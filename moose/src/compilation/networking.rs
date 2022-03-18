use crate::computation::*;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Applies the networking pass to the entire computation
pub fn networking_pass(comp: &Computation) -> anyhow::Result<Option<Computation>> {
    let graph = comp.as_graph();
    let mut pass = NetworkingPass::new(comp);

    let mut created_cache = HashMap::new();
    for er in graph.edge_references() {
        let src_op = &comp.operations[graph[er.source()].index];
        let dst_op = &comp.operations[graph[er.target()].index];
        match placement_discrimnator(src_op, dst_op) {
            // We only operate on edges that jump from a host to a different host
            (Some(src), Some(dst)) if src != dst => {
                // Create a jump or use the existing one, if already passed `src_op` to host `dst`
                let receive_op_name = created_cache
                    .entry((dst, &src_op.name))
                    .or_insert_with(|| pass.create_networking_jump(src_op, dst_op, src, dst));

                // Update target operation's input to the receive operation's name
                if let Some(input) = pass.operations[graph[er.target()].index]
                    .inputs
                    .iter_mut()
                    .find(|r| *r == &src_op.name)
                {
                    *input = receive_op_name.clone();
                }
            }
            _ => (),
        }
    }

    pass.operations.extend(pass.extra_ops);
    Ok(Some(Computation {
        operations: pass.operations,
    }))
}

pub struct NetworkingPass {
    operations: Vec<Operation>,
    extra_ops: Vec<Operation>,
    counter: std::ops::RangeFrom<usize>,
    rendezvous: std::ops::RangeFrom<usize>,
}

impl NetworkingPass {
    /// Create a new context for the Networking pass
    fn new(comp: &Computation) -> NetworkingPass {
        NetworkingPass {
            operations: comp.operations.clone(),
            extra_ops: Vec::new(),
            counter: 0..,
            rendezvous: 0..,
        }
    }

    /// Create operations necessary for a networking jump between two hosts
    fn create_networking_jump(
        &mut self,
        src_op: &Operation,
        dst_op: &Operation,
        src: &str,
        dst: &str,
    ) -> String {
        let index = self.counter.next().unwrap();

        let rendezvous_key = RendezvousKey::from(self.rendezvous.next().unwrap() as u128);

        let send_op = Operation {
            name: format!("send_{}", index),
            kind: SendOp {
                sig: Signature::unary(src_op.kind.sig().ret(), Ty::HostUnit),
                rendezvous_key: rendezvous_key.clone(),
                receiver: Role::from(dst),
            }
            .into(),
            inputs: vec![src_op.name.clone()],
            placement: src_op.placement.clone(),
        };
        self.extra_ops.push(send_op);

        let receive_op_name = format!("receive_{}", index);
        let receive_op = Operation {
            name: receive_op_name.clone(),
            kind: ReceiveOp {
                sig: Signature::nullary(src_op.kind.sig().ret()),
                rendezvous_key,
                sender: Role::from(src),
            }
            .into(),
            inputs: vec![],
            placement: dst_op.placement.clone(),
        };
        self.extra_ops.push(receive_op);

        receive_op_name
    }
}

/// The discriminator to find the signature of an edge projected as a jump between hosts
fn placement_discrimnator<'a>(
    op1: &'a Operation,
    op2: &'a Operation,
) -> (Option<&'a str>, Option<&'a str>) {
    fn placement(op: &Operation) -> Option<&str> {
        match &op.placement {
            Placement::Host(host) => Some(host.owner.0.as_str()),
            _ => None,
        }
    }

    (placement(op1), placement(op2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textual::ToTextual;
    use std::convert::TryInto;

    #[test]
    fn test_all_on_one_host() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        x = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        y = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        dot = Dot: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        mean = Mean{}: (HostFloat32Tensor) -> HostFloat32Tensor (dot) @Host(alice)"#;

        let comp = networking_pass(&source.try_into()?)?.unwrap().to_textual();
        // Networking should not introduce any changes to such a computation
        assert!(comp.contains(
            "mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)"
        ));
        assert!(comp.contains(
            "dot = Dot: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)"
        ));
        assert!(comp
            .contains("mean = Mean: (HostFloat32Tensor) -> HostFloat32Tensor (dot) @Host(alice)"));
        Ok(())
    }

    #[test]
    fn test_regular_jumps() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        x = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        y = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(bob)
        mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        dot = Dot: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        mean = Mean{}: (HostFloat32Tensor) -> HostFloat32Tensor (dot) @Host(alice)"#;
        let comp = networking_pass(&source.try_into()?)?.unwrap().to_textual();

        // Networking should introduce one new networking operation (not 2) for the 2 jumps. And leave the mean unchaged (dot already on the right host)
        assert!(comp.contains(
            r#"send_0 = Send{rendezvous_key = 00000000000000000000000000000000, receiver = "alice"}: (HostFloat32Tensor) -> HostUnit (y) @Host(bob)"#
        ));
        assert!(comp.contains(r#"receive_0 = Receive{rendezvous_key = 00000000000000000000000000000000, sender = "bob"}: () -> HostFloat32Tensor () @Host(alice)"#));
        assert!(comp.contains("mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, receive_0) @Host(alice)"));
        assert!(comp.contains("dot = Dot: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, receive_0) @Host(alice)"));
        assert!(comp
            .contains("mean = Mean: (HostFloat32Tensor) -> HostFloat32Tensor (dot) @Host(alice)"));
        Ok(())
    }

    #[test]
    fn test_jumps_cache() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        x = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        y = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(bob)
        add = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(bob)"#;
        let comp = networking_pass(&source.try_into()?)?.unwrap().to_textual();
        // Should have one send/receive pair per each variable being sent
        assert!(comp.contains(
            r#"send_0 = Send{rendezvous_key = 00000000000000000000000000000000, receiver = "bob"}: (HostFloat32Tensor) -> HostUnit (x) @Host(alice)"#
        ));
        assert!(comp.contains(r#"receive_0 = Receive{rendezvous_key = 00000000000000000000000000000000, sender = "alice"}: () -> HostFloat32Tensor () @Host(bob)"#));
        assert!(comp.contains(
            r#"send_1 = Send{rendezvous_key = 01000000000000000000000000000000, receiver = "bob"}: (HostFloat32Tensor) -> HostUnit (y) @Host(alice)"#
        ));
        assert!(comp.contains(r#"receive_1 = Receive{rendezvous_key = 01000000000000000000000000000000, sender = "alice"}: () -> HostFloat32Tensor () @Host(bob)"#));
        // Should use the same pair of operators for both computations on both (asserting for no extra jumps)
        assert!(comp.contains(r#"add = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (receive_0, receive_1) @Host(bob)"#));
        assert!(comp.contains(r#"mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (receive_0, receive_1) @Host(bob)"#));
        Ok(())
    }

    #[test]
    fn test_ignore_replicated() -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        y = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(bob)
        mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Replicated(alice, bob, charlie)"#;

        let comp = networking_pass(&source.try_into()?)?.unwrap().to_textual();
        // Networking should not make any changes to the replicated placement (should probably never see it in real life)
        assert!(comp.contains("mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Replicated(alice, bob, charlie)"));
        Ok(())
    }
}
