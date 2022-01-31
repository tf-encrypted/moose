use crate::computation::*;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

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

    /// Applies the networking pass to the entire computation
    pub fn pass(comp: &Computation) -> anyhow::Result<Option<Computation>> {
        // We clone the operations to make the changes to them.
        let mut pass = NetworkingPass::new(comp);

        let graph = comp.as_graph();

        let mut created_cache = HashMap::new();
        for er in graph.edge_references() {
            let src_op = &comp.operations[graph[er.source()].1];
            let dst_op = &comp.operations[graph[er.target()].1];
            match placement_discrimnator(src_op, dst_op) {
                // We only operate on edges that jump from a host to a different host
                (Some(src), Some(dst)) if src != dst => {
                    // Create a jump or use the existing one, if already passed `src_op` to host `dst`
                    let receive_op_name = created_cache
                        .entry((dst, &src_op.name))
                        .or_insert_with(|| pass.create_networking_jump(src_op, dst_op, src, dst));

                    // Update target operation's input to the receive operation's name
                    if let Some(input) = pass.operations[graph[er.target()].1]
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

        let send_operation = Operation {
            name: format!("send_{}", index),
            kind: SendOp {
                sig: Signature::unary(src_op.kind.sig().ret(), Ty::Unit),
                rendezvous_key: rendezvous_key.clone(),
                receiver: Role::from(dst),
            }
            .into(),
            inputs: vec![src_op.name.clone()],
            placement: src_op.placement.clone(),
        };
        self.extra_ops.push(send_operation);

        let receive_operation = Operation {
            name: format!("receive_{}", index),
            kind: ReceiveOp {
                sig: Signature::nullary(src_op.kind.sig().ret()),
                rendezvous_key,
                sender: Role::from(src),
            }
            .into(),
            inputs: vec![],
            placement: dst_op.placement.clone(),
        };
        self.extra_ops.push(receive_operation);

        // Return the name of the receive operation
        format!("receive_{}", index)
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
        x = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(alice)
        y = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(alice)
        mul = Mul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)
        dot = HostDot: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)
        mean = HostMean{}: (Float32Tensor) -> Float32Tensor (dot) @Host(alice)"#;

        let comp = NetworkingPass::pass(&source.try_into()?)?
            .unwrap()
            .to_textual();
        // Networking should not introduce any changes to such a computation
        assert!(comp.contains(
            "mul = Mul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)"
        ));
        assert!(comp.contains(
            "dot = HostDot: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)"
        ));
        assert!(
            comp.contains("mean = HostMean: (Float32Tensor) -> Float32Tensor (dot) @Host(alice)")
        );
        Ok(())
    }

    #[test]
    fn test_regular_jumps() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        x = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(alice)
        y = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(bob)
        mul = Mul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)
        dot = HostDot: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)
        mean = HostMean{}: (Float32Tensor) -> Float32Tensor (dot) @Host(alice)"#;
        let comp = NetworkingPass::pass(&source.try_into()?)?
            .unwrap()
            .to_textual();

        // Networking should introduce one new networking operation (not 2) for the 2 jumps. And leave the mean unchaged (dot already on the right host)
        assert!(comp.contains(
            r#"send_0 = Send{rendezvous_key = 00000000000000000000000000000000, receiver = "alice"}: (Float32Tensor) -> Unit (y) @Host(bob)"#
        ));
        assert!(comp.contains(r#"receive_0 = Receive{rendezvous_key = 00000000000000000000000000000000, sender = "bob"}: () -> Float32Tensor () @Host(alice)"#));
        assert!(comp.contains("mul = Mul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, receive_0) @Host(alice)"));
        assert!(comp.contains("dot = HostDot: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, receive_0) @Host(alice)"));
        assert!(
            comp.contains("mean = HostMean: (Float32Tensor) -> Float32Tensor (dot) @Host(alice)")
        );
        Ok(())
    }

    #[test]
    fn test_jumps_cache() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        x = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(alice)
        y = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(alice)
        mul = Mul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(bob)
        add = Add: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(bob)"#;
        let comp = NetworkingPass::pass(&source.try_into()?)?
            .unwrap()
            .to_textual();
        // Should have one send/receive pair per each variable being sent
        assert!(comp.contains(
            r#"send_0 = Send{rendezvous_key = 00000000000000000000000000000000, receiver = "bob"}: (Float32Tensor) -> Unit (x) @Host(alice)"#
        ));
        assert!(comp.contains(r#"receive_0 = Receive{rendezvous_key = 00000000000000000000000000000000, sender = "alice"}: () -> Float32Tensor () @Host(bob)"#));
        assert!(comp.contains(
            r#"send_1 = Send{rendezvous_key = 01000000000000000000000000000000, receiver = "bob"}: (Float32Tensor) -> Unit (y) @Host(alice)"#
        ));
        assert!(comp.contains(r#"receive_1 = Receive{rendezvous_key = 01000000000000000000000000000000, sender = "alice"}: () -> Float32Tensor () @Host(bob)"#));
        // Should use the same pair of operators for both computations on both (asserting for no extra jumps)
        assert!(comp.contains(r#"add = Add: (Float32Tensor, Float32Tensor) -> Float32Tensor (receive_0, receive_1) @Host(bob)"#));
        assert!(comp.contains(r#"mul = Mul: (Float32Tensor, Float32Tensor) -> Float32Tensor (receive_0, receive_1) @Host(bob)"#));
        Ok(())
    }

    #[test]
    fn test_ignore_replicated() -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(alice)
        y = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(bob)
        mul = Mul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Replicated(alice, bob, charlie)"#;

        let comp = NetworkingPass::pass(&source.try_into()?)?
            .unwrap()
            .to_textual();
        // Networking should not make any changes to the replicated placement (should probably never see it in real life)
        assert!(comp.contains("mul = Mul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Replicated(alice, bob, charlie)"));
        Ok(())
    }
}
