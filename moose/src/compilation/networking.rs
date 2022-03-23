use crate::{computation::*, host::HostPlacement};
use std::collections::HashMap;

/// Applies the networking pass to the entire computation
pub fn networking_pass(mut comp: Computation) -> anyhow::Result<Computation> {
    let graph = comp.as_graph();
    let mut state = NetworkingPassState::new();
    // Per src_op cache; allocated here to reuse memory but cleared for each src_op
    let mut cache: HashMap<HostPlacement, String> = HashMap::new();

    for src_node in graph.node_indices() {
        let src_idx = graph[src_node].index;

        cache.clear();

        for dst_node in graph.neighbors(src_node) {
            let dst_idx = graph[dst_node].index;

            let receive_op_name = {
                let src_op = &comp.operations[src_idx];
                let dst_op = &comp.operations[dst_idx];

                use Placement::*;
                match (&src_op.placement, &dst_op.placement) {
                    // We only operate on edges across different hosts
                    (Host(src_host), Host(dst_host)) if src_host != dst_host => {
                        let src_op_name = src_op.name.clone();

                        // Create a new jump, or use an existing if `src_op` has already been sent to `dst_host`
                        if let Some(receive_op_name) = cache.get(dst_host) {
                            Some((src_op_name, receive_op_name.clone()))
                        } else {
                            let receive_op_name = state.create_networking_jump(src_op, dst_op);
                            cache.insert(dst_host.clone(), receive_op_name.clone());
                            Some((src_op_name, receive_op_name))
                        }
                    }
                    _ => {
                        // No updates needed
                        None
                    }
                }
            };

            if let Some((src_op_name, receive_op_name)) = receive_op_name {
                // Update target operation's input to the receive operation's name
                let dst_op = &mut comp.operations[dst_idx];
                for input_op_name in &mut dst_op.inputs {
                    if *input_op_name == src_op_name {
                        *input_op_name = receive_op_name.clone();
                    }
                }
            };
        }
    }

    comp.operations.extend(state.extra_ops);
    Ok(comp)
}

struct NetworkingPassState {
    extra_ops: Vec<Operation>,
    counter: std::ops::RangeFrom<usize>,
    rendezvous: std::ops::RangeFrom<usize>,
}

impl NetworkingPassState {
    fn new() -> NetworkingPassState {
        NetworkingPassState {
            extra_ops: Vec::new(),
            counter: 0..,
            rendezvous: 0..,
        }
    }

    /// Create operations necessary for a networking jump between two hosts
    fn create_networking_jump(&mut self, src_op: &Operation, dst_op: &Operation) -> String {
        let index = self.counter.next().unwrap();

        let rendezvous_key = RendezvousKey::from(self.rendezvous.next().unwrap() as u128);

        let receiver = match &dst_op.placement {
            Placement::Host(plc) => plc.owner.clone(),
            _ => unimplemented!(), // should never happen
        };

        let sender = match &src_op.placement {
            Placement::Host(plc) => plc.owner.clone(),
            _ => unimplemented!(), // should never happen
        };

        let send_op = Operation {
            name: format!("send_{}", index),
            kind: SendOp {
                sig: Signature::unary(src_op.kind.sig().ret(), Ty::HostUnit),
                rendezvous_key: rendezvous_key.clone(),
                receiver,
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
                sender,
            }
            .into(),
            inputs: vec![],
            placement: dst_op.placement.clone(),
        };
        self.extra_ops.push(receive_op);

        receive_op_name
    }
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

        let comp = networking_pass(source.try_into()?)?.to_textual();
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
        let comp = networking_pass(source.try_into()?)?.to_textual();

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
        let comp = networking_pass(source.try_into()?)?.to_textual();
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

        let comp = networking_pass(source.try_into()?)?.to_textual();
        // Networking should not make any changes to the replicated placement (should probably never see it in real life)
        assert!(comp.contains("mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Replicated(alice, bob, charlie)"));
        Ok(())
    }
}
