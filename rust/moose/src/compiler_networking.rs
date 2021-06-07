use crate::computation::*;
use std::collections::HashMap;

pub fn print_graph(comp: &Computation) -> anyhow::Result<Computation> {
    comp.graph_operation(|graph, _| {
        println!("{:?}", petgraph::dot::Dot::new(&graph));
        Ok(())
    })?;
    Ok(Computation {
        operations: comp.operations.clone(),
    })
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

    /// Applies the networking pass to the entire computation
    ///
    /// TODO: Types
    pub fn compiler_networking(comp: &Computation) -> anyhow::Result<Computation> {
        // We clone the operations to make the changes to them.
        let mut pass = NetworkingPass::new(comp);

        comp.graph_operation(|graph, inv_map| {
            use petgraph::visit::EdgeRef;

            let mut created_cache = HashMap::new();
            for er in graph.edge_references() {
                let src_op = &comp.operations[inv_map[&er.source()]];
                let dst_op = &comp.operations[inv_map[&er.target()]];
                match placement_discrimnator(src_op, dst_op) {
                    // We only operate on edges that jump from a host to a different host
                    (Some(src), Some(dst)) if src != dst => {
                        // Create a jump, if we never jumped from host `src` to host `dst`
                        let receive_op_name =
                            created_cache.entry((src, dst)).or_insert_with(|| {
                                pass.create_networking_jump(src_op, dst_op, src, dst)
                            });

                        // Update target operation's input to the receive operation's name
                        let position = dst_op
                            .inputs
                            .iter()
                            .position(|r| src_op.name.eq(r))
                            .unwrap();
                        pass.operations[inv_map[&er.target()]].inputs[position] =
                            receive_op_name.clone();
                    }
                    _ => (),
                }
            }
            Ok(())
        })?;

        pass.operations.extend(pass.extra_ops);
        Computation {
            operations: pass.operations,
        }
        .toposort()
        .map_err(|e| anyhow::anyhow!("Failed to sort the ops {}", e))
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

        let rendezvous_key = format!("rendezvous_key_{}", self.rendezvous.next().unwrap());

        let send_operation = Operation {
            name: format!("send_{}", index),
            kind: Operator::Send(SendOp {
                rendezvous_key: rendezvous_key.clone(),
                receiver: Role::from(dst),
            }),
            inputs: vec![src_op.name.clone()],
            placement: src_op.placement.clone(),
        };
        self.extra_ops.push(send_operation);

        let receive_operation = Operation {
            name: format!("receive_{}", index),
            kind: Operator::Receive(ReceiveOp {
                rendezvous_key: rendezvous_key.clone(),
                sender: Role::from(src),
                ty: Ty::Float32TensorTy,
            }), // TODO Types
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
