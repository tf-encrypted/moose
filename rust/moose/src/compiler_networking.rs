use crate::computation::*;

pub fn print_graph(comp: &Computation) -> anyhow::Result<Computation> {
    comp.graph_operation(|graph, _| {
        println!("{:?}", petgraph::dot::Dot::new(&graph));
        Ok(())
    })?;
    Ok(Computation {
        operations: comp.operations.clone(),
    })
}

/// Applies the networking pass to the entire computation
///
/// TODO: Cache, Types, rendezvous_key_
pub fn compiler_networking(comp: &Computation) -> anyhow::Result<Computation> {
    // We clone the operations to make the changes to them.
    let mut operations = comp.operations.clone();

    let extra_ops = comp.graph_operation(|graph, inv_map| {
        use petgraph::visit::EdgeRef;

        let mut extra_ops = Vec::new();
        let mut counter = 0..;
        for er in graph.edge_references() {
            let src_op = &comp.operations[inv_map[&er.source()]];
            let dst_op = &comp.operations[inv_map[&er.target()]];
            match placement_discrimnator(src_op, dst_op) {
                // We only operate on edges that jump from a host to a different host
                (Some(src), Some(trg)) if src != trg => {
                    let index = counter.next().unwrap();
                    print!(
                        "\nFound a {} networking edge to cut!\n{} -> {}\n",
                        index, src, trg
                    );

                    // rendezvous_key = context.get_fresh_name("rendezvous_key")
                    let rendezvous_key = format!("rendezvous_key_{}", index); // TODO

                    let send_operation = Operation {
                        name: format!("send_{}", index),
                        kind: Operator::Send(SendOp {
                            rendezvous_key: rendezvous_key.clone(),
                            receiver: Role::from(trg),
                        }),
                        inputs: vec![src_op.name.clone()],
                        placement: src_op.placement.clone(),
                    };
                    extra_ops.push(send_operation);

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
                    extra_ops.push(receive_operation);

                    let position = dst_op
                        .inputs
                        .iter()
                        .position(|r| src_op.name.eq(r))
                        .unwrap();
                    operations[inv_map[&er.target()]].inputs[position] =
                        format!("receive_{}", index);
                }
                _ => (),
            }
        }
        Ok(extra_ops)
    })?;

    operations.extend(extra_ops);
    print!("\nNetworking pass complete!\n");
    Computation { operations }
        .toposort()
        .map_err(|e| anyhow::anyhow!("Failed to sort the ops {}", e))
}

/// The discriminator to find signature of an edge projected as a jump between hosts
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
