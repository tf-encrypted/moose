use crate::computation::*;
use std::collections::HashMap;
use std::mem::discriminant;

/// Applies the networking pass to the entire computation
pub fn compiler_networking(comp: &Computation) -> anyhow::Result<Computation> {
    let mut operations = comp.operations.clone();
    let extra_ops = cut_networking_edges(&mut operations);
    operations.extend(extra_ops);
    Computation { operations }
        .toposort()
        .map_err(|e| anyhow::anyhow!("Failed to sort the ops {}", e))
}

fn cut_networking_edges(ops: &mut Vec<Operation>) -> Vec<Operation> {
    let placement_lookup_table = compute_operations_lookup_dict(ops);
    let host_discriminant = discriminant(&Placement::Host(HostPlacement {
        owner: Role::from("ignored"),
    }));
    // Edges are tuples of destination operation and input name (name of the source destination)
    let edges: Vec<(&mut Operation, Vec<String>)> = ops
        .iter_mut()
        // Only consider operators placed on Hosts
        .filter(|op| discriminant(&op.placement) == host_discriminant)
        // map to the vector of edges spanning hosts
        .map(list_host_jumps(&placement_lookup_table))
        // Nothing to do with operators that do not have edges between the hosts
        .filter(|(_, inputs)| !inputs.is_empty())
        .collect();

    let mut extra_ops = Vec::new();
    for (op, inputs) in edges {
        for input in inputs {
            // rendezvous_key = context.get_fresh_name("rendezvous_key")
            let rendezvous_key = "rendezvous_key_todo";

            let send_operation = Operation {
                name: "sendXX_todo".into(), // TODO
                kind: Operator::Send(SendOp {
                    rendezvous_key: rendezvous_key.into(),
                    receiver: Role::from("bob"),
                }), // TODO, receiver
                inputs: vec!["y".into()],   // TODO: source op
                placement: op.placement.clone(), // TODO: source
            };
            extra_ops.push(send_operation);

            let receive_operation = Operation {
                name: "receiveXX_todo".into(), // TODO
                kind: Operator::Receive(ReceiveOp {
                    rendezvous_key: rendezvous_key.into(),
                    sender: Role::from("alice"),
                    ty: Ty::Float32TensorTy,
                }), // TODO Types, sender
                inputs: vec![],
                placement: op.placement.clone(),
            };
            extra_ops.push(receive_operation);

            let index = op.inputs.iter().position(|r| input.eq(r)).unwrap();
            op.inputs[index] = "receiveXX_todo".into(); // TODO find the index of `input` - perhaps just work with the index from the start?
        }
    }
    extra_ops
}

fn compute_operations_lookup_dict(ops: &[Operation]) -> HashMap<String, String> {
    let mut dict = HashMap::new();
    for op in ops {
        dict.insert(op.name.clone(), format!("{:?}", op.placement)); // TODO: Consider textual format instead of debug print?
    }
    dict
}

fn list_host_jumps<'a>(
    placement_lookup_table: &'a HashMap<String, String>,
) -> impl Fn(&'a mut Operation) -> (&'a mut Operation, Vec<String>) {
    move |op| {
        let dst_placement = format!("{:?}", op.placement); // TODO: Consider textual format instead of debug print?
        let edges: Vec<String> = op
            .inputs
            .iter()
            .filter_map(|i| {
                if placement_lookup_table.get(i) != Some(&dst_placement) {
                    Some(i.clone())
                } else {
                    None
                }
            })
            .collect();
        (op, edges)
    }
}
