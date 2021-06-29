use crate::computation::*;
use petgraph::Direction;

/// Updates the operators such that the tpye information is inferred by one-hop check, without any graph traversal.
pub fn update_types_one_hop(comp: &Computation) -> anyhow::Result<Option<Computation>> {
    let mut operations = comp.operations.clone();
    let graph = comp.as_graph();

    for n in graph.node_indices() {
        let op = &comp.operations[graph[n].1];
        for (pos, &t) in op.kind.sig().args().iter().enumerate() {
            if *t != Ty::UnknownTy {
                continue;
            }
            let name = &op.inputs[pos];
            let src_op = graph
                .neighbors_directed(n, Direction::Incoming)
                .find(|i| name == &graph[*i].0);
            if let Some(new_type) = src_op.map(|i| comp.operations[graph[i].1].kind.sig().ret()) {
                // We found a better type, let's use it
                operations[graph[n].1].kind = op
                    .kind
                    .new_with_sig(op.kind.sig().new_with_arg(pos, new_type));
            }
        }
    }
    Ok(Some(Computation { operations }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_computation::ToTextual;
    use std::convert::TryInto;

    #[test]
    fn test_all_on_one_host() -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])} @Host(alice)
        y = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])} @Host(alice)
        mul = StdMul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)
        dot = StdDot: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(alice)
        mean = StdMean: (Float32Tensor) -> Float32Tensor (dot) @Host(alice)
        constant_0 = Constant{value = String("regression_weights")} () @Host(alice)
        save = Save: (String, Unknown) -> Unit (constant_0, mean) @Host(alice)
        "#;

        let comp = update_types_one_hop(&source.try_into()?)?
            .unwrap()
            .to_textual();
        // The computation should now contain the type information
        assert!(comp.contains(
            "save = Save: (String, Float32Tensor) -> Unit (constant_0, mean) @Host(alice)"
        ));
        Ok(())
    }
}
