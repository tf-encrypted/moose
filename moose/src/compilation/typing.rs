use crate::computation::*;
use petgraph::Direction;
use std::collections::HashMap;

/// Updates the operators such that the type information is inferred by one-hop check, without any recursive graph traversal.
pub(crate) fn update_types_one_hop(comp: &Computation) -> anyhow::Result<Option<Computation>> {
    let mut operations = comp.operations.clone();
    let graph = comp.as_graph();

    for n in graph.node_indices() {
        // Prepare the raw data for the signature computation
        let inputs = &comp.operations[graph[n].1].inputs;
        let types: HashMap<&String, Ty> = graph
            .neighbors_directed(n, Direction::Incoming)
            .map(|i| (&graph[i].0, comp.operations[graph[i].1].kind.sig().ret()))
            .collect();
        let ret = comp.operations[graph[n].1].kind.sig().ret();

        let find_type = |i: usize| -> anyhow::Result<Ty> {
            match types.get(&inputs[i]) {
                Some(ty) => Ok(*ty),
                _ => Err(anyhow::anyhow!(
                    "Could not find type of input {}",
                    inputs[i]
                )),
            }
        };

        // Compute the new signature from the graph
        let new_sig = match inputs.len() {
            0 => Signature::nullary(ret),
            1 => Signature::unary(find_type(0)?, ret),
            2 => Signature::binary(find_type(0)?, find_type(1)?, ret),
            3 => Signature::ternary(find_type(0)?, find_type(1)?, find_type(2)?, ret),
            n => {
                assert!((0..n).all(|i| find_type(i).ok() == find_type(0).ok()));
                Signature::variadic(find_type(0)?, ret)
            }
        };

        // Update the existing signature with it.
        operations[graph[n].1].kind.sig_mut().merge(new_sig)?;
    }
    Ok(Some(Computation { operations }))
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
        mean = HostMean{}: (HostFloat32Tensor) -> HostFloat32Tensor (dot) @Host(alice)
        constant_0 = Constant{value = HostString("regression_weights")}: () -> HostString () @Host(alice)
        save = Save: (HostString, Unknown) -> Unit (constant_0, mean) @Host(alice)
        "#;

        let comp = update_types_one_hop(&source.try_into()?)?
            .unwrap()
            .to_textual();
        // The computation should now contain the type information
        assert!(comp.contains(
            "save = Save: (HostString, HostFloat32Tensor) -> Unit (constant_0, mean) @Host(alice)"
        ));
        Ok(())
    }
}
