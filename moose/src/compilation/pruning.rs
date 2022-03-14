use crate::computation::{Computation, Operator};
use petgraph::visit::{depth_first_search, DfsEvent};

/// Prunes the computation from anything not relevant for the output
pub fn prune_graph(comp: &Computation) -> anyhow::Result<Option<Computation>> {
    // Need to reverse the graph, because we will be traversing it from the outputs
    let mut graph = comp.as_graph();
    graph.reverse();
    // Operations to keep
    let mut keep = Vec::with_capacity(comp.operations.len());
    // Identify all the output nodes
    let outputs = graph
        .node_indices()
        .filter(|i| matches!(comp.operations[graph[*i].1].kind, Operator::Output(_)));

    // Perform a DFS
    depth_first_search(&graph, outputs, |event| {
        if let DfsEvent::Discover(visited, _) = event {
            keep.push(comp.operations[graph[visited].1].clone());
        };
    });

    // Construct a new computation. NB: we did not toposort it.
    Ok(Some(Computation { operations: keep }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textual::ToTextual;
    use std::convert::TryInto;

    #[test]
    fn test_nothing_to_prune() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        x = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        y = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        z = Output: (HostFloat32Tensor) -> HostFloat32Tensor (mul) @Host(alice)"#;

        let comp = prune_graph(&source.try_into()?)?.unwrap();
        // Pruning should not introduce any changes to such a computation
        assert_eq!(comp.operations.len(), 4);
        let comp = comp.to_textual();
        assert!(comp.contains(
            "x = Constant{value = HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor () @Host(alice)"
        ));
        assert!(comp.contains(
            "y = Constant{value = HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor () @Host(alice)"
        ));
        assert!(comp.contains(
            "mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)"
        ));
        assert!(comp
            .contains("z = Output: (HostFloat32Tensor) -> HostFloat32Tensor (mul) @Host(alice)"));
        Ok(())
    }

    #[test]
    fn test_simple_prune() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        x = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        y = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        add = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        dot = Dot: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        z = Output: (HostFloat32Tensor) -> HostFloat32Tensor (mul) @Host(alice)"#;

        let comp = prune_graph(&source.try_into()?)?.unwrap();
        // Pruning should remove `add` and `dot`
        assert_eq!(comp.operations.len(), 4);
        let comp = comp.to_textual();
        assert!(comp.contains(
            "x = Constant{value = HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor () @Host(alice)"
        ));
        assert!(comp.contains(
            "y = Constant{value = HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor () @Host(alice)"
        ));
        assert!(comp.contains(
            "mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)"
        ));
        assert!(comp
            .contains("z = Output: (HostFloat32Tensor) -> HostFloat32Tensor (mul) @Host(alice)"));
        Ok(())
    }

    #[test]
    fn test_network_prune() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        x = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        y = Constant {value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(bob)
        send_mul = Send {rendezvous_key=30303030303030303030303030303031, receiver="alice"}: (HostFloat32Tensor) -> HostUnit (y) @Host(bob)
        recv_mul = Receive {rendezvous_key=30303030303030303030303030303031, sender="bob"} : () -> HostFloat32Tensor () @Host(alice)
        send_add = Send {rendezvous_key=30303030303030303030303030303032, receiver="alice"}: (HostFloat32Tensor) -> HostUnit (y) @Host(bob)
        recv_add = Receive {rendezvous_key=30303030303030303030303030303032, sender="bob"} : () -> HostFloat32Tensor () @Host(alice)
        mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, recv_mul) @Host(alice)
        add = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, recv_add) @Host(alice)
        z = Output: (HostFloat32Tensor) -> HostFloat32Tensor (mul) @Host(alice)"#;

        let comp = prune_graph(&source.try_into()?)?.unwrap();

        assert_eq!(comp.operations.len(), 6);
        let comp = comp.to_textual();
        assert!(comp.contains(
            "x = Constant{value = HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor () @Host(alice)"
        ));
        assert!(comp.contains(
            "y = Constant{value = HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor () @Host(bob)"
        ));
        assert!(comp.contains(
            "mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, recv_mul) @Host(alice)"
        ));
        assert!(comp.contains(
            r#"send_mul = Send{rendezvous_key = 30303030303030303030303030303031, receiver = "alice"}: (HostFloat32Tensor) -> HostUnit (y) @Host(bob)"#
        ));
        assert!(comp.contains(
            r#"recv_mul = Receive{rendezvous_key = 30303030303030303030303030303031, sender = "bob"}: () -> HostFloat32Tensor () @Host(alice)"#
        ));
        assert!(comp.contains(
            "mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, recv_mul) @Host(alice)"
        ));
        assert!(comp
            .contains("z = Output: (HostFloat32Tensor) -> HostFloat32Tensor (mul) @Host(alice)"));
        Ok(())
    }

    #[test]
    fn test_multiple_output_prune() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        x = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        y = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        add = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        dot = Dot: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        z = Output: (HostFloat32Tensor) -> HostFloat32Tensor (mul) @Host(alice)
        z2 = Output: (HostFloat32Tensor) -> HostFloat32Tensor (add) @Host(alice)"#;

        let comp = prune_graph(&source.try_into()?)?.unwrap();
        // Pruning should remove only  `dot`
        assert_eq!(comp.operations.len(), 6);
        let comp = comp.to_textual();
        assert!(comp.contains(
            "x = Constant{value = HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor () @Host(alice)"
        ));
        assert!(comp.contains(
            "y = Constant{value = HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor () @Host(alice)"
        ));
        assert!(comp.contains(
            "mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)"
        ));
        assert!(comp
            .contains("z = Output: (HostFloat32Tensor) -> HostFloat32Tensor (mul) @Host(alice)"));
        assert!(comp.contains(
            "add = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)"
        ));
        assert!(comp
            .contains("z2 = Output: (HostFloat32Tensor) -> HostFloat32Tensor (add) @Host(alice)"));
        Ok(())
    }
}
