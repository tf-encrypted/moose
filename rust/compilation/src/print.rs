use moose::computation::Computation;
use petgraph::dot::Dot;

/// Prints the computation's graph DOT representation to stdout
pub fn print_graph(comp: &Computation) -> anyhow::Result<Computation> {
    comp.graph_operation(|graph, _| {
        println!("{:?}", Dot::new(&graph));
        Ok(())
    })?;
    Ok(Computation {
        operations: comp.operations.clone(),
    })
}
