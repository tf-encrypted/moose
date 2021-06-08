use moose::computation::Computation;
use petgraph::dot::Dot;

/// Prints the computation's graph DOT representation to stdout
pub fn print_graph(comp: &Computation) -> anyhow::Result<Computation> {
    let graph = comp.as_graph();
    println!("{:?}", Dot::new(&graph));
    Ok(Computation {
        operations: comp.operations.clone(),
    })
}
