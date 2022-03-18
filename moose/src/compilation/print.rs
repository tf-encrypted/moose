use crate::computation::{Computation, Operation, Operator, ReceiveOp};
use crate::textual::ToTextual;
use petgraph::dot::Config::{EdgeNoLabel, NodeNoLabel};
use petgraph::dot::Dot;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

const COLORS: [&str; 9] = [
    "#336699", "#ff0000", "#ff6600", "#92cd00", "#ffcc00", "#ffa4b6", "#f765a3", "#a155b9",
    "#3caea3",
];

// TODO remove `get` prefix
/// Prints the computation's graph DOT representation to stdout
pub fn get_dot_graph(comp: &Computation) -> String {
    let graph = comp.as_graph();

    // We need to compute the color lookup table ahead of time, because `Dot`'s closures capture everything as immutable
    let mut color_cache = HashMap::new();
    let mut color = 0..;
    for n in graph.node_indices() {
        let placement = comp.operations[graph[n].index].placement.to_textual();
        color_cache
            .entry(placement)
            .or_insert_with(|| COLORS[color.next().unwrap_or_default() % COLORS.len()]);
    }

    format!(
        "{:?}",
        Dot::with_attr_getters(
            &graph,
            // We will be manually adding label elements, turn off automatic labels.
            &[EdgeNoLabel, NodeNoLabel],
            // Edge formatter.
            &|_, edge| {
                let source = &comp.operations[graph[edge.source()].index];
                let target = &comp.operations[graph[edge.target()].index];
                // Annotate jumps in between the hosts
                match (&source.kind, &target.kind) {
                    (
                        Operator::Send(_),
                        Operator::Receive(ReceiveOp {
                            rendezvous_key: key,
                            ..
                        }),
                    ) => format!("label={} style = dotted", key),
                    _ => "".into(),
                }
            },
            // Node formatter.
            &|_, (_, n)| {
                format!(
                    "label = \"{}\" shape = {} color = \"{}\"",
                    pretty(&comp.operations[n.index]),
                    shape(&comp.operations[n.index]),
                    color_cache[&comp.operations[n.index].placement.to_textual()]
                )
            }
        )
    )
}

/// Prints the computation's graph DOT representation to stdout
pub fn print_graph(comp: Computation) -> anyhow::Result<Computation> {
    let graph = get_dot_graph(&comp);
    println!("{}", graph);
    Ok(comp)
}

/// Prints only the bits of the operation that we want visible on the graph's DOT
fn pretty(op: &Operation) -> String {
    format!(
        "{} = {}\\l{}",
        op.name,
        op.kind.short_name(),
        op.placement.to_textual()
    )
}

fn shape(op: &Operation) -> String {
    match op.kind {
        Operator::Input(_) => "invhouse".into(),
        Operator::Output(_) => "house".into(),
        Operator::Send(_) => "rarrow".into(),
        Operator::Receive(_) => "larrow".into(),
        _ => "rectangle".into(),
    }
}
