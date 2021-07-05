use crate::computation::{Computation, Operation, Operator, ReceiveOp};
use crate::text_computation::ToTextual;
use petgraph::dot::Config::{EdgeNoLabel, NodeNoLabel};
use petgraph::dot::Dot;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

const COLORS: [&str; 9] = [
    "#336699", "#ff0000", "#ff6600", "#92cd00", "#ffcc00", "#ffa4b6", "#f765a3", "#a155b9",
    "#3caea3",
];

/// Prints the computation's graph DOT representation to stdout
pub fn print_graph(comp: &Computation) -> anyhow::Result<Option<Computation>> {
    let graph = comp.as_graph();

    // We need to compute the color lookup table ahead of time, because `Dot`'s closures capture everything as immutable
    let mut color_cache = HashMap::new();
    let mut color = 0..;
    for n in graph.node_indices() {
        let placement = comp.operations[graph[n].1].placement.to_textual();
        color_cache
            .entry(placement)
            .or_insert_with(|| COLORS[color.next().unwrap_or_default() % COLORS.len()]);
    }

    println!(
        "{:?}",
        Dot::with_attr_getters(
            &graph,
            // We will be manually adding label elements, turn off automatic labels.
            &[EdgeNoLabel, NodeNoLabel],
            // Edge formatter.
            &|_, edge| {
                let source = &comp.operations[graph[edge.source()].1];
                let target = &comp.operations[graph[edge.target()].1];
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
                    pretty(&comp.operations[n.1]),
                    shape(&comp.operations[n.1]),
                    color_cache[&comp.operations[n.1].placement.to_textual()]
                )
            }
        )
    );
    Ok(None)
}

/// Prints only the bits of the operation that we want visible on the graph's DOT
fn pretty(op: &Operation) -> String {
    let op_kind = match &op.kind {
        Operator::Identity(o) => o.short_name(),
        Operator::Load(o) => o.short_name(),
        Operator::Save(o) => o.short_name(),
        Operator::Send(o) => o.short_name(),
        Operator::Receive(o) => o.short_name(),
        Operator::Input(o) => o.short_name(),
        Operator::Output(o) => o.short_name(),
        Operator::Constant(o) => o.short_name(),
        Operator::Shape(o) => o.short_name(),
        Operator::BitFill(o) => o.short_name(),
        Operator::RingFill(o) => o.short_name(),
        Operator::AdtFill(o) => o.short_name(),
        Operator::StdAdd(o) => o.short_name(),
        Operator::StdSub(o) => o.short_name(),
        Operator::StdMul(o) => o.short_name(),
        Operator::StdDiv(o) => o.short_name(),
        Operator::StdDot(o) => o.short_name(),
        Operator::StdMean(o) => o.short_name(),
        Operator::StdExpandDims(o) => o.short_name(),
        Operator::StdReshape(o) => o.short_name(),
        Operator::StdAtLeast2D(o) => o.short_name(),
        Operator::StdSlice(o) => o.short_name(),
        Operator::StdSum(o) => o.short_name(),
        Operator::StdOnes(o) => o.short_name(),
        Operator::StdConcatenate(o) => o.short_name(),
        Operator::StdTranspose(o) => o.short_name(),
        Operator::StdInverse(o) => o.short_name(),
        Operator::RingAdd(o) => o.short_name(),
        Operator::RingSub(o) => o.short_name(),
        Operator::RingNeg(o) => o.short_name(),
        Operator::RingMul(o) => o.short_name(),
        Operator::RingDot(o) => o.short_name(),
        Operator::RingSum(o) => o.short_name(),
        Operator::RingSample(o) => o.short_name(),
        Operator::RingShl(o) => o.short_name(),
        Operator::RingShr(o) => o.short_name(),
        Operator::RingInject(o) => o.short_name(),
        Operator::BitExtract(o) => o.short_name(),
        Operator::BitSample(o) => o.short_name(),
        Operator::BitXor(o) => o.short_name(),
        Operator::BitAnd(o) => o.short_name(),
        Operator::PrimDeriveSeed(o) => o.short_name(),
        Operator::PrimPrfKeyGen(o) => o.short_name(),
        Operator::FixedpointRingEncode(o) => o.short_name(),
        Operator::FixedpointRingDecode(o) => o.short_name(),
        Operator::FixedpointRingMean(o) => o.short_name(),
        Operator::FixedAdd(o) => o.short_name(),
        Operator::FixedMul(o) => o.short_name(),
        Operator::AdtReveal(o) => o.short_name(),
        Operator::AdtAdd(o) => o.short_name(),
        Operator::AdtSub(o) => o.short_name(),
        Operator::AdtMul(o) => o.short_name(),
        Operator::AdtShl(o) => o.short_name(),
        Operator::RepSetup(o) => o.short_name(),
        Operator::RepShare(o) => o.short_name(),
        Operator::RepReveal(o) => o.short_name(),
        Operator::RepAdd(o) => o.short_name(),
        Operator::RepMul(o) => o.short_name(),
        Operator::RepTruncPr(o) => o.short_name(),
        Operator::RepToAdt(o) => o.short_name(),
        Operator::AdtToRep(o) => o.short_name(),
    };
    format!("{} = {}\\l{}", op.name, op_kind, op.placement.to_textual())
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
