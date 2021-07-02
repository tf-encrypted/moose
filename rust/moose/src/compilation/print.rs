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
    let op_kind = match op.kind {
        Operator::Identity(_) => "Identity",
        Operator::Load(_) => "Load",
        Operator::Save(_) => "Save",
        Operator::Send(_) => "Send",
        Operator::Receive(_) => "Receive",
        Operator::Input(_) => "Input",
        Operator::Output(_) => "Output",
        Operator::Constant(_) => "Constant",
        Operator::Shape(_) => "Shape",
        Operator::StdAdd(_) => "StdAdd",
        Operator::StdSub(_) => "StdSub",
        Operator::StdMul(_) => "StdMul",
        Operator::StdDiv(_) => "StdDiv",
        Operator::StdDot(_) => "StdDot",
        Operator::StdMean(_) => "StdMean",
        Operator::StdExpandDims(_) => "StdExpandDims",
        Operator::StdReshape(_) => "StdReshape",
        Operator::StdAtLeast2D(_) => "StdAtLeast2D",
        Operator::StdShape(_) => "StdShape",
        Operator::StdSlice(_) => "StdSlice",
        Operator::StdSum(_) => "StdSum",
        Operator::StdOnes(_) => "StdOnes",
        Operator::StdConcatenate(_) => "StdConcatenate",
        Operator::StdTranspose(_) => "StdTranspose",
        Operator::StdInverse(_) => "StdInverse",
        Operator::RingAdd(_) => "RingAdd",
        Operator::RingSub(_) => "RingSub",
        Operator::RingNeg(_) => "RingNeg",
        Operator::RingMul(_) => "RingMul",
        Operator::RingDot(_) => "RingDot",
        Operator::RingSum(_) => "RingSum",
        Operator::RingSample(_) => "RingSample",
        Operator::RingFill(_) => "RingFill",
        Operator::RingShl(_) => "RingShl",
        Operator::RingShr(_) => "RingShr",
        Operator::RingInject(_) => "RingInject",
        Operator::BitExtract(_) => "BitExtract",
        Operator::BitSample(_) => "BitSample",
        Operator::BitFill(_) => "BitFill",
        Operator::BitXor(_) => "BitXor",
        Operator::BitAnd(_) => "BitAnd",
        Operator::PrimDeriveSeed(_) => "PrimDeriveSeed",
        Operator::PrimPrfKeyGen(_) => "PrimPrfKeyGen",
        Operator::FixedpointRingEncode(_) => "FixedpointRingEncode",
        Operator::FixedpointRingDecode(_) => "FixedpointRingDecode",
        Operator::FixedpointRingMean(_) => "FixedpointRingMean",
        Operator::FixedAdd(_) => "FixedAdd",
        Operator::FixedMul(_) => "FixedMul",
        Operator::AdtReveal(_) => "AdtReveal",
        Operator::AdtAdd(_) => "AdtAdd",
        Operator::AdtSub(_) => "AdtSub",
        Operator::AdtMul(_) => "AdtMul",
        Operator::RepSetup(_) => "RepSetup",
        Operator::RepShare(_) => "RepShare",
        Operator::RepReveal(_) => "RepReveal",
        Operator::RepAdd(_) => "RepAdd",
        Operator::RepMul(_) => "RepMul",
        Operator::RepToAdt(_) => "RepToAdt",
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
