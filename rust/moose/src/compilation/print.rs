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
                        Operator::SendOp(_),
                        Operator::ReceiveOp(ReceiveOp {
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
        Operator::IdentityOp(_) => "Identity",
        Operator::LoadOp(_) => "Load",
        Operator::SaveOp(_) => "Save",
        Operator::SendOp(_) => "Send",
        Operator::ReceiveOp(_) => "Receive",
        Operator::InputOp(_) => "Input",
        Operator::OutputOp(_) => "Output",
        Operator::ConstantOp(_) => "Constant",
        Operator::StdAddOp(_) => "StdAdd",
        Operator::StdSubOp(_) => "StdSub",
        Operator::StdMulOp(_) => "StdMul",
        Operator::StdDivOp(_) => "StdDiv",
        Operator::StdDotOp(_) => "StdDot",
        Operator::StdMeanOp(_) => "StdMean",
        Operator::StdExpandDimsOp(_) => "StdExpandDims",
        Operator::StdReshapeOp(_) => "StdReshape",
        Operator::StdAtLeast2DOp(_) => "StdAtLeast2D",
        Operator::StdShapeOp(_) => "StdShape",
        Operator::StdSliceOp(_) => "StdSlice",
        Operator::StdSumOp(_) => "StdSum",
        Operator::StdOnesOp(_) => "StdOnes",
        Operator::StdConcatenateOp(_) => "StdConcatenate",
        Operator::StdTransposeOp(_) => "StdTranspose",
        Operator::StdInverseOp(_) => "StdInverse",
        Operator::RingAddOp(_) => "RingAdd",
        Operator::RingSubOp(_) => "RingSub",
        Operator::RingMulOp(_) => "RingMul",
        Operator::RingDotOp(_) => "RingDot",
        Operator::RingSumOp(_) => "RingSum",
        Operator::RingShapeOp(_) => "RingShape",
        Operator::RingSampleOp(_) => "RingSample",
        Operator::RingFillOp(_) => "RingFill",
        Operator::RingShlOp(_) => "RingShl",
        Operator::RingShrOp(_) => "RingShr",
        Operator::RingInjectOp(_) => "RingInject",
        Operator::BitExtractOp(_) => "BitExtract",
        Operator::BitSampleOp(_) => "BitSample",
        Operator::BitFillOp(_) => "BitFill",
        Operator::BitXorOp(_) => "BitXor",
        Operator::BitAndOp(_) => "BitAnd",
        Operator::PrimDeriveSeedOp(_) => "PrimDeriveSeed",
        Operator::PrimGenPrfKeyOp(_) => "PrimGenPrfKey",
        Operator::FixedpointRingEncodeOp(_) => "FixedpointRingEncode",
        Operator::FixedpointRingDecodeOp(_) => "FixedpointRingDecode",
        Operator::FixedpointRingMeanOp(_) => "FixedpointRingMean",
        Operator::FixedAddOp(_) => "FixedAdd",
        Operator::FixedMulOp(_) => "FixedMul",
        Operator::AdtRevealOp(_) => "AdtReveal",
        Operator::AdtAddOp(_) => "AdtAdd",
        Operator::AdtMulOp(_) => "AdtMul",
        Operator::RepSetupOp(_) => "RepSetup",
        Operator::RepShareOp(_) => "RepShare",
        Operator::RepRevealOp(_) => "RepReveal",
        Operator::RepAddOp(_) => "RepAdd",
        Operator::RepMulOp(_) => "RepMul",
        Operator::RepToAdtOp(_) => "RepToAdt",
    };
    format!("{} = {}\\l{}", op.name, op_kind, op.placement.to_textual())
}

fn shape(op: &Operation) -> String {
    match op.kind {
        Operator::InputOp(_) => "invhouse".into(),
        Operator::OutputOp(_) => "house".into(),
        Operator::SendOp(_) => "rarrow".into(),
        Operator::ReceiveOp(_) => "larrow".into(),
        _ => "rectangle".into(),
    }
}
