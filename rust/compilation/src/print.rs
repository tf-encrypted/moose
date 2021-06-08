use moose::computation::{Computation, HostPlacement, Operation, Operator, Placement};
use petgraph::dot::Dot;
use petgraph::graph::{EdgeIndex, Graph};

/// Prints the computation's graph DOT representation to stdout
pub fn print_graph(comp: &Computation) -> anyhow::Result<Computation> {
    let graph = comp.as_graph();
    let graph2 = graph.map(
        |_, n| pretty(&comp.operations[n.1]),
        |e, _| pretty_edge(&graph, comp, e),
    );
    // Place Config::EdgeNoLabel inside the config element to have no labels on edges
    println!("{}", Dot::with_config(&graph2, &[]));
    Ok(Computation {
        operations: comp.operations.clone(),
    })
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
        Operator::RingMul(_) => "RingMul",
        Operator::RingDot(_) => "RingDot",
        Operator::RingSum(_) => "RingSum",
        Operator::RingShape(_) => "RingShape",
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
        Operator::PrimGenPrfKey(_) => "PrimGenPrfKey",
        Operator::FixedpointRingEncode(_) => "FixedpointRingEncode",
        Operator::FixedpointRingDecode(_) => "FixedpointRingDecode",
        Operator::FixedpointRingMean(_) => "FixedpointRingMean",
    };
    format!("{}, {}", op.name, op_kind)
}

fn pretty_edge(g: &Graph<(String, usize), ()>, comp: &Computation, e: EdgeIndex) -> String {
    let edge = &g.raw_edges()[e.index()];
    let source = &comp.operations[g[edge.source()].1];
    let target = &comp.operations[g[edge.target()].1];
    match (&source.placement, &target.placement) {
        (
            Placement::Host(HostPlacement { owner: host_a }),
            Placement::Host(HostPlacement { owner: host_b }),
        ) if host_a != host_b => format!("net {} to {}", host_a, host_b),
        _ => "".into(),
    }
}
