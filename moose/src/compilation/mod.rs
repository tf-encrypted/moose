use crate::compilation::networking::NetworkingPass;
use crate::compilation::print::print_graph;
use crate::compilation::pruning::prune_graph;
use crate::compilation::replicated_lowering::replicated_lowering;
use crate::compilation::typing::update_types_one_hop;
use crate::computation::Computation;
use crate::textual::ToTextual;

use self::deprecated_logical::deprecated_logical_lowering;

pub mod deprecated_logical;
pub mod networking;
pub mod print;
pub mod pruning;
pub mod replicated_lowering;
pub mod typing;

pub enum Pass {
    Networking,
    Print,
    Prune,
    Symbolic,
    Toposort,
    Typing,
    Dump,
    DeprecatedLogical, // A simple pass to support older Python compiler
}

fn parse_pass(name: &str) -> anyhow::Result<Pass> {
    match name {
        "networking" => Ok(Pass::Networking),
        "print" => Ok(Pass::Print),
        "prune" => Ok(Pass::Prune),
        "full" => Ok(Pass::Symbolic),
        "toposort" => Ok(Pass::Toposort),
        "typing" => Ok(Pass::Typing),
        "dump" => Ok(Pass::Dump),
        missing_pass => Err(anyhow::anyhow!("Unknown pass requested: {}", missing_pass)),
    }
}

pub fn into_pass(passes: &[String]) -> anyhow::Result<Vec<Pass>> {
    passes.iter().map(|s| parse_pass(s.as_str())).collect()
}

pub fn compile_passes(comp: &Computation, passes: &[Pass]) -> anyhow::Result<Computation> {
    let mut computation = comp.toposort()?;

    for pass in passes {
        if let Some(new_comp) = do_pass(pass, &computation)? {
            computation = new_comp;
        }
    }
    Ok(computation)
}

fn do_pass(pass: &Pass, comp: &Computation) -> anyhow::Result<Option<Computation>> {
    match pass {
        Pass::Networking => NetworkingPass::pass(comp),
        Pass::Print => print_graph(comp),
        Pass::Prune => prune_graph(comp),
        Pass::Symbolic => replicated_lowering(comp),
        Pass::Typing => update_types_one_hop(comp),
        Pass::DeprecatedLogical => deprecated_logical_lowering(comp),
        Pass::Dump => {
            println!("\nDumping a computation:\n{}\n\n", comp.to_textual());
            Ok(None)
        }
        Pass::Toposort => comp
            .toposort()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("Toposort failed due to {}", e)),
    }
}
