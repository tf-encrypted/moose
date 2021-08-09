use crate::compilation::networking::NetworkingPass;
use crate::compilation::print::print_graph;
use crate::compilation::pruning::prune_graph;
use crate::compilation::replicated_lowering::replicated_lowering;
use crate::compilation::typing::update_types_one_hop;
use crate::computation::Computation;
use crate::text_computation::ToTextual;

pub mod networking;
pub mod print;
pub mod pruning;
pub mod replicated_lowering;
pub mod typing;

pub fn compile_passes(comp: &Computation, passes: &[String]) -> anyhow::Result<Computation> {
    let mut computation = comp.toposort()?;

    for pass in passes {
        if let Some(new_comp) = do_pass(pass, &computation)? {
            computation = new_comp;
        }
    }
    Ok(computation)
}

fn do_pass(pass: &str, comp: &Computation) -> anyhow::Result<Option<Computation>> {
    match pass {
        "networking" => NetworkingPass::pass(comp),
        "print" => print_graph(comp),
        "prune" => prune_graph(comp),
        "replicated-lowering" => replicated_lowering(comp),
        "typing" => update_types_one_hop(comp),
        "dump" => {
            println!("\nDumping a computation:\n{}\n\n", comp.to_textual());
            Ok(None)
        }
        missing_pass => Err(anyhow::anyhow!("Unknwon pass requested: {}", missing_pass)),
    }
}
