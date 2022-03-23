use self::deprecated_logical::deprecated_logical_lowering;
use crate::compilation::lowering::lowering;
use crate::compilation::networking::networking_pass;
use crate::compilation::print::print_graph;
use crate::compilation::pruning::prune_graph;
use crate::compilation::typing::update_types_one_hop;
use crate::computation::Computation;
use crate::textual::ToTextual;
use std::convert::TryFrom;

pub mod deprecated_logical;
pub mod lowering;
pub mod networking;
pub mod print;
pub mod pruning;
pub mod typing;

#[derive(Clone)]
pub enum Pass {
    Networking,
    Print,
    Prune,
    Lowering,
    Toposort,
    Typing,
    Dump,
    DeprecatedLogical, // A simple pass to support older Python compiler
}

impl TryFrom<&str> for Pass {
    type Error = anyhow::Error;

    fn try_from(name: &str) -> anyhow::Result<Pass> {
        match name {
            "networking" => Ok(Pass::Networking),
            "print" => Ok(Pass::Print),
            "prune" => Ok(Pass::Prune),
            "lowering" => Ok(Pass::Lowering),
            "toposort" => Ok(Pass::Toposort),
            "typing" => Ok(Pass::Typing),
            "dump" => Ok(Pass::Dump),
            missing_pass => Err(anyhow::anyhow!("Unknown pass requested: {}", missing_pass)),
        }
    }
}

impl TryFrom<&String> for Pass {
    type Error = anyhow::Error;
    fn try_from(name: &String) -> anyhow::Result<Pass> {
        Pass::try_from(name.as_str())
    }
}

impl TryFrom<&Pass> for Pass {
    type Error = anyhow::Error;
    fn try_from(pass: &Pass) -> anyhow::Result<Pass> {
        Ok(pass.clone())
    }
}

#[deprecated]
pub const DEFAULT_PASSES: [Pass; 5] = [
    Pass::Typing,
    Pass::Lowering,
    Pass::Prune,
    Pass::Networking,
    Pass::Toposort,
];

#[deprecated]
pub fn compile_passes<'p, P>(
    mut computation: Computation,
    passes: &'p [P],
) -> anyhow::Result<Computation>
where
    Pass: TryFrom<&'p P, Error = anyhow::Error>,
{
    let passes = passes
        .iter()
        .map(Pass::try_from)
        .collect::<anyhow::Result<Vec<Pass>>>()?;

    for pass in passes {
        computation = pass.run(computation)?;
    }
    Ok(computation)
}

pub fn compile<P>(comp: Computation, passes: Option<Vec<P>>) -> anyhow::Result<Computation>
where
    for<'p> Pass: TryFrom<&'p P, Error = anyhow::Error>,
{
    #[allow(deprecated)]
    match passes {
        None => compile_passes::<Pass>(comp, DEFAULT_PASSES.as_slice()),
        Some(passes) => {
            let passes = passes.as_ref();
            compile_passes(comp, passes)
        }
    }
}

impl Pass {
    fn run(&self, comp: Computation) -> anyhow::Result<Computation> {
        match self {
            Pass::Networking => networking_pass(comp),
            Pass::Print => print_graph(comp),
            Pass::Prune => prune_graph(comp),
            Pass::Lowering => lowering(comp),
            Pass::Typing => update_types_one_hop(comp),
            Pass::DeprecatedLogical => deprecated_logical_lowering(comp),
            Pass::Dump => {
                println!("\nDumping a computation:\n{}\n\n", comp.to_textual());
                Ok(comp)
            }
            Pass::Toposort => {
                let comp = comp
                    .toposort()
                    .map_err(|e| anyhow::anyhow!("Toposort failed due to {}", e))?;
                Ok(comp)
            }
        }
    }
}
