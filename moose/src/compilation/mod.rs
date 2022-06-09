//! Moose compilation framework.

use crate::computation::Computation;
use crate::textual::ToTextual;
use std::convert::TryFrom;

mod deprecated_shape;
mod lowering;
mod networking;
mod print;
mod pruning;
pub mod toposort;
mod typing;
mod well_formed;

/// Default compiler passes in order.
pub const DEFAULT_PASSES: [Pass; 6] = [
    Pass::Typing,
    Pass::DeprecatedShape,
    Pass::Lowering,
    Pass::Prune,
    Pass::Networking,
    Pass::Toposort,
];

/// Supported compiler passes.
#[derive(Clone)]
pub enum Pass {
    /// Insert send and receive nodes where needed.
    Networking,
    /// Print DOT representation of computation.
    Print,
    /// Print textual representation of computation.
    Dump,
    /// Prune unneeded operations.
    Prune,
    /// Lower computation.
    Lowering,
    /// Sort computation in topological order.
    Toposort,
    /// Perform basic type inference of operations.
    Typing,
    /// Check well-formedness.
    WellFormed,
    DeprecatedShape, // Support HostShape in the logical dialect (for pre-0.2.0 computations)
}

impl Pass {
    fn run(&self, comp: Computation) -> anyhow::Result<Computation> {
        match self {
            Pass::Toposort => self::toposort::toposort(comp),
            Pass::Networking => self::networking::networking_pass(comp),
            Pass::Print => self::print::print_graph(comp),
            Pass::Prune => self::pruning::prune_graph(comp),
            Pass::Lowering => self::lowering::lowering(comp),
            Pass::Typing => self::typing::update_types_one_hop(comp),
            Pass::WellFormed => self::well_formed::well_formed(comp),
            Pass::DeprecatedShape => self::deprecated_shape::deprecated_shape_support(comp),
            Pass::Dump => {
                println!("{}", comp.to_textual());
                Ok(comp)
            }
        }
    }
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
            "wellformed" => Ok(Pass::WellFormed),
            "dump" => Ok(Pass::Dump),
            "deprecatedShape" => Ok(Pass::DeprecatedShape),
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

/// Compile computation using specified or default passes.
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
