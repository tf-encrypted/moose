use crate::computation::Computation;
use crate::execution::SymbolicExecutor;

pub fn lowering(comp: &Computation) -> anyhow::Result<Option<Computation>> {
    let compiled = SymbolicExecutor::default().run_computation(comp)?;
    Ok(Some(compiled))
}
