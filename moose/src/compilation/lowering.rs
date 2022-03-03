use crate::computation::NamedComputation;
use crate::execution::{SymbolicExecutor, SymbolicSession};

pub fn lowering(comp: &NamedComputation) -> anyhow::Result<Option<NamedComputation>> {
    let sess = SymbolicSession::default();
    let compiled = SymbolicExecutor::default().run_computation(comp, &sess)?;
    Ok(Some(compiled))
}
