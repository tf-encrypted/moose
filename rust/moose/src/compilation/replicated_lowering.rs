use crate::{
    computation::Computation,
    symbolic::{SymbolicExecutor, SymbolicSession},
};

pub fn replicated_lowering(comp: &Computation) -> anyhow::Result<Option<Computation>> {
    let sess = SymbolicSession::default();
    let compiled = SymbolicExecutor::default().run_computation(comp, &sess)?;
    Ok(Some(compiled))
}
