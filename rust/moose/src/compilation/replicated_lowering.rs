use crate::{
    computation::Computation,
    error::Error,
    symbolic::{SymbolicExecutor, SymbolicSession},
};

pub fn replicated_lowering(comp: &Computation) -> anyhow::Result<Option<Computation>> {
    let sess = SymbolicSession::default();
    SymbolicExecutor::default().run_computation(&comp.toposort()?, &sess);

    let ops = sess.ops.read().map_err(|e| {
        Error::Compilation(format!(
            "Failed to get operations from the Symbolic Session due to an error: {}",
            e
        ))
    })?;

    Ok(Some(Computation {
        operations: ops.clone(),
    }))
}
