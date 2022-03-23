use crate::computation::Computation;
use crate::execution::SymbolicExecutor;

pub fn lowering(comp: Computation) -> anyhow::Result<Computation> {
    SymbolicExecutor::default().run_computation(&comp)
}
