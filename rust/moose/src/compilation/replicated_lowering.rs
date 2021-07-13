use std::collections::HashMap;

use crate::{computation::*, kernels::Session, symbolic::SymbolicSession};

pub fn replicated_lowering(comp: &Computation) -> anyhow::Result<Option<Computation>> {
    let sess = SymbolicSession::default();
    let mut env: HashMap<String, SymbolicValue> = HashMap::default();

    for op in comp.operations.iter() {
        let operator = op.kind.clone();
        let operands = op
            .inputs
            .iter()
            .map(|input_name| env.get(input_name).unwrap().clone())
            .collect();
        let res = sess.execute(operator, &op.placement, operands);
        env.insert(op.name.clone(), res);
    }

    println!("\n\n\nDUMPING ENTIRE ENV\n{:?}\n\n\n", env);

    let ops = sess.ops.read().unwrap();

    Ok(Some(Computation {
        operations: ops.clone(),
    }))
}
