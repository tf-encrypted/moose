use moose::computation::Operator;
use moose::computation::{CompactestComputation, CompactestOperation};
use moose::execution::{AsyncNetworkingImpl, AsyncStorageImpl};
use moose::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

pub struct CompactExecutionContext {
    own_identity: Identity,
    networking: AsyncNetworkingImpl,
    storage: AsyncStorageImpl,
}

type Environment = Vec<Option<<AsyncSession as Session>::Value>>;

type OutputEnvironment = Vec<(usize, <AsyncSession as Session>::Value)>;

impl CompactExecutionContext {
    pub fn new(
        own_identity: Identity,
        networking: AsyncNetworkingImpl,
        storage: AsyncStorageImpl,
    ) -> CompactExecutionContext {
        CompactExecutionContext {
            own_identity,
            networking,
            storage,
        }
    }

    pub async fn execute_computation(
        &self,
        session_id: SessionId,
        computation: &CompactestComputation,
        role_assignments: HashMap<Role, Identity>,
    ) -> Result<OutputEnvironment, Box<dyn std::error::Error>> {
        let session = AsyncSession::new(
            session_id,
            HashMap::new(),
            role_assignments.clone(),
            Arc::clone(&self.networking),
            Arc::clone(&self.storage),
        );

        let mut outputs: OutputEnvironment = Vec::default();

        {
            tracing::info!(
                "Computation has {:?} operations",
                computation.operations.len()
            );
            let mut env: Environment = Vec::with_capacity(computation.operations.len());

            for (op_index, op) in computation.operations.iter().enumerate() {
                // TODO(Morten) move filtering logic to the session
                match &computation.placements[op.placement_index] {
                    Placement::Host(host) => {
                        let owning_identity = role_assignments.get(&host.owner).unwrap();
                        if owning_identity == &self.own_identity {
                            // ok
                        } else {
                            // skip operation
                            env.push(None);
                            continue;
                        }
                    }
                    _ => {
                        env.push(None);
                        // skip operation
                        continue;
                    }
                };

                let operands = op
                    .inputs
                    .iter()
                    .map(|input_index| env.get(*input_index).unwrap().clone().unwrap().clone())
                    .collect();

                let result = session.execute(
                    computation.kinds[op.kind_index].clone(),
                    &computation.placements[op.placement_index],
                    operands,
                )?;

                if matches!(computation.kinds[op.kind_index], Operator::Output(_)) {
                    // If it is an output, we need to make sure we capture it for returning.
                    outputs.push((op_index, result.clone()));
                } else {
                    // Everything else should be available in the env for other ops to use.
                    env.push(Some(result)); // assume computations are top sorted
                }
            }
        }

        // let session_handle = AsyncSessionHandle::for_session(&session);
        // session_handle.join_on_first_error().await?;

        Ok(outputs)
    }
}
