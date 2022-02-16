use moose::computation::Operator;
use moose::execution::{AsyncNetworkingImpl, AsyncStorageImpl};
use moose::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

pub struct ExecutionContext {
    own_identity: Identity,
    networking: AsyncNetworkingImpl,
    storage: AsyncStorageImpl,
}

impl ExecutionContext {
    pub fn new(
        own_identity: Identity,
        networking: AsyncNetworkingImpl,
        storage: AsyncStorageImpl,
    ) -> ExecutionContext {
        ExecutionContext {
            own_identity,
            networking,
            storage,
        }
    }

    pub async fn execute_computation(
        &self,
        session_id: SessionId,
        computation: &Computation,
        role_assignments: HashMap<Role, Identity>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let session = AsyncSession::new(
            session_id,
            HashMap::new(),
            role_assignments.clone(),
            Arc::clone(&self.networking),
            Arc::clone(&self.storage),
        );

        let mut env: HashMap<String, <AsyncSession as Session>::Value> = HashMap::default();
        let mut outputs: HashMap<String, <AsyncSession as Session>::Value> = HashMap::default();

        for op in computation.operations.iter() {
            // TODO(Morten) move filtering logic to the session
            match &op.placement {
                Placement::Host(host) => {
                    let owning_identity = role_assignments.get(&host.owner).unwrap();
                    if owning_identity == &self.own_identity {
                        // ok
                    } else {
                        // skip operation
                        continue;
                    }
                }
                _ => {
                    // skip operation
                    continue;
                }
            };

            let operands = op
                .inputs
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();

            let result = session.execute(op.kind.clone(), &op.placement, operands)?;

            if matches!(op.kind, Operator::Output(_)) {
                // If it is an output, we need to make sure we capture it for returning.
                outputs.insert(op.name.clone(), result.clone());
            } else {
                // Everything else should be available in the env for other ops to use.
                env.insert(op.name.clone(), result);
            }
        }

        for (output_name, output_value) in outputs {
            tokio::spawn(async move {
                let value = output_value.await.unwrap();
                tracing::info!("Output '{}': {:?}", output_name, value);
            });
        }

        let session_handle = AsyncSessionHandle::for_session(&session);
        session_handle.join_on_first_error().await?;

        Ok(())
    }
}
