use moose::computation::IndexedComputation;
use moose::computation::Operator;
use moose::execution::{AsyncNetworkingImpl, AsyncStorageImpl};
use moose::prelude::*;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;

pub struct ExecutionContext {
    own_identity: Identity,
    networking: AsyncNetworkingImpl,
    storage: AsyncStorageImpl,
}

type Environment = HashMap<String, <AsyncSession as Session>::Value>;

type CompactEnvironment = Vec<Option<<AsyncSession as Session>::Value>>;
type CompactOutputEnvironment = Vec<(usize, <AsyncSession as Session>::Value)>;

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

    #[tracing::instrument(skip(self, computation, role_assignments))]
    pub async fn execute_computation(
        &self,
        session_id: SessionId,
        computation: &NamedComputation,
        role_assignments: HashMap<Role, Identity>,
    ) -> Result<(AsyncSessionHandle, Environment), Box<dyn std::error::Error>> {
        let session = AsyncSession::new(
            session_id,
            HashMap::new(),
            role_assignments.clone(),
            Arc::clone(&self.networking),
            Arc::clone(&self.storage),
        );

        let mut outputs: Environment = HashMap::default();

        {
            let mut env: Environment = HashMap::with_capacity(computation.operations.len());

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
        }

        let handle = AsyncSessionHandle::for_session(&session);
        Ok((handle, outputs))
    }

    #[tracing::instrument(skip(self, computation, role_assignments))]
    pub async fn execute_compact_computation(
        &self,
        session_id: SessionId,
        computation: &NamedComputation,
        role_assignments: HashMap<Role, Identity>,
    ) -> Result<(AsyncSessionHandle, CompactOutputEnvironment), Box<dyn std::error::Error>> {
        let session = AsyncSession::new(
            session_id,
            HashMap::new(),
            role_assignments.clone(),
            Arc::clone(&self.networking),
            Arc::clone(&self.storage),
        );

        let computation = IndexedComputation::try_from(computation)?;
        let mut outputs: CompactOutputEnvironment = Vec::default();
        {
            let mut env: CompactEnvironment = Vec::with_capacity(computation.operations.len());

            for (op_index, op) in computation.operations.iter().enumerate() {
                // TODO(Morten) move filtering logic to the session
                let placement = computation.placements.get(op.placement).ok_or_else(|| {
                    moose::Error::MalformedComputation(format!(
                        "Missing placement for operation '{}'",
                        op_index
                    ))
                })?;
                match placement {
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
                        // skip operation
                        env.push(None);
                        continue;
                    }
                };

                let operands = op
                    .inputs
                    .iter()
                    .map(|input_index| env.get(*input_index).unwrap().clone().unwrap())
                    .collect();

                let operator =
                    computation
                        .operators
                        .get(op.operator)
                        .cloned()
                        .ok_or_else(|| {
                            moose::Error::MalformedComputation(format!(
                                "Missing operator for operation '{}'",
                                op_index
                            ))
                        })?;
                let is_output = matches!(&operator, Operator::Output(_));

                let result = session.execute(operator, placement, operands)?;

                if is_output {
                    // If it is an output, we need to make sure we capture it for returning.
                    outputs.push((op_index, result));
                } else {
                    // Everything else should be available in the env for other ops to use.
                    env.push(Some(result)); // assume computations are top sorted
                }
            }
        }

        let handle = AsyncSessionHandle::for_session(&session);
        Ok((handle, outputs))
    }
}
