pub(crate) mod gen {
    tonic::include_proto!("moose_choreography");
}

use crate::execution::ExecutionContext;
use self::gen::choreography_server::{Choreography, ChoreographyServer};
use self::gen::{
    AbortComputationRequest, AbortComputationResponse, LaunchComputationRequest,
    LaunchComputationResponse,
};
use super::{NetworkingStrategy, StorageStrategy};
use async_trait::async_trait;
use dashmap::DashMap;
use moose::computation::{Computation, Role, SessionId, Value};
use moose::execution::Identity;
use std::collections::HashMap;
use std::sync::Arc;

type ResultsStores = DashMap<SessionId, HashMap<String, Value>>;

pub struct GrpcChoreography {
    own_identity: Arc<Identity>,
    result_stores: Arc<ResultsStores>,
    networking_strategy: NetworkingStrategy,
    storage_strategy: StorageStrategy,
}

impl GrpcChoreography {
    pub fn new(
        own_identity: Identity,
        networking_strategy: NetworkingStrategy,
        storage_strategy: StorageStrategy,
    ) -> GrpcChoreography {
        GrpcChoreography {
            own_identity: Arc::new(own_identity),
            result_stores: Arc::new(ResultsStores::default()),
            networking_strategy,
            storage_strategy,
        }
    }

    pub fn into_server(self) -> ChoreographyServer<impl Choreography> {
        ChoreographyServer::new(self)
    }
}

#[async_trait]
impl Choreography for GrpcChoreography {
    async fn launch_computation(
        &self,
        request: tonic::Request<LaunchComputationRequest>,
    ) -> Result<tonic::Response<LaunchComputationResponse>, tonic::Status> {
        tracing::info!("Launching computation");

        // TODO(Morten) extract session_id, computation, and role_assignments; then create new execution context and launch

        let request = request.into_inner();

        let session_id = bincode::deserialize::<SessionId>(&request.session_id).map_err(|_e| {
            tonic::Status::new(
                tonic::Code::Aborted,
                "failed to parse session id".to_string(),
            )
        })?;

        let computation =
            bincode::deserialize::<Computation>(&request.computation).map_err(|_e| {
                tonic::Status::new(
                    tonic::Code::Aborted,
                    "failed to parse computation".to_string(),
                )
            })?;

        let role_assignments = bincode::deserialize::<HashMap<Role, Identity>>(
            &request.role_assignment,
        )
        .map_err(|_e| {
            tonic::Status::new(
                tonic::Code::Aborted,
                "failed to parse role assignment".to_string(),
            )
        })?;

        let networking = (self.networking_strategy)(session_id.clone());
        let storage = (self.storage_strategy)();
        let context =
            ExecutionContext::new(self.own_identity.as_ref().clone(), networking, storage);

        let (_handle, outputs) = context
            .execute_computation(session_id.clone(), &computation, role_assignments)
            .await
            .map_err(|_e| {
                tonic::Status::new(
                    tonic::Code::Aborted,
                    "failed launch computation".to_string(),
                )
            })?;

        let result_stores = Arc::clone(&self.result_stores);
        tokio::spawn(async move {
            let mut results = HashMap::with_capacity(outputs.len());
            for (output_name, output_value) in outputs {
                let value = output_value.await.unwrap();
                results.insert(output_name, value);
            }
            tracing::info!("Results ready, {:?}", results.keys());
            result_stores.insert(session_id, results);
        });

        Ok(tonic::Response::new(LaunchComputationResponse::default()))
    }

    async fn abort_computation(
        &self,
        request: tonic::Request<AbortComputationRequest>,
    ) -> Result<tonic::Response<AbortComputationResponse>, tonic::Status> {
        unimplemented!()
    }
}
