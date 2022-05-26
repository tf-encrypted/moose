use crate::choreography::grpc::gen::choreography_client::ChoreographyClient;
use crate::choreography::grpc::gen::{
    AbortComputationRequest, LaunchComputationRequest, RetrieveResultsRequest,
};
use moose::prelude::{Computation, Identity, Role, SessionId, Value};
use std::collections::HashMap;
use tonic::transport::{Channel, ClientTlsConfig, Uri};

pub struct GrpcMooseRuntime {
    role_assignments: HashMap<Role, Identity>,
    channels: HashMap<Role, Channel>,
}

impl GrpcMooseRuntime {
    pub fn new(
        role_assignments: HashMap<Role, Identity>,
        tls_config: Option<ClientTlsConfig>,
    ) -> Result<GrpcMooseRuntime, Box<dyn std::error::Error>> {
        let channels = role_assignments
            .iter()
            .map(|(role, identity)| {
                let endpoint: Uri = format!("http://{}", identity).parse()?;
                let mut channel = Channel::builder(endpoint);
                if let Some(ref tls_config) = tls_config {
                    channel = channel.tls_config(tls_config.clone())?;
                };
                let channel = channel.connect_lazy();
                Ok((role.clone(), channel))
            })
            .collect::<Result<_, Box<dyn std::error::Error>>>()?;

        Ok(GrpcMooseRuntime {
            role_assignments,
            channels,
        })
    }

    pub async fn run_computation(
        &self,
        session_id: &SessionId,
        comp: &Computation,
    ) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
        self.launch_computation(session_id, comp).await?;
        self.retrieve_results(session_id).await
    }

    pub async fn launch_computation(
        &self,
        session_id: &SessionId,
        comp: &Computation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let session_id = bincode::serialize(&session_id)?;
        let computation = bincode::serialize(&comp)?;
        let role_assignment = bincode::serialize(&self.role_assignments)?;

        for channel in self.channels.values() {
            let mut client = ChoreographyClient::new(channel.clone());

            let request = LaunchComputationRequest {
                session_id: session_id.clone(),
                computation: computation.clone(),
                role_assignment: role_assignment.clone(),
            };

            let _response = client.launch_computation(request).await?;
        }

        Ok(())
    }

    pub async fn abort_computation(
        &self,
        session_id: &SessionId,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let session_id = bincode::serialize(&session_id)?;

        for channel in self.channels.values() {
            let mut client = ChoreographyClient::new(channel.clone());

            let request = AbortComputationRequest {
                session_id: session_id.clone(),
            };

            let _response = client.abort_computation(request).await?;
        }

        Ok(())
    }

    pub async fn retrieve_results(
        &self,
        session_id: &SessionId,
    ) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
        let session_id = bincode::serialize(&session_id)?;

        let mut combined_results = HashMap::new();
        for channel in self.channels.values() {
            let mut client = ChoreographyClient::new(channel.clone());

            let request = RetrieveResultsRequest {
                session_id: session_id.clone(),
            };

            let response = client.retrieve_results(request).await?;
            let vals = bincode::deserialize::<HashMap<String, Value>>(&response.get_ref().values)?;
            combined_results.extend(vals);
        }

        Ok(combined_results)
    }
}
