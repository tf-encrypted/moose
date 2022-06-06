use crate::choreography::grpc::gen::choreography_client::ChoreographyClient;
use crate::choreography::grpc::gen::{
    AbortComputationRequest, LaunchComputationRequest, RetrieveResultsRequest,
};
use moose::prelude::{Computation, Identity, Role, SessionId, Value};
use std::collections::HashMap;
use std::time::Duration;
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
        computation: &Computation,
        arguments: HashMap<String, Value>,
    ) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
        self.launch_computation(session_id, computation, arguments)
            .await?;
        self.retrieve_results(session_id).await
    }

    pub async fn run_computation_with_stats(
        &self,
        session_id: &SessionId,
        computation: &Computation,
        arguments: HashMap<String, Value>,
    ) -> Result<(HashMap<String, Value>, HashMap<String, Duration>), Box<dyn std::error::Error>>
    {
        self.launch_computation(session_id, computation, arguments)
            .await?;
        self.retrieve_results_with_stats(session_id).await
    }

    pub async fn launch_computation(
        &self,
        session_id: &SessionId,
        computation: &Computation,
        arguments: HashMap<String, Value>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let session_id = bincode::serialize(session_id)?;
        let computation = bincode::serialize(computation)?;
        let arguments = bincode::serialize(&arguments)?;
        let role_assignment = bincode::serialize(&self.role_assignments)?;

        for channel in self.channels.values() {
            let mut client = ChoreographyClient::new(channel.clone());

            // TODO(Morten) SECURITY: note that _all_ arguments are sent to _all_ workers;
            // this may still be okay/needed if/when we send value references around
            let request = LaunchComputationRequest {
                session_id: session_id.clone(),
                computation: computation.clone(),
                arguments: arguments.clone(),
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

    pub async fn retrieve_results_with_stats(
        &self,
        session_id: &SessionId,
    ) -> Result<(HashMap<String, Value>, HashMap<String, Duration>), Box<dyn std::error::Error>>
    {
        let session_id = bincode::serialize(&session_id)?;

        let mut combined_outputs = HashMap::new();
        let mut combined_stats = HashMap::new();

        for channel in self.channels.values() {
            let mut client = ChoreographyClient::new(channel.clone());

            let request = RetrieveResultsRequest {
                session_id: session_id.clone(),
            };

            let response = client.retrieve_results(request).await?;
            let vals = bincode::deserialize::<HashMap<String, Value>>(&response.get_ref().values)?;
            combined_outputs.extend(vals);

            let metrics = bincode::deserialize::<HashMap<String, Duration>>(
                &response.get_ref().metric_values,
            )?;
            combined_stats.extend(metrics);
        }

        Ok((combined_outputs, combined_stats))
    }
}
