use crate::choreography::grpc::gen::choreography_client::ChoreographyClient;
use crate::choreography::grpc::gen::{
    AbortComputationRequest, LaunchComputationRequest, RetrieveResultsRequest,
};
use crate::choreography::grpc::ComputationOutputs;
use crate::prelude::{Computation, Identity, Role, SessionId, Value};
use std::collections::HashMap;
use std::time::Duration;
use tonic::transport::{Channel, ClientTlsConfig, Uri};

pub struct GrpcMooseRuntime {
    role_assignments: HashMap<Role, Identity>,
    channels: HashMap<Role, Channel>,
}

#[derive(Debug)]
pub struct GrpcOutputs {
    pub outputs: HashMap<String, Value>,
    pub elapsed_time: Option<HashMap<Role, Duration>>,
}

impl GrpcMooseRuntime {
    pub fn new(
        role_assignments: HashMap<Role, Identity>,
        tls_config: Option<ClientTlsConfig>,
    ) -> Result<GrpcMooseRuntime, Box<dyn std::error::Error>> {
        let channels = role_assignments
            .iter()
            .map(|(role, identity)| {
                let endpoint: Uri = format!("http://{identity}").parse()?;
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
    ) -> Result<GrpcOutputs, Box<dyn std::error::Error>> {
        self.launch_computation(session_id, computation, arguments)
            .await?;
        self.retrieve_results(session_id).await
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
    ) -> Result<GrpcOutputs, Box<dyn std::error::Error>> {
        let session_id = bincode::serialize(&session_id)?;

        let mut combined_outputs = HashMap::new();
        let mut combined_stats = HashMap::new();

        for (role, channel) in self.channels.iter() {
            let mut client = ChoreographyClient::new(channel.clone());

            let request = RetrieveResultsRequest {
                session_id: session_id.clone(),
            };

            let response = client.retrieve_results(request).await?;

            let ComputationOutputs {
                outputs,
                elapsed_time,
            } = bincode::deserialize::<ComputationOutputs>(&response.get_ref().values)?;
            combined_outputs.extend(outputs);

            if let Some(time) = elapsed_time {
                combined_stats.insert(role.clone(), time);
            }
        }

        if combined_stats.is_empty() {
            Ok(GrpcOutputs {
                outputs: combined_outputs,
                elapsed_time: None,
            })
        } else {
            Ok(GrpcOutputs {
                outputs: combined_outputs,
                elapsed_time: Some(combined_stats),
            })
        }
    }
}
