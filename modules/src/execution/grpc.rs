use moose::prelude::{Computation, Identity, Role, SessionId};
use tonic::transport::{Channel, Uri};
use std::collections::HashMap;
use crate::choreography::grpc::gen::choreography_client::ChoreographyClient;
use crate::choreography::grpc::gen::LaunchComputationRequest;

pub struct GrpcMooseRuntime {
    role_assignments: HashMap<Role, Identity>,
    channels: HashMap<Role, Channel>,
}

impl GrpcMooseRuntime {
    pub fn new(role_assignments: HashMap<Role, Identity>) -> Result<GrpcMooseRuntime, Box<dyn std::error::Error>> {
        let channels = role_assignments.iter()
            .map(|(role, identity)| {
                let endpoint: Uri = format!("http://{}", identity).parse()?;
                let channel = Channel::builder(endpoint);
                // if let Some(ref tls_config) = self.tls_config {
                //     channel = channel.tls_config(tls_config.clone()).map_err(|e| {
                //         moose::Error::Networking(format!(
                //             "failed to TLS config {:?}",
                //             e.to_string()
                //         ))
                //     })?;
                // };
                let channel = channel.connect_lazy();
                Ok((role.clone(), channel))
            })
            .collect::<Result<_, Box<dyn std::error::Error>>>()?;

        Ok(GrpcMooseRuntime { role_assignments, channels })
    }

    pub async fn launch_computation(&self, session_id: &SessionId, comp: &Computation) -> Result<(), Box<dyn std::error::Error>> {
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

    pub fn abort_computation(&self, session_id: &SessionId) {
        // TODO(Morten) tell every identity in self.role_assignments to abort session_id
    }
}
