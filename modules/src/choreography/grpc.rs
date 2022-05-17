mod gen {
    tonic::include_proto!("moose_choreography");
}

use self::gen::choreography_server::{Choreography, ChoreographyServer};
use self::gen::{RunComputationRequest, RunComputationResponse};
use async_trait::async_trait;
use moose::error::Error;
use moose::execution::Identity;
use tonic::transport::Server;

pub struct GrpcChoreography {
    pub own_identity: Identity,
}

impl GrpcChoreography {
    pub fn new_server(&self) -> ChoreographyServer<impl Choreography> {
        ChoreographyServer::new(ChoreographyImpl {})
    }
}

struct ChoreographyImpl {}

#[async_trait]
impl Choreography for ChoreographyImpl {
    async fn run_computation(
        &self,
        request: tonic::Request<RunComputationRequest>,
    ) -> Result<tonic::Response<RunComputationResponse>, tonic::Status> {
        unimplemented!()
    }
}
