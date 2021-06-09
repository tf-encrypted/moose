use tonic::{transport::Server, Request, Response, Status};

use grpc::moose::worker_server::{Worker, WorkerServer};
use grpc::moose::{RunComputationResponse, RunComputationRequest, GetValueResponse, GetValueRequest};


#[derive(Debug, Default)]
pub struct GrpcWorker {}

#[tonic::async_trait]
impl Worker for GrpcWorker {
    async fn run_computation(
        &self,
        request: Request<RunComputationRequest>,
    ) -> Result<Response<RunComputationResponse>, Status> {
        println!("Got a request: {:?}", request);

        Err(Status::unimplemented("todo"))
    }

    async fn get_value(
        &self,
        request: Request<GetValueRequest>,
    ) -> Result<Response<GetValueResponse>, Status> {
        println!("Got a request: {:?}", request);

        Err(Status::unimplemented("todo"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:50051".parse()?;

    let worker = GrpcWorker::default();

    Server::builder()
        .add_service(WorkerServer::new(worker))
        .serve(addr)
        .await?;

    Ok(())
}
