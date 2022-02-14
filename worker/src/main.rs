mod gen {
    tonic::include_proto!("moose");
}
mod networking;

use crate::gen::networking_server::NetworkingServer;
use crate::networking::grpc::NetworkingImpl;
use tonic::{transport::Server, Request, Response, Status};
use structopt::StructOpt;

#[derive(Debug, StructOpt, Clone)]
struct Opt {
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    let addr = "0.0.0.0:50051".parse()?;

    let networking = NetworkingImpl::default();

    Server::builder()
        .add_service(NetworkingServer::new(networking))
        .serve(addr)
        .await?;

    Ok(())
}
