mod gen {
    tonic::include_proto!("moose");
}
mod choreography;
mod execution;
mod networking;

use crate::choreography::filesystem::FilesystemChoreography;
use crate::networking::grpc::GrpcNetworkingManager;
use moose::prelude::*;
use moose::storage::LocalAsyncStorage;
use std::sync::Arc;
use structopt::StructOpt;
use tonic::transport::Server;

#[derive(Debug, StructOpt, Clone)]
struct Opt {
    #[structopt(env, long, default_value = "50000")]
    port: u16,

    #[structopt(env, long)]
    identity: String,

    #[structopt(env, long, default_value = "./examples")]
    sessions: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let opt = Opt::from_args();

    let manager = GrpcNetworkingManager::default();

    let server_task = {
        // TODO(Morten) construct `addr` in a nicer way
        let addr = format!("0.0.0.0:{}", opt.port).parse()?;
        let manager = manager.clone();
        tokio::spawn(async move {
            let _res = Server::builder()
                .add_service(manager.new_server())
                .serve(addr)
                .await;
        })
    };
    tracing::debug!("gRPC server launched");

    // TODO(Morten) we should not have to do this; add retry logic on client side instead
    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

    let _res = FilesystemChoreography::new(
        Identity::from(opt.identity),
        opt.sessions,
        Box::new(move || manager.new_session()),
        Box::new(|| Arc::new(LocalAsyncStorage::default())),
    )
    .listen()
    .await;
    tracing::debug!("Choreography launched");

    server_task.await?;

    Ok(())
}
