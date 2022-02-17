use moose::prelude::*;
use moose::storage::LocalAsyncStorage;
use moose_choreography::choreography::filesystem::FilesystemChoreography;
use moose_networking::networking::grpc::GrpcNetworkingManager;
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

    #[structopt(env, long)]
    ignore_existing: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let opt = Opt::from_args();

    let manager = GrpcNetworkingManager::default();

    let _server_task = {
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

    // TODO(Morten) we should not have to do this; add retry logic on client side instead
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    let _res = FilesystemChoreography::new(
        Identity::from(opt.identity),
        opt.sessions,
        Box::new(move || manager.new_session()),
        Box::new(|| Arc::new(LocalAsyncStorage::default())),
    )
    .listen(opt.ignore_existing)
    .await?;

    Ok(())
}
