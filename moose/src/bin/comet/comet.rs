//! Reindeer using gRPC choreography and gRPC networking.

use moose::choreography::grpc::GrpcChoreography;
use moose::networking::grpc::GrpcNetworkingManager;
use moose::prelude::*;
use moose::storage::filesystem::AsyncFilesystemStorage;
use moose::tokio;
use std::sync::Arc;
use structopt::StructOpt;
use tonic::transport::Server;

#[derive(Debug, StructOpt, Clone)]
pub struct Opt {
    #[structopt(env, long)]
    /// Own identity in sessions
    identity: String,

    #[structopt(env, long, default_value = "50000")]
    /// Port to use for gRPC server
    port: u16,

    #[structopt(env, long)]
    /// Directory to read certificates from
    certs: Option<String>,

    #[structopt(env, long)]
    /// Expected identity of choreographer; `certs` must be specified
    choreographer: Option<String>,

    #[structopt(long)]
    /// Report telemetry to Jaeger
    telemetry: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();
    if !opt.telemetry {
        tracing_subscriber::fmt::init();
    } else {
        moose::reindeer::setup_tracing(&opt.identity, "comet")?;
    }

    let root_span = tracing::span!(tracing::Level::INFO, "app_start");
    let _enter = root_span.enter();

    let my_cert_name = opt.identity.replace(':', "_");
    let own_identity = Identity::from(opt.identity);

    let networking = match opt.certs {
        Some(ref certs_dir) => {
            let client = moose::reindeer::load_client_tls_config(&my_cert_name, certs_dir)?;
            GrpcNetworkingManager::from_tls_config(client)
        }
        None => GrpcNetworkingManager::without_tls(),
    };

    let networking_server = networking.new_server();
    let choreography = GrpcChoreography::new(
        own_identity,
        opt.choreographer,
        Box::new(move |session_id| networking.new_session(session_id)),
        Box::new(|| Arc::new(AsyncFilesystemStorage::default())),
    );

    let mut server = Server::builder();

    if let Some(ref certs_dir) = opt.certs {
        let tls_server_config = moose::reindeer::load_server_tlc_config(&my_cert_name, certs_dir)?;
        server = server.tls_config(tls_server_config)?;
    }

    let router = server
        .add_service(networking_server)
        .add_service(choreography.into_server());

    let addr = format!("0.0.0.0:{}", &opt.port).parse()?;
    let res = router.serve(addr).await;
    if let Err(e) = res {
        tracing::error!("gRPC error: {}", e);
    }
    Ok(())
}
