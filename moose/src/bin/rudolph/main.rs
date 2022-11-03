//! Reindeer using file-based choreography and gRPC networking.

use moose::choreography::filesystem::FilesystemChoreography;
use moose::networking::grpc::GrpcNetworkingManager;
use moose::prelude::*;
use moose::storage::filesystem::AsyncFilesystemStorage;
use moose::tokio;
use std::sync::Arc;
use clap::StructOpt;
use tonic::transport::Server;

#[derive(Debug, StructOpt, Clone)]
pub struct Opt {
    #[structopt(env, long)]
    /// Own identity in sessions
    identity: String,

    #[structopt(env, long, default_value = "50000")]
    /// Port to use for gRPC server
    port: u16,

    #[structopt(env, long, default_value = "./examples")]
    /// Directory to read sessions from
    sessions: String,

    #[structopt(env, long)]
    /// Directory to read certificates from
    certs: Option<String>,

    #[structopt(long)]
    /// Ignore any existing files in the sessions directory and only listen for new
    ignore_existing: bool,

    #[structopt(long)]
    /// Do not listen for new files but exit when existing have been processed
    no_listen: bool,

    #[structopt(long)]
    /// Report telemetry to Jaeger
    telemetry: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::parse();
    if !opt.telemetry {
        tracing_subscriber::fmt::init();
    } else {
        moose::reindeer::setup_tracing(&opt.identity, "rudolph")?;
    }

    let root_span = tracing::span!(tracing::Level::INFO, "app_start");
    let _enter = root_span.enter();

    let my_cert_name = opt.identity.replace(':', "_");

    let manager = match opt.certs {
        Some(ref certs_dir) => {
            let client = moose::reindeer::load_client_tls_config(&my_cert_name, certs_dir)?;
            GrpcNetworkingManager::from_tls_config(client)
        }
        None => GrpcNetworkingManager::without_tls(),
    };

    let own_identity = Identity::from(opt.identity);

    let mut server = Server::builder();

    if let Some(ref certs_dir) = opt.certs {
        let tls_server_config = moose::reindeer::load_server_tlc_config(&my_cert_name, certs_dir)?;
        server = server.tls_config(tls_server_config)?;
    }

    let router = server.add_service(manager.new_server());

    let addr = format!("0.0.0.0:{}", &opt.port).parse()?;
    let _server_task = tokio::spawn(async move {
        let res = router.serve(addr).await;
        if let Err(e) = res {
            tracing::error!("gRPC error: {}", e);
        }
    });

    // NOTE(Morten) if we want to move this into separate task then we need
    // to make sure AsyncSessionHandle::join_on_first_error is Send, which
    // means fixing the use of RwLock
    FilesystemChoreography::new(
        own_identity,
        opt.sessions,
        Box::new(move |session_id| manager.new_session(session_id)),
        Box::new(|| Arc::new(AsyncFilesystemStorage::default())),
    )
    .process(opt.ignore_existing, opt.no_listen)
    .await?;

    Ok(())
}
