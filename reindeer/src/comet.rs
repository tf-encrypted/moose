use moose::prelude::*;
use moose::storage::LocalAsyncStorage;
use moose_modules::choreography::grpc::GrpcChoreography;
use moose_modules::networking::grpc::GrpcNetworkingManager;
use std::sync::Arc;
use structopt::StructOpt;
use tonic::transport::Server;

#[derive(Debug, StructOpt, Clone)]
struct Opt {
    #[structopt(env, long)]
    /// Own identity in sessions
    identity: String,

    #[structopt(env, long, default_value = "50000")]
    /// Port to use for gRPC server
    port: u16,

    #[structopt(env, long)]
    /// Directory to read certificates from
    certs: Option<String>,

    #[structopt(long)]
    /// Report telemetry to Jaeger
    telemetry: bool,
}

pub fn certificate(endpoint: &str) -> String {
    endpoint.replace(':', "_")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();
    reindeer::setup_tracing(opt.telemetry, &opt.identity)?;

    let root_span = tracing::span!(tracing::Level::INFO, "app_start");
    let _enter = root_span.enter();

    let my_cert_name = opt.identity.replace(':', "_");
    let own_identity = Identity::from(opt.identity);

    let networking = match opt.certs {
        Some(ref certs_dir) => {
            let client = reindeer::setup_tls_client(&my_cert_name, &certs_dir)?;
            GrpcNetworkingManager::from_tls_config(client)
        }
        None => GrpcNetworkingManager::without_tls(),
    };

    let choreography = GrpcChoreography {
        own_identity: own_identity.clone(),
    };

    let mut server = Server::builder();

    if let Some(ref certs_dir) = opt.certs {
        let tls_server_config = reindeer::setup_tls_server(&my_cert_name, &certs_dir)?;
        server = server.tls_config(tls_server_config)?;
    }

    let router = server
        .add_service(networking.new_server())
        .add_service(choreography.new_server());

    let addr = format!("0.0.0.0:{}", &opt.port).parse()?;
    let _server_task = tokio::spawn(async move {
        let res = router.serve(addr).await;
        if let Err(e) = res {
            tracing::error!("gRPC error: {}", e);
        }
    });

    Ok(())
}
