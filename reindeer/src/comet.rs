use moose::prelude::*;
use moose::storage::LocalAsyncStorage;
use moose_modules::networking::grpc::GrpcNetworkingManager;
use moose_modules::choreography::grpc::GrpcChoreography;
use std::sync::Arc;
use structopt::StructOpt;
use tonic::transport::{ClientTlsConfig, Server, ServerTlsConfig};

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

fn setup_tls_server(
    my_cert_name: &str,
    certs_dir: &str,
) -> Result<ServerTlsConfig, Box<dyn std::error::Error>> {
    let (identity, ca_cert) = reindeer::load_identity_and_ca(my_cert_name, certs_dir)?;
    let server_tls = ServerTlsConfig::new()
        .identity(identity)    
        .client_ca_root(ca_cert);
    Ok(server_tls)
}

fn setup_tls_client(
    my_cert_name: &str,
    certs_dir: &str,
) -> Result<ClientTlsConfig, Box<dyn std::error::Error>> {
    let (client_identity, ca_cert) = reindeer::load_identity_and_ca(my_cert_name, certs_dir)?;
    let client_tls = ClientTlsConfig::new()
        .identity(client_identity)    
        .ca_certificate(ca_cert);
    Ok(client_tls)
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();
    reindeer::setup_tracing(opt.telemetry, &opt.identity)?;

    let root_span = tracing::span!(tracing::Level::INFO, "app_start");
    let _enter = root_span.enter();

    let my_cert_name = certificate(&opt.identity);
    let own_identity = Identity::from(opt.identity);

    let networking = match opt.certs {
        Some(ref certs_dir) => {
            let client = setup_tls_client(&my_cert_name, &certs_dir)?;
            GrpcNetworkingManager::from_tls_config(client)
        }
        None => GrpcNetworkingManager::without_tls(),
    };

    let choreography = GrpcChoreography {
        own_identity: own_identity.clone(),
    };

    let addr = format!("0.0.0.0:{}", &opt.port).parse()?;

    let mut server = Server::builder();

    match opt.certs {
        Some(ref certs_dir) => {
            let tls_server_config = setup_tls_server(&my_cert_name, &certs_dir)?;
            server = server.tls_config(tls_server_config)?;
        }
        None => (),
    };

    let router = server
        .add_service(networking.new_server())
        .add_service(choreography.new_server());

    let _server_task = tokio::spawn(async move {
        let res = router.serve(addr).await;
        if let Err(e) = res {
            tracing::error!("gRPC error: {}", e);
        }
    });

    Ok(())
}
