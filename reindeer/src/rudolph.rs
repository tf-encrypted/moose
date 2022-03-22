use moose::prelude::*;
use moose::storage::LocalAsyncStorage;
use moose_modules::choreography::filesystem::FilesystemChoreography;
use moose_modules::networking::grpc::GrpcNetworkingManager;
use std::sync::Arc;
use structopt::StructOpt;
use tonic::transport::{Certificate, ClientTlsConfig, ServerTlsConfig};

#[derive(Debug, StructOpt, Clone)]
struct Opt {
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

const CA_NAME: &str = "ca";
pub fn certificate(endpoint: &str) -> String {
    endpoint.replace(':', "_")
}

fn setup_tls_server(
    my_cert_name: &str,
    certs_dir: &str,
) -> Result<ServerTlsConfig, Box<dyn std::error::Error>> {
    use tonic::transport::Identity;
    let cert_raw = std::fs::read(format!("{}/{}.crt", certs_dir, my_cert_name))?;
    let key_raw = std::fs::read(format!("{}/{}.key", certs_dir, my_cert_name))?;
    let identity = Identity::from_pem(cert_raw, key_raw);

    let ca_cert_raw = std::fs::read(format!("{}/{}.crt", certs_dir, CA_NAME))?;
    let ca_cert = Certificate::from_pem(ca_cert_raw);

    let server_tls = ServerTlsConfig::new()
        .identity(identity)
        .client_ca_root(ca_cert);

    Ok(server_tls)
}

fn setup_tls_client(
    my_cert_name: &str,
    certs_dir: &str,
) -> Result<ClientTlsConfig, Box<dyn std::error::Error>> {
    use tonic::transport::Identity;
    let server_root_ca_cert =
        Certificate::from_pem(std::fs::read(format!("{}/{}.crt", certs_dir, CA_NAME))?);

    let client_cert = std::fs::read(format!("{}/{}.crt", certs_dir, my_cert_name))?;
    let client_key = std::fs::read(format!("{}/{}.key", certs_dir, my_cert_name))?;
    let client_identity = Identity::from_pem(client_cert, client_key);

    let client_tls = ClientTlsConfig::new()
        .ca_certificate(server_root_ca_cert)
        .identity(client_identity);

    Ok(client_tls)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    if !opt.telemetry {
        tracing_subscriber::fmt::init();
    } else {
        use opentelemetry::sdk::trace::Config;
        use opentelemetry::sdk::Resource;
        use opentelemetry::KeyValue;
        use tracing_subscriber::{prelude::*, EnvFilter};

        let tracer =
            opentelemetry_jaeger::new_pipeline()
                .with_service_name("rudolph")
                .with_trace_config(Config::default().with_resource(Resource::new(vec![
                    KeyValue::new("identity", opt.identity.clone()),
                ])))
                .install_simple()?;
        let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

        tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(telemetry)
            .try_init()?;
    };

    let root_span = tracing::span!(tracing::Level::INFO, "app_start");
    let _enter = root_span.enter();

    let my_cert_name = certificate(&opt.identity);

    let manager = match opt.certs {
        Some(certs_dir) => {
            let client = setup_tls_client(&my_cert_name, &certs_dir)?;
            let server = setup_tls_server(&my_cert_name, &certs_dir)?;
            GrpcNetworkingManager::from_tls_config(client, server)
        }
        None => GrpcNetworkingManager::without_tls(),
    };
    let _server_task = manager.start_server(opt.port)?;
    let own_identity = Identity::from(opt.identity);

    // NOTE(Morten) if we want to move this into separate task then we need
    // to make sure AsyncSessionHandle::join_on_first_error is Send, which
    // means fixing the use of RwLock
    FilesystemChoreography::new(
        own_identity,
        opt.sessions,
        Box::new(move |session_id| manager.new_session(session_id)),
        Box::new(|| Arc::new(LocalAsyncStorage::default())),
    )
    .process(opt.ignore_existing, opt.no_listen)
    .await?;

    Ok(())
}
