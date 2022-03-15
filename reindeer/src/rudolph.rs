use moose::prelude::*;
use moose::storage::LocalAsyncStorage;
use moose_modules::choreography::filesystem::FilesystemChoreography;
use moose_modules::networking::grpc::GrpcNetworkingManager;
use std::sync::Arc;
use structopt::StructOpt;

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

fn certificate(endpoint: &str) -> String {
    endpoint.replace(":", "_")
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

    let manager = GrpcNetworkingManager::default();

    let _server_task = {
        use tonic::transport::{Server, Identity, ServerTlsConfig, Certificate};

        // TODO(Morten) construct `addr` in a nicer way
        let addr = format!("0.0.0.0:{}", opt.port).parse()?;
        let manager = manager.clone();

        let mut server = Server::builder();

        if let Some(cert_dir) = opt.certs {
            let cert_name = certificate(&opt.identity);
            let cert_raw = tokio::fs::read(format!("examples/certs/{}.crt", cert_name)).await?;
            let key_raw = tokio::fs::read(format!("examples/certs/{}.key", cert_name)).await?;
            let identity = Identity::from_pem(cert_raw, key_raw);

            let ca_cert_raw = tokio::fs::read("examples/certs/ca.crt").await?;
            let ca_cert = Certificate::from_pem(ca_cert_raw);

            let tls = ServerTlsConfig::new()
                .identity(identity)
                .client_ca_root(ca_cert);

            server = server.tls_config(tls)?;
        }

        let router = server
            .add_service(manager.new_server());

        tokio::spawn(async move {
            if let Err(e) = router
                .serve(addr)
                .await {
                tracing::error!("gRPC error: {}", e);
            }
        })
    };

    // NOTE(Morten) if we want to move this into separate task then we need
    // to make sure AsyncSessionHandle::join_on_first_error is Send, which
    // means fixing the use of RwLock
    FilesystemChoreography::new(
        Identity::from(opt.identity),
        opt.sessions,
        Box::new(move |session_id| manager.new_session(session_id)),
        Box::new(|| Arc::new(LocalAsyncStorage::default())),
    )
    .process(opt.ignore_existing, opt.no_listen)
    .await?;

    Ok(())
}
