use moose::prelude::*;
use moose::storage::LocalAsyncStorage;
use moose_modules::choreography::filesystem::FilesystemChoreography;
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

    #[structopt(env, long, default_value = "./examples")]
    /// Directory to read sessions from
    sessions: String,

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
        // TODO(Morten) construct `addr` in a nicer way
        let addr = format!("0.0.0.0:{}", opt.port).parse()?;
        let manager = manager.clone();


        let tls = ServerTlsConfig::new()
            .identity(server_identity)
            .client_ca_root(client_ca_cert);

        tokio::spawn(async move {
            let res = Server::builder()
                .add_service(manager.new_server())
                .serve(addr)
                .await;
            if let Err(e) = res {
                tracing::error!("gRPC error: {}", e);
            }
        })
    };

    // TODO(Morten) we should not have to do this; add retry logic on client side instead
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

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
