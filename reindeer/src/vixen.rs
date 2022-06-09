use moose::computation::Role;
use moose::execution::Identity;
use moose::prelude::*;
use moose::storage::local::LocalAsyncStorage;
use moose::tokio;
use moose::networking::tcpstream::TcpStreamNetworking;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;
use structopt::StructOpt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Debug, StructOpt, Clone)]
struct Opt {
    #[structopt(long)]
    comp: String,

    #[structopt(long)]
    session_id: String,

    #[structopt(long)]
    placement: String,

    #[structopt(long)]
    role_assignment: String,

    #[structopt(long)]
    hosts: String,
}

fn init_tracer() {
    let fmt_layer = Some(tracing_subscriber::fmt::Layer::default());
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env()) // The tracing formatter defaults to the max log level set by RUST_LOG
        .with(fmt_layer)
        .try_init()
        .unwrap_or_else(|e| println!("Failed to initialize telemetry subscriber: {}", e));
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracer();
    tracing::info!("starting up");
    let opt = Opt::from_args();

    let hosts: HashMap<String, String> = serde_json::from_str(&opt.hosts)?;

    let computation_bytes = std::fs::read(&opt.comp)?;

    let computation = Computation::from_msgpack(computation_bytes)?;

    let storage = Arc::new(LocalAsyncStorage::default());

    let networking = TcpStreamNetworking::new(&opt.placement, hosts).await?;
    let networking = Arc::new(networking);

    let arguments = HashMap::new();

    let session_id = moose::computation::SessionId::try_from(opt.session_id.as_ref())?;

    let role_assignment_map: HashMap<String, String> = serde_json::from_str(&opt.role_assignment)?;
    let role_assignment: HashMap<Role, Identity> = role_assignment_map
        .iter()
        .map(|(key, value)| {
            let role = Role::from(key);
            let identity = Identity::from(value);
            (role, identity)
        })
        .collect();

    let moose_session = moose::execution::AsyncSession::new(
        session_id,
        arguments,
        role_assignment.clone(),
        networking,
        storage,
    );

    let own_identity: Identity = Identity::from(&opt.placement);
    let mut executor = AsyncExecutor::default();

    let outputs_handle = executor.run_computation(
        &computation,
        &role_assignment,
        &own_identity,
        &moose_session,
    )?;

    let moose_session_handle = moose_session.into_handle()?;
    tracing::info!("joining on tasks");
    moose_session_handle.join_on_first_error().await?;

    let mut outputs = HashMap::new();

    tracing::info!("collecting outputs");
    for (output_name, output_future) in outputs_handle {
        tracing::info!("awaiting output: {}", output_name);
        let value = output_future.await.unwrap();
        tracing::info!("got output: {}", output_name);
        outputs.insert(output_name, value);
    }

    tracing::info!("outputs: {:?}", outputs);
    Ok(())
}
