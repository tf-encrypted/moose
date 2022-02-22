use maplit::hashmap;
use moose::computation::Role;
use moose::execution::Identity;
use moose::prelude::*;
use moose::storage::LocalAsyncStorage;
use moose_modules::networking::tcpstream::TcpStreamNetworking;
use moose_modules::storage::csv::read_csv;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;
use structopt::StructOpt;

#[derive(Debug, StructOpt, Clone)]
struct Opt {
    #[structopt(long)]
    data: String,

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    let hosts: HashMap<String, String> = serde_json::from_str(&opt.hosts)?;

    let computation_bytes = std::fs::read(&opt.comp)?;

    let computation = Computation::from_bytes(computation_bytes)?;

    let input = read_csv(&opt.data, None, &[], &opt.placement.clone()).await?;

    let storage = Arc::new(LocalAsyncStorage::default());

    let networking = TcpStreamNetworking::new(&opt.placement, hosts).await?;
    let networking = Arc::new(networking);

    let arguments = hashmap!["x".to_string() => input];

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

    let moose_session_handle = AsyncSessionHandle::for_session(&moose_session);

    let own_identity: Identity = Identity::from(&opt.placement);
    let mut executor = AsyncExecutor::default();

    let outputs_handle = executor.run_computation(
        &computation,
        &role_assignment,
        &own_identity,
        &moose_session,
    )?;

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
