use maplit::hashmap;
use moose::computation::Role;
use moose::execution::Identity;
use moose::prelude::*;
use moose::storage::LocalAsyncStorage;
use moose_modules::networking::tcpstream::TcpStreamNetworking;
use moose_modules::storage::csv::read_csv;
use std::convert::TryFrom;
use std::fs::File;
use std::sync::Arc;
use std::{collections::HashMap, io::Read};
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

fn read_comp_file(filename: &str) -> anyhow::Result<Vec<u8>> {
    let mut file = File::open(filename)?;
    let mut vec = Vec::new();
    file.read_to_end(&mut vec)?;
    if vec.is_empty() {
        Err(anyhow::anyhow!("computation is empty"))
    } else {
        Ok(vec)
    }
}

#[tokio::main]
async fn main() {
    let opt = Opt::from_args();

    let hosts: HashMap<String, String> = serde_json::from_str(&opt.hosts).unwrap();

    let computation_bytes = read_comp_file(&opt.comp).unwrap();

    let computation = Computation::from_bytes(computation_bytes).unwrap();

    let input = read_csv(&opt.data, None, &[], &opt.placement.clone())
        .await
        .unwrap();

    let storage = Arc::new(LocalAsyncStorage::default());

    let networking = TcpStreamNetworking::new(&opt.placement, hosts)
        .await
        .unwrap();
    let networking = Arc::new(networking);

    let arguments = hashmap!["x".to_string() => input];

    let session_id = moose::computation::SessionId::try_from(opt.session_id.as_ref()).unwrap();

    let role_assignment_map: HashMap<String, String> =
        serde_json::from_str(&opt.role_assignment).unwrap();
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

    let outputs_handle = executor
        .run_computation(
            &computation,
            &role_assignment,
            &own_identity,
            &moose_session,
        )
        .unwrap();

    println!("joining on tasks");
    moose_session_handle.join_on_first_error().await.unwrap();

    let mut outputs = HashMap::new();

    println!("collecting outputs");
    for (output_name, output_future) in outputs_handle {
        println!("awaiting output: {}", output_name);
        let value = output_future.await.unwrap();
        println!("got output: {}", output_name);
        outputs.insert(output_name, value);
    }

    println!("outputs: {:?}", outputs);
}
