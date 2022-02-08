pub mod cape;

use crate::cape::csv::read_csv;
use crate::cape::networking_tcp_stream::TcpStreamNetworking;
use crate::cape::storage_stub::StubAsyncStorage;
use maplit::hashmap;
use moose::computation::Role;
use moose::execution::Identity;
use moose::prelude::*;
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

    //#[structopt(long)]
    //output: String,
    #[structopt(long)]
    placement: String,

    #[structopt(long)]
    role_assignments: String,

    #[structopt(long)]
    hosts: String,
}

fn read_comp_file(filename: &str) -> anyhow::Result<Vec<u8>> {
    let mut file = File::open(filename)?;
    let mut vec = Vec::new();
    file.read_to_end(&mut vec)?;
    Ok(vec)
}

#[tokio::main]
async fn main() {
    let opt = Opt::from_args();

    let _hosts: HashMap<String, String> = serde_json::from_str(&opt.hosts).unwrap();

    let computation_bytes = read_comp_file(&opt.comp).unwrap();

    let _computation = Computation::from_bytes(computation_bytes).unwrap();

    let input = read_csv(&opt.data, None, &[], &opt.placement)
        .await
        .unwrap();

    let _host = Arc::new(moose::computation::Placement::Host(HostPlacement {
        owner: opt.placement.into(),
    }));

    let storage = Arc::new(StubAsyncStorage::default());

    let networking = Arc::new(TcpStreamNetworking::default());

    let arguments = hashmap!["x".to_string() => input];

    let session_id = moose::computation::SessionId::try_from(opt.session_id.as_ref()).unwrap();

    let role_assignments_map: HashMap<String, String> =
        serde_json::from_str(&opt.role_assignments).unwrap();
    let role_assignments: HashMap<Role, Identity> = role_assignments_map
        .iter()
        .map(|(key, value)| {
            let role = Role::from(key);
            let identity = Identity::from(value);
            (role, identity)
        })
        .collect();

    let _moose_session = moose::execution::AsyncSession::new(
        session_id,
        arguments.clone(),
        role_assignments.clone(),
        networking,
        storage,
    );
}
