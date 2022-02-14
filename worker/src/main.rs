mod gen {
    tonic::include_proto!("moose");
}
mod networking;

use crate::gen::networking_server::NetworkingServer;
use crate::networking::grpc::{Channels, GrpcNetworking, NetworkingImpl, SessionStores};
use moose::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::{convert::TryFrom, sync::Arc};
use structopt::StructOpt;
use tonic::transport::Channel;
use tonic::transport::{Server, Uri};

#[derive(Debug, StructOpt, Clone)]
struct Opt {
    #[structopt(env, long, default_value = "50000")]
    port: u16,

    #[structopt(env, long)]
    identity: String,

    #[structopt(env, long, default_value = "./examples")]
    sessions: String,
}

#[derive(Debug, Deserialize)]
struct SessionConfig {
    computation: ComputationConfig,
    roles: Vec<RoleConfig>,
}

#[derive(Debug, Deserialize)]
struct ComputationConfig {
    path: String,
}

#[derive(Debug, Deserialize)]
struct RoleConfig {
    name: String,
    endpoint: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let opt = Opt::from_args();

    let stores: Arc<SessionStores> = Arc::default();

    let server_task = {
        let stores = Arc::clone(&stores);
        // TODO(Morten) construct `addr` in a nicer way
        let addr = format!("0.0.0.0:{}", opt.port).parse()?;
        tokio::spawn(async move {
            let _server = Server::builder()
                .add_service(NetworkingServer::new(NetworkingImpl { stores }))
                .serve(addr)
                .await;
        })
    };

    // TODO(Morten) we should not have to do this; add retry logic on client side instead
    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

    let mut sessions: HashMap<SessionId, SessionConfig> = HashMap::new();
    let channels = Arc::default();

    let own_identity = Identity::from(opt.identity);
    for entry in std::fs::read_dir(opt.sessions)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            match path.extension() {
                Some(ext) if ext == "toml" => {
                    let config = std::fs::read_to_string(&path)?;
                    let session_config: SessionConfig = toml::from_str(&config)?;
                    let raw_session_id = path.file_stem().unwrap().to_string_lossy().to_string();
                    let session_id = SessionId::try_from(raw_session_id.as_str()).unwrap();
                    sessions.insert(session_id, session_config);
                }
                Some(ext) if ext == "moose" => {
                    // ok
                }
                _ => {
                    tracing::warn!("Skipping {:?}", path);
                }
            }
        }
    }

    for (session_id, session_config) in sessions {
        process_session(
            &stores,
            &channels,
            &own_identity,
            session_id,
            session_config,
        )
        .await?
    }

    tracing::debug!("Done launching");
    server_task.await?;

    Ok(())
}

#[tracing::instrument(skip(stores, channels, own_identity, session_id, session_config))]
async fn process_session(
    stores: &Arc<SessionStores>,
    channels: &Arc<Channels>,
    own_identity: &Identity,
    session_id: SessionId,
    session_config: SessionConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let comp_path = &session_config.computation.path;

    let comp_raw = std::fs::read_to_string(comp_path)?;
    let computation = moose::textual::verbose_parse_computation(&comp_raw)?;

    let mut role_assignments = HashMap::new();
    for role_config in session_config.roles {
        let role = Role::from(&role_config.name);
        let identity = Identity::from(&role_config.endpoint);

        role_assignments.insert(role, identity.clone());

        // TODO(Morten) better Uri construction
        let endpoint: Uri = format!("http://{}", role_config.endpoint).parse()?;
        channels
            .entry(identity)
            .or_insert_with(|| Channel::builder(endpoint).connect_lazy());
    }

    let networking = Arc::new(GrpcNetworking {
        stores: Arc::clone(stores),
        channels: Arc::clone(channels),
    });
    let storage = Arc::new(moose::storage::LocalAsyncStorage::default());
    let session = AsyncSession::new(
        session_id,
        HashMap::new(),
        role_assignments.clone(),
        networking,
        storage,
    );

    let mut env: HashMap<String, <AsyncSession as Session>::Value> = HashMap::default();
    let mut outputs: HashMap<String, <AsyncSession as Session>::Value> = HashMap::default();

    for op in computation.operations {
        // TODO(Morten) move filtering logic to the session
        match &op.placement {
            Placement::Host(host) => {
                let owning_identity = role_assignments.get(&host.owner).unwrap();
                if owning_identity == own_identity {
                    // ok
                } else {
                    // skip operation
                    continue;
                }
            }
            _ => {
                // skip operation
                continue;
            }
        };

        let operands = op
            .inputs
            .iter()
            .map(|input_name| env.get(input_name).unwrap().clone())
            .collect();

        let result = session.execute(op.kind.clone(), &op.placement, operands)?;

        if matches!(op.kind, moose::computation::Operator::Output(_)) {
            // If it is an output, we need to make sure we capture it for returning.
            outputs.insert(op.name.clone(), result.clone());
        } else {
            // Everything else should be available in the env for other ops to use.
            env.insert(op.name.clone(), result);
        }
    }

    for (output_name, output_value) in outputs {
        tokio::spawn(async move {
            let value = output_value.await.unwrap();
            tracing::info!("Output '{}': {:?}", output_name, value);
        });
    }

    let session_handle = AsyncSessionHandle::for_session(&session);
    session_handle.join_on_first_error().await?;

    Ok(())
}
