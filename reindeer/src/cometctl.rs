use clap::{Parser, Subcommand};
use moose::prelude::*;
use moose_modules::choreography::{Format, SessionConfig};
use moose_modules::execution::grpc::GrpcMooseRuntime;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser, Debug)]
#[clap(name = "cometctl")]
#[clap(about = "A simple CLI for interacting with Comets")]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Launch {
        /// Session config file to use
        session_config: PathBuf,
    },
    Abort {
        /// Session config file to use
        session_config: PathBuf,
    },
    Results {
        /// Session config file to use
        session_config: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    tracing_subscriber::fmt::init();

    match args.command {
        Commands::Launch {
            session_config: session_config_file,
        } => {
            let (session_id, computation, role_assignments) =
                parse_session_file(&session_config_file, true)?;
            let runtime = GrpcMooseRuntime::new(role_assignments)?;
            runtime
                .launch_computation(&session_id, &computation.unwrap())
                .await?;
        }
        Commands::Abort {
            session_config: session_config_file,
        } => {}
        Commands::Results {
            session_config: session_config_file,
        } => {
            let (session_id, _computation, role_assignments) =
                parse_session_file(&session_config_file, false)?;
            let runtime = GrpcMooseRuntime::new(role_assignments)?;
            let results = runtime.retrieve_results(&session_id).await?;
            tracing::info!("Results: {:?}", results);
        }
    }

    Ok(())
}

fn parse_session_file(
    session_config_file: &PathBuf,
    load_computation: bool,
) -> Result<(SessionId, Option<Computation>, RoleAssignments), Box<dyn std::error::Error>> {
    tracing::info!("Loading session from {:?}", session_config_file);

    let session_config = SessionConfig::from_str(&std::fs::read_to_string(session_config_file)?)?;

    let computation = {
        if load_computation {
            let comp_path = &session_config.computation.path;
            tracing::debug!("Loading computation from {:?}", comp_path);
            match session_config.computation.format {
                Format::Binary => {
                    let comp_raw = std::fs::read(comp_path)?;
                    Some(moose::computation::Computation::from_msgpack(comp_raw)?)
                }
                Format::Textual => {
                    let comp_raw = std::fs::read_to_string(comp_path)?;
                    Some(moose::computation::Computation::from_textual(&comp_raw)?)
                }
            }
        } else {
            None
        }
    };

    let role_assignments: RoleAssignments = session_config
        .roles
        .into_iter()
        .map(|role_config| {
            let role = Role::from(&role_config.name);
            let identity = Identity::from(&role_config.endpoint);
            (role, identity)
        })
        .collect();

    let session_id: SessionId = SessionId::try_from(
        session_config_file
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .borrow(),
    )?;

    Ok((session_id, computation, role_assignments))
}

type RoleAssignments = HashMap<Role, Identity>;
