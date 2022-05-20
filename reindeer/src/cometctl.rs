use clap::{Parser, Subcommand};
use moose::prelude::*;
use moose_modules::choreography::{Format, SessionConfig};
use moose_modules::execution::grpc::GrpcMooseRuntime;
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
        /// Session ID to use; defaults to the name of the session config
        #[clap(long)]
        session_id: Option<String>,

        /// Session config file to use
        session_config: PathBuf,
    },
    Abort {
        /// Session ID to use; defaults to the name of the session config
        #[clap(long)]
        session_id: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    tracing_subscriber::fmt::init();

    match args.command {
        Commands::Launch {
            session_id,
            session_config: session_config_file,
        } => {
            tracing::info!("Launching session from {:?}", session_config_file);

            let session_config =
                SessionConfig::from_str(&std::fs::read_to_string(&session_config_file)?)?;

            let computation = {
                let comp_path = &session_config.computation.path;
                tracing::debug!("Loading computation from {:?}", comp_path);
                match session_config.computation.format {
                    Format::Binary => {
                        let comp_raw = std::fs::read(comp_path)?;
                        moose::computation::Computation::from_msgpack(comp_raw)?
                    }
                    Format::Textual => {
                        let comp_raw = std::fs::read_to_string(comp_path)?;
                        moose::computation::Computation::from_textual(&comp_raw)?
                    }
                }
            };

            let role_assignments: HashMap<Role, Identity> = session_config
                .roles
                .into_iter()
                .map(|role_config| {
                    let role = Role::from(&role_config.name);
                    let identity = Identity::from(&role_config.endpoint);
                    (role, identity)
                })
                .collect();

            let session_id: SessionId = SessionId::try_from(
                session_id
                    .unwrap_or_else(|| {
                        session_config_file
                            .file_stem()
                            .unwrap()
                            .to_string_lossy()
                            .to_string()
                    })
                    .as_str(),
            )?;

            let runtime = GrpcMooseRuntime::new(role_assignments)?;
            runtime
                .launch_computation(&session_id, &computation)
                .await?;
        }
        Commands::Abort { session_id } => {}
    }

    Ok(())
}
