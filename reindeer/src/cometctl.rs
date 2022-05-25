use clap::{Parser, Subcommand};
use moose_modules::choreography::{
    parse_session_config_file_with_computation, parse_session_config_file_without_computation,
};
use moose_modules::execution::grpc::GrpcMooseRuntime;
use std::path::PathBuf;

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
    Run {
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
            let (_, session_id, role_assignments, computation) =
                parse_session_config_file_with_computation(&session_config_file)?;
            let runtime = GrpcMooseRuntime::new(role_assignments)?;
            runtime
                .launch_computation(&session_id, &computation)
                .await?;
        }
        Commands::Abort {
            session_config: session_config_file,
        } => {
            let (_, session_id, role_assignments) =
                parse_session_config_file_without_computation(&session_config_file)?;
            let runtime = GrpcMooseRuntime::new(role_assignments)?;
            runtime.abort_computation(&session_id).await?;
        }
        Commands::Results {
            session_config: session_config_file,
        } => {
            let (_, session_id, role_assignments) =
                parse_session_config_file_without_computation(&session_config_file)?;
            let runtime = GrpcMooseRuntime::new(role_assignments)?;
            let results = runtime.retrieve_results(&session_id).await?;
            println!("Results: {:?}", results);
        }
        Commands::Run {
            session_config: session_config_file,
        } => {
            let (_, session_id, role_assignments, computation) =
                parse_session_config_file_with_computation(&session_config_file)?;
            let runtime = GrpcMooseRuntime::new(role_assignments)?;
            let results = runtime.run_computation(&session_id, &computation).await?;
            println!("Results: {:?}", results);
        }
    }

    Ok(())
}
