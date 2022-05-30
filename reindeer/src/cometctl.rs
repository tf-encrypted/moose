use clap::{Parser, Subcommand};
use moose::computation::SessionId;
use moose_modules::choreography::{
    parse_session_config_file_with_computation, parse_session_config_file_without_computation,
};
use moose_modules::execution::grpc::GrpcMooseRuntime;
use std::convert::TryFrom;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[clap(name = "cometctl")]
#[clap(about = "A simple CLI for interacting with Comets")]
struct Cli {
    #[clap(subcommand)]
    command: Commands,

    #[clap(long)]
    /// Directory to read certificates from
    certs: Option<String>,

    #[clap(long)]
    /// Own identity; `certs` must be specified
    identity: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Launch {
        /// Session config file to use
        session_config: PathBuf,

        #[clap(long)]
        /// Session id to use
        session_id: Option<String>,
    },
    Abort {
        /// Session config file to use
        session_config: PathBuf,

        #[clap(long)]
        /// Session id to use
        session_id: Option<String>,
    },
    Results {
        /// Session config file to use
        session_config: PathBuf,

        #[clap(long)]
        /// Session id to use
        session_id: Option<String>,
    },
    Run {
        /// Session config file to use
        session_config: PathBuf,

        #[clap(long)]
        /// Session id to use
        session_id: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    tracing_subscriber::fmt::init();

    let tls_config = match (args.certs, args.identity) {
        (Some(certs_dir), Some(identity)) => {
            Some(reindeer::setup_tls_client(&identity, &certs_dir)?)
        }
        (None, None) => None,
        _ => panic!("both --certs and --identity must be specified"),
    };

    match args.command {
        Commands::Launch {
            session_config,
            session_id,
        } => {
            let (_, default_session_id, role_assignments, computation) =
                parse_session_config_file_with_computation(&session_config)?;
            let runtime = GrpcMooseRuntime::new(role_assignments, tls_config)?;
            let session_id = session_id
                .map(|session_id| SessionId::try_from(session_id.as_ref()))
                .unwrap_or(Ok(default_session_id))?;
            runtime
                .launch_computation(&session_id, &computation)
                .await?;
        }
        Commands::Abort {
            session_config,
            session_id,
        } => {
            let (_, default_session_id, role_assignments) =
                parse_session_config_file_without_computation(&session_config)?;
            let runtime = GrpcMooseRuntime::new(role_assignments, tls_config)?;
            let session_id = session_id
                .map(|session_id| SessionId::try_from(session_id.as_ref()))
                .unwrap_or(Ok(default_session_id))?;
            runtime.abort_computation(&session_id).await?;
        }
        Commands::Results {
            session_config,
            session_id,
        } => {
            let (_, default_session_id, role_assignments) =
                parse_session_config_file_without_computation(&session_config)?;
            let runtime = GrpcMooseRuntime::new(role_assignments, tls_config)?;
            let session_id = session_id
                .map(|session_id| SessionId::try_from(session_id.as_ref()))
                .unwrap_or(Ok(default_session_id))?;
            let results = runtime.retrieve_results(&session_id).await?;
            println!("Results: {:?}", results);
        }
        Commands::Run {
            session_config,
            session_id,
        } => {
            let (_, default_session_id, role_assignments, computation) =
                parse_session_config_file_with_computation(&session_config)?;
            let runtime = GrpcMooseRuntime::new(role_assignments, tls_config)?;
            let session_id = session_id
                .map(|session_id| SessionId::try_from(session_id.as_ref()))
                .unwrap_or(Ok(default_session_id))?;
            let results = runtime.run_computation(&session_id, &computation).await?;
            println!("Results: {:?}", results);
        }
    }

    Ok(())
}
