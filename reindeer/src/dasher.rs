use moose::computation::Operator;
use moose::prelude::*;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::sync::Arc;
use structopt::StructOpt;

#[derive(Debug, StructOpt, Clone)]
#[structopt(about = "Run computation locally by simulating all roles as seperate identities")]
struct Opt {
    computation: String,

    #[structopt(short, long)]
    binary: bool,

    #[structopt(short, long, default_value = "dasher-session")]
    session_id: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let opt = Opt::from_args();

    let computation = {
        let comp_path = &opt.computation;
        if opt.binary {
            let comp_raw = std::fs::read(comp_path)?;
            moose::computation::Computation::from_msgpack(&comp_raw)?
        } else {
            let comp_raw = std::fs::read_to_string(comp_path)?;
            moose::computation::Computation::from_textual(&comp_raw)?
        }
    };

    let session_id = SessionId::try_from(opt.session_id.as_str())?;

    let role_assignments: HashMap<Role, Identity> = {
        let roles: HashSet<Role> = computation
            .operations
            .iter()
            .flat_map(|op| match &op.placement {
                Placement::Host(plc) => vec![plc.owner.clone()],
                Placement::Replicated(plc) => plc.owners.to_vec(),
                Placement::Mirrored3(plc) => plc.owners.to_vec(),
                Placement::Additive(plc) => plc.owners.to_vec(),
            })
            .collect();

        println!(
            "Roles found: {:?}",
            roles.iter().map(|role| &role.0).collect::<Vec<_>>()
        );

        roles
            .into_iter()
            .map(|role| {
                let identity = Identity::from(&role.0);
                (role, identity)
            })
            .collect()
    };

    let networking = Arc::new(moose::networking::LocalAsyncNetworking::default());

    let storage = Arc::new(moose::storage::LocalAsyncStorage::default());

    let session = AsyncSession::new(
        session_id,
        HashMap::new(),
        role_assignments,
        networking,
        storage,
    );

    let mut outputs: HashMap<String, <AsyncSession as Session>::Value> = HashMap::new();

    {
        let mut env: HashMap<String, <AsyncSession as Session>::Value> =
            HashMap::with_capacity(computation.operations.len());

        for op in computation.operations.iter() {
            let operands = op
                .inputs
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();

            let result = session.execute(&op.kind, &op.placement, operands)?;

            if matches!(op.kind, Operator::Output(_)) {
                // If it is an output, we need to make sure we capture it for returning.
                outputs.insert(op.name.clone(), result.clone());
            } else {
                // Everything else should be available in the env for other ops to use.
                env.insert(op.name.clone(), result);
            }
        }
    }

    for (output_name, output_value) in outputs {
        tokio::spawn(async move {
            let value = output_value.await;
            println!("Output '{}' ready:\n{:?}\n", output_name, value);
        });
    }

    let session_handle = session.into_handle()?;
    session_handle.join_on_first_error().await?;

    Ok(())
}
