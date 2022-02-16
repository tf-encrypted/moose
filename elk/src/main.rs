use clap::{Parser, Subcommand};
use moose::compilation::compile;
use moose::textual::verbose_parse_computation;
use moose::textual::ToTextual;
use std::fs::{read_to_string, write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[clap(name = "elk")]
#[clap(
    about = "A Moose compiler wrapper CLI",
    long_about = "Takes an input file with the Computation's Textual representation and applies the specified passes to it."
)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compiles the moose computation
    Compile {
        /// Input file
        input: PathBuf,

        /// Output file, stdout if not present
        output: Option<PathBuf>,

        /// List of passes to apply. In order. Default to run all the passes
        #[clap(short, long, required = true)]
        passes: Option<Vec<String>>,
    },
    /// Prints stats about a computation without transforming it
    Stats {
        /// Input file
        input: PathBuf,

        /// The kind of the stats to produce
        flavour: String,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    match &args.command {
        Commands::Compile {
            input,
            output,
            passes,
        } => {
            let source = read_to_string(input)?;
            let comp = verbose_parse_computation(&source)?;
            let comp = compile(&comp, passes.clone())?;
            match output {
                Some(path) => write(path, comp.to_textual())?,
                None => println!("{}", comp.to_textual()),
            }
        }
        Commands::Stats { input, flavour } => {
            println!("Computing {}", flavour);
            let source = read_to_string(input)?;
            let _comp = verbose_parse_computation(&source)?;
        }
    }
    Ok(())
}
