use clap::{Parser, Subcommand};
use moose::compilation::compile;
use moose::textual::verbose_parse_computation;
use moose::textual::ToTextual;
use std::fs::{read_to_string, write};
use std::path::PathBuf;

#[derive(Parser)]
#[clap(name = "elk")]
#[clap(
    about = "A Moose compiler wrapper CLI",
    long_about = "Takes an input file with the Computation's Textual representation and applies the specified passes to it."
)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[clap(arg_required_else_help = true)]
    Compile {
        /// Input file
        input: PathBuf,

        /// Output file, stdout if not present
        output: Option<PathBuf>,

        /// List of passes to apply. In order. Default to run all the passes
        passes: Option<Vec<String>>,
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
            let comp = compile(&comp, passes)?;
            match output {
                Some(path) => write(path, comp.to_textual())?,
                None => println!("{}", comp.to_textual()),
            }
        }
    }
    Ok(())
}
