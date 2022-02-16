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

        /// Comma-separated list of passes to apply. In order. Default to run all the passes
        #[clap(short, long, required = true)]
        passes: Option<String>,
    },
    /// Prints stats about a computation without transforming it
    Stats {
        /// The kind of the stats to produce
        flavour: String,

        /// Input file
        input: PathBuf,
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
            let passes: Option<Vec<String>> = passes
                .clone()
                .map(|p| p.split(',').map(|s| s.to_string()).collect());
            let comp = compile(&comp, passes)?;
            match output {
                Some(path) => write(path, comp.to_textual())?,
                None => println!("{}", comp.to_textual()),
            }
        }
        Commands::Stats { flavour, input } => {
            let source = read_to_string(input)?;
            let comp = verbose_parse_computation(&source)?;
            match flavour.as_str() {
                "op_hist" => {
                    use std::collections::HashMap;
                    let hist: HashMap<String, usize> = comp
                        .operations
                        .iter()
                        .map(|op| op.kind.short_name())
                        .fold(HashMap::new(), |mut map, name| {
                            *map.entry(name.to_string()).or_insert(0) += 1;
                            map
                        });
                    let mut sorted_hist: Vec<(&String, &usize)> = hist.iter().collect();
                    sorted_hist.sort_by(|a, b| b.1.cmp(a.1));
                    for op in sorted_hist {
                        println!("{}\t{}", op.1, op.0);
                    }
                }
                _ => return Err(anyhow::anyhow!("Unexpected stats flavour {}", flavour)),
            }
        }
    }
    Ok(())
}
