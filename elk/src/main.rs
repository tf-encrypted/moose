use clap::{Parser, Subcommand};
use moose::compilation::compile;
use moose::prelude::Computation;
use moose::textual::parallel_parse_computation;
use moose::textual::ToTextual;
use std::collections::HashMap;
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
        #[clap(short, long)]
        passes: Option<String>,
    },
    /// Prints stats about a computation without transforming it
    Stats {
        /// The kind of the stats to produce
        flavor: String,

        /// Input file
        input: PathBuf,

        /// Include placement in the category
        #[clap(short, long)]
        by_placement: bool,
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
            let comp = parse_computation(input)?;
            let passes: Option<Vec<String>> = passes
                .clone()
                .map(|p| p.split(',').map(|s| s.to_string()).collect());
            let comp = compile(&comp, passes)?;
            match output {
                Some(path) => write(path, comp.to_textual())?,
                None => println!("{}", comp.to_textual()),
            }
        }
        Commands::Stats {
            flavor,
            input,
            by_placement,
        } => {
            let comp = parse_computation(input)?;
            match flavor.as_str() {
                "op_hist" => {
                    let hist: HashMap<String, usize> = comp
                        .operations
                        .iter()
                        .map(|op| {
                            if *by_placement {
                                format!("{} {}", op.kind.short_name(), op.placement.to_textual())
                            } else {
                                op.kind.short_name().to_string()
                            }
                        })
                        .fold(HashMap::new(), |mut map, name| {
                            *map.entry(name).or_insert(0) += 1;
                            map
                        });
                    print_sorted(&hist);
                }
                "op_count" => {
                    if *by_placement {
                        let hist: HashMap<String, usize> = comp
                            .operations
                            .iter()
                            .map(|op| op.placement.to_textual())
                            .fold(HashMap::new(), |mut map, name| {
                                *map.entry(name).or_insert(0) += 1;
                                map
                            });
                        print_sorted(&hist);
                    } else {
                        println!("{}", comp.operations.len())
                    }
                }
                _ => return Err(anyhow::anyhow!("Unexpected stats flavor {}", flavor)),
            }
        }
    }
    Ok(())
}

fn parse_computation(input: &PathBuf) -> anyhow::Result<Computation> {
    let source = read_to_string(input)?;
    parallel_parse_computation(&source, 12)
}

fn print_sorted(map: &HashMap<String, usize>) {
    let mut sorted_hist: Vec<(&String, &usize)> = map.iter().collect();
    sorted_hist.sort_by(|a, b| b.1.cmp(a.1));
    for op in sorted_hist {
        println!("{:8} {}", op.1, op.0);
    }
}
