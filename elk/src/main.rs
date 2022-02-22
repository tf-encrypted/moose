use clap::{Parser, Subcommand};
use moose::compilation::compile;
use moose::textual::verbose_parse_computation;
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
        flavour: String,

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
        Commands::Stats {
            flavour,
            input,
            by_placement,
        } => {
            let source = read_to_string(input)?;
            let comp = verbose_parse_computation(&source)?;
            match flavour.as_str() {
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
                    print_sorted("Operator", &hist);
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
                        print_sorted("Placement", &hist);
                    } else {
                        println!("{}", comp.operations.len())
                    }
                }
                "out_degree" => {
                    let op_name_to_out_degree: HashMap<&String, usize> =
                        comp.operations.iter().fold(HashMap::new(), |mut map, op| {
                            for input_op_name in op.inputs.iter() {
                                *map.entry(input_op_name).or_insert(0) += 1;
                            }
                            map
                        });
                    let out_degree_distribution: HashMap<usize, usize> = op_name_to_out_degree
                        .into_iter()
                        .fold(HashMap::new(), |mut map, (_op_name, out_degree)| {
                            *map.entry(out_degree).or_insert(0) += 1;
                            map
                        });
                    print_sorted("Out degree", &out_degree_distribution);
                }
                _ => return Err(anyhow::anyhow!("Unexpected stats flavour {}", flavour)),
            }
        }
    }
    Ok(())
}

fn print_sorted<S>(key_label: &str, map: &HashMap<S, usize>)
where
    S: std::fmt::Display,
{
    let mut sorted_hist: Vec<(&S, &usize)> = map.iter().collect();
    sorted_hist.sort_by(|a, b| b.1.cmp(a.1));
    println!("{:>10} {}", "Count", key_label);
    for op in sorted_hist {
        println!("{:>10} {}", op.1, op.0);
    }
}
