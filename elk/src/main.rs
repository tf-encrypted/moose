use clap::{Parser, Subcommand};
use moose::compilation::compile;
use moose::prelude::Computation;
use moose::textual::ToTextual;
use std::collections::HashMap;
use std::fs::{read_to_string, write};
use std::path::{Path, PathBuf};

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

        #[clap(short, long)]
        format: String,

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

        format: String,

        /// Include placement in the category, where supported
        #[clap(short, long)]
        by_placement: bool,

        /// Include operation kind in the category, where supported
        #[clap(long)]
        by_op_kind: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    match &args.command {
        Commands::Compile {
            input,
            output,
            format,
            passes,
        } => {
            let comp = parse_computation(input, format)?;
            let passes: Option<Vec<String>> = passes
                .clone()
                .map(|p| p.split(',').map(|s| s.to_string()).collect());
            let comp = compile(&comp, passes)?;
            match (output, &format[..]) {
                (Some(path), "Textual") => write(path, comp.to_textual())?,
                (Some(path), "Binary") => {
                    let comp_bytes = comp.to_msgpack()?;
                    write(path, comp_bytes)?
                }
                (None, _) => println!("{}", comp.to_textual()),
                (&Some(_), &_) => println!("{}", comp.to_textual()),
            }
        }
        Commands::Stats {
            flavor,
            input,
            format,
            by_placement,
            by_op_kind,
        } => {
            let comp = parse_computation(input, format)?;
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
                    let op_kind_map: HashMap<&String, &str> = if *by_op_kind {
                        comp.operations
                            .iter()
                            .map(|op| (&op.name, op.kind.short_name()))
                            .collect()
                    } else {
                        HashMap::new()
                    };
                    let out_degree_distribution: HashMap<NumericalHistKey, usize> =
                        op_name_to_out_degree.into_iter().fold(
                            HashMap::new(),
                            |mut map, (op_name, out_degree)| {
                                *map.entry(NumericalHistKey {
                                    value: out_degree,
                                    category: op_kind_map.get(op_name),
                                })
                                .or_insert(0) += 1;
                                map
                            },
                        );
                    print_sorted("Out degree", &out_degree_distribution);
                }
                _ => return Err(anyhow::anyhow!("Unexpected stats flavor {}", flavor)),
            }
        }
    }
    Ok(())
}

fn parse_computation(input: &Path, format: &String) -> anyhow::Result<Computation> {
    match &format[..] {
        "Binary" => {
            let comp_raw = std::fs::read(input)?;
            Computation::from_msgpack(comp_raw)
                .map_err(|e| anyhow::anyhow!("Failed to parse the input computation due to {}", e))
        }
        _ => {
            let source = read_to_string(input)?;
            Computation::from_textual(&source)
                .map_err(|e| anyhow::anyhow!("Failed to parse the input computation due to {}", e))
        }
    }
}

#[derive(Eq, PartialEq, Hash, Debug)]
struct NumericalHistKey<'a> {
    value: usize,
    category: Option<&'a &'a str>,
}

impl std::fmt::Display for NumericalHistKey<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:10} {}", self.value, self.category.unwrap_or(&""))
    }
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
