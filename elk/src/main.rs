use clap::{ArgEnum, Parser, Subcommand};
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
    /// Compile a Moose computation
    Compile {
        /// Input file
        input: PathBuf,

        /// Output file, stdout if not present
        output: Option<PathBuf>,

        /// Computation format
        #[clap(arg_enum, short, long, default_value = "textual")]
        input_format: ComputationFormat,

        /// Computation format
        #[clap(arg_enum, short, long, default_value = "textual")]
        output_format: ComputationFormat,

        /// Comma-separated list of passes to apply in-order; default to all passes
        #[clap(short, long)]
        passes: Option<String>,
    },
    /// Print stats about a computation without transforming it
    #[clap(subcommand)]
    Stats(StatsCommands),
}

#[derive(Subcommand, Debug)]
enum StatsCommands {
    /// Print operator histogram
    OpHist {
        /// Input file
        input: PathBuf,

        /// Computation format
        #[clap(arg_enum, short, long, default_value = "textual")]
        input_format: ComputationFormat,

        /// Include placement in the category
        #[clap(long)]
        by_placement: bool,
    },
    /// Print operator counts
    OpCount {
        /// Input file
        input: PathBuf,

        /// Computation format
        #[clap(arg_enum, short, long, default_value = "textual")]
        input_format: ComputationFormat,

        /// Include placement in the category
        #[clap(long)]
        by_placement: bool,
    },
    /// Print out degree
    OutDegree {
        /// Input file
        input: PathBuf,

        /// Computation format
        #[clap(arg_enum, short, long, default_value = "textual")]
        input_format: ComputationFormat,

        /// Include operator in the category
        #[clap(long)]
        by_operator: bool,
    },
}

#[derive(Clone, Debug, ArgEnum)]
enum ComputationFormat {
    Bincode,
    Msgpack,
    Textual,
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    match &args.command {
        Commands::Compile {
            input,
            output,
            input_format,
            output_format,
            passes,
        } => {
            let comp = input_computation(input, input_format)?;
            let passes: Option<Vec<String>> = passes
                .clone()
                .map(|p| p.split(',').map(|s| s.to_string()).collect());
            let comp = compile(&comp, passes)?;
            output_computation(&comp, output, output_format)?;
        }
        Commands::Stats(StatsCommands::OpHist {
            input,
            input_format,
            by_placement,
        }) => {
            let comp = input_computation(input, input_format)?;
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
        Commands::Stats(StatsCommands::OpCount {
            input,
            input_format,
            by_placement,
        }) => {
            let comp = input_computation(input, input_format)?;
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
        Commands::Stats(StatsCommands::OutDegree {
            input,
            input_format,
            by_operator,
        }) => {
            let comp = input_computation(input, input_format)?;
            let op_name_to_out_degree: HashMap<&String, usize> =
                comp.operations.iter().fold(HashMap::new(), |mut map, op| {
                    for input_op_name in op.inputs.iter() {
                        *map.entry(input_op_name).or_insert(0) += 1;
                    }
                    map
                });
            let operator_map: HashMap<&String, &str> = if *by_operator {
                comp.operations
                    .iter()
                    .map(|op| (&op.name, op.kind.short_name()))
                    .collect()
            } else {
                HashMap::new()
            };
            let out_degree_distribution: HashMap<NumericalHistKey, usize> = op_name_to_out_degree
                .into_iter()
                .fold(HashMap::new(), |mut map, (op_name, out_degree)| {
                    *map.entry(NumericalHistKey {
                        value: out_degree,
                        category: operator_map.get(op_name),
                    })
                    .or_insert(0) += 1;
                    map
                });
            print_sorted("Out degree", &out_degree_distribution);
        }
    }
    Ok(())
}

fn input_computation(input: &Path, format: &ComputationFormat) -> anyhow::Result<Computation> {
    match format {
        ComputationFormat::Textual => {
            let source = read_to_string(input)?;
            Computation::from_textual(&source)
                .map_err(|e| anyhow::anyhow!("Failed to parse the input computation due to {}", e))
        }
        ComputationFormat::Msgpack => {
            let comp_raw = std::fs::read(input)?;
            Computation::from_msgpack(comp_raw)
                .map_err(|e| anyhow::anyhow!("Failed to parse the input computation due to {}", e))
        }
        ComputationFormat::Bincode => {
            let comp_raw = std::fs::read(input)?;
            Computation::from_bincode(comp_raw)
                .map_err(|e| anyhow::anyhow!("Failed to parse the input computation due to {}", e))
        }
    }
}

fn output_computation(
    comp: &Computation,
    output: &Option<PathBuf>,
    format: &ComputationFormat,
) -> anyhow::Result<()> {
    match format {
        ComputationFormat::Textual => {
            let result = comp.to_textual();
            match output {
                Some(path) => {
                    write(path, result)?;
                    Ok(())
                }
                None => {
                    println!("{}", result);
                    Ok(())
                }
            }
        }
        ComputationFormat::Msgpack => {
            let result = comp.to_msgpack()?;
            match output {
                Some(path) => {
                    write(path, result)?;
                    Ok(())
                }
                None => todo!(),
            }
        }
        ComputationFormat::Bincode => {
            let result = comp.to_bincode()?;
            match output {
                Some(path) => {
                    write(path, result)?;
                    Ok(())
                }
                None => todo!(),
            }
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
