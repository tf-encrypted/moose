use compilation::networking::{print_graph, NetworkingPass};
use moose::computation::Computation;
use moose::text_computation::verbose_parse_computation;
use moose::text_computation::ToTextual;
use std::fs::{read_to_string, write};
use std::path::PathBuf;

use structopt::StructOpt;

#[derive(Debug, StructOpt)]
/// A Moose compiler wrapper CLI
///
/// Takes an input file with the Computation's Textual representation and applies the specified passes to it.
struct Opt {
    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// Output file, stdout if not present
    #[structopt(parse(from_os_str))]
    output: Option<PathBuf>,

    /// List of passes to apply. In order. Default to run all the passes
    #[structopt(short, long)]
    passes: Option<Vec<String>>,
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();
    let source = read_to_string(opt.input)?;
    let mut comp = verbose_parse_computation(&source)?;
    for pass in opt.passes.unwrap_or_else(all_passes) {
        comp = do_pass(&pass, &comp)?;
    }
    match opt.output {
        Some(path) => write(path, comp.to_textual())?,
        None => println!("{}", comp.to_textual()),
    }
    Ok(())
}

/// Finds all the passes and the proper order for them
fn all_passes() -> Vec<String> {
    // Currently is not doing any magical discover and sorting, just returns a hard-coded list.
    vec!["networking".into()]
}

fn do_pass(pass: &str, comp: &Computation) -> anyhow::Result<Computation> {
    match pass {
        "networking" => NetworkingPass::pass(comp),
        "print" => print_graph(comp),
        missing_pass => Err(anyhow::anyhow!("Unknwon pass requested: {}", missing_pass)),
    }
}
