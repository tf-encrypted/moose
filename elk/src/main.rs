use moose::compilation::compile_passes;
use moose::textual::verbose_parse_computation;
use moose::textual::ToTextual;
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
    let comp = verbose_parse_computation(&source)?;
    let passes = opt.passes.unwrap_or_else(all_passes);
    let comp = compile_passes(&comp, &passes)?;
    match opt.output {
        Some(path) => write(path, comp.to_textual())?,
        None => println!("{}", comp.to_textual()),
    }
    Ok(())
}

/// Finds all the passes and the proper order for them
fn all_passes() -> Vec<String> {
    // Currently is not doing any magical discover and sorting, just returns a hard-coded list.
    vec!["networking".into(), "prune".into()]
}
