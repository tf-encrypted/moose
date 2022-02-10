use moose::compilation::compile;
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
    let comp = compile(&comp, opt.passes)?;
    match opt.output {
        Some(path) => write(path, comp.to_textual())?,
        None => println!("{}", comp.to_textual()),
    }
    Ok(())
}
