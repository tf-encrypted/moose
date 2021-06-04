use moose::compiler_networking::compiler_networking;
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

    /// List of passes to apply. In order.
    #[structopt(short, long)]
    passes: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();
    let source = read_to_string(opt.input)?;
    let mut comp = verbose_parse_computation(&source)?;
    for pass in opt.passes {
        comp = do_pass(&pass, &comp)?;
    }
    // TODO: Apply some passes
    match opt.output {
        Some(path) => write(path, comp.to_textual())?,
        None => println!("{}", comp.to_textual()),
    }
    Ok(())
}

fn do_pass(pass: &str, comp: &Computation) -> anyhow::Result<Computation> {
    match pass {
        "networking" => compiler_networking(comp),
        missing_pass => Err(anyhow::anyhow!("Unknwon pass requested: {}", missing_pass)),
    }
}
