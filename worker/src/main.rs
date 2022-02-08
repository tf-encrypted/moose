pub mod cape;

use crate::cape::csv::read_csv;
use moose::prelude::*;
use ndarray::array;
use std::fs::File;
use std::{collections::HashMap, io::Read};
use structopt::StructOpt;

#[derive(Debug, StructOpt, Clone)]
struct Opt {
    #[structopt(long)]
    data: String,

    #[structopt(long)]
    comp: String,

    //#[structopt(long)]
    //session_id: String,

    //#[structopt(long)]
    //output: String,
    #[structopt(long)]
    placement: String,

    #[structopt(long)]
    hosts: String,
}

fn read_comp_file(filename: &str) -> anyhow::Result<Vec<u8>> {
    let mut file = File::open(filename)?;
    let mut vec = Vec::new();
    file.read_to_end(&mut vec)?;
    Ok(vec)
}

#[tokio::main]
async fn main() {
    let opt = Opt::from_args();

    let hosts: HashMap<String, String> = serde_json::from_str(&opt.hosts).unwrap();

    let computation_bytes = read_comp_file(&opt.comp).unwrap();

    let computation = Computation::from_bytes(computation_bytes).unwrap();

    let input = read_csv(&opt.data, None, &[], &opt.placement)
        .await
        .unwrap();

    println!("input = {:?}", input);
}
