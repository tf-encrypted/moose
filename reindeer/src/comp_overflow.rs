use structopt::StructOpt;
use moose::computation::CompactComputation;

#[derive(Debug, StructOpt, Clone)]
struct Opt {
    #[structopt(env, long, default_value = "./xgboost-perf/xgboost.moose")]
    comp_path: String,

    #[structopt(env, long, default_value = "5")]
    num_clones: u16,

    #[structopt(env, long, default_value = "3")]
    iterations: u16,

    #[structopt(env, long)]
    bench_compact: u8,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let opt = Opt::from_args();

    tracing::info!("Started to parse the computation from {:?}", opt.comp_path);

    let computation = {
        let comp_path = &opt.comp_path;
        let comp_raw = std::fs::read_to_string(comp_path)?;
        moose::computation::Computation::from_textual(&comp_raw)?
    };


    tracing::info!("Finished parsing the computation from");

    if opt.bench_compact == 1 {
        tracing::info!("Cloning the compact computation");
        let compact = CompactComputation::from(&computation);
        for x in 0..opt.iterations {
            tracing::info!("batch of cloning {:?}", x);
            let _cloned: Vec<_> = (0..opt.num_clones).map(|_| compact.clone()).collect();
        }
    } else {
        tracing::info!("Cloning the un-optimized computation");
        for x in 0..opt.iterations {
            tracing::info!("batch of cloning {:?}", x);
            let _cloned: Vec<_> = (0..opt.num_clones).map(|_| computation.clone()).collect();
        }
    }

    Ok(())
}
