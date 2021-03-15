use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;

struct std_AddOperation;

#[derive(Deserialize, Debug)]
struct Operation {
    // name: String,
    // inputs: HashMap<String, String>,
    // placement_name: String,
    __type__: String,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
enum Placement {
    host_HostPlacement(HostPlacement),
    rep_ReplicatedPlacement(ReplicatedPlacement),
}

#[derive(Deserialize, Debug)]
struct HostPlacement {
    name: String
}

#[derive(Deserialize, Debug)]
struct ReplicatedPlacement {
    name: String
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
struct Computation {
    // operations: HashMap<String, Operation>,
    placements: HashMap<String, Placement>,
}

#[test]
fn test_deserialize_python_computation() {
    // TODO call Python code to generate and serialize computation,
    // store and load from tmp directory instead of in `src`
    let file = File::open("/home/dahl/work/tf-encrypted/runtime/examples/replicated/computation.tmp").unwrap();

    let comp: Computation = rmp_serde::from_read(file).unwrap();
    println!("{:?}", comp);

    // let computation: Computation = rmp_serde::from_slice(&serialized).unwrap();
    // println!("deserialized = {:?}", deserialized);

}