use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyBytes, PyModule},
};
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt::Binary;

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
enum Operation {
    prim_SampleKeyOperation(SampleKeyOperation),
    prim_DeriveSeedOperation(DeriveSeedOperation),
    ring_RingShapeOperation(RingShapeOperation),
    ring_RingSampleOperation(RingSampleOperation),
    ring_RingSubOperation(RingSubOperation),
    ring_RingShlOperation(RingShlOperation),
    ring_RingShrOperation(RingShrOperation),
    ring_RingAddOperation(RingAddOperation),
    ring_FillTensorOperation(FillTensorOperation),
    ring_RingMulOperation(RingMulOperation),
    std_AddOperation(AddOperation),
    std_ConstantOperation(ConstantOperation),
    std_LoadOperation(LoadOperation),
    std_SaveOperation(SaveOperation),
    std_OutputOperation(OutputOperation),
    std_SerializeOperation(SerializeOperation),
    std_DeserializeOperation(DeserializeOperation),
    std_SendOperation(SendOperation),
    std_ReceiveOperation(ReceiveOperation),
    fixed_RingEncodeOperation(RingEncodeOperation),
    fixed_RingDecodeOperation(RingDecodeOperation),
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
enum ValueType {
    prim_PRFKeyType,
    prim_SeedType,
    std_ShapeType,
    ring_RingTensorType,
}

#[derive(Deserialize, Debug)]
struct SampleKeyOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct DeriveSeedOperation {
    name: String,
    #[serde(with = "serde_bytes")]
    nonce: Vec<u8>,
}

#[derive(Deserialize, Debug)]
struct RingShapeOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct RingSampleOperation {
    name: String,
    max_value: Option<u64>,
}

#[derive(Deserialize, Debug)]
struct RingSubOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct RingAddOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct RingMulOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct RingShlOperation {
    name: String,
    amount: u64,
}

#[derive(Deserialize, Debug)]
struct RingShrOperation {
    name: String,
    amount: u64,
}

#[derive(Deserialize, Debug)]
struct FillTensorOperation {
    name: String,
    value: i64,
}

#[derive(Deserialize, Debug)]
struct ConstantOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct LoadOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct AddOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct SaveOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct OutputOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct SerializeOperation {
    name: String,
    value_type: ValueType,
}

#[derive(Deserialize, Debug)]
struct DeserializeOperation {
    name: String,
    value_type: ValueType,
}

#[derive(Deserialize, Debug)]
struct SendOperation {
    name: String,
    sender: String,
    receiver: String,
    rendezvous_key: String,
}

#[derive(Deserialize, Debug)]
struct ReceiveOperation {
    name: String,
    sender: String,
    receiver: String,
    rendezvous_key: String,
}

#[derive(Deserialize, Debug)]
struct RingEncodeOperation {
    name: String,
    scaling_factor: u64,
}

#[derive(Deserialize, Debug)]
struct RingDecodeOperation {
    name: String,
    scaling_factor: u64,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
enum Placement {
    host_HostPlacement(HostPlacement),
    rep_ReplicatedPlacement(ReplicatedPlacement),
}

#[derive(Deserialize, Debug)]
struct HostPlacement {
    name: String,
}

#[derive(Deserialize, Debug)]
struct ReplicatedPlacement {
    name: String,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
struct Computation {
    operations: HashMap<String, Operation>,
    placements: HashMap<String, Placement>,
}

#[test]
fn test_deserialize_python_computation() {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let comp_graph_py = PyModule::from_code(
        py,
        r#"
from moose import edsl
from moose.computation.utils import serialize_computation

alice = edsl.host_placement(name="alice")
bob = edsl.host_placement(name="bob")
carole = edsl.host_placement(name="carole")
dave = edsl.host_placement(name="dave")
rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

def f():
    @edsl.computation
    def my_comp():
        with alice:
            x = edsl.load("x", dtype=edsl.float64)
        with bob:
            y = edsl.load("y", dtype=edsl.float64)
        with rep:
            z1 = edsl.mul(x, y)

        with dave:
            res_dave = edsl.save("res", z1)

        return res_dave

    concrete_comp = edsl.trace(my_comp)
    return serialize_computation(concrete_comp)

"#,
        "comp_graph.py",
        "comp_graph",
    )
    .unwrap();
    let py_any: &PyAny = comp_graph_py.getattr("f").unwrap().call0().unwrap();
    let buf: Vec<u8> = py_any.extract().unwrap();

    let comp: Computation = rmp_serde::from_read_ref(&buf).unwrap();
    println!("{:?}", comp);

    assert_eq!(true, false);
    // let computation: Computation = rmp_serde::from_slice(&serialized).unwrap();
    // println!("deserialized = {:?}", deserialized);
}
