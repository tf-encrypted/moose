use moose::execution;
use moose::execution::*;
use pyo3::{prelude::*, types::PyModule};
use serde::Deserialize;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::convert::TryInto;

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
enum PyOperation {
    prim_SampleKeyOperation(PySampleKeyOperation),
    prim_DeriveSeedOperation(PyDeriveSeedOperation),
    ring_RingAddOperation(PyRingAddOperation),
    ring_RingSubOperation(PyRingSubOperation),
    ring_RingMulOperation(PyRingMulOperation),
    ring_RingDotOperation(PyRingDotOperation),
    ring_RingShapeOperation(PyRingShapeOperation),
    ring_RingSampleOperation(PyRingSampleOperation),
    ring_FillTensorOperation(PyFillTensorOperation),
    ring_RingShlOperation(PyRingShlOperation),
    ring_RingShrOperation(PyRingShrOperation),
    std_AddOperation(PyAddOperation),
    std_ConstantOperation(PyConstantOperation),
    std_LoadOperation(PyLoadOperation),
    std_SaveOperation(PySaveOperation),
    std_OutputOperation(PyOutputOperation),
    std_SerializeOperation(PySerializeOperation),
    std_DeserializeOperation(PyDeserializeOperation),
    std_SendOperation(PySendOperation),
    std_ReceiveOperation(PyReceiveOperation),
    fixed_RingEncodeOperation(PyRingEncodeOperation),
    fixed_RingDecodeOperation(PyRingDecodeOperation),
    // std_UnknownOperation(PyUnknownOperation),
}

#[derive(Deserialize, Debug)]
struct PyUnknownOperation {
    name: String,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
enum ValueType {
    prim_PRFKeyType,
    prim_SeedType,
    std_ShapeType,
    ring_RingTensorType,
}

#[derive(Deserialize, Debug)]
struct PySampleKeyOperation {
    name: String,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyDeriveSeedOperation {
    name: String,
    inputs: HashMap<String, String>,
    placement_name: String,
    #[serde(with = "serde_bytes")]
    nonce: Vec<u8>,
}

#[derive(Deserialize, Debug)]
struct PyRingAddOperation {
    name: String,
    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingSubOperation {
    name: String,
    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingMulOperation {
    name: String,
    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingDotOperation {
    name: String,
    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingShapeOperation {
    name: String,
    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingSampleOperation {
    name: String,
    max_value: Option<u64>,

    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyFillTensorOperation {
    name: String,
    value: u64,
    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingShlOperation {
    name: String,
    amount: u64,
    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingShrOperation {
    name: String,
    amount: u64,

    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyConstantOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct PyLoadOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct PyAddOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct PySaveOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct PyOutputOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct PySerializeOperation {
    name: String,
    value_type: ValueType,
}

#[derive(Deserialize, Debug)]
struct PyDeserializeOperation {
    name: String,
    value_type: ValueType,
}

#[derive(Deserialize, Debug)]
struct PySendOperation {
    name: String,
    sender: String,
    receiver: String,
    rendezvous_key: String,

    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyReceiveOperation {
    name: String,
    sender: String,
    receiver: String,
    rendezvous_key: String,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingEncodeOperation {
    name: String,
    scaling_factor: u64,
    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingDecodeOperation {
    name: String,
    scaling_factor: u64,
    inputs: HashMap<String, String>,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
enum PyPlacement {
    host_HostPlacement(PyHostPlacement),
    rep_ReplicatedPlacement(PyReplicatedPlacement),
}

#[derive(Deserialize, Debug)]
struct PyHostPlacement {
    name: String,
}

#[derive(Deserialize, Debug)]
struct PyReplicatedPlacement {
    player_names: Vec<String>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
struct PyComputation {
    operations: HashMap<String, PyOperation>,
    placements: HashMap<String, PyPlacement>,
}

impl TryFrom<&PyPlacement> for Placement {
    type Error = anyhow::Error;
    fn try_from(placement: &PyPlacement) -> anyhow::Result<Placement> {
        match placement {
            PyPlacement::host_HostPlacement(plc) => {
                Ok(execution::Placement::Host(execution::HostPlacement {
                    name: plc.name.clone(),
                }))
            }
            PyPlacement::rep_ReplicatedPlacement(plc) => {
                if plc.player_names.len() != 3 {
                    return Err(anyhow::anyhow!("Placement doesn't have 3 players"));
                }
                Ok(execution::Placement::Replicated(
                    execution::ReplicatedPlacement {
                        players: [
                            plc.player_names[0].clone(),
                            plc.player_names[1].clone(),
                            plc.player_names[2].clone(),
                        ],
                    },
                ))
            }
        }
    }
}

fn map_inputs(
    inputs: &HashMap<String, String>,
    expected_inputs: &[&str],
) -> anyhow::Result<Vec<String>> {
    expected_inputs
        .iter()
        .map(|item| {
            inputs
                .get(*item)
                .cloned()
                .ok_or(anyhow::anyhow!("No value found in input vector"))
        })
        .collect::<anyhow::Result<Vec<_>>>()
}
fn map_placement(
    plc: &HashMap<String, execution::Placement>,
    name: &str,
) -> anyhow::Result<execution::Placement> {
    plc.get(name)
        .cloned()
        .ok_or(anyhow::anyhow!("No key found in placement dictionary"))
}

impl TryFrom<PyComputation> for execution::Computation {
    type Error = anyhow::Error;
    fn try_from(python_computation: PyComputation) -> anyhow::Result<execution::Computation> {
        let placements: HashMap<String, execution::Placement> = python_computation
            .placements
            .iter()
            .map(|(placement_name, python_placement)| {
                Ok((placement_name.clone(), python_placement.try_into()?))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;
        let operations: Vec<execution::Operation> = python_computation
            .operations
            .values()
            .map(|op| {
                use execution::Operator::*;
                use PyOperation::*;
                match op {
                    prim_SampleKeyOperation(op) => Ok(execution::Operation {
                        kind: PrimGenPrfKey(PrimGenPrfKeyOp {}),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    prim_DeriveSeedOperation(op) => Ok(execution::Operation {
                        kind: PrimDeriveSeed(PrimDeriveSeedOp {
                            nonce: Nonce(op.nonce.clone()),
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["key"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingAddOperation(op) => Ok(execution::Operation {
                        kind: RingAdd(RingAddOp {
                            lhs: execution::Ty::Ring64TensorTy,
                            rhs: execution::Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingSubOperation(op) => Ok(execution::Operation {
                        kind: RingSub(RingSubOp {
                            lhs: execution::Ty::Ring64TensorTy,
                            rhs: execution::Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingMulOperation(op) => Ok(execution::Operation {
                        kind: RingMul(RingMulOp {
                            lhs: execution::Ty::Ring64TensorTy,
                            rhs: execution::Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingDotOperation(op) => Ok(execution::Operation {
                        kind: RingDot(RingDotOp {
                            lhs: execution::Ty::Ring64TensorTy,
                            rhs: execution::Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),

                    ring_RingShapeOperation(op) => Ok(execution::Operation {
                        kind: RingShape(RingShapeOp {
                            ty: execution::Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["tensor"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingSampleOperation(op) => Ok(execution::Operation {
                        kind: RingSample(RingSampleOp {
                            output: execution::Ty::Ring64TensorTy,
                            max_value: op.max_value,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["shape", "seed"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_FillTensorOperation(op) => Ok(execution::Operation {
                        kind: RingFill(RingFillOp { value: op.value }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["shape"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingShlOperation(op) => Ok(execution::Operation {
                        kind: RingShl(RingShlOp {
                            amount: op.amount as usize,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingShrOperation(op) => Ok(execution::Operation {
                        kind: RingShr(RingShrOp {
                            amount: op.amount as usize,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SendOperation(op) => Ok(execution::Operation {
                        kind: Send(execution::SendOp {
                            rendezvous_key: op.rendezvous_key.clone(),
                            sender: execution::HostPlacement { name: op.sender.clone() },
                            receiver: execution::HostPlacement { name: op.receiver.clone() },
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_ReceiveOperation(op) => Ok(execution::Operation {
                        kind: Receive(execution::ReceiveOp {
                            rendezvous_key: op.rendezvous_key.clone(),
                            sender: execution::HostPlacement { name: op.sender.clone() },
                            receiver: execution::HostPlacement { name: op.receiver.clone() },
                        }),
                        name: op.name.clone(),
                        inputs: Vec::new(), //  TODO(Dragos): is this OK?
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_RingEncodeOperation(op) => Ok(execution::Operation {
                        kind: FixedpointRingEncode(execution::FixedpointRingEncodeOp {
                            scaling_factor: op.scaling_factor,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_RingDecodeOperation(op) => Ok(execution::Operation {
                        kind: FixedpointRingDecode(execution::FixedpointRingDecodeOp {
                            scaling_factor: op.scaling_factor,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    _ => Err(anyhow::anyhow!(
                        "Python to Rust op conversion not implemented",
                    )),
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(execution::Computation { operations })
    }
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

    let comp: PyComputation = rmp_serde::from_read_ref(&buf).unwrap();

    println!("{:?}", comp);
    let rust_comp: Computation = comp.try_into().unwrap();
    for operation in rust_comp.operations {
        println!("{:?}", operation);
    }

    assert_eq!(true, false);
    // let computation: Computation = rmp_serde::from_slice(&serialized).unwrap();
    // println!("deserialized = {:?}", deserialized);
}
