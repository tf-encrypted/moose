use moose::{computation::*, standard::Float64Tensor};
use ndarray::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::convert::TryInto;

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
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
    std_ConstantOperation(PyConstantOperation),
    std_SendOperation(PySendOperation),
    std_ReceiveOperation(PyReceiveOperation),
    fixed_RingEncodeOperation(PyRingEncodeOperation),
    fixed_RingDecodeOperation(PyRingDecodeOperation),
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
enum PyValueType {
    std_ShapeType,
    std_StringType,
    std_TensorType,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
enum PyValue {
    std_ShapeValue { value: Vec<u8> },
    std_StringValue { value: String },
    std_Float64Tensor { items: Vec<f64>, shape: Vec<u8> },
}

#[derive(Deserialize, Debug)]
struct PySampleKeyOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
}

type Inputs = HashMap<String, String>;

#[derive(Deserialize, Debug)]
struct PyDeriveSeedOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    #[serde(with = "serde_bytes")]
    nonce: Vec<u8>,
}

#[derive(Deserialize, Debug)]
struct PyRingAddOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingSubOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingMulOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingDotOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingShapeOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingSampleOperation {
    name: String,
    max_value: Option<u64>,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyFillTensorOperation {
    name: String,
    value: u64,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingShlOperation {
    name: String,
    amount: u64,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingShrOperation {
    name: String,
    amount: u64,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyConstantOperation {
    name: String,
    placement_name: String,
    output_type: PyValueType,
    value: PyValue,
}

#[derive(Deserialize, Debug)]
struct PyAddOperation {
    name: String,
}

#[derive(Deserialize, Debug)]
struct PySerializeOperation {
    name: String,
    value_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyDeserializeOperation {
    name: String,
    value_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PySendOperation {
    name: String,
    sender: String,
    receiver: String,
    rendezvous_key: String,
    inputs: Inputs,
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
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingDecodeOperation {
    name: String,
    scaling_factor: u64,
    inputs: Inputs,
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
            PyPlacement::host_HostPlacement(plc) => Ok(Placement::Host(HostPlacement {
                name: plc.name.clone(),
            })),
            PyPlacement::rep_ReplicatedPlacement(plc) => {
                if plc.player_names.len() != 3 {
                    return Err(anyhow::anyhow!("Placement doesn't have 3 players"));
                }
                Ok(Placement::Replicated(ReplicatedPlacement {
                    players: [
                        plc.player_names[0].clone(),
                        plc.player_names[1].clone(),
                        plc.player_names[2].clone(),
                    ],
                }))
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
                .ok_or_else(|| anyhow::anyhow!("No value found in input vector"))
        })
        .collect::<anyhow::Result<Vec<_>>>()
}

fn map_placement(plc: &HashMap<String, Placement>, name: &str) -> anyhow::Result<Placement> {
    plc.get(name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("No key found in placement dictionary"))
}

fn map_constant_value(constant_value: &PyValue) -> anyhow::Result<Value> {
    match constant_value {
        PyValue::std_ShapeValue { ref value } => {
            Ok(moose::standard::Shape(value.iter().map(|i| *i as usize).collect()).into())
        }
        &PyValue::std_Float64Tensor {
            ref items,
            ref shape,
        } => {
            let shape: Vec<usize> = shape.iter().map(|i| *i as usize).collect();
            let tensor = ArrayD::from_shape_vec(shape, items.clone())?;
            Ok(Float64Tensor::from(tensor).into())
        }
        _ => unimplemented!(),
    }
}

impl TryFrom<PyComputation> for Computation {
    type Error = anyhow::Error;
    fn try_from(python_computation: PyComputation) -> anyhow::Result<Computation> {
        let placements: HashMap<String, Placement> = python_computation
            .placements
            .iter()
            .map(|(placement_name, python_placement)| {
                Ok((placement_name.clone(), python_placement.try_into()?))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;
        let operations: Vec<Operation> = python_computation
            .operations
            .values()
            .map(|op| {
                use moose::computation::Operator::*;
                use PyOperation::*;
                match op {
                    prim_SampleKeyOperation(op) => Ok(Operation {
                        kind: PrimGenPrfKey(PrimGenPrfKeyOp {}),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    prim_DeriveSeedOperation(op) => Ok(Operation {
                        kind: PrimDeriveSeed(PrimDeriveSeedOp {
                            nonce: moose::prim::Nonce(op.nonce.clone()),
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["key"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingAddOperation(op) => Ok(Operation {
                        kind: RingAdd(RingAddOp {
                            lhs: Ty::Ring64TensorTy,
                            rhs: Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingSubOperation(op) => Ok(Operation {
                        kind: RingSub(RingSubOp {
                            lhs: Ty::Ring64TensorTy,
                            rhs: Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingMulOperation(op) => Ok(Operation {
                        kind: RingMul(RingMulOp {
                            lhs: Ty::Ring64TensorTy,
                            rhs: Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingDotOperation(op) => Ok(Operation {
                        kind: RingDot(RingDotOp {
                            lhs: Ty::Ring64TensorTy,
                            rhs: Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingShapeOperation(op) => Ok(Operation {
                        kind: RingShape(RingShapeOp {
                            ty: Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["tensor"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingSampleOperation(op) => Ok(Operation {
                        kind: RingSample(RingSampleOp {
                            output: Ty::Ring64TensorTy,
                            max_value: op.max_value,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["shape", "seed"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_FillTensorOperation(op) => Ok(Operation {
                        kind: RingFill(RingFillOp { value: op.value }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["shape"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingShlOperation(op) => Ok(Operation {
                        kind: RingShl(RingShlOp {
                            amount: op.amount as usize,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingShrOperation(op) => Ok(Operation {
                        kind: RingShr(RingShrOp {
                            amount: op.amount as usize,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_ConstantOperation(op) => Ok(Operation {
                        kind: Constant(ConstantOp {
                            value: map_constant_value(&op.value)?,
                        }),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SendOperation(op) => Ok(Operation {
                        kind: Send(SendOp {
                            rendezvous_key: op.rendezvous_key.clone(),
                            sender: HostPlacement {
                                name: op.sender.clone(),
                            },
                            receiver: HostPlacement {
                                name: op.receiver.clone(),
                            },
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_ReceiveOperation(op) => Ok(Operation {
                        kind: Receive(ReceiveOp {
                            rendezvous_key: op.rendezvous_key.clone(),
                            sender: HostPlacement {
                                name: op.sender.clone(),
                            },
                            receiver: HostPlacement {
                                name: op.receiver.clone(),
                            },
                        }),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_RingEncodeOperation(op) => Ok(Operation {
                        kind: FixedpointRingEncode(FixedpointRingEncodeOp {
                            scaling_factor: op.scaling_factor,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_RingDecodeOperation(op) => Ok(Operation {
                        kind: FixedpointRingDecode(FixedpointRingDecodeOp {
                            scaling_factor: op.scaling_factor,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    _ => Err(anyhow::anyhow!(
                        "Python to Rust op conversion not implemented"
                    )),
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Computation { operations })
    }
}

#[test]
fn test_deserialize_python_simple_computation() {
    use maplit::hashmap;
    use moose::execution;
    use pyo3::{prelude::*, types::PyModule};

    let gil = Python::acquire_gil();
    let py = gil.python();
    let comp_graph_py = PyModule::from_code(
        py,
        r#"

import numpy as np
from moose.computation import ring as ring_dialect
from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.utils import serialize_computation
from moose.computation.standard import TensorType
from moose.computation import dtypes
def f():
    comp = Computation(operations={}, placements={})
    alice = comp.add_placement(HostPlacement(name="alice"))
    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="x_shape",
            placement_name=alice.name,
            inputs={},
            value=standard_dialect.ShapeValue(value = (2, 2)),
            output_type=standard_dialect.ShapeType(),
        )
    )
    comp.add_operation(
        ring_dialect.FillTensorOperation(
            name="x",
            placement_name=alice.name,
            value=1,
            inputs={"shape": "x_shape"},
        )
    )
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="alice_input",
            value=x,
            placement_name=alice.name,
            inputs={},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    return serialize_computation(comp)

"#,
        "comp_graph.py",
        "comp_graph",
    )
    .unwrap();
    let py_any: &PyAny = comp_graph_py.getattr("f").unwrap().call0().unwrap();
    let buf: Vec<u8> = py_any.extract().unwrap();

    let comp: PyComputation = rmp_serde::from_read_ref(&buf).unwrap();
    let rust_comp: Computation = comp.try_into().unwrap();

    let env = hashmap![];
    let exec = execution::EagerExecutor::new();
    exec.run_computation(&rust_comp, 12345, env).ok();
}
