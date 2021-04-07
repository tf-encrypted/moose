use moose::{computation::*, standard::Float32Tensor, standard::Float64Tensor};
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
    std_AddOperation(PyAddOperation),
    std_SubOperation(PySubOperation),
    std_MulOperation(PyMulOperation),
    std_DotOperation(PyDotOperation),
    std_SerializeOperation(PySerializeOperation),
    std_DeserializeOperation(PyDeserializeOperation),
    std_SendOperation(PySendOperation),
    std_OutputOperation(PyOutputOperation),
    std_SaveOperation(PySaveOperation),
    std_ReceiveOperation(PyReceiveOperation),
    fixed_RingEncodeOperation(PyRingEncodeOperation),
    fixed_RingDecodeOperation(PyRingDecodeOperation),
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
#[allow(clippy::upper_case_acronyms)]
enum PyValueType {
    prim_PRFKeyType,
    prim_SeedType,
    std_BytesType,
    std_ShapeType,
    std_StringType,
    std_TensorType,
    std_UnitType,
    ring_RingTensorType,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
enum PyValue {
    std_ShapeValue { value: Vec<u8> },
    std_StringConstant { value: String },
    std_TensorConstant { value: PyNdarray },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "dtype")]
#[allow(non_camel_case_types)]
enum PyNdarray {
    float32 { items: Vec<f32>, shape: Vec<u8> },
    float64 { items: Vec<f64>, shape: Vec<u8> },
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
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    value: PyValue,
}

#[derive(Deserialize, Debug)]
struct PyAddOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PySubOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyMulOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyDotOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PySerializeOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyDeserializeOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
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
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyOutputOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PySaveOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
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
    name: String,
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
        PyValue::std_ShapeValue { value } => {
            Ok(moose::standard::Shape(value.iter().map(|i| *i as usize).collect()).into())
        }
        PyValue::std_StringConstant { value } => Ok(Value::String(String::from(value))),
        PyValue::std_TensorConstant { value } => match value {
            PyNdarray::float32 {
                ref items,
                ref shape,
            } => {
                let shape: Vec<usize> = shape.iter().map(|i| *i as usize).collect();
                let tensor = ArrayD::from_shape_vec(shape, items.clone())?;
                Ok(Float32Tensor::from(tensor).into())
            }
            PyNdarray::float64 {
                ref items,
                ref shape,
            } => {
                let shape: Vec<usize> = shape.iter().map(|i| *i as usize).collect();
                let tensor = ArrayD::from_shape_vec(shape, items.clone())?;
                Ok(Float64Tensor::from(tensor).into())
            }
        },
    }
}

fn map_type(py_type: &PyValueType) -> Ty {
    match py_type {
        PyValueType::prim_PRFKeyType => Ty::PrfKeyTy,
        PyValueType::prim_SeedType => Ty::SeedTy,
        PyValueType::std_ShapeType => Ty::ShapeTy,
        PyValueType::std_UnitType => Ty::UnitTy,
        PyValueType::std_StringType => Ty::StringTy,
        PyValueType::std_TensorType => Ty::Float64TensorTy,
        PyValueType::std_BytesType => unimplemented!(),
        PyValueType::ring_RingTensorType => Ty::Ring64TensorTy,
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
                    std_AddOperation(op) => Ok(Operation {
                        kind: StdAdd(StdAddOp {
                            lhs: Ty::Float64TensorTy,
                            rhs: Ty::Float64TensorTy,
                        }),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SubOperation(op) => Ok(Operation {
                        kind: StdSub(StdSubOp {
                            lhs: Ty::Float64TensorTy,
                            rhs: Ty::Float64TensorTy,
                        }),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_MulOperation(op) => Ok(Operation {
                        kind: StdMul(StdMulOp {
                            lhs: Ty::Float64TensorTy,
                            rhs: Ty::Float64TensorTy,
                        }),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_DotOperation(op) => Ok(Operation {
                        kind: StdDot(StdDotOp {
                            lhs: Ty::Float64TensorTy,
                            rhs: Ty::Float64TensorTy,
                        }),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])?,
                        name: op.name.clone(),
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
                            ty: map_type(&op.output_type),
                        }),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SerializeOperation(op) => Ok(Operation {
                        kind: Identity(IdentityOp {
                            ty: map_type(&op.output_type),
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_DeserializeOperation(op) => Ok(Operation {
                        kind: Identity(IdentityOp {
                            ty: map_type(&op.output_type),
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_OutputOperation(op) => Ok(Operation {
                        kind: Output(OutputOp {
                            ty: map_type(&op.output_type),
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SaveOperation(op) => Ok(Operation {
                        kind: Save(SaveOp {
                            ty: map_type(&op.output_type),
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["key", "value"])?,
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
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Computation { operations })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maplit::hashmap;
    use moose::execution::EagerExecutor;
    use numpy::ToPyArray;
    use pyo3::prelude::*;

    fn create_computation_graph_from_python(py_any: &PyAny) -> Computation {
        let buf: Vec<u8> = py_any.extract().unwrap();
        let comp: PyComputation = rmp_serde::from_read_ref(&buf).unwrap();

        let rust_comp: Computation = comp.try_into().unwrap();
        rust_comp.toposort().unwrap()
    }

    fn run_computation(computation: &Computation) -> HashMap<String, Value> {
        let exec = EagerExecutor::new();
        let env = hashmap![];
        exec.run_computation(&computation, 12345, env).unwrap()
    }

    fn run_binary_func(x: &ArrayD<f64>, y: &ArrayD<f64>, py_code: &str) -> Value {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let xc = x.to_pyarray(py);
        let yc = y.to_pyarray(py);

        let comp_graph_py = PyModule::from_code(py, py_code, "comp_graph.py", "comp_graph")
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap();

        let py_any = comp_graph_py
            .getattr("f")
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap()
            .call1((xc, yc))
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap();

        let outputs = run_computation(&create_computation_graph_from_python(py_any));
        outputs["result"].clone()
    }

    fn run_call0_func(py_code: &str) -> Value {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let comp_graph_py = PyModule::from_code(py, py_code, "comp_graph.py", "comp_graph")
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap();

        let py_any = comp_graph_py
            .getattr("f")
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap()
            .call0()
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap();

        let comp = &create_computation_graph_from_python(py_any);

        let exec = EagerExecutor::new();
        let env = hashmap![];
        exec.run_computation(&comp, 12345, env).unwrap();
        // println!("weight = {:?}", exec.storage.load("w_uri".to_string()).unwrap());
        Value::Unit
    }

    #[test]
    fn test_deserialize_host_op() {
        let py_code = r#"
import numpy as np
from moose.computation import ring as ring_dialect
from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.utils import serialize_computation
from moose.computation.standard import TensorType
from moose.computation.standard import TensorConstant
from moose.computation.standard import UnitType
from moose.computation import dtypes
def f(arg1, arg2):
    comp = Computation(operations={}, placements={})
    alice = comp.add_placement(HostPlacement(name="alice"))

    x = np.array(arg1, dtype=np.float64)
    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="alice_input_x",
            value=TensorConstant(value = x),
            placement_name=alice.name,
            inputs={},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    y = np.array(arg2, dtype=np.float64)
    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="alice_input_y",
            value=TensorConstant(value = y),
            placement_name=alice.name,
            inputs={},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )
    comp.add_operation(
        standard_dialect.SPECIAL_OP(
                name="add",
                inputs={"lhs": "alice_input_x", "rhs": "alice_input_y"},
                placement_name=alice.name,
                output_type=TensorType(dtype=dtypes.float64),
        )
    )
    comp.add_operation(
        standard_dialect.OutputOperation(
                name="result",
                inputs={"value": "add"},
                placement_name=alice.name,
                output_type=UnitType(),
        )
    )

    return serialize_computation(comp)
        "#;

        let mul_code = py_code.replace("SPECIAL_OP", "MulOperation");
        let x1 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y1 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let result = run_binary_func(&x1, &y1, &mul_code);

        assert_eq!(
            result,
            Value::Float64Tensor(Float64Tensor::from(x1) * Float64Tensor::from(y1))
        );

        let x2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let add_code = py_code.replace("SPECIAL_OP", "AddOperation");
        let result = run_binary_func(&x2, &y2, &add_code);

        assert_eq!(
            result,
            Value::Float64Tensor(Float64Tensor::from(x2) + Float64Tensor::from(y2))
        );

        let x3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let sub_code = py_code.replace("SPECIAL_OP", "SubOperation");
        let result = run_binary_func(&x3, &y3, &sub_code);

        assert_eq!(
            result,
            Value::Float64Tensor(Float64Tensor::from(x3) - Float64Tensor::from(y3))
        );

        let x4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let dot_code = py_code.replace("SPECIAL_OP", "DotOperation");
        let result = run_binary_func(&x4, &y4, &dot_code);

        assert_eq!(
            result,
            Value::Float64Tensor(Float64Tensor::from(x4).dot(Float64Tensor::from(y4)))
        );
    }

    #[test]
    fn test_deserialize_replicated_op() {
        let py_code = r#"
import numpy as np

from moose.compiler.compiler import Compiler
from moose.computation import dtypes
from moose.computation import ring as ring_dialect
from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.standard import TensorType
from moose.computation.standard import UnitType
from moose.computation.utils import serialize_computation
from moose.computation.ring import RingTensorType

alice = HostPlacement(name="alice")
bob = HostPlacement(name="bob")
carole = HostPlacement(name="carole")
rep = ReplicatedPlacement(name="rep", player_names=["alice", "bob", "carole"])


def f(arg1, arg2):
    comp = Computation(operations={}, placements={})
    comp.add_placement(alice)
    comp.add_placement(bob)
    comp.add_placement(carole)
    comp.add_placement(rep)

    x = np.array(arg1, dtype=np.float64)
    y = np.array(arg2, dtype=np.float64)

    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="alice_input",
            value=standard_dialect.TensorConstant(value=x),
            placement_name=alice.name,
            inputs={},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="bob_input",
            value=standard_dialect.TensorConstant(value=y),
            placement_name=bob.name,
            inputs={},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    comp.add_operation(
        standard_dialect.SPECIAL_OP(
            name="rep_add",
            placement_name=rep.name,
            inputs={"lhs": "alice_input", "rhs": "bob_input"},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    comp.add_operation(
        standard_dialect.OutputOperation(
            name="result", placement_name=carole.name, inputs={"value": "rep_add"},
            output_type=RingTensorType(),
        )
    )

    compiler = Compiler()
    comp = compiler.run_passes(comp)

    return serialize_computation(comp)

"#;
        let mul_code = py_code.replace("SPECIAL_OP", "MulOperation");
        let x1 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y1 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let result = run_binary_func(&x1, &y1, &mul_code);

        assert_eq!(
            result,
            Value::Float64Tensor(Float64Tensor::from(x1) * Float64Tensor::from(y1))
        );

        let x2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let add_code = py_code.replace("SPECIAL_OP", "AddOperation");
        let result = run_binary_func(&x2, &y2, &add_code);

        assert_eq!(
            result,
            Value::Float64Tensor(Float64Tensor::from(x2) + Float64Tensor::from(y2))
        );

        let x3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let sub_code = py_code.replace("SPECIAL_OP", "SubOperation");
        let result = run_binary_func(&x3, &y3, &sub_code);

        assert_eq!(
            result,
            Value::Float64Tensor(Float64Tensor::from(x3) - Float64Tensor::from(y3))
        );

        let x4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let dot_code = py_code.replace("SPECIAL_OP", "DotOperation");
        let result = run_binary_func(&x4, &y4, &dot_code);

        assert_eq!(
            result,
            Value::Float64Tensor(Float64Tensor::from(x4).dot(Float64Tensor::from(y4)))
        );
    }
    #[test]
    fn test_constant() {
        let py_code = r#"
import numpy as np
from moose.computation import ring as ring_dialect
from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.utils import serialize_computation
from moose.computation.standard import TensorType
from moose.computation.standard import UnitType
from moose.computation import dtypes

def f():
    comp = Computation(operations={}, placements={})
    alice = comp.add_placement(HostPlacement(name="alice"))

    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="constant_0",
            inputs={},
            placement_name="alice",
            value=standard_dialect.StringConstant(value="w_uri"),
            output_type=TensorType(dtype=dtypes.string),
        )
    )

    return serialize_computation(comp)

    "#;

        let _ = run_call0_func(&py_code);
    }
    #[test]
    fn test_deserialize_linear_regression() {
        let py_code = r#"
import numpy as np

from moose import edsl
from moose.computation.utils import serialize_computation
from moose.computation import standard as standard_dialect


def generate_data(seed, n_instances, n_features, coeff=3, shift=10):
    rng = np.random.default_rng()
    x_data = rng.normal(size=(n_instances, n_features))
    y_data = np.dot(x_data, np.ones(n_features) * coeff) + shift
    return x_data, y_data


def f():
    x_owner = edsl.host_placement(name="x-owner")
    model_owner = edsl.host_placement(name="model-owner")

    x_uri, y_uri = generate_data(seed=42, n_instances=10, n_features=1)

    @edsl.computation
    def my_comp():

        with x_owner:
            X = edsl.constant(x_uri, dtype=edsl.float64)

        with model_owner:
            res = (
                edsl.save("w_uri", X),
            )

        return res

    concrete_comp = edsl.trace(my_comp)
    return serialize_computation(concrete_comp)

"#;
        let _ = run_call0_func(&py_code);
    }
}
