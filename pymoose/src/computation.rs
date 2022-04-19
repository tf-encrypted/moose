//! Parser for computations defined in Python

use moose::computation::*;
use moose::host::{FromRaw, HostPlacement, RawShape, SliceInfo, SliceInfoElem};
use moose::logical::{TensorDType, TensorShape};
use moose::mirrored::Mirrored3Placement;
use moose::replicated::ReplicatedPlacement;
use ndarray::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
enum PyOperation {
    AbsOperation(PyAbsOperation),
    AddNOperation(PyAddNOperation),
    IdentityOperation(PyIdentityOperation),
    ConstantOperation(PyConstantOperation),
    AddOperation(PyAddOperation),
    SubOperation(PySubOperation),
    MulOperation(PyMulOperation),
    DotOperation(PyDotOperation),
    BitwiseOrOperation(PyBitwiseOrOperation),
    LessOperation(PyLessOperation),
    GreaterOperation(PyGreaterOperation),
    MuxOperation(PyMuxOperation),
    AtLeast2DOperation(PyAtLeast2DOperation),
    ShapeOperation(PyShapeOperation),
    IndexAxisOperation(PyIndexAxisOperation),
    SliceOperation(PySliceOperation),
    OnesOperation(PyOnesOperation),
    ZerosOperation(PyZerosOperation),
    ConcatenateOperation(PyConcatenateOperation),
    MaximumOperation(PyMaximumOperation),
    DecryptOperation(PyDecryptOperation),
    TransposeOperation(PyTransposeOperation),
    ExpandDimsOperation(PyExpandDimsOperation),
    ExpOperation(PyExpOperation),
    InverseOperation(PyInverseOperation),
    MeanOperation(PyMeanOperation),
    SigmoidOperation(PySigmoidOperation),
    LogOperation(PyLogOperation),
    Log2Operation(PyLog2Operation),
    SoftmaxOperation(PySoftmaxOperation),
    ArgmaxOperation(PyArgmaxOperation),
    SqrtOperation(PySqrtOperation),
    SumOperation(PySumOperation),
    DivOperation(PyDivOperation),
    InputOperation(PyInputOperation),
    OutputOperation(PyOutputOperation),
    SaveOperation(PySaveOperation),
    LoadOperation(PyLoadOperation),
    CastOperation(PyCastOperation),
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
#[allow(clippy::upper_case_acronyms)]
enum PyValueType {
    BytesType,
    ShapeType,
    StringType,
    TensorType { dtype: PyDType },
    AesKeyType,
    AesTensorType { dtype: PyDType },
    UnitType,
    UnknownType,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "name")]
#[allow(non_camel_case_types)]
enum PyDType {
    float32,
    float64,
    int32,
    int64,
    uint32,
    uint64,
    bool_,
    fixed {
        integral_precision: u32,
        fractional_precision: u32,
    },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
enum PyConstant {
    ShapeConstant { value: Vec<usize> },
    StringConstant { value: String },
    TensorConstant { value: PyNdarray },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "dtype")]
#[allow(non_camel_case_types)]
enum PyNdarray {
    float32 { items: Vec<f32>, shape: Vec<usize> },
    float64 { items: Vec<f64>, shape: Vec<usize> },
    uint64 { items: Vec<u64>, shape: Vec<usize> },
    bool { items: Vec<bool>, shape: Vec<usize> },
}

type Inputs = HashMap<String, String>;

#[derive(Deserialize, Debug)]
struct PyOpSignature {
    input_types: HashMap<String, PyValueType>,
    return_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyAbsOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyAddNOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyIdentityOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyConstantOperation {
    name: String,
    #[allow(dead_code)]
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
    value: PyConstant,
}

#[derive(Deserialize, Debug)]
struct PyAddOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PySubOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyMulOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyDotOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyLessOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyGreaterOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyBitwiseOrOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyMuxOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyAtLeast2DOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
    to_column_vector: bool,
}

#[derive(Deserialize, Debug)]
struct PyShapeOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyIndexAxisOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
    axis: usize,
    index: usize,
}

#[derive(Deserialize, Debug)]
struct PySliceOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
    begin: u32,
    end: u32,
}

#[derive(Deserialize, Debug)]
struct PyOnesOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyZerosOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyExpandDimsOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
    axis: Vec<usize>,
}

#[derive(Deserialize, Debug)]
struct PyExpOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PySigmoidOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyLogOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyLog2Operation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PySoftmaxOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
    axis: usize,
    upmost_index: u32,
}

#[derive(Deserialize, Debug)]
struct PyArgmaxOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
    axis: usize,
    upmost_index: u32,
}

#[derive(Deserialize, Debug)]
struct PyConcatenateOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
    axis: u32,
}

#[derive(Deserialize, Debug)]
struct PyMaximumOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyDecryptOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyTransposeOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyInverseOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyMeanOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
    axis: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct PySqrtOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PySumOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
    axis: Option<usize>,
}

#[derive(Deserialize, Debug)]
struct PyDivOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}
#[derive(Deserialize, Debug)]
struct PyCastOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyInputOperation {
    name: String,
    #[allow(dead_code)]
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyOutputOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PyLoadOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
struct PySaveOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    signature: PyOpSignature,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
enum PyPlacement {
    HostPlacement(PyHostPlacement),
    MirroredPlacement(PyMirroredPlacement),
    ReplicatedPlacement(PyReplicatedPlacement),
}

#[derive(Deserialize, Debug)]
struct PyHostPlacement {
    name: String,
}

#[derive(Deserialize, Debug)]
struct PyReplicatedPlacement {
    #[allow(dead_code)]
    name: String,
    player_names: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct PyMirroredPlacement {
    #[allow(dead_code)]
    name: String,
    player_names: Vec<String>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
pub struct PyComputation {
    operations: HashMap<String, PyOperation>,
    placements: HashMap<String, PyPlacement>,
}

impl PyComputation {
    pub fn from_read(computation: &[u8]) -> anyhow::Result<Self> {
        let py_comp: PyComputation = rmp_serde::from_read_ref(&computation)?;
        Ok(py_comp)
    }
}

impl TryFrom<&PyPlacement> for Placement {
    type Error = anyhow::Error;
    fn try_from(placement: &PyPlacement) -> anyhow::Result<Placement> {
        match placement {
            PyPlacement::HostPlacement(plc) => Ok(Placement::Host(HostPlacement {
                owner: Role::from(&plc.name),
            })),
            PyPlacement::MirroredPlacement(plc) => {
                if plc.player_names.len() != 3 {
                    return Err(anyhow::anyhow!("Placement doesn't have 3 players"));
                }
                Ok(Placement::Mirrored3(Mirrored3Placement {
                    owners: [
                        Role::from(&plc.player_names[0]),
                        Role::from(&plc.player_names[1]),
                        Role::from(&plc.player_names[2]),
                    ],
                }))
            }
            PyPlacement::ReplicatedPlacement(plc) => {
                if plc.player_names.len() != 3 {
                    return Err(anyhow::anyhow!("Placement doesn't have 3 players"));
                }
                Ok(Placement::Replicated(ReplicatedPlacement {
                    owners: [
                        Role::from(&plc.player_names[0]),
                        Role::from(&plc.player_names[1]),
                        Role::from(&plc.player_names[2]),
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
                .ok_or_else(|| anyhow::anyhow!("'{:?}' not found in input vector", item))
        })
        .collect::<anyhow::Result<Vec<_>>>()
}

fn map_placement(plc: &HashMap<String, Placement>, name: &str) -> anyhow::Result<Placement> {
    plc.get(name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("No key found in placement dictionary"))
}

fn map_constant_value(constant_value: &PyConstant) -> anyhow::Result<Constant> {
    match constant_value {
        PyConstant::ShapeConstant { value } => {
            Ok(RawShape(value.iter().map(|i| *i as usize).collect()).into())
        }
        PyConstant::StringConstant { value } => Ok(Constant::String(String::from(value))),
        PyConstant::TensorConstant { value } => match value {
            PyNdarray::float32 {
                ref items,
                ref shape,
            } => {
                let shape: Vec<usize> = shape.iter().map(|i| *i as usize).collect();
                let tensor = ArrayD::from_shape_vec(shape, items.clone())?;
                let plc = HostPlacement::from("TODO");
                Ok(Constant::HostFloat32Tensor(plc.from_raw(tensor)))
            }
            PyNdarray::float64 {
                ref items,
                ref shape,
            } => {
                let shape: Vec<usize> = shape.iter().map(|i| *i as usize).collect();
                let tensor = ArrayD::from_shape_vec(shape, items.clone())?;
                let plc = HostPlacement::from("TODO");
                Ok(Constant::HostFloat64Tensor(plc.from_raw(tensor)))
            }
            PyNdarray::uint64 {
                ref items,
                ref shape,
            } => {
                let shape: Vec<usize> = shape.iter().map(|i| *i as usize).collect();
                let tensor = ArrayD::from_shape_vec(shape, items.clone())?;
                let plc = HostPlacement::from("TODO");
                Ok(Constant::HostUint64Tensor(plc.from_raw(tensor)))
            }
            PyNdarray::bool {
                ref items,
                ref shape,
            } => {
                use ::moose::host::BitArrayRepr;
                use ::moose::host::HostBitTensor;
                let shape: RawShape = RawShape(shape.iter().map(|i| *i as usize).collect());
                let items: Vec<u8> = items.iter().map(|x| *x as u8).collect();
                let tensor = BitArrayRepr::from_vec(items, &shape);
                let plc = HostPlacement::from("TODO");
                Ok(Constant::HostBitTensor(HostBitTensor(tensor, plc)))
            }
        },
    }
}

fn map_type(py_type: &PyValueType, placement: Option<&Placement>) -> anyhow::Result<Ty> {
    match py_type {
        PyValueType::ShapeType => match placement {
            Some(Placement::Host(_)) => Ok(Ty::Shape(TensorShape::Host)),
            Some(Placement::Replicated(_)) => Ok(Ty::Shape(TensorShape::Replicated)),
            Some(other) => Err(anyhow::anyhow!(
                "Do not know to map ShapeType to a placement {:?}",
                other
            )),
            None => Err(anyhow::anyhow!(
                "Expected placement information to map ShapeType, found None"
            )),
        },
        PyValueType::UnitType => Ok(Ty::HostUnit),
        PyValueType::StringType => Ok(Ty::HostString),
        PyValueType::TensorType { dtype } => match dtype {
            PyDType::float32 => Ok(Ty::Tensor(TensorDType::Float32)),
            PyDType::float64 => Ok(Ty::Tensor(TensorDType::Float64)),
            PyDType::bool_ => Ok(Ty::Tensor(TensorDType::Bool)),
            PyDType::uint64 => Ok(Ty::Tensor(TensorDType::Uint64)),
            // PyDType::int32 => Ok(Ty::HostInt32Tensor),
            // PyDType::int64 => Ok(Ty::HostInt64Tensor),
            // PyDType::uint32 => Ok(Ty::HostUint32Tensor),
            PyDType::fixed {
                integral_precision,
                fractional_precision,
            } => Ok(Ty::Tensor(TensorDType::Fixed128 {
                integral_precision: *integral_precision,
                fractional_precision: *fractional_precision,
            })),
            _ => Err(anyhow::anyhow!("unimplemented dtype '{:?}'", dtype)),
        },
        PyValueType::AesTensorType { dtype } => match dtype {
            PyDType::fixed {
                integral_precision: 24,
                fractional_precision: 40,
            } => Ok(Ty::AesTensor),
            // TODO we are erasing fixedpoint precision here on purpose
            //  -- but we pprobably want to support other precisions down the road
            PyDType::fixed { .. } => Err(anyhow::anyhow!(
                "Unsupported precision for the fixedpoint AES Tensor '{:?}'",
                dtype
            )),
            _ => Err(anyhow::anyhow!("unimplemented dtype '{:?}'", dtype)),
        },
        PyValueType::AesKeyType => Ok(Ty::AesKey),
        PyValueType::UnknownType => Ok(Ty::Unknown),
        PyValueType::BytesType => Err(anyhow::anyhow!("unimplemented type 'bytes'")),
    }
}

fn map_signature(
    pysig: &PyOpSignature,
    plc: &HashMap<String, Placement>,
    placement_name: &str,
    expected_inputs: &[&str],
) -> anyhow::Result<Signature> {
    let placement = plc.get(placement_name);
    let get_arg = |name: &str| {
        pysig
            .input_types
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Missing type information for argument {}", name))
    };
    match expected_inputs {
        [] => Ok(Signature::nullary(map_type(&pysig.return_type, placement)?)),
        [arg0] => Ok(Signature::unary(
            map_type(get_arg(arg0)?, placement)?,
            map_type(&pysig.return_type, placement)?,
        )),
        [arg0, arg1] => Ok(Signature::binary(
            map_type(get_arg(arg0)?, placement)?,
            map_type(get_arg(arg1)?, placement)?,
            map_type(&pysig.return_type, placement)?,
        )),
        [arg0, arg1, arg2] => Ok(Signature::ternary(
            map_type(get_arg(arg0)?, placement)?,
            map_type(get_arg(arg1)?, placement)?,
            map_type(get_arg(arg2)?, placement)?,
            map_type(&pysig.return_type, placement)?,
        )),
        inputs => Err(anyhow::anyhow!(
            "Too many inputs to map into a signature of fixed arity {:?}",
            inputs
        )),
    }
}

fn map_signature_variadic(
    pysig: &PyOpSignature,
    plc: &HashMap<String, Placement>,
    placement_name: &str,
    any_arg: &str,
) -> anyhow::Result<Signature> {
    let placement = plc.get(placement_name);
    Ok(Signature::variadic(
        map_type(
            pysig.input_types.get(any_arg).ok_or_else(|| {
                anyhow::anyhow!("Missing type information for argument {}", any_arg)
            })?,
            placement,
        )?,
        map_type(&pysig.return_type, placement)?,
    ))
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
                use anyhow::Context;
                use PyOperation::*;
                match op {
                    AbsOperation(op) => Ok(Operation {
                        kind: AbsOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    AddNOperation(op) => {
                        let mut inputs: Vec<(&String, &String)> = op.inputs.iter().collect();
                        inputs.sort_by_key(|x| x.0);
                        let sorted_input_names: Vec<String> =
                            inputs.into_iter().map(|(_k, v)| v.clone()).collect();
                        Ok(Operation {
                            kind: AddNOp {
                                sig: map_signature_variadic(
                                    &op.signature,
                                    &placements,
                                    &op.placement_name,
                                    "array0",
                                )?,
                            }
                            .into(),
                            inputs: sorted_input_names,
                            name: op.name.clone(),
                            placement: map_placement(&placements, &op.placement_name)?,
                        })
                    }
                    IdentityOperation(op) => Ok(Operation {
                        kind: IdentityOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ConstantOperation(op) => Ok(Operation {
                        kind: ConstantOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &[],
                            )?,
                            value: map_constant_value(&op.value)?,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    AddOperation(op) => Ok(Operation {
                        kind: AddOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["lhs", "rhs"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    SubOperation(op) => Ok(Operation {
                        kind: SubOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["lhs", "rhs"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    MulOperation(op) => Ok(Operation {
                        kind: MulOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["lhs", "rhs"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    DotOperation(op) => Ok(Operation {
                        kind: DotOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["lhs", "rhs"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    LessOperation(op) => Ok(Operation {
                        kind: LessOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["lhs", "rhs"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    GreaterOperation(op) => Ok(Operation {
                        kind: GreaterOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["lhs", "rhs"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    BitwiseOrOperation(op) => Ok(Operation {
                        kind: OrOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["lhs", "rhs"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    MuxOperation(op) => Ok(Operation {
                        kind: MuxOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["selector", "x", "y"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["selector", "x", "y"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    AtLeast2DOperation(op) => Ok(Operation {
                        kind: AtLeast2DOp {
                            // we can use output type type to determine input type
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                            to_column_vector: op.to_column_vector,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ShapeOperation(op) => Ok(Operation {
                        kind: ShapeOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    IndexAxisOperation(op) => Ok(Operation {
                        kind: IndexAxisOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                            axis: op.axis,
                            index: op.index,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    SliceOperation(op) => Ok(Operation {
                        kind: SliceOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                            slice: SliceInfo(vec![SliceInfoElem {
                                start: op.begin as isize,
                                step: Some(1),
                                end: Some(op.end as isize),
                            }]),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    OnesOperation(op) => Ok(Operation {
                        kind: OnesOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["shape"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["shape"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ZerosOperation(op) => Ok(Operation {
                        kind: ZerosOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["shape"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["shape"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ExpandDimsOperation(op) => Ok(Operation {
                        kind: ExpandDimsOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                            axis: op.axis.clone(),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ExpOperation(op) => Ok(Operation {
                        kind: ExpOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    SigmoidOperation(op) => Ok(Operation {
                        kind: SigmoidOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    LogOperation(op) => Ok(Operation {
                        kind: LogOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    Log2Operation(op) => Ok(Operation {
                        kind: Log2Op {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    SoftmaxOperation(op) => Ok(Operation {
                        kind: SoftmaxOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                            axis: op.axis,
                            upmost_index: op.upmost_index as usize,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ArgmaxOperation(op) => Ok(Operation {
                        kind: ArgmaxOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                            axis: op.axis,
                            upmost_index: op.upmost_index as usize,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ConcatenateOperation(op) => {
                        let mut inputs: Vec<(&String, &String)> = op.inputs.iter().collect();
                        inputs.sort_by_key(|x| x.0);
                        let sorted_input_names: Vec<String> =
                            inputs.into_iter().map(|(_k, v)| v.clone()).collect();
                        Ok(Operation {
                            kind: ConcatOp {
                                // TODO[jason] input_types should actually just be a single type, not one for each array
                                sig: map_signature_variadic(
                                    &op.signature,
                                    &placements,
                                    &op.placement_name,
                                    "array0",
                                )?,
                                axis: op.axis,
                            }
                            .into(),
                            inputs: sorted_input_names,
                            name: op.name.clone(),
                            placement: map_placement(&placements, &op.placement_name)?,
                        })
                    }
                    MaximumOperation(op) => {
                        let mut inputs: Vec<(&String, &String)> = op.inputs.iter().collect();
                        inputs.sort_by_key(|x| x.0);
                        let sorted_input_names: Vec<String> =
                            inputs.into_iter().map(|(_k, v)| v.clone()).collect();
                        Ok(Operation {
                            kind: MaximumOp {
                                // TODO[jason] input_types should actually just be a single type, not one for each array
                                sig: map_signature_variadic(
                                    &op.signature,
                                    &placements,
                                    &op.placement_name,
                                    "array0",
                                )?,
                            }
                            .into(),
                            inputs: sorted_input_names,
                            name: op.name.clone(),
                            placement: map_placement(&placements, &op.placement_name)?,
                        })
                    }
                    DecryptOperation(op) => Ok(Operation {
                        kind: DecryptOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["key", "ciphertext"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["key", "ciphertext"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    TransposeOperation(op) => Ok(Operation {
                        kind: TransposeOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),

                    InverseOperation(op) => Ok(Operation {
                        kind: InverseOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    MeanOperation(op) => Ok(Operation {
                        kind: MeanOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                            axis: op.axis,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    SqrtOperation(op) => Ok(Operation {
                        kind: SqrtOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    SumOperation(op) => Ok(Operation {
                        kind: SumOp {
                            // we can use output type type to determine input type
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                            axis: op.axis,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    DivOperation(op) => Ok(Operation {
                        kind: DivOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["lhs", "rhs"],
                            )?,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    InputOperation(op) => Ok(Operation {
                        kind: InputOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &[],
                            )?,

                            arg_name: op.name.clone(),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    OutputOperation(op) => Ok(Operation {
                        kind: OutputOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["value"],
                            )?,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    SaveOperation(op) => Ok(Operation {
                        kind: SaveOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["key", "value"],
                            )?,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["key", "value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    LoadOperation(op) => Ok(Operation {
                        kind: LoadOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["key", "query"],
                            )?,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["key", "query"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    CastOperation(op) => Ok(Operation {
                        kind: CastOp {
                            sig: map_signature(
                                &op.signature,
                                &placements,
                                &op.placement_name,
                                &["x"],
                            )?,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
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
    use moose::textual::ToTextual;

    #[test]
    fn test_shape_replicated() -> Result<(), anyhow::Error> {
        let py_op = PyShapeOperation {
            name: "x".to_string(),
            inputs: HashMap::from([("x".to_string(), "op_1".to_string())]),
            placement_name: "rep".to_string(),
            signature: PyOpSignature {
                input_types: HashMap::from([(
                    "x".to_string(),
                    PyValueType::TensorType {
                        dtype: PyDType::float32,
                    },
                )]),
                return_type: PyValueType::ShapeType,
            },
        };
        let py_comp = PyComputation {
            operations: HashMap::from([("op1".to_string(), PyOperation::ShapeOperation(py_op))]),
            placements: HashMap::from([(
                "rep".to_string(),
                PyPlacement::ReplicatedPlacement(PyReplicatedPlacement {
                    name: "rep".to_string(),
                    player_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                }),
            )]),
        };
        let comp: Computation = py_comp.try_into()?;
        assert_eq!(
            comp.operations[0].to_textual(),
            "x = Shape: (Tensor<Float32>) -> Shape<Replicated> (op_1) @Replicated(a, b, c)"
        );
        Ok(())
    }

    #[test]
    fn test_shape_host() -> Result<(), anyhow::Error> {
        let py_op = PyShapeOperation {
            name: "x".to_string(),
            inputs: HashMap::from([("x".to_string(), "op_1".to_string())]),
            placement_name: "host".to_string(),
            signature: PyOpSignature {
                input_types: HashMap::from([(
                    "x".to_string(),
                    PyValueType::TensorType {
                        dtype: PyDType::float32,
                    },
                )]),
                return_type: PyValueType::ShapeType,
            },
        };
        let py_comp = PyComputation {
            operations: HashMap::from([("op1".to_string(), PyOperation::ShapeOperation(py_op))]),
            placements: HashMap::from([(
                "host".to_string(),
                PyPlacement::HostPlacement(PyHostPlacement {
                    name: "host".to_string(),
                }),
            )]),
        };
        let comp: Computation = py_comp.try_into()?;
        assert_eq!(
            comp.operations[0].to_textual(),
            "x = Shape: (Tensor<Float32>) -> Shape<Host> (op_1) @Host(host)"
        );
        Ok(())
    }
}
