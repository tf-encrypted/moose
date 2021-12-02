//! Parser for computations defined in Python

use moose::computation::*;
use moose::host::{HostFloat32Tensor, HostFloat64Tensor, RawShape, SliceInfo, SliceInfoElem};
use moose::logical::TensorDType;
use ndarray::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
enum PyOperation {
    std_ConstantOperation(PyConstantOperation),
    std_AddOperation(PyAddOperation),
    std_SubOperation(PySubOperation),
    std_MulOperation(PyMulOperation),
    std_DotOperation(PyDotOperation),
    std_LessOperation(PyLessOperation),
    std_AtLeast2DOperation(PyAtLeast2DOperation),
    std_ShapeOperation(PyShapeOperation),
    std_SliceOperation(PySliceOperation),
    std_OnesOperation(PyOnesOperation),
    std_ConcatenateOperation(PyConcatenateOperation),
    std_DecryptOperation(PyDecryptOperation),
    std_TransposeOperation(PyTransposeOperation),
    std_ExpandDimsOperation(PyExpandDimsOperation),
    std_ExpOperation(PyExpOperation),
    std_InverseOperation(PyInverseOperation),
    std_MeanOperation(PyMeanOperation),
    std_SigmoidOperation(PySigmoidOperation),
    std_SqrtOperation(PySqrtOperation),
    std_SumOperation(PySumOperation),
    std_DivOperation(PyDivOperation),
    std_InputOperation(PyInputOperation),
    std_OutputOperation(PyOutputOperation),
    std_SaveOperation(PySaveOperation),
    std_LoadOperation(PyLoadOperation),
    std_CastOperation(PyCastOperation),
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
#[allow(clippy::upper_case_acronyms)]
enum PyValueType {
    std_BytesType,
    std_ShapeType,
    std_StringType,
    std_TensorType { dtype: PyDType },
    std_AesKeyType,
    std_AesTensorType { dtype: PyDType },
    std_UnitType,
    std_UnknownType,
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
    fixed8_27,
    fixed14_23,
    fixed46_40,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "__type__")]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
enum PyConstant {
    std_ShapeConstant { value: Vec<u8> },
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

type Inputs = HashMap<String, String>;

#[derive(Deserialize, Debug)]
struct PyConstantOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    value: PyConstant,
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
struct PyLessOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    input_type: PyValueType,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyAtLeast2DOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    to_column_vector: bool,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyShapeOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PySliceOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    begin: u32,
    end: u32,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyOnesOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    // dtype: Option<PyNdArray>,
}

#[derive(Deserialize, Debug)]
struct PyExpandDimsOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    axis: Vec<u32>,
}

#[derive(Deserialize, Debug)]
struct PyExpOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PySigmoidOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyConcatenateOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    axis: u32,
}

#[derive(Deserialize, Debug)]
struct PyDecryptOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyTransposeOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyInverseOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyMeanOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    axis: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct PySqrtOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PySumOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    axis: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct PyDivOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}
#[derive(Deserialize, Debug)]
struct PyCastOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyInputOperation {
    name: String,
    inputs: Inputs,
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
struct PyLoadOperation {
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
            PyPlacement::host_HostPlacement(plc) => Ok(Placement::Host(HostPlacement {
                owner: Role::from(&plc.name),
            })),
            PyPlacement::rep_ReplicatedPlacement(plc) => {
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
        PyConstant::std_ShapeConstant { value } => {
            Ok(RawShape(value.iter().map(|i| *i as usize).collect()).into())
        }
        PyConstant::std_StringConstant { value } => Ok(Constant::String(String::from(value))),
        PyConstant::std_TensorConstant { value } => match value {
            PyNdarray::float32 {
                ref items,
                ref shape,
            } => {
                let shape: Vec<usize> = shape.iter().map(|i| *i as usize).collect();
                let tensor = ArrayD::from_shape_vec(shape, items.clone())?;
                Ok(HostFloat32Tensor::from(tensor).into())
            }
            PyNdarray::float64 {
                ref items,
                ref shape,
            } => {
                let shape: Vec<usize> = shape.iter().map(|i| *i as usize).collect();
                let tensor = ArrayD::from_shape_vec(shape, items.clone())?;
                Ok(HostFloat64Tensor::from(tensor).into())
            }
        },
    }
}

fn map_type(py_type: &PyValueType) -> anyhow::Result<Ty> {
    match py_type {
        PyValueType::std_ShapeType => Ok(Ty::HostShape),
        PyValueType::std_UnitType => Ok(Ty::Unit),
        PyValueType::std_StringType => Ok(Ty::HostString),
        PyValueType::std_TensorType { dtype } => match dtype {
            PyDType::float32 => Ok(Ty::Tensor(TensorDType::Float32)),
            PyDType::float64 => Ok(Ty::Tensor(TensorDType::Float64)),
            PyDType::bool_ => Ok(Ty::Tensor(TensorDType::Bool)),
            // PyDType::int32 => Ok(Ty::HostInt32Tensor),
            // PyDType::int64 => Ok(Ty::HostInt64Tensor),
            // PyDType::uint32 => Ok(Ty::HostUint32Tensor),
            // PyDType::uint64 => Ok(Ty::HostUint64Tensor),
            PyDType::fixed14_23 => Ok(Ty::Tensor(TensorDType::Fixed128 {
                integral_precision: 14,
                fractional_precision: 23,
            })),
            PyDType::fixed8_27 => Ok(Ty::Tensor(TensorDType::Fixed128 {
                integral_precision: 8,
                fractional_precision: 27,
            })),
            PyDType::fixed46_40 => Ok(Ty::Tensor(TensorDType::Fixed128 {
                integral_precision: 46,
                fractional_precision: 40,
            })),
            _ => Err(anyhow::anyhow!("unimplemented dtype '{:?}'", dtype)),
        },
        PyValueType::std_AesTensorType { dtype } => match dtype {
            // TODO we are erasing fixedpoint precision here on purpose
            //  -- but we robably want to avoid this down the road
            PyDType::fixed46_40 => Ok(Ty::AesTensor),
            _ => Err(anyhow::anyhow!("unimplemented dtype '{:?}'", dtype)),
        },
        PyValueType::std_AesKeyType => Ok(Ty::AesKey),
        PyValueType::std_UnknownType => Ok(Ty::Unknown),
        PyValueType::std_BytesType => Err(anyhow::anyhow!("unimplemented type 'bytes'")),
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
                use anyhow::Context;
                use PyOperation::*;
                match op {
                    std_ConstantOperation(op) => Ok(Operation {
                        kind: ConstantOp {
                            sig: Signature::nullary(map_type(&op.output_type)?),
                            value: map_constant_value(&op.value)?,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_AddOperation(op) => Ok(Operation {
                        kind: AddOp {
                            // we can use output type type to determine input type
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SubOperation(op) => Ok(Operation {
                        kind: SubOp {
                            // we can use output type type to determine input type
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_MulOperation(op) => Ok(Operation {
                        kind: MulOp {
                            // we can use output type type to determine input type
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_DotOperation(op) => Ok(Operation {
                        kind: DotOp {
                            // we can use output type type to determine input type
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_LessOperation(op) => Ok(Operation {
                        kind: LessThanOp {
                            // we can use output type type to determine input type
                            sig: Signature::binary(
                                map_type(&op.input_type)?,
                                map_type(&op.input_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_AtLeast2DOperation(op) => Ok(Operation {
                        kind: AtLeast2DOp {
                            // we can use output type type to determine input type
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            to_column_vector: op.to_column_vector,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_ShapeOperation(op) => {
                        let plc = map_placement(&placements, &op.placement_name)?;
                        let ret = match plc {
                            Placement::Host(_) => Ty::HostShape,
                            Placement::Replicated(_) => Ty::ReplicatedShape,
                            _ => Ty::HostShape, // TODO(lvorona): Do we want to support std_Shape on any other placements?
                        };

                        Ok(Operation {
                            kind: ShapeOp {
                                sig: Signature::unary(Ty::Tensor(TensorDType::Unknown), ret),
                            }
                            .into(),
                            inputs: map_inputs(&op.inputs, &["x"])
                                .with_context(|| format!("Failed at op {:?}", op))?,
                            name: op.name.clone(),
                            placement: plc,
                        })
                    }
                    std_SliceOperation(op) => {
                        let typ = map_type(&op.output_type)?;
                        let plc = map_placement(&placements, &op.placement_name)?;
                        match &plc {
                            // TODO [kyle] Should all shape ops (HostSliceOp, RepSliceOp, SliceOp)
                            // be unified under the common SliceOp?
                            Placement::Replicated(_) => {
                                let typ = Ty::ReplicatedShape;
                                Ok(Operation {
                                    kind: RepSliceOp {
                                        sig: Signature::unary(typ, typ),
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
                                    placement: plc,
                                })
                            }
                            _ => Ok(Operation {
                                kind: SliceOp {
                                    sig: Signature::unary(typ, typ),
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
                                placement: plc,
                            }),
                        }
                    }
                    std_OnesOperation(op) => Ok(Operation {
                        kind: OnesOp {
                            sig: Signature::unary(Ty::HostShape, map_type(&op.output_type)?),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["shape"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_ExpandDimsOperation(op) => Ok(Operation {
                        kind: ExpandDimsOp {
                            // assume input type is the same as the output type
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            axis: op.axis.clone(),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_ExpOperation(op) => Ok(Operation {
                        kind: ExpOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SigmoidOperation(op) => Ok(Operation {
                        kind: SigmoidOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_ConcatenateOperation(op) => {
                        let mut inputs: Vec<(&String, &String)> = op.inputs.iter().collect();
                        inputs.sort_by_key(|x| x.0);
                        let sorted_input_names: Vec<String> =
                            inputs.into_iter().map(|(_k, v)| v.clone()).collect();
                        Ok(Operation {
                            kind: ConcatOp {
                                // assume input type is the same as output type
                                sig: Signature::variadic(
                                    map_type(&op.output_type)?,
                                    map_type(&op.output_type)?,
                                ),
                                axis: op.axis,
                            }
                            .into(),
                            inputs: sorted_input_names,
                            name: op.name.clone(),
                            placement: map_placement(&placements, &op.placement_name)?,
                        })
                    }
                    std_DecryptOperation(op) => Ok(Operation {
                        kind: AesDecryptOp {
                            sig: Signature::binary(
                                Ty::AesKey,
                                Ty::AesTensor,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["key", "ciphertext"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_TransposeOperation(op) => Ok(Operation {
                        kind: TransposeOp {
                            // we can use output type type to determine input type
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),

                    std_InverseOperation(op) => Ok(Operation {
                        kind: InverseOp {
                            // we can use output type type to determine input type
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_MeanOperation(op) => Ok(Operation {
                        kind: MeanOp {
                            // we can use output type type to determine input type
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            axis: op.axis,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SqrtOperation(op) => Ok(Operation {
                        kind: HostSqrtOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SumOperation(op) => Ok(Operation {
                        kind: SumOp {
                            // we can use output type type to determine input type
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            axis: op.axis,
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_DivOperation(op) => Ok(Operation {
                        kind: DivOp {
                            // we can use output type type to determine input type
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_InputOperation(op) => Ok(Operation {
                        kind: InputOp {
                            sig: Signature::nullary(map_type(&op.output_type)?),
                            arg_name: op.name.clone(),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_OutputOperation(op) => Ok(Operation {
                        kind: OutputOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SaveOperation(op) => Ok(Operation {
                        kind: SaveOp {
                            sig: Signature::binary(Ty::HostString, Ty::Unknown, Ty::Unit),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["key", "value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_LoadOperation(op) => Ok(Operation {
                        kind: LoadOp {
                            sig: Signature::binary(
                                Ty::HostString,
                                Ty::HostString,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["key", "query"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_CastOperation(op) => Ok(Operation {
                        kind: CastOp {
                            sig: Signature::unary(Ty::Unknown, map_type(&op.output_type)?),
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
