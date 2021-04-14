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
    ring_RingSumOperation(PyRingSumOperation),
    ring_RingMeanOperation(PyRingMeanOperation),
    ring_FillTensorOperation(PyFillTensorOperation),
    ring_RingShlOperation(PyRingShlOperation),
    ring_RingShrOperation(PyRingShrOperation),
    std_ConstantOperation(PyConstantOperation),
    std_AddOperation(PyAddOperation),
    std_SubOperation(PySubOperation),
    std_MulOperation(PyMulOperation),
    std_DotOperation(PyDotOperation),
    std_AtLeast2DOperation(PyAtLeast2DOperation),
    std_ShapeOperation(PyShapeOperation),
    std_SliceOperation(PySliceOperation),
    std_OnesOperation(PyOnesOperation),
    std_ConcatenateOperation(PyConcatenateOperation),
    std_TransposeOperation(PyTransposeOperation),
    std_ExpandDimsOperation(PyExpandDimsOperation),
    std_InverseOperation(PyInverseOperation),
    std_MeanOperation(PyMeanOperation),
    std_SumOperation(PySumOperation),
    std_DivOperation(PyDivOperation),
    std_SerializeOperation(PySerializeOperation),
    std_DeserializeOperation(PyDeserializeOperation),
    std_SendOperation(PySendOperation),
    std_OutputOperation(PyOutputOperation),
    std_SaveOperation(PySaveOperation),
    std_LoadOperation(PyLoadOperation),
    std_ReceiveOperation(PyReceiveOperation),
    fixed_RingEncodeOperation(PyRingEncodeOperation),
    fixed_RingDecodeOperation(PyRingDecodeOperation),
    fixed_RingMeanOperation(PyFixedRingMeanOperation),
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
    std_UnknownType,
    ring_RingTensorType,
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
struct PyRingSumOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    axis: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct PyRingMeanOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    axis: Option<u32>,
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
    axis: u32,
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
struct PyRingEncodeOperation {
    name: String,
    scaling_factor: u64,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyFixedRingMeanOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    axis: Option<u32>,
    precision: u64, // TODO(Dragos) change this to precision
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
                .ok_or_else(|| anyhow::anyhow!("'{:?}' not found in input vector", item))
        })
        .collect::<anyhow::Result<Vec<_>>>()
}

fn map_placement(plc: &HashMap<String, Placement>, name: &str) -> anyhow::Result<Placement> {
    plc.get(name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("No key found in placement dictionary"))
}

fn map_constant_value(constant_value: &PyConstant) -> anyhow::Result<Value> {
    match constant_value {
        PyConstant::std_ShapeConstant { value } => {
            Ok(moose::standard::Shape(value.iter().map(|i| *i as usize).collect()).into())
        }
        PyConstant::std_StringConstant { value } => Ok(Value::String(String::from(value))),
        PyConstant::std_TensorConstant { value } => match value {
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
        PyValueType::std_UnknownType => Ty::Float64TensorTy, // TODO(Dragos) fixme
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
                use anyhow::Context;
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
                        inputs: map_inputs(&op.inputs, &["key"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingAddOperation(op) => Ok(Operation {
                        kind: RingAdd(RingAddOp {
                            lhs: Ty::Ring64TensorTy,
                            rhs: Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingSubOperation(op) => Ok(Operation {
                        kind: RingSub(RingSubOp {
                            lhs: Ty::Ring64TensorTy,
                            rhs: Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingMulOperation(op) => Ok(Operation {
                        kind: RingMul(RingMulOp {
                            lhs: Ty::Ring64TensorTy,
                            rhs: Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingDotOperation(op) => Ok(Operation {
                        kind: RingDot(RingDotOp {
                            lhs: Ty::Ring64TensorTy,
                            rhs: Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingShapeOperation(op) => Ok(Operation {
                        kind: RingShape(RingShapeOp {
                            ty: Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["tensor"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingSampleOperation(op) => Ok(Operation {
                        kind: RingSample(RingSampleOp {
                            output: Ty::Ring64TensorTy,
                            max_value: op.max_value,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["shape", "seed"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingSumOperation(op) => Ok(Operation {
                        kind: RingSum(RingSumOp {
                            ty: Ty::Ring64TensorTy,
                            axis: op.axis,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingMeanOperation(op) => Ok(Operation {
                        // TODO
                        kind: Identity(IdentityOp {
                            ty: Ty::Ring64TensorTy,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_FillTensorOperation(op) => Ok(Operation {
                        kind: RingFill(RingFillOp { value: op.value }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["shape"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingShlOperation(op) => Ok(Operation {
                        kind: RingShl(RingShlOp {
                            amount: op.amount as usize,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingShrOperation(op) => Ok(Operation {
                        kind: RingShr(RingShrOp {
                            amount: op.amount as usize,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
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
                            // we can use output type type to determine input type
                            lhs: map_type(&op.output_type),
                            rhs: map_type(&op.output_type),
                        }),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SubOperation(op) => Ok(Operation {
                        kind: StdSub(StdSubOp {
                            // we can use output type type to determine input type
                            lhs: map_type(&op.output_type),
                            rhs: map_type(&op.output_type),
                        }),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_MulOperation(op) => Ok(Operation {
                        kind: StdMul(StdMulOp {
                            // we can use output type type to determine input type
                            lhs: map_type(&op.output_type),
                            rhs: map_type(&op.output_type),
                        }),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_DotOperation(op) => Ok(Operation {
                        kind: StdDot(StdDotOp {
                            // we can use output type type to determine input type
                            lhs: map_type(&op.output_type),
                            rhs: map_type(&op.output_type),
                        }),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_AtLeast2DOperation(op) => Ok(Operation {
                        kind: StdAtLeast2D(StdAtLeast2DOp {
                            // we can use output type type to determine input type
                            ty: map_type(&op.output_type),
                            to_column_vector: op.to_column_vector,
                        }),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_ShapeOperation(op) => Ok(Operation {
                        kind: StdShape(StdShapeOp {
                            ty: Ty::Float64TensorTy,
                        }),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SliceOperation(op) => Ok(Operation {
                        kind: StdSlice(StdSliceOp {
                            ty: Ty::ShapeTy,
                            start: op.begin,
                            end: op.end,
                        }),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_OnesOperation(op) => Ok(Operation {
                        kind: StdOnes(StdOnesOp {
                            // we can use output type type to determine input type
                            ty: map_type(&op.output_type),
                        }),
                        inputs: map_inputs(&op.inputs, &["shape"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_ExpandDimsOperation(op) => Ok(Operation {
                        kind: StdExpandDims(StdExpandDimsOp {
                            ty: Ty::Float64TensorTy, // TODO
                            axis: op.axis,
                        }),
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
                            kind: StdConcatenate(StdConcatenateOp {
                                ty: Ty::Float64TensorTy, // TODO
                                axis: op.axis,
                            }),
                            inputs: sorted_input_names,
                            name: op.name.clone(),
                            placement: map_placement(&placements, &op.placement_name)?,
                        })
                    }
                    std_TransposeOperation(op) => Ok(Operation {
                        kind: StdTranspose(StdTransposeOp {
                            // we can use output type type to determine input type
                            ty: map_type(&op.output_type),
                        }),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),

                    std_InverseOperation(op) => Ok(Operation {
                        kind: StdInverse(StdInverseOp {
                            // we can use output type type to determine input type
                            ty: map_type(&op.output_type),
                        }),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_MeanOperation(op) => Ok(Operation {
                        kind: StdMean(StdMeanOp {
                            // we can use output type type to determine input type
                            ty: map_type(&op.output_type),
                            axis: op.axis,
                        }),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SumOperation(op) => Ok(Operation {
                        kind: StdSum(StdSumOp {
                            // we can use output type type to determine input type
                            ty: map_type(&op.output_type),
                            axis: op.axis,
                        }),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_DivOperation(op) => Ok(Operation {
                        kind: StdDiv(StdDivOp {
                            // we can use output type type to determine input type
                            lhs: map_type(&op.output_type),
                            rhs: map_type(&op.output_type),
                        }),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
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
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
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
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_DeserializeOperation(op) => Ok(Operation {
                        kind: Identity(IdentityOp {
                            ty: map_type(&op.output_type),
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_OutputOperation(op) => Ok(Operation {
                        kind: Output(OutputOp {
                            ty: map_type(&op.output_type),
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SaveOperation(op) => Ok(Operation {
                        kind: Save(SaveOp {
                            ty: Ty::Float64TensorTy, // TODO
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["key", "value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_LoadOperation(op) => Ok(Operation {
                        kind: Load(LoadOp {
                            ty: map_type(&op.output_type),
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["key", "query"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_RingEncodeOperation(op) => Ok(Operation {
                        kind: FixedpointRingEncode(FixedpointRingEncodeOp {
                            scaling_factor: op.scaling_factor,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_RingDecodeOperation(op) => Ok(Operation {
                        kind: FixedpointRingDecode(FixedpointRingDecodeOp {
                            scaling_factor: op.scaling_factor,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_RingMeanOperation(op) => Ok(Operation {
                        kind: FixedpointRingMean(FixedpointRingMeanOp {
                            axis: op.axis.map(|x| x as usize),
                            scaling_factor: op.precision,
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
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
    use approx::AbsDiffEq;
    use maplit::hashmap;
    use moose::execution::EagerExecutor;
    use moose::storage::{LocalSyncStorage, SyncStorage};
    use numpy::ToPyArray;
    use pyo3::prelude::*;
    use rand::Rng;
    use std::rc::Rc;

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

    fn generate_python_names() -> (String, String) {
        const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
        const STRING_LEN: usize = 30;
        let mut rng = rand::thread_rng();

        let file_name: String = (0..STRING_LEN)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect();
        let module_name = file_name.clone();

        (file_name + ".py", module_name)
    }
    fn run_binary_func(x: &ArrayD<f64>, y: &ArrayD<f64>, py_code: &str) -> Value {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let xc = x.to_pyarray(py);
        let yc = y.to_pyarray(py);

        let (file_name, module_name) = generate_python_names();
        let comp_graph_py = PyModule::from_code(py, py_code, &file_name, &module_name)
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

    fn graph_from_run_call0_func(py_code: &str) -> Computation {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let (file_name, module_name) = generate_python_names();
        let comp_graph_py = PyModule::from_code(py, py_code, &file_name, &module_name)
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

        create_computation_graph_from_python(py_any)
    }
    fn run_executor(comp: &Computation) -> Value {
        let exec = EagerExecutor::new();
        let env = hashmap![];
        exec.run_computation(&comp, 12345, env).unwrap();
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
            output_type=standard_dialect.StringType(),
        )
    )

    return serialize_computation(comp)

    "#;

        let _ = run_executor(&graph_from_run_call0_func(&py_code));
    }
    #[test]
    fn test_deserialize_linear_regression() {
        let py_code = r#"
import numpy as np
from moose import edsl
from moose.computation import standard as standard_dialect
from moose.computation.utils import serialize_computation


def mse(y_pred, y_true):
    return edsl.mean(edsl.square(edsl.sub(y_pred, y_true)), axis=0)


def ss_res(y_pred, y_true):
    squared_residuals = edsl.square(edsl.sub(y_true, y_pred))
    return edsl.sum(squared_residuals, axis=0)


def ss_tot(y_true):
    y_mean = edsl.mean(y_true)
    squared_deviations = edsl.square(edsl.sub(y_true, y_mean))
    return edsl.sum(squared_deviations, axis=0)


def r_squared(ss_res, ss_tot):
    residuals_ratio = edsl.div(ss_res, ss_tot)
    return edsl.sub(edsl.constant(np.array([1], dtype=np.float64), dtype=edsl.float64), residuals_ratio)


def f():
    x_owner = edsl.host_placement(name="x-owner")
    model_owner = edsl.host_placement(name="model-owner")
    y_owner = edsl.host_placement(name="y-owner")
    replicated_plc = edsl.replicated_placement(
        players=[x_owner, y_owner, model_owner], name="replicated-plc"
    )


    @edsl.computation
    def my_comp():

        with x_owner:
            X = edsl.atleast_2d(
                edsl.load("x_uri", dtype=edsl.float64),to_column_vector=True
            )
            bias_shape = edsl.slice(edsl.shape(X), begin=0, end=1)
            bias = edsl.ones(bias_shape, dtype=edsl.float64)
            reshaped_bias = edsl.expand_dims(bias, 1)
            X_b = edsl.concatenate([reshaped_bias, X], axis=1)
            A = edsl.inverse(edsl.dot(edsl.transpose(X_b), X_b))
            B = edsl.dot(A, edsl.transpose(X_b))

        with y_owner:
            y_true = edsl.atleast_2d(
                edsl.load("y_uri", dtype=edsl.float64), to_column_vector=True
            )
            totals_ss = ss_tot(y_true)

        with replicated_plc:
            w = edsl.dot(B, y_true)
            y_pred = edsl.dot(X_b, w)
            mse_result = mse(y_pred, y_true)
            residuals_ss = ss_res(y_pred, y_true)

        with model_owner:
            rsquared_result = r_squared(residuals_ss, totals_ss)

        with model_owner:
            res = (
                edsl.save("regression_weights", w),
                edsl.save("mse_result", mse_result),
                edsl.save("rsquared_result", rsquared_result),
            )

        return res

    concrete_comp = edsl.trace(my_comp)
    return serialize_computation(concrete_comp)

"#;

        let comp = graph_from_run_call0_func(&py_code);
        let x = Value::from(Float64Tensor::from(
            array![
                [-0.76943992],
                [0.32067753],
                [-0.61509169],
                [0.11511809],
                [1.49598442],
                [0.37012138],
                [-0.49693762],
                [0.96914636],
                [0.19892362],
                [-0.98655745]
            ]
            .into_dimensionality::<IxDyn>()
            .unwrap(),
        ));

        let y = Value::from(Float64Tensor::from(
            array![
                7.69168025,
                10.9620326,
                8.15472493,
                10.34535427,
                14.48795325,
                11.11036415,
                8.50918715,
                12.90743909,
                10.59677087,
                7.04032766
            ]
            .into_dimensionality::<IxDyn>()
            .unwrap(),
        ));

        let mut storage_inputs: HashMap<String, Value> = HashMap::new();
        storage_inputs.insert("x_uri".to_string(), x);
        storage_inputs.insert("y_uri".to_string(), y);

        let storage: Rc<dyn SyncStorage> = Rc::new(LocalSyncStorage::from_hashmap(storage_inputs));
        let exec = EagerExecutor::new_from_storage(&storage);
        let env = hashmap![];
        exec.run_computation(&comp, 12345, env).unwrap();

        let res = array![[9.9999996], [2.999999]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let diff = Float64Tensor::try_from(storage.load("regression_weights".to_string()).unwrap())
            .unwrap();

        assert!(diff.0.abs_diff_eq(&res, 0.000001));
    }
}
