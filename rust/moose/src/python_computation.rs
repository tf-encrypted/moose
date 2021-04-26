use crate::{
    computation::*, prim, standard::Float32Tensor, standard::Float64Tensor, standard::Shape,
};
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
    std_InputOperation(PyInputOperation),
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
    std_TensorType(PyDType),
    std_UnitType,
    std_UnknownType,
    ring_RingTensorType,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "dtype")]
#[allow(non_camel_case_types)]
enum PyDType {
    float32,
    float64,
    int32,
    int64,
    uint32,
    uint64,
    fixed14_23,
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

fn map_constant_value(constant_value: &PyConstant) -> anyhow::Result<Value> {
    match constant_value {
        PyConstant::std_ShapeConstant { value } => {
            Ok(Shape(value.iter().map(|i| *i as usize).collect()).into())
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
        PyValueType::std_TensorType(py_dtype) => match py_dtype {
            PyDType::float32 => Ty::Float32TensorTy,
            PyDType::float64 => Ty::Float64TensorTy,
            PyDType::int32 => Ty::Int32TensorTy,
            PyDType::int64 => Ty::Int64TensorTy,
            PyDType::uint32 => Ty::Uint32TensorTy,
            PyDType::uint64 => Ty::Uint64TensorTy,
            PyDType::fixed14_23 => unimplemented!(),
        },
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
                use crate::computation::Operator::*;
                use anyhow::Context;
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
                            nonce: prim::Nonce(op.nonce.clone()),
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
                            receiver: Role::from(&op.receiver),
                        }),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_ReceiveOperation(op) => Ok(Operation {
                        kind: Receive(ReceiveOp {
                            rendezvous_key: op.rendezvous_key.clone(),
                            sender: Role::from(&op.sender),
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
                    std_InputOperation(op) => Ok(Operation {
                        kind: Input(InputOp {
                            arg_name: op.name.clone(),
                            ty: map_type(&op.output_type),
                        }),
                        name: op.name.clone(),
                        inputs: Vec::new(),
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
