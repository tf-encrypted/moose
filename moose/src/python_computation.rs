//! Parser for computations defined in Python

use crate::computation::*;
use crate::host::{HostFloat32Tensor, HostFloat64Tensor, RawShape, SliceInfo, SliceInfoElem};
use ndarray::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};

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
    bit_BitExtractOperation(PyBitExtractOperation),
    bit_BitSampleOperation(PyBitSampleOperation),
    bit_BitFillTensorOperation(PyBitFillOperation),
    bit_BitXorOperation(PyBitXorOperation),
    bit_BitAndOperation(PyBitAndOperation),
    bit_RingInjectOperation(PyRingInjectOperation),
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
    std_SqrtOperation(PySqrtOperation),
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
    std_CastOperation(PyCastOperation),
    fixed_EncodeOperation(PyFixedEncodeOperation),
    fixed_DecodeOperation(PyFixedDecodeOperation),
    fixed_AddOperation(PyFixedAddOperation),
    fixed_SubOperation(PyFixedSubOperation),
    fixed_MulOperation(PyFixedMulOperation),
    fixed_DotOperation(PyFixedDotOperation),
    fixed_TruncPrOperation(PyFixedTruncPrOperation),
    fixed_MeanOperation(PyFixedMeanOperation),
    fixed_SumOperation(PyFixedSumOperation),
    fixed_RingEncodeOperation(PyFixedRingEncodeOperation),
    fixed_RingDecodeOperation(PyFixedRingDecodeOperation),
    fixed_RingMeanOperation(PyFixedRingMeanOperation),
    rep_SetupOperation(PyRepSetupOperation),
    rep_ShareOperation(PyRepShareOperation),
    rep_DotOperation(PyRepDotOperation),
    rep_TruncPrOperation(PyRepTruncPrOperation),
    rep_SubOperation(PyRepSubOperation),
    rep_MulOperation(PyRepMulOperation),
    rep_MeanOperation(PyRepMeanOperation),
    rep_SumOperation(PyRepSumOperation),
    rep_RevealOperation(PyRepRevealOperation),
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
    std_TensorType { dtype: PyDType },
    std_UnitType,
    std_UnknownType,
    ring_RingTensorType,
    bit_BitTensorType,
    rep_ReplicatedSetupType,
    rep_ReplicatedRingTensorType,
    fixed_EncodedTensorType,
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
    fixed8_27,
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
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyRingSubOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyRingMulOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyRingDotOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
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
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyRingSumOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    axis: Option<u32>,
    output_type: PyValueType,
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
    value: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyBitFillOperation {
    name: String,
    value: u8,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingShlOperation {
    name: String,
    amount: u64,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyRingShrOperation {
    name: String,
    amount: u64,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyBitExtractOperation {
    name: String,
    bit_idx: u64,
    inputs: Inputs,
    placement_name: String,
    ring_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyBitSampleOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyBitXorOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
}
#[derive(Deserialize, Debug)]
struct PyBitAndOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
}

#[derive(Deserialize, Debug)]
struct PyRingInjectOperation {
    name: String,
    inputs: Inputs,
    bit_idx: u64,
    placement_name: String,
    output_type: PyValueType,
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
    axis: Vec<u32>,
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
struct PyFixedEncodeOperation {
    name: String,
    precision: u32,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyFixedDecodeOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    precision: u32,
}

#[derive(Deserialize, Debug)]
struct PyFixedAddOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyFixedSubOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyFixedMulOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyFixedDotOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyFixedTruncPrOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    precision: u32,
}

#[derive(Deserialize, Debug)]
struct PyFixedMeanOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    axis: Option<u32>,
    precision: u32,
    scaling_base: u64,
    scaling_exp: u32,
}

#[derive(Deserialize, Debug)]
struct PyFixedSumOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    axis: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct PyFixedRingEncodeOperation {
    name: String,
    scaling_base: u64,
    scaling_exp: u32,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyFixedRingMeanOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    axis: Option<u32>,
    scaling_base: u64,
    scaling_exp: u32,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyFixedRingDecodeOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    input_type: PyValueType,
    scaling_base: u64,
    scaling_exp: u32,
}

#[derive(Deserialize, Debug)]
struct PyRepSetupOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyRepShareOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyRepDotOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyRepTruncPrOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    precision: u32,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyRepSubOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyRepMulOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
}

#[derive(Deserialize, Debug)]
struct PyRepMeanOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    axis: Option<u32>,
    precision: u64,
}

#[derive(Deserialize, Debug)]
struct PyRepSumOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    axis: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct PyRepRevealOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    output_type: PyValueType,
    recipient_name: String,
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
        PyValueType::prim_PRFKeyType => Ok(Ty::PrfKey),
        PyValueType::prim_SeedType => Ok(Ty::Seed),
        PyValueType::std_ShapeType => Ok(Ty::HostShape),
        PyValueType::std_UnitType => Ok(Ty::Unit),
        PyValueType::std_StringType => Ok(Ty::String),
        PyValueType::std_TensorType { dtype } => match dtype {
            PyDType::float32 => Ok(Ty::Tensor(InnerTy::Float32)),
            PyDType::float64 => Ok(Ty::Tensor(InnerTy::Float64)),
            // PyDType::int32 => Ok(Ty::HostInt32Tensor),
            // PyDType::int64 => Ok(Ty::HostInt64Tensor),
            // PyDType::uint32 => Ok(Ty::HostUint32Tensor),
            // PyDType::uint64 => Ok(Ty::HostUint64Tensor),
            // PyDType::fixed14_23 => Err(anyhow::anyhow!("unimplemented dtype 'fixed14_23'")),
            PyDType::fixed8_27 => Ok(Ty::Tensor(InnerTy::Fixed128)), // TODO: store the precision (27)
            _ => Err(anyhow::anyhow!("unimplemented dtype '{:?}'", py_type)),
        },
        PyValueType::std_UnknownType => Ok(Ty::Unknown),
        PyValueType::std_BytesType => Err(anyhow::anyhow!("unimplemented type 'bytes'")),
        PyValueType::ring_RingTensorType => Ok(Ty::HostRing128Tensor),
        PyValueType::bit_BitTensorType => Ok(Ty::HostBitTensor),
        PyValueType::rep_ReplicatedSetupType => Ok(Ty::ReplicatedSetup),
        PyValueType::rep_ReplicatedRingTensorType => Ok(Ty::ReplicatedRing128Tensor),
        PyValueType::fixed_EncodedTensorType => Ok(Ty::Fixed128Tensor),
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
                use std::str::FromStr;
                use PyOperation::*;
                match op {
                    prim_SampleKeyOperation(op) => Ok(Operation {
                        kind: PrimPrfKeyGenOp {
                            sig: Signature::nullary(Ty::PrfKey),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    prim_DeriveSeedOperation(op) => Ok(Operation {
                        kind: PrimDeriveSeedOp {
                            sig: Signature::unary(Ty::PrfKey, Ty::Seed),
                            sync_key: op.nonce.clone().try_into()?,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["key"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingAddOperation(op) => Ok(Operation {
                        kind: RingAddOp {
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingSubOperation(op) => Ok(Operation {
                        kind: RingSubOp {
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingMulOperation(op) => Ok(Operation {
                        kind: RingMulOp {
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingDotOperation(op) => Ok(Operation {
                        kind: RingDotOp {
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingShapeOperation(op) => Ok(Operation {
                        kind: ShapeOp {
                            sig: Signature::unary(Ty::HostRing128Tensor, Ty::HostShape),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["tensor"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingSampleOperation(op) => Ok(Operation {
                        // NOTE(Morten) the old Python op was RingSampleOp, ie without Seeded
                        kind: RingSampleSeededOp {
                            sig: Signature::binary(
                                Ty::HostShape,
                                Ty::Seed,
                                map_type(&op.output_type)?,
                            ),
                            max_value: op.max_value,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["shape", "seed"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingSumOperation(op) => Ok(Operation {
                        kind: RingSumOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            axis: op.axis,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingMeanOperation(op) => {
                        Err(anyhow::anyhow!("unsupported operation: {:?}", op))
                    }
                    ring_FillTensorOperation(op) => {
                        let ty = map_type(&op.output_type)?;
                        let value = match ty {
                            Ty::HostRing64Tensor => Constant::Ring64(u64::from_str(&op.value)?),
                            Ty::HostRing128Tensor => Constant::Ring128(u128::from_str(&op.value)?),
                            _ => {
                                return Err(anyhow::anyhow!(
                                    "unsupported return type for ring fill: {:?}",
                                    ty
                                ));
                            }
                        };
                        Ok(Operation {
                            kind: RingFillOp {
                                sig: Signature::unary(Ty::HostShape, ty),
                                value,
                            }
                            .into(),
                            name: op.name.clone(),
                            inputs: map_inputs(&op.inputs, &["shape"])
                                .with_context(|| format!("Failed at op {:?}", op))?,
                            placement: map_placement(&placements, &op.placement_name)?,
                        })
                    }
                    ring_RingShlOperation(op) => Ok(Operation {
                        kind: RingShlOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            amount: op.amount as usize,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    ring_RingShrOperation(op) => Ok(Operation {
                        kind: RingShrOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            amount: op.amount as usize,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    bit_BitExtractOperation(op) => Ok(Operation {
                        kind: BitExtractOp {
                            sig: Signature::unary(map_type(&op.ring_type)?, Ty::HostBitTensor),
                            bit_idx: op.bit_idx as usize,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["tensor"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    bit_BitSampleOperation(op) => Ok(Operation {
                        // NOTE(Morten) mapping from BitSample in Python to BitSampleSeeded in Rust
                        kind: BitSampleSeededOp {
                            sig: Signature::binary(Ty::HostShape, Ty::Seed, Ty::HostBitTensor),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["shape", "seed"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    bit_BitFillTensorOperation(op) => Ok(Operation {
                        kind: BitFillOp {
                            sig: Signature::unary(Ty::HostShape, Ty::HostBitTensor),
                            value: Constant::Ring64(u64::from(op.value)),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["shape"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    bit_BitXorOperation(op) => Ok(Operation {
                        kind: BitXorOp {
                            sig: Signature::binary(
                                Ty::HostBitTensor,
                                Ty::HostBitTensor,
                                Ty::HostBitTensor,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    bit_BitAndOperation(op) => Ok(Operation {
                        kind: BitAndOp {
                            sig: Signature::binary(
                                Ty::HostBitTensor,
                                Ty::HostBitTensor,
                                Ty::HostBitTensor,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    bit_RingInjectOperation(op) => Ok(Operation {
                        kind: RingInjectOp {
                            sig: Signature::unary(Ty::HostBitTensor, map_type(&op.output_type)?),
                            bit_idx: op.bit_idx as usize,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["tensor"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
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
                    std_ShapeOperation(op) => Ok(Operation {
                        // TODO (lvorona): We can actually use InnerTy::Unknown and let the type inference figure the type out.
                        kind: ShapeOp {
                            sig: Signature::unary(Ty::Tensor(InnerTy::Float64), Ty::HostShape),
                        }
                        .into(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        name: op.name.clone(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SliceOperation(op) => Ok(Operation {
                        kind: SliceOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
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
                    std_ConcatenateOperation(op) => {
                        let mut inputs: Vec<(&String, &String)> = op.inputs.iter().collect();
                        inputs.sort_by_key(|x| x.0);
                        let sorted_input_names: Vec<String> =
                            inputs.into_iter().map(|(_k, v)| v.clone()).collect();
                        Ok(Operation {
                            kind: ConcatOp {
                                // assume input type is the same as output type
                                // TODO: Support variadic signature
                                sig: Signature::binary(
                                    map_type(&op.output_type)?,
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
                    std_SendOperation(op) => Ok(Operation {
                        kind: SendOp {
                            sig: Signature::unary(Ty::Unknown, Ty::Unit),
                            rendezvous_key: op.rendezvous_key.clone().try_into()?,
                            receiver: Role::from(&op.receiver),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_ReceiveOperation(op) => Ok(Operation {
                        kind: ReceiveOp {
                            sig: Signature::nullary(map_type(&op.output_type)?),
                            rendezvous_key: op.rendezvous_key.clone().try_into()?,
                            sender: Role::from(&op.sender),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    std_SerializeOperation(op) => Ok(Operation {
                        kind: IdentityOp {
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
                    std_DeserializeOperation(op) => Ok(Operation {
                        kind: IdentityOp {
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
                            sig: Signature::binary(Ty::String, Ty::Unknown, Ty::Unit),
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
                                Ty::String,
                                Ty::String,
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
                    fixed_EncodeOperation(op) => Ok(Operation {
                        kind: FixedpointEncodeOp {
                            sig: Signature::unary(Ty::Unknown, map_type(&op.output_type)?),
                            precision: op.precision,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_DecodeOperation(op) => Ok(Operation {
                        kind: FixedpointDecodeOp {
                            sig: Signature::unary(
                                Ty::Fixed128Tensor, // TODO: Derive from the output type
                                map_type(&op.output_type)?,
                            ),
                            precision: op.precision,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_AddOperation(op) => Ok(Operation {
                        kind: FixedpointAddOp {
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_SubOperation(op) => Ok(Operation {
                        kind: FixedpointSubOp {
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_MulOperation(op) => Ok(Operation {
                        kind: FixedpointMulOp {
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_DotOperation(op) => Ok(Operation {
                        kind: FixedpointDotOp {
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_TruncPrOperation(op) => Ok(Operation {
                        kind: FixedpointTruncPrOp {
                            sig: Signature::unary(
                                Ty::Fixed128Tensor, // TODO: Derive from the output type
                                map_type(&op.output_type)?,
                            ),
                            precision: op.precision,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_MeanOperation(op) => Ok(Operation {
                        kind: FixedpointMeanOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            axis: op.axis,
                            // TODO(Morten) fix on Python side as needed due to removing the two below
                            // scaling_base: op.scaling_base,
                            // scaling_exp: op.scaling_exp,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_SumOperation(op) => Ok(Operation {
                        kind: FixedpointSumOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            axis: op.axis,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_RingEncodeOperation(op) => Ok(Operation {
                        kind: RingFixedpointEncodeOp {
                            sig: Signature::unary(Ty::Unknown, map_type(&op.output_type)?),
                            scaling_base: op.scaling_base,
                            scaling_exp: op.scaling_exp,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_RingDecodeOperation(op) => Ok(Operation {
                        kind: RingFixedpointDecodeOp {
                            sig: Signature::unary(
                                map_type(&op.input_type)?,
                                map_type(&op.output_type)?,
                            ),
                            scaling_base: op.scaling_base,
                            scaling_exp: op.scaling_exp,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    fixed_RingMeanOperation(op) => Ok(Operation {
                        kind: RingFixedpointMeanOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            axis: op.axis,
                            scaling_base: op.scaling_base,
                            scaling_exp: op.scaling_exp,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    rep_TruncPrOperation(op) => Ok(Operation {
                        kind: RepTruncPrOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            amount: op.precision,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    rep_RevealOperation(op) => Ok(Operation {
                        kind: RepRevealOp {
                            sig: Signature::unary(
                                Ty::ReplicatedRing128Tensor, // TODO: deduct from the output type
                                Ty::HostRing128Tensor,
                                // map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.recipient_name)?,
                    }),
                    rep_ShareOperation(op) => Ok(Operation {
                        kind: RepShareOp {
                            sig: Signature::binary(
                                Ty::ReplicatedSetup,
                                Ty::HostRing128Tensor, // TODO: should actually deduct from the output type
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["setup", "value"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    rep_MulOperation(op) => Ok(Operation {
                        kind: RepMulOp {
                            sig: Signature::ternary(
                                Ty::ReplicatedSetup,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["setup", "lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    rep_SubOperation(op) => Ok(Operation {
                        kind: RepSubOp {
                            sig: Signature::binary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    rep_DotOperation(op) => Ok(Operation {
                        kind: RepDotOp {
                            sig: Signature::ternary(
                                Ty::ReplicatedSetup,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["setup", "lhs", "rhs"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    rep_MeanOperation(op) => Ok(Operation {
                        kind: RepFixedpointMeanOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            axis: op.axis,
                            scaling_base: 2,
                            scaling_exp: op.precision as u32, // TODO avoid cast
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    rep_SumOperation(op) => Ok(Operation {
                        kind: RepSumOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            axis: op.axis,
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: map_inputs(&op.inputs, &["x"])
                            .with_context(|| format!("Failed at op {:?}", op))?,
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                    rep_SetupOperation(op) => Ok(Operation {
                        kind: RepSetupOp {
                            sig: Signature::nullary(map_type(&op.output_type)?),
                        }
                        .into(),
                        name: op.name.clone(),
                        inputs: Vec::new(),
                        placement: map_placement(&placements, &op.placement_name)?,
                    }),
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Computation { operations })
    }
}
