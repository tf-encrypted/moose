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
    std_IndexAxisOperation(PyIndexAxisOperation),
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
    std_AesKeyType,
    std_AesTensorType { dtype: PyDType },
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
struct PyIndexAxisOperation {
    name: String,
    inputs: Inputs,
    placement_name: String,
    axis: usize,
    index: usize,
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
        PyValueType::std_StringType => Ok(Ty::HostString),
        PyValueType::std_TensorType { dtype } => match dtype {
            PyDType::float32 => Ok(Ty::Tensor(TensorDType::Float32)),
            PyDType::float64 => Ok(Ty::Tensor(TensorDType::Float64)),
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
                    std_IndexAxisOperation(op) => Ok(Operation {
                        kind: IndexAxisOp {
                            sig: Signature::unary(
                                map_type(&op.output_type)?,
                                map_type(&op.output_type)?,
                            ),
                            axis: op.axis.clone(),
                            index: op.index.clone(),
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
                    fixed_EncodeOperation(op) => Ok(Operation {
                        kind: FixedpointEncodeOp {
                            sig: Signature::unary(Ty::Unknown, map_type(&op.output_type)?),
                            fractional_precision: op.precision,
                            integral_precision: 8, // just because
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
                            fractional_precision: op.precision,
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

#[cfg(test)]
mod tests {
    use super::PyComputation;
    use maplit::hashmap;
    use moose::compilation::{compile_passes, Pass};
    use moose::computation::*;
    use moose::host::HostFloat64Tensor;
    use moose::kernels::{SyncSession, TestSyncExecutor};
    use moose::storage::{LocalSyncStorage, SyncStorage};
    use ndarray::prelude::*;
    use numpy::ToPyArray;
    use pyo3::prelude::*;
    use rand::Rng;
    use std::collections::{HashMap, HashSet};
    use std::convert::TryFrom;
    use std::convert::TryInto;
    use std::rc::Rc;

    fn create_computation_graph_from_python(py_any: &PyAny) -> Computation {
        let buf: Vec<u8> = py_any.extract().unwrap();
        let comp: PyComputation = rmp_serde::from_read_ref(&buf).unwrap();

        let rust_comp: Computation = comp.try_into().unwrap();
        compile_passes(
            &rust_comp,
            &[Pass::Typing, Pass::DeprecatedLogical, Pass::Toposort],
        )
        .unwrap()
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
        pyo3::prepare_freethreaded_python();
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

        let computation = create_computation_graph_from_python(py_any);
        let all_roles = computation
            .operations
            .iter()
            .flat_map(|op| -> Box<dyn Iterator<Item = &Role>> {
                match &op.placement {
                    // TODO(Morten) box seems too complicated..?
                    Placement::Host(plc) => Box::new(std::iter::once(&plc.owner)),
                    Placement::Replicated(plc) => Box::new(plc.owners.iter()),
                    Placement::Additive(plc) => Box::new(plc.owners.iter()),
                }
            })
            .collect::<HashSet<_>>();
        let executor = TestSyncExecutor::default();
        let session = SyncSession::from_roles(all_roles.iter().cloned());
        let outputs = executor.run_computation(&computation, &session).unwrap();
        outputs["result"].clone()
    }
    fn run_unary_func(x: &ArrayD<f64>, py_code: &str) -> Value {
        pyo3::prepare_freethreaded_python();
        let gil = Python::acquire_gil();
        let py = gil.python();

        let xc = x.to_pyarray(py);

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
            .call1((xc,))
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap();

        let computation = create_computation_graph_from_python(py_any);
        let all_roles = computation
            .operations
            .iter()
            .flat_map(|op| -> Box<dyn Iterator<Item = &Role>> {
                match &op.placement {
                    // TODO(Morten) box seems too complicated..?
                    Placement::Host(plc) => Box::new(std::iter::once(&plc.owner)),
                    Placement::Replicated(plc) => Box::new(plc.owners.iter()),
                    Placement::Additive(plc) => Box::new(plc.owners.iter()),
                }
            })
            .collect::<HashSet<_>>();
        let executor = TestSyncExecutor::default();
        let session = SyncSession::from_roles(all_roles.iter().cloned());
        let outputs = executor.run_computation(&computation, &session).unwrap();
        outputs["result"].clone()
    }

    fn graph_from_run_call0_func(py_code: &str) -> Computation {
        pyo3::prepare_freethreaded_python();
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

    #[test]
    fn test_deserialize_host_op() {
        let py_code = r#"
import numpy as np
from pymoose.computation import standard as standard_dialect
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.utils import serialize_computation
from pymoose.computation.standard import TensorType
from pymoose.computation.standard import TensorConstant
from pymoose.computation.standard import UnitType
from pymoose.computation import dtypes
from pymoose.deprecated.computation import ring as ring_dialect
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
                output_type=TensorType(dtype=dtypes.float64),
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

        let unwrapped_result = match result {
            Value::HostFloat64Tensor(t) => t,
            _ => panic!("Unexpected result type. Expected HostFloat64Tensor"),
        };
        assert_eq!(unwrapped_result.0, x1 * y1);

        let x2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let add_code = py_code.replace("SPECIAL_OP", "AddOperation");
        let result = run_binary_func(&x2, &y2, &add_code);

        let unwrapped_result = match result {
            Value::HostFloat64Tensor(t) => t,
            _ => panic!("Unexpected result type. Expected HostFloat64Tensor"),
        };
        assert_eq!(unwrapped_result.0, x2 + y2);

        let x3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let sub_code = py_code.replace("SPECIAL_OP", "SubOperation");
        let result = run_binary_func(&x3, &y3, &sub_code);

        let unwrapped_result = match result {
            Value::HostFloat64Tensor(t) => t,
            _ => panic!("Unexpected result type. Expected HostFloat64Tensor"),
        };
        assert_eq!(unwrapped_result.0, x3 - y3);

        let x4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let dot_code = py_code.replace("SPECIAL_OP", "DotOperation");
        let result = run_binary_func(&x4, &y4, &dot_code);

        let unwrapped_result = match result {
            Value::HostFloat64Tensor(t) => t,
            _ => panic!("Unexpected result type. Expected HostFloat64Tensor"),
        };
        assert_eq!(
            unwrapped_result.0,
            HostFloat64Tensor::from(x4)
                .dot(HostFloat64Tensor::from(y4))
                .0
        );
    }

    #[test]
    fn test_deserialize_replicated_op() {
        let py_code = r#"
import numpy as np

from pymoose.computation import dtypes
from pymoose.computation import standard as standard_dialect
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.replicated import ReplicatedPlacement
from pymoose.computation.standard import TensorType
from pymoose.computation.standard import UnitType
from pymoose.computation.utils import serialize_computation
from pymoose.computation import dtypes
from pymoose.deprecated.compiler.compiler import Compiler
from pymoose.deprecated.computation import fixedpoint as fixedpoint_ops
from pymoose.deprecated.computation import ring as ring_dialect
from pymoose.deprecated.computation.ring import RingTensorType

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

    fp_dtype = dtypes.fixed(8, 27)


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
        fixedpoint_ops.EncodeOperation(
            name="encode_alice",
            inputs={"value": "alice_input"},
            placement_name="alice",
            output_type=fixedpoint_ops.EncodedTensorType(
                dtype=fp_dtype, precision=fp_dtype.fractional_precision
            ),
            precision=fp_dtype.fractional_precision,
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
        fixedpoint_ops.EncodeOperation(
            name="encode_bob",
            inputs={"value": "bob_input"},
            placement_name="bob",
            output_type=fixedpoint_ops.EncodedTensorType(
                dtype=fp_dtype, precision=fp_dtype.fractional_precision
            ),
            precision=fp_dtype.fractional_precision,
        )
    )

    comp.add_operation(
        standard_dialect.SPECIAL_OP(
            name="rep_add",
            placement_name=rep.name,
            inputs={"lhs": "encode_alice", "rhs": "encode_bob"},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    comp.add_operation(
        fixedpoint_ops.DecodeOperation(
            name="decode_carole",
            inputs={"value": "rep_add"},
            placement_name=carole.name,
            output_type=TensorType(dtype=dtypes.float64),
            precision=fp_dtype.fractional_precision,
        )
    )

    comp.add_operation(
        standard_dialect.OutputOperation(
            name="result", placement_name=carole.name, inputs={"value": "decode_carole"},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    compiler = Compiler(ring=128)
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

        let unwrapped_result = match result {
            Value::HostFloat64Tensor(t) => t,
            _ => panic!("Unexpected result type. Expected HostFloat64Tensor"),
        };
        assert_eq!(unwrapped_result.0, x1 * y1);

        let x2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let add_code = py_code.replace("SPECIAL_OP", "AddOperation");
        let result = run_binary_func(&x2, &y2, &add_code);

        let unwrapped_result = match result {
            Value::HostFloat64Tensor(t) => t,
            _ => panic!("Unexpected result type. Expected HostFloat64Tensor"),
        };
        assert_eq!(unwrapped_result.0, x2 + y2);

        let x3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let sub_code = py_code.replace("SPECIAL_OP", "SubOperation");
        let result = run_binary_func(&x3, &y3, &sub_code);

        let unwrapped_result = match result {
            Value::HostFloat64Tensor(t) => t,
            _ => panic!("Unexpected result type. Expected HostFloat64Tensor"),
        };
        assert_eq!(unwrapped_result.0, x3 - y3);

        let x4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let dot_code = py_code.replace("SPECIAL_OP", "DotOperation");
        let result = run_binary_func(&x4, &y4, &dot_code);

        let unwrapped_result = match result {
            Value::HostFloat64Tensor(t) => t,
            _ => panic!("Unexpected result type. Expected HostFloat64Tensor"),
        };
        assert_eq!(
            unwrapped_result.0,
            HostFloat64Tensor::from(x4)
                .dot(HostFloat64Tensor::from(y4))
                .0
        );
    }
    #[test]
    fn test_constant() {
        let py_code = r#"
import numpy as np
from pymoose.computation import standard as standard_dialect
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.utils import serialize_computation
from pymoose.computation import dtypes
from pymoose.deprecated.computation import ring as ring_dialect

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

        let computation = graph_from_run_call0_func(py_code);
        let executor = TestSyncExecutor::default();
        let session = SyncSession::default();
        let _ = executor.run_computation(&computation, &session).unwrap();
    }
    #[test]
    fn test_deserialize_linear_regression() {
        let py_code = r#"
import numpy as np
from pymoose import edsl
from pymoose.computation import standard as standard_dialect
from pymoose.computation.utils import serialize_computation


FIXED = edsl.fixed(8, 27)

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
            X_b = edsl.cast(X_b, dtype=FIXED)
            B = edsl.cast(B, dtype=FIXED)


        with y_owner:
            y_true = edsl.atleast_2d(
                edsl.load("y_uri", dtype=edsl.float64), to_column_vector=True
            )
            totals_ss = ss_tot(y_true)
            y_true = edsl.cast(y_true, dtype=FIXED)


        with replicated_plc:
            w = edsl.dot(B, y_true)
            y_pred = edsl.dot(X_b, w)
            mse_result = mse(y_pred, y_true)
            residuals_ss = ss_res(y_pred, y_true)

        with model_owner:
            residuals_ss = edsl.cast(residuals_ss, dtype=edsl.float64)
            rsquared_result = r_squared(residuals_ss, totals_ss)

        with model_owner:
            w = edsl.cast(w, dtype=edsl.float64)
            mse_result = edsl.cast(mse_result, dtype=edsl.float64)
            res = (
                edsl.save("regression_weights", w),
                edsl.save("mse_result", mse_result),
                edsl.save("rsquared_result", rsquared_result),
            )

        return res

    concrete_comp = edsl.trace_and_compile(my_comp, ring=128)
    return serialize_computation(concrete_comp)

"#;

        let comp = graph_from_run_call0_func(py_code);
        let x = Value::from(HostFloat64Tensor::from(
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

        let y = Value::from(HostFloat64Tensor::from(
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
        let executor = TestSyncExecutor::default();
        let own_identity = moose::execution::Identity::from("tester");
        let role_assignments = hashmap!(
            Role::from("x-owner") => own_identity.clone(),
            Role::from("y-owner") => own_identity.clone(),
            Role::from("model-owner") => own_identity,
        );
        let session = SyncSession::from_storage(
            SessionId::try_from("foobar").unwrap(),
            hashmap!(),
            role_assignments,
            storage.clone(),
        );
        let _ = executor.run_computation(&comp, &session).unwrap();

        let res = array![[9.9999996], [2.999999]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let diff = HostFloat64Tensor::try_from(
            storage
                .load(
                    "regression_weights",
                    &SessionId::try_from("foobar").unwrap(),
                    None,
                    "",
                )
                .unwrap(),
        )
        .unwrap();

        assert!(diff.0.abs_diff_eq(&res, 0.000001));
    }

    #[test]
    fn test_deserialize_replicated_abs() {
        let py_code = r#"
import numpy as np

from pymoose.computation import dtypes
from pymoose.computation import standard as standard_dialect
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.replicated import ReplicatedPlacement
from pymoose.computation.standard import TensorType
from pymoose.computation.standard import UnitType
from pymoose.computation.utils import serialize_computation
from pymoose.deprecated.compiler.compiler import Compiler
from pymoose.deprecated.computation import fixedpoint as fixedpoint_ops
from pymoose.deprecated.computation import ring as ring_dialect
from pymoose.deprecated.computation.ring import RingTensorType

alice = HostPlacement(name="alice")
bob = HostPlacement(name="bob")
carole = HostPlacement(name="carole")
rep = ReplicatedPlacement(name="rep", player_names=["alice", "bob", "carole"])


def f(arg1):
    comp = Computation(operations={}, placements={})
    comp.add_placement(alice)
    comp.add_placement(bob)
    comp.add_placement(carole)
    comp.add_placement(rep)

    x = np.array(arg1, dtype=np.float64)

    fp_dtype = dtypes.fixed(8, 27)

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
        fixedpoint_ops.EncodeOperation(
            name="encode_alice",
            inputs={"value": "alice_input"},
            placement_name="alice",
            output_type=fixedpoint_ops.EncodedTensorType(
                dtype=fp_dtype, precision=fp_dtype.fractional_precision
            ),
            precision=fp_dtype.fractional_precision,
        )
    )

    comp.add_operation(
        standard_dialect.AbsOperation(
            name="rep_abs",
            placement_name=rep.name,
            inputs={"x": "encode_alice"},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    comp.add_operation(
        fixedpoint_ops.DecodeOperation(
            name="decode_carole",
            inputs={"value": "rep_abs"},
            placement_name=carole.name,
            output_type=TensorType(dtype=dtypes.float64),
            precision=fp_dtype.fractional_precision,
        )
    )

    comp.add_operation(
        standard_dialect.OutputOperation(
            name="result", placement_name=carole.name, inputs={"value": "decode_carole"},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    compiler = Compiler(ring=128)
    comp = compiler.run_passes(comp)

    return serialize_computation(comp)

"#;
        let x1 = array![[-1.0, -2.0], [-3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let result = run_unary_func(&x1, py_code);

        let unwrapped_result = match result {
            Value::HostFloat64Tensor(t) => t,
            _ => panic!("Unexpected result type. Expected HostFloat64Tensor"),
        };
        assert_eq!(unwrapped_result.0, x1.mapv(f64::abs));
    }
}
