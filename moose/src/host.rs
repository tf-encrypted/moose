use crate::computation::{
    BitAndOp, BitExtractOp, BitFillOp, BitSampleOp, BitSampleSeededOp, BitXorOp, CanonicalType,
    Constant, HostAddOp, HostBitDecOp, HostConcatOp, HostDiagOp, HostDivOp, HostDotOp,
    HostExpandDimsOp, HostIndexAxisOp, HostInverseOp, HostMeanOp, HostMulOp, HostOnesOp,
    HostPlacement, HostReshapeOp, HostShlDimOp, HostSliceOp, HostSqrtOp, HostSqueezeOp, HostSubOp,
    HostSumOp, HostTransposeOp, KnownType, Placed, Placement, ReplicatedPlacement, RingAddOp,
    RingDotOp, RingFillOp, RingFixedpointMeanOp, RingInjectOp, RingMulOp, RingNegOp, RingSampleOp,
    RingSampleSeededOp, RingShlOp, RingShrOp, RingSubOp, RingSumOp, Role, ShapeOp, SymbolicType,
};
use crate::error::{Error, Result};
use crate::fixedpoint::Fixed128Tensor;
use crate::kernels::{
    PlacementAdd, PlacementAnd, PlacementBitDec, PlacementBitExtract, PlacementDiag, PlacementDot,
    PlacementFill, PlacementIndex, PlacementMean, PlacementMul, PlacementNeg, PlacementPlace,
    PlacementSample, PlacementSampleSeeded, PlacementSampleUniform, PlacementSampleUniformSeeded,
    PlacementShl, PlacementShlDim, PlacementShr, PlacementSlice, PlacementSub, PlacementSum,
    PlacementTruncPr, PlacementXor, RuntimeSession, Session, SyncSession, Tensor,
};
use crate::prim::{RawSeed, Seed};
use crate::prng::AesRng;
use crate::symbolic::{Symbolic, SymbolicHandle, SymbolicSession};
use macros::with_context;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use ndarray::Slice;
use ndarray_linalg::types::{Lapack, Scalar};
use ndarray_linalg::*;
use num_traits::Zero;
use num_traits::{Float, FromPrimitive};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::num::Wrapping;
use std::ops::{Add, Div, Mul, Sub}; // related to TODOs
use std::ops::{BitAnd, BitXor, Neg, Shl, Shr};

impl SymbolicType for String {
    type Type = Symbolic<String>;
}

impl Placed for String {
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        // TODO we need a wrapper for strings that contains placement info
        unimplemented!()
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawShape(pub Vec<usize>);

impl RawShape {
    pub fn extend_singletons(self, mut axis: Vec<usize>) -> Self {
        let ax = axis.pop();
        match ax {
            Some(ax) => {
                let (left, right) = self.0.split_at(ax);
                RawShape::extend_singletons(RawShape([left, right].join(&1usize)), axis)
            }
            None => self,
        }
    }

    pub fn slice(self, begin: usize, end: usize) -> Self {
        let slc = &self.0[begin..end];
        RawShape(slc.to_vec())
    }

    pub fn unsqueeze(mut self, axis: usize) -> Self {
        self.0.insert(axis, 1);
        self
    }

    pub fn squeeze(mut self, axis: Option<usize>) -> Self {
        match axis {
            Some(axis) => {
                let removed_axis = self.0.remove(axis);
                match removed_axis {
                    1 => self,
                    _ => panic!(
                        "The axis selected has a value of {:?}. Cannot select an axis to squeeze out
                        which has size not equal to one", removed_axis
                    ),
                }
            }
            None => RawShape(self.0.into_iter().filter(|x| *x != 1).collect::<Vec<_>>()),
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct HostShape(pub RawShape, pub HostPlacement);

impl SymbolicType for HostShape {
    type Type = Symbolic<HostShape>;
}

impl Placed for HostShape {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl PlacementPlace<SyncSession, HostShape> for HostPlacement {
    fn place(&self, _sess: &SyncSession, shape: HostShape) -> HostShape {
        match shape.placement() {
            Ok(place) if self == &place => shape,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                HostShape(shape.0, self.clone())
            }
        }
    }
}

impl PlacementPlace<SymbolicSession, Symbolic<HostShape>> for HostPlacement {
    fn place(&self, _sess: &SymbolicSession, x: Symbolic<HostShape>) -> Symbolic<HostShape> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                match x {
                    Symbolic::Concrete(shape) => {
                        // TODO insert Place ops?
                        Symbolic::Concrete(HostShape(shape.0, self.clone()))
                    }
                    Symbolic::Symbolic(SymbolicHandle { op, plc: _ }) => {
                        // TODO insert `Place` ops here?
                        Symbolic::Symbolic(SymbolicHandle {
                            op,
                            plc: self.clone(),
                        })
                    }
                }
            }
        }
    }
}

/// One slice for slicing op
#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct SliceInfoElem {
    /// start index; negative are counted from the back of the axis
    pub start: isize,
    /// end index; negative are counted from the back of the axis; when not present
    /// the default is the full length of the axis.
    pub end: Option<isize>,
    /// step size in elements; the default is 1, for every element.
    pub step: Option<isize>,
}

/// An ndarray slice needs a SliceInfoElem for each shape dimension
#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct SliceInfo(pub Vec<SliceInfoElem>);

impl From<SliceInfo> for ndarray::SliceInfo<Vec<ndarray::SliceInfoElem>, IxDyn, IxDyn> {
    fn from(s: SliceInfo) -> ndarray::SliceInfo<Vec<ndarray::SliceInfoElem>, IxDyn, IxDyn> {
        let all_slices: Vec<ndarray::SliceInfoElem> = s
            .0
            .iter()
            .map(|x| ndarray::SliceInfoElem::from(Slice::new(x.start, x.end, x.step.unwrap_or(1))))
            .collect();
        ndarray::SliceInfo::<Vec<ndarray::SliceInfoElem>, IxDyn, IxDyn>::try_from(all_slices)
            .unwrap()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HostTensor<T>(pub ArrayD<T>, pub HostPlacement);

impl<T> Placed for HostTensor<T> {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

pub type HostFloat32Tensor = HostTensor<f32>;
pub type HostFloat64Tensor = HostTensor<f64>;
pub type HostInt8Tensor = HostTensor<i8>;
pub type HostInt16Tensor = HostTensor<i16>;
pub type HostInt32Tensor = HostTensor<i32>;
pub type HostInt64Tensor = HostTensor<i64>;
pub type HostUint8Tensor = HostTensor<u8>;
pub type HostUint16Tensor = HostTensor<u16>;
pub type HostUint32Tensor = HostTensor<u32>;
pub type HostUint64Tensor = HostTensor<u64>;

impl<T> SymbolicType for HostTensor<T> {
    type Type = Symbolic<HostTensor<T>>;
}

impl<T> TryFrom<Symbolic<HostTensor<T>>> for HostTensor<T> {
    type Error = Error;
    fn try_from(v: Symbolic<HostTensor<T>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl From<HostShape> for Symbolic<HostShape> {
    fn from(x: HostShape) -> Self {
        Symbolic::Concrete(x)
    }
}

impl From<HostRing64Tensor> for Symbolic<HostRing64Tensor> {
    fn from(x: HostRing64Tensor) -> Self {
        Symbolic::Concrete(x)
    }
}

impl From<HostRing128Tensor> for Symbolic<HostRing128Tensor> {
    fn from(x: HostRing128Tensor) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<RingT: Placed> From<AbstractHostFixedTensor<RingT>>
    for Symbolic<AbstractHostFixedTensor<RingT>>
{
    fn from(x: AbstractHostFixedTensor<RingT>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl From<HostBitTensor> for Symbolic<HostBitTensor> {
    fn from(x: HostBitTensor) -> Self {
        Symbolic::Concrete(x)
    }
}

impl TryFrom<Symbolic<HostBitTensor>> for HostBitTensor {
    type Error = Error;
    fn try_from(v: Symbolic<HostBitTensor>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl TryFrom<Symbolic<HostRing64Tensor>> for HostRing64Tensor {
    type Error = Error;
    fn try_from(v: Symbolic<HostRing64Tensor>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl TryFrom<Symbolic<HostRing128Tensor>> for HostRing128Tensor {
    type Error = Error;
    fn try_from(v: Symbolic<HostRing128Tensor>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl<RingT: Placed> TryFrom<Symbolic<AbstractHostFixedTensor<RingT>>>
    for AbstractHostFixedTensor<RingT>
{
    type Error = Error;
    fn try_from(v: Symbolic<AbstractHostFixedTensor<RingT>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl<T> From<HostTensor<T>> for Symbolic<HostTensor<T>> {
    fn from(x: HostTensor<T>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<T> PlacementPlace<SyncSession, HostTensor<T>> for HostPlacement {
    fn place(&self, _sess: &SyncSession, x: HostTensor<T>) -> HostTensor<T> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => HostTensor(x.0, self.clone()),
        }
    }
}

/// This implementation is required to do the `plc.place(sess, x)`
impl<T> PlacementPlace<SymbolicSession, Symbolic<HostTensor<T>>> for HostPlacement {
    fn place(
        &self,
        _sess: &SymbolicSession,
        x: Symbolic<HostTensor<T>>,
    ) -> Symbolic<HostTensor<T>> {
        match x {
            Symbolic::Concrete(x) => Symbolic::Concrete(x),
            Symbolic::Symbolic(SymbolicHandle { op, plc: _ }) => {
                Symbolic::Symbolic(SymbolicHandle {
                    op,
                    plc: self.clone(),
                })
            }
        }
    }
}

kernel! {
    HostAddOp,
    [
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::rep_kernel),
    ]
}

impl HostAddOp {
    fn rep_kernel<S: Session>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: cs!(Fixed128Tensor),
        y: cs!(Fixed128Tensor),
    ) -> cs!(Fixed128Tensor)
    where
        Fixed128Tensor: KnownType<S>,
        ReplicatedPlacement:
            PlacementAdd<S, cs!(Fixed128Tensor), cs!(Fixed128Tensor), cs!(Fixed128Tensor)>,
    {
        with_context!(plc, sess, x + y)
    }
}

modelled!(PlacementMean::mean, HostPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (HostRing64Tensor) -> HostRing64Tensor, RingFixedpointMeanOp);
modelled!(PlacementMean::mean, HostPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (HostRing128Tensor) -> HostRing128Tensor, RingFixedpointMeanOp);

kernel! {
    RingFixedpointMeanOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] attributes[axis, scaling_base, scaling_exp] Self::ring64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] attributes[axis, scaling_base, scaling_exp] Self::ring128_kernel),
    ]
}

impl RingFixedpointMeanOp {
    fn ring64_kernel<S: RuntimeSession>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: HostRing64Tensor,
    ) -> HostRing64Tensor
    where
        HostPlacement: PlacementPlace<S, HostRing64Tensor>,
    {
        let scaling_factor = u64::pow(scaling_base, scaling_exp);
        let axis = axis.map(|a| a as usize);
        let mean = HostRing64Tensor::fixedpoint_mean(x, axis, scaling_factor);
        plc.place(sess, mean)
    }

    fn ring128_kernel<S: RuntimeSession>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: HostRing128Tensor,
    ) -> HostRing128Tensor
    where
        HostPlacement: PlacementPlace<S, HostRing128Tensor>,
    {
        let scaling_factor = u128::pow(scaling_base as u128, scaling_exp);
        let axis = axis.map(|a| a as usize);
        let mean = HostRing128Tensor::fixedpoint_mean(x, axis, scaling_factor);
        plc.place(sess, mean)
    }
}

impl HostAddOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        plc.place(sess, x + y)
    }
}

kernel! {
    HostSubOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::rep_kernel),
    ]
}

impl HostSubOp {
    fn rep_kernel<S: Session>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: cs!(Fixed128Tensor),
        y: cs!(Fixed128Tensor),
    ) -> cs!(Fixed128Tensor)
    where
        Fixed128Tensor: KnownType<S>,
        ReplicatedPlacement:
            PlacementSub<S, cs!(Fixed128Tensor), cs!(Fixed128Tensor), cs!(Fixed128Tensor)>,
    {
        with_context!(plc, sess, x - y)
    }

    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        plc.place(sess, x - y)
    }
}

kernel! {
    HostMulOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::rep_kernel),
    ]
}

impl HostMulOp {
    fn rep_kernel<S: Session>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: cs!(Fixed128Tensor),
        y: cs!(Fixed128Tensor),
    ) -> cs!(Fixed128Tensor)
    where
        Fixed128Tensor: KnownType<S>,
        ReplicatedPlacement:
            PlacementMul<S, cs!(Fixed128Tensor), cs!(Fixed128Tensor), cs!(Fixed128Tensor)>,
        ReplicatedPlacement: PlacementTruncPr<S, cs!(Fixed128Tensor), cs!(Fixed128Tensor)>,
    {
        let mul = with_context!(plc, sess, x * y);
        // TODO: Grab precision (27) from the type
        plc.trunc_pr(sess, 27, &mul)
    }
}

impl HostMulOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        plc.place(sess, x * y)
    }
}

kernel! {
    HostDivOp,
    [
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::rep_kernel),
    ]
}

impl HostDivOp {
    fn rep_kernel<S: Session>(
        _sess: &S,
        _plc: &ReplicatedPlacement,
        _x: cs!(Fixed128Tensor),
        _y: cs!(Fixed128Tensor),
    ) -> cs!(Fixed128Tensor)
    where
        Fixed128Tensor: KnownType<S>,
        // ReplicatedPlacement: PlacementDiv<S, cs!(Fixed128Tensor), cs!(Fixed128Tensor), cs!(Fixed128Tensor)>,
    {
        unimplemented!("Lacking division protocols yet") // TODO: implement those
    }

    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        plc.place(sess, x / y)
    }
}

kernel! {
    HostDotOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::rep_kernel),
    ]
}

impl HostDotOp {
    fn rep_kernel<S: Session>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: cs!(Fixed128Tensor),
        y: cs!(Fixed128Tensor),
    ) -> cs!(Fixed128Tensor)
    where
        Fixed128Tensor: KnownType<S>,
        ReplicatedPlacement:
            PlacementDot<S, cs!(Fixed128Tensor), cs!(Fixed128Tensor), cs!(Fixed128Tensor)>,
        ReplicatedPlacement: PlacementTruncPr<S, cs!(Fixed128Tensor), cs!(Fixed128Tensor)>,
    {
        let mul = plc.dot(sess, &x, &y);
        // TODO: Grab precision (27) from the type
        plc.trunc_pr(sess, 27, &mul)
    }
}

impl HostDotOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        plc.place(sess, x.dot(y))
    }
}

impl HostOnesOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar>(
        sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        plc.place(sess, HostTensor::ones(shape))
    }
}

impl ShapeOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> HostShape {
        let raw_shape = RawShape(x.0.shape().into());
        HostShape(raw_shape, plc.clone())
    }
}

modelled!(PlacementSlice::slice, HostPlacement, attributes[slice: SliceInfo] (HostShape) -> HostShape, HostSliceOp);
modelled!(PlacementSlice::slice, HostPlacement, attributes[slice: SliceInfo] (HostRing64Tensor) -> HostRing64Tensor, HostSliceOp);
modelled!(PlacementSlice::slice, HostPlacement, attributes[slice: SliceInfo] (HostRing128Tensor) -> HostRing128Tensor, HostSliceOp);

kernel! {
    HostSliceOp,
    [
        (HostPlacement, (HostShape) -> HostShape => [runtime] attributes[slice] Self::shape_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] attributes[slice] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] attributes[slice] Self::kernel),
    ]
}

impl HostSliceOp {
    pub fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        slice_info: SliceInfo,
        x: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T>
    where
        T: Clone,
    {
        let slice_info =
            ndarray::SliceInfo::<Vec<ndarray::SliceInfoElem>, IxDyn, IxDyn>::from(slice_info);
        let sliced = x.0.slice(slice_info).to_owned();
        AbstractHostRingTensor(sliced, plc.clone())
    }

    pub fn shape_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        slice_info: SliceInfo,
        x: HostShape,
    ) -> HostShape {
        let slice = x.0.slice(
            slice_info.0[0].start as usize,
            slice_info.0[0].end.unwrap() as usize,
        );
        HostShape(slice, plc.clone())
    }
}

modelled!(PlacementDiag::diag, HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor, HostDiagOp);
modelled!(PlacementDiag::diag, HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor, HostDiagOp);
modelled!(PlacementDiag::diag, HostPlacement, (HostInt8Tensor) -> HostInt8Tensor, HostDiagOp);
modelled!(PlacementDiag::diag, HostPlacement, (HostInt16Tensor) -> HostInt16Tensor, HostDiagOp);
modelled!(PlacementDiag::diag, HostPlacement, (HostInt32Tensor) -> HostInt32Tensor, HostDiagOp);
modelled!(PlacementDiag::diag, HostPlacement, (HostInt64Tensor) -> HostInt64Tensor, HostDiagOp);
modelled!(PlacementDiag::diag, HostPlacement, (HostUint16Tensor) -> HostUint16Tensor, HostDiagOp);
modelled!(PlacementDiag::diag, HostPlacement, (HostUint32Tensor) -> HostUint32Tensor, HostDiagOp);
modelled!(PlacementDiag::diag, HostPlacement, (HostUint64Tensor) -> HostUint64Tensor, HostDiagOp);
modelled!(PlacementDiag::diag, HostPlacement, (HostBitTensor) -> HostBitTensor, HostDiagOp);
modelled!(PlacementDiag::diag, HostPlacement, (HostRing64Tensor) -> HostRing64Tensor, HostDiagOp);
modelled!(PlacementDiag::diag, HostPlacement, (HostRing128Tensor) -> HostRing128Tensor, HostDiagOp);

kernel! {
    HostDiagOp,
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint16Tensor) -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::bit_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring_kernel),
    ]
}

impl HostDiagOp {
    pub fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> HostTensor<T> {
        let diag = x.0.into_diag().into_dimensionality::<IxDyn>().unwrap();
        HostTensor::<T>(diag, plc.clone())
    }

    pub fn ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T> {
        let diag = x.0.into_diag().into_dimensionality::<IxDyn>().unwrap();
        AbstractHostRingTensor::<T>(diag, plc.clone())
    }

    pub fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
    ) -> HostBitTensor {
        let diag = x.0.into_diag().into_dimensionality::<IxDyn>().unwrap();
        HostBitTensor(diag, plc.clone())
    }
}

modelled!(PlacementIndex::index_axis, HostPlacement, attributes[axis:usize, index: usize] (HostRing64Tensor) -> HostRing64Tensor, HostIndexAxisOp);
modelled!(PlacementIndex::index_axis, HostPlacement, attributes[axis:usize, index: usize] (HostRing128Tensor) -> HostRing128Tensor, HostIndexAxisOp);
modelled!(PlacementIndex::index_axis, HostPlacement, attributes[axis:usize, index: usize] (HostBitTensor) -> HostBitTensor, HostIndexAxisOp);

kernel! {
    HostIndexAxisOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] attributes[axis, index] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] attributes[axis, index] Self::kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] attributes[axis, index] Self::bit_kernel),
    ]
}

impl HostIndexAxisOp {
    pub fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T>
    where
        T: Clone,
    {
        let axis = Axis(axis);
        let result = x.0.index_axis(axis, index);
        AbstractHostRingTensor(result.to_owned(), plc.clone())
    }

    pub fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: HostBitTensor,
    ) -> HostBitTensor {
        let axis = Axis(axis);
        let result = x.0.index_axis(axis, index);
        HostBitTensor(result.to_owned(), plc.clone())
    }
}

modelled!(PlacementShlDim::shl_dim, HostPlacement, attributes[amount:usize, bit_length: usize] (HostBitTensor) -> HostBitTensor, HostShlDimOp);

kernel! {
    HostShlDimOp,
    [
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] attributes[amount, bit_length] Self::bit_kernel),
    ]
}

impl HostShlDimOp {
    pub fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        amount: usize,
        bit_length: usize,
        x: HostBitTensor,
    ) -> HostBitTensor {
        let axis = Axis(0);
        let mut raw_tensor_shape = x.0.shape().to_vec();
        raw_tensor_shape.remove(0);
        let raw_shape = raw_tensor_shape.as_ref();

        let zero = ArrayD::from_elem(raw_shape, 0);
        let zero_view = zero.view();

        let concatenated: Vec<_> = (0..bit_length)
            .map(|i| {
                if i < bit_length - amount {
                    x.0.index_axis(axis, i + amount)
                } else {
                    zero_view.clone()
                }
            })
            .collect();

        let result = ndarray::stack(Axis(0), &concatenated).unwrap();

        HostBitTensor(result, plc.clone())
    }
}

modelled!(PlacementBitDec::bit_decompose, HostPlacement, (HostRing64Tensor) -> HostRing64Tensor, HostBitDecOp);
modelled!(PlacementBitDec::bit_decompose, HostPlacement, (HostRing128Tensor) -> HostRing128Tensor, HostBitDecOp);
modelled!(PlacementBitDec::bit_decompose, HostPlacement, (HostRing64Tensor) -> HostBitTensor, HostBitDecOp);
modelled!(PlacementBitDec::bit_decompose, HostPlacement, (HostRing128Tensor) -> HostBitTensor, HostBitDecOp);

kernel! {
    HostBitDecOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring128_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostBitTensor => [runtime] Self::bit64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostBitTensor => [runtime] Self::bit128_kernel),
    ]
}

impl HostBitDecOp {
    fn ring64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing64Tensor,
    ) -> HostRing64Tensor
where {
        let shape = x.shape();
        let raw_shape = shape.0 .0;
        let ones = ArrayD::from_elem(raw_shape, Wrapping(1));

        let bit_rep: Vec<_> = (0..HostRing64Tensor::SIZE)
            .map(|i| (&x.0 >> i) & (&ones))
            .collect();
        let bit_rep_view: Vec<_> = bit_rep.iter().map(ArrayView::from).collect();

        // by default we put bits as rows, ie access i'th bit from tensor T is done through index_axis(Axis(0), T)
        // in the current protocols it's easier to reason that the bits are stacked on axis(0)
        let result = ndarray::stack(Axis(0), &bit_rep_view).unwrap();
        AbstractHostRingTensor(result, plc.clone())
    }

    fn ring128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing128Tensor,
    ) -> HostRing128Tensor
where {
        let shape = x.shape();
        let raw_shape = shape.0 .0;
        let ones = ArrayD::from_elem(raw_shape, Wrapping(1));

        let bit_rep: Vec<_> = (0..HostRing128Tensor::SIZE)
            .map(|i| (&x.0 >> i) & (&ones))
            .collect();

        let bit_rep_view: Vec<_> = bit_rep.iter().map(ArrayView::from).collect();
        let result = ndarray::stack(Axis(0), &bit_rep_view).unwrap();
        AbstractHostRingTensor(result, plc.clone())
    }

    fn bit64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing64Tensor,
    ) -> HostBitTensor
where {
        let shape = x.shape();
        let raw_shape = shape.0 .0;
        let ones = ArrayD::from_elem(raw_shape, Wrapping(1));

        let bit_rep: Vec<_> = (0..HostRing64Tensor::SIZE)
            .map(|i| (&x.0 >> i) & (&ones))
            .collect();

        let bit_rep_view: Vec<_> = bit_rep.iter().map(ArrayView::from).collect();
        let result = ndarray::stack(Axis(0), &bit_rep_view).unwrap();
        // we unwrap only at the end since shifting can cause overflow
        HostBitTensor(result.map(|v| v.0 as u8), plc.clone())
    }

    fn bit128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing128Tensor,
    ) -> HostBitTensor
where {
        let shape = x.shape();
        let raw_shape = shape.0 .0;
        let ones = ArrayD::from_elem(raw_shape, Wrapping(1));

        let bit_rep: Vec<_> = (0..HostRing128Tensor::SIZE)
            .map(|i| (&x.0 >> i) & (&ones))
            .collect();

        let bit_rep_view: Vec<_> = bit_rep.iter().map(ArrayView::from).collect();
        let result = ndarray::stack(Axis(0), &bit_rep_view).unwrap();
        // we unwrap only at the end since shifting can cause overflow
        HostBitTensor(result.map(|v| v.0 as u8), plc.clone())
    }
}

impl<T> HostTensor<T>
where
    T: LinalgScalar,
{
    pub fn place(plc: &HostPlacement, x: ArrayD<T>) -> HostTensor<T> {
        HostTensor::<T>(x, plc.clone())
    }

    pub fn atleast_2d(self, to_column_vector: bool) -> HostTensor<T> {
        match self.0.ndim() {
            0 => HostTensor::<T>(self.0.into_shape(IxDyn(&[1, 1])).unwrap(), self.1),
            1 => {
                let length = self.0.len();
                let newshape = if to_column_vector {
                    IxDyn(&[length, 1])
                } else {
                    IxDyn(&[1, length])
                };
                HostTensor::<T>(self.0.into_shape(newshape).unwrap(), self.1)
            }
            2 => self,
            otherwise => panic!(
                "Tensor input for `atleast_2d` must have rank <= 2, found rank {:?}.",
                otherwise
            ),
        }
    }

    pub fn dot(self, other: HostTensor<T>) -> HostTensor<T> {
        match (self.0.ndim(), other.0.ndim()) {
            (1, 1) => {
                let l = self.0.into_dimensionality::<Ix1>().unwrap();
                let r = other.0.into_dimensionality::<Ix1>().unwrap();
                let res = Array::from_elem([], l.dot(&r))
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                HostTensor::<T>(res, self.1)
            }
            (1, 2) => {
                let l = self.0.into_dimensionality::<Ix1>().unwrap();
                let r = other.0.into_dimensionality::<Ix2>().unwrap();
                let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                HostTensor::<T>(res, self.1)
            }
            (2, 1) => {
                let l = self.0.into_dimensionality::<Ix2>().unwrap();
                let r = other.0.into_dimensionality::<Ix1>().unwrap();
                let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                HostTensor::<T>(res, self.1)
            }
            (2, 2) => {
                let l = self.0.into_dimensionality::<Ix2>().unwrap();
                let r = other.0.into_dimensionality::<Ix2>().unwrap();
                let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                HostTensor::<T>(res, self.1)
            }
            (self_rank, other_rank) => panic!(
                // TODO: replace with proper error handling
                "Dot<HostTensor> not implemented between tensors of rank {:?} and {:?}.",
                self_rank, other_rank,
            ),
        }
    }

    pub fn ones(shape: HostShape) -> Self {
        HostTensor::<T>(ArrayD::ones(shape.0 .0), shape.1)
    }

    pub fn reshape(self, newshape: HostShape) -> Self {
        HostTensor::<T>(self.0.into_shape(newshape.0 .0).unwrap(), self.1) // TODO need to be fix (unwrap)
    }

    pub fn expand_dims(self, mut axis: Vec<usize>) -> Self {
        let plc = (&self.1).clone();
        axis.sort_by_key(|ax| Reverse(*ax));
        let newshape = self.shape().0.extend_singletons(axis);
        self.reshape(HostShape(newshape, plc))
    }

    pub fn squeeze(self, axis: Option<usize>) -> Self {
        let plc = (&self.1).clone();
        let newshape = self.shape().0.squeeze(axis);
        self.reshape(HostShape(newshape, plc))
    }

    pub fn shape(&self) -> HostShape {
        HostShape(RawShape(self.0.shape().into()), self.1.clone())
    }

    pub fn sum(self, axis: Option<usize>) -> Self {
        if let Some(i) = axis {
            HostTensor::<T>(self.0.sum_axis(Axis(i)), self.1)
        } else {
            let out = Array::from_elem([], self.0.sum())
                .into_dimensionality::<IxDyn>()
                .unwrap();
            HostTensor::<T>(out, self.1)
        }
    }

    pub fn transpose(self) -> Self {
        HostTensor::<T>(self.0.reversed_axes(), self.1)
    }
}

impl<T> HostTensor<T>
where
    T: LinalgScalar + FromPrimitive,
{
    pub fn mean(self, axis: Option<usize>) -> Self {
        match axis {
            Some(i) => {
                let reduced = self.0.mean_axis(Axis(i)).unwrap();
                HostTensor::<T>(reduced, self.1)
            }
            None => {
                let mean = self.0.mean().unwrap();
                let out = Array::from_elem([], mean)
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                HostTensor::<T>(out, self.1)
            }
        }
    }
}

impl HostMeanOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        match axis {
            Some(i) => {
                let reduced: ArrayD<T> = x.0.mean_axis(Axis(i as usize)).unwrap();
                HostTensor::place(plc, reduced)
            }
            None => {
                let mean = x.0.mean().unwrap();
                let out = Array::from_elem([], mean)
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                HostTensor::place(plc, out)
            }
        }
    }

    // TODO: Make it generic for any FixedTensor
    pub fn rep_kernel<S: Session>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: cs!(Fixed128Tensor),
    ) -> cs!(Fixed128Tensor)
    where
        Fixed128Tensor: KnownType<S>,
        ReplicatedPlacement: PlacementMean<S, cs!(Fixed128Tensor), cs!(Fixed128Tensor)>,
        ReplicatedPlacement: PlacementTruncPr<S, cs!(Fixed128Tensor), cs!(Fixed128Tensor)>,
    {
        // TODO: grab scaling base and exp from somewhere else
        let mean = plc.mean(sess, axis, 2, 27, &x);
        plc.trunc_pr(sess, 27, &mean)
    }
}

impl HostSqrtOp {
    pub fn kernel<S: RuntimeSession, T: 'static + Float>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let x_sqrt = x.0.mapv(T::sqrt);
        HostTensor::place(plc, x_sqrt)
    }
}

impl HostSumOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let axis = axis.map(|a| a as usize);
        plc.place(sess, x.sum(axis))
    }

    // TODO: Make it generic for any FixedTensor
    pub fn rep_kernel<S: Session>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: cs!(Fixed128Tensor),
    ) -> cs!(Fixed128Tensor)
    where
        Fixed128Tensor: KnownType<S>,
        ReplicatedPlacement: PlacementSum<S, cs!(Fixed128Tensor), cs!(Fixed128Tensor)>,
    {
        plc.sum(sess, axis, &x)
    }
}

impl HostExpandDimsOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<u32>,
        x: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let axis = axis.iter().map(|a| *a as usize).collect();
        plc.place(sess, x.expand_dims(axis))
    }
}

impl HostSqueezeOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let axis = axis.map(|a| a as usize);
        plc.place(sess, x.squeeze(axis))
    }
}

impl HostConcatOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        axis: u32,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> HostTensor<T> {
        let ax = Axis(axis as usize);
        let x = x.0.view();
        let y = y.0.view();

        let c =
            ndarray::concatenate(ax, &[x, y]).expect("Failed to concatenate arrays with ndarray");
        HostTensor(c, plc.clone())
    }
}

impl HostTransposeOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        plc.place(sess, x.transpose())
    }
}

impl HostInverseOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive + Lapack>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        plc.place(sess, x.inv())
    }
}

impl<T> HostTensor<T>
where
    T: Scalar + Lapack,
{
    pub fn inv(self) -> Self {
        match self.0.ndim() {
            2 => {
                let two_dim: Array2<T> = self.0.into_dimensionality::<Ix2>().unwrap();
                HostTensor::<T>(
                    two_dim
                        .inv()
                        .unwrap()
                        .into_dimensionality::<IxDyn>()
                        .unwrap(),
                    self.1,
                )
            }
            other_rank => panic!(
                "Inverse only defined for rank 2 matrices, not rank {:?}",
                other_rank,
            ),
        }
    }
}

// This implementation is only used by the old kernels. Construct HostTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<ArrayD<T>> for HostTensor<T>
where
    T: LinalgScalar,
{
    fn from(v: ArrayD<T>) -> HostTensor<T> {
        HostTensor::<T>(
            v,
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the old kernels
            },
        )
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> Add for HostTensor<T>
where
    T: LinalgScalar,
{
    type Output = HostTensor<T>;
    fn add(self, other: HostTensor<T>) -> Self::Output {
        match self.0.broadcast(other.0.dim()) {
            Some(self_broadcasted) => {
                HostTensor::<T>(self_broadcasted.to_owned() + other.0, self.1.clone())
            }
            None => HostTensor::<T>(self.0 + other.0, self.1.clone()),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> Sub for HostTensor<T>
where
    T: LinalgScalar,
{
    type Output = HostTensor<T>;
    fn sub(self, other: HostTensor<T>) -> Self::Output {
        match self.0.broadcast(other.0.dim()) {
            Some(self_broadcasted) => {
                HostTensor::<T>(self_broadcasted.to_owned() - other.0, self.1.clone())
            }
            None => HostTensor::<T>(self.0 - other.0, self.1.clone()),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> Mul for HostTensor<T>
where
    T: LinalgScalar,
{
    type Output = HostTensor<T>;
    fn mul(self, other: HostTensor<T>) -> Self::Output {
        match self.0.broadcast(other.0.dim()) {
            Some(self_broadcasted) => {
                HostTensor::<T>(self_broadcasted.to_owned() * other.0, self.1.clone())
            }
            None => HostTensor::<T>(self.0 * other.0, self.1.clone()),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> Div for HostTensor<T>
where
    T: LinalgScalar,
{
    type Output = HostTensor<T>;
    fn div(self, other: HostTensor<T>) -> Self::Output {
        match self.0.broadcast(other.0.dim()) {
            Some(self_broadcasted) => {
                HostTensor::<T>(self_broadcasted.to_owned() / other.0, self.1.clone())
            }
            None => HostTensor::<T>(self.0 / other.0, self.1.clone()),
        }
    }
}

// This implementation is only used by the old kernels. Construct HostTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<Vec<T>> for HostTensor<T> {
    fn from(v: Vec<T>) -> HostTensor<T> {
        HostTensor(
            Array::from(v).into_dyn(),
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the old kernel
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct HostTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<Array1<T>> for HostTensor<T> {
    fn from(v: Array1<T>) -> HostTensor<T> {
        HostTensor(
            v.into_dyn(),
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the old kernel
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct HostTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<Array2<T>> for HostTensor<T> {
    fn from(v: Array2<T>) -> HostTensor<T> {
        HostTensor(
            v.into_dyn(),
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the old kernel
            },
        )
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
pub fn concatenate<T>(axis: usize, arrays: &[HostTensor<T>]) -> HostTensor<T>
where
    T: LinalgScalar,
{
    let ax = Axis(axis);
    let inner_arrays: Vec<_> = arrays.iter().map(|a| a.0.view()).collect();

    let c = ndarray::concatenate(ax, &inner_arrays).unwrap();
    HostTensor::<T>(
        c,
        HostPlacement {
            owner: "TODO".into(),
        },
    )
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HostBitTensor(pub ArrayD<u8>, HostPlacement);

impl SymbolicType for HostBitTensor {
    type Type = Symbolic<HostBitTensor>;
}

impl<S: Session> Tensor<S> for HostBitTensor {
    type Scalar = u8;
}

impl<S: Session> Tensor<S> for Symbolic<HostBitTensor> {
    type Scalar = u8;
}

impl Placed for HostBitTensor {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl PlacementPlace<SyncSession, HostBitTensor> for HostPlacement {
    fn place(&self, _sess: &SyncSession, x: HostBitTensor) -> HostBitTensor {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                HostBitTensor(x.0, self.clone())
            }
        }
    }
}

impl PlacementPlace<SymbolicSession, Symbolic<HostBitTensor>> for HostPlacement {
    fn place(
        &self,
        _sess: &SymbolicSession,
        x: Symbolic<HostBitTensor>,
    ) -> Symbolic<HostBitTensor> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                match x {
                    Symbolic::Concrete(x) => {
                        // TODO insert Place ops?
                        Symbolic::Concrete(HostBitTensor(x.0, self.clone()))
                    }
                    Symbolic::Symbolic(SymbolicHandle { op, plc: _ }) => {
                        // TODO insert `Place` ops here?
                        Symbolic::Symbolic(SymbolicHandle {
                            op,
                            plc: self.clone(),
                        })
                    }
                }
            }
        }
    }
}

impl ShapeOp {
    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
    ) -> HostShape {
        let raw_shape = RawShape(x.0.shape().into());
        HostShape(raw_shape, plc.clone())
    }
}

impl HostReshapeOp {
    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
        shape: HostShape,
    ) -> HostBitTensor {
        HostBitTensor(x.0.into_shape(shape.0 .0).unwrap(), plc.clone()) // TODO need to be fix (unwrap)
    }
}

impl HostReshapeOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        shape: HostShape,
    ) -> HostTensor<T>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        plc.place(sess, x.reshape(shape))
    }
}

modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> HostBitTensor, BitFillOp);

kernel! {
    BitFillOp,
    [
        (HostPlacement, (HostShape) -> HostBitTensor => [runtime] attributes[value: Bit] Self::kernel),
    ]
}

impl BitFillOp {
    fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u8,
        shape: HostShape,
    ) -> HostBitTensor {
        assert!(value == 0 || value == 1);
        let raw_shape = shape.0 .0;
        let raw_tensor = ArrayD::from_elem(raw_shape.as_ref(), value as u8);
        HostBitTensor(raw_tensor, plc.clone())
    }
}

modelled!(PlacementSampleUniform::sample_uniform, HostPlacement, (HostShape) -> HostBitTensor, BitSampleOp);

kernel! {
    BitSampleOp,
    [
        (HostPlacement, (HostShape) -> HostBitTensor => [runtime] Self::kernel),
    ]
}

impl BitSampleOp {
    fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> HostBitTensor {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        HostBitTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }
}

modelled!(PlacementSampleUniformSeeded::sample_uniform_seeded, HostPlacement, (HostShape, Seed) -> HostBitTensor, BitSampleSeededOp);

kernel! {
    BitSampleSeededOp,
    [
        (HostPlacement, (HostShape, Seed) -> HostBitTensor => [runtime] Self::kernel),
    ]
}

impl BitSampleSeededOp {
    fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> HostBitTensor {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        HostBitTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }
}

modelled!(PlacementXor::xor, HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor, BitXorOp);
modelled_alias!(PlacementAdd::add, HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => PlacementXor::xor); // add = xor in Z2
modelled_alias!(PlacementSub::sub, HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => PlacementXor::xor); // sub = xor in Z2

kernel! {
    BitXorOp,
    [
        (HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => [runtime] Self::kernel),
    ]
}

impl BitXorOp {
    fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
        y: HostBitTensor,
    ) -> HostBitTensor {
        HostBitTensor(x.0 ^ y.0, plc.clone())
    }
}

modelled!(PlacementAnd::and, HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor, BitAndOp);
modelled!(PlacementAnd::and, HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor, BitAndOp);
modelled!(PlacementAnd::and, HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor, BitAndOp);

modelled_alias!(PlacementMul::mul, HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => PlacementAnd::and); // mul = and in Z2

kernel! {
    BitAndOp,
    [
        (HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => [runtime] Self::bit_kernel),
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring_kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring_kernel),
    ]
}

impl BitAndOp {
    fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
        y: HostBitTensor,
    ) -> HostBitTensor {
        HostBitTensor(x.0 & y.0, plc.clone())
    }

    fn ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractHostRingTensor<T>,
        y: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: BitAnd<Wrapping<T>, Output = Wrapping<T>>,
    {
        AbstractHostRingTensor(x.0 & y.0, plc.clone())
    }
}

impl HostBitTensor {
    #[cfg_attr(
        feature = "exclude_old_framework",
        deprecated(
            note = "This function is only used by the old kernels, which are not aware of the placements. See BitSampleSeededOp::kernel for the new code"
        )
    )]
    pub fn sample_uniform(shape: &RawShape) -> Self {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let ix = IxDyn(shape.0.as_ref());
        HostBitTensor(
            Array::from_shape_vec(ix, values).unwrap(),
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the older kernels.
            },
        )
    }

    #[cfg_attr(
        feature = "exclude_old_framework",
        deprecated(
            note = "This function is only used by the old kernels, which are not aware of the placements. See BitSampleSeededOp::kernel for the new code"
        )
    )]
    pub fn sample_uniform_seeded(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let ix = IxDyn(shape.0.as_ref());
        HostBitTensor(
            Array::from_shape_vec(ix, values).unwrap(),
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the older kernels.
            },
        )
    }
}

impl HostBitTensor {
    #[cfg_attr(
        feature = "exclude_old_framework",
        deprecated(
            note = "This function is only used by the old kernels, which are not aware of the placements. See BitFillOp::kernel for the new code"
        )
    )]
    pub fn fill(shape: &RawShape, el: u8) -> HostBitTensor {
        assert!(
            el == 0 || el == 1,
            "cannot fill a HostBitTensor with a value {:?}",
            el
        );
        HostBitTensor(
            ArrayD::from_elem(shape.0.as_ref(), el & 1),
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the older kernels.
            },
        )
    }
}

#[allow(dead_code)]
impl HostBitTensor {
    pub(crate) fn from_raw_plc(raw_tensor: ArrayD<u8>, plc: HostPlacement) -> HostBitTensor {
        HostBitTensor(raw_tensor.into_dyn(), plc)
    }
}

// This implementation is only used by the old kernels. Construct HostBitTensor(raw_tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl From<ArrayD<u8>> for HostBitTensor {
    fn from(a: ArrayD<u8>) -> HostBitTensor {
        let wrapped = a.mapv(|ai| (ai & 1) as u8);
        HostBitTensor(
            wrapped,
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the older kernels.
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct HostBitTensor(raw_tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl From<Vec<u8>> for HostBitTensor {
    fn from(v: Vec<u8>) -> HostBitTensor {
        let ix = IxDyn(&[v.len()]);
        HostBitTensor(
            Array::from_shape_vec(ix, v).unwrap(),
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the older kernels.
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct HostBitTensor(raw_tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl From<&[u8]> for HostBitTensor {
    fn from(v: &[u8]) -> HostBitTensor {
        let ix = IxDyn(&[v.len()]);
        let v_wrapped: Vec<_> = v.iter().map(|vi| *vi & 1).collect();
        HostBitTensor(
            Array::from_shape_vec(ix, v_wrapped).unwrap(),
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the older kernels.
            },
        )
    }
}

impl From<HostBitTensor> for ArrayD<u8> {
    fn from(b: HostBitTensor) -> ArrayD<u8> {
        b.0
    }
}

impl BitXor for HostBitTensor {
    type Output = HostBitTensor;
    fn bitxor(self, other: Self) -> Self::Output {
        assert_eq!(self.1, other.1);
        HostBitTensor(self.0 ^ other.0, self.1)
    }
}

impl BitAnd for HostBitTensor {
    type Output = HostBitTensor;
    fn bitand(self, other: Self) -> Self::Output {
        assert_eq!(self.1, other.1);
        HostBitTensor(self.0 & other.0, self.1)
    }
}

modelled!(PlacementBitExtract::bit_extract, HostPlacement, attributes[bit_idx: usize] (HostRing64Tensor) -> HostBitTensor, BitExtractOp);
modelled!(PlacementBitExtract::bit_extract, HostPlacement, attributes[bit_idx: usize] (HostRing128Tensor) -> HostBitTensor, BitExtractOp);

kernel! {
    BitExtractOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostBitTensor => [runtime] attributes[bit_idx] Self::kernel64),
        (HostPlacement, (HostRing128Tensor) -> HostBitTensor => [runtime] attributes[bit_idx] Self::kernel128),
    ]
}

impl BitExtractOp {
    fn kernel64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        bit_idx: usize,
        x: HostRing64Tensor,
    ) -> HostBitTensor {
        HostBitTensor((x >> bit_idx).0.mapv(|ai| (ai.0 & 1) as u8), plc.clone())
    }
    fn kernel128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        bit_idx: usize,
        x: HostRing128Tensor,
    ) -> HostBitTensor {
        HostBitTensor((x >> bit_idx).0.mapv(|ai| (ai.0 & 1) as u8), plc.clone())
    }
}

impl RingInjectOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        bit_idx: usize,
        x: HostBitTensor,
    ) -> AbstractHostRingTensor<T>
    where
        T: From<u8>,
        Wrapping<T>: Shl<usize, Output = Wrapping<T>>,
    {
        AbstractHostRingTensor(x.0.mapv(|ai| Wrapping(T::from(ai)) << bit_idx), plc.clone())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AbstractHostFixedTensor<HostRingT>(pub HostRingT);

pub type HostFixed64Tensor = AbstractHostFixedTensor<HostRing64Tensor>;

impl SymbolicType for HostFixed64Tensor {
    type Type = Symbolic<AbstractHostFixedTensor<Symbolic<HostRing64Tensor>>>;
}

impl<T> From<T> for HostFixed64Tensor
where
    HostRing64Tensor: From<T>,
{
    fn from(x: T) -> Self {
        AbstractHostFixedTensor(HostRing64Tensor::from(x))
    }
}

pub type HostFixed128Tensor = AbstractHostFixedTensor<HostRing128Tensor>;

impl SymbolicType for HostFixed128Tensor {
    type Type = Symbolic<AbstractHostFixedTensor<Symbolic<HostRing128Tensor>>>;
}

impl<T> From<T> for HostFixed128Tensor
where
    HostRing128Tensor: From<T>,
{
    fn from(x: T) -> Self {
        AbstractHostFixedTensor(HostRing128Tensor::from(x))
    }
}

impl<RingT: Placed> Placed for AbstractHostFixedTensor<RingT> {
    type Placement = RingT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.0.placement()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AbstractHostRingTensor<T>(pub ArrayD<Wrapping<T>>, pub HostPlacement);

/// Tensor for ring arithmetic over Z_{2^64}
pub type HostRing64Tensor = AbstractHostRingTensor<u64>;

/// Tensor for ring arithmetic over Z_{2^128}
pub type HostRing128Tensor = AbstractHostRingTensor<u128>;

impl SymbolicType for HostRing64Tensor {
    type Type = Symbolic<HostRing64Tensor>;
}

impl SymbolicType for HostRing128Tensor {
    type Type = Symbolic<HostRing128Tensor>;
}

impl CanonicalType for HostBitTensor {
    type Type = HostBitTensor;
}

impl CanonicalType for Symbolic<HostBitTensor> {
    type Type = HostBitTensor;
}

impl CanonicalType for HostShape {
    type Type = HostShape;
}

impl CanonicalType for Symbolic<HostShape> {
    type Type = HostShape;
}

impl<S: Session, T> Tensor<S> for AbstractHostRingTensor<T> {
    type Scalar = T;
}

impl<S: Session, T> Tensor<S> for Symbolic<AbstractHostRingTensor<T>> {
    type Scalar = T;
}

impl<T> Placed for AbstractHostRingTensor<T> {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

pub trait RingSize {
    const SIZE: usize;
}

impl RingSize for HostRing64Tensor {
    const SIZE: usize = 64;
}

impl RingSize for HostRing128Tensor {
    const SIZE: usize = 128;
}

pub trait FromRawPlc<P, T> {
    fn from_raw_plc(raw_tensor: ArrayD<T>, plc: P) -> AbstractHostRingTensor<T>;
}

impl<P> FromRawPlc<P, u64> for HostRing64Tensor
where
    P: Into<HostPlacement>,
{
    fn from_raw_plc(raw_tensor: ArrayD<u64>, plc: P) -> HostRing64Tensor {
        let tensor = raw_tensor.mapv(Wrapping).into_dyn();
        AbstractHostRingTensor(tensor, plc.into())
    }
}

impl<P> FromRawPlc<P, u128> for HostRing128Tensor
where
    P: Into<HostPlacement>,
{
    fn from_raw_plc(raw_tensor: ArrayD<u128>, plc: P) -> HostRing128Tensor {
        let tensor = raw_tensor.mapv(Wrapping).into_dyn();
        AbstractHostRingTensor(tensor, plc.into())
    }
}

impl<R: RingSize + Placed> RingSize for Symbolic<R> {
    const SIZE: usize = R::SIZE;
}

impl<T> PlacementPlace<SyncSession, AbstractHostRingTensor<T>> for HostPlacement
where
    AbstractHostRingTensor<T>: Placed<Placement = HostPlacement>,
{
    fn place(
        &self,
        _sess: &SyncSession,
        x: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                AbstractHostRingTensor(x.0, self.clone())
            }
        }
    }
}

impl<T> PlacementPlace<SymbolicSession, Symbolic<AbstractHostRingTensor<T>>> for HostPlacement {
    fn place(
        &self,
        _sess: &SymbolicSession,
        x: Symbolic<AbstractHostRingTensor<T>>,
    ) -> Symbolic<AbstractHostRingTensor<T>> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                match x {
                    Symbolic::Concrete(x) => {
                        // TODO insert Place ops?
                        Symbolic::Concrete(AbstractHostRingTensor(x.0, self.clone()))
                    }
                    Symbolic::Symbolic(SymbolicHandle { op, plc: _ }) => {
                        // TODO insert `Place` ops here?
                        Symbolic::Symbolic(SymbolicHandle {
                            op,
                            plc: self.clone(),
                        })
                    }
                }
            }
        }
    }
}

modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> HostRing64Tensor, RingFillOp);
modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> HostRing128Tensor, RingFillOp);

kernel! {
    RingFillOp,
    [
        (HostPlacement, (HostShape) -> HostRing64Tensor => [runtime] attributes[value: Ring64] Self::ring64_kernel),
        (HostPlacement, (HostShape) -> HostRing128Tensor => [runtime] attributes[value: Ring128] Self::ring128_kernel),
    ]
}

impl RingFillOp {
    fn ring64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u64,
        shape: HostShape,
    ) -> HostRing64Tensor {
        let raw_shape = shape.0 .0;
        let raw_tensor = ArrayD::from_elem(raw_shape.as_ref(), Wrapping(value));
        AbstractHostRingTensor(raw_tensor, plc.clone())
    }

    fn ring128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u128,
        shape: HostShape,
    ) -> HostRing128Tensor {
        let raw_shape = shape.0 .0;
        let raw_tensor = ArrayD::from_elem(raw_shape.as_ref(), Wrapping(value));
        AbstractHostRingTensor(raw_tensor, plc.clone())
    }
}

impl ShapeOp {
    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractHostRingTensor<T>,
    ) -> HostShape {
        let raw_shape = RawShape(x.0.shape().into());
        HostShape(raw_shape, plc.clone())
    }
}

impl HostReshapeOp {
    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractHostRingTensor<T>,
        shape: HostShape,
    ) -> AbstractHostRingTensor<T> {
        AbstractHostRingTensor::<T>(x.0.into_shape(shape.0 .0).unwrap(), plc.clone())
        // TODO need to be fix (unwrap)
    }
}

modelled!(PlacementAdd::add, HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor, RingAddOp);
modelled!(PlacementAdd::add, HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor, RingAddOp);

kernel! {
    RingAddOp,
    [
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

impl RingAddOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractHostRingTensor<T>,
        y: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Add<Wrapping<T>, Output = Wrapping<T>>,
    {
        AbstractHostRingTensor(x.0 + y.0, plc.clone())
    }
}

modelled!(PlacementSub::sub, HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor, RingSubOp);
modelled!(PlacementSub::sub, HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor, RingSubOp);

kernel! {
    RingSubOp,
    [
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

impl RingSubOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractHostRingTensor<T>,
        y: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Sub<Wrapping<T>, Output = Wrapping<T>>,
    {
        AbstractHostRingTensor(x.0 - y.0, plc.clone())
    }
}

modelled!(PlacementNeg::neg, HostPlacement, (HostRing64Tensor) -> HostRing64Tensor, RingNegOp);
modelled!(PlacementNeg::neg, HostPlacement, (HostRing128Tensor) -> HostRing128Tensor, RingNegOp);

kernel! {
    RingNegOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

impl RingNegOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Neg<Output = Wrapping<T>>,
    {
        AbstractHostRingTensor(x.0.neg(), plc.clone())
    }
}

modelled!(PlacementMul::mul, HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor, RingMulOp);
modelled!(PlacementMul::mul, HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor, RingMulOp);

kernel! {
    RingMulOp,
    [
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

impl RingMulOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractHostRingTensor<T>,
        y: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Mul<Wrapping<T>, Output = Wrapping<T>>,
    {
        AbstractHostRingTensor(x.0 * y.0, plc.clone())
    }
}

modelled!(PlacementDot::dot, HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor, RingDotOp);
modelled!(PlacementDot::dot, HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor, RingDotOp);

kernel! {
    RingDotOp,
    [
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

impl RingDotOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractHostRingTensor<T>,
        y: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Mul<Wrapping<T>, Output = Wrapping<T>>,
        Wrapping<T>: LinalgScalar,
    {
        AbstractHostRingTensor(x.dot(y).0, plc.clone())
    }
}

modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (HostRing64Tensor) -> HostRing64Tensor, RingSumOp);
modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (HostRing128Tensor) -> HostRing128Tensor, RingSumOp);

kernel! {
    RingSumOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] attributes[axis] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] attributes[axis] Self::kernel),
    ]
}

impl RingSumOp {
    fn kernel<S: RuntimeSession, T>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T>
    where
        T: FromPrimitive + Zero,
        Wrapping<T>: Clone,
        Wrapping<T>: Add<Output = Wrapping<T>>,
        HostPlacement: PlacementPlace<S, AbstractHostRingTensor<T>>,
    {
        let sum = x.sum(axis.map(|a| a as usize));
        plc.place(sess, sum)
    }
}

modelled!(PlacementShl::shl, HostPlacement, attributes[amount: usize] (HostRing64Tensor) -> HostRing64Tensor, RingShlOp);
modelled!(PlacementShl::shl, HostPlacement, attributes[amount: usize] (HostRing128Tensor) -> HostRing128Tensor, RingShlOp);

kernel! {
    RingShlOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] attributes[amount] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] attributes[amount] Self::kernel),
    ]
}

impl RingShlOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        amount: usize,
        x: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Shl<usize, Output = Wrapping<T>>,
    {
        AbstractHostRingTensor(x.0 << amount, plc.clone())
    }
}

modelled!(PlacementShr::shr, HostPlacement, attributes[amount: usize] (HostRing64Tensor) -> HostRing64Tensor, RingShrOp);
modelled!(PlacementShr::shr, HostPlacement, attributes[amount: usize] (HostRing128Tensor) -> HostRing128Tensor, RingShrOp);

kernel! {
    RingShrOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] attributes[amount] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] attributes[amount] Self::kernel),
    ]
}

impl RingShrOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        amount: usize,
        x: AbstractHostRingTensor<T>,
    ) -> AbstractHostRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Shr<usize, Output = Wrapping<T>>,
    {
        AbstractHostRingTensor(x.0 >> amount, plc.clone())
    }
}

modelled!(PlacementSample::sample, HostPlacement, attributes[max_value: Option<u64>] (HostShape) -> HostRing64Tensor, RingSampleOp);
modelled!(PlacementSample::sample, HostPlacement, attributes[max_value: Option<u64>] (HostShape) -> HostRing128Tensor, RingSampleOp);

kernel! {
    RingSampleOp,
    [
        (HostPlacement, (HostShape) -> HostRing64Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_uniform_u64(ctx, plc, shape)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_bits_u64(ctx, plc, shape)
                })),
                _ => unimplemented!(),  // TODO: replace
            }
        }),
        (HostPlacement, (HostShape) -> HostRing128Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_uniform_u128(ctx, plc, shape)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_bits_u128(ctx, plc, shape)
                })),
                _ => unimplemented!(),  // TODO: replace
            }
        }),
    ]
}

impl RingSampleOp {
    fn kernel_uniform_u64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> HostRing64Tensor {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let raw_array = Array::from_shape_vec(ix, values).unwrap();
        AbstractHostRingTensor(raw_array, plc.clone())
    }

    fn kernel_bits_u64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> HostRing64Tensor {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u64)).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        AbstractHostRingTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }

    fn kernel_uniform_u128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> HostRing128Tensor {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size)
            .map(|_| Wrapping(((rng.next_u64() as u128) << 64) + rng.next_u64() as u128))
            .collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        AbstractHostRingTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }

    fn kernel_bits_u128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> HostRing128Tensor {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u128)).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        AbstractHostRingTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }
}

modelled!(PlacementSampleSeeded::sample_seeded, HostPlacement, attributes[max_value: Option<u64>] (HostShape, Seed) -> HostRing64Tensor, RingSampleSeededOp);
modelled!(PlacementSampleSeeded::sample_seeded, HostPlacement, attributes[max_value: Option<u64>] (HostShape, Seed) -> HostRing128Tensor, RingSampleSeededOp);

kernel! {
    RingSampleSeededOp,
    [
        (HostPlacement, (HostShape, Seed) -> HostRing64Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_uniform_u64(ctx, plc, shape, seed)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_bits_u64(ctx, plc, shape, seed)
                })),
                _ => unimplemented!(),
            }
        }),
        (HostPlacement, (HostShape, Seed) -> HostRing128Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_uniform_u128(ctx, plc, shape, seed)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_bits_u128(ctx, plc, shape, seed)
                })),
                _ => unimplemented!(),
            }
        }),
    ]
}

impl RingSampleSeededOp {
    fn kernel_uniform_u64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> HostRing64Tensor {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let raw_array = Array::from_shape_vec(ix, values).unwrap();
        AbstractHostRingTensor(raw_array, plc.clone())
    }

    fn kernel_bits_u64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> HostRing64Tensor {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u64)).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        AbstractHostRingTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }

    fn kernel_uniform_u128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> HostRing128Tensor {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size)
            .map(|_| Wrapping(((rng.next_u64() as u128) << 64) + rng.next_u64() as u128))
            .collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        AbstractHostRingTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }

    fn kernel_bits_u128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> HostRing128Tensor {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u128)).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        AbstractHostRingTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl HostRing64Tensor {
    pub fn sample_uniform(shape: &RawShape) -> HostRing64Tensor {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape.0.as_ref());
        HostRing64Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
    pub fn sample_bits(shape: &RawShape) -> Self {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u64)).collect();
        let ix = IxDyn(shape.0.as_ref());
        HostRing64Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl HostRing128Tensor {
    pub fn sample_uniform(shape: &RawShape) -> Self {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size)
            .map(|_| {
                let upper = rng.next_u64() as u128;
                let lower = rng.next_u64() as u128;
                Wrapping((upper << 64) + lower)
            })
            .collect();
        let ix = IxDyn(shape.0.as_ref());
        HostRing128Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }

    pub fn sample_bits(shape: &RawShape) -> Self {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u128)).collect();
        let ix = IxDyn(shape.0.as_ref());
        HostRing128Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl HostRing64Tensor {
    pub fn sample_uniform_seeded(shape: &RawShape, seed: &RawSeed) -> HostRing64Tensor {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape.0.as_ref());
        HostRing64Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
    pub fn sample_bits_seeded(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u64)).collect();
        let ix = IxDyn(shape.0.as_ref());
        HostRing64Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl HostRing128Tensor {
    pub fn sample_uniform_seeded(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size)
            .map(|_| {
                let upper = rng.next_u64() as u128;
                let lower = rng.next_u64() as u128;
                Wrapping((upper << 64) + lower)
            })
            .collect();
        let ix = IxDyn(shape.0.as_ref());
        HostRing128Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }

    pub fn sample_bits_seeded(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u128)).collect();
        let ix = IxDyn(shape.0.as_ref());
        HostRing128Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl HostRing64Tensor {
    pub fn bit_extract(&self, bit_idx: usize) -> HostBitTensor {
        let temp = &self.0 >> bit_idx;
        let lsb = temp.mapv(|ai| (ai.0 & 1) as u8);
        HostBitTensor::from(lsb)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl HostRing128Tensor {
    pub fn bit_extract(&self, bit_idx: usize) -> HostBitTensor {
        let temp = &self.0 >> bit_idx;
        let lsb = temp.mapv(|ai| (ai.0 & 1) as u8);
        HostBitTensor::from(lsb)
    }
}

impl<T> AbstractHostRingTensor<T>
where
    T: Clone,
{
    pub fn from_raw_plc<D: ndarray::Dimension, P: Into<HostPlacement>>(
        raw_tensor: Array<T, D>,
        plc: P,
    ) -> AbstractHostRingTensor<T> {
        let tensor = raw_tensor.mapv(Wrapping).into_dyn();
        AbstractHostRingTensor(tensor, plc.into())
    }
}

// This implementation is only used by the old kernels. Construct AbstractHostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> AbstractHostRingTensor<T>
where
    Wrapping<T>: Clone,
{
    #[cfg_attr(
        feature = "exclude_old_framework",
        deprecated(
            note = "This function is only used by the old kernels, which are not aware of the placements."
        )
    )]
    pub fn fill(shape: &RawShape, el: T) -> AbstractHostRingTensor<T> {
        AbstractHostRingTensor(
            ArrayD::from_elem(shape.0.as_ref(), Wrapping(el)),
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

impl<T> AbstractHostRingTensor<T> {
    pub(crate) fn shape(&self) -> HostShape {
        HostShape(RawShape(self.0.shape().into()), self.1.clone())
    }
}

// This implementation is only used by the old kernels. Construct AbstractHostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<ArrayD<T>> for AbstractHostRingTensor<T>
where
    T: Clone,
{
    fn from(a: ArrayD<T>) -> AbstractHostRingTensor<T> {
        let wrapped = a.mapv(Wrapping);
        AbstractHostRingTensor(
            wrapped,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct AbstractHostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl From<ArrayD<i64>> for AbstractHostRingTensor<u64> {
    fn from(a: ArrayD<i64>) -> AbstractHostRingTensor<u64> {
        let ring_rep = a.mapv(|ai| Wrapping(ai as u64));
        AbstractHostRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct AbstractHostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl From<ArrayD<i128>> for AbstractHostRingTensor<u128> {
    fn from(a: ArrayD<i128>) -> AbstractHostRingTensor<u128> {
        let ring_rep = a.mapv(|ai| Wrapping(ai as u128));
        AbstractHostRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> AbstractHostRingTensor<T> {
    pub fn new(a: ArrayD<Wrapping<T>>) -> AbstractHostRingTensor<T> {
        AbstractHostRingTensor(
            a,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct AbstractHostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<HostBitTensor> for AbstractHostRingTensor<T>
where
    T: From<u8>,
{
    fn from(b: HostBitTensor) -> AbstractHostRingTensor<T> {
        let ring_rep = b.0.mapv(|ai| Wrapping(ai.into()));
        AbstractHostRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

impl From<&AbstractHostRingTensor<u64>> for ArrayD<i64> {
    fn from(r: &AbstractHostRingTensor<u64>) -> ArrayD<i64> {
        r.0.mapv(|element| element.0 as i64)
    }
}

impl From<&AbstractHostRingTensor<u128>> for ArrayD<i128> {
    fn from(r: &AbstractHostRingTensor<u128>) -> ArrayD<i128> {
        r.0.mapv(|element| element.0 as i128)
    }
}

// This implementation is only used by the old kernels. Construct AbstractHostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<Vec<T>> for AbstractHostRingTensor<T> {
    fn from(v: Vec<T>) -> AbstractHostRingTensor<T> {
        let ix = IxDyn(&[v.len()]);
        use vec_utils::VecExt;
        let v_wrapped: Vec<_> = v.map(Wrapping);
        AbstractHostRingTensor(
            Array::from_shape_vec(ix, v_wrapped).unwrap(),
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct AbstractHostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<&[T]> for AbstractHostRingTensor<T>
where
    T: Copy,
{
    fn from(v: &[T]) -> AbstractHostRingTensor<T> {
        let ix = IxDyn(&[v.len()]);
        let v_wrapped: Vec<_> = v.iter().map(|vi| Wrapping(*vi)).collect();
        AbstractHostRingTensor(
            Array::from_shape_vec(ix, v_wrapped).unwrap(),
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> Add<AbstractHostRingTensor<T>> for AbstractHostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Add<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = AbstractHostRingTensor<T>;
    fn add(self, other: AbstractHostRingTensor<T>) -> Self::Output {
        AbstractHostRingTensor(self.0 + other.0, self.1)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> Mul<AbstractHostRingTensor<T>> for AbstractHostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Mul<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = AbstractHostRingTensor<T>;
    fn mul(self, other: AbstractHostRingTensor<T>) -> Self::Output {
        AbstractHostRingTensor(self.0 * other.0, self.1)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> Sub<AbstractHostRingTensor<T>> for AbstractHostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Sub<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = AbstractHostRingTensor<T>;
    fn sub(self, other: AbstractHostRingTensor<T>) -> Self::Output {
        AbstractHostRingTensor(self.0 - other.0, self.1)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> Shl<usize> for AbstractHostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Shl<usize, Output = Wrapping<T>>,
{
    type Output = AbstractHostRingTensor<T>;
    fn shl(self, other: usize) -> Self::Output {
        AbstractHostRingTensor(self.0 << other, self.1)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> Shr<usize> for AbstractHostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Shr<usize, Output = Wrapping<T>>,
{
    type Output = AbstractHostRingTensor<T>;
    fn shr(self, other: usize) -> Self::Output {
        AbstractHostRingTensor(self.0 >> other, self.1)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> BitAnd<AbstractHostRingTensor<T>> for AbstractHostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: BitAnd<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = AbstractHostRingTensor<T>;
    fn bitand(self, other: AbstractHostRingTensor<T>) -> Self::Output {
        AbstractHostRingTensor(self.0 & other.0, self.1)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> AbstractHostRingTensor<T>
where
    Wrapping<T>: LinalgScalar,
{
    pub fn dot(self, rhs: AbstractHostRingTensor<T>) -> AbstractHostRingTensor<T> {
        match self.0.ndim() {
            1 => match rhs.0.ndim() {
                1 => {
                    let l = self.0.into_dimensionality::<Ix1>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix1>().unwrap();
                    let res = Array::from_elem([], l.dot(&r))
                        .into_dimensionality::<IxDyn>()
                        .unwrap();
                    AbstractHostRingTensor(res, self.1)
                }
                2 => {
                    let l = self.0.into_dimensionality::<Ix1>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix2>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    AbstractHostRingTensor(res, self.1)
                }
                other => panic!(
                    "Dot<AbstractHostRingTensor> cannot handle argument of rank {:?} ",
                    other
                ),
            },
            2 => match rhs.0.ndim() {
                1 => {
                    let l = self.0.into_dimensionality::<Ix2>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix1>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    AbstractHostRingTensor(res, self.1)
                }
                2 => {
                    let l = self.0.into_dimensionality::<Ix2>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix2>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    AbstractHostRingTensor(res, self.1)
                }
                other => panic!(
                    "Dot<AbstractHostRingTensor> cannot handle argument of rank {:?} ",
                    other
                ),
            },
            other => panic!(
                "Dot<AbstractHostRingTensor> not implemented for tensors of rank {:?}",
                other
            ),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> AbstractHostRingTensor<T>
where
    Wrapping<T>: Clone + Zero,
{
    pub fn sum(self, axis: Option<usize>) -> AbstractHostRingTensor<T> {
        if let Some(i) = axis {
            AbstractHostRingTensor(self.0.sum_axis(Axis(i)), self.1)
        } else {
            let out = Array::from_elem([], self.0.sum())
                .into_dimensionality::<IxDyn>()
                .unwrap();
            AbstractHostRingTensor(out, self.1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_prod_f32() {
        let x = HostTensor::<f32>::from(
            array![[1.0, -2.0], [3.0, -4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let y = x.clone();
        let z = x.dot(y);
        assert_eq!(
            z,
            HostTensor::<f32>::from(
                array![[-5.0, 6.0], [-9.0, 10.0]]
                    .into_dimensionality::<IxDyn>()
                    .unwrap()
            )
        );
    }

    #[test]
    fn test_inverse() {
        let x = HostTensor::<f32>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );

        let x_inv = x.inv();

        assert_eq!(
            x_inv,
            HostTensor::<f32>::from(
                array![[-2.0, 1.0], [1.5, -0.5]]
                    .into_dimensionality::<IxDyn>()
                    .unwrap()
            )
        );
    }

    #[test]
    fn test_shape_slice() {
        let x_shape = RawShape(vec![1, 2, 3]);
        let x_slice = x_shape.slice(1, 3);
        assert_eq!(x_slice, RawShape(vec![2, 3]))
    }

    #[test]
    fn test_tensor_slice() {
        let x_backing: ArrayD<u64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let x = HostRing64Tensor::from_raw_plc(x_backing, alice.clone());

        let slice = SliceInfo(vec![
            SliceInfoElem {
                start: 1,
                end: None,
                step: None,
            },
            SliceInfoElem {
                start: 0,
                end: None,
                step: None,
            },
        ]);

        let sess = SyncSession::default();
        let y = alice.slice(&sess, slice, &x);

        let target: ArrayD<u64> = array![[3, 4]].into_dimensionality::<IxDyn>().unwrap();

        assert_eq!(y, HostRing64Tensor::from_raw_plc(target, alice))
    }

    #[test]
    fn test_diag() {
        let x_backing: ArrayD<f64> = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let alice = HostPlacement {
            owner: "alice".into(),
        };

        let x = HostTensor::<f64>(x_backing, alice.clone());
        let sess = SyncSession::default();
        let y = alice.diag(&sess, &x);
        let target: ArrayD<f64> = array![1.0, 4.0].into_dimensionality::<IxDyn>().unwrap();
        assert_eq!(y, HostTensor::<f64>(target, alice))
    }

    #[test]
    fn test_index() {
        let x_backing: ArrayD<u64> = array![[[1, 2], [3, 4]], [[4, 5], [6, 7]]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let x = HostRing64Tensor::from_raw_plc(x_backing, alice.clone());
        let sess = SyncSession::default();
        let y = alice.index_axis(&sess, 0, 1, &x);

        let target: ArrayD<u64> = array![[4, 5], [6, 7]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        assert_eq!(y, HostRing64Tensor::from_raw_plc(target, alice))
    }

    #[test]
    fn test_transpose() {
        let x = HostTensor::<f32>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let y = x.transpose();
        assert_eq!(
            y,
            HostTensor::<f32>::from(
                array![[1.0, 3.0], [2.0, 4.0]]
                    .into_dimensionality::<IxDyn>()
                    .unwrap()
            )
        );
    }

    #[test]
    fn test_concatenate() {
        let a = HostTensor::<f32>::from(
            array![[[1.0, 2.0], [3.0, 4.0]]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let b = HostTensor::<f32>::from(
            array![[[1.0, 2.0], [3.0, 4.0]]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let expected = HostTensor::<f32>::from(
            array![[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let conc = concatenate(0, &vec![a, b]);
        assert_eq!(conc, expected)
    }

    #[test]
    fn test_atleast_2d() {
        let a = HostTensor::<f32>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let a_exp = a.clone();
        let b = HostTensor::<f32>::from(
            array![1.0, 2.0, 3.0, 4.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let b_exp = HostTensor::<f32>::from(
            array![[1.0, 2.0, 3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let c = HostTensor::<f32>::from(
            array![1.0, 2.0, 3.0, 4.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let c_exp = HostTensor::<f32>::from(
            array![[1.0], [2.0], [3.0], [4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let d = HostTensor::<f32>::from(
            Array::from_elem([], 1.0)
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let d_exp = HostTensor::<f32>::from(array![[1.0]].into_dimensionality::<IxDyn>().unwrap());
        let ax = a.atleast_2d(true);
        let bx = b.atleast_2d(false);
        let cx = c.atleast_2d(true);
        let dx = d.atleast_2d(true);
        assert_eq!(ax, a_exp);
        assert_eq!(bx, b_exp);
        assert_eq!(cx, c_exp);
        assert_eq!(dx, d_exp);
    }

    #[test]
    fn test_add_broadcasting() {
        let x_1 = HostTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_1 = HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.add(y_1);
        let z_1_exp =
            HostTensor::<f32>::from(array![3.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let x_2 = HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_2 = HostTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_2 = x_2.add(y_2);
        let z_2_exp =
            HostTensor::<f32>::from(array![3.0, 4.0].into_dimensionality::<IxDyn>().unwrap());

        assert_eq!(z_1, z_1_exp);
        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_sub_broadcasting() {
        let x_1 = HostTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_1 = HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.sub(y_1);
        let z_1_exp =
            HostTensor::<f32>::from(array![1.0, 0.0].into_dimensionality::<IxDyn>().unwrap());
        let x_2 = HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_2 = HostTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_2 = x_2.sub(y_2);
        let z_2_exp =
            HostTensor::<f32>::from(array![-1.0, 0.0].into_dimensionality::<IxDyn>().unwrap());

        assert_eq!(z_1, z_1_exp);
        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_mul_broadcasting() {
        let x_1 = HostTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_1 = HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.mul(y_1);
        let z_1_exp =
            HostTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let x_2 = HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_2 = HostTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_2 = x_2.mul(y_2);
        let z_2_exp =
            HostTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());

        assert_eq!(z_1, z_1_exp);
        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_div_broadcasting() {
        let x_1 = HostTensor::<f32>::from(array![1.0].into_dimensionality::<IxDyn>().unwrap());
        let y_1 = HostTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.div(y_1);
        let z_1_exp =
            HostTensor::<f32>::from(array![0.5, 0.25].into_dimensionality::<IxDyn>().unwrap());
        let x_2 = HostTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let y_2 = HostTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_2 = x_2.div(y_2);
        let z_2_exp =
            HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());

        assert_eq!(z_1, z_1_exp);
        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_kernel_inverse() {
        use crate::kernels::PlacementInverse;
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let sess = SyncSession::default();
        let x = crate::host::HostTensor::<f64>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let inv = alice.inverse(&sess, &x);
        assert_eq!("[[-2, 1],\n [1.5, -0.5]]", format!("{}", inv.0));
    }

    #[test]
    fn test_kernel_sqrt() {
        use crate::kernels::PlacementSqrt;
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let sess = SyncSession::default();
        let x = crate::host::HostTensor::<f64>::from(
            array![[4.0, 9.0], [16.0, 25.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let exp = crate::host::HostTensor::<f64>(
            array![[2.0, 3.0], [4.0, 5.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            alice.clone(),
        );
        let sqrt = alice.sqrt(&sess, &x);
        assert_eq!(exp, sqrt)
    }

    use rstest::rstest;
    #[rstest]
    #[case(None)]
    #[case(Some(2))]
    fn test_kernel_squeeze(#[case] axis: Option<u32>) {
        use crate::kernels::PlacementSqueeze;
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let sess = SyncSession::default();
        let x = crate::host::HostTensor::<f64>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let x_expanded = x.expand_dims(vec![2]);
        let exp_shape = RawShape(vec![2, 2]);

        let x_squeezed = alice.squeeze(&sess, axis, &x_expanded);

        assert_eq!(exp_shape, x_squeezed.shape().0)
    }

    #[test]
    fn test_kernel_transpose() {
        use crate::kernels::PlacementTranspose;
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let sess = SyncSession::default();
        let x = crate::host::HostTensor::<f64>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let t = alice.transpose(&sess, &x);
        assert_eq!("[[1, 3],\n [2, 4]]", format!("{}", t.0));
    }

    #[test]
    fn test_kernel_concatenate() {
        use crate::kernels::PlacementConcatenate;
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let sess = SyncSession::default();
        let x = crate::host::HostTensor::<f64>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let y = crate::host::HostTensor::<f64>::from(
            array![[5.0, 6.0], [7.0, 8.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let c = alice.concatenate(&sess, 0, &x, &y);
        assert_eq!("[[1, 2],\n [3, 4],\n [5, 6],\n [7, 8]]", format!("{}", c.0));
    }

    #[test]
    fn bit_sample() {
        let shape = RawShape(vec![5]);
        let seed = RawSeed([0u8; 16]);
        let r = HostBitTensor::sample_uniform_seeded(&shape, &seed);
        assert_eq!(r, HostBitTensor::from(vec![0, 1, 1, 0, 0,]));
    }

    #[test]
    fn bit_fill() {
        let shape = RawShape(vec![2]);
        let r = HostBitTensor::fill(&shape, 1);
        assert_eq!(r, HostBitTensor::from(vec![1, 1]))
    }

    #[test]
    fn bit_ops() {
        let shape = RawShape(vec![5]);

        // test xor
        assert_eq!(
            HostBitTensor::fill(&shape, 0) ^ HostBitTensor::fill(&shape, 1),
            HostBitTensor::fill(&shape, 1)
        );
        assert_eq!(
            HostBitTensor::fill(&shape, 1) ^ HostBitTensor::fill(&shape, 0),
            HostBitTensor::fill(&shape, 1)
        );
        assert_eq!(
            HostBitTensor::fill(&shape, 1) ^ HostBitTensor::fill(&shape, 1),
            HostBitTensor::fill(&shape, 0)
        );
        assert_eq!(
            HostBitTensor::fill(&shape, 0) ^ HostBitTensor::fill(&shape, 0),
            HostBitTensor::fill(&shape, 0)
        );

        // test and
        assert_eq!(
            HostBitTensor::fill(&shape, 0) & HostBitTensor::fill(&shape, 1),
            HostBitTensor::fill(&shape, 0)
        );
        assert_eq!(
            HostBitTensor::fill(&shape, 1) & HostBitTensor::fill(&shape, 0),
            HostBitTensor::fill(&shape, 0)
        );
        assert_eq!(
            HostBitTensor::fill(&shape, 1) & HostBitTensor::fill(&shape, 1),
            HostBitTensor::fill(&shape, 1)
        );
        assert_eq!(
            HostBitTensor::fill(&shape, 0) & HostBitTensor::fill(&shape, 0),
            HostBitTensor::fill(&shape, 0)
        );
    }

    #[test]
    fn ring_matrix_vector_prod() {
        let array_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = HostRing64Tensor::from(array_backing);
        let y = HostRing64Tensor::from(vec![1, 1]);
        let z = x.dot(y);

        let result = HostRing64Tensor::from(vec![3, 7]);
        assert_eq!(result, z)
    }

    #[test]
    fn ring_matrix_matrix_prod() {
        let x_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y_backing: ArrayD<i64> = array![[1, 0], [0, 1]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = HostRing64Tensor::from(x_backing);
        let y = HostRing64Tensor::from(y_backing);
        let z = x.dot(y);

        let r_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let result = HostRing64Tensor::from(r_backing);
        assert_eq!(result, z)
    }

    #[test]
    fn ring_vector_prod() {
        let x_backing = vec![1, 2];
        let y_backing = vec![1, 1];
        let x = HostRing64Tensor::from(x_backing);
        let y = HostRing64Tensor::from(y_backing);
        let z = x.dot(y);

        let r_backing = Array::from_elem([], Wrapping(3))
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let result = HostRing64Tensor::new(r_backing);
        assert_eq!(result, z)
    }

    #[test]
    fn ring_sample() {
        let shape = RawShape(vec![5]);
        let seed = RawSeed([0u8; 16]);
        let r = HostRing64Tensor::sample_uniform_seeded(&shape, &seed);
        assert_eq!(
            r,
            HostRing64Tensor::from(vec![
                4263935709876578662,
                3326810793440857224,
                17325099178452873543,
                15208531650305571673,
                9619880027406922172
            ])
        );

        let r128 = HostRing128Tensor::sample_uniform_seeded(&shape, &seed);
        assert_eq!(
            r128,
            HostRing128Tensor::from(vec![
                78655730786844307471556614669614075016,
                319591670596555766473793801091584867161,
                177455464885365520564027128957528354027,
                72628979995024532377123578937486303732,
                299726520301515014350190124791858941972
            ])
        );

        let r_bits = HostRing64Tensor::sample_bits_seeded(&shape, &seed);
        assert_eq!(r_bits, HostRing64Tensor::from(vec![0, 1, 1, 0, 0]));

        let r128_bits = HostRing128Tensor::sample_bits_seeded(&shape, &seed);
        assert_eq!(r128_bits, HostRing128Tensor::from(vec![0, 1, 1, 0, 0]));
    }

    #[test]
    fn ring_fill() {
        let r = HostRing64Tensor::fill(&RawShape(vec![2]), 1);
        assert_eq!(r, HostRing64Tensor::from(vec![1, 1]))
    }

    #[test]
    fn ring_sum_with_axis() {
        let x_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = HostRing64Tensor::from(x_backing);
        let out = x.sum(Some(0));
        assert_eq!(out, HostRing64Tensor::from(vec![4, 6]))
    }

    #[test]
    fn ring_sum_without_axis() {
        let x_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = HostRing64Tensor::from(x_backing);
        let exp_v: u64 = 10;
        let exp_backing = Array::from_elem([], exp_v)
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let exp = HostRing64Tensor::from(exp_backing);
        let out = x.sum(None);
        assert_eq!(out, exp)
    }

    #[test]
    fn bit_extract() {
        let shape = RawShape(vec![5]);
        let value = 7;

        let r0 = HostRing64Tensor::fill(&shape, value).bit_extract(0);
        assert_eq!(HostBitTensor::fill(&shape, 1), r0,);

        let r1 = HostRing64Tensor::fill(&shape, value).bit_extract(1);
        assert_eq!(HostBitTensor::fill(&shape, 1), r1,);

        let r2 = HostRing64Tensor::fill(&shape, value).bit_extract(2);
        assert_eq!(HostBitTensor::fill(&shape, 1), r2,);

        let r3 = HostRing64Tensor::fill(&shape, value).bit_extract(3);
        assert_eq!(HostBitTensor::fill(&shape, 0), r3,)
    }

    #[test]
    fn test_bit_dec() {
        let x_backing: ArrayD<u64> = array![[[1, 2], [3, 4]], [[4, 5], [6, 7]]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let x = HostRing64Tensor::from_raw_plc(x_backing, alice.clone());
        let sess = SyncSession::default();
        let x_bits: HostBitTensor = alice.bit_decompose(&sess, &x);
        let targets: Vec<_> = (0..64).map(|i| alice.bit_extract(&sess, i, &x)).collect();

        for (i, target) in targets.iter().enumerate() {
            let sliced = alice.index_axis(&sess, 0, i, &x_bits);
            assert_eq!(&sliced, target);
        }
    }
}
