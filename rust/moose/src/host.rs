use crate::computation::{
    BitAndOp, BitExtractOp, BitFillOp, BitSampleOp, BitXorOp, Constant, HostAddOp,
    HostConcatenateOp, HostDivOp, HostDotOp, HostExpandDimsOp, HostInverseOp, HostMeanOp,
    HostMulOp, HostOnesOp, HostPlacement, HostSliceOp, HostSubOp, HostSumOp, HostTransposeOp,
    Placed, Placement, RingAddOp, RingDotOp, RingFillOp, RingMulOp, RingNegOp, RingSampleOp,
    RingShlOp, RingShrOp, RingSubOp, RingSumOp, Role, ShapeOp,
};
use crate::error::Result;
use crate::kernels::{
    PlacementAdd, PlacementAnd, PlacementBitExtract, PlacementDot, PlacementFill, PlacementMul,
    PlacementNeg, PlacementPlace, PlacementSample, PlacementSampleUniform, PlacementShl,
    PlacementShr, PlacementSlice, PlacementSub, PlacementSum, PlacementXor, RuntimeSession,
    Session, SyncSession, Tensor,
};
use crate::prim::{RawSeed, Seed};
use crate::prng::AesRng;
use crate::symbolic::{Symbolic, SymbolicHandle, SymbolicSession};
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use ndarray_linalg::types::{Lapack, Scalar};
use ndarray_linalg::*;
use num_traits::FromPrimitive;
use num_traits::Zero;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::fmt::Debug;
use std::num::Wrapping;
use std::ops::{Add, Div, Mul, Sub}; // related to TODOs
use std::ops::{BitAnd, BitXor, Neg, Shl, Shr};

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
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct HostShape(pub RawShape, pub HostPlacement);

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

impl HostSubOp {
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

impl HostDivOp {
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

modelled!(PlacementSlice::slice, HostPlacement, attributes[start: u32, end: u32] (HostShape) -> HostShape, HostSliceOp);

kernel! {
    HostSliceOp,
    [
        (HostPlacement, (HostShape) -> HostShape => attributes[start, end] Self::kernel),
    ]
}

impl HostSliceOp {
    pub(crate) fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        start: u32,
        end: u32,
        x: HostShape,
    ) -> HostShape {
        let slice = x.0.slice(start as usize, end as usize);
        HostShape(slice, plc.clone())
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

impl HostConcatenateOp {
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
#[cfg(not(feature = "symbolic"))]
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
#[cfg(not(feature = "symbolic"))]
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
#[cfg(not(feature = "symbolic"))]
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
#[cfg(not(feature = "symbolic"))]
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

modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> HostBitTensor, BitFillOp);

kernel! {
    BitFillOp,
    [
        (HostPlacement, (HostShape) -> HostBitTensor => attributes[value: Bit] Self::kernel),
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

modelled!(PlacementSampleUniform::sample_uniform, HostPlacement, (HostShape, Seed) -> HostBitTensor, BitSampleOp);

kernel! {
    BitSampleOp,
    [
        (HostPlacement, (HostShape, Seed) -> HostBitTensor => Self::kernel),
    ]
}

impl BitSampleOp {
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
        (HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => Self::kernel),
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
modelled!(PlacementAnd::and, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, BitAndOp);
modelled!(PlacementAnd::and, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, BitAndOp);

modelled_alias!(PlacementMul::mul, HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => PlacementAnd::and); // mul = and in Z2

kernel! {
    BitAndOp,
    [
        (HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => Self::bit_kernel),
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor => Self::ring_kernel),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor => Self::ring_kernel),
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
        x: AbstractRingTensor<T>,
        y: AbstractRingTensor<T>,
    ) -> AbstractRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: BitAnd<Wrapping<T>, Output = Wrapping<T>>,
    {
        AbstractRingTensor(x.0 & y.0, plc.clone())
    }
}

impl HostBitTensor {
    #[cfg_attr(
        feature = "symbolic",
        deprecated(
            note = "This function is only used by the old kernels, which are not aware of the placements. See BitSampleOp::kernel for the new code"
        )
    )]
    pub fn sample_uniform(shape: &RawShape, seed: &RawSeed) -> Self {
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
        feature = "symbolic",
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
#[cfg(not(feature = "symbolic"))]
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
#[cfg(not(feature = "symbolic"))]
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
#[cfg(not(feature = "symbolic"))]
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

modelled!(PlacementBitExtract::bit_extract, HostPlacement, attributes[bit_idx: usize] (Ring64Tensor) -> HostBitTensor, BitExtractOp);
modelled!(PlacementBitExtract::bit_extract, HostPlacement, attributes[bit_idx: usize] (Ring128Tensor) -> HostBitTensor, BitExtractOp);

kernel! {
    BitExtractOp,
    [
        (HostPlacement, (Ring64Tensor) -> HostBitTensor => attributes[bit_idx] Self::kernel64),
        (HostPlacement, (Ring128Tensor) -> HostBitTensor => attributes[bit_idx] Self::kernel128),
    ]
}

impl BitExtractOp {
    fn kernel64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        bit_idx: usize,
        x: Ring64Tensor,
    ) -> HostBitTensor {
        HostBitTensor((x >> bit_idx).0.mapv(|ai| (ai.0 & 1) as u8), plc.clone())
    }
    fn kernel128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        bit_idx: usize,
        x: Ring128Tensor,
    ) -> HostBitTensor {
        HostBitTensor((x >> bit_idx).0.mapv(|ai| (ai.0 & 1) as u8), plc.clone())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AbstractRingTensor<T>(pub ArrayD<Wrapping<T>>, pub HostPlacement);

/// Tensor for ring arithmetic over Z_{2^64}
pub type Ring64Tensor = AbstractRingTensor<u64>;

/// Tensor for ring arithmetic over Z_{2^128}
pub type Ring128Tensor = AbstractRingTensor<u128>;

impl<S: Session, T> Tensor<S> for AbstractRingTensor<T> {
    type Scalar = T;
}

impl<S: Session, T> Tensor<S> for Symbolic<AbstractRingTensor<T>> {
    type Scalar = T;
}

impl<T> Placed for AbstractRingTensor<T> {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

pub trait RingSize {
    const SIZE: usize;
}

impl RingSize for Ring64Tensor {
    const SIZE: usize = 64;
}

impl RingSize for Ring128Tensor {
    const SIZE: usize = 128;
}

pub trait FromRawPlc<P, T> {
    fn from_raw_plc(raw_tensor: ArrayD<T>, plc: P) -> AbstractRingTensor<T>;
}

impl<P> FromRawPlc<P, u64> for Ring64Tensor
where
    P: Into<HostPlacement>,
{
    fn from_raw_plc(raw_tensor: ArrayD<u64>, plc: P) -> Ring64Tensor {
        let tensor = raw_tensor.mapv(Wrapping).into_dyn();
        AbstractRingTensor(tensor, plc.into())
    }
}

impl<P> FromRawPlc<P, u128> for Ring128Tensor
where
    P: Into<HostPlacement>,
{
    fn from_raw_plc(raw_tensor: ArrayD<u128>, plc: P) -> Ring128Tensor {
        let tensor = raw_tensor.mapv(Wrapping).into_dyn();
        AbstractRingTensor(tensor, plc.into())
    }
}

impl<R: RingSize + Placed> RingSize for Symbolic<R> {
    const SIZE: usize = R::SIZE;
}

impl<T> PlacementPlace<SyncSession, AbstractRingTensor<T>> for HostPlacement
where
    AbstractRingTensor<T>: Placed<Placement = HostPlacement>,
{
    fn place(&self, _sess: &SyncSession, x: AbstractRingTensor<T>) -> AbstractRingTensor<T> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                AbstractRingTensor(x.0, self.clone())
            }
        }
    }
}

impl<T> PlacementPlace<SymbolicSession, Symbolic<AbstractRingTensor<T>>> for HostPlacement {
    fn place(
        &self,
        _sess: &SymbolicSession,
        x: Symbolic<AbstractRingTensor<T>>,
    ) -> Symbolic<AbstractRingTensor<T>> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                match x {
                    Symbolic::Concrete(x) => {
                        // TODO insert Place ops?
                        Symbolic::Concrete(AbstractRingTensor(x.0, self.clone()))
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

modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> Ring64Tensor, RingFillOp);
modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> Ring128Tensor, RingFillOp);

kernel! {
    RingFillOp,
    [
        (HostPlacement, (HostShape) -> Ring64Tensor => attributes[value: Ring64] Self::ring64_kernel),
        (HostPlacement, (HostShape) -> Ring128Tensor => attributes[value: Ring128] Self::ring128_kernel),
    ]
}

impl RingFillOp {
    fn ring64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u64,
        shape: HostShape,
    ) -> Ring64Tensor {
        let raw_shape = shape.0 .0;
        let raw_tensor = ArrayD::from_elem(raw_shape.as_ref(), Wrapping(value));
        AbstractRingTensor(raw_tensor, plc.clone())
    }

    fn ring128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u128,
        shape: HostShape,
    ) -> Ring128Tensor {
        let raw_shape = shape.0 .0;
        let raw_tensor = ArrayD::from_elem(raw_shape.as_ref(), Wrapping(value));
        AbstractRingTensor(raw_tensor, plc.clone())
    }
}

impl ShapeOp {
    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractRingTensor<T>,
    ) -> HostShape {
        let raw_shape = RawShape(x.0.shape().into());
        HostShape(raw_shape, plc.clone())
    }
}

modelled!(PlacementAdd::add, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingAddOp);
modelled!(PlacementAdd::add, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingAddOp);

kernel! {
    RingAddOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor => Self::kernel),
    ]
}

impl RingAddOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractRingTensor<T>,
        y: AbstractRingTensor<T>,
    ) -> AbstractRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Add<Wrapping<T>, Output = Wrapping<T>>,
    {
        AbstractRingTensor(x.0 + y.0, plc.clone())
    }
}

modelled!(PlacementSub::sub, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingSubOp);
modelled!(PlacementSub::sub, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingSubOp);

kernel! {
    RingSubOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor => Self::kernel),
    ]
}

impl RingSubOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractRingTensor<T>,
        y: AbstractRingTensor<T>,
    ) -> AbstractRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Sub<Wrapping<T>, Output = Wrapping<T>>,
    {
        AbstractRingTensor(x.0 - y.0, plc.clone())
    }
}

modelled!(PlacementNeg::neg, HostPlacement, (Ring64Tensor) -> Ring64Tensor, RingNegOp);
modelled!(PlacementNeg::neg, HostPlacement, (Ring128Tensor) -> Ring128Tensor, RingNegOp);

kernel! {
    RingNegOp,
    [
        (HostPlacement, (Ring64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Ring128Tensor) -> Ring128Tensor => Self::kernel),
    ]
}

impl RingNegOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractRingTensor<T>,
    ) -> AbstractRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Neg<Output = Wrapping<T>>,
    {
        AbstractRingTensor(x.0.neg(), plc.clone())
    }
}

modelled!(PlacementMul::mul, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingMulOp);
modelled!(PlacementMul::mul, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingMulOp);

kernel! {
    RingMulOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor => Self::kernel),
    ]
}

impl RingMulOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractRingTensor<T>,
        y: AbstractRingTensor<T>,
    ) -> AbstractRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Mul<Wrapping<T>, Output = Wrapping<T>>,
    {
        AbstractRingTensor(x.0 * y.0, plc.clone())
    }
}

modelled!(PlacementDot::dot, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingDotOp);
modelled!(PlacementDot::dot, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingDotOp);

kernel! {
    RingDotOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor => Self::kernel),
    ]
}

impl RingDotOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: AbstractRingTensor<T>,
        y: AbstractRingTensor<T>,
    ) -> AbstractRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Mul<Wrapping<T>, Output = Wrapping<T>>,
        Wrapping<T>: LinalgScalar,
    {
        AbstractRingTensor(x.dot(y).0, plc.clone())
    }
}

modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (Ring64Tensor) -> Ring64Tensor, RingSumOp);
modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (Ring128Tensor) -> Ring128Tensor, RingSumOp);

kernel! {
    RingSumOp,
    [
        (HostPlacement, (Ring64Tensor) -> Ring64Tensor => attributes[axis] Self::kernel),
        (HostPlacement, (Ring128Tensor) -> Ring128Tensor => attributes[axis] Self::kernel),
    ]
}

impl RingSumOp {
    fn kernel<S: RuntimeSession, T>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: AbstractRingTensor<T>,
    ) -> AbstractRingTensor<T>
    where
        T: FromPrimitive + Zero,
        Wrapping<T>: Clone,
        Wrapping<T>: Add<Output = Wrapping<T>>,
        HostPlacement: PlacementPlace<S, AbstractRingTensor<T>>,
    {
        let sum = x.sum(axis.map(|a| a as usize));
        plc.place(sess, sum)
    }
}

modelled!(PlacementShl::shl, HostPlacement, attributes[amount: usize] (Ring64Tensor) -> Ring64Tensor, RingShlOp);
modelled!(PlacementShl::shl, HostPlacement, attributes[amount: usize] (Ring128Tensor) -> Ring128Tensor, RingShlOp);

kernel! {
    RingShlOp,
    [
        (HostPlacement, (Ring64Tensor) -> Ring64Tensor => attributes[amount] Self::kernel),
        (HostPlacement, (Ring128Tensor) -> Ring128Tensor => attributes[amount] Self::kernel),
    ]
}

impl RingShlOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        amount: usize,
        x: AbstractRingTensor<T>,
    ) -> AbstractRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Shl<usize, Output = Wrapping<T>>,
    {
        AbstractRingTensor(x.0 << amount, plc.clone())
    }
}

modelled!(PlacementShr::shr, HostPlacement, attributes[amount: usize] (Ring64Tensor) -> Ring64Tensor, RingShrOp);
modelled!(PlacementShr::shr, HostPlacement, attributes[amount: usize] (Ring128Tensor) -> Ring128Tensor, RingShrOp);

kernel! {
    RingShrOp,
    [
        (HostPlacement, (Ring64Tensor) -> Ring64Tensor => attributes[amount] Self::kernel),
        (HostPlacement, (Ring128Tensor) -> Ring128Tensor => attributes[amount] Self::kernel),
    ]
}

impl RingShrOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        amount: usize,
        x: AbstractRingTensor<T>,
    ) -> AbstractRingTensor<T>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: Shr<usize, Output = Wrapping<T>>,
    {
        AbstractRingTensor(x.0 >> amount, plc.clone())
    }
}

modelled!(PlacementSample::sample, HostPlacement, attributes[max_value: Option<u64>] (HostShape, Seed) -> Ring64Tensor, RingSampleOp);
modelled!(PlacementSample::sample, HostPlacement, attributes[max_value: Option<u64>] (HostShape, Seed) -> Ring128Tensor, RingSampleOp);

kernel! {
    RingSampleOp,
    [
        (HostPlacement, (HostShape, Seed) -> Ring64Tensor => custom |op| {
            match op.max_value {
                None => Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_uniform_u64(ctx, plc, shape, seed)
                }),
                Some(max_value) if max_value == 1 => Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_bits_u64(ctx, plc, shape, seed)
                }),
                _ => unimplemented!(),
            }
        }),
        (HostPlacement, (HostShape, Seed) -> Ring128Tensor => custom |op| {
            match op.max_value {
                None => Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_uniform_u128(ctx, plc, shape, seed)
                }),
                Some(max_value) if max_value == 1 => Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_bits_u128(ctx, plc, shape, seed)
                }),
                _ => unimplemented!(),
            }
        }),
    ]
}

impl RingSampleOp {
    fn kernel_uniform_u64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> Ring64Tensor {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let raw_array = Array::from_shape_vec(ix, values).unwrap();
        AbstractRingTensor(raw_array, plc.clone())
    }

    fn kernel_bits_u64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> Ring64Tensor {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u64)).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        AbstractRingTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }

    fn kernel_uniform_u128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> Ring128Tensor {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size)
            .map(|_| Wrapping(((rng.next_u64() as u128) << 64) + rng.next_u64() as u128))
            .collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        AbstractRingTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }

    fn kernel_bits_u128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> Ring128Tensor {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u128)).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        AbstractRingTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }
}

impl Ring64Tensor {
    pub fn sample_uniform(shape: &RawShape, seed: &RawSeed) -> Ring64Tensor {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape.0.as_ref());
        Ring64Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
    pub fn sample_bits(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u64)).collect();
        let ix = IxDyn(shape.0.as_ref());
        Ring64Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
}

impl Ring128Tensor {
    pub fn sample_uniform(shape: &RawShape, seed: &RawSeed) -> Self {
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
        Ring128Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }

    pub fn sample_bits(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u128)).collect();
        let ix = IxDyn(shape.0.as_ref());
        Ring128Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
}

impl Ring64Tensor {
    pub fn bit_extract(&self, bit_idx: usize) -> HostBitTensor {
        let temp = &self.0 >> bit_idx;
        let lsb = temp.mapv(|ai| (ai.0 & 1) as u8);
        HostBitTensor::from(lsb)
    }
}

impl Ring128Tensor {
    pub fn bit_extract(&self, bit_idx: usize) -> HostBitTensor {
        let temp = &self.0 >> bit_idx;
        let lsb = temp.mapv(|ai| (ai.0 & 1) as u8);
        HostBitTensor::from(lsb)
    }
}

impl<T> AbstractRingTensor<T>
where
    T: Clone,
{
    pub fn from_raw_plc<D: ndarray::Dimension, P: Into<HostPlacement>>(
        raw_tensor: Array<T, D>,
        plc: P,
    ) -> AbstractRingTensor<T> {
        let tensor = raw_tensor.mapv(Wrapping).into_dyn();
        AbstractRingTensor(tensor, plc.into())
    }
}

// This implementation is only used by the old kernels. Construct AbstractRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "symbolic"))]
impl<T> AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
{
    #[cfg_attr(
        feature = "symbolic",
        deprecated(
            note = "This function is only used by the old kernels, which are not aware of the placements."
        )
    )]
    pub fn fill(shape: &RawShape, el: T) -> AbstractRingTensor<T> {
        AbstractRingTensor(
            ArrayD::from_elem(shape.0.as_ref(), Wrapping(el)),
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

impl<T> AbstractRingTensor<T> {
    pub(crate) fn shape(&self) -> HostShape {
        HostShape(RawShape(self.0.shape().into()), self.1.clone())
    }
}

// This implementation is only used by the old kernels. Construct AbstractRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "symbolic"))]
impl<T> From<ArrayD<T>> for AbstractRingTensor<T>
where
    T: Clone,
{
    fn from(a: ArrayD<T>) -> AbstractRingTensor<T> {
        let wrapped = a.mapv(Wrapping);
        AbstractRingTensor(
            wrapped,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct AbstractRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "symbolic"))]
impl From<ArrayD<i64>> for AbstractRingTensor<u64> {
    fn from(a: ArrayD<i64>) -> AbstractRingTensor<u64> {
        let ring_rep = a.mapv(|ai| Wrapping(ai as u64));
        AbstractRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct AbstractRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "symbolic"))]
impl From<ArrayD<i128>> for AbstractRingTensor<u128> {
    fn from(a: ArrayD<i128>) -> AbstractRingTensor<u128> {
        let ring_rep = a.mapv(|ai| Wrapping(ai as u128));
        AbstractRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

impl<T> AbstractRingTensor<T> {
    #[cfg_attr(
        feature = "symbolic",
        deprecated(
            note = "This function is only used by the old kernels, which are not aware of the placements."
        )
    )]
    pub fn new(a: ArrayD<Wrapping<T>>) -> AbstractRingTensor<T> {
        AbstractRingTensor(
            a,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct AbstractRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "symbolic"))]
impl<T> From<HostBitTensor> for AbstractRingTensor<T>
where
    T: From<u8>,
{
    fn from(b: HostBitTensor) -> AbstractRingTensor<T> {
        let ring_rep = b.0.mapv(|ai| Wrapping(ai.into()));
        AbstractRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

impl From<&AbstractRingTensor<u64>> for ArrayD<i64> {
    fn from(r: &AbstractRingTensor<u64>) -> ArrayD<i64> {
        r.0.mapv(|element| element.0 as i64)
    }
}

impl From<&AbstractRingTensor<u128>> for ArrayD<i128> {
    fn from(r: &AbstractRingTensor<u128>) -> ArrayD<i128> {
        r.0.mapv(|element| element.0 as i128)
    }
}

// This implementation is only used by the old kernels. Construct AbstractRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "symbolic"))]
impl<T> From<Vec<T>> for AbstractRingTensor<T> {
    fn from(v: Vec<T>) -> AbstractRingTensor<T> {
        let ix = IxDyn(&[v.len()]);
        use vec_utils::VecExt;
        let v_wrapped: Vec<_> = v.map(Wrapping);
        AbstractRingTensor(
            Array::from_shape_vec(ix, v_wrapped).unwrap(),
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct AbstractRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "symbolic"))]
impl<T> From<&[T]> for AbstractRingTensor<T>
where
    T: Copy,
{
    fn from(v: &[T]) -> AbstractRingTensor<T> {
        let ix = IxDyn(&[v.len()]);
        let v_wrapped: Vec<_> = v.iter().map(|vi| Wrapping(*vi)).collect();
        AbstractRingTensor(
            Array::from_shape_vec(ix, v_wrapped).unwrap(),
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

impl<T> Add<AbstractRingTensor<T>> for AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Add<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = AbstractRingTensor<T>;
    fn add(self, other: AbstractRingTensor<T>) -> Self::Output {
        AbstractRingTensor(self.0 + other.0, self.1)
    }
}

impl<T> Mul<AbstractRingTensor<T>> for AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Mul<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = AbstractRingTensor<T>;
    fn mul(self, other: AbstractRingTensor<T>) -> Self::Output {
        AbstractRingTensor(self.0 * other.0, self.1)
    }
}

impl<T> Sub<AbstractRingTensor<T>> for AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Sub<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = AbstractRingTensor<T>;
    fn sub(self, other: AbstractRingTensor<T>) -> Self::Output {
        AbstractRingTensor(self.0 - other.0, self.1)
    }
}

impl<T> Shl<usize> for AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Shl<usize, Output = Wrapping<T>>,
{
    type Output = AbstractRingTensor<T>;
    fn shl(self, other: usize) -> Self::Output {
        AbstractRingTensor(self.0 << other, self.1)
    }
}

impl<T> Shr<usize> for AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Shr<usize, Output = Wrapping<T>>,
{
    type Output = AbstractRingTensor<T>;
    fn shr(self, other: usize) -> Self::Output {
        AbstractRingTensor(self.0 >> other, self.1)
    }
}

impl<T> BitAnd<AbstractRingTensor<T>> for AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: BitAnd<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = AbstractRingTensor<T>;
    fn bitand(self, other: AbstractRingTensor<T>) -> Self::Output {
        AbstractRingTensor(self.0 & other.0, self.1)
    }
}

impl<T> AbstractRingTensor<T>
where
    Wrapping<T>: LinalgScalar,
{
    pub fn dot(self, rhs: AbstractRingTensor<T>) -> AbstractRingTensor<T> {
        match self.0.ndim() {
            1 => match rhs.0.ndim() {
                1 => {
                    let l = self.0.into_dimensionality::<Ix1>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix1>().unwrap();
                    let res = Array::from_elem([], l.dot(&r))
                        .into_dimensionality::<IxDyn>()
                        .unwrap();
                    AbstractRingTensor(res, self.1)
                }
                2 => {
                    let l = self.0.into_dimensionality::<Ix1>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix2>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    AbstractRingTensor(res, self.1)
                }
                other => panic!(
                    "Dot<AbstractRingTensor> cannot handle argument of rank {:?} ",
                    other
                ),
            },
            2 => match rhs.0.ndim() {
                1 => {
                    let l = self.0.into_dimensionality::<Ix2>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix1>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    AbstractRingTensor(res, self.1)
                }
                2 => {
                    let l = self.0.into_dimensionality::<Ix2>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix2>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    AbstractRingTensor(res, self.1)
                }
                other => panic!(
                    "Dot<AbstractRingTensor> cannot handle argument of rank {:?} ",
                    other
                ),
            },
            other => panic!(
                "Dot<AbstractRingTensor> not implemented for tensors of rank {:?}",
                other
            ),
        }
    }
}

impl<T> AbstractRingTensor<T>
where
    Wrapping<T>: Clone + Zero,
{
    pub fn sum(self, axis: Option<usize>) -> AbstractRingTensor<T> {
        if let Some(i) = axis {
            AbstractRingTensor(self.0.sum_axis(Axis(i)), self.1)
        } else {
            let out = Array::from_elem([], self.0.sum())
                .into_dimensionality::<IxDyn>()
                .unwrap();
            AbstractRingTensor(out, self.1)
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
        let r = HostBitTensor::sample_uniform(&shape, &seed);
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
        let x = Ring64Tensor::from(array_backing);
        let y = Ring64Tensor::from(vec![1, 1]);
        let z = x.dot(y);

        let result = Ring64Tensor::from(vec![3, 7]);
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
        let x = Ring64Tensor::from(x_backing);
        let y = Ring64Tensor::from(y_backing);
        let z = x.dot(y);

        let r_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let result = Ring64Tensor::from(r_backing);
        assert_eq!(result, z)
    }

    #[test]
    fn ring_vector_prod() {
        let x_backing = vec![1, 2];
        let y_backing = vec![1, 1];
        let x = Ring64Tensor::from(x_backing);
        let y = Ring64Tensor::from(y_backing);
        let z = x.dot(y);

        let r_backing = Array::from_elem([], Wrapping(3))
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let result = Ring64Tensor::new(r_backing);
        assert_eq!(result, z)
    }

    #[test]
    fn ring_sample() {
        let shape = RawShape(vec![5]);
        let seed = RawSeed([0u8; 16]);
        let r = Ring64Tensor::sample_uniform(&shape, &seed);
        assert_eq!(
            r,
            Ring64Tensor::from(vec![
                4263935709876578662,
                3326810793440857224,
                17325099178452873543,
                15208531650305571673,
                9619880027406922172
            ])
        );

        let r128 = Ring128Tensor::sample_uniform(&shape, &seed);
        assert_eq!(
            r128,
            Ring128Tensor::from(vec![
                78655730786844307471556614669614075016,
                319591670596555766473793801091584867161,
                177455464885365520564027128957528354027,
                72628979995024532377123578937486303732,
                299726520301515014350190124791858941972
            ])
        );

        let r_bits = Ring64Tensor::sample_bits(&shape, &seed);
        assert_eq!(r_bits, Ring64Tensor::from(vec![0, 1, 1, 0, 0]));

        let r128_bits = Ring128Tensor::sample_bits(&shape, &seed);
        assert_eq!(r128_bits, Ring128Tensor::from(vec![0, 1, 1, 0, 0]));
    }

    #[test]
    fn ring_fill() {
        let r = Ring64Tensor::fill(&RawShape(vec![2]), 1);
        assert_eq!(r, Ring64Tensor::from(vec![1, 1]))
    }

    #[test]
    fn ring_sum_with_axis() {
        let x_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = Ring64Tensor::from(x_backing);
        let out = x.sum(Some(0));
        assert_eq!(out, Ring64Tensor::from(vec![4, 6]))
    }

    #[test]
    fn ring_sum_without_axis() {
        let x_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = Ring64Tensor::from(x_backing);
        let exp_v: u64 = 10;
        let exp_backing = Array::from_elem([], exp_v)
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let exp = Ring64Tensor::from(exp_backing);
        let out = x.sum(None);
        assert_eq!(out, exp)
    }

    #[test]
    fn bit_extract() {
        let shape = RawShape(vec![5]);
        let value = 7;

        let r0 = Ring64Tensor::fill(&shape, value).bit_extract(0);
        assert_eq!(HostBitTensor::fill(&shape, 1), r0,);

        let r1 = Ring64Tensor::fill(&shape, value).bit_extract(1);
        assert_eq!(HostBitTensor::fill(&shape, 1), r1,);

        let r2 = Ring64Tensor::fill(&shape, value).bit_extract(2);
        assert_eq!(HostBitTensor::fill(&shape, 1), r2,);

        let r3 = Ring64Tensor::fill(&shape, value).bit_extract(3);
        assert_eq!(HostBitTensor::fill(&shape, 0), r3,)
    }
}
