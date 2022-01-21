//! Placement for plaintext operations on a single role

use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::symbolic::Symbolic;
use crate::execution::Session;
use crate::kernels::*;
use crate::prng::AesRng;
use crate::types::*;
use crate::{BitArray, Const, Ring, N128, N224, N256, N64};
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use ndarray::Slice;
#[cfg(feature = "blas")]
use ndarray_linalg::{Inverse, Lapack, Scalar};
use num_traits::FromPrimitive;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::num::Wrapping;

mod fixedpoint;
mod ops;
mod prim;
pub use prim::*;

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct HostPlacement {
    pub owner: Role,
}

impl From<&str> for HostPlacement {
    fn from(role: &str) -> Self {
        HostPlacement {
            owner: Role::from(role),
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct HostString(pub String, pub HostPlacement);

impl Placed for HostString {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl<S: Session> PlacementPlace<S, HostString> for HostPlacement {
    fn place(&self, _sess: &S, string: HostString) -> HostString {
        match string.placement() {
            Ok(place) if self == &place => string,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                HostString(string.0, self.clone())
            }
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawShape(pub Vec<usize>);

impl RawShape {
    fn extend_singletons(self, mut axis: Vec<usize>) -> Self {
        let ax = axis.pop();
        match ax {
            Some(ax) => {
                let (left, right) = self.0.split_at(ax);
                RawShape::extend_singletons(RawShape([left, right].join(&1usize)), axis)
            }
            None => self,
        }
    }

    fn slice(self, begin: usize, end: usize) -> Self {
        let slc = &self.0[begin..end];
        RawShape(slc.to_vec())
    }

    fn squeeze(mut self, axis: Option<usize>) -> Self {
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

impl Placed for HostShape {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl<S: Session> PlacementPlace<S, HostShape> for HostPlacement {
    fn place(&self, _sess: &S, shape: HostShape) -> HostShape {
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

impl<S: Session, T> PlacementPlace<S, HostTensor<T>> for HostPlacement {
    fn place(&self, _sess: &S, x: HostTensor<T>) -> HostTensor<T> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => HostTensor(x.0, self.clone()),
        }
    }
}

impl<S: Session> PlacementPlace<S, Unit> for HostPlacement {
    fn place(&self, _sess: &S, x: Unit) -> Unit {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => Unit(self.clone()),
        }
    }
}

impl<T> HostTensor<T>
where
    T: LinalgScalar,
{
    pub fn place(plc: &HostPlacement, x: ArrayD<T>) -> HostTensor<T> {
        HostTensor::<T>(x, plc.clone())
    }

    fn atleast_2d(self, to_column_vector: bool) -> HostTensor<T> {
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

    fn dot(self, other: HostTensor<T>) -> HostTensor<T> {
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

    fn ones(shape: HostShape) -> Self {
        HostTensor::<T>(ArrayD::ones(shape.0 .0), shape.1)
    }

    pub fn reshape(self, newshape: HostShape) -> Self {
        HostTensor::<T>(self.0.into_shape(newshape.0 .0).unwrap(), self.1) // TODO need to be fix (unwrap)
    }

    fn expand_dims(self, mut axis: Vec<usize>) -> Self {
        let plc = (&self.1).clone();
        axis.sort_by_key(|ax| Reverse(*ax));
        let newshape = self.shape().0.extend_singletons(axis);
        self.reshape(HostShape(newshape, plc))
    }

    fn squeeze(self, axis: Option<usize>) -> Self {
        let plc = (&self.1).clone();
        let newshape = self.shape().0.squeeze(axis);
        self.reshape(HostShape(newshape, plc))
    }

    pub fn shape(&self) -> HostShape {
        HostShape(RawShape(self.0.shape().into()), self.1.clone())
    }

    fn sum(self, axis: Option<usize>) -> Result<Self> {
        if let Some(i) = axis {
            Ok(HostTensor::<T>(self.0.sum_axis(Axis(i)), self.1))
        } else {
            let out = Array::from_elem([], self.0.sum())
                .into_dimensionality::<IxDyn>()
                .map_err(|e| Error::KernelError(e.to_string()))?;
            Ok(HostTensor::<T>(out, self.1))
        }
    }

    fn transpose(self) -> Self {
        HostTensor::<T>(self.0.reversed_axes(), self.1)
    }
}

#[cfg(feature = "blas")]
impl<T> HostTensor<T>
where
    T: Scalar + Lapack,
{
    fn inv(self) -> Self {
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
impl<T> std::ops::Add for HostTensor<T>
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
impl<T> std::ops::Sub for HostTensor<T>
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
impl<T> std::ops::Mul for HostTensor<T>
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
impl<T> std::ops::Div for HostTensor<T>
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

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct HostBitTensor(pub ArrayD<u8>, pub HostPlacement);

impl std::fmt::Debug for HostBitTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.as_slice().unwrap().fmt(f)
    }
}

impl<S: Session> TensorLike<S> for HostBitTensor {
    type Scalar = u8;
}

impl<S: Session> TensorLike<S> for Symbolic<HostBitTensor> {
    type Scalar = u8;
}

impl Placed for HostBitTensor {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl<S: Session> PlacementPlace<S, HostBitTensor> for HostPlacement {
    fn place(&self, _sess: &S, x: HostBitTensor) -> HostBitTensor {
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

impl HostBitTensor {
    pub fn place(plc: &HostPlacement, x: ArrayD<u8>) -> HostBitTensor {
        HostBitTensor(x, plc.clone())
    }

    fn reshape(self, newshape: HostShape) -> Self {
        HostBitTensor(self.0.into_shape(newshape.0 .0).unwrap(), self.1) // TODO need to be fix (unwrap)
    }

    fn expand_dims(self, mut axis: Vec<usize>) -> Self {
        let plc = (&self.1).clone();
        axis.sort_by_key(|ax| Reverse(*ax));
        let newshape = self.shape().0.extend_singletons(axis);
        self.reshape(HostShape(newshape, plc))
    }

    fn shape(&self) -> HostShape {
        HostShape(RawShape(self.0.shape().into()), self.1.clone())
    }
}

#[allow(dead_code)]
impl HostBitTensor {
    pub(crate) fn from_raw_plc(raw_tensor: ArrayD<u8>, plc: HostPlacement) -> HostBitTensor {
        HostBitTensor(raw_tensor, plc)
    }

    pub(crate) fn from_vec_plc(vec: Vec<u8>, plc: HostPlacement) -> HostBitTensor {
        let raw_tensor = ArrayBase::from_vec(vec).into_dyn();
        Self::from_raw_plc(raw_tensor, plc)
    }

    pub(crate) fn from_slice_plc(slice: &[u8], plc: HostPlacement) -> HostBitTensor {
        let data = slice.to_vec();
        Self::from_vec_plc(data, plc)
    }

    pub(crate) fn from_array_plc<const N: usize>(
        array: [u8; N],
        plc: HostPlacement,
    ) -> HostBitTensor {
        let data = array.to_vec();
        Self::from_vec_plc(data, plc)
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

impl std::ops::BitXor for HostBitTensor {
    type Output = HostBitTensor;
    fn bitxor(self, other: Self) -> Self::Output {
        assert_eq!(self.1, other.1);
        HostBitTensor(self.0 ^ other.0, self.1)
    }
}

impl std::ops::BitAnd for HostBitTensor {
    type Output = HostBitTensor;
    fn bitand(self, other: Self) -> Self::Output {
        assert_eq!(self.1, other.1);
        HostBitTensor(self.0 & other.0, self.1)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HostBitArray<HostBitTensorT, N>(pub HostBitTensorT, pub PhantomData<N>);

impl<HostBitT: CanonicalType, N> CanonicalType for HostBitArray<HostBitT, N> {
    type Type = HostBitArray<<HostBitT as CanonicalType>::Type, N>;
}

impl<HostBitT, N: Const> BitArray for HostBitArray<HostBitT, N> {
    type Len = N;
}

impl<HostBitT: Placed, N: Const> BitArray for Symbolic<HostBitArray<HostBitT, N>> {
    type Len = N;
}

#[cfg(test)]
impl<N> HostBitArray<HostBitTensor, N> {
    pub(crate) fn from_raw_plc(raw_tensor: ArrayD<u8>, plc: HostPlacement) -> Self {
        // TODO check that first dimension equals N
        HostBitArray::<_, N>(HostBitTensor::from_raw_plc(raw_tensor, plc), PhantomData)
    }
}

// TODO implement using moose_type macro
impl<HostBitTensorT: Placed, N> Placed for HostBitArray<HostBitTensorT, N> {
    type Placement = HostBitTensorT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.0.placement()
    }
}

impl PartiallySymbolicType for HostBitArray64 {
    type Type = HostBitArray<<HostBitTensor as SymbolicType>::Type, N64>;
}

impl PartiallySymbolicType for HostBitArray128 {
    type Type = HostBitArray<<HostBitTensor as SymbolicType>::Type, N128>;
}

impl PartiallySymbolicType for HostBitArray224 {
    type Type = HostBitArray<<HostBitTensor as SymbolicType>::Type, N224>;
}

impl PartiallySymbolicType for HostBitArray256 {
    type Type = HostBitArray<<HostBitTensor as SymbolicType>::Type, N256>;
}

impl<HostBitT: Placed, N> From<HostBitArray<HostBitT, N>> for Symbolic<HostBitArray<HostBitT, N>>
where
    HostBitT: Placed<Placement = HostPlacement>,
{
    fn from(x: HostBitArray<HostBitT, N>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<HostBitT, N> TryFrom<Symbolic<HostBitArray<HostBitT, N>>> for HostBitArray<HostBitT, N>
where
    HostBitT: Placed<Placement = HostPlacement>,
{
    type Error = crate::error::Error;
    fn try_from(v: Symbolic<HostBitArray<HostBitT, N>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(crate::error::Error::Unexpected(None)), // TODO err message
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AbstractHostAesKey<HostBitArrayT>(pub(crate) HostBitArrayT);

impl<HostBitArrayT: Placed<Placement = HostPlacement>> Placed
    for AbstractHostAesKey<HostBitArrayT>
{
    type Placement = HostBitArrayT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.0.placement()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HostFixedAesTensor<HostBitArrayT> {
    pub tensor: HostBitArrayT,
    pub integral_precision: u32,
    pub fractional_precision: u32,
}

impl<HostBitArrayT: Placed> Placed for HostFixedAesTensor<HostBitArrayT>
where
    <HostBitArrayT as Placed>::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.tensor.placement()?.into())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HostFixedTensor<HostRingT> {
    pub tensor: HostRingT,
    pub fractional_precision: u32,
    pub integral_precision: u32,
}

impl<RingT: Placed> Placed for HostFixedTensor<RingT> {
    type Placement = RingT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.tensor.placement()
    }
}

impl<S: Session, RingT> PlacementPlace<S, HostFixedTensor<RingT>> for HostPlacement
where
    HostFixedTensor<RingT>: Placed<Placement = HostPlacement>,
    HostPlacement: PlacementPlace<S, RingT>,
{
    fn place(&self, sess: &S, x: HostFixedTensor<RingT>) -> HostFixedTensor<RingT> {
        match x.placement() {
            Ok(place) if self == &place => x,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                HostFixedTensor {
                    tensor: self.place(sess, x.tensor),
                    integral_precision: x.integral_precision,
                    fractional_precision: x.fractional_precision,
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HostRingTensor<T>(pub ArrayD<Wrapping<T>>, pub HostPlacement);

impl Ring for HostRing64Tensor {
    type BitLength = N64;
}

impl Ring for HostRing128Tensor {
    type BitLength = N128;
}

impl<T> Placed for HostRingTensor<T> {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl<S: Session, T> TensorLike<S> for HostRingTensor<T> {
    type Scalar = T;
}

impl<S: Session, T> TensorLike<S> for Symbolic<HostRingTensor<T>> {
    type Scalar = T;
}

pub trait FromRawPlc<P, T> {
    fn from_raw_plc(raw_tensor: ArrayD<T>, plc: P) -> Self;
}

impl<P, T> FromRawPlc<P, T> for HostRingTensor<T>
where
    P: Into<HostPlacement>,
    T: Clone,
{
    fn from_raw_plc(raw_tensor: ArrayD<T>, plc: P) -> HostRingTensor<T> {
        let tensor = raw_tensor.mapv(Wrapping).into_dyn();
        HostRingTensor(tensor, plc.into())
    }
}

impl<P, T> FromRawPlc<P, T> for HostTensor<T>
where
    P: Into<HostPlacement>,
{
    fn from_raw_plc(raw_tensor: ArrayD<T>, plc: P) -> HostTensor<T> {
        HostTensor(raw_tensor, plc.into())
    }
}

impl<R: Ring + Placed> Ring for Symbolic<R> {
    type BitLength = R::BitLength;
}

impl<S: Session, T> PlacementPlace<S, HostRingTensor<T>> for HostPlacement
where
    HostRingTensor<T>: Placed<Placement = HostPlacement>,
{
    fn place(&self, _sess: &S, x: HostRingTensor<T>) -> HostRingTensor<T> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                HostRingTensor(x.0, self.clone())
            }
        }
    }
}

impl<T> HostRingTensor<T> {
    pub fn place(plc: &HostPlacement, x: ArrayD<Wrapping<T>>) -> HostRingTensor<T> {
        HostRingTensor::<T>(x, plc.clone())
    }

    fn reshape(self, newshape: HostShape) -> Self {
        HostRingTensor::<T>(self.0.into_shape(newshape.0 .0).unwrap(), self.1) // TODO need to be fix (unwrap)
    }

    fn expand_dims(self, mut axis: Vec<usize>) -> Self {
        let plc = (&self.1).clone();
        axis.sort_by_key(|ax| Reverse(*ax));
        let newshape = self.shape().0.extend_singletons(axis);
        self.reshape(HostShape(newshape, plc))
    }
}

pub trait Convert<T> {
    type Scale: num_traits::One + Clone;
    fn encode(x: &T, scaling_factor: Self::Scale) -> Self;
    fn decode(x: &Self, scaling_factor: Self::Scale) -> T;
}

impl Convert<HostFloat64Tensor> for HostRing64Tensor {
    type Scale = u64;
    fn encode(x: &HostFloat64Tensor, scaling_factor: Self::Scale) -> HostRing64Tensor {
        let x_upshifted = &x.0 * (scaling_factor as f64);
        let x_converted: ArrayD<u64> = x_upshifted.mapv(|el| (el as i64) as u64);
        HostRing64Tensor::from(x_converted)
    }
    fn decode(x: &Self, scaling_factor: Self::Scale) -> HostFloat64Tensor {
        let x_upshifted: ArrayD<i64> = ArrayD::from(x);
        let x_converted = x_upshifted.mapv(|el| el as f64);
        HostFloat64Tensor::from(x_converted / scaling_factor as f64)
    }
}

impl Convert<HostFloat64Tensor> for HostRing128Tensor {
    type Scale = u128;
    fn encode(x: &HostFloat64Tensor, scaling_factor: Self::Scale) -> HostRing128Tensor {
        let x_upshifted = &x.0 * (scaling_factor as f64);
        let x_converted: ArrayD<u128> = x_upshifted.mapv(|el| (el as i128) as u128);
        HostRing128Tensor::from(x_converted)
    }
    fn decode(x: &Self, scaling_factor: Self::Scale) -> HostFloat64Tensor {
        let x_upshifted: ArrayD<i128> = ArrayD::from(x);
        let x_converted = x_upshifted.mapv(|el| el as f64);
        HostFloat64Tensor::from(x_converted / scaling_factor as f64)
    }
}

impl<T> HostRingTensor<T>
where
    Wrapping<T>: Clone + num_traits::Zero + std::ops::Mul<Wrapping<T>, Output = Wrapping<T>>,
    HostRingTensor<T>: Convert<HostFloat64Tensor>,
{
    fn compute_mean_weight(x: &Self, axis: &Option<usize>) -> Result<HostFloat64Tensor> {
        let shape: &[usize] = x.0.shape();
        if let Some(ax) = axis {
            let dim_len = shape[*ax] as f64;
            Ok(HostFloat64Tensor::from(
                Array::from_elem([], 1.0 / dim_len)
                    .into_dimensionality::<IxDyn>()
                    .map_err(|e| Error::KernelError(e.to_string()))?,
            ))
        } else {
            let dim_prod: usize = std::iter::Product::product(shape.iter());
            let prod_inv = 1.0 / dim_prod as f64;
            Ok(HostFloat64Tensor::from(
                Array::from_elem([], prod_inv)
                    .into_dimensionality::<IxDyn>()
                    .map_err(|e| Error::KernelError(e.to_string()))?,
            ))
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl HostRing64Tensor {
    fn sample_uniform_seeded(shape: &RawShape, seed: &RawSeed) -> HostRing64Tensor {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape.0.as_ref());
        HostRing64Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }

    fn sample_bits_seeded(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u64)).collect();
        let ix = IxDyn(shape.0.as_ref());
        HostRing64Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl HostRing128Tensor {
    fn sample_uniform_seeded(shape: &RawShape, seed: &RawSeed) -> Self {
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

    fn sample_bits_seeded(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u128)).collect();
        let ix = IxDyn(shape.0.as_ref());
        HostRing128Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl HostRing64Tensor {
    fn bit_extract(&self, bit_idx: usize) -> HostBitTensor {
        let temp = &self.0 >> bit_idx;
        let lsb = temp.mapv(|ai| (ai.0 & 1) as u8);
        HostBitTensor::from(lsb)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl HostRing128Tensor {
    fn bit_extract(&self, bit_idx: usize) -> HostBitTensor {
        let temp = &self.0 >> bit_idx;
        let lsb = temp.mapv(|ai| (ai.0 & 1) as u8);
        HostBitTensor::from(lsb)
    }
}

impl<T> HostRingTensor<T>
where
    T: Clone,
{
    pub fn from_raw_plc<D: ndarray::Dimension, P: Into<HostPlacement>>(
        raw_tensor: Array<T, D>,
        plc: P,
    ) -> HostRingTensor<T> {
        let tensor = raw_tensor.mapv(Wrapping).into_dyn();
        HostRingTensor(tensor, plc.into())
    }
}

impl<T> HostTensor<T>
where
    T: Clone,
{
    fn from_raw_plc<D: ndarray::Dimension, P: Into<HostPlacement>>(
        raw_tensor: Array<T, D>,
        plc: P,
    ) -> HostTensor<T> {
        HostTensor(raw_tensor.into_dyn(), plc.into())
    }
}

// This implementation is only used by the old kernels. Construct HostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> HostRingTensor<T>
where
    Wrapping<T>: Clone,
{
    #[cfg_attr(
        feature = "exclude_old_framework",
        deprecated(
            note = "This function is only used by the old kernels, which are not aware of the placements."
        )
    )]
    fn fill(shape: &RawShape, el: T) -> HostRingTensor<T> {
        HostRingTensor(
            ArrayD::from_elem(shape.0.as_ref(), Wrapping(el)),
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

impl<T> HostRingTensor<T> {
    fn shape(&self) -> HostShape {
        HostShape(RawShape(self.0.shape().into()), self.1.clone())
    }
}

// This implementation is only used by the old kernels. Construct HostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<ArrayD<T>> for HostRingTensor<T>
where
    T: Clone,
{
    fn from(a: ArrayD<T>) -> HostRingTensor<T> {
        let wrapped = a.mapv(Wrapping);
        HostRingTensor(
            wrapped,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct HostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl From<ArrayD<i64>> for HostRingTensor<u64> {
    fn from(a: ArrayD<i64>) -> HostRingTensor<u64> {
        let ring_rep = a.mapv(|ai| Wrapping(ai as u64));
        HostRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct HostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl From<ArrayD<i128>> for HostRingTensor<u128> {
    fn from(a: ArrayD<i128>) -> HostRingTensor<u128> {
        let ring_rep = a.mapv(|ai| Wrapping(ai as u128));
        HostRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> HostRingTensor<T> {
    fn new(a: ArrayD<Wrapping<T>>) -> HostRingTensor<T> {
        HostRingTensor(
            a,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct HostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<HostBitTensor> for HostRingTensor<T>
where
    T: From<u8>,
{
    fn from(b: HostBitTensor) -> HostRingTensor<T> {
        let ring_rep = b.0.mapv(|ai| Wrapping(ai.into()));
        HostRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

impl From<&HostRingTensor<u64>> for ArrayD<i64> {
    fn from(r: &HostRingTensor<u64>) -> ArrayD<i64> {
        r.0.mapv(|element| element.0 as i64)
    }
}

impl From<&HostRingTensor<u128>> for ArrayD<i128> {
    fn from(r: &HostRingTensor<u128>) -> ArrayD<i128> {
        r.0.mapv(|element| element.0 as i128)
    }
}

// This implementation is only used by the old kernels. Construct HostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<Vec<T>> for HostRingTensor<T> {
    fn from(v: Vec<T>) -> HostRingTensor<T> {
        let ix = IxDyn(&[v.len()]);
        use vec_utils::VecExt;
        let v_wrapped: Vec<_> = v.map(Wrapping);
        HostRingTensor(
            Array::from_shape_vec(ix, v_wrapped).unwrap(),
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

// This implementation is only used by the old kernels. Construct HostRingTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "exclude_old_framework"))]
impl<T> From<&[T]> for HostRingTensor<T>
where
    T: Copy,
{
    fn from(v: &[T]) -> HostRingTensor<T> {
        let ix = IxDyn(&[v.len()]);
        let v_wrapped: Vec<_> = v.iter().map(|vi| Wrapping(*vi)).collect();
        HostRingTensor(
            Array::from_shape_vec(ix, v_wrapped).unwrap(),
            HostPlacement {
                owner: Role::from("TODO"), // Fake owner for the old kernels
            },
        )
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> std::ops::Add<HostRingTensor<T>> for HostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: std::ops::Add<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = HostRingTensor<T>;
    fn add(self, other: HostRingTensor<T>) -> Self::Output {
        HostRingTensor(self.0 + other.0, self.1)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> std::ops::Mul<HostRingTensor<T>> for HostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: std::ops::Mul<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = HostRingTensor<T>;
    fn mul(self, other: HostRingTensor<T>) -> Self::Output {
        HostRingTensor(self.0 * other.0, self.1)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> std::ops::Sub<HostRingTensor<T>> for HostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: std::ops::Sub<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = HostRingTensor<T>;
    fn sub(self, other: HostRingTensor<T>) -> Self::Output {
        HostRingTensor(self.0 - other.0, self.1)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> std::ops::Shl<usize> for HostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: std::ops::Shl<usize, Output = Wrapping<T>>,
{
    type Output = HostRingTensor<T>;
    fn shl(self, other: usize) -> Self::Output {
        HostRingTensor(self.0 << other, self.1)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> std::ops::Shr<usize> for HostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: std::ops::Shr<usize, Output = Wrapping<T>>,
{
    type Output = HostRingTensor<T>;
    fn shr(self, other: usize) -> Self::Output {
        HostRingTensor(self.0 >> other, self.1)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl<T> std::ops::BitAnd<HostRingTensor<T>> for HostRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: std::ops::BitAnd<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = HostRingTensor<T>;
    fn bitand(self, other: HostRingTensor<T>) -> Self::Output {
        HostRingTensor(self.0 & other.0, self.1)
    }
}

impl<T> HostRingTensor<T>
where
    Wrapping<T>: LinalgScalar,
{
    fn dot(self, rhs: HostRingTensor<T>) -> Result<HostRingTensor<T>> {
        match self.0.ndim() {
            1 => match rhs.0.ndim() {
                1 => {
                    let l = self
                        .0
                        .into_dimensionality::<Ix1>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    let r = rhs
                        .0
                        .into_dimensionality::<Ix1>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    let res = Array::from_elem([], l.dot(&r))
                        .into_dimensionality::<IxDyn>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    Ok(HostRingTensor(res, self.1))
                }
                2 => {
                    let l = self
                        .0
                        .into_dimensionality::<Ix1>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    let r = rhs
                        .0
                        .into_dimensionality::<Ix2>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    let res = l
                        .dot(&r)
                        .into_dimensionality::<IxDyn>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    Ok(HostRingTensor(res, self.1))
                }
                other => Err(Error::KernelError(format!(
                    "Dot<HostRingTensor> cannot handle argument of rank {:?} ",
                    other
                ))),
            },
            2 => match rhs.0.ndim() {
                1 => {
                    let l = self
                        .0
                        .into_dimensionality::<Ix2>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    let r = rhs
                        .0
                        .into_dimensionality::<Ix1>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    let res = l
                        .dot(&r)
                        .into_dimensionality::<IxDyn>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    Ok(HostRingTensor(res, self.1))
                }
                2 => {
                    let l = self
                        .0
                        .into_dimensionality::<Ix2>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    let r = rhs
                        .0
                        .into_dimensionality::<Ix2>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    let res = l
                        .dot(&r)
                        .into_dimensionality::<IxDyn>()
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                    Ok(HostRingTensor(res, self.1))
                }
                other => Err(Error::KernelError(format!(
                    "Dot<HostRingTensor> cannot handle argument of rank {:?} ",
                    other
                ))),
            },
            other => Err(Error::KernelError(format!(
                "Dot<HostRingTensor> not implemented for tensors of rank {:?}",
                other
            ))),
        }
    }
}

impl<T> HostRingTensor<T>
where
    Wrapping<T>: Clone + num_traits::Zero,
{
    fn sum(self, axis: Option<usize>) -> Result<HostRingTensor<T>> {
        if let Some(i) = axis {
            Ok(HostRingTensor(self.0.sum_axis(Axis(i)), self.1))
        } else {
            let out = Array::from_elem([], self.0.sum())
                .into_dimensionality::<IxDyn>()
                .map_err(|e| Error::KernelError(e.to_string()))?;
            Ok(HostRingTensor(out, self.1))
        }
    }
}

trait FromRaw<T, O> {
    fn from_raw(&self, raw: T) -> O;
}

impl<T: Clone, D: ndarray::Dimension> FromRaw<Array<T, D>, HostTensor<T>> for HostPlacement {
    fn from_raw(&self, raw: Array<T, D>) -> HostTensor<T> {
        HostTensor(raw.into_dyn(), self.clone())
    }
}

impl<T: Clone, D: ndarray::Dimension> FromRaw<Array<T, D>, HostRingTensor<T>> for HostPlacement {
    fn from_raw(&self, raw: Array<T, D>) -> HostRingTensor<T> {
        HostRingTensor(raw.mapv(Wrapping).into_dyn(), self.clone())
    }
}

impl<D: ndarray::Dimension> FromRaw<Array<u8, D>, HostBitTensor> for HostPlacement {
    fn from_raw(&self, raw: Array<u8, D>) -> HostBitTensor {
        HostBitTensor(raw.into_dyn(), self.clone())
    }
}

impl FromRaw<RawShape, HostShape> for HostPlacement {
    fn from_raw(&self, raw: RawShape) -> HostShape {
        HostShape(raw, self.clone())
    }
}

impl FromRaw<RawSeed, Seed> for HostPlacement {
    fn from_raw(&self, raw: RawSeed) -> Seed {
        Seed(raw, self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::SyncSession;
    use rstest::rstest;

    #[test]
    fn test_host_shape_op() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x: HostRing64Tensor = plc.from_raw(array![1024, 5, 4]);
        let shape = plc.shape(&sess, &x).0;

        let expected = RawShape(vec![3]);
        assert_eq!(expected, shape);
    }

    #[test]
    fn dot_prod_f32() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x: HostFloat32Tensor = plc.from_raw(array![[1.0, -2.0], [3.0, -4.0]]);
        let z = plc.dot(&sess, &x, &x);

        let expected: HostFloat32Tensor = plc.from_raw(array![[-5.0, 6.0], [-9.0, 10.0]]);
        assert_eq!(z, expected);
    }

    #[cfg(feature = "blas")]
    #[test]
    fn test_inverse() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x: HostFloat32Tensor = plc.from_raw(array![[1.0, 2.0], [3.0, 4.0]]);
        let x_inv = plc.inverse(&sess, &x);

        let expected: HostFloat32Tensor = plc.from_raw(array![[-2.0, 1.0], [1.5, -0.5]]);
        assert_eq!(x_inv, expected);
    }

    #[test]
    fn test_shape_slice() {
        let x_shape = RawShape(vec![1, 2, 3]);
        let x_slice = x_shape.slice(1, 3);
        assert_eq!(x_slice, RawShape(vec![2, 3]))
    }

    #[test]
    fn test_tensor_slice() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x: HostRing64Tensor = plc.from_raw(array![[1, 2], [3, 4]]);

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
        let y = plc.slice(&sess, slice, &x);

        let expected: HostRing64Tensor = plc.from_raw(array![[3, 4]]);
        assert_eq!(y, expected);
    }

    #[test]
    fn test_tensor_slice_neg_indicies() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x: HostRing64Tensor = plc.from_raw(array![[1, 2], [3, 4]]);

        let slice = SliceInfo(vec![
            SliceInfoElem {
                start: -1,
                end: None,
                step: Some(2),
            },
            SliceInfoElem {
                start: -1,
                end: None,
                step: Some(2),
            },
        ]);
        let y = plc.slice(&sess, slice, &x);

        // This example we take the last element of the last element in dimension 1, which is just 4.
        let expected: HostRing64Tensor = plc.from_raw(array![[4]]);
        assert_eq!(y, expected);
    }

    #[test]
    #[should_panic]
    fn test_tensor_slice_index_out_of_range() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x: HostRing64Tensor = plc.from_raw(array![[1, 2], [3, 4]]);

        let slice = SliceInfo(vec![
            SliceInfoElem {
                start: -1,
                end: None,
                step: Some(2),
            },
            SliceInfoElem {
                start: -1,
                end: None,
                step: Some(2),
            },
            SliceInfoElem {
                start: -1,
                end: None,
                step: Some(2),
            },
        ]);
        let _y = plc.slice(&sess, slice, &x);
        // This example we expect a panic from the underlying slice implementation.
    }

    #[test]
    fn test_diag() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x: HostFloat64Tensor = plc.from_raw(array![[1.0, 2.0], [3.0, 4.0]]);
        let y = plc.diag(&sess, &x);

        let expected: HostFloat64Tensor = plc.from_raw(array![1.0, 4.0]);
        assert_eq!(y, expected);
    }

    #[test]
    fn test_index() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x: HostRing64Tensor = plc.from_raw(array![[[1_u64, 2], [3, 4]], [[4, 5], [6, 7]]]);
        let y = plc.index_axis(&sess, 0, 1, &x);

        let expected: HostRing64Tensor = plc.from_raw(array![[4, 5], [6, 7]]);
        assert_eq!(y, expected);
    }

    #[test]
    fn test_transpose() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x: HostFloat32Tensor = plc.from_raw(array![[1.0, 2.0], [3.0, 4.0]]);
        let y = plc.transpose(&sess, &x);

        let expected: HostFloat32Tensor = plc.from_raw(array![[1.0, 3.0], [2.0, 4.0]]);
        assert_eq!(y, expected);
    }

    #[test]
    fn test_concatenate() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let a: HostFloat32Tensor = plc.from_raw(array![[[1.0, 2.0], [3.0, 4.0]]]);
        let b: HostFloat32Tensor = plc.from_raw(array![[[1.0, 2.0], [3.0, 4.0]]]);
        let conc = plc.concatenate(&sess, 0, &vec![a, b]);

        let expected: HostFloat32Tensor =
            plc.from_raw(array![[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]);
        assert_eq!(conc, expected)
    }

    #[test]
    fn test_atleast_2d() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let a: HostFloat32Tensor = plc.from_raw(array![[1.0, 2.0], [3.0, 4.0]]);
        let a_exp = a.clone();
        let b: HostFloat32Tensor = plc.from_raw(array![1.0, 2.0, 3.0, 4.0]);
        let b_exp: HostFloat32Tensor = plc.from_raw(array![[1.0, 2.0, 3.0, 4.0]]);
        let c: HostFloat32Tensor = plc.from_raw(array![1.0, 2.0, 3.0, 4.0]);
        let c_exp: HostFloat32Tensor = plc.from_raw(array![[1.0], [2.0], [3.0], [4.0]]);
        let d: HostFloat32Tensor = plc.from_raw(Array::from_elem([], 1.0));
        let d_exp: HostFloat32Tensor = plc.from_raw(array![[1.0]]);

        let ax = plc.at_least_2d(&sess, true, &a);
        let bx = plc.at_least_2d(&sess, false, &b);
        let cx = plc.at_least_2d(&sess, true, &c);
        let dx = plc.at_least_2d(&sess, true, &d);
        assert_eq!(ax, a_exp);
        assert_eq!(bx, b_exp);
        assert_eq!(cx, c_exp);
        assert_eq!(dx, d_exp);
    }

    #[test]
    fn test_add_broadcasting() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x_1: HostFloat32Tensor = plc.from_raw(array![2.0]);
        let y_1: HostFloat32Tensor = plc.from_raw(array![1.0, 2.0]);
        let z_1 = plc.add(&sess, &x_1, &y_1);
        let z_1_exp: HostFloat32Tensor = plc.from_raw(array![3.0, 4.0]);
        assert_eq!(z_1, z_1_exp);

        let x_2: HostFloat32Tensor = plc.from_raw(array![1.0, 2.0]);
        let y_2: HostFloat32Tensor = plc.from_raw(array![2.0]);
        let z_2 = plc.add(&sess, &x_2, &y_2);
        let z_2_exp: HostFloat32Tensor = plc.from_raw(array![3.0, 4.0]);
        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_sub_broadcasting() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x_1: HostFloat32Tensor = plc.from_raw(array![2.0]);
        let y_1: HostFloat32Tensor = plc.from_raw(array![1.0, 2.0]);
        let z_1 = plc.sub(&sess, &x_1, &y_1);
        let z_1_exp: HostFloat32Tensor = plc.from_raw(array![1.0, 0.0]);
        assert_eq!(z_1, z_1_exp);

        let x_2: HostFloat32Tensor = plc.from_raw(array![1.0, 2.0]);
        let y_2: HostFloat32Tensor = plc.from_raw(array![2.0]);
        let z_2 = plc.sub(&sess, &x_2, &y_2);
        let z_2_exp: HostFloat32Tensor = plc.from_raw(array![-1.0, 0.0]);

        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_mul_broadcasting() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x_1: HostFloat32Tensor = plc.from_raw(array![2.0]);
        let y_1: HostFloat32Tensor = plc.from_raw(array![1.0, 2.0]);
        let z_1 = plc.mul(&sess, &x_1, &y_1);
        let z_1_exp: HostFloat32Tensor = plc.from_raw(array![2.0, 4.0]);
        assert_eq!(z_1, z_1_exp);

        let x_2: HostFloat32Tensor = plc.from_raw(array![1.0, 2.0]);
        let y_2: HostFloat32Tensor = plc.from_raw(array![2.0]);
        let z_2 = plc.mul(&sess, &x_2, &y_2);
        let z_2_exp: HostFloat32Tensor = plc.from_raw(array![2.0, 4.0]);
        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_div_broadcasting() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x_1: HostFloat32Tensor = plc.from_raw(array![1.0]);
        let y_1: HostFloat32Tensor = plc.from_raw(array![2.0, 4.0]);
        let z_1 = plc.div(&sess, &x_1, &y_1);
        let z_1_exp: HostFloat32Tensor = plc.from_raw(array![0.5, 0.25]);
        assert_eq!(z_1, z_1_exp);

        let x_2: HostFloat32Tensor = plc.from_raw(array![2.0, 4.0]);
        let y_2: HostFloat32Tensor = plc.from_raw(array![2.0]);
        let z_2 = plc.div(&sess, &x_2, &y_2);
        let z_2_exp: HostFloat32Tensor = plc.from_raw(array![1.0, 2.0]);
        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_kernel_sqrt() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostFloat64Tensor = plc.from_raw(array![[4.0, 9.0], [16.0, 25.0]]);
        let sqrt = plc.sqrt(&sess, &x);

        let expected: HostFloat64Tensor = plc.from_raw(array![[2.0, 3.0], [4.0, 5.0]]);
        assert_eq!(expected, sqrt)
    }

    #[rstest]
    #[case(None)]
    #[case(Some(2))]
    fn test_kernel_squeeze(#[case] axis: Option<u32>) {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostFloat64Tensor = plc.from_raw(array![[1.0, 2.0], [3.0, 4.0]]);
        let x_expanded = plc.expand_dims(&sess, vec![2], &x);
        let x_squeezed = plc.squeeze(&sess, axis, &x_expanded);
        let shape = plc.shape(&sess, &x_squeezed);

        let expected = RawShape(vec![2, 2]);
        assert_eq!(expected, shape.0)
    }

    #[test]
    fn test_kernel_transpose() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostFloat64Tensor = plc.from_raw(array![[1.0, 2.0], [3.0, 4.0]]);
        let t = plc.transpose(&sess, &x);

        let expected: HostFloat64Tensor = plc.from_raw(array![[1.0, 3.0], [2.0, 4.0]]);
        assert_eq!(expected, t);
    }

    #[test]
    fn test_kernel_concatenate() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostFloat64Tensor = plc.from_raw(array![[1.0, 2.0], [3.0, 4.0]]);
        let y: HostFloat64Tensor = plc.from_raw(array![[5.0, 6.0], [7.0, 8.0]]);
        let c = plc.concatenate(&sess, 0, &[x, y]);

        let expected: HostFloat64Tensor =
            plc.from_raw(array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        assert_eq!(expected, c);
    }

    #[test]
    fn bit_sample() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let shape: HostShape = plc.from_raw(RawShape(vec![5]));
        let seed: Seed = plc.from_raw(RawSeed([0u8; 16]));
        let r: HostBitTensor = plc.sample_uniform_seeded(&sess, &shape, &seed);

        let expected: HostBitTensor = plc.from_raw(array![0, 1, 1, 0, 0]);
        assert_eq!(r, expected);
    }

    #[test]
    fn bit_fill() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let shape: HostShape = plc.from_raw(RawShape(vec![2]));
        let x: HostBitTensor = plc.fill(&sess, 1_u8.into(), &shape);

        let expected: HostBitTensor = plc.from_raw(array![1, 1]);
        assert_eq!(x, expected);
    }

    #[test]
    fn bit_xor() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let shape: HostShape = plc.from_raw(RawShape(vec![5]));
        let zero = plc.fill(&sess, 0_u8.into(), &shape);
        let one = plc.fill(&sess, 1_u8.into(), &shape);

        assert_eq!(&plc.xor(&sess, &zero, &one), &one);
        assert_eq!(&plc.xor(&sess, &one, &zero), &one);
        assert_eq!(&plc.xor(&sess, &one, &one), &zero);
        assert_eq!(&plc.xor(&sess, &zero, &zero), &zero);
    }

    #[test]
    fn bit_or() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let shape: HostShape = plc.from_raw(RawShape(vec![5]));
        let zero: HostBitTensor = plc.fill(&sess, 0_u8.into(), &shape);
        let one: HostBitTensor = plc.fill(&sess, 1_u8.into(), &shape);

        assert_eq!(&plc.or(&sess, &zero, &one), &one);
        assert_eq!(&plc.or(&sess, &one, &zero), &one);
        assert_eq!(&plc.or(&sess, &one, &one), &one);
        assert_eq!(&plc.or(&sess, &zero, &zero), &zero);
    }

    #[test]
    fn bit_and() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let shape: HostShape = plc.from_raw(RawShape(vec![5]));
        let zero: HostBitTensor = plc.fill(&sess, 0_u8.into(), &shape);
        let one: HostBitTensor = plc.fill(&sess, 1_u8.into(), &shape);

        assert_eq!(&plc.and(&sess, &zero, &one), &zero);
        assert_eq!(&plc.and(&sess, &one, &zero), &zero);
        assert_eq!(&plc.and(&sess, &one, &one), &one);
        assert_eq!(&plc.and(&sess, &zero, &zero), &zero);
    }

    #[test]
    fn ring_matrix_vector_prod() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostRing64Tensor = plc.from_raw(array![[1, 2], [3, 4]]);
        let y: HostRing64Tensor = plc.from_raw(array![1, 1]);
        let z = plc.dot(&sess, &x, &y);

        let expected: HostRing64Tensor = plc.from_raw(array![3, 7]);
        assert_eq!(expected, z);
    }

    #[test]
    fn ring_matrix_matrix_prod() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostRing64Tensor = plc.from_raw(array![[1, 2], [3, 4]]);
        let y: HostRing64Tensor = plc.from_raw(array![[1, 0], [0, 1]]);
        let z = plc.dot(&sess, &x, &y);

        let expected: HostRing64Tensor = plc.from_raw(array![[1, 2], [3, 4]]);
        assert_eq!(expected, z);
    }

    #[test]
    fn ring_vector_prod() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostRing64Tensor = plc.from_raw(array![1, 2]);
        let y: HostRing64Tensor = plc.from_raw(array![1, 1]);
        let z = plc.dot(&sess, &x, &y);

        let expected: HostRing64Tensor = plc.from_raw(Array::from_elem([], 3));
        assert_eq!(expected, z);
    }

    #[test]
    fn ring_sample() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let shape = plc.from_raw(RawShape(vec![5]));
        let seed = plc.from_raw(RawSeed([0u8; 16]));

        let r64: HostRing64Tensor = plc.sample_uniform_seeded(&sess, &shape, &seed);
        assert_eq!(
            r64,
            plc.from_raw(array![
                4263935709876578662,
                3326810793440857224,
                17325099178452873543,
                15208531650305571673,
                9619880027406922172
            ])
        );

        let r128: HostRing128Tensor = plc.sample_uniform_seeded(&sess, &shape, &seed);
        assert_eq!(
            r128,
            plc.from_raw(array![
                78655730786844307471556614669614075016,
                319591670596555766473793801091584867161,
                177455464885365520564027128957528354027,
                72628979995024532377123578937486303732,
                299726520301515014350190124791858941972
            ])
        );

        let r64_bits: HostRing64Tensor = plc.sample_bits_seeded(&sess, &shape, &seed);
        assert_eq!(r64_bits, plc.from_raw(array![0, 1, 1, 0, 0]));

        let r128_bits: HostRing128Tensor = plc.sample_bits_seeded(&sess, &shape, &seed);
        assert_eq!(r128_bits, plc.from_raw(array![0, 1, 1, 0, 0]));
    }

    #[test]
    fn ring_fill() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let shape = plc.from_raw(RawShape(vec![2]));
        let r: HostRing64Tensor = plc.fill(&sess, 1_u64.into(), &shape);
        assert_eq!(r, plc.from_raw(array![1, 1]));
    }

    #[test]
    fn ring_sum_with_axis() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostRing64Tensor = plc.from_raw(array![[1, 2], [3, 4]]);
        let out = plc.sum(&sess, Some(0), &x);
        assert_eq!(out, plc.from_raw(array![4, 6]))
    }

    #[test]
    fn ring_sum_without_axis() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostRing64Tensor = plc.from_raw(array![[1, 2], [3, 4]]);
        let out = plc.sum(&sess, None, &x);
        assert_eq!(out, plc.from_raw(Array::from_elem([], 10_u64)))
    }

    #[test]
    fn ring_add_n() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        // only 1 tensor
        let x: HostRing64Tensor = plc.from_raw(array![[1, 4], [9, 16], [25, 36]]);
        let expected: HostRing64Tensor = plc.from_raw(array![[1, 4], [9, 16], [25, 36]]);
        let out = plc.add_n(&sess, &[x]);
        assert_eq!(out, expected);

        // 64 bit
        // I'll buy you a beer if you tell me what all of these sequences are ;)
        let x: HostRing64Tensor = plc.from_raw(array![[1, 4], [9, 16], [25, 36]]);
        let y: HostRing64Tensor = plc.from_raw(array![[1, 3], [6, 10], [15, 21]]);
        let z: HostRing64Tensor = plc.from_raw(array![[1, 36], [1225, 41616], [1413721, 48024900]]);
        let expected: HostRing64Tensor =
            plc.from_raw(array![[3, 43], [1240, 41642], [1413761, 48024957]]);
        let out = plc.add_n(&sess, &[x, y, z]);
        assert_eq!(out, expected);

        // 128 bit
        let w: HostRing128Tensor = plc.from_raw(array![[6, 3, 10], [5, 16, 8], [4, 2, 1]]);
        let x: HostRing128Tensor = plc.from_raw(array![[40, 20, 10], [5, 16, 8], [4, 2, 1]]);
        let y: HostRing128Tensor = plc.from_raw(array![[42, 21, 64], [32, 16, 8], [4, 2, 1]]);
        let z: HostRing128Tensor = plc.from_raw(array![[256, 128, 64], [32, 16, 8], [4, 2, 1]]);
        let expected: HostRing128Tensor =
            plc.from_raw(array![[344, 172, 148], [74, 64, 32], [16, 8, 4]]);
        let out = plc.add_n(&sess, &[w, x, y, z]);
        assert_eq!(out, expected);
    }

    #[test]
    fn bit_extract() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let shape = plc.from_raw(RawShape(vec![5]));
        let value: HostRing64Tensor = plc.fill(&sess, 7_u64.into(), &shape);

        let r0 = plc.bit_extract(&sess, 0, &value);
        let r0_expected: HostBitTensor = plc.from_raw(array![1, 1, 1, 1, 1]);
        assert_eq!(r0, r0_expected);

        let r1 = plc.bit_extract(&sess, 1, &value);
        let r1_expected: HostBitTensor = plc.from_raw(array![1, 1, 1, 1, 1]);
        assert_eq!(r1, r1_expected);

        let r2 = plc.bit_extract(&sess, 2, &value);
        let r2_expected: HostBitTensor = plc.from_raw(array![1, 1, 1, 1, 1]);
        assert_eq!(r2, r2_expected);

        let r3 = plc.bit_extract(&sess, 3, &value);
        let r3_expected: HostBitTensor = plc.from_raw(array![0, 0, 0, 0, 0]);
        assert_eq!(r3, r3_expected);
    }

    #[test]
    fn bit_decompose1() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostRing64Tensor = plc.from_raw(array![[[1, 2], [3, 4]], [[4, 5], [6, 7]]]);
        let x_bits: HostBitTensor = plc.bit_decompose(&sess, &x);
        let targets: Vec<_> = (0..64).map(|i| plc.bit_extract(&sess, i, &x)).collect();

        for (i, target) in targets.iter().enumerate() {
            let sliced = plc.index_axis(&sess, 0, i, &x_bits);
            assert_eq!(&sliced, target);
        }
    }

    #[test]
    fn bit_decompose2() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostRing64Tensor = plc.from_raw(array![6743216615002642708]);
        let x_bits: HostBitTensor = plc.bit_decompose(&sess, &x);

        let expected: HostBitTensor = plc.from_raw(
            array![
                0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0,
                0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
                1, 0, 1, 1, 1, 0, 1, 0
            ]
            .into_shape((64, 1))
            .unwrap(),
        );
        assert_eq!(x_bits, expected);
    }
}
