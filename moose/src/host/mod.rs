//! Placement for plaintext operations by a single role

use crate::computation::*;
use crate::error::Result;
#[cfg(feature = "compile")]
use crate::execution::symbolic::Symbolic;
use crate::execution::Session;
use crate::kernels::*;
use crate::types::*;
use crate::{BitArray, Const, Ring, TensorLike, N128, N64};
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use ndarray::Slice;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::num::Wrapping;

mod bitarray;
mod fixedpoint;
mod ops;
mod prim;
pub use bitarray::*;
pub use fixedpoint::Convert;
pub use prim::*;

pub type ArcArrayD<A> = ArcArray<A, IxDyn>;

/// Placement type for single role plaintext operations
#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Debug)]
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

impl<S: Session> PlacementPlace<S, HostUnit> for HostPlacement {
    fn place(&self, _sess: &S, x: HostUnit) -> HostUnit {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => HostUnit(self.clone()),
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

#[derive(Serialize, Deserialize, PartialEq, Hash, Clone, Debug)]
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

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct SliceInfoElem {
    /// Start index; negative are counted from the back of the axis.
    pub start: isize,
    /// End index; negative are counted from the back of the axis; when not present
    /// the default is the full length of the axis.
    pub end: Option<isize>,
    /// Step size in elements; the default is 1, for every element.
    pub step: Option<isize>,
}

// Slicing needs a SliceInfoElem for each shape dimension
#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug)]
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

#[derive(Serialize, Deserialize, Hash, Clone, Debug, PartialEq)]
pub struct HostTensor<T>(pub ArcArrayD<T>, pub HostPlacement);

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

impl<T> HostTensor<T>
where
    T: LinalgScalar,
{
    pub(crate) fn place(plc: &HostPlacement, x: ArcArrayD<T>) -> HostTensor<T> {
        HostTensor::<T>(x, plc.clone())
    }

    pub(crate) fn reshape(self, newshape: HostShape) -> Self {
        HostTensor::<T>(self.0.into_shape(newshape.0 .0).unwrap(), self.1) // TODO need to be fix (unwrap)
    }

    pub(crate) fn shape(&self) -> HostShape {
        HostShape(RawShape(self.0.shape().into()), self.1.clone())
    }
}

#[derive(Serialize, Deserialize, Hash, Clone, PartialEq)]
pub struct HostBitTensor(pub BitArrayRepr, pub HostPlacement);

impl std::fmt::Debug for HostBitTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.data.as_bitslice().fmt(f)
    }
}

impl TensorLike for HostBitTensor {
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
    pub(crate) fn place(plc: &HostPlacement, x: BitArrayRepr) -> HostBitTensor {
        HostBitTensor(x, plc.clone())
    }

    fn reshape(self, newshape: HostShape) -> Self {
        let arr = BitArrayRepr {
            data: self.0.data.clone(),
            dim: std::sync::Arc::new(IxDyn(&newshape.0 .0)),
        };
        HostBitTensor(arr, self.1)
    }

    fn expand_dims(self, mut axis: Vec<usize>) -> Self {
        let plc = self.1.clone();
        axis.sort_by_key(|ax| Reverse(*ax));
        let newshape = self.shape().0.extend_singletons(axis);
        self.reshape(HostShape(newshape, plc))
    }

    fn shape(&self) -> HostShape {
        HostShape(RawShape(self.0.shape().into()), self.1.clone())
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

#[cfg(feature = "compile")]
impl<HostBitT: Placed, N: Const> BitArray for Symbolic<HostBitArray<HostBitT, N>> {
    type Len = N;
}

// TODO implement using moose_type macro
impl<HostBitTensorT: Placed, N> Placed for HostBitArray<HostBitTensorT, N> {
    type Placement = HostBitTensorT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.0.placement()
    }
}

#[cfg(feature = "compile")]
impl<HostBitTensorT, N: Const> PartiallySymbolicType for HostBitArray<HostBitTensorT, N>
where
    HostBitTensorT: SymbolicType,
{
    type Type = HostBitArray<<HostBitTensorT as SymbolicType>::Type, N>;
}

#[cfg(feature = "compile")]
impl<HostBitT, N> From<HostBitArray<HostBitT, N>> for Symbolic<HostBitArray<HostBitT, N>>
where
    HostBitT: Placed<Placement = HostPlacement>,
{
    fn from(x: HostBitArray<HostBitT, N>) -> Self {
        Symbolic::Concrete(x)
    }
}

#[cfg(feature = "compile")]
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

#[derive(Serialize, Deserialize, Hash, Clone, Debug, PartialEq)]
pub struct HostRingTensor<T>(pub ArcArrayD<Wrapping<T>>, pub HostPlacement);

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

impl<T> TensorLike for HostRingTensor<T> {
    type Scalar = T;
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
    pub(crate) fn place(plc: &HostPlacement, x: ArcArrayD<Wrapping<T>>) -> HostRingTensor<T> {
        HostRingTensor::<T>(x, plc.clone())
    }

    fn reshape(self, newshape: HostShape) -> Self {
        HostRingTensor::<T>(self.0.into_shape(newshape.0 .0).unwrap(), self.1) // TODO need to be fix (unwrap)
    }

    fn expand_dims(self, mut axis: Vec<usize>) -> Self {
        let plc = self.1.clone();
        axis.sort_by_key(|ax| Reverse(*ax));
        let newshape = self.shape().0.extend_singletons(axis);
        self.reshape(HostShape(newshape, plc))
    }
}

#[cfg(test)]
impl<T: Clone> HostRingTensor<T> {
    pub(crate) fn from_raw_plc<D: ndarray::Dimension, P: Into<HostPlacement>>(
        raw_tensor: Array<T, D>,
        plc: P,
    ) -> HostRingTensor<T> {
        let tensor = raw_tensor.mapv(Wrapping).into_dyn();
        HostRingTensor(tensor.into_shared(), plc.into())
    }
}

impl<T> HostRingTensor<T> {
    fn shape(&self) -> HostShape {
        HostShape(RawShape(self.0.shape().into()), self.1.clone())
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

impl<T> HostRingTensor<T>
where
    Wrapping<T>: Clone + num_traits::Zero,
{
    fn sum(self, axis: Option<usize>) -> Result<HostRingTensor<T>> {
        if let Some(i) = axis {
            Ok(HostRingTensor(
                self.0.sum_axis(Axis(i)).into_shared(),
                self.1,
            ))
        } else {
            let out = Array::from_elem([], self.0.sum()).into_dyn();
            Ok(HostRingTensor(out.into_shared(), self.1))
        }
    }
}

pub trait FromRaw<T, O> {
    #![allow(clippy::wrong_self_convention)]
    fn from_raw(&self, raw: T) -> O;
}

impl<T: Clone, O> FromRaw<&[T], O> for HostPlacement
where
    HostPlacement: FromRaw<Array1<T>, O>,
{
    fn from_raw(&self, raw: &[T]) -> O {
        self.from_raw(Array::from_vec(raw.to_vec()))
    }
}

impl<T, O> FromRaw<Vec<T>, O> for HostPlacement
where
    HostPlacement: FromRaw<Array1<T>, O>,
{
    fn from_raw(&self, raw: Vec<T>) -> O {
        self.from_raw(Array::from_vec(raw))
    }
}

impl<T: Clone, D: ndarray::Dimension> FromRaw<Array<T, D>, HostTensor<T>> for HostPlacement {
    fn from_raw(&self, raw: Array<T, D>) -> HostTensor<T> {
        HostTensor(raw.into_dyn().into_shared(), self.clone())
    }
}

impl<T: Clone, D: ndarray::Dimension> FromRaw<Array<T, D>, HostRingTensor<T>> for HostPlacement {
    fn from_raw(&self, raw: Array<T, D>) -> HostRingTensor<T> {
        HostRingTensor(raw.mapv(Wrapping).into_dyn().into_shared(), self.clone())
    }
}

impl<D: ndarray::Dimension> FromRaw<Array<u8, D>, HostBitTensor> for HostPlacement {
    fn from_raw(&self, raw: Array<u8, D>) -> HostBitTensor {
        let raw = raw.into_dyn();
        let data = raw.as_slice().unwrap().iter().map(|&ai| ai != 0).collect();
        HostBitTensor(BitArrayRepr::from_raw(data, raw.dim()), self.clone())
    }
}

impl<T: Clone, D: ndarray::Dimension, N: Const> FromRaw<Array<T, D>, HostBitArray<HostBitTensor, N>>
    for HostPlacement
where
    HostPlacement: FromRaw<Array<T, D>, HostBitTensor>,
{
    fn from_raw(&self, raw: Array<T, D>) -> HostBitArray<HostBitTensor, N> {
        assert_eq!(raw.shape()[0], N::VALUE);
        let raw_bits: HostBitTensor = self.from_raw(raw);
        HostBitArray(raw_bits, PhantomData)
    }
}

impl FromRaw<RawShape, HostShape> for HostPlacement {
    fn from_raw(&self, raw: RawShape) -> HostShape {
        HostShape(raw, self.clone())
    }
}

impl FromRaw<RawSeed, HostSeed> for HostPlacement {
    fn from_raw(&self, raw: RawSeed) -> HostSeed {
        HostSeed(raw, self.clone())
    }
}

#[cfg(feature = "sync_execute")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
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
    fn test_bit_diag() {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x: HostBitTensor = plc.from_raw(array![[1, 1], [1, 0]]);
        let y = plc.diag(&sess, &x);

        let expected: HostBitTensor = plc.from_raw(array![1, 0]);
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

        let x: HostRing64Tensor = plc.from_raw(array![0_u64, 1, 2, 3]);
        let y = plc.index_axis(&sess, 0, 1, &x);
        let expected: HostRing64Tensor = plc.from_raw(array![1]);
        let y = plc.expand_dims(&sess, [0].to_vec(), &y);
        assert_eq!(y, expected);
    }

    #[rstest]
    #[case(
        array![[[0, 1], [0, 0]], [[1, 1], [0, 0]]].into_dyn(),
        0,
        1,
        array![[1, 1], [0, 0]].into_dyn(),
    )]
    #[case(
        array![[[0, 1], [0, 0]], [[1, 1], [0, 0]]].into_dyn(),
        0,
        0,
        array![[0, 1], [0, 0]].into_dyn(),
    )]
    fn test_index_bit(
        #[case] x: ArrayD<u8>,
        #[case] axis: usize,
        #[case] index: usize,
        #[case] expected: ArrayD<u8>,
    ) {
        let sess = SyncSession::default();
        let plc = HostPlacement::from("host");

        let x: HostBitTensor = plc.from_raw(x);
        let y = plc.index_axis(&sess, axis, index, &x);

        let expected: HostBitTensor = plc.from_raw(expected);
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
        let seed: HostSeed = plc.from_raw(RawSeed([0u8; 16]));
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

    #[test]
    fn bit_shl_dim() {
        let plc = HostPlacement::from("host");
        let sess = SyncSession::default();

        let x: HostRing64Tensor = plc.from_raw(array![6743216615002642708]);
        let x_bits: HostBitTensor = plc.bit_decompose(&sess, &x);
        let x_bits = plc.shl_dim(&sess, 6, 64, &x_bits);

        let expected: HostBitTensor = plc.from_raw(
            array![
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1,
                0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0,
                1, 0, 1, 0, 0, 1, 1, 0,
            ]
            .into_shape((64, 1))
            .unwrap(),
        );
        assert_eq!(x_bits, expected);
    }

    #[test]
    fn test_host_mul() {
        let plc = HostPlacement::from("host");
        let x: HostRing128Tensor = plc.from_raw(array![340282366920938463463374415046855271599]);
        let sess = SyncSession::default();
        let y = plc.mul(&sess, &x, &x);

        let expected: HostRing128Tensor = plc.from_raw(array![37011954726876357358499180449]);
        assert_eq!(y, expected);
    }
}
