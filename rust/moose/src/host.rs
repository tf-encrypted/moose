use crate::bit::BitTensor;
use crate::computation::{
    HostPlacement, Placed, Placement, ReplicatedPlacement, ShapeOp, StdAddOp, StdConcatenateOp,
    StdDivOp, StdDotOp, StdExpandDimsOp, StdInverseOp, StdMeanOp, StdMulOp, StdOnesOp, StdSliceOp,
    StdSubOp, StdSumOp, StdTransposeOp,
};
use crate::error::Result;
use crate::kernels::{PlacementPlace, PlacementShape, PlacementSlice, RuntimeSession, SyncSession};
use crate::replicated::{
    Replicated128Tensor, Replicated64Tensor, ReplicatedBitTensor, ReplicatedShape,
};
use crate::ring::{Ring128Tensor, Ring64Tensor};
use crate::symbolic::{Symbolic, SymbolicHandle, SymbolicSession};
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use ndarray_linalg::types::{Lapack, Scalar};
use ndarray_linalg::*;
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::ops::{Add, Div, Mul, Sub}; // related to TODOs

impl Placed for String {
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        // TODO we need a wrapper for strings that contains placement info
        unimplemented!()
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawShape(pub Vec<usize>);

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

// TODO these should have Host prefix
pub type HostFloat32Tensor = HostTensor<f32>;
pub type Float64Tensor = HostTensor<f64>;
pub type Int8Tensor = HostTensor<i8>;
pub type Int16Tensor = HostTensor<i16>;
pub type Int32Tensor = HostTensor<i32>;
pub type Int64Tensor = HostTensor<i64>;
pub type Uint8Tensor = HostTensor<u8>;
pub type Uint16Tensor = HostTensor<u16>;
pub type Uint32Tensor = HostTensor<u32>;
pub type Uint64Tensor = HostTensor<u64>;

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

impl StdAddOp {
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

impl StdSubOp {
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

impl StdMulOp {
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

impl StdDivOp {
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

impl StdDotOp {
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

impl StdOnesOp {
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

modelled!(PlacementShape::shape, HostPlacement, (Ring64Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (Ring128Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (BitTensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (Float64Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (Replicated64Tensor) -> ReplicatedShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (Replicated128Tensor) -> ReplicatedShape, ShapeOp);

kernel! {
    ShapeOp,
    [
        (HostPlacement, (Ring64Tensor) -> HostShape => Self::ring_kernel),
        (HostPlacement, (Ring128Tensor) -> HostShape => Self::ring_kernel),
        (HostPlacement, (BitTensor) -> HostShape => Self::bit_kernel),
        (HostPlacement, (Float64Tensor) -> HostShape => Self::std_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedShape => Self::rep_kernel),
        (ReplicatedPlacement, (Replicated64Tensor) -> ReplicatedShape => Self::rep_kernel),
        (ReplicatedPlacement, (Replicated128Tensor) -> ReplicatedShape => Self::rep_kernel),
    ]
}

impl ShapeOp {
    pub(crate) fn std_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> HostShape {
        let raw_shape = RawShape(x.0.shape().into());
        HostShape(raw_shape, plc.clone())
    }
}

modelled!(PlacementSlice::slice, HostPlacement, attributes[start: u32, end: u32] (HostShape) -> HostShape, StdSliceOp);

kernel! {
    StdSliceOp,
    [
        (HostPlacement, (HostShape) -> HostShape => attributes[start, end] Self::kernel),
    ]
}

impl StdSliceOp {
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

impl StdMeanOp {
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

impl StdSumOp {
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

impl StdExpandDimsOp {
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

impl StdConcatenateOp {
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

impl StdTransposeOp {
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

impl StdInverseOp {
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
        let d_exp =
            HostTensor::<f32>::from(array![[1.0]].into_dimensionality::<IxDyn>().unwrap());
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
        let y_1 =
            HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.add(y_1);
        let z_1_exp =
            HostTensor::<f32>::from(array![3.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let x_2 =
            HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
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
        let y_1 =
            HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.sub(y_1);
        let z_1_exp =
            HostTensor::<f32>::from(array![1.0, 0.0].into_dimensionality::<IxDyn>().unwrap());
        let x_2 =
            HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
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
        let y_1 =
            HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.mul(y_1);
        let z_1_exp =
            HostTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let x_2 =
            HostTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
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
        let y_1 =
            HostTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.div(y_1);
        let z_1_exp =
            HostTensor::<f32>::from(array![0.5, 0.25].into_dimensionality::<IxDyn>().unwrap());
        let x_2 =
            HostTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
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
}
