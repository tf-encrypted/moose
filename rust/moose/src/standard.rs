use crate::bit::BitTensor;
use crate::computation::{
    HostPlacement, Placed, Placement, ShapeOp, StdAddOp, StdConcatenateOp, StdDivOp, StdDotOp,
    StdExpandDimsOp, StdInverseOp, StdMeanOp, StdMulOp, StdOnesOp, StdSliceOp, StdSubOp, StdSumOp,
    StdTransposeOp,
};
use crate::error::Result;
use crate::kernels::{PlacementPlace, PlacementShape, PlacementSlice, RuntimeSession, SyncSession};
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
pub struct Shape(pub RawShape, pub Placement);

impl Placed for Shape {
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl<S: RuntimeSession> PlacementPlace<S, Shape> for Placement {
    fn place(&self, _sess: &S, shape: Shape) -> Shape {
        match shape.placement() {
            Ok(place) if &place == self => shape,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                Shape(shape.0, self.clone())
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct StandardTensor<T>(pub ArrayD<T>, pub Placement);

impl<T> Placed for StandardTensor<T> {
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

pub type Float32Tensor = StandardTensor<f32>;
pub type Float64Tensor = StandardTensor<f64>;
pub type Int8Tensor = StandardTensor<i8>;
pub type Int16Tensor = StandardTensor<i16>;
pub type Int32Tensor = StandardTensor<i32>;
pub type Int64Tensor = StandardTensor<i64>;
pub type Uint8Tensor = StandardTensor<u8>;
pub type Uint16Tensor = StandardTensor<u16>;
pub type Uint32Tensor = StandardTensor<u32>;
pub type Uint64Tensor = StandardTensor<u64>;

impl<T> PlacementPlace<SyncSession, StandardTensor<T>> for HostPlacement {
    fn place(&self, _sess: &SyncSession, x: StandardTensor<T>) -> StandardTensor<T> {
        match x.placement() {
            Ok(Placement::Host(place)) if &place == self => x,
            _ => StandardTensor(x.0, Placement::Host(self.clone())),
        }
    }
}

/// This implementation is required to do the `plc.place(sess, x)`
impl<T> PlacementPlace<SymbolicSession, Symbolic<StandardTensor<T>>> for HostPlacement {
    fn place(
        &self,
        _sess: &SymbolicSession,
        x: Symbolic<StandardTensor<T>>,
    ) -> Symbolic<StandardTensor<T>> {
        match x {
            Symbolic::Concrete(x) => Symbolic::Concrete(x),
            Symbolic::Symbolic(SymbolicHandle { op, plc: _ }) => {
                Symbolic::Symbolic(SymbolicHandle {
                    op,
                    plc: Placement::Host(self.clone()),
                })
            }
        }
    }
}

impl StdAddOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: StandardTensor<T>,
        y: StandardTensor<T>,
    ) -> StandardTensor<T>
    where
        HostPlacement: PlacementPlace<S, StandardTensor<T>>,
    {
        plc.place(sess, x + y)
    }
}

impl StdSubOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: StandardTensor<T>,
        y: StandardTensor<T>,
    ) -> StandardTensor<T>
    where
        HostPlacement: PlacementPlace<S, StandardTensor<T>>,
    {
        plc.place(sess, x - y)
    }
}

impl StdMulOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: StandardTensor<T>,
        y: StandardTensor<T>,
    ) -> StandardTensor<T>
    where
        HostPlacement: PlacementPlace<S, StandardTensor<T>>,
    {
        plc.place(sess, x * y)
    }
}

impl StdDivOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: StandardTensor<T>,
        y: StandardTensor<T>,
    ) -> StandardTensor<T>
    where
        HostPlacement: PlacementPlace<S, StandardTensor<T>>,
    {
        plc.place(sess, x / y)
    }
}

impl StdDotOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: StandardTensor<T>,
        y: StandardTensor<T>,
    ) -> StandardTensor<T>
    where
        HostPlacement: PlacementPlace<S, StandardTensor<T>>,
    {
        plc.place(sess, x.dot(y))
    }
}

impl StdOnesOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar>(
        sess: &S,
        plc: &HostPlacement,
        shape: Shape,
    ) -> StandardTensor<T>
    where
        HostPlacement: PlacementPlace<S, StandardTensor<T>>,
    {
        plc.place(sess, StandardTensor::ones(shape))
    }
}

modelled!(PlacementShape::shape, HostPlacement, (Ring64Tensor) -> Shape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (Ring128Tensor) -> Shape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (BitTensor) -> Shape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (Float64Tensor) -> Shape, ShapeOp);

kernel! {
    ShapeOp,
    [
        (HostPlacement, (Ring64Tensor) -> Shape => Self::ring_kernel),
        (HostPlacement, (Ring128Tensor) -> Shape => Self::ring_kernel),
        (HostPlacement, (BitTensor) -> Shape => Self::bit_kernel),
        (HostPlacement, (Float64Tensor) -> Shape => Self::std_kernel),
    ]
}

impl ShapeOp {
    pub(crate) fn std_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: StandardTensor<T>,
    ) -> Shape {
        let raw_shape = RawShape(x.0.shape().into());
        Shape(raw_shape, plc.clone().into())
    }
}

modelled!(PlacementSlice::slice, HostPlacement, attributes[start: u32, end: u32] (Shape) -> Shape, StdSliceOp);

kernel! {
    StdSliceOp,
    [
        (HostPlacement, (Shape) -> Shape => attributes[start, end] Self::kernel),
    ]
}

impl StdSliceOp {
    pub(crate) fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        start: u32,
        end: u32,
        x: Shape,
    ) -> Shape {
        let slice = x.0.slice(start as usize, end as usize);
        Shape(slice, plc.clone().into())
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

impl<T> StandardTensor<T>
where
    T: LinalgScalar,
{
    pub fn place(plc: &HostPlacement, x: ArrayD<T>) -> StandardTensor<T> {
        StandardTensor::<T>(x, Placement::Host(plc.clone()))
    }

    pub fn atleast_2d(self, to_column_vector: bool) -> StandardTensor<T> {
        match self.0.ndim() {
            0 => StandardTensor::<T>(self.0.into_shape(IxDyn(&[1, 1])).unwrap(), self.1),
            1 => {
                let length = self.0.len();
                let newshape = if to_column_vector {
                    IxDyn(&[length, 1])
                } else {
                    IxDyn(&[1, length])
                };
                StandardTensor::<T>(self.0.into_shape(newshape).unwrap(), self.1)
            }
            2 => self,
            otherwise => panic!(
                "Tensor input for `atleast_2d` must have rank <= 2, found rank {:?}.",
                otherwise
            ),
        }
    }

    pub fn dot(self, other: StandardTensor<T>) -> StandardTensor<T> {
        match (self.0.ndim(), other.0.ndim()) {
            (1, 1) => {
                let l = self.0.into_dimensionality::<Ix1>().unwrap();
                let r = other.0.into_dimensionality::<Ix1>().unwrap();
                let res = Array::from_elem([], l.dot(&r))
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                StandardTensor::<T>(res, self.1)
            }
            (1, 2) => {
                let l = self.0.into_dimensionality::<Ix1>().unwrap();
                let r = other.0.into_dimensionality::<Ix2>().unwrap();
                let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                StandardTensor::<T>(res, self.1)
            }
            (2, 1) => {
                let l = self.0.into_dimensionality::<Ix2>().unwrap();
                let r = other.0.into_dimensionality::<Ix1>().unwrap();
                let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                StandardTensor::<T>(res, self.1)
            }
            (2, 2) => {
                let l = self.0.into_dimensionality::<Ix2>().unwrap();
                let r = other.0.into_dimensionality::<Ix2>().unwrap();
                let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                StandardTensor::<T>(res, self.1)
            }
            (self_rank, other_rank) => panic!(
                // TODO: replace with proper error handling
                "Dot<StandardTensor> not implemented between tensors of rank {:?} and {:?}.",
                self_rank, other_rank,
            ),
        }
    }

    pub fn ones(shape: Shape) -> Self {
        StandardTensor::<T>(ArrayD::ones(shape.0 .0), shape.1)
    }

    pub fn reshape(self, newshape: Shape) -> Self {
        StandardTensor::<T>(self.0.into_shape(newshape.0 .0).unwrap(), self.1) // TODO need to be fix (unwrap)
    }

    pub fn expand_dims(self, mut axis: Vec<usize>) -> Self {
        let plc = (&self.1).clone();
        axis.sort_by_key(|ax| Reverse(*ax));
        let newshape = self.shape().0.extend_singletons(axis);
        self.reshape(Shape(newshape, plc))
    }

    pub fn shape(&self) -> Shape {
        Shape(RawShape(self.0.shape().into()), self.1.clone())
    }

    pub fn sum(self, axis: Option<usize>) -> Self {
        if let Some(i) = axis {
            StandardTensor::<T>(self.0.sum_axis(Axis(i)), self.1)
        } else {
            let out = Array::from_elem([], self.0.sum())
                .into_dimensionality::<IxDyn>()
                .unwrap();
            StandardTensor::<T>(out, self.1)
        }
    }

    pub fn transpose(self) -> Self {
        StandardTensor::<T>(self.0.reversed_axes(), self.1)
    }
}

impl<T> StandardTensor<T>
where
    T: LinalgScalar + FromPrimitive,
{
    pub fn mean(self, axis: Option<usize>) -> Self {
        match axis {
            Some(i) => {
                let reduced = self.0.mean_axis(Axis(i)).unwrap();
                StandardTensor::<T>(reduced, self.1)
            }
            None => {
                let mean = self.0.mean().unwrap();
                let out = Array::from_elem([], mean)
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                StandardTensor::<T>(out, self.1)
            }
        }
    }
}

impl StdMeanOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: StandardTensor<T>,
    ) -> StandardTensor<T>
    where
        HostPlacement: PlacementPlace<S, StandardTensor<T>>,
    {
        match axis {
            Some(i) => {
                let reduced: ArrayD<T> = x.0.mean_axis(Axis(i as usize)).unwrap();
                StandardTensor::place(plc, reduced)
            }
            None => {
                let mean = x.0.mean().unwrap();
                let out = Array::from_elem([], mean)
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                StandardTensor::place(plc, out)
            }
        }
    }
}

impl StdSumOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: StandardTensor<T>,
    ) -> StandardTensor<T>
    where
        HostPlacement: PlacementPlace<S, StandardTensor<T>>,
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
        x: StandardTensor<T>,
    ) -> StandardTensor<T>
    where
        HostPlacement: PlacementPlace<S, StandardTensor<T>>,
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
        x: StandardTensor<T>,
        y: StandardTensor<T>,
    ) -> StandardTensor<T> {
        let ax = Axis(axis as usize);
        let x = x.0.view();
        let y = y.0.view();

        let c =
            ndarray::concatenate(ax, &[x, y]).expect("Failed to concatenate arrays with ndarray");
        StandardTensor(c, plc.clone().into())
    }
}

impl StdTransposeOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: StandardTensor<T>,
    ) -> StandardTensor<T>
    where
        HostPlacement: PlacementPlace<S, StandardTensor<T>>,
    {
        plc.place(sess, x.transpose())
    }
}

impl StdInverseOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive + Lapack>(
        _sess: &S,
        plc: &HostPlacement,
        x: StandardTensor<T>,
    ) -> StandardTensor<T> {
        match x.0.ndim() {
            2 => {
                let dim2 = x.0.into_dimensionality::<Ix2>().unwrap();
                let inv = Inverse::inv(&dim2).unwrap();
                let dim_d = inv.into_dimensionality::<IxDyn>().unwrap();
                StandardTensor::<T>(dim_d, plc.clone().into())
            }
            _ => unimplemented!("No implementation for reversing non-2D tensors"),
        }
    }
}

impl<T> StandardTensor<T>
where
    T: Scalar + Lapack,
{
    pub fn inv(self) -> Self {
        match self.0.ndim() {
            2 => {
                let two_dim: Array2<T> = self.0.into_dimensionality::<Ix2>().unwrap();
                StandardTensor::<T>::from(
                    two_dim
                        .inv()
                        .unwrap()
                        .into_dimensionality::<IxDyn>()
                        .unwrap(),
                )
            }
            other_rank => panic!(
                "Inverse only defined for rank 2 matrices, not rank {:?}",
                other_rank,
            ),
        }
    }
}

// This implementation is only used by the old kernels. Construct StandardTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "symbolic"))]
impl<T> From<ArrayD<T>> for StandardTensor<T>
where
    T: LinalgScalar,
{
    fn from(v: ArrayD<T>) -> StandardTensor<T> {
        StandardTensor::<T>(
            v,
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the old kernels
            }
            .into(),
        )
    }
}

impl<T> Add for StandardTensor<T>
where
    T: LinalgScalar,
{
    type Output = StandardTensor<T>;
    fn add(self, other: StandardTensor<T>) -> Self::Output {
        match self.0.broadcast(other.0.dim()) {
            Some(self_broadcasted) => {
                StandardTensor::<T>(self_broadcasted.to_owned() + other.0, self.1.clone())
            }
            None => StandardTensor::<T>(self.0 + other.0, self.1.clone()),
        }
    }
}

impl<T> Sub for StandardTensor<T>
where
    T: LinalgScalar,
{
    type Output = StandardTensor<T>;
    fn sub(self, other: StandardTensor<T>) -> Self::Output {
        match self.0.broadcast(other.0.dim()) {
            Some(self_broadcasted) => {
                StandardTensor::<T>(self_broadcasted.to_owned() - other.0, self.1.clone())
            }
            None => StandardTensor::<T>(self.0 - other.0, self.1.clone()),
        }
    }
}

impl<T> Mul for StandardTensor<T>
where
    T: LinalgScalar,
{
    type Output = StandardTensor<T>;
    fn mul(self, other: StandardTensor<T>) -> Self::Output {
        match self.0.broadcast(other.0.dim()) {
            Some(self_broadcasted) => {
                StandardTensor::<T>(self_broadcasted.to_owned() * other.0, self.1.clone())
            }
            None => StandardTensor::<T>(self.0 * other.0, self.1.clone()),
        }
    }
}

impl<T> Div for StandardTensor<T>
where
    T: LinalgScalar,
{
    type Output = StandardTensor<T>;
    fn div(self, other: StandardTensor<T>) -> Self::Output {
        match self.0.broadcast(other.0.dim()) {
            Some(self_broadcasted) => {
                StandardTensor::<T>(self_broadcasted.to_owned() / other.0, self.1.clone())
            }
            None => StandardTensor::<T>(self.0 / other.0, self.1.clone()),
        }
    }
}

// This implementation is only used by the old kernels. Construct StandardTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "symbolic"))]
impl<T> From<Vec<T>> for StandardTensor<T> {
    fn from(v: Vec<T>) -> StandardTensor<T> {
        StandardTensor(
            Array::from(v).into_dyn(),
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the old kernel
            }
            .into(),
        )
    }
}

// This implementation is only used by the old kernels. Construct StandardTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "symbolic"))]
impl<T> From<Array1<T>> for StandardTensor<T> {
    fn from(v: Array1<T>) -> StandardTensor<T> {
        StandardTensor(
            v.into_dyn(),
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the old kernel
            }
            .into(),
        )
    }
}

// This implementation is only used by the old kernels. Construct StandardTensor(tensor, plc.clone()) with a proper placement instead.
#[cfg(not(feature = "symbolic"))]
impl<T> From<Array2<T>> for StandardTensor<T> {
    fn from(v: Array2<T>) -> StandardTensor<T> {
        StandardTensor(
            v.into_dyn(),
            HostPlacement {
                owner: "TODO".into(), // Fake owner for the old kernel
            }
            .into(),
        )
    }
}

pub fn concatenate<T>(axis: usize, arrays: &[StandardTensor<T>]) -> StandardTensor<T>
where
    T: LinalgScalar,
{
    let ax = Axis(axis);
    let inner_arrays: Vec<_> = arrays.iter().map(|a| a.0.view()).collect();

    let c = ndarray::concatenate(ax, &inner_arrays).unwrap();
    StandardTensor::<T>(
        c,
        HostPlacement {
            owner: "TODO".into(),
        }
        .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_prod_f32() {
        let x = StandardTensor::<f32>::from(
            array![[1.0, -2.0], [3.0, -4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let y = x.clone();
        let z = x.dot(y);
        assert_eq!(
            z,
            StandardTensor::<f32>::from(
                array![[-5.0, 6.0], [-9.0, 10.0]]
                    .into_dimensionality::<IxDyn>()
                    .unwrap()
            )
        );
    }

    #[test]
    fn test_inverse() {
        let x = StandardTensor::<f32>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );

        let x_inv = x.inv();

        assert_eq!(
            x_inv,
            StandardTensor::<f32>::from(
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
        let x = StandardTensor::<f32>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let y = x.transpose();
        assert_eq!(
            y,
            StandardTensor::<f32>::from(
                array![[1.0, 3.0], [2.0, 4.0]]
                    .into_dimensionality::<IxDyn>()
                    .unwrap()
            )
        );
    }

    #[test]
    fn test_concatenate() {
        let a = StandardTensor::<f32>::from(
            array![[[1.0, 2.0], [3.0, 4.0]]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let b = StandardTensor::<f32>::from(
            array![[[1.0, 2.0], [3.0, 4.0]]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let expected = StandardTensor::<f32>::from(
            array![[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let conc = concatenate(0, &vec![a, b]);
        assert_eq!(conc, expected)
    }

    #[test]
    fn test_atleast_2d() {
        let a = StandardTensor::<f32>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let a_exp = a.clone();
        let b = StandardTensor::<f32>::from(
            array![1.0, 2.0, 3.0, 4.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let b_exp = StandardTensor::<f32>::from(
            array![[1.0, 2.0, 3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let c = StandardTensor::<f32>::from(
            array![1.0, 2.0, 3.0, 4.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let c_exp = StandardTensor::<f32>::from(
            array![[1.0], [2.0], [3.0], [4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let d = StandardTensor::<f32>::from(
            Array::from_elem([], 1.0)
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let d_exp =
            StandardTensor::<f32>::from(array![[1.0]].into_dimensionality::<IxDyn>().unwrap());
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
        let x_1 = StandardTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_1 =
            StandardTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.add(y_1);
        let z_1_exp =
            StandardTensor::<f32>::from(array![3.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let x_2 =
            StandardTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_2 = StandardTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_2 = x_2.add(y_2);
        let z_2_exp =
            StandardTensor::<f32>::from(array![3.0, 4.0].into_dimensionality::<IxDyn>().unwrap());

        assert_eq!(z_1, z_1_exp);
        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_sub_broadcasting() {
        let x_1 = StandardTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_1 =
            StandardTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.sub(y_1);
        let z_1_exp =
            StandardTensor::<f32>::from(array![1.0, 0.0].into_dimensionality::<IxDyn>().unwrap());
        let x_2 =
            StandardTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_2 = StandardTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_2 = x_2.sub(y_2);
        let z_2_exp =
            StandardTensor::<f32>::from(array![-1.0, 0.0].into_dimensionality::<IxDyn>().unwrap());

        assert_eq!(z_1, z_1_exp);
        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_mul_broadcasting() {
        let x_1 = StandardTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_1 =
            StandardTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.mul(y_1);
        let z_1_exp =
            StandardTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let x_2 =
            StandardTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());
        let y_2 = StandardTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_2 = x_2.mul(y_2);
        let z_2_exp =
            StandardTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());

        assert_eq!(z_1, z_1_exp);
        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_div_broadcasting() {
        let x_1 = StandardTensor::<f32>::from(array![1.0].into_dimensionality::<IxDyn>().unwrap());
        let y_1 =
            StandardTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.div(y_1);
        let z_1_exp =
            StandardTensor::<f32>::from(array![0.5, 0.25].into_dimensionality::<IxDyn>().unwrap());
        let x_2 =
            StandardTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let y_2 = StandardTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_2 = x_2.div(y_2);
        let z_2_exp =
            StandardTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());

        assert_eq!(z_1, z_1_exp);
        assert_eq!(z_2, z_2_exp);
    }

    #[test]
    fn test_kernel_inverse() {
        use crate::kernels::PlacementStdInverse;
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let sess = SyncSession::default();
        let x = crate::standard::StandardTensor::<f64>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let inv = alice.std_inverse(&sess, &x);
        assert_eq!("[[-2, 1],\n [1.5, -0.5]]", format!("{}", inv.0));
    }

    #[test]
    fn test_kernel_transpose() {
        use crate::kernels::PlacementStdTranspose;
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let sess = SyncSession::default();
        let x = crate::standard::StandardTensor::<f64>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let t = alice.std_transpose(&sess, &x);
        assert_eq!("[[1, 3],\n [2, 4]]", format!("{}", t.0));
    }

    #[test]
    fn test_kernel_concatenate() {
        use crate::kernels::PlacementStdConcatenate;
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let sess = SyncSession::default();
        let x = crate::standard::StandardTensor::<f64>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let y = crate::standard::StandardTensor::<f64>::from(
            array![[5.0, 6.0], [7.0, 8.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let c = alice.std_concatenate(&sess, 0, &x, &y);
        assert_eq!("[[1, 2],\n [3, 4],\n [5, 6],\n [7, 8]]", format!("{}", c.0));
    }
}
