use super::*;
use crate::error::{Error, Result};
use crate::execution::{RuntimeSession, Session};
use crate::prng::AesRng;
use crate::{Const, Ring, N128, N224, N64};
use ndarray::LinalgScalar;
#[cfg(feature = "blas")]
use ndarray_linalg::{Inverse, Lapack};
use num_traits::{Float, FromPrimitive, Zero};
use std::marker::PhantomData;
use std::num::Wrapping;

impl ConstantOp {
    pub(crate) fn kernel<S: RuntimeSession, T: Placed>(
        sess: &S,
        plc: &HostPlacement,
        value: T,
    ) -> Result<T>
    where
        HostPlacement: PlacementPlace<S, T>,
    {
        Ok(plc.place(sess, value))
    }
}

macro_rules! wrapping_constant_kernel {
    ($name:ident for $wrapping:tt($inner:ty)) => {
        impl ConstantOp {
            pub(crate) fn $name<S: RuntimeSession>(
                _sess: &S,
                plc: &HostPlacement,
                value: $inner,
            ) -> Result<$wrapping> {
                Ok($wrapping(value.clone(), plc.clone()))
            }
        }
    };
}

wrapping_constant_kernel!(string_kernel for HostString(String));
wrapping_constant_kernel!(shape_kernel for HostShape(RawShape));
wrapping_constant_kernel!(prf_key_kernel for PrfKey(RawPrfKey));
wrapping_constant_kernel!(seed_kernel for Seed(RawSeed));

impl SendOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _receiver: Role,
        _x: T,
    ) -> Result<Unit>
    where
        Value: From<T>,
    {
        // let x: Value = x.into();
        // sess.networking.send(&x, &receiver, &rendezvous_key)?;
        // Ok(Unit(plc.clone()))
        todo!()
    }
}

impl ReceiveOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _sender: Role,
    ) -> Result<T>
    where
        T: TryFrom<Value, Error = Error>,
        T: std::fmt::Debug,
        HostPlacement: PlacementPlace<S, T>,
    {
        // use std::convert::TryInto;
        // let value = sess.networking.receive(&sender, &rendezvous_key)?;
        // Ok(plc.place(sess, value.try_into()?))
        todo!()
    }

    pub(crate) fn missing_kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _sender: Role,
    ) -> Result<T>
    where
        T: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<T as KnownType<S>>::TY
        )))
    }
}

impl IdentityOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(sess: &S, plc: &HostPlacement, x: T) -> Result<T>
    where
        HostPlacement: PlacementPlace<S, T>,
    {
        let value = plc.place(sess, x);
        Ok(value)
    }

    pub(crate) fn missing_kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _x: T,
    ) -> Result<T>
    where
        T: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<T as KnownType<S>>::TY
        )))
    }
}

impl InputOp {
    pub(crate) fn kernel<S: RuntimeSession, O>(
        sess: &S,
        plc: &HostPlacement,
        arg_name: String,
    ) -> Result<O>
    where
        O: TryFrom<Value, Error = Error>,
        HostPlacement: PlacementPlace<S, O>,
    {
        use std::convert::TryInto;
        let value = sess
            .find_argument(&arg_name)
            .ok_or_else(|| Error::MissingArgument(arg_name.clone()))?;
        let value = plc.place(sess, value.try_into()?);
        Ok(value)
    }

    pub(crate) fn missing_kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _arg_name: String,
    ) -> Result<O>
    where
        O: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<O as KnownType<S>>::TY
        )))
    }

    pub(crate) fn host_bitarray64<S: Session, HostBitTensorT>(
        sess: &S,
        plc: &HostPlacement,
        arg_name: String,
    ) -> Result<HostBitArray<HostBitTensorT, N64>>
    where
        HostPlacement: PlacementInput<S, HostBitTensorT>,
    {
        // TODO(Morten) ideally we should verify that shape of bit tensor
        let bit_tensor = plc.input(sess, arg_name);
        Ok(HostBitArray(bit_tensor, PhantomData))
    }

    pub(crate) fn host_bitarray128<S: Session, HostBitTensorT>(
        sess: &S,
        plc: &HostPlacement,
        arg_name: String,
    ) -> Result<HostBitArray<HostBitTensorT, N128>>
    where
        HostPlacement: PlacementInput<S, HostBitTensorT>,
    {
        // TODO(Morten) ideally we should verify that shape of bit tensor
        let bit_tensor = plc.input(sess, arg_name);
        Ok(HostBitArray(bit_tensor, PhantomData))
    }

    pub(crate) fn host_bitarray224<S: Session, HostBitTensorT>(
        sess: &S,
        plc: &HostPlacement,
        arg_name: String,
    ) -> Result<HostBitArray<HostBitTensorT, N224>>
    where
        HostPlacement: PlacementInput<S, HostBitTensorT>,
    {
        // TODO(Morten) ideally we should verify that shape of bit tensor
        let bit_tensor = plc.input(sess, arg_name);
        Ok(HostBitArray(bit_tensor, PhantomData))
    }
}

impl OutputOp {
    pub(crate) fn kernel<S: RuntimeSession, O>(sess: &S, plc: &HostPlacement, x: O) -> Result<O>
    where
        HostPlacement: PlacementPlace<S, O>,
    {
        // Output is not doing anything now, it is just a marker on the graph.
        // But it has to return a value because that's how we collect outputs in the old framework
        let x = plc.place(sess, x);
        Ok(x)
    }

    pub(crate) fn non_placing_kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        x: O,
    ) -> Result<O> {
        // Output is not doing anything now, it is just a marker on the graph.
        // But it has to return a value because that's how we collect outputs in the old framework
        Ok(x)
    }
}

impl LoadOp {
    pub(crate) fn kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: HostString,
        _query: HostString,
    ) -> Result<O>
    where
        O: KnownType<S>,
        O: TryFrom<Value, Error = Error>,
        HostPlacement: PlacementPlace<S, O>,
    {
        // use std::convert::TryInto;
        // let value = sess.storage.load(&key.0, &query.0, Some(<O as KnownType<S>>::TY))?;
        // let value = plc.place(sess, value.try_into()?);
        // Ok(value)
        todo!()
    }

    pub(crate) fn missing_kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: HostString,
        _query: HostString,
    ) -> Result<O>
    where
        O: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<O as KnownType<S>>::TY
        )))
    }
}

impl SaveOp {
    pub(crate) fn kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: HostString,
        _x: O,
    ) -> Result<Unit>
    where
        Value: From<O>,
    {
        // let x: Value = x.into();
        // sess.storage.save(&key.0, &x)?;
        // Ok(Unit(plc.clone()))
        todo!()
    }
}

impl HostAddOp {
    pub(crate) fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(HostTensor(x.0 + y.0, plc.clone()))
    }
}

impl HostSubOp {
    pub(crate) fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(HostTensor(x.0 - y.0, plc.clone()))
    }
}

impl HostMulOp {
    pub(crate) fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(HostTensor(x.0 * y.0, plc.clone()))
    }
}

impl HostDivOp {
    pub(crate) fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        match x.0.broadcast(y.0.dim()) {
            Some(x_broadcasted) => Ok(HostTensor::<T>(x_broadcasted.to_owned() / y.0, plc.clone())),
            None => Ok(HostTensor::<T>(x.0 / y.0, plc.clone())),
        }
    }

    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRingTensor<T>,
        y: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Div<Wrapping<T>, Output = Wrapping<T>>,
    {
        Ok(HostRingTensor(x.0 / y.0, plc.clone()))
    }
}

impl<T> HostTensor<T>
where
    T: LinalgScalar,
{
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
}

impl HostDotOp {
    pub(crate) fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(plc.place(sess, x.dot(y)))
    }
}

impl HostOnesOp {
    pub(crate) fn kernel<S: RuntimeSession, T: LinalgScalar>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> Result<HostTensor<T>> {
        let raw_shape = shape.0;
        Ok(HostTensor::<T>(ArrayD::ones(raw_shape.0), plc.clone()))
    }
}

impl ShapeOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> Result<HostShape> {
        let raw_shape = RawShape(x.0.shape().into());
        Ok(HostShape(raw_shape, plc.clone()))
    }
}

impl HostAtLeast2DOp {
    pub(crate) fn kernel<S: RuntimeSession, T: LinalgScalar>(
        sess: &S,
        plc: &HostPlacement,
        to_column_vector: bool,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let y = match x.0.ndim() {
            0 => HostTensor::<T>(x.0.into_shape(IxDyn(&[1, 1])).unwrap(), x.1),
            1 => {
                let length = x.0.len();
                let newshape = if to_column_vector {
                    IxDyn(&[length, 1])
                } else {
                    IxDyn(&[1, length])
                };
                HostTensor::<T>(x.0.into_shape(newshape).unwrap(), x.1)
            }
            2 => x,
            otherwise => panic!(
                "Tensor input for `at_least_2d` must have rank <= 2, found rank {:?}.",
                otherwise
            ),
        };
        Ok(plc.place(sess, y))
    }
}

impl SliceOp {
    // TODO(lvorona): type inferring fails if I try to make it more generic and have one kernel work for all the types
    pub(crate) fn kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        slice_info: SliceInfo,
        x: cs!(HostShape),
    ) -> Result<cs!(HostShape)>
    where
        HostShape: KnownType<S>,
        HostPlacement: PlacementSlice<S, cs!(HostShape), cs!(HostShape)>,
    {
        Ok(plc.slice(sess, slice_info, &x))
    }
}

impl HostSliceOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        slice_info: SliceInfo,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        T: Clone,
    {
        let slice_info =
            ndarray::SliceInfo::<Vec<ndarray::SliceInfoElem>, IxDyn, IxDyn>::from(slice_info);
        let sliced = x.0.slice(slice_info).to_owned();
        Ok(HostRingTensor(sliced, plc.clone()))
    }

    pub(crate) fn shape_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        slice_info: SliceInfo,
        x: HostShape,
    ) -> Result<HostShape> {
        let slice = x.0.slice(
            slice_info.0[0].start as usize,
            slice_info.0[0].end.unwrap() as usize,
        );
        Ok(HostShape(slice, plc.clone()))
    }
}

impl HostDiagOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>> {
        let diag =
            x.0.into_diag()
                .into_dimensionality::<IxDyn>()
                .map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostTensor::<T>(diag, plc.clone()))
    }

    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>> {
        let diag =
            x.0.into_diag()
                .into_dimensionality::<IxDyn>()
                .map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor::<T>(diag, plc.clone()))
    }

    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
    ) -> Result<HostBitTensor> {
        let diag =
            x.0.into_diag()
                .into_dimensionality::<IxDyn>()
                .map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostBitTensor(diag, plc.clone()))
    }
}

impl<T: LinalgScalar> HostTensor<T> {
    fn index_axis(&self, axis: usize, index: usize) -> Result<HostTensor<T>> {
        if axis >= self.0.ndim() {
            return Err(Error::InvalidArgument(format!(
                "axis too large in index axis, used axis {} with dimension {}",
                axis,
                self.0.ndim()
            )));
        }
        if index >= self.0.shape()[axis] {
            return Err(Error::InvalidArgument(format!(
                "index too large in index axis, used index {} in shape {:?}",
                index,
                self.0.shape()
            )));
        }
        let axis = Axis(axis);
        let result = self.0.index_axis(axis, index);
        Ok(HostTensor(result.to_owned(), self.1.clone()))
    }
}

impl<T: Clone> HostRingTensor<T> {
    fn index_axis(self, axis: usize, index: usize) -> Result<HostRingTensor<T>> {
        if axis >= self.0.ndim() {
            return Err(Error::InvalidArgument(format!(
                "axis too large in index axis, used axis {} with dimension {}",
                axis,
                self.0.ndim()
            )));
        }
        if index >= self.0.shape()[axis] {
            return Err(Error::InvalidArgument(format!(
                "index too large in index axis, used index {} in shape {:?}",
                index,
                self.0.shape()
            )));
        }
        let axis = Axis(axis);
        let result = self.0.index_axis(axis, index);
        Ok(HostRingTensor(result.to_owned(), self.1))
    }
}

impl HostBitTensor {
    fn index_axis(self, axis: usize, index: usize) -> Result<HostBitTensor> {
        if axis >= self.0.ndim() {
            return Err(Error::InvalidArgument(format!(
                "axis too large in index axis, used axis {} with dimension {}",
                axis,
                self.0.ndim()
            )));
        }
        if index >= self.0.shape()[axis] {
            return Err(Error::InvalidArgument(format!(
                "index too large in index axis, used index {} in shape {:?}",
                index,
                self.0.shape()
            )));
        }
        let axis = Axis(axis);
        let result = self.0.index_axis(axis, index);
        Ok(HostBitTensor(result.to_owned(), self.1))
    }
}

impl IndexAxisOp {
    pub(crate) fn host_float_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(plc.place(sess, x.index_axis(axis, index)?))
    }

    pub(crate) fn host_bit_kernel<S: RuntimeSession>(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: HostBitTensor,
    ) -> Result<HostBitTensor>
    where
        HostPlacement: PlacementPlace<S, HostBitTensor>,
    {
        Ok(plc.place(sess, x.index_axis(axis, index)?))
    }

    pub(crate) fn host_ring_kernel<S: RuntimeSession, T>(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        T: Clone,
        HostPlacement: PlacementPlace<S, HostRingTensor<T>>,
    {
        Ok(plc.place(sess, x.index_axis(axis, index)?))
    }
}

impl IndexOp {
    pub(crate) fn host_kernel<S: Session, HostBitT, N>(
        sess: &S,
        plc: &HostPlacement,
        index: usize,
        x: HostBitArray<HostBitT, N>,
    ) -> Result<HostBitT>
    where
        HostPlacement: PlacementIndexAxis<S, HostBitT, HostBitT>,
    {
        Ok(plc.index_axis(sess, 0, index, &x.0))
    }
}

impl HostShlDimOp {
    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        amount: usize,
        bit_length: usize,
        x: HostBitTensor,
    ) -> Result<HostBitTensor> {
        let axis = Axis(0);
        let mut raw_tensor_shape = x.0.shape().to_vec();
        raw_tensor_shape.remove(0);
        let raw_shape = raw_tensor_shape.as_ref();

        let zero = ArrayD::from_elem(raw_shape, 0);
        let zero_view = zero.view();

        let concatenated: Vec<_> = (0..bit_length)
            .map(|i| {
                if i < amount {
                    zero_view.clone()
                } else {
                    x.0.index_axis(axis, i - amount)
                }
            })
            .collect();

        let result = ndarray::stack(Axis(0), &concatenated)
            .map_err(|e| Error::KernelError(e.to_string()))?;

        Ok(HostBitTensor(result, plc.clone()))
    }
}

impl HostBitDecOp {
    pub(crate) fn ring64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing64Tensor,
    ) -> Result<HostRing64Tensor> {
        let shape = x.shape();
        let raw_shape = shape.0 .0;
        let ones = ArrayD::from_elem(raw_shape, Wrapping(1));

        let bit_rep: Vec<_> = (0..<HostRing64Tensor as Ring>::BitLength::VALUE)
            .map(|i| (&x.0 >> i) & (&ones))
            .collect();
        let bit_rep_view: Vec<_> = bit_rep.iter().map(ArrayView::from).collect();

        // by default we put bits as rows, ie access i'th bit from tensor T is done through index_axis(Axis(0), T)
        // in the current protocols it's easier to reason that the bits are stacked on axis(0)
        let result = ndarray::stack(Axis(0), &bit_rep_view)
            .map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(result, plc.clone()))
    }

    pub(crate) fn ring128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing128Tensor,
    ) -> Result<HostRing128Tensor> {
        let shape = x.shape();
        let raw_shape = shape.0 .0;
        let ones = ArrayD::from_elem(raw_shape, Wrapping(1));

        let bit_rep: Vec<_> = (0..<HostRing128Tensor as Ring>::BitLength::VALUE)
            .map(|i| (&x.0 >> i) & (&ones))
            .collect();

        let bit_rep_view: Vec<_> = bit_rep.iter().map(ArrayView::from).collect();
        let result = ndarray::stack(Axis(0), &bit_rep_view)
            .map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(result, plc.clone()))
    }

    pub(crate) fn bit64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing64Tensor,
    ) -> Result<HostBitTensor> {
        let shape = x.shape();
        let raw_shape = shape.0 .0;
        let ones = ArrayD::from_elem(raw_shape, Wrapping(1));

        let bit_rep: Vec<_> = (0..<HostRing64Tensor as Ring>::BitLength::VALUE)
            .map(|i| (&x.0 >> i) & (&ones))
            .collect();

        let bit_rep_view: Vec<_> = bit_rep.iter().map(ArrayView::from).collect();
        let result = ndarray::stack(Axis(0), &bit_rep_view)
            .map_err(|e| Error::KernelError(e.to_string()))?;
        // we unwrap only at the end since shifting can cause overflow
        Ok(HostBitTensor(result.map(|v| v.0 as u8), plc.clone()))
    }

    pub(crate) fn bit128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing128Tensor,
    ) -> Result<HostBitTensor> {
        let shape = x.shape();
        let raw_shape = shape.0 .0;
        let ones = ArrayD::from_elem(raw_shape, Wrapping(1));

        let bit_rep: Vec<_> = (0..<HostRing128Tensor as Ring>::BitLength::VALUE)
            .map(|i| (&x.0 >> i) & (&ones))
            .collect();

        let bit_rep_view: Vec<_> = bit_rep.iter().map(ArrayView::from).collect();
        let result = ndarray::stack(Axis(0), &bit_rep_view)
            .map_err(|e| Error::KernelError(e.to_string()))?;
        // we unwrap only at the end since shifting can cause overflow
        Ok(HostBitTensor(result.map(|v| v.0 as u8), plc.clone()))
    }
}

impl HostMeanOp {
    pub(crate) fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        match axis {
            Some(i) => {
                let reduced: Option<ArrayD<T>> = x.0.mean_axis(Axis(i as usize));
                if reduced.is_none() {
                    return Err(Error::KernelError(
                        "HostMeanOp cannot reduce over an empty axis.".to_string(),
                    ));
                };
                Ok(HostTensor::place(plc, reduced.unwrap()))
            }
            None => {
                let mean = x.0.mean();
                if mean.is_none() {
                    return Err(Error::KernelError(
                        "HostMeanOp cannot reduce over an empty tensor.".to_string(),
                    ));
                };
                let out = Array::from_elem([], mean.unwrap())
                    .into_dimensionality::<IxDyn>()
                    .map_err(|e| Error::KernelError(e.to_string()))?;
                Ok(HostTensor::place(plc, out))
            }
        }
    }
}

impl HostSqrtOp {
    pub(crate) fn kernel<S: RuntimeSession, T: 'static + Float>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let x_sqrt = x.0.mapv(T::sqrt);
        Ok(HostTensor::place(plc, x_sqrt))
    }
}

// TODO(Morten) inline
impl<T: LinalgScalar> HostTensor<T> {
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
}

impl SumOp {
    pub(crate) fn host_float_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<usize>,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let axis = axis.map(|a| a as usize);
        Ok(plc.place(sess, x.sum(axis)?))
    }

    pub(crate) fn host_ring_kernel<S: RuntimeSession, T>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<usize>,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        T: FromPrimitive + Zero,
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Add<Wrapping<T>, Output = Wrapping<T>>,
        HostPlacement: PlacementPlace<S, HostRingTensor<T>>,
    {
        let sum = x.sum(axis.map(|a| a as usize))?;
        Ok(plc.place(sess, sum))
    }
}

impl AddNOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        xs: &[HostRingTensor<T>],
    ) -> Result<HostRingTensor<T>>
    where
        T: Clone + LinalgScalar,
        Wrapping<T>: std::ops::Add<Wrapping<T>, Output = Wrapping<T>>,
    {
        if xs.is_empty() {
            Err(Error::InvalidArgument(
                "cannot reduce on empty array of tensors".to_string(),
            ))
        } else {
            let base = xs[0].0.clone();
            let sum = xs[1..].iter().fold(base, |acc, item| acc + &item.0);
            Ok(HostRingTensor(sum, plc.clone()))
        }
    }

    pub(crate) fn host_float_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        xs: &[HostTensor<T>],
    ) -> Result<HostTensor<T>>
    where
        T: Clone + LinalgScalar,
    {
        if xs.is_empty() {
            Err(Error::InvalidArgument(
                "cannot reduce on empty array of tensors".to_string(),
            ))
        } else {
            let base = xs[0].0.clone();
            let sum = xs[1..].iter().fold(base, |acc, item| acc + &item.0);
            Ok(HostTensor(sum, plc.clone()))
        }
    }
}

// TODO(Morten) inline
impl<T: LinalgScalar> HostTensor<T> {
    fn expand_dims(self, mut axis: Vec<usize>) -> Self {
        let plc = (&self.1).clone();
        axis.sort_by_key(|ax| Reverse(*ax));
        let newshape = self.shape().0.extend_singletons(axis);
        self.reshape(HostShape(newshape, plc))
    }
}

impl ExpandDimsOp {
    pub(crate) fn host_int_float_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<usize>,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>> {
        Ok(plc.place(sess, x.expand_dims(axis)))
    }

    pub(crate) fn host_bit_kernel<S: RuntimeSession>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<usize>,
        x: HostBitTensor,
    ) -> Result<HostBitTensor> {
        Ok(plc.place(sess, x.expand_dims(axis)))
    }

    pub(crate) fn host_ring_kernel<S: RuntimeSession, T>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<usize>,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>> {
        Ok(plc.place(sess, x.expand_dims(axis)))
    }
}

// TODO(Morten) inline
impl<T: LinalgScalar> HostTensor<T> {
    fn squeeze(self, axis: Option<usize>) -> Self {
        let plc = (&self.1).clone();
        let newshape = self.shape().0.squeeze(axis);
        self.reshape(HostShape(newshape, plc))
    }
}

impl HostSqueezeOp {
    pub(crate) fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let axis = axis.map(|a| a as usize);
        Ok(plc.place(sess, x.squeeze(axis)))
    }
}

impl ConcatOp {
    pub(crate) fn kernel<S: Session, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        axis: u32,
        xs: &[HostTensor<T>],
    ) -> Result<HostTensor<T>> {
        use ndarray::IxDynImpl;
        use ndarray::ViewRepr;
        let ax = Axis(axis as usize);
        let arr: Vec<ArrayBase<ViewRepr<&T>, Dim<IxDynImpl>>> =
            xs.iter().map(|x| x.0.view()).collect();

        let c = ndarray::concatenate(ax, &arr).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostTensor(c, plc.clone()))
    }

    pub(crate) fn ring_kernel<S: Session, T>(
        _sess: &S,
        plc: &HostPlacement,
        axis: u32,
        xs: &[HostRingTensor<T>],
    ) -> Result<HostRingTensor<T>>
    where
        T: Clone,
    {
        use ndarray::IxDynImpl;
        use ndarray::ViewRepr;
        let arr: Vec<ArrayBase<ViewRepr<&std::num::Wrapping<T>>, Dim<IxDynImpl>>> =
            xs.iter().map(|x| x.0.view()).collect();
        let ax = Axis(axis as usize);
        let concatenated =
            ndarray::concatenate(ax, &arr).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(concatenated, plc.clone()))
    }
}

impl HostTransposeOp {
    pub(crate) fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>> {
        let raw_tensor = x.0.reversed_axes();
        Ok(HostTensor(raw_tensor, plc.clone()))
    }
}

impl HostInverseOp {
    #[cfg(feature = "blas")]
    pub(crate) fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive + Lapack>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let x = plc.place(sess, x);
        // TODO(Morten) better error handling below
        let x_inv = match x.0.ndim() {
            2 => {
                let two_dim: Array2<T> = x.0.into_dimensionality::<Ix2>().unwrap();
                HostTensor::<T>(
                    two_dim
                        .inv()
                        .unwrap()
                        .into_dimensionality::<IxDyn>()
                        .unwrap(),
                    x.1,
                )
            }
            other_rank => panic!(
                "Inverse only defined for rank 2 matrices, not rank {:?}",
                other_rank,
            ),
        };
        Ok(x_inv)
    }

    #[cfg(not(feature = "blas"))]
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _x: HostTensor<T>,
    ) -> Result<HostTensor<T>> {
        Err(Error::UnimplementedOperator(format!(
            "Please enable 'blas' feature"
        )))
    }
}

impl RingFixedpointEncodeOp {
    pub(crate) fn float32_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        scaling_base: u64,
        scaling_exp: u32,
        x: HostFloat32Tensor,
    ) -> Result<HostRing64Tensor> {
        let scaling_factor = u64::pow(scaling_base, scaling_exp);
        let x_upshifted = &x.0 * (scaling_factor as f32);
        let x_converted: ArrayD<Wrapping<u64>> =
            x_upshifted.mapv(|el| Wrapping((el as i64) as u64));
        Ok(HostRingTensor(x_converted, plc.clone()))
    }

    pub(crate) fn float64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        scaling_base: u64,
        scaling_exp: u32,
        x: HostFloat64Tensor,
    ) -> Result<HostRing128Tensor> {
        let scaling_factor = u128::pow(scaling_base as u128, scaling_exp);
        let x_upshifted = &x.0 * (scaling_factor as f64);
        let x_converted: ArrayD<Wrapping<u128>> =
            x_upshifted.mapv(|el| Wrapping((el as i128) as u128));
        Ok(HostRingTensor(x_converted, plc.clone()))
    }
}

impl RingFixedpointDecodeOp {
    pub(crate) fn float32_kernel<S: RuntimeSession>(
        _sess: &S,
        _plc: &HostPlacement,
        _scaling_base: u64,
        _scaling_exp: u32,
        _x: HostRing64Tensor,
    ) -> Result<HostFloat32Tensor> {
        unimplemented!()
    }

    pub(crate) fn float64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        scaling_base: u64,
        scaling_exp: u32,
        x: HostRing128Tensor,
    ) -> Result<HostFloat64Tensor> {
        let scaling_factor = u128::pow(scaling_base as u128, scaling_exp);
        let x_upshifted: ArrayD<i128> = x.0.mapv(|xi| xi.0 as i128);
        let x_converted = x_upshifted.mapv(|el| el as f64);
        Ok(HostTensor(x_converted / scaling_factor as f64, plc.clone()))
    }
}

impl SignOp {
    pub(crate) fn ring64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing64Tensor,
    ) -> Result<HostRing64Tensor> {
        let sign = x.0.mapv(|Wrapping(item)| {
            let s = item as i64;
            if s < 0 {
                Wrapping(-1_i64 as u64)
            } else {
                Wrapping(1_u64)
            }
        });
        Ok(HostRingTensor::<u64>(sign, plc.clone()))
    }

    pub(crate) fn ring128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing128Tensor,
    ) -> Result<HostRing128Tensor> {
        let sign = x.0.mapv(|Wrapping(item)| {
            let s = item as i128;
            if s < 0 {
                Wrapping(-1_i128 as u128)
            } else {
                Wrapping(1_u128)
            }
        });
        Ok(HostRingTensor::<u128>(sign, plc.clone()))
    }
}

impl ShapeOp {
    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
    ) -> Result<HostShape> {
        let raw_shape = RawShape(x.0.shape().into());
        Ok(HostShape(raw_shape, plc.clone()))
    }
}

impl HostReshapeOp {
    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
        shape: HostShape,
    ) -> Result<HostBitTensor> {
        let res =
            x.0.into_shape(shape.0 .0)
                .map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostBitTensor(res, plc.clone()))
    }
}

impl HostReshapeOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        shape: HostShape,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let res =
            x.0.into_shape(shape.0 .0)
                .map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostTensor::<T>(res, plc.clone()))
    }
}

impl FillOp {
    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u8,
        shape: HostShape,
    ) -> Result<HostBitTensor> {
        let raw_shape = shape.0 .0;
        let raw_tensor = ArrayD::from_elem(raw_shape.as_ref(), value);
        Ok(HostBitTensor(raw_tensor, plc.clone()))
    }
}

impl BitSampleOp {
    pub(crate) fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> Result<HostBitTensor> {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let arr =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostBitTensor(arr, plc.clone()))
    }
}

impl BitSampleSeededOp {
    pub(crate) fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> Result<HostBitTensor> {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let res =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostBitTensor(res, plc.clone()))
    }
}

impl BitXorOp {
    pub(crate) fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
        y: HostBitTensor,
    ) -> Result<HostBitTensor> {
        Ok(HostBitTensor(x.0 ^ y.0, plc.clone()))
    }
}

impl BitNegOp {
    pub(crate) fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
    ) -> Result<HostBitTensor> {
        Ok(HostBitTensor((!x.0) & 1, plc.clone()))
    }
}

impl BitAndOp {
    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
        y: HostBitTensor,
    ) -> Result<HostBitTensor> {
        Ok(HostBitTensor(x.0 & y.0, plc.clone()))
    }

    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRingTensor<T>,
        y: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::BitAnd<Wrapping<T>, Output = Wrapping<T>>,
    {
        Ok(HostRingTensor(x.0 & y.0, plc.clone()))
    }
}

impl BitOrOp {
    pub(crate) fn host_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
        y: HostBitTensor,
    ) -> Result<HostBitTensor> {
        Ok(HostBitTensor(x.0 | y.0, plc.clone()))
    }
}

impl BitExtractOp {
    pub(crate) fn kernel64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        bit_idx: usize,
        x: HostRing64Tensor,
    ) -> Result<HostBitTensor> {
        Ok(HostBitTensor(
            (x.0 >> bit_idx).mapv(|ai| (ai.0 & 1) as u8),
            plc.clone(),
        ))
    }

    pub(crate) fn kernel128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        bit_idx: usize,
        x: HostRing128Tensor,
    ) -> Result<HostBitTensor> {
        Ok(HostBitTensor(
            (x.0 >> bit_idx).mapv(|ai| (ai.0 & 1) as u8),
            plc.clone(),
        ))
    }
}

impl RingInjectOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        bit_idx: usize,
        x: HostBitTensor,
    ) -> Result<HostRingTensor<T>>
    where
        T: From<u8>,
        Wrapping<T>: std::ops::Shl<usize, Output = Wrapping<T>>,
    {
        Ok(HostRingTensor(
            x.0.mapv(|ai| Wrapping(T::from(ai)) << bit_idx),
            plc.clone(),
        ))
    }
}

impl RingFillOp {
    pub(crate) fn ring64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u64,
        shape: HostShape,
    ) -> Result<HostRing64Tensor> {
        let raw_shape = shape.0 .0;
        let raw_tensor = ArrayD::from_elem(raw_shape.as_ref(), Wrapping(value));
        Ok(HostRingTensor(raw_tensor, plc.clone()))
    }

    pub(crate) fn ring128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u128,
        shape: HostShape,
    ) -> Result<HostRing128Tensor> {
        let raw_shape = shape.0 .0;
        let raw_tensor = ArrayD::from_elem(raw_shape.as_ref(), Wrapping(value));
        Ok(HostRingTensor(raw_tensor, plc.clone()))
    }
}

impl ShapeOp {
    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRingTensor<T>,
    ) -> Result<HostShape> {
        let raw_shape = RawShape(x.0.shape().into());
        Ok(HostShape(raw_shape, plc.clone()))
    }
}

impl BroadcastOp {
    pub(crate) fn host_ring_kernel<S: RuntimeSession, T: Clone + std::fmt::Debug>(
        _sess: &S,
        plc: &HostPlacement,
        s: HostShape,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>> {
        match x.0.broadcast(s.clone().0 .0) {
            Some(y) => Ok(HostRingTensor(y.to_owned(), plc.clone())),
            None => Err(Error::KernelError(format!(
                "Tensor {:?} not broadcastable to shape {:?}.",
                x, s
            ))),
        }
    }

    pub(crate) fn host_bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        s: HostShape,
        x: HostBitTensor,
    ) -> Result<HostBitTensor> {
        match x.0.broadcast(s.clone().0 .0) {
            Some(y) => Ok(HostBitTensor(y.to_owned(), plc.clone())),
            None => Err(Error::KernelError(format!(
                "Tensor {:?} not broadcastable to shape {:?}.",
                x, s
            ))),
        }
    }
}

impl HostReshapeOp {
    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRingTensor<T>,
        shape: HostShape,
    ) -> Result<HostRingTensor<T>> {
        let res =
            x.0.into_shape(shape.0 .0)
                .map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor::<T>(res, plc.clone()))
    }
}

impl RingAddOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRingTensor<T>,
        y: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Add<Wrapping<T>, Output = Wrapping<T>>,
    {
        Ok(HostRingTensor(x.0 + y.0, plc.clone()))
    }
}

impl RingSubOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRingTensor<T>,
        y: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Sub<Wrapping<T>, Output = Wrapping<T>>,
    {
        Ok(HostRingTensor(x.0 - y.0, plc.clone()))
    }
}

impl RingNegOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Neg<Output = Wrapping<T>>,
    {
        use std::ops::Neg;
        Ok(HostRingTensor(x.0.neg(), plc.clone()))
    }
}

impl RingMulOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRingTensor<T>,
        y: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Mul<Wrapping<T>, Output = Wrapping<T>>,
    {
        Ok(HostRingTensor(x.0 * y.0, plc.clone()))
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

impl RingDotOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRingTensor<T>,
        y: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Mul<Wrapping<T>, Output = Wrapping<T>>,
        Wrapping<T>: LinalgScalar,
    {
        let dot = x.dot(y)?;
        Ok(HostRingTensor(dot.0, plc.clone()))
    }
}

impl RingShlOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        amount: usize,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Shl<usize, Output = Wrapping<T>>,
    {
        Ok(HostRingTensor(x.0 << amount, plc.clone()))
    }
}

impl RingShrOp {
    pub(crate) fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        amount: usize,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Shr<usize, Output = Wrapping<T>>,
    {
        Ok(HostRingTensor(x.0 >> amount, plc.clone()))
    }
}

impl RingSampleOp {
    pub(crate) fn kernel_uniform_u64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> Result<HostRing64Tensor> {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let raw_array =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(raw_array, plc.clone()))
    }

    pub(crate) fn kernel_bits_u64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> Result<HostRing64Tensor> {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u64)).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let arr =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(arr, plc.clone()))
    }

    pub(crate) fn kernel_uniform_u128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> Result<HostRing128Tensor> {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size)
            .map(|_| Wrapping(((rng.next_u64() as u128) << 64) + rng.next_u64() as u128))
            .collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let arr =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(arr, plc.clone()))
    }

    pub(crate) fn kernel_bits_u128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> Result<HostRing128Tensor> {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u128)).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let arr =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(arr, plc.clone()))
    }
}

impl RingSampleSeededOp {
    pub(crate) fn kernel_uniform_u64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> Result<HostRing64Tensor> {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let raw_array =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(raw_array, plc.clone()))
    }

    pub(crate) fn kernel_bits_u64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> Result<HostRing64Tensor> {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u64)).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let arr =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(arr, plc.clone()))
    }

    pub(crate) fn kernel_uniform_u128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> Result<HostRing128Tensor> {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size)
            .map(|_| Wrapping(((rng.next_u64() as u128) << 64) + rng.next_u64() as u128))
            .collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let arr =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(arr, plc.clone()))
    }

    pub(crate) fn kernel_bits_u128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
        seed: Seed,
    ) -> Result<HostRing128Tensor> {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u128)).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let arr =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(arr, plc.clone()))
    }
}

impl LessOp {
    pub(crate) fn host_fixed_kernel<S: Session, HostRingT, HostBitT>(
        sess: &S,
        plc: &HostPlacement,
        x: HostFixedTensor<HostRingT>,
        y: HostFixedTensor<HostRingT>,
    ) -> Result<HostBitT>
    where
        HostPlacement: PlacementLessThan<S, HostRingT, HostRingT, HostBitT>,
    {
        Ok(plc.less(sess, &x.tensor, &y.tensor))
    }

    pub(crate) fn host_ring64_kernel<S: Session>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing64Tensor,
        y: HostRing64Tensor,
    ) -> Result<HostBitTensor> {
        let result = (x.0 - y.0).mapv(|Wrapping(item)| if (item as i64) < 0 { 1_u8 } else { 0_u8 });
        Ok(HostBitTensor(result, plc.clone()))
    }

    pub(crate) fn host_ring128_kernel<S: Session>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing128Tensor,
        y: HostRing128Tensor,
    ) -> Result<HostBitTensor> {
        let result = (x.0 - y.0).mapv(
            |Wrapping(item)| {
                if (item as i128) < 0 {
                    1_u8
                } else {
                    0_u8
                }
            },
        );

        Ok(HostBitTensor(result, plc.clone()))
    }

    pub(crate) fn host_float_kernel<S: Session, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostBitTensor>
    where
        T: std::cmp::PartialOrd + Zero,
    {
        let result = (x.0 - y.0).mapv(|item| (item < T::zero()) as u8);
        Ok(HostBitTensor(result, plc.clone()))
    }
}

impl GreaterThanOp {
    pub(crate) fn host_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: HostRingT,
        y: HostRingT,
    ) -> Result<HostRingT>
    where
        HostPlacement: PlacementSign<S, HostRingT, HostRingT>,
        HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
    {
        let z = plc.sub(sess, &y, &x);
        Ok(plc.sign(sess, &z))
    }
}

impl IdentityOp {
    pub(crate) fn host_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementIdentity<S, HostRingT, HostRingT>,
    {
        let tensor = plc.identity(sess, &x.tensor);
        Ok(HostFixedTensor::<HostRingT> {
            tensor,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}
