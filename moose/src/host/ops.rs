use super::*;
use crate::error::{Error, Result};
use crate::execution::{RuntimeSession, Session};
use crate::host::bitarray::BitArrayRepr;
use crate::prng::AesRng;
use crate::{Const, Ring, N128, N224, N64};
use ndarray::LinalgScalar;
use ndarray::Zip;
#[cfg(feature = "blas")]
use ndarray_linalg::{Inverse, Lapack};
use num_traits::{Float, FromPrimitive, Zero};
use std::convert::TryInto;
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
wrapping_constant_kernel!(prf_key_kernel for HostPrfKey(RawPrfKey));
wrapping_constant_kernel!(seed_kernel for HostSeed(RawSeed));

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
    ) -> Result<HostUnit>
    where
        Value: From<O>,
    {
        // let x: Value = x.into();
        // sess.storage.save(&key.0, &x)?;
        // Ok(HostUnit(plc.clone()))
        todo!()
    }
}

impl AddOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
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

impl SubOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
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

impl MulOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
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

    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
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

impl DivOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        match x.0.broadcast(y.0.dim()) {
            Some(x_broadcasted) => Ok(HostTensor::<T>(
                (x_broadcasted.to_owned() / y.0).into_shared(),
                plc.clone(),
            )),
            None => Ok(HostTensor::<T>((x.0 / y.0).into_shared(), plc.clone())),
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
                    .into_shared()
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                HostTensor(res, self.1)
            }
            (1, 2) => {
                let l = self.0.into_dimensionality::<Ix1>().unwrap();
                let r = other.0.into_dimensionality::<Ix2>().unwrap();
                let res = l
                    .dot(&r)
                    .into_shared()
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                HostTensor(res, self.1)
            }
            (2, 1) => {
                let l = self.0.into_dimensionality::<Ix2>().unwrap();
                let r = other.0.into_dimensionality::<Ix1>().unwrap();
                let res = l
                    .dot(&r)
                    .into_shared()
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                HostTensor(res, self.1)
            }
            (2, 2) => {
                let l = self.0.into_dimensionality::<Ix2>().unwrap();
                let r = other.0.into_dimensionality::<Ix2>().unwrap();
                let res = l
                    .dot(&r)
                    .into_shared()
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                HostTensor(res, self.1)
            }
            (self_rank, other_rank) => panic!(
                // TODO: replace with proper error handling
                "Dot<HostTensor> not implemented between tensors of rank {:?} and {:?}.",
                self_rank, other_rank,
            ),
        }
    }
}

impl DotOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let x = plc.place(sess, x);
        let y = plc.place(sess, y);
        Ok(x.dot(y))
    }
}

impl OnesOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar>(
        _sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> Result<HostTensor<T>> {
        let raw_shape = shape.0;
        Ok(HostTensor(ArcArrayD::ones(raw_shape.0), plc.clone()))
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

impl AtLeast2DOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar>(
        sess: &S,
        plc: &HostPlacement,
        to_column_vector: bool,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let x = plc.place(sess, x);
        match x.0.ndim() {
            0 => Ok(HostTensor(x.0.into_shape(IxDyn(&[1, 1])).unwrap(), x.1)),
            1 => {
                let length = x.0.len();
                let newshape = if to_column_vector {
                    IxDyn(&[length, 1])
                } else {
                    IxDyn(&[1, length])
                };
                Ok(HostTensor(x.0.into_shape(newshape).unwrap(), x.1))
            }
            2 => Ok(x),
            otherwise => Err(Error::InvalidArgument(format!(
                "Tensor input for `at_least_2d` must have rank <= 2, found rank {:?}.",
                otherwise
            ))),
        }
    }
}

impl SliceOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T>(
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
        Ok(HostRingTensor(sliced.to_shared(), plc.clone()))
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

impl DiagOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T>(
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
        let diag = x.0.into_diag();
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
        Ok(HostTensor(result.to_owned().into_shared(), self.1.clone()))
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
        Ok(HostRingTensor(result.to_owned().into_shared(), self.1))
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
        let result = self.0.index_axis(axis, index);
        Ok(HostBitTensor(result, self.1))
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
        let x = plc.place(sess, x);
        x.index_axis(axis, index)
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
        let x = plc.place(sess, x);
        x.index_axis(axis, index)
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
        let x = plc.place(sess, x);
        x.index_axis(axis, index)
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

impl ShlDimOp {
    pub(crate) fn host_bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        amount: usize,
        bit_length: usize,
        x: HostBitTensor,
    ) -> Result<HostBitTensor> {
        use bitvec::prelude::BitVec;
        let height = x.0.dim.default_strides()[0];
        let mut data = BitVec::repeat(false, height * amount); // Left portion is zeroes
        let tail = height * (bit_length - amount);
        data.extend_from_bitslice(&x.0.data[0..tail]); // The rest is just a portion of the input bitarray
        let result = BitArrayRepr {
            data: std::sync::Arc::new(data),
            dim: x.0.dim.clone(),
        };
        Ok(HostBitTensor(result, plc.clone()))
    }
}

impl BitDecomposeOp {
    pub(crate) fn host_ring64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing64Tensor,
    ) -> Result<HostRing64Tensor> {
        let shape = x.shape();
        let raw_shape = shape.0 .0;
        let ones = ArcArrayD::from_elem(raw_shape, Wrapping(1));

        let bit_rep: Vec<_> = (0..<HostRing64Tensor as Ring>::BitLength::VALUE)
            .map(|i| (&x.0 >> i) & (&ones))
            .collect();
        let bit_rep_view: Vec<_> = bit_rep.iter().map(ArrayView::from).collect();

        // by default we put bits as rows, ie access i'th bit from tensor T is done through index_axis(Axis(0), T)
        // in the current protocols it's easier to reason that the bits are stacked on axis(0)
        let result = ndarray::stack(Axis(0), &bit_rep_view)
            .map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(result.into_shared(), plc.clone()))
    }

    pub(crate) fn host_ring128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing128Tensor,
    ) -> Result<HostRing128Tensor> {
        let shape = x.shape();
        let raw_shape = shape.0 .0;
        let ones = ArcArrayD::from_elem(raw_shape, Wrapping(1));

        let bit_rep: Vec<_> = (0..<HostRing128Tensor as Ring>::BitLength::VALUE)
            .map(|i| (&x.0 >> i) & (&ones))
            .collect();

        let bit_rep_view: Vec<_> = bit_rep.iter().map(ArrayView::from).collect();
        let result = ndarray::stack(Axis(0), &bit_rep_view)
            .map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(result.into_shared(), plc.clone()))
    }

    pub(crate) fn host_bit64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing64Tensor,
    ) -> Result<HostBitTensor> {
        use bitvec::prelude::*;
        let mut dim = x.0.dim().insert_axis(Axis(0));
        dim.slice_mut()[0] = <HostRing64Tensor as Ring>::BitLength::VALUE;

        let mut data = BitVec::EMPTY;
        for i in 0..<HostRing64Tensor as Ring>::BitLength::VALUE {
            let slice: BitVec<u8, Lsb0> = x.0.iter().map(|ai| ((ai >> i).0 & 1) != 0).collect();
            data.extend_from_bitslice(&slice);
        }

        Ok(HostBitTensor(
            BitArrayRepr::from_raw(data, dim),
            plc.clone(),
        ))
    }

    pub(crate) fn host_bit128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing128Tensor,
    ) -> Result<HostBitTensor> {
        use bitvec::prelude::*;
        let mut dim = x.0.dim().insert_axis(Axis(0));
        dim.slice_mut()[0] = <HostRing128Tensor as Ring>::BitLength::VALUE;

        let mut data = BitVec::EMPTY;
        for i in 0..<HostRing128Tensor as Ring>::BitLength::VALUE {
            let slice: BitVec<u8, Lsb0> = x.0.iter().map(|ai| ((ai >> i).0 & 1) != 0).collect();
            data.extend_from_bitslice(&slice);
        }

        Ok(HostBitTensor(
            BitArrayRepr::from_raw(data, dim),
            plc.clone(),
        ))
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
                Ok(HostTensor::place(plc, reduced.unwrap().into_shared()))
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
                Ok(HostTensor::place(plc, out.into_shared()))
            }
        }
    }
}

impl SqrtOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: 'static + Float>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let x_sqrt = x.0.mapv(T::sqrt);
        Ok(HostTensor::place(plc, x_sqrt.into_shared()))
    }
}

impl<T: LinalgScalar> HostTensor<T> {
    fn sum(self, axis: Option<usize>) -> Result<Self> {
        if let Some(i) = axis {
            Ok(HostTensor::<T>(
                self.0.sum_axis(Axis(i)).into_shared(),
                self.1,
            ))
        } else {
            let out = Array::from_elem([], self.0.sum())
                .into_dimensionality::<IxDyn>()
                .map_err(|e| Error::KernelError(e.to_string()))?;
            Ok(HostTensor::<T>(out.into_shared(), self.1))
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
        let x = plc.place(sess, x);
        x.sum(axis)
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
        let axis = axis.map(|a| a as usize);
        let x = plc.place(sess, x);
        x.sum(axis)
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
        let plc = self.1.clone();
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
        let x = plc.place(sess, x);
        Ok(x.expand_dims(axis))
    }

    pub(crate) fn host_bit_kernel<S: RuntimeSession>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<usize>,
        x: HostBitTensor,
    ) -> Result<HostBitTensor> {
        let x = plc.place(sess, x);
        Ok(x.expand_dims(axis))
    }

    pub(crate) fn host_ring_kernel<S: RuntimeSession, T>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<usize>,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>> {
        let x = plc.place(sess, x);
        Ok(x.expand_dims(axis))
    }
}

impl SqueezeOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let x = plc.place(sess, x);
        let axis = axis.map(|a| a as usize);
        let newshape = HostShape(x.shape().0.squeeze(axis), plc.clone());
        Ok(x.reshape(newshape))
    }
}

impl ConcatOp {
    pub(crate) fn host_kernel<S: Session, T: LinalgScalar + FromPrimitive>(
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
        Ok(HostTensor(c.into_shared(), plc.clone()))
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
        Ok(HostRingTensor(concatenated.into_shared(), plc.clone()))
    }

    pub(crate) fn bit_kernel<S: Session>(
        _sess: &S,
        plc: &HostPlacement,
        axis: u32,
        xs: &[HostBitTensor],
    ) -> Result<HostBitTensor> {
        use bitvec::prelude::*;
        let mut data = BitVec::<u8, Lsb0>::EMPTY;
        for x in xs {
            data.extend_from_bitslice(&x.0.data);
        }
        // Computing the dimension
        let mut res_dim = xs[0].0.shape().to_vec();
        let stacked_dim: usize = xs.iter().fold(0, |acc, a| acc + a.0.shape()[axis as usize]);
        res_dim[axis as usize] = stacked_dim;

        Ok(HostBitTensor(
            BitArrayRepr::from_raw(data, IxDyn(&res_dim)),
            plc.clone(),
        ))
    }
}

impl TransposeOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>> {
        let raw_tensor = x.0.reversed_axes();
        Ok(HostTensor(raw_tensor, plc.clone()))
    }
}

impl InverseOp {
    #[cfg(feature = "blas")]
    pub(crate) fn host_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive + Lapack>(
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
                let two_dim: ndarray::ArcArray2<T> = x.0.into_dimensionality::<Ix2>().unwrap();
                HostTensor::<T>(
                    two_dim
                        .inv()
                        .unwrap()
                        .into_shared()
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
    pub(crate) fn host_kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _x: HostTensor<T>,
    ) -> Result<HostTensor<T>> {
        Err(Error::UnimplementedOperator(format!(
            "Please enable 'blas' feature"
        )))
    }
}

impl LogOp {
    pub(crate) fn host_kernel<S: RuntimeSession, T: num_traits::Float>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let x = plc.place(sess, x);
        Ok(HostTensor::<T>(x.0.map(|e| e.ln()).into_shared(), x.1))
    }
}

impl Log2Op {
    pub(crate) fn host_kernel<S: RuntimeSession, T: num_traits::Float>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let x = plc.place(sess, x);
        Ok(HostTensor::<T>(x.0.map(|e| e.log2()).into_shared(), x.1))
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
        Ok(HostRingTensor(x_converted.into_shared(), plc.clone()))
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
        Ok(HostRingTensor(x_converted.into_shared(), plc.clone()))
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
        Ok(HostTensor(
            (x_converted / scaling_factor as f64).into_shared(),
            plc.clone(),
        ))
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
        Ok(HostRingTensor::<u64>(sign.into_shared(), plc.clone()))
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
        Ok(HostRingTensor::<u128>(sign.into_shared(), plc.clone()))
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

impl ReshapeOp {
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

    pub(crate) fn host_bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
        shape: HostShape,
    ) -> Result<HostBitTensor> {
        let res = BitArrayRepr {
            data: x.0.data,
            dim: std::sync::Arc::new(IxDyn(&shape.0 .0)),
        };
        Ok(HostBitTensor(res, plc.clone()))
    }

    pub(crate) fn host_ring_kernel<S: RuntimeSession, T>(
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

impl XorOp {
    pub(crate) fn host_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
        y: HostBitTensor,
    ) -> Result<HostBitTensor> {
        let arr = &x.0 ^ &y.0;
        Ok(HostBitTensor(arr, plc.clone()))
    }
}

impl NegOp {
    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
    ) -> Result<HostBitTensor> {
        let arr = !(&x.0);
        Ok(HostBitTensor(arr, plc.clone()))
    }
}

impl AndOp {
    pub(crate) fn host_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
        y: HostBitTensor,
    ) -> Result<HostBitTensor> {
        let arr = &x.0 & &y.0;
        Ok(HostBitTensor(arr, plc.clone()))
    }

    pub(crate) fn host_ring_kernel<S: RuntimeSession, T>(
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

impl OrOp {
    pub(crate) fn host_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
        y: HostBitTensor,
    ) -> Result<HostBitTensor> {
        let arr = &x.0 | &y.0;
        Ok(HostBitTensor(arr, plc.clone()))
    }
}

impl BitExtractOp {
    pub(crate) fn kernel64<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        bit_idx: usize,
        x: HostRing64Tensor,
    ) -> Result<HostBitTensor> {
        let dim = x.0.dim();
        let data = x.0.iter().map(|ai| ((ai >> bit_idx).0 & 1) != 0).collect();
        Ok(HostBitTensor(
            BitArrayRepr::from_raw(data, dim),
            plc.clone(),
        ))
    }

    pub(crate) fn kernel128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        bit_idx: usize,
        x: HostRing128Tensor,
    ) -> Result<HostBitTensor> {
        let dim = x.0.dim();
        let data = x.0.iter().map(|ai| ((ai >> bit_idx).0 & 1) != 0).collect();
        Ok(HostBitTensor(
            BitArrayRepr::from_raw(data, dim),
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
        let values: Vec<_> =
            x.0.data
                .iter()
                .map(|ai| {
                    let bit = if *ai { 1 } else { 0 };
                    Wrapping(T::from(bit)) << bit_idx
                })
                .collect();
        let ix = IxDyn(x.0.shape());
        let arr =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(arr.into_shared(), plc.clone()))
    }
}

impl FillOp {
    pub(crate) fn host_bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u8,
        shape: HostShape,
    ) -> Result<HostBitTensor> {
        let raw_tensor = BitArrayRepr::from_elem(&shape.0, value);
        Ok(HostBitTensor(raw_tensor, plc.clone()))
    }

    pub(crate) fn host_ring64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u64,
        shape: HostShape,
    ) -> Result<HostRing64Tensor> {
        let raw_shape = shape.0 .0;
        let raw_tensor = ArcArrayD::from_elem(raw_shape.as_ref(), Wrapping(value));
        Ok(HostRingTensor(raw_tensor, plc.clone()))
    }

    pub(crate) fn host_ring128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u128,
        shape: HostShape,
    ) -> Result<HostRing128Tensor> {
        let raw_shape = shape.0 .0;
        let raw_tensor = ArcArrayD::from_elem(raw_shape.as_ref(), Wrapping(value));
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
            Some(y) => Ok(HostRingTensor(y.to_owned().into_shared(), plc.clone())),
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
        let dim = IxDyn(&s.0 .0);
        let old_len = x.0.dim.size();
        let new_len = dim.size();
        if new_len < old_len || new_len % old_len != 0 {
            return Err(Error::KernelError(format!(
                "Tensor {:?} not broadcastable to shape {:?}.",
                x, s
            )));
        }
        use bitvec::prelude::*;
        let mut data = BitVec::EMPTY;
        for _ in 0..(new_len / old_len) {
            data.extend_from_bitslice(&x.0.data);
        }
        Ok(HostBitTensor(
            BitArrayRepr::from_raw(data, dim),
            plc.clone(),
        ))
    }
}

impl AddOp {
    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
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

impl SubOp {
    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
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

impl NegOp {
    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
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
                    Ok(HostRingTensor(res.into_shared(), self.1))
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
                    Ok(HostRingTensor(res.into_shared(), self.1))
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
                    Ok(HostRingTensor(res.into_shared(), self.1))
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
                    Ok(HostRingTensor(res.into_shared(), self.1))
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

impl DotOp {
    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
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

impl ShlOp {
    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
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

impl ShrOp {
    pub(crate) fn ring_kernel<S: RuntimeSession, T>(
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

impl SampleOp {
    pub(crate) fn ring64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        max_value: Option<u64>,
        shape: HostShape,
    ) -> Result<HostRing64Tensor> {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let element_sampler: Box<dyn FnMut(_) -> _> = match max_value {
            None => Box::new(|_| Wrapping(rng.next_u64())),
            Some(x) => {
                if x == 1 {
                    Box::new(|_| Wrapping(rng.get_bit() as u64))
                } else {
                    return Err(Error::UnimplementedOperator(
                        "SampleOp for HostRingTensor @ HostPlacement does not yet support max_value != 1".to_string()
                    ));
                }
            }
        };
        let values: Vec<_> = (0..size).map(element_sampler).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let raw_array =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(raw_array.into_shared(), plc.clone()))
    }

    pub(crate) fn ring128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        max_value: Option<u64>,
        shape: HostShape,
    ) -> Result<HostRing128Tensor> {
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let element_sampler: Box<dyn FnMut(_) -> _> = match max_value {
            None => {
                Box::new(|_| Wrapping(((rng.next_u64() as u128) << 64) + rng.next_u64() as u128))
            }
            Some(x) => {
                if x == 1 {
                    Box::new(|_| Wrapping(rng.get_bit() as u128))
                } else {
                    return Err(Error::UnimplementedOperator(
                        "SampleOp for HostRingTensor @ HostPlacement does not yet support max_value != 1".to_string()
                    ));
                }
            }
        };
        let values: Vec<_> = (0..size).map(element_sampler).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let arr =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(arr.into_shared(), plc.clone()))
    }

    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        max_value: Option<u64>,
        shape: HostShape,
    ) -> Result<HostBitTensor> {
        if max_value.is_some() {
            return Err(Error::UnimplementedOperator(
                "SampleOp for HostBitTensor @ HostPlacement does not support max_value".to_string(),
            ));
        };
        let mut rng = AesRng::from_random_seed();
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let arr = BitArrayRepr::from_vec(values, &shape.0);
        Ok(HostBitTensor(arr, plc.clone()))
    }
}

impl SampleSeededOp {
    pub(crate) fn ring64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        max_value: Option<u64>,
        shape: HostShape,
        seed: HostSeed,
    ) -> Result<HostRing64Tensor> {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let element_sampler: Box<dyn FnMut(_) -> _> = match max_value {
            None => Box::new(|_| Wrapping(rng.next_u64())),
            Some(x) => {
                if x == 1 {
                    Box::new(|_| Wrapping(rng.get_bit() as u64))
                } else {
                    return Err(Error::UnimplementedOperator(
                        "SampleOp for HostRingTensor @ HostPlacement does not yet support max_value != 1".to_string()
                    ));
                }
            }
        };
        let values: Vec<_> = (0..size).map(element_sampler).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let raw_array =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(raw_array.into_shared(), plc.clone()))
    }

    pub(crate) fn ring128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        max_value: Option<u64>,
        shape: HostShape,
        seed: HostSeed,
    ) -> Result<HostRing128Tensor> {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let element_sampler: Box<dyn FnMut(_) -> _> = match max_value {
            None => {
                Box::new(|_| Wrapping(((rng.next_u64() as u128) << 64) + rng.next_u64() as u128))
            }
            Some(x) => {
                if x == 1 {
                    Box::new(|_| Wrapping(rng.get_bit() as u128))
                } else {
                    return Err(Error::UnimplementedOperator(
                        "SampleOp for HostRingTensor @ HostPlacement does not yet support max_value != 1".to_string()
                    ));
                }
            }
        };
        let values: Vec<_> = (0..size).map(element_sampler).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let arr =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(arr.into_shared(), plc.clone()))
    }

    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        max_value: Option<u64>,
        shape: HostShape,
        seed: HostSeed,
    ) -> Result<HostBitTensor> {
        if max_value.is_some() {
            return Err(Error::UnimplementedOperator(
                "SampleOp for HostBitTensor @ HostPlacement does not support max_value".to_string(),
            ));
        };
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let res = BitArrayRepr::from_vec(values, &shape.0);
        Ok(HostBitTensor(res, plc.clone()))
    }
}

impl LessThanOp {
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
        use bitvec::prelude::*;
        let dim = x.0.dim();
        let data: BitVec<u8, Lsb0> = (x.0 - y.0)
            .as_slice()
            .ok_or_else(|| Error::KernelError("Failed to get tensor's slice".to_string()))?
            .iter()
            .map(|&Wrapping(item)| (item as i64) < 0)
            .collect();
        let result = BitArrayRepr::from_raw(data, dim);
        Ok(HostBitTensor(result, plc.clone()))
    }

    pub(crate) fn host_ring128_kernel<S: Session>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing128Tensor,
        y: HostRing128Tensor,
    ) -> Result<HostBitTensor> {
        use bitvec::prelude::*;
        let dim = x.0.dim();
        let data: BitVec<u8, Lsb0> = (x.0 - y.0)
            .as_slice()
            .ok_or_else(|| Error::KernelError("Failed to get tensor's slice".to_string()))?
            .iter()
            .map(|&Wrapping(item)| (item as i128) < 0)
            .collect();
        let result = BitArrayRepr::from_raw(data, dim);
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
        use bitvec::prelude::*;
        let dim = x.0.dim();
        let data: BitVec<u8, Lsb0> = (x.0 - y.0)
            .as_slice()
            .ok_or_else(|| Error::KernelError("Failed to get tensor's slice".to_string()))?
            .iter()
            .map(|&item| item < T::zero())
            .collect();
        let result = BitArrayRepr::from_raw(data, dim);
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

impl MuxOp {
    pub(crate) fn host_float_int_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        s: HostBitTensor,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        T: From<u8>,
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        // Seems to be the right approach for now but in the future this
        // expression could be implemented at the HostPlacement level
        // (Add, Sub & Mul) instead of ndarray
        // [s] * ([x] - [y]) + [y] <=> if s=1 choose x, otherwise y
        let s_t: ArrayD<T> =
            s.0.into_array()
                .map_err(|e| Error::KernelError(e.to_string()))?;
        let res = s_t * (x.0 - y.0.clone()) + y.0;
        Ok(HostTensor::<T>(res.into_shared(), plc.clone()))
    }

    pub(crate) fn host_ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        s: HostBitTensor,
        x: HostRingTensor<T>,
        y: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        T: LinalgScalar + FromPrimitive,
        T: From<u8>,
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Add<Output = Wrapping<T>>,
        Wrapping<T>: std::ops::Sub<Output = Wrapping<T>>,
        Wrapping<T>: std::ops::Mul<Output = Wrapping<T>>,
    {
        // Seems to be the right approach for now but in the future this
        // expression could be implemented at the HostPlacement level
        // (Add, Sub & Mul) instead of ndarray
        // [s] * ([x] - [y]) + [y] <=> if s=1 choose x, otherwise y
        let s_t: ArrayD<Wrapping<T>> =
            s.0.into_array()
                .map_err(|e| Error::KernelError(e.to_string()))?
                .mapv(|item| Wrapping(item));
        let res = s_t * (x.0 - y.0.clone()) + y.0;
        Ok(HostRingTensor::<T>(res.into_shared(), plc.clone()))
    }
}

impl CastOp {
    pub(crate) fn no_op_reduction_kernel<S: RuntimeSession, T>(
        sess: &S,
        plc: &HostPlacement,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostRingTensor<T>>,
    {
        let x = plc.place(sess, x);
        Ok(x)
    }

    pub(crate) fn hr64_hu64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing64Tensor,
    ) -> Result<HostTensor<u64>> {
        let unwrapped = x.0.mapv(|item| item.0);
        Ok(HostTensor(unwrapped.into_shared(), plc.clone()))
    }

    pub(crate) fn ring_reduction_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostRing128Tensor,
    ) -> Result<HostRing64Tensor> {
        let x_downshifted: ArrayD<Wrapping<u64>> = x.0.mapv(|el| {
            let reduced = el.0 % ((1_u128) << 64);
            Wrapping(reduced as u64)
        });

        Ok(HostRingTensor(x_downshifted.into_shared(), plc.clone()))
    }

    // standard casts
    // pub(crate) fn to_f64_kernel<S, T>(
    //     _sess: &S,
    //     plc: &HostPlacement,
    //     x: HostTensor<T>,
    // ) -> Result<HostTensor<f64>>
    // where
    //     T: num_traits::ToPrimitive
    // {
    //     Ok(HostTensor::<f64>(x.0.map(|x| numcast(x)), x.1))
    // }

    pub(crate) fn standard_host_kernel<S: RuntimeSession, T1, T2>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T1>,
    ) -> Result<HostTensor<T2>>
    where
        T1: num_traits::NumCast + Clone,
        T2: num_traits::NumCast + Clone,
    {
        Ok(HostTensor::<T2>(
            x.0.mapv(|x| num_traits::cast(x).unwrap()).into(),
            plc.clone(),
        ))
    }
}

impl RingFixedpointArgmaxOp {
    pub(crate) fn host_ring64_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        axis: usize,
        _upmost_index: usize,
        x: HostRing64Tensor,
    ) -> Result<HostRing64Tensor> {
        let axis = Axis(axis);
        let signed_tensor = x.0.mapv(|entry| entry.0 as i64);

        let mut current_max = signed_tensor.index_axis(axis, 0).to_owned();
        let mut current_pattern_max = current_max.mapv(|_x| 0_u64);

        for (index, subview) in signed_tensor.axis_iter(axis).enumerate() {
            let index = index as u64;
            Zip::from(&mut current_max)
                .and(&mut current_pattern_max)
                .and(&subview)
                .for_each(|max_entry, pattern_entry, &subview_entry| {
                    if *max_entry < subview_entry {
                        *max_entry = subview_entry;
                        *pattern_entry = index;
                    }
                });
        }
        Ok(HostRingTensor(
            current_pattern_max.mapv(Wrapping).into_shared(),
            plc.clone(),
        ))
    }

    pub(crate) fn host_ring128_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        axis: usize,
        _upmost_index: usize,
        x: HostRing128Tensor,
    ) -> Result<HostRing64Tensor> {
        let axis = Axis(axis);
        let signed_tensor = x.0.mapv(|entry| entry.0 as i128);

        let mut current_max = signed_tensor.index_axis(axis, 0).to_owned();
        let mut current_pattern_max = current_max.mapv(|_x| 0_u64);

        for (index, subview) in signed_tensor.axis_iter(axis).enumerate() {
            let index = index as u64;
            Zip::from(&mut current_max)
                .and(&mut current_pattern_max)
                .and(&subview)
                .for_each(|max_entry, pattern_entry, &subview_entry| {
                    if *max_entry < subview_entry {
                        *max_entry = subview_entry;
                        *pattern_entry = index;
                    }
                });
        }
        Ok(HostRingTensor(
            current_pattern_max.mapv(Wrapping).into_shared(),
            plc.clone(),
        ))
    }
}

impl ArgmaxOp {
    pub(crate) fn host_fixed_uint_kernel<S: Session, HostRingT, HostRingT2>(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        upmost_index: usize,
        x: HostFixedTensor<HostRingT>,
    ) -> Result<m!(HostUint64Tensor)>
    where
        HostUint64Tensor: KnownType<S>,
        HostPlacement: PlacementArgmax<S, HostRingT, HostRingT2>,
        HostPlacement: PlacementCast<S, HostRingT2, m!(HostUint64Tensor)>,
    {
        let arg_out = plc.argmax(sess, axis, upmost_index, &x.tensor);
        Ok(plc.cast(sess, &arg_out))
    }
}
