use super::*;
use crate::error::{Error, Result};
use crate::prng::AesRng;
use crate::{Const, Ring, N128, N224, N64};
use ndarray::LinalgScalar;
#[cfg(feature = "blas")]
use ndarray_linalg::Lapack;
use num_traits::{Float, FromPrimitive, Zero};
use std::marker::PhantomData;
use std::num::Wrapping;

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

modelled_kernel! {
    PlacementMeanAsFixedpoint::mean_as_fixedpoint, RingFixedpointMeanOp{axis: Option<u32>, scaling_base: u64, scaling_exp: u32},
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring128_kernel),
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
    ) -> Result<HostRing64Tensor>
    where
        HostPlacement: PlacementPlace<S, HostRing64Tensor>,
    {
        let scaling_factor = u64::pow(scaling_base, scaling_exp);
        let axis = axis.map(|a| a as usize);
        let mean = HostRing64Tensor::fixedpoint_mean(x, axis, scaling_factor)?;
        Ok(plc.place(sess, mean))
    }

    fn ring128_kernel<S: RuntimeSession>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: HostRing128Tensor,
    ) -> Result<HostRing128Tensor>
    where
        HostPlacement: PlacementPlace<S, HostRing128Tensor>,
    {
        let scaling_factor = u128::pow(scaling_base as u128, scaling_exp);
        let axis = axis.map(|a| a as usize);
        let mean = HostRing128Tensor::fixedpoint_mean(x, axis, scaling_factor)?;
        Ok(plc.place(sess, mean))
    }
}

modelled_kernel! {
    PlacementAdd::add, HostAddOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor, HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor, HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor, HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

impl HostAddOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(plc.place(sess, x + y))
    }
}

modelled_kernel! {
    PlacementSub::sub, HostSubOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor, HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor, HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor, HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

impl HostSubOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(plc.place(sess, x - y))
    }
}

modelled_kernel! {
    PlacementMul::mul, HostMulOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor, HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor, HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor, HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

impl HostMulOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(plc.place(sess, x * y))
    }
}

modelled_kernel! {
    PlacementDiv::div, HostDivOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring_kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring_kernel),
        (HostPlacement, (HostInt8Tensor, HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor, HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor, HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

impl HostDivOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
        y: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(plc.place(sess, x / y))
    }

    fn ring_kernel<S: RuntimeSession, T>(
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

modelled_kernel! {
    PlacementDot::dot, HostDotOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
    ]
}

impl HostDotOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
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
    pub fn kernel<S: RuntimeSession, T: LinalgScalar>(
        sess: &S,
        plc: &HostPlacement,
        shape: HostShape,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(plc.place(sess, HostTensor::ones(shape)))
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

modelled_kernel! {
    PlacementAtLeast2D::at_least_2d, HostAtLeast2DOp{to_column_vector: bool},
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

impl HostAtLeast2DOp {
    fn kernel<S: RuntimeSession, T: LinalgScalar>(
        sess: &S,
        plc: &HostPlacement,
        to_column_vector: bool,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let y = x.atleast_2d(to_column_vector);
        Ok(plc.place(sess, y))
    }
}

unmodelled!(HostPlacement, attributes[slice: SliceInfo] (HostShape) -> HostShape, SliceOp);

kernel! {
    SliceOp,
    [
        (HostPlacement, (HostShape) -> HostShape => [hybrid] attributes[slice] Self::kernel),
        // (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [hybrid] attributes[slice] Self::kernel),
        // (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [hybrid] attributes[slice] Self::kernel),
    ]
}

impl SliceOp {
    // TODO(lvorona): type inferring fails if I try to make it more generic and have one kernel work for all the types
    pub fn kernel<S: Session>(
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

modelled_kernel! {
    PlacementSlice::slice, HostSliceOp{slice: SliceInfo},
    [
        (HostPlacement, (HostShape) -> HostShape => [runtime] Self::shape_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

impl HostSliceOp {
    pub fn kernel<S: RuntimeSession, T>(
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

    pub fn shape_kernel<S: RuntimeSession>(
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

modelled_kernel! {
    PlacementDiag::diag, HostDiagOp,
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
    ) -> Result<HostTensor<T>> {
        let diag =
            x.0.into_diag()
                .into_dimensionality::<IxDyn>()
                .map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostTensor::<T>(diag, plc.clone()))
    }

    pub fn ring_kernel<S: RuntimeSession, T>(
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

    pub fn bit_kernel<S: RuntimeSession>(
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

impl IndexAxisOp {
    pub(crate) fn host_float_kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        _sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        T: Clone,
    {
        let axis = Axis(axis);
        let result = x.0.index_axis(axis, index);
        Ok(HostTensor(result.to_owned(), plc.clone()))
    }

    pub(crate) fn host_bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: HostBitTensor,
    ) -> Result<HostBitTensor> {
        let axis = Axis(axis);
        let result = x.0.index_axis(axis, index);
        Ok(HostBitTensor(result.to_owned(), plc.clone()))
    }

    pub(crate) fn host_ring_kernel<S: RuntimeSession, T>(
        _sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        T: Clone,
    {
        let axis = Axis(axis);
        let result = x.0.index_axis(axis, index);
        Ok(HostRingTensor(result.to_owned(), plc.clone()))
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

modelled_kernel! {
    PlacementShlDim::shl_dim, HostShlDimOp{amount: usize, bit_length: usize},
    [
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::bit_kernel),
    ]
}

impl HostShlDimOp {
    pub fn bit_kernel<S: RuntimeSession>(
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

modelled_kernel! {
    PlacementBitDec::bit_decompose, HostBitDecOp,
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

    fn ring128_kernel<S: RuntimeSession>(
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

    fn bit64_kernel<S: RuntimeSession>(
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

    fn bit128_kernel<S: RuntimeSession>(
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

modelled_kernel! {
    PlacementMean::mean, HostMeanOp{axis: Option<u32>},
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
    ]
}

impl HostMeanOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
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

modelled_kernel! {
    PlacementSqrt::sqrt, HostSqrtOp,
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
    ]
}

impl HostSqrtOp {
    pub fn kernel<S: RuntimeSession, T: 'static + Float>(
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

modelled_kernel! {
    PlacementSum::sum, HostSumOp{axis: Option<u32>}, [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
    ]
}

impl HostSumOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let axis = axis.map(|a| a as usize);
        Ok(plc.place(sess, x.sum(axis)?))
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

modelled_kernel! {
    PlacementExpandDims::expand_dims, HostExpandDimsOp{axis: Vec<u32>}, [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

impl HostExpandDimsOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<u32>,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        let axis = axis.iter().map(|a| *a as usize).collect();
        Ok(plc.place(sess, x.expand_dims(axis)))
    }
}

modelled_kernel! {
    PlacementSqueeze::squeeze, HostSqueezeOp{axis: Option<u32>}, [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> HostUint64Tensor => [runtime] Self::kernel),
    ]
}

impl HostSqueezeOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
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

modelled_kernel! {
    PlacementTranspose::transpose, HostTransposeOp,
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

impl HostTransposeOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(plc.place(sess, x.transpose()))
    }
}

modelled_kernel! {
    PlacementInverse::inverse, HostInverseOp,
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
    ]
}

#[cfg(feature = "blas")]
impl HostInverseOp {
    pub fn kernel<S: RuntimeSession, T: LinalgScalar + FromPrimitive + Lapack>(
        sess: &S,
        plc: &HostPlacement,
        x: HostTensor<T>,
    ) -> Result<HostTensor<T>>
    where
        HostPlacement: PlacementPlace<S, HostTensor<T>>,
    {
        Ok(plc.place(sess, x.inv()))
    }
}

#[cfg(not(feature = "blas"))]
impl HostInverseOp {
    pub fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _x: HostTensor<T>,
    ) -> Result<HostTensor<T>> {
        unimplemented!("Please enable 'blas' feature");
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

modelled_kernel! {
    PlacementSign::sign, SignOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring128_kernel),
    ]
}

impl SignOp {
    fn ring64_kernel<S: RuntimeSession>(
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

    fn ring128_kernel<S: RuntimeSession>(
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

modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> HostBitTensor, FillOp);

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
    ) -> Result<HostBitTensor> {
        Ok(HostBitTensor(x.0 ^ y.0, plc.clone()))
    }
}

modelled!(PlacementNeg::neg, HostPlacement, (HostBitTensor) -> HostBitTensor, BitNegOp);

kernel! {
    BitNegOp,
    [
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::kernel),
    ]
}

impl BitNegOp {
    fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: HostBitTensor,
    ) -> Result<HostBitTensor> {
        Ok(HostBitTensor((!x.0) & 1, plc.clone()))
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
    ) -> Result<HostBitTensor> {
        Ok(HostBitTensor(x.0 & y.0, plc.clone()))
    }

    fn ring_kernel<S: RuntimeSession, T>(
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

modelled!(PlacementOr::or, HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor, BitOrOp);

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
    ) -> Result<HostBitTensor> {
        Ok(HostBitTensor(
            (x >> bit_idx).0.mapv(|ai| (ai.0 & 1) as u8),
            plc.clone(),
        ))
    }
    fn kernel128<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        bit_idx: usize,
        x: HostRing128Tensor,
    ) -> Result<HostBitTensor> {
        Ok(HostBitTensor(
            (x >> bit_idx).0.mapv(|ai| (ai.0 & 1) as u8),
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
    ) -> Result<HostRing64Tensor> {
        let raw_shape = shape.0 .0;
        let raw_tensor = ArrayD::from_elem(raw_shape.as_ref(), Wrapping(value));
        Ok(HostRingTensor(raw_tensor, plc.clone()))
    }

    fn ring128_kernel<S: RuntimeSession>(
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
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        T: FromPrimitive + Zero,
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Add<Output = Wrapping<T>>,
        HostPlacement: PlacementPlace<S, HostRingTensor<T>>,
    {
        let sum = x.sum(axis.map(|a| a as usize))?;
        Ok(plc.place(sess, sum))
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
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Shl<usize, Output = Wrapping<T>>,
    {
        Ok(HostRingTensor(x.0 << amount, plc.clone()))
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
        x: HostRingTensor<T>,
    ) -> Result<HostRingTensor<T>>
    where
        Wrapping<T>: Clone,
        Wrapping<T>: std::ops::Shr<usize, Output = Wrapping<T>>,
    {
        Ok(HostRingTensor(x.0 >> amount, plc.clone()))
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
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleOp with max_value != 1".to_string()
                )),
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
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleOp with max_value != 1".to_string()
                )),
            }
        }),
    ]
}

impl RingSampleOp {
    fn kernel_uniform_u64<S: RuntimeSession>(
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

    fn kernel_bits_u64<S: RuntimeSession>(
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

    fn kernel_uniform_u128<S: RuntimeSession>(
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

    fn kernel_bits_u128<S: RuntimeSession>(
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
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleSeededOp with max_value != 1".to_string()
                )),
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
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleSeededOp with max_value != 1".to_string()
                )),
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
    ) -> Result<HostRing64Tensor> {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        let raw_array =
            Array::from_shape_vec(ix, values).map_err(|e| Error::KernelError(e.to_string()))?;
        Ok(HostRingTensor(raw_array, plc.clone()))
    }

    fn kernel_bits_u64<S: RuntimeSession>(
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

    fn kernel_uniform_u128<S: RuntimeSession>(
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

    fn kernel_bits_u128<S: RuntimeSession>(
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

modelled!(PlacementLessThan::less, HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostBitTensor, LessOp);
modelled!(PlacementLessThan::less, HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostBitTensor, LessOp);
modelled!(PlacementLessThan::less, HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostBitTensor, LessOp);
modelled!(PlacementLessThan::less, HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostBitTensor, LessOp);
modelled!(PlacementLessThan::less, HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostBitTensor, LessOp);
modelled!(PlacementLessThan::less, HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostBitTensor, LessOp);

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

modelled!(PlacementGreaterThan::greater_than, HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor, GreaterThanOp);
modelled!(PlacementGreaterThan::greater_than, HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor, GreaterThanOp);

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
