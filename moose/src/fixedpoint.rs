//! Support for fixed-point arithmetic

use crate::computation::{
    FixedpointAddOp, FixedpointDecodeOp, FixedpointDotOp, FixedpointEncodeOp, FixedpointMeanOp,
    FixedpointMulOp, FixedpointRingMeanOp, FixedpointSubOp, FixedpointSumOp, FixedpointTruncPrOp,
    HostPlacement, KnownType, Placed, Placement, ReplicatedPlacement,
};
use crate::error::Result;
use crate::host::{HostRing64Tensor, HostRing128Tensor, AbstractHostFixedTensor,
    AbstractHostRingTensor, HostFloat32Tensor, HostFloat64Tensor, HostFixed128Tensor,
    HostFixed64Tensor,
};
use crate::kernels::{
    PlacementAdd, PlacementDot, PlacementDotSetup, PlacementFixedpointDecode,
    PlacementFixedpointEncode, PlacementFixedpointRingDecode, PlacementFixedpointRingEncode,
    PlacementMean, PlacementMul, PlacementMulSetup, PlacementPlace, PlacementReveal,
    PlacementRingMean, PlacementSetupGen, PlacementShareSetup, PlacementShr, PlacementSub,
    PlacementSum, PlacementTruncPr, RuntimeSession, Session,
};
use crate::replicated::{ReplicatedFixed64Tensor, ReplicatedFixed128Tensor, AbstractReplicatedFixedTensor};
use macros::with_context;
use ndarray::prelude::*;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::num::Wrapping;
use std::ops::Mul;

/// Fixed-point tensor backed by Z_{2^64} arithmetic
pub type Fixed64Tensor = FixedTensor<HostFixed64Tensor, ReplicatedFixed64Tensor>;

/// Fixed-point tensor backed by Z_{2^128} arithmetic
pub type Fixed128Tensor = FixedTensor<HostFixed128Tensor, ReplicatedFixed128Tensor>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FixedTensor<HostTensorT, ReplicatedTensorT> {
    Host(HostTensorT),
    Replicated(ReplicatedTensorT),
}

impl<HostTensorT, ReplicatedTensorT> Placed for FixedTensor<HostTensorT, ReplicatedTensorT>
where
    HostTensorT: Placed,
    HostTensorT::Placement: Into<Placement>,
    ReplicatedTensorT: Placed,
    ReplicatedTensorT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            FixedTensor::Host(x) => Ok(x.placement()?.into()),
            FixedTensor::Replicated(x) => Ok(x.placement()?.into()),
        }
    }
}

pub trait Convert<T> {
    type Scale: One + Clone;
    fn encode(x: &T, scaling_factor: Self::Scale) -> Self;
    fn decode(x: &Self, scaling_factor: Self::Scale) -> T;
}

impl Convert<HostFloat64Tensor> for HostFixed64Tensor {
    type Scale = u64;
    fn encode(x: &HostFloat64Tensor, scaling_factor: Self::Scale) -> HostFixed64Tensor {
        let x_upshifted = &x.0 * (scaling_factor as f64);
        let x_converted: ArrayD<u64> = x_upshifted.mapv(|el| (el as i64) as u64);
        AbstractHostFixedTensor(HostRing64Tensor::from(x_converted))
    }
    fn decode(x: &Self, scaling_factor: Self::Scale) -> HostFloat64Tensor {
        let x_upshifted: ArrayD<i64> = x.0.into();
        let x_converted = x_upshifted.mapv(|el| el as f64);
        HostFloat64Tensor::from(x_converted / scaling_factor as f64)
    }
}

impl Convert<HostFloat64Tensor> for HostFixed128Tensor {
    type Scale = u128;
    fn encode(x: &HostFloat64Tensor, scaling_factor: Self::Scale) -> HostFixed128Tensor {
        let x_upshifted = &x.0 * (scaling_factor as f64);
        let x_converted: ArrayD<u128> = x_upshifted.mapv(|el| (el as i128) as u128);
        AbstractHostFixedTensor(HostRing128Tensor::from(x_converted))
    }

    fn decode(x: &Self, scaling_factor: Self::Scale) -> HostFloat64Tensor {
        let x_upshifted: ArrayD<i128> = x.into();
        let x_converted = x_upshifted.mapv(|el| el as f64);
        HostFloat64Tensor::from(x_converted / scaling_factor as f64)
    }
}

impl<T> AbstractHostRingTensor<T>
where
    Wrapping<T>: Clone + Zero + Mul<Wrapping<T>, Output = Wrapping<T>>,
    AbstractHostRingTensor<T>: Convert<HostFloat64Tensor>,
{
    pub fn ring_mean(
        x: Self,
        axis: Option<usize>,
        scaling_factor: <AbstractHostRingTensor<T> as Convert<HostFloat64Tensor>>::Scale,
    ) -> AbstractHostRingTensor<T> {
        let mean_weight = Self::compute_mean_weight(&x, &axis);
        let encoded_weight = AbstractHostRingTensor::<T>::encode(&mean_weight, scaling_factor);
        let operand_sum = x.sum(axis);
        operand_sum.mul(encoded_weight)
    }

    fn compute_mean_weight(x: &Self, &axis: &Option<usize>) -> HostFloat64Tensor {
        let shape: &[usize] = x.0.shape();
        if let Some(ax) = axis {
            let dim_len = shape[ax] as f64;
            HostFloat64Tensor::from(
                Array::from_elem([], 1.0 / dim_len)
                    .into_dimensionality::<IxDyn>()
                    .unwrap(),
            )
        } else {
            let dim_prod: usize = std::iter::Product::product(shape.iter());
            let prod_inv = 1.0 / dim_prod as f64;
            HostFloat64Tensor::from(
                Array::from_elem([], prod_inv)
                    .into_dimensionality::<IxDyn>()
                    .unwrap(),
            )
        }
    }
}

modelled!(PlacementRingMean::ring_mean, HostPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (HostRing64Tensor) -> HostRing64Tensor, FixedpointRingMeanOp);
modelled!(PlacementRingMean::ring_mean, HostPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (HostRing128Tensor) -> HostRing128Tensor, FixedpointRingMeanOp);

kernel! {
    FixedpointRingMeanOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => attributes[axis, scaling_base, scaling_exp] Self::kernel_ring64tensor),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => attributes[axis, scaling_base, scaling_exp] Self::kernel_ring128tensor),
    ]
}

impl FixedpointRingMeanOp {
    fn kernel_ring64tensor<S: RuntimeSession>(
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
        let mean = HostRing64Tensor::ring_mean(x, axis, scaling_factor);
        plc.place(sess, mean)
    }

    fn kernel_ring128tensor<S: RuntimeSession>(
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
        let mean = HostRing128Tensor::ring_mean(x, axis, scaling_factor);
        plc.place(sess, mean)
    }
}

modelled!(PlacementFixedpointEncode::fixedpoint_encode, HostPlacement, attributes[precision: u32] (HostFloat32Tensor) -> HostFixed64Tensor, FixedpointEncodeOp);
modelled!(PlacementFixedpointEncode::fixedpoint_encode, HostPlacement, attributes[precision: u32] (HostFloat64Tensor) -> HostFixed128Tensor, FixedpointEncodeOp);
modelled!(PlacementFixedpointEncode::fixedpoint_encode, HostPlacement, attributes[precision: u32] (HostFloat32Tensor) -> Fixed64Tensor, FixedpointEncodeOp);
modelled!(PlacementFixedpointEncode::fixedpoint_encode, HostPlacement, attributes[precision: u32] (HostFloat64Tensor) -> Fixed128Tensor, FixedpointEncodeOp);

hybrid_kernel! {
    FixedpointEncodeOp,
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFixed64Tensor => attributes[precision] Self::host_fixed_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFixed128Tensor => attributes[precision] Self::host_fixed_kernel),
        (HostPlacement, (HostFloat32Tensor) -> Fixed64Tensor => attributes[precision] Self::fixed_kernel),
        (HostPlacement, (HostFloat64Tensor) -> Fixed128Tensor => attributes[precision] Self::fixed_kernel),
    ]
}

impl FixedpointEncodeOp {
    fn fixed_kernel<S: Session, HostFloatT, HostFixedT, RepRingT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: HostFloatT,
    ) -> FixedTensor<HostFixedT, RepRingT>
    where
        HostPlacement: PlacementFixedpointEncode<S, HostFloatT, HostFixedT>,
    {
        let x = plc.fixedpoint_encode(sess, precision, &x);
        FixedTensor::Host(x)
    }

    fn host_fixed_kernel<S: Session, HostFloatT, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: HostFloatT,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementFixedpointRingEncode<S, HostFloatT, HostRingT>,
    {
        let y = plc.fixedpoint_ring_encode(sess, 2, precision, &x);
        AbstractHostFixedTensor(y)
    }
}

modelled!(PlacementFixedpointDecode::fixedpoint_decode, HostPlacement, attributes[precision: u32] (Fixed64Tensor) -> HostFloat32Tensor, FixedpointDecodeOp);
modelled!(PlacementFixedpointDecode::fixedpoint_decode, HostPlacement, attributes[precision: u32] (Fixed128Tensor) -> HostFloat64Tensor, FixedpointDecodeOp);
modelled!(PlacementFixedpointDecode::fixedpoint_decode, HostPlacement, attributes[precision: u32] (HostFixed64Tensor) -> HostFloat32Tensor, FixedpointDecodeOp);
modelled!(PlacementFixedpointDecode::fixedpoint_decode, HostPlacement, attributes[precision: u32] (HostFixed128Tensor) -> HostFloat64Tensor, FixedpointDecodeOp);

// TODO(Morten) should these produce Float32Tensor and Float64Tensor instead?
hybrid_kernel! {
    FixedpointDecodeOp,
    [
        (HostPlacement, (Fixed64Tensor) -> HostFloat32Tensor => attributes[precision] Self::fixed_kernel),
        (HostPlacement, (Fixed128Tensor) -> HostFloat64Tensor => attributes[precision] Self::fixed_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFloat32Tensor => attributes[precision] Self::host_fixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFloat64Tensor => attributes[precision] Self::host_fixed_kernel),
    ]
}

impl FixedpointDecodeOp {
    fn fixed_kernel<S: Session, HostFixedT, RepFixedT, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> HostFloatT
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementFixedpointDecode<S, HostFixedT, HostFloatT>,
    {
        let v = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        plc.fixedpoint_decode(sess, precision, &v)
    }

    fn host_fixed_kernel<S: Session, HostRingT, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: AbstractHostFixedTensor<HostRingT>,
    ) -> HostFloatT
    where
        HostPlacement: PlacementFixedpointRingDecode<S, HostRingT, HostFloatT>,
    {
        plc.fixedpoint_ring_decode(sess, 2, precision, &x.0)
    }
}

modelled!(PlacementAdd::add, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointAddOp);
modelled!(PlacementAdd::add, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointAddOp);
modelled!(PlacementAdd::add, HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor, FixedpointAddOp);
modelled!(PlacementAdd::add, HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor, FixedpointAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, FixedpointAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, FixedpointAddOp);

hybrid_kernel! {
    FixedpointAddOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::fixed_on_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::fixed_on_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::fixed_on_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::fixed_on_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => Self::hostfixed_on_host_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => Self::hostfixed_on_host_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => Self::repfixed_on_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => Self::repfixed_on_rep_kernel),
    ]
}

impl FixedpointAddOp {
    fn fixed_on_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementAdd<S, HostFixedT, HostFixedT, HostFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        let result = with_context!(plc, sess, x + y);
        FixedTensor::Host(result)
    }

    fn fixed_on_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementAdd<S, RepFixedT, RepFixedT, RepFixedT>,
    {
        let setup = plc.gen_setup(sess);

        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &setup, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &setup, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.add(sess, &x, &y);
        FixedTensor::Replicated(z)
    }

    fn hostfixed_on_host_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
    {
        AbstractHostFixedTensor(plc.add(sess, &x.0, &y.0))
    }

    fn repfixed_on_rep_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> AbstractReplicatedFixedTensor<RepRingT>
    where
        HostPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
    {
        AbstractReplicatedFixedTensor(plc.add(sess, &x.0, &y.0))
    }
}

modelled!(PlacementSub::sub, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor, FixedpointSubOp);

hybrid_kernel! {
    FixedpointSubOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::fixed_on_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::fixed_on_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::fixed_on_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::fixed_on_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => Self::hostfixed_on_host_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => Self::hostfixed_on_host_kernel),
    ]
}

impl FixedpointSubOp {
    fn fixed_on_host_kernel<S: Session, HostFixedT, RepRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepRingT>,
        y: FixedTensor<HostFixedT, RepRingT>,
    ) -> FixedTensor<HostFixedT, RepRingT>
    where
        HostPlacement: PlacementReveal<S, RepRingT, HostFixedT>,
        HostPlacement: PlacementSub<S, HostFixedT, HostFixedT, HostFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        let result = with_context!(plc, sess, x - y);
        FixedTensor::Host(result)
    }

    fn hostfixed_on_host_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
    {
        AbstractHostFixedTensor(plc.sub(sess, &x.0, &y.0))
    }

    fn fixed_on_rep_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<RingT, RepT>,
        y: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, RingT, RepT>,
        ReplicatedPlacement: PlacementSub<S, RepT, RepT, RepT>,
    {
        let setup = plc.gen_setup(sess);

        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &setup, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &setup, &v),
            FixedTensor::Replicated(v) => v,
        };

        let result = with_context!(plc, sess, x - y);
        FixedTensor::Replicated(result)
    }
}

modelled!(PlacementMul::mul, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor, FixedpointMulOp);

hybrid_kernel! {
    FixedpointMulOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::fixed_on_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::fixed_on_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::fixed_on_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::fixed_on_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => Self::hostfixed_on_host_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => Self::hostfixed_on_host_kernel),
    ]
}

impl FixedpointMulOp {
    fn fixed_on_host_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<RingT, RepT>,
        y: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        HostPlacement: PlacementReveal<S, RepT, RingT>,
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
    {
        let x = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        let result = with_context!(plc, sess, x * y);
        FixedTensor::Host(result)
    }

    fn hostfixed_on_host_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementMul<S, HostRingT, HostRingT, HostRingT>,
    {
        AbstractHostFixedTensor(plc.mul(sess, &x.0, &y.0))
    }

    fn fixed_on_rep_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<RingT, RepT>,
        y: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, RingT, RepT>,
        ReplicatedPlacement: PlacementMulSetup<S, S::ReplicatedSetup, RepT, RepT, RepT>,
    {
        let setup = plc.gen_setup(sess);

        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &setup, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &setup, &v),
            FixedTensor::Replicated(v) => v,
        };

        let result = with_context!(plc, sess, mul_setup(&setup, &x, &y));
        FixedTensor::Replicated(result)
    }
}

modelled!(PlacementDot::dot, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor, FixedpointDotOp);

hybrid_kernel! {
    FixedpointDotOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::fixed_on_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::fixed_on_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::rep_on_host_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::rep_on_host_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => Self::hostfixed_on_host_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => Self::hostfixed_on_host_kernel),
    ]
}

impl FixedpointDotOp {
    fn fixed_on_host_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<RingT, RepT>,
        y: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        HostPlacement: PlacementReveal<S, RepT, RingT>,
        HostPlacement: PlacementDot<S, RingT, RingT, RingT>,
    {
        let x_revealed = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };
        let y_revealed = match y {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let result = plc.dot(sess, &x_revealed, &y_revealed);
        FixedTensor::Host(result)
    }

    fn hostfixed_on_host_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementDot<S, HostRingT, HostRingT, HostRingT>,
    {
        AbstractHostFixedTensor(plc.dot(sess, &x.0, &y.0))
    }

    fn rep_on_host_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<RingT, RepT>,
        y: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, RingT, RepT>,
        ReplicatedPlacement: PlacementDotSetup<S, S::ReplicatedSetup, RepT, RepT, RepT>,
    {
        let setup = plc.gen_setup(sess);

        let x_shared = match x {
            FixedTensor::Host(x) => plc.share(sess, &setup, &x),
            FixedTensor::Replicated(x) => x,
        };
        let y_shared = match y {
            FixedTensor::Host(x) => plc.share(sess, &setup, &x),
            FixedTensor::Replicated(x) => x,
        };

        let result = plc.dot_setup(sess, &setup, &x_shared, &y_shared);
        FixedTensor::Replicated(result)
    }
}

modelled!(PlacementTruncPr::trunc_pr, HostPlacement, attributes[precision: u32] (Fixed64Tensor) -> Fixed64Tensor, FixedpointTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, HostPlacement, attributes[precision: u32] (Fixed128Tensor) -> Fixed128Tensor, FixedpointTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[precision: u32] (Fixed64Tensor) -> Fixed64Tensor, FixedpointTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[precision: u32] (Fixed128Tensor) -> Fixed128Tensor, FixedpointTruncPrOp);

hybrid_kernel! {
    FixedpointTruncPrOp,
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => attributes[precision] Self::host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => attributes[precision] Self::host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => attributes[precision] Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => attributes[precision] Self::rep_kernel),
    ]
}

impl FixedpointTruncPrOp {
    fn host_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        HostPlacement: PlacementReveal<S, RepT, RingT>,
        HostPlacement: PlacementShr<S, RingT, RingT>,
    {
        let x_revealed = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let result = plc.shr(sess, precision as usize, &x_revealed);
        FixedTensor::Host(result)
    }

    fn rep_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        precision: u32,
        x: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, RingT, RepT>,
        ReplicatedPlacement: PlacementTruncPr<S, RepT, RepT>,
    {
        let setup = plc.gen_setup(sess);

        let x_shared = match x {
            FixedTensor::Host(x) => plc.share(sess, &setup, &x),
            FixedTensor::Replicated(x) => x,
        };

        let result = plc.trunc_pr(sess, precision, &x_shared);
        FixedTensor::Replicated(result)
    }
}

modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (Fixed64Tensor) -> Fixed64Tensor, FixedpointSumOp);
modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (Fixed128Tensor) -> Fixed128Tensor, FixedpointSumOp);
modelled!(PlacementSum::sum, ReplicatedPlacement, attributes[axis: Option<u32>] (Fixed64Tensor) -> Fixed64Tensor, FixedpointSumOp);
modelled!(PlacementSum::sum, ReplicatedPlacement, attributes[axis: Option<u32>] (Fixed128Tensor) -> Fixed128Tensor, FixedpointSumOp);

hybrid_kernel! {
    FixedpointSumOp,
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => attributes[axis] Self::host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => attributes[axis] Self::host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => attributes[axis] Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => attributes[axis] Self::rep_kernel),
    ]
}

impl FixedpointSumOp {
    fn host_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        HostPlacement: PlacementReveal<S, RepT, RingT>,
        HostPlacement: PlacementSum<S, RingT, RingT>,
    {
        let x_revealed = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let result = plc.sum(sess, axis, &x_revealed);
        FixedTensor::Host(result)
    }

    fn rep_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, RingT, RepT>,
        ReplicatedPlacement: PlacementSum<S, RepT, RepT>,
    {
        let x_shared = match x {
            FixedTensor::Host(x) => {
                let setup = plc.gen_setup(sess);
                plc.share(sess, &setup, &x)
            }
            FixedTensor::Replicated(x) => x,
        };

        let result = plc.sum(sess, axis, &x_shared);
        FixedTensor::Replicated(result)
    }
}

modelled!(PlacementRingMean::ring_mean, HostPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (Fixed64Tensor) -> Fixed64Tensor, FixedpointMeanOp);
modelled!(PlacementRingMean::ring_mean, HostPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (Fixed128Tensor) -> Fixed128Tensor, FixedpointMeanOp);
modelled!(PlacementRingMean::ring_mean, ReplicatedPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (Fixed64Tensor) -> Fixed64Tensor, FixedpointMeanOp);
modelled!(PlacementRingMean::ring_mean, ReplicatedPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (Fixed128Tensor) -> Fixed128Tensor, FixedpointMeanOp);

hybrid_kernel! {
    FixedpointMeanOp,
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => attributes[axis, scaling_base, scaling_exp] Self::host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => attributes[axis, scaling_base, scaling_exp] Self::host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => attributes[axis, scaling_base, scaling_exp] Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => attributes[axis, scaling_base, scaling_exp] Self::rep_kernel),
    ]
}

impl FixedpointMeanOp {
    fn host_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        HostPlacement: PlacementReveal<S, RepT, RingT>,
        HostPlacement: PlacementRingMean<S, RingT, RingT>,
    {
        let x_revealed = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let result = plc.ring_mean(sess, axis, scaling_base, scaling_exp, &x_revealed);
        FixedTensor::Host(result)
    }

    fn rep_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        _scaling_base: u64,
        scaling_exp: u32,
        x: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, RingT, RepT>,
        ReplicatedPlacement: PlacementMean<S, RepT, RepT>,
    {
        let x_shared = match x {
            FixedTensor::Host(x) => {
                let setup = plc.gen_setup(sess);
                plc.share(sess, &setup, &x)
            }
            FixedTensor::Replicated(x) => x,
        };

        let result = plc.mean(sess, axis, scaling_exp.into(), &x_shared);
        FixedTensor::Replicated(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::SyncSession;
    use proptest::prelude::*;

    #[test]
    fn ring_fixedpoint() {
        let x = HostFloat64Tensor::from(
            array![1.0, -2.0, 3.0, -4.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );

        let scaling_factor = 2u64.pow(16);
        let x_encoded = HostRing64Tensor::encode(&x, scaling_factor);
        assert_eq!(
            x_encoded,
            HostRing64Tensor::from(vec![
                65536,
                18446744073709420544,
                196608,
                18446744073709289472
            ])
        );

        let x_decoded = HostRing64Tensor::decode(&x_encoded, scaling_factor);
        assert_eq!(x_decoded, x);

        let scaling_factor_long = 2u128.pow(80);
        let x_encoded = HostRing128Tensor::encode(&x, scaling_factor_long);
        assert_eq!(
            x_encoded,
            HostRing128Tensor::from(vec![
                1208925819614629174706176,
                340282366920936045611735378173418799104,
                3626777458843887524118528,
                340282366920933627760096148915069386752
            ])
        );

        let x_decoded_long = HostRing128Tensor::decode(&x_encoded, scaling_factor_long);
        assert_eq!(x_decoded_long, x);
    }

    #[test]
    fn ring_mean_with_axis() {
        let x_backing: HostFloat64Tensor = HostFloat64Tensor::from(
            array![[1., 2.], [3., 4.]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let encoding_factor = 2u64.pow(16);
        let decoding_factor = 2u64.pow(32);
        let x = HostRing64Tensor::encode(&x_backing, encoding_factor);
        let out = HostRing64Tensor::ring_mean(x, Some(0), encoding_factor);
        let dec = HostRing64Tensor::decode(&out, decoding_factor);
        assert_eq!(
            dec,
            HostFloat64Tensor::from(array![2., 3.].into_dimensionality::<IxDyn>().unwrap())
        );
    }

    #[test]
    fn ring_mean_no_axis() {
        let x_backing: HostFloat64Tensor = HostFloat64Tensor::from(
            array![[1., 2.], [3., 4.]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let encoding_factor = 2u64.pow(16);
        let decoding_factor = 2u64.pow(32);
        let x = HostRing64Tensor::encode(&x_backing, encoding_factor);
        let out = HostRing64Tensor::ring_mean(x, None, encoding_factor);
        let dec = HostRing64Tensor::decode(&out, decoding_factor);
        assert_eq!(
            dec.0.into_shape((1,)).unwrap(),
            array![2.5].into_shape((1,)).unwrap()
        );
    }

    macro_rules! host_binary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };

                let x = FixedTensor::Host(AbstractHostRingTensor::from_raw_plc(
                    xs,
                    alice.clone(),
                ));
                let y = FixedTensor::Host(AbstractHostRingTensor::from_raw_plc(
                    ys,
                    alice.clone(),
                ));

                let sess = SyncSession::default();

                let sum = alice.$test_func(&sess, &x, &y);
                let opened_product = match sum {
                    FixedTensor::Host(r) => r,
                    _ => panic!("Should not produce a replicated tensor on a host placement"),
                };
                assert_eq!(
                    opened_product,
                    AbstractHostRingTensor::from_raw_plc(zs, alice.clone())
                );
            }
        };
    }

    host_binary_func_test!(test_host_add64, add<u64>);
    host_binary_func_test!(test_host_add128, add<u128>);
    host_binary_func_test!(test_host_sub64, sub<u64>);
    host_binary_func_test!(test_host_sub128, sub<u128>);
    host_binary_func_test!(test_host_mul64, mul<u64>);
    host_binary_func_test!(test_host_mul128, mul<u128>);
    host_binary_func_test!(test_host_dot64, dot<u64>);
    host_binary_func_test!(test_host_dot128, dot<u128>);

    #[test]
    fn test_fixed_host_mul64() {
        let a = vec![1u64, 2];
        let b = vec![3u64, 4];
        let a = Array::from_shape_vec(IxDyn(&[a.len()]), a).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[b.len()]), b).unwrap();
        let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
        for i in 0..a.len() {
            target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
        }
        test_host_mul64(a, b, target);
    }

    macro_rules! rep_binary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = FixedTensor::HostTensor(AbstractHostRingTensor::from_raw_plc(xs, alice.clone()));
                let y = FixedTensor::HostTensor(AbstractHostRingTensor::from_raw_plc(ys, alice.clone()));

                let sess = SyncSession::default();

                let sum = rep.$test_func(&sess, &x, &y);
                let opened_product = match sum {
                    FixedTensor::ReplicatedTensor(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an unreplicated tensor on a replicated placement"),
                };
                assert_eq!(
                    opened_product,
                    AbstractHostRingTensor::from_raw_plc(zs, alice.clone())
                );
            }
        };
    }

    rep_binary_func_test!(test_rep_add64, add<u64>);
    rep_binary_func_test!(test_rep_add128, add<u128>);
    rep_binary_func_test!(test_rep_sub64, sub<u64>);
    rep_binary_func_test!(test_rep_sub128, sub<u128>);
    rep_binary_func_test!(test_rep_mul64, mul<u64>);
    rep_binary_func_test!(test_rep_mul128, mul<u128>);
    rep_binary_func_test!(test_rep_dot64, dot<u64>);
    rep_binary_func_test!(test_rep_dot128, dot<u128>);

    macro_rules! pairwise_same_length {
        ($func_name:ident, $tt: ident) => {
            fn $func_name() -> impl Strategy<Value = (ArrayD<$tt>, ArrayD<$tt>)> {
                (1usize..25)
                    .prop_flat_map(|length| {
                        (
                            proptest::collection::vec(any::<$tt>(), length),
                            proptest::collection::vec(any::<$tt>(), length),
                        )
                    })
                    .prop_map(|(x, y)| {
                        let a = Array::from_shape_vec(IxDyn(&[x.len()]), x).unwrap();
                        let b = Array::from_shape_vec(IxDyn(&[y.len()]), y).unwrap();
                        (a, b)
                    })
                    .boxed()
            }
        };
    }

    pairwise_same_length!(pairwise_same_length64, u64);
    pairwise_same_length!(pairwise_same_length128, u128);

    #[test]
    fn test_fixed_rep_mul64() {
        let a = vec![1u64, 2];
        let b = vec![3u64, 4];
        let a = Array::from_shape_vec(IxDyn(&[a.len()]), a).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[b.len()]), b).unwrap();
        let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
        for i in 0..a.len() {
            target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
        }
        test_rep_mul64(a, b, target);
    }

    proptest! {
        #[test]
        fn test_fuzzy_host_add64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) + std::num::Wrapping(b[i])).0;
            }
            test_host_add64(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_add128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) + std::num::Wrapping(b[i])).0;
            }
            test_host_add128(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_sub64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) - std::num::Wrapping(b[i])).0;
            }
            test_host_sub64(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_sub128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) - std::num::Wrapping(b[i])).0;
            }
            test_host_sub128(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_mul64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_host_mul64(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_mul128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_host_mul128(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_dot64((a,b) in pairwise_same_length64())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_host_dot64(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_dot128((a,b) in pairwise_same_length128())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_host_dot128(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_add64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) + std::num::Wrapping(b[i])).0;
            }
            test_rep_add64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_add128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) + std::num::Wrapping(b[i])).0;
            }
            test_rep_add128(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_sub64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) - std::num::Wrapping(b[i])).0;
            }
            test_rep_sub64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_sub128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) - std::num::Wrapping(b[i])).0;
            }
            test_rep_sub128(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_mul64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_rep_mul64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_mul128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_rep_mul128(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_dot64((a,b) in pairwise_same_length64())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_rep_dot64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_dot128((a,b) in pairwise_same_length128())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_rep_dot128(a, b, target);
        }
    }
}
