//! Support for fixed-point arithmetic

use crate::computation::*;
use crate::error::Result;
use crate::floatingpoint::{Float32Tensor, Float64Tensor, FloatTensor};
use crate::host::*;
use crate::kernels::*;
use crate::replicated::{
    AbstractReplicatedFixedTensor, ReplicatedFixed128Tensor, ReplicatedFixed64Tensor,
    ReplicatedShape,
};
use macros::with_context;
use ndarray::prelude::*;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::num::Wrapping;
use std::ops::Mul;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FixedTensor<HostFixedT, RepFixedT> {
    Host(HostFixedT),
    Replicated(RepFixedT),
}

moose_type!(Fixed64Tensor = FixedTensor<HostFixed64Tensor, ReplicatedFixed64Tensor>);
moose_type!(Fixed128Tensor = FixedTensor<HostFixed128Tensor, ReplicatedFixed128Tensor>);

impl<HostFixedT, RepFixedT> Placed for FixedTensor<HostFixedT, RepFixedT>
where
    HostFixedT: Placed,
    HostFixedT::Placement: Into<Placement>,
    RepFixedT: Placed,
    RepFixedT::Placement: Into<Placement>,
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

impl<T> AbstractHostRingTensor<T>
where
    Wrapping<T>: Clone + Zero + Mul<Wrapping<T>, Output = Wrapping<T>>,
    AbstractHostRingTensor<T>: Convert<HostFloat64Tensor>,
{
    pub fn fixedpoint_mean(
        x: Self,
        axis: Option<usize>,
        scaling_factor: <AbstractHostRingTensor<T> as Convert<HostFloat64Tensor>>::Scale,
    ) -> AbstractHostRingTensor<T> {
        let mean_weight = Self::compute_mean_weight(&x, &axis);
        let encoded_weight = AbstractHostRingTensor::<T>::encode(&mean_weight, scaling_factor);
        let operand_sum = x.sum(axis);
        operand_sum.mul(encoded_weight)
    }

    fn compute_mean_weight(x: &Self, axis: &Option<usize>) -> HostFloat64Tensor {
        let shape: &[usize] = x.0.shape();
        if let Some(ax) = axis {
            let dim_len = shape[*ax] as f64;
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

modelled!(PlacementFixedpointEncode::fixedpoint_encode, HostPlacement, attributes[fractional_precision: u32, integral_precision: u32] (Float32Tensor) -> Fixed64Tensor, FixedpointEncodeOp);
modelled!(PlacementFixedpointEncode::fixedpoint_encode, HostPlacement, attributes[fractional_precision: u32, integral_precision: u32] (Float64Tensor) -> Fixed128Tensor, FixedpointEncodeOp);
modelled!(PlacementFixedpointEncode::fixedpoint_encode, HostPlacement, attributes[fractional_precision: u32, integral_precision: u32] (HostFloat32Tensor) -> HostFixed64Tensor, FixedpointEncodeOp);
modelled!(PlacementFixedpointEncode::fixedpoint_encode, HostPlacement, attributes[fractional_precision: u32, integral_precision: u32] (HostFloat64Tensor) -> HostFixed128Tensor, FixedpointEncodeOp);

kernel! {
    FixedpointEncodeOp,
    [
        (HostPlacement, (Float32Tensor) -> Fixed64Tensor => [hybrid] attributes[fractional_precision, integral_precision] Self::fixed_kernel),
        (HostPlacement, (Float64Tensor) -> Fixed128Tensor => [hybrid] attributes[fractional_precision, integral_precision] Self::fixed_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFixed64Tensor => [hybrid] attributes[fractional_precision, integral_precision] Self::hostfixed_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFixed128Tensor => [hybrid] attributes[fractional_precision, integral_precision] Self::hostfixed_kernel),
    ]
}

impl FixedpointEncodeOp {
    fn fixed_kernel<S: Session, HostFloatT, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        fractional_precision: u32,
        integral_precision: u32,
        x: FloatTensor<HostFloatT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        HostPlacement: PlacementFixedpointEncode<S, HostFloatT, HostFixedT>,
    {
        match x {
            FloatTensor::Host(x) => {
                let x = plc.fixedpoint_encode(sess, fractional_precision, integral_precision, &x);
                FixedTensor::Host(x)
            }
        }
    }

    fn hostfixed_kernel<S: Session, HostFloatT, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        fractional_precision: u32,
        integral_precision: u32,
        x: HostFloatT,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementRingFixedpointEncode<S, HostFloatT, HostRingT>,
    {
        // TODO(Morten) inline this function?
        let y = plc.fixedpoint_ring_encode(sess, 2, fractional_precision, &x);
        AbstractHostFixedTensor {
            tensor: y,
            fractional_precision,
            integral_precision,
        }
    }
}

modelled!(PlacementFixedpointDecode::fixedpoint_decode, HostPlacement, attributes[fractional_precision: u32] (Fixed64Tensor) -> Float32Tensor, FixedpointDecodeOp);
modelled!(PlacementFixedpointDecode::fixedpoint_decode, HostPlacement, attributes[fractional_precision: u32] (Fixed128Tensor) -> Float64Tensor, FixedpointDecodeOp);
modelled!(PlacementFixedpointDecode::fixedpoint_decode, HostPlacement, attributes[fractional_precision: u32] (HostFixed64Tensor) -> HostFloat32Tensor, FixedpointDecodeOp);
modelled!(PlacementFixedpointDecode::fixedpoint_decode, HostPlacement, attributes[fractional_precision: u32] (HostFixed128Tensor) -> HostFloat64Tensor, FixedpointDecodeOp);

kernel! {
    FixedpointDecodeOp,
    [
        (HostPlacement, (Fixed64Tensor) -> Float32Tensor => [hybrid] attributes[fractional_precision] Self::fixed_kernel),
        (HostPlacement, (Fixed128Tensor) -> Float64Tensor => [hybrid] attributes[fractional_precision] Self::fixed_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFloat32Tensor => [hybrid] attributes[fractional_precision] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFloat64Tensor => [hybrid] attributes[fractional_precision] Self::hostfixed_kernel),
    ]
}

impl FixedpointDecodeOp {
    fn fixed_kernel<S: Session, HostFixedT, RepFixedT, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FloatTensor<HostFloatT>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementFixedpointDecode<S, HostFixedT, HostFloatT>,
    {
        let v = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = plc.fixedpoint_decode(sess, precision, &v);
        FloatTensor::Host(y)
    }

    fn hostfixed_kernel<S: Session, HostRingT, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: AbstractHostFixedTensor<HostRingT>,
    ) -> HostFloatT
    where
        HostPlacement: PlacementRingFixedpointDecode<S, HostRingT, HostFloatT>,
    {
        // TODO(Morten) inline this function?
        assert_eq!(x.fractional_precision, precision);
        plc.fixedpoint_ring_decode(sess, 2, precision, &x.tensor)
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

kernel! {
    FixedpointAddOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [hybrid] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [hybrid] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [hybrid] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [hybrid] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::repfixed_kernel),
    ]
}

impl FixedpointAddOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
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

        let z = plc.add(sess, &x, &y);
        FixedTensor::Host(z)
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
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

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.add(sess, &x.tensor, &y.tensor);
        AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        }
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> AbstractReplicatedFixedTensor<RepRingT>
    where
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.add(sess, &x.tensor, &y.tensor);
        AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        }
    }
}

modelled!(PlacementSub::sub, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, FixedpointSubOp);

kernel! {
    FixedpointSubOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [hybrid] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [hybrid] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [hybrid] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [hybrid] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::repfixed_kernel),
    ]
}

impl FixedpointSubOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
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

        let z = plc.sub(sess, &x, &y);
        FixedTensor::Host(z)
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementSub<S, RepFixedT, RepFixedT, RepFixedT>,
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

        let z = plc.sub(sess, &x, &y);
        FixedTensor::Replicated(z)
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.sub(sess, &x.tensor, &y.tensor);
        AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        }
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> AbstractReplicatedFixedTensor<RepRingT>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.sub(sess, &x.tensor, &y.tensor);
        AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        }
    }
}

modelled!(PlacementMul::mul, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, FixedpointMulOp);

kernel! {
    FixedpointMulOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [hybrid] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [hybrid] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [hybrid] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [hybrid] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::repfixed_kernel),
    ]
}

impl FixedpointMulOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementMul<S, HostFixedT, HostFixedT, HostFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        let z = with_context!(plc, sess, x * y);
        FixedTensor::Host(z)
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementMul<S, RepFixedT, RepFixedT, RepFixedT>,
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

        let z = with_context!(plc, sess, x * y);
        FixedTensor::Replicated(z)
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementMul<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.mul(sess, &x.tensor, &y.tensor);
        AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        }
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> AbstractReplicatedFixedTensor<RepRingT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementMulSetup<S, S::ReplicatedSetup, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let setup = plc.gen_setup(sess);
        let z = plc.mul_setup(sess, &setup, &x.tensor, &y.tensor);
        AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.fractional_precision, y.fractional_precision),
        }
    }
}

modelled!(PlacementDiv::div, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointDivOp);
modelled!(PlacementDiv::div, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointDivOp);
modelled!(PlacementDiv::div, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointDivOp);
modelled!(PlacementDiv::div, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointDivOp);
modelled!(PlacementDiv::div, HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor, FixedpointDivOp);
modelled!(PlacementDiv::div, HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor, FixedpointDivOp);
modelled!(PlacementDiv::div, ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, FixedpointDivOp);
modelled!(PlacementDiv::div, ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, FixedpointDivOp);

kernel! {
    FixedpointDivOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [hybrid] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [hybrid] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [hybrid] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [hybrid] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_rep_kernel),
    ]
}

impl FixedpointDivOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDiv<S, HostFixedT, HostFixedT, HostFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        let z = plc.div(sess, &x, &y);
        FixedTensor::Host(z)
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementDiv<S, RepFixedT, RepFixedT, RepFixedT>,
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

        let z = plc.div(sess, &x, &y);
        FixedTensor::Replicated(z)
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementDiv<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.div(sess, &x.tensor, &y.tensor);
        AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        }
    }
}

modelled!(PlacementDot::dot, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, FixedpointDotOp);

kernel! {
    FixedpointDotOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [hybrid] Self::fixed_on_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::fixed_on_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [hybrid] Self::fixed_on_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::fixed_on_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [hybrid] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [hybrid] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::repfixed_kernel),
    ]
}

impl FixedpointDotOp {
    fn fixed_on_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDot<S, HostFixedT, HostFixedT, HostFixedT>,
    {
        let x_revealed = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };
        let y_revealed = match y {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let z = plc.dot(sess, &x_revealed, &y_revealed);
        FixedTensor::Host(z)
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
        ReplicatedPlacement: PlacementDot<S, RepFixedT, RepFixedT, RepFixedT>,
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

        let z = plc.dot(sess, &x_shared, &y_shared);
        FixedTensor::Replicated(z)
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementDot<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.dot(sess, &x.tensor, &y.tensor);
        AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        }
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> AbstractReplicatedFixedTensor<RepRingT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementDotSetup<S, S::ReplicatedSetup, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let setup = plc.gen_setup(sess);
        let z = plc.dot_setup(sess, &setup, &x.tensor, &y.tensor);
        AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        }
    }
}

modelled!(PlacementTruncPr::trunc_pr, HostPlacement, attributes[precision: u32] (Fixed64Tensor) -> Fixed64Tensor, FixedpointTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, HostPlacement, attributes[precision: u32] (Fixed128Tensor) -> Fixed128Tensor, FixedpointTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[precision: u32] (Fixed64Tensor) -> Fixed64Tensor, FixedpointTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[precision: u32] (Fixed128Tensor) -> Fixed128Tensor, FixedpointTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, HostPlacement, attributes[precision: u32] (HostFixed64Tensor) -> HostFixed64Tensor, FixedpointTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, HostPlacement, attributes[precision: u32] (HostFixed128Tensor) -> HostFixed128Tensor, FixedpointTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[precision: u32] (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, FixedpointTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[precision: u32] (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, FixedpointTruncPrOp);

kernel! {
    FixedpointTruncPrOp,
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [hybrid] attributes[precision] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [hybrid] attributes[precision] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [hybrid] attributes[precision] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [hybrid] attributes[precision] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [hybrid] attributes[precision] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [hybrid] attributes[precision] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] attributes[precision] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] attributes[precision] Self::repfixed_kernel),
    ]
}

impl FixedpointTruncPrOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementTruncPr<S, HostFixedT, HostFixedT>,
    {
        let v = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let z = plc.trunc_pr(sess, precision, &v);
        FixedTensor::Host(z)
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        precision: u32,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementTruncPr<S, RepFixedT, RepFixedT>,
    {
        let setup = plc.gen_setup(sess);

        let v = match x {
            FixedTensor::Host(x) => plc.share(sess, &setup, &x),
            FixedTensor::Replicated(x) => x,
        };

        let z = plc.trunc_pr(sess, precision, &v);
        FixedTensor::Replicated(z)
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementShr<S, HostRingT, HostRingT>,
    {
        // NOTE(Morten) we assume fixedpoint base is 2 so that truncation becomes (integer) division by 2**precision
        let z = plc.shr(sess, precision as usize, &x.tensor);
        AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision - precision,
            integral_precision: x.integral_precision,
        }
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        precision: u32,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> AbstractReplicatedFixedTensor<RepRingT>
    where
        ReplicatedPlacement: PlacementTruncPr<S, RepRingT, RepRingT>,
    {
        let z = plc.trunc_pr(sess, precision, &x.tensor);
        AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision - precision,
            integral_precision: x.integral_precision,
        }
    }
}

modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (Fixed64Tensor) -> Fixed64Tensor, FixedpointSumOp);
modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (Fixed128Tensor) -> Fixed128Tensor, FixedpointSumOp);
modelled!(PlacementSum::sum, ReplicatedPlacement, attributes[axis: Option<u32>] (Fixed64Tensor) -> Fixed64Tensor, FixedpointSumOp);
modelled!(PlacementSum::sum, ReplicatedPlacement, attributes[axis: Option<u32>] (Fixed128Tensor) -> Fixed128Tensor, FixedpointSumOp);
modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (HostFixed64Tensor) -> HostFixed64Tensor, FixedpointSumOp);
modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (HostFixed128Tensor) -> HostFixed128Tensor, FixedpointSumOp);
modelled!(PlacementSum::sum, ReplicatedPlacement, attributes[axis: Option<u32>] (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, FixedpointSumOp);
modelled!(PlacementSum::sum, ReplicatedPlacement, attributes[axis: Option<u32>] (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, FixedpointSumOp);

kernel! {
    FixedpointSumOp,
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [hybrid] attributes[axis] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [hybrid] attributes[axis] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [hybrid] attributes[axis] Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [hybrid] attributes[axis] Self::rep_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [hybrid] attributes[axis] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [hybrid] attributes[axis] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] attributes[axis] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] attributes[axis] Self::repfixed_kernel),
    ]
}

impl FixedpointSumOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementSum<S, HostFixedT, HostFixedT>,
    {
        let v = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let result = plc.sum(sess, axis, &v);
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

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementSum<S, HostRingT, HostRingT>,
    {
        let z = plc.sum(sess, axis, &x.tensor);
        AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        }
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> AbstractReplicatedFixedTensor<RepRingT>
    where
        ReplicatedPlacement: PlacementSum<S, RepRingT, RepRingT>,
    {
        let z = plc.sum(sess, axis, &x.tensor);
        AbstractReplicatedFixedTensor {
            tensor: z,
            integral_precision: x.integral_precision,
            fractional_precision: x.fractional_precision,
        }
    }
}

modelled!(PlacementMean::mean, HostPlacement, attributes[axis: Option<u32>] (Fixed64Tensor) -> Fixed64Tensor, FixedpointMeanOp);
modelled!(PlacementMean::mean, HostPlacement, attributes[axis: Option<u32>] (Fixed128Tensor) -> Fixed128Tensor, FixedpointMeanOp);
modelled!(PlacementMean::mean, ReplicatedPlacement, attributes[axis: Option<u32>] (Fixed64Tensor) -> Fixed64Tensor, FixedpointMeanOp);
modelled!(PlacementMean::mean, ReplicatedPlacement, attributes[axis: Option<u32>] (Fixed128Tensor) -> Fixed128Tensor, FixedpointMeanOp);
modelled!(PlacementMean::mean, HostPlacement, attributes[axis: Option<u32>] (HostFixed64Tensor) -> HostFixed64Tensor, FixedpointMeanOp);
modelled!(PlacementMean::mean, HostPlacement, attributes[axis: Option<u32>] (HostFixed128Tensor) -> HostFixed128Tensor, FixedpointMeanOp);
modelled!(PlacementMean::mean, ReplicatedPlacement, attributes[axis: Option<u32>] (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, FixedpointMeanOp);
modelled!(PlacementMean::mean, ReplicatedPlacement, attributes[axis: Option<u32>] (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, FixedpointMeanOp);

kernel! {
    FixedpointMeanOp,
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [hybrid] attributes[axis] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [hybrid] attributes[axis] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [hybrid] attributes[axis] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [hybrid] attributes[axis] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [hybrid] attributes[axis] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [hybrid] attributes[axis] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] attributes[axis] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] attributes[axis] Self::repfixed_kernel),
    ]
}

impl FixedpointMeanOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementMean<S, HostFixedT, HostFixedT>,
    {
        let x_revealed = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let result = plc.mean(sess, axis, &x_revealed);
        FixedTensor::Host(result)
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> FixedTensor<HostFixedT, RepFixedT>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementMean<S, RepFixedT, RepFixedT>,
    {
        let x_shared = match x {
            FixedTensor::Host(x) => {
                let setup = plc.gen_setup(sess);
                plc.share(sess, &setup, &x)
            }
            FixedTensor::Replicated(x) => x,
        };

        let result = plc.mean(sess, axis, &x_shared);
        FixedTensor::Replicated(result)
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementMeanAsFixedpoint<S, HostRingT, HostRingT>,
    {
        let y = plc.mean_as_fixedpoint(sess, axis, 2, x.fractional_precision, &x.tensor);
        AbstractHostFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision * 2,
            integral_precision: x.integral_precision,
        }
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> AbstractReplicatedFixedTensor<RepRingT>
    where
        ReplicatedPlacement: PlacementMeanAsFixedpoint<S, RepRingT, RepRingT>,
    {
        let y = plc.mean_as_fixedpoint(sess, axis, 2, x.fractional_precision, &x.tensor);
        AbstractReplicatedFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision * 2,
            integral_precision: x.integral_precision,
        }
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
        let x_encoded = HostFixed64Tensor {
            tensor: HostRing64Tensor::encode(&x, scaling_factor),
            fractional_precision: 16,
            integral_precision: 5,
        };

        assert_eq!(
            x_encoded,
            HostFixed64Tensor {
                tensor: HostRing64Tensor::from(vec![
                    65536,
                    18446744073709420544,
                    196608,
                    18446744073709289472
                ]),
                fractional_precision: 16,
                integral_precision: 5,
            }
        );
        let x_decoded = HostRing64Tensor::decode(&x_encoded.tensor, scaling_factor);
        assert_eq!(x_decoded, x);

        let scaling_factor_long = 2u128.pow(80);
        let x_encoded = HostFixed128Tensor {
            tensor: HostRing128Tensor::encode(&x, scaling_factor_long),
            fractional_precision: 80,
            integral_precision: 5,
        };

        assert_eq!(
            x_encoded,
            HostFixed128Tensor {
                tensor: HostRing128Tensor::from(vec![
                    1208925819614629174706176,
                    340282366920936045611735378173418799104,
                    3626777458843887524118528,
                    340282366920933627760096148915069386752
                ]),
                fractional_precision: 80,
                integral_precision: 5,
            }
        );

        let x_decoded_long = HostRing128Tensor::decode(&x_encoded.tensor, scaling_factor_long);
        assert_eq!(x_decoded_long, x);
    }

    #[test]
    fn fixedpoint_mean_with_axis() {
        let x_backing = HostFloat64Tensor::from(
            array![[1., 2.], [3., 4.]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let encoding_factor = 2u64.pow(16);
        let decoding_factor = 2u64.pow(32);
        let x = HostRing64Tensor::encode(&x_backing, encoding_factor);
        let out = HostRing64Tensor::fixedpoint_mean(x, Some(0), encoding_factor);
        let dec = HostRing64Tensor::decode(&out, decoding_factor);
        assert_eq!(
            dec,
            HostFloat64Tensor::from(array![2., 3.].into_dimensionality::<IxDyn>().unwrap())
        );
    }

    #[test]
    fn fixedpoint_mean_no_axis() {
        let x_backing = HostFloat64Tensor::from(
            array![[1., 2.], [3., 4.]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let encoding_factor = 2u64.pow(16);
        let decoding_factor = 2u64.pow(32);
        let x = HostRing64Tensor::encode(&x_backing, encoding_factor);
        let out = HostRing64Tensor::fixedpoint_mean(x, None, encoding_factor);
        let dec = HostRing64Tensor::decode(&out, decoding_factor);
        assert_eq!(
            dec.0.into_shape((1,)).unwrap(),
            array![2.5].into_shape((1,)).unwrap()
        );
    }

    fn new_host_fixed_tensor<HostRingT>(x: HostRingT) -> AbstractHostFixedTensor<HostRingT> {
        AbstractHostFixedTensor {
            tensor: x,
            fractional_precision: 15,
            integral_precision: 8,
        }
    }

    macro_rules! host_binary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $factor: expr) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };

                let x = FixedTensor::Host(new_host_fixed_tensor(
                    AbstractHostRingTensor::from_raw_plc(xs, alice.clone()),
                ));
                let y = FixedTensor::Host(new_host_fixed_tensor(
                    AbstractHostRingTensor::from_raw_plc(ys, alice.clone()),
                ));

                let sess = SyncSession::default();

                let sum = alice.$test_func(&sess, &x, &y);
                let opened_product = match sum {
                    FixedTensor::Host(r) => r,
                    _ => panic!("Should not produce a replicated tensor on a host placement"),
                };
                assert_eq!(
                    opened_product.tensor,
                    AbstractHostRingTensor::from_raw_plc(zs, alice.clone())
                );
                let expected_precision = match x {
                    FixedTensor::Host(x) => x.fractional_precision * $factor,
                    _ => unreachable!(),
                };
                assert_eq!(opened_product.fractional_precision, expected_precision);
            }
        };
    }

    host_binary_func_test!(test_host_add64, add<u64>, 1);
    host_binary_func_test!(test_host_add128, add<u128>, 1);
    host_binary_func_test!(test_host_sub64, sub<u64>, 1);
    host_binary_func_test!(test_host_sub128, sub<u128>, 1);
    host_binary_func_test!(test_host_mul64, mul<u64>, 2);
    host_binary_func_test!(test_host_mul128, mul<u128>, 2);
    host_binary_func_test!(test_host_dot64, dot<u64>, 2);
    host_binary_func_test!(test_host_dot128, dot<u128>, 2);

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
        ($func_name:ident, $test_func: ident<$tt: ty>, $factor: expr) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = FixedTensor::Host(new_host_fixed_tensor(AbstractHostRingTensor::from_raw_plc(xs, alice.clone())));
                let y = FixedTensor::Host(new_host_fixed_tensor(AbstractHostRingTensor::from_raw_plc(ys, alice.clone())));

                let sess = SyncSession::default();

                let sum = rep.$test_func(&sess, &x, &y);
                let opened_product = match sum {
                    FixedTensor::Replicated(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an unreplicated tensor on a replicated placement"),
                };
                assert_eq!(
                    opened_product.tensor,
                    AbstractHostRingTensor::from_raw_plc(zs, alice.clone())
                );
                println!("Result from division: {:?}", opened_product);
                let expected_precision = match x {
                    FixedTensor::Host(x) => x.fractional_precision * $factor,
                    _ => unreachable!(),
                };
                assert_eq!(
                    opened_product.fractional_precision,
                    expected_precision
                );
            }
        };
    }

    rep_binary_func_test!(test_rep_add64, add<u64>, 1);
    rep_binary_func_test!(test_rep_add128, add<u128>, 1);
    rep_binary_func_test!(test_rep_sub64, sub<u64>, 1);
    rep_binary_func_test!(test_rep_sub128, sub<u128>, 1);
    rep_binary_func_test!(test_rep_mul64, mul<u64>, 2);
    rep_binary_func_test!(test_rep_mul128, mul<u128>, 2);
    rep_binary_func_test!(test_rep_dot64, dot<u64>, 2);
    rep_binary_func_test!(test_rep_dot128, dot<u128>, 2);

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

    macro_rules! rep_div_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $factor: expr) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };
                // this actually doesn't do any encoding
                let x = FixedTensor::Host(new_host_fixed_tensor(AbstractHostRingTensor::from_raw_plc(xs, alice.clone())));
                let y = FixedTensor::Host(new_host_fixed_tensor(AbstractHostRingTensor::from_raw_plc(ys, alice.clone())));


                let sess = SyncSession::default();

                let sum = rep.$test_func(&sess, &x, &y);
                let opened_product = match sum {
                    FixedTensor::Replicated(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an unreplicated tensor on a replicated placement"),
                };
                let expected_result = AbstractHostRingTensor::from_raw_plc(zs,alice.clone());
                let expected_f64 = Convert::decode(&expected_result, 2_u64.pow(15));

                println!("Result from division: {:?}", opened_product);
                let result = Convert::decode(&opened_product.tensor, 2_u64.pow(15));

                assert_eq!(
                    result, expected_f64
                );

            }
        };
    }

    rep_div_func_test!(test_rep_div64, div<u64>, 1);

    #[test]
    fn test_fixed_rep_div64() {
        let a: Vec<u64> = vec![1.0, 2.0, 2.0]
            .iter()
            .map(|item| encode(*item))
            .collect();
        let b: Vec<u64> = vec![3.0, 7.0, 1.41]
            .iter()
            .map(|item| encode(*item))
            .collect();

        fn encode(item: f64) -> u64 {
            (2_i64.pow(15) as f64 * item) as u64
        }

        let a = Array::from_shape_vec(IxDyn(&[a.len()]), a).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[b.len()]), b).unwrap();

        let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
        // fixed(res) = (a/b) * 2^27
        for i in 0..a.len() {
            let div_result = (a[i] as f64) / (b[i] as f64);
            target[i] = encode(div_result);
        }
        test_rep_div64(a, b, target);
    }
}
