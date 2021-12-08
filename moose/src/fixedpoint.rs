//! Support for fixed-point arithmetic

use crate::computation::*;
use crate::error::{Error, Result};
use crate::floatingpoint::{Float32Tensor, Float64Tensor, FloatTensor};
use crate::host::*;
use crate::kernels::*;
use crate::replicated::*;
use crate::symbolic::Symbolic;
use macros::with_context;
use ndarray::prelude::*;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
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
    ) -> Result<AbstractHostRingTensor<T>> {
        let mean_weight = Self::compute_mean_weight(&x, &axis)?;
        let encoded_weight = AbstractHostRingTensor::<T>::encode(&mean_weight, scaling_factor);
        let operand_sum = x.sum(axis)?;
        Ok(operand_sum.mul(encoded_weight))
    }

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

impl IdentityOp {
    pub(crate) fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        HostPlacement: PlacementIdentity<S, HostFixedT, HostFixedT>,
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
    {
        match x {
            FixedTensor::Host(x) => Ok(FixedTensor::Host(plc.identity(sess, &x))),
            FixedTensor::Replicated(x) => {
                let x = plc.reveal(sess, &x);
                Ok(FixedTensor::Host(plc.identity(sess, &x)))
            }
        }
    }

    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementIdentity<S, RepFixedT, RepFixedT>,
    {
        match x {
            FixedTensor::Host(x) => {
                let x = plc.share(sess, &x);
                Ok(FixedTensor::Replicated(plc.identity(sess, &x)))
            }
            FixedTensor::Replicated(x) => Ok(FixedTensor::Replicated(plc.identity(sess, &x))),
        }
    }
}

modelled_kernel! {
    PlacementFixedpointEncode::fixedpoint_encode, FixedpointEncodeOp{fractional_precision: u32, integral_precision: u32},
    [
        (HostPlacement, (Float32Tensor) -> Fixed64Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (Float64Tensor) -> Fixed128Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFixed64Tensor => [hybrid] Self::hostfixed_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFixed128Tensor => [hybrid] Self::hostfixed_kernel),
    ]
}

impl FixedpointEncodeOp {
    fn fixed_kernel<S: Session, HostFloatT, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        fractional_precision: u32,
        integral_precision: u32,
        x: FloatTensor<HostFloatT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        HostPlacement: PlacementFixedpointEncode<S, HostFloatT, HostFixedT>,
    {
        match x {
            FloatTensor::Host(x) => {
                let x = plc.fixedpoint_encode(sess, fractional_precision, integral_precision, &x);
                Ok(FixedTensor::Host(x))
            }
        }
    }

    fn hostfixed_kernel<S: Session, HostFloatT, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        fractional_precision: u32,
        integral_precision: u32,
        x: HostFloatT,
    ) -> Result<AbstractHostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementRingFixedpointEncode<S, HostFloatT, HostRingT>,
    {
        // TODO(Morten) inline this function?
        let y = plc.fixedpoint_ring_encode(sess, 2, fractional_precision, &x);
        Ok(AbstractHostFixedTensor {
            tensor: y,
            fractional_precision,
            integral_precision,
        })
    }
}

modelled_kernel! {
    PlacementFixedpointDecode::fixedpoint_decode, FixedpointDecodeOp{fractional_precision: u32},
    [
        (HostPlacement, (Fixed64Tensor) -> Float32Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (Fixed128Tensor) -> Float64Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFloat32Tensor => [hybrid] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFloat64Tensor => [hybrid] Self::hostfixed_kernel),
    ]
}

impl FixedpointDecodeOp {
    fn fixed_kernel<S: Session, HostFixedT, RepFixedT, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementFixedpointDecode<S, HostFixedT, HostFloatT>,
    {
        let v = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = plc.fixedpoint_decode(sess, precision, &v);
        Ok(FloatTensor::Host(y))
    }

    fn hostfixed_kernel<S: Session, HostRingT, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: AbstractHostFixedTensor<HostRingT>,
    ) -> Result<HostFloatT>
    where
        HostPlacement: PlacementRingFixedpointDecode<S, HostRingT, HostFloatT>,
    {
        // TODO(Morten) inline this function?
        assert_eq!(x.fractional_precision, precision);
        Ok(plc.fixedpoint_ring_decode(sess, 2, precision, &x.tensor))
    }
}

modelled_kernel! {
    PlacementAdd::add, FixedpointAddOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_mirfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_mirfixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::mirfixed_repfixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::mirfixed_repfixed_kernel),
    ]
}

impl FixedpointAddOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
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
        Ok(FixedTensor::Host(z))
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementAdd<S, RepFixedT, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.add(sess, &x, &y);
        Ok(FixedTensor::Replicated(z))
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> Result<AbstractHostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.add(sess, &x.tensor, &y.tensor);
        Ok(AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.add(sess, &x.tensor, &y.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    fn repfixed_mirfixed_kernel<S: Session, RepRingT, MirroredRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractMirroredFixedTensor<MirroredRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementAdd<S, RepRingT, MirroredRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.add(sess, &x.tensor, &y.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: u32::max(x.fractional_precision, y.fractional_precision),
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    fn mirfixed_repfixed_kernel<S: Session, RepRingT, MirroredRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractMirroredFixedTensor<MirroredRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementAdd<S, MirroredRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.add(sess, &x.tensor, &y.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: u32::max(x.fractional_precision, y.fractional_precision),
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }
}

modelled_kernel! {
    PlacementSub::sub, FixedpointSubOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
    ]
}

impl FixedpointSubOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
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
        Ok(FixedTensor::Host(z))
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementSub<S, RepFixedT, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.sub(sess, &x, &y);
        Ok(FixedTensor::Replicated(z))
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> Result<AbstractHostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.sub(sess, &x.tensor, &y.tensor);
        Ok(AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.sub(sess, &x.tensor, &y.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}

modelled_kernel! {
    PlacementMul::mul, FixedpointMulOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_mirfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_mirfixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::mirfixed_repfixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::mirfixed_repfixed_kernel),
    ]
}

impl FixedpointMulOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
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
        Ok(FixedTensor::Host(z))
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementMul<S, RepFixedT, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = with_context!(plc, sess, x * y);
        Ok(FixedTensor::Replicated(z))
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> Result<AbstractHostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementMul<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.mul(sess, &x.tensor, &y.tensor);
        Ok(AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.mul(sess, &x.tensor, &y.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    fn repfixed_mirfixed_kernel<S: Session, RepRingT, MirroredRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractMirroredFixedTensor<MirroredRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementMul<S, RepRingT, MirroredRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.mul(sess, &x.tensor, &y.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    fn mirfixed_repfixed_kernel<S: Session, RepRingT, MirroredRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractMirroredFixedTensor<MirroredRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementMul<S, MirroredRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.mul(sess, &x.tensor, &y.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }
}

modelled_kernel! {
    PlacementDiv::div, FixedpointDivOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::rep_rep_kernel),
    ]
}

impl FixedpointDivOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
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
        Ok(FixedTensor::Host(z))
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementDiv<S, RepFixedT, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.div(sess, &x, &y);
        Ok(FixedTensor::Replicated(z))
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> Result<AbstractHostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementDiv<S, HostRingT, HostRingT, HostRingT>,
        HostPlacement: PlacementShl<S, HostRingT, HostRingT>,
        HostPlacement: PlacementSign<S, HostRingT, HostRingT>,
        HostPlacement: PlacementMul<S, HostRingT, HostRingT, HostRingT>,
    {
        // Fx(x) / Fx(y) = ((x*2^f)*2^f)/ (y*2^f)
        assert_eq!(x.fractional_precision, y.fractional_precision);

        let sgn_x = plc.sign(sess, &x.tensor);
        let sgn_y = plc.sign(sess, &y.tensor);

        let abs_x = plc.mul(sess, &x.tensor, &sgn_x);
        let abs_y = plc.mul(sess, &y.tensor, &sgn_y);

        let x_upshifted = plc.shl(sess, x.fractional_precision as usize, &abs_x);

        let abs_z: HostRingT = plc.div(sess, &x_upshifted, &abs_y);
        let sgn_z = plc.mul(sess, &sgn_x, &sgn_y);

        Ok(AbstractHostFixedTensor {
            tensor: plc.mul(sess, &abs_z, &sgn_z),
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}

modelled_kernel! {
    PlacementDot::dot, FixedpointDotOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_on_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_on_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_on_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_on_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
    ]
}

impl FixedpointDotOp {
    fn fixed_on_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
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
        Ok(FixedTensor::Host(z))
    }

    fn fixed_on_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementDot<S, RepFixedT, RepFixedT, RepFixedT>,
    {
        let x_shared = match x {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };
        let y_shared = match y {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };

        let z = plc.dot(sess, &x_shared, &y_shared);
        Ok(FixedTensor::Replicated(z))
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
        y: AbstractHostFixedTensor<HostRingT>,
    ) -> Result<AbstractHostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementDot<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.dot(sess, &x.tensor, &y.tensor);
        Ok(AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementDot<S, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.dot(sess, &x.tensor, &y.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }
}

modelled_kernel! {
    PlacementTruncPr::trunc_pr, FixedpointTruncPrOp{precision: u32},
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
    ]
}

impl FixedpointTruncPrOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementTruncPr<S, HostFixedT, HostFixedT>,
    {
        let v = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let z = plc.trunc_pr(sess, precision, &v);
        Ok(FixedTensor::Host(z))
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        precision: u32,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementTruncPr<S, RepFixedT, RepFixedT>,
    {
        let v = match x {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };

        let z = plc.trunc_pr(sess, precision, &v);
        Ok(FixedTensor::Replicated(z))
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: AbstractHostFixedTensor<HostRingT>,
    ) -> Result<AbstractHostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementShr<S, HostRingT, HostRingT>,
    {
        // NOTE(Morten) we assume fixedpoint base is 2 so that truncation becomes (integer) division by 2**precision
        let z = plc.shr(sess, precision as usize, &x.tensor);
        Ok(AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision - precision,
            integral_precision: x.integral_precision,
        })
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        precision: u32,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementTruncPr<S, RepRingT, RepRingT>,
    {
        let z = plc.trunc_pr(sess, precision, &x.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision - precision,
            integral_precision: x.integral_precision,
        })
    }
}

modelled_kernel! {
    PlacementSum::sum, FixedpointSumOp{axis: Option<u32>},
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::rep_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
    ]
}

impl FixedpointSumOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementSum<S, HostFixedT, HostFixedT>,
    {
        let v = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let result = plc.sum(sess, axis, &v);
        Ok(FixedTensor::Host(result))
    }

    fn rep_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: FixedTensor<RingT, RepT>,
    ) -> Result<FixedTensor<RingT, RepT>>
    where
        ReplicatedPlacement: PlacementShare<S, RingT, RepT>,
        ReplicatedPlacement: PlacementSum<S, RepT, RepT>,
    {
        let x_shared = match x {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };

        let result = plc.sum(sess, axis, &x_shared);
        Ok(FixedTensor::Replicated(result))
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: AbstractHostFixedTensor<HostRingT>,
    ) -> Result<AbstractHostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementSum<S, HostRingT, HostRingT>,
    {
        let z = plc.sum(sess, axis, &x.tensor);
        Ok(AbstractHostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementSum<S, RepRingT, RepRingT>,
    {
        let z = plc.sum(sess, axis, &x.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: z,
            integral_precision: x.integral_precision,
            fractional_precision: x.fractional_precision,
        })
    }
}

modelled!(PlacementIndexAxis::index_axis, ReplicatedPlacement, attributes[axis:usize, index: usize] (Fixed64Tensor) -> Fixed64Tensor, IndexAxisOp);
modelled!(PlacementIndexAxis::index_axis, ReplicatedPlacement, attributes[axis:usize, index: usize] (Fixed128Tensor) -> Fixed128Tensor, IndexAxisOp);
modelled!(PlacementIndexAxis::index_axis, ReplicatedPlacement, attributes[axis: usize, index: usize] (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, IndexAxisOp);
modelled!(PlacementIndexAxis::index_axis, ReplicatedPlacement, attributes[axis: usize, index: usize] (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, IndexAxisOp);

impl IndexAxisOp {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        index: usize,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementIndexAxis<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.index_axis(sess, axis, index, &x);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        index: usize,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementIndexAxis<S, RepRingT, RepRingT>,
    {
        let y = plc.index_axis(sess, axis, index, &x.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}

impl ShapeOp {
    pub(crate) fn host_fixed_kernel<S: Session, HostFixedT, RepFixedT, HostShapeT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<HostShapeT>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementShape<S, HostFixedT, HostShapeT>,
    {
        let x_revealed = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        Ok(plc.shape(sess, &x_revealed))
    }

    pub(crate) fn rep_fixed_kernel<S: Session, HostFixedT, RepFixedT, RepShapeT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<RepShapeT>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShape<S, RepFixedT, RepShapeT>,
    {
        let x_shared = match x {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };

        Ok(plc.shape(sess, &x_shared))
    }

    pub(crate) fn host_hostfixed_kernel<S: Session, HostRingT, HostShapeT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractHostFixedTensor<HostRingT>,
    ) -> Result<HostShapeT>
    where
        HostPlacement: PlacementShape<S, HostRingT, HostShapeT>,
    {
        Ok(plc.shape(sess, &x.tensor))
    }
}

modelled_kernel! {
    PlacementMean::mean, FixedpointMeanOp{axis: Option<u32>},
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
    ]
}

impl FixedpointMeanOp {
    fn fixed_host_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementMean<S, HostFixedT, HostFixedT>,
    {
        let x_revealed = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let result = plc.mean(sess, axis, &x_revealed);
        Ok(FixedTensor::Host(result))
    }

    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementMean<S, RepFixedT, RepFixedT>,
    {
        let x_shared = match x {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };

        let result = plc.mean(sess, axis, &x_shared);
        Ok(FixedTensor::Replicated(result))
    }

    fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: AbstractHostFixedTensor<HostRingT>,
    ) -> Result<AbstractHostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementMeanAsFixedpoint<S, HostRingT, HostRingT>,
    {
        let y = plc.mean_as_fixedpoint(sess, axis, 2, x.fractional_precision, &x.tensor);
        Ok(AbstractHostFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision * 2,
            integral_precision: x.integral_precision,
        })
    }

    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementMeanAsFixedpoint<S, RepRingT, RepRingT>,
    {
        let y = plc.mean_as_fixedpoint(sess, axis, 2, x.fractional_precision, &x.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision * 2,
            integral_precision: x.integral_precision,
        })
    }
}

modelled_kernel! {
    PlacementNeg::neg, NegOp,
    [
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
    ]
}

impl NegOp {
    fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementNeg<S, RepRingT, RepRingT>,
    {
        let y = plc.neg(sess, &x.tensor);
        Ok(AbstractReplicatedFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}

impl AddNOp {
    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        xs: &[AbstractReplicatedFixedTensor<RepRingT>],
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementAddN<S, RepRingT, RepRingT>,
        RepRingT: Clone,
    {
        let fractional_precision = xs[0].fractional_precision;
        let integral_precision = xs
            .iter()
            .fold(xs[0].integral_precision, |a, b| a.max(b.integral_precision));

        assert!(xs
            .iter()
            .all(|x| x.fractional_precision == fractional_precision));

        let zs: Vec<RepRingT> = xs.iter().map(|item| item.tensor.clone()).collect();

        Ok(AbstractReplicatedFixedTensor {
            tensor: rep.add_n(sess, &zs),
            fractional_precision,
            integral_precision,
        })
    }
}

pub trait FixedpointTensor {
    fn fractional_precision(&self) -> u32;
    fn integral_precision(&self) -> u32;
}

impl<RepRingT> FixedpointTensor for AbstractReplicatedFixedTensor<RepRingT> {
    fn fractional_precision(&self) -> u32 {
        self.fractional_precision
    }

    fn integral_precision(&self) -> u32 {
        self.integral_precision
    }
}

impl<RepRingT: Placed> FixedpointTensor for Symbolic<AbstractReplicatedFixedTensor<RepRingT>> {
    fn fractional_precision(&self) -> u32 {
        match self {
            Symbolic::Symbolic(_) => unimplemented!("ouch"), // TODO(Dragos) extract from underlying op signature
            Symbolic::Concrete(x) => x.fractional_precision,
        }
    }

    fn integral_precision(&self) -> u32 {
        match self {
            Symbolic::Symbolic(_) => unimplemented!("oof"), // TODO(Dragos) extract from underlying op signature
            Symbolic::Concrete(x) => x.integral_precision,
        }
    }
}

modelled_kernel! {
    PlacementPow2::pow2, Pow2Op,
    [
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::rep_rep_kernel),
    ]
}

impl Pow2Op {
    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementPow2<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.pow2(sess, &x);
        Ok(FixedTensor::Replicated(z))
    }
}

modelled_kernel! {
    PlacementExp::exp, ExpOp,
    [
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [transparent] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [transparent] Self::rep_rep_kernel),
        (ReplicatedPlacement, (crate::logical::Tensor) -> crate::logical::Tensor => [concrete] Self::logical_kernel),
    ]
}

impl ExpOp {
    fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementExp<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.exp(sess, &x);
        Ok(FixedTensor::Replicated(z))
    }
}

modelled!(PlacementSigmoid::sigmoid, ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor, SigmoidOp);
modelled!(PlacementSigmoid::sigmoid, ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor, SigmoidOp);
modelled!(PlacementSigmoid::sigmoid, ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, SigmoidOp);
modelled!(PlacementSigmoid::sigmoid, ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, SigmoidOp);
modelled!(PlacementSigmoid::sigmoid, ReplicatedPlacement, (crate::logical::Tensor) -> crate::logical::Tensor, SigmoidOp);

impl SigmoidOp {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementSigmoid<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.sigmoid(sess, &x);
        Ok(FixedTensor::Replicated(z))
    }
}

pub(crate) trait PrefixMul<S: Session, RepFixedT> {
    fn prefix_mul(&self, sess: &S, x: Vec<RepFixedT>) -> Vec<RepFixedT>;
}

impl<S: Session, RepFixedT> PrefixMul<S, RepFixedT> for ReplicatedPlacement
where
    RepFixedT: FixedpointTensor,
    ReplicatedPlacement: PlacementMul<S, RepFixedT, RepFixedT, RepFixedT>,
    ReplicatedPlacement: PlacementTruncPr<S, RepFixedT, RepFixedT>,
{
    fn prefix_mul(&self, sess: &S, x: Vec<RepFixedT>) -> Vec<RepFixedT> {
        let elementwise_mul =
            |rep: &ReplicatedPlacement, sess: &S, x: &RepFixedT, y: &RepFixedT| -> RepFixedT {
                assert_eq!(x.fractional_precision(), y.fractional_precision());
                rep.trunc_pr(sess, x.fractional_precision(), &rep.mul(sess, x, y))
            };

        self.prefix_op(sess, x, elementwise_mul)
    }
}

pub(crate) trait PolynomialEval<S: Session, RepFixedTensorT> {
    fn polynomial_eval(&self, sess: &S, coeffs: Vec<f64>, x: RepFixedTensorT) -> RepFixedTensorT;
}

impl<S: Session, RepFixedTensorT, MirFixedT> PolynomialEval<S, RepFixedTensorT>
    for ReplicatedPlacement
where
    RepFixedTensorT: FixedpointTensor,
    RepFixedTensorT: Clone,
    ReplicatedPlacement: PlacementMul<S, MirFixedT, RepFixedTensorT, RepFixedTensorT>,
    ReplicatedPlacement: PlacementTruncPr<S, RepFixedTensorT, RepFixedTensorT>,
    ReplicatedPlacement: PlacementAddN<S, RepFixedTensorT, RepFixedTensorT>,
    ReplicatedPlacement: PlacementAdd<S, RepFixedTensorT, MirFixedT, RepFixedTensorT>,
    ReplicatedPlacement: ShapeFill<S, RepFixedTensorT, Result = MirFixedT>,
    ReplicatedPlacement: PrefixMul<S, RepFixedTensorT>,
{
    fn polynomial_eval(&self, sess: &S, coeffs: Vec<f64>, x: RepFixedTensorT) -> RepFixedTensorT {
        assert!(!coeffs.is_empty());
        let mut degree = coeffs.len() - 1;

        // Exclude coefficients under precision
        for coeff in coeffs.iter().rev() {
            if *coeff < 2f64.powi(-(x.fractional_precision() as i32 + 1)) as f64 {
                degree -= 1
            } else {
                break;
            }
        }

        let coeffs_mir: Vec<_> = coeffs[0..degree + 1]
            .iter()
            .map(|coeff| {
                self.shape_fill(
                    sess,
                    coeff.as_fixedpoint(x.fractional_precision() as usize),
                    &x,
                )
            })
            .collect();

        let x_n: Vec<RepFixedTensorT> = (0..degree).map(|_| x.clone()).collect();

        let x_pre_mul = self.prefix_mul(sess, x_n);

        // TODO [Yann]
        // If x_pre_mul could be concatenated in one tensor, we could use a single
        // multiplication instead of doing a for loop.
        let x_mul_coeffs: Vec<RepFixedTensorT> = (0..x_pre_mul.len())
            .map(|i| self.mul(sess, &coeffs_mir[i + 1], &x_pre_mul[i]))
            .collect();

        let x_mul_coeffs_added = self.add_n(sess, &x_mul_coeffs);
        let x_mul_coeffs_added_fixed_trunc =
            self.trunc_pr(sess, x.fractional_precision(), &x_mul_coeffs_added);

        self.add(sess, &x_mul_coeffs_added_fixed_trunc, &coeffs_mir[0])
    }
}

modelled!(PlacementLessThan::less_than, ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor, LessThanOp);
modelled!(PlacementLessThan::less_than, ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor, LessThanOp);
modelled!(PlacementLessThan::less_than, ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedBitTensor, LessThanOp);
modelled!(PlacementLessThan::less_than, ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor, LessThanOp);
modelled!(PlacementLessThan::less_than, ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedBitTensor, LessThanOp);
modelled!(PlacementLessThan::less_than, ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor, LessThanOp);

impl LessThanOp {
    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementLessThan<S, RepRingT, RepRingT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.less_than(sess, &x.tensor, &y.tensor))
    }

    pub(crate) fn rep_mir_fixed_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractMirroredFixedTensor<MirRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementLessThan<S, MirRingT, RepRingT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.less_than(sess, &x.tensor, &y.tensor))
    }

    pub(crate) fn rep_fixed_mir_kernel<S: Session, RepRingT, MirroredT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractMirroredFixedTensor<MirroredT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementLessThan<S, RepRingT, MirroredT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.less_than(sess, &x.tensor, &y.tensor))
    }
}

modelled!(PlacementGreaterThan::greater_than, ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor, GreaterThanOp);
modelled!(PlacementGreaterThan::greater_than, ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor, GreaterThanOp);
modelled!(PlacementGreaterThan::greater_than, ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedBitTensor, GreaterThanOp);
modelled!(PlacementGreaterThan::greater_than, ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor, GreaterThanOp);
modelled!(PlacementGreaterThan::greater_than, ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedBitTensor, GreaterThanOp);
modelled!(PlacementGreaterThan::greater_than, ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor, GreaterThanOp);

impl GreaterThanOp {
    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementGreaterThan<S, RepRingT, RepRingT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.greater_than(sess, &x.tensor, &y.tensor))
    }

    pub(crate) fn rep_mir_fixed_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractMirroredFixedTensor<MirRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementGreaterThan<S, MirRingT, RepRingT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.greater_than(sess, &x.tensor, &y.tensor))
    }

    pub(crate) fn rep_fixed_mir_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractMirroredFixedTensor<MirRingT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementGreaterThan<S, RepRingT, MirRingT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.greater_than(sess, &x.tensor, &y.tensor))
    }
}

modelled!(PlacementFill::fill, ReplicatedPlacement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3Fixed64Tensor, FillOp);
modelled!(PlacementFill::fill, ReplicatedPlacement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3Fixed128Tensor, FillOp);

impl FillOp {
    pub(crate) fn mir_fixed_kernel<S: Session, MirRingT, ShapeT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        value: Constant,
        shape: ShapeT,
        fractional_precision: u32,
        integral_precision: u32,
    ) -> Result<AbstractMirroredFixedTensor<MirRingT>>
    where
        ReplicatedPlacement: PlacementFill<S, ShapeT, MirRingT>,
    {
        let filled = plc.fill(sess, value, &shape);
        Ok(AbstractMirroredFixedTensor {
            tensor: filled,
            integral_precision,
            fractional_precision,
        })
    }
}

modelled!(PlacementIfElse::if_else, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, IfElseOp);
modelled!(PlacementIfElse::if_else, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, IfElseOp);

impl IfElseOp {
    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        s: RepRingT,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementIfElse<S, RepRingT, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(AbstractReplicatedFixedTensor {
            tensor: plc.if_else(sess, &s, &x.tensor, &y.tensor),
            fractional_precision: x.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::SyncSession;
    use crate::replicated::AbstractReplicatedRingTensor;
    use crate::symbolic::{Symbolic, SymbolicHandle, SymbolicSession};
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
        let out = HostRing64Tensor::fixedpoint_mean(x, Some(0), encoding_factor).unwrap();
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
        let out = HostRing64Tensor::fixedpoint_mean(x, None, encoding_factor).unwrap();
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

    fn new_host_fixed_tensor_with_precision<HostRingT>(
        x: HostRingT,
        integral_precision: u32,
        fractional_precision: u32,
    ) -> AbstractHostFixedTensor<HostRingT> {
        AbstractHostFixedTensor {
            tensor: x,
            integral_precision,
            fractional_precision,
        }
    }

    fn new_replicated_fixed_tensor<RepRingT>(
        x: RepRingT,
    ) -> AbstractReplicatedFixedTensor<RepRingT> {
        AbstractReplicatedFixedTensor {
            tensor: x,
            fractional_precision: 15,
            integral_precision: 8,
        }
    }

    macro_rules! host_binary_approx_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $factor: expr, $error: expr) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<f64>,
                integral_precision: u32, fractional_precision: u32
            ) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let one_r: $tt = 1;

                let encode = |item: &$tt| (Wrapping(one_r << fractional_precision) * Wrapping(*item)).0;
                let xs = xs.clone().map(encode);
                let ys = ys.clone().map(encode);

                let x = FixedTensor::Host(new_host_fixed_tensor_with_precision(
                    AbstractHostRingTensor::from_raw_plc(xs.clone(), alice.clone()), integral_precision, fractional_precision)
                );
                let y = FixedTensor::Host(new_host_fixed_tensor_with_precision(
                    AbstractHostRingTensor::from_raw_plc(ys.clone(), alice.clone()), integral_precision, fractional_precision)
                );

                let sess = SyncSession::default();

                let sum = alice.$test_func(&sess, &x, &y);
                let opened_product = match sum {
                    FixedTensor::Host(r) => r,
                    _ => panic!("Should not produce a replicated tensor on a host placement"),
                };
                let r64 = Convert::decode(&opened_product.tensor, one_r << (opened_product.fractional_precision));
                let distance = squared_distance(&r64, &zs);

                let _: Vec<_> = distance.iter().enumerate().map(|(i, d)| {
                    assert!(*d < $error, "failed at index {:?} when dividing {:?} / {:?}, result is {:?}, should have been {:?}", i, xs[i], ys[i], r64.0[i], zs[i]);
                }).collect();

                let expected_precision = match x {
                    FixedTensor::Host(x) => x.fractional_precision * $factor,
                    _ => unreachable!(),
                };
                assert_eq!(opened_product.fractional_precision, expected_precision);
            }
        };
    }

    host_binary_approx_func_test!(test_host_div64, div<u64>, 1, 0.00000001);
    host_binary_approx_func_test!(test_host_div128, div<u128>, 1, 0.00000001);

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

    macro_rules! pairwise_bounded_same_length {
        ($func_name:ident, $tt: ident) => {
            fn $func_name(bit_room: usize) -> impl Strategy<Value = (ArrayD<$tt>, ArrayD<$tt>)> {
                (1usize..25)
                    .prop_flat_map(move |length: usize| {
                        let foo1 = bit_room;
                        let foo2 = bit_room;
                        (
                            proptest::collection::vec(
                                any::<$tt>().prop_map(move |x| x >> foo1),
                                length,
                            ),
                            proptest::collection::vec(
                                any::<$tt>().prop_map(move |x| {
                                    let y = x >> foo2;
                                    if y == 0 {
                                        1
                                    } else {
                                        y
                                    }
                                }),
                                length,
                            ),
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

    pairwise_bounded_same_length!(pairwise_bounded_same_length64, i64);
    pairwise_bounded_same_length!(pairwise_bounded_same_length128, i128);

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
        fn test_fuzzy_host_div64((a,b) in pairwise_bounded_same_length64(2 * 15))
        {
            let fractional_precision = 15;
            let integral_precision = 49;
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0f64; a.len()]).unwrap();
            for i in 0..a.len() {
                let d = (a[i] as f64) / (b[i] as f64);
                target[i] = d;
            }
            test_host_div64(a.map(|x| *x as u64), b.map(|x| *x as u64), target, integral_precision, fractional_precision);
        }

        #[test]
        fn test_fuzzy_host_div128((a,b) in pairwise_bounded_same_length64(2 * 15))
        {
            let fractional_precision = 15;
            let integral_precision = 49;
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0f64; a.len()]).unwrap();
            for i in 0..a.len() {
                let d = (a[i] as f64) / (b[i] as f64);
                target[i] = d;
            }
            test_host_div128(a.map(|x| *x as u128), b.map(|x| *x as u128), target, integral_precision, fractional_precision);
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

        #[test]
        fn test_fuzzy_fixed_rep_greater_than64((a,b) in pairwise_bounded_same_length64(10 + 1))
        {
            let mut target: Vec<u64> = vec![0_u64; a.len()];
            for i in 0..a.len() {
                target[i] = (a[i] > b[i]) as u64;
            }
            test_rep_greater_than64(a.map(|x| *x as f64), b.map(|x| *x as f64), target);
        }

        #[test]
        fn test_fuzzy_fixed_rep_greater_than128((a,b) in pairwise_bounded_same_length128(10 + 1))
        {
            let mut target: Vec<u128> = vec![0_u128; a.len()];
            for i in 0..a.len() {
                target[i] = (a[i] > b[i]) as u128;
            }
            test_rep_greater_than128(a.map(|x| *x as f64), b.map(|x| *x as f64), target);
        }
    }

    fn squared_distance(x: &HostFloat64Tensor, target: &ArrayD<f64>) -> ArrayD<f64> {
        assert_eq!(x.shape().0 .0, target.shape());
        let x = x.0.clone();
        let y = target.clone();
        (x.clone() - y.clone()) * (x - y)
    }

    macro_rules! rep_div_func_concrete_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $i_precision: expr, $f_precision: expr) => {
            fn $func_name(xs: ArrayD<f64>, ys: ArrayD<f64>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let encode = |item: &f64| (2_i64.pow($f_precision) as f64 * item) as $tt;

                let xs = xs.clone().map(encode);
                let ys = ys.clone().map(encode);
                let x = FixedTensor::Host(new_host_fixed_tensor_with_precision(
                    AbstractHostRingTensor::from_raw_plc(xs.clone(), alice.clone()), $i_precision, $f_precision)
                );
                let y = FixedTensor::Host(new_host_fixed_tensor_with_precision(
                    AbstractHostRingTensor::from_raw_plc(ys.clone(), alice.clone()), $i_precision, $f_precision)
                );

                let sess = SyncSession::default();

                let sum = rep.$test_func(&sess, &x, &y);
                let opened_product = match sum {
                    FixedTensor::Replicated(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an non-replicated tensor on a replicated placement"),
                };

                let mut expected_result = Array::from_shape_vec(IxDyn(&[xs.clone().len()]), vec![0 as f64; xs.clone().len()]).unwrap();
                for i in 0..xs.len() {
                    expected_result[i] = (xs[i] as f64) / (ys[i] as f64);
                }
                let result = Convert::decode(&opened_product.tensor, (2 as $tt).pow($f_precision));
                let distance = squared_distance(&result, &expected_result);
                let error: f64 = (1_f64) / ((2 as $tt).pow($f_precision) as f64);

                let _: Vec<_> = distance.iter().enumerate().map(|(i, d)| {
                    assert!(*d < error, "failed at index {:?} when dividing {:?} / {:?}, result is {:?}, should have been {:?}", i, xs[i], ys[i], result.0[i], expected_result[i]);
                }).collect();


            }
        };
    }

    #[test]
    fn test_fixed_rep_div64() {
        let a: Vec<f64> = vec![1.0, 2.0, 2.0];
        let b: Vec<f64> = vec![3.0, 7.0, 1.41];
        let a = Array::from_shape_vec(IxDyn(&[a.len()]), a).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[b.len()]), b).unwrap();

        rep_div_func_concrete_test!(test_rep_div64, div<u64>, 10, 15);
        test_rep_div64(a, b);
    }

    #[test]
    fn test_fixed_rep_div128() {
        let a: Vec<f64> = vec![1.0, 2.0, 2.0];
        let b: Vec<f64> = vec![3.0, 7.0, 1.41];
        let a = Array::from_shape_vec(IxDyn(&[a.len()]), a).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[b.len()]), b).unwrap();

        rep_div_func_concrete_test!(test_rep_div128, div<u128>, 10, 15);
        test_rep_div128(a, b);
    }

    macro_rules! new_symbolic_replicated_tensor {
        ($func_name:ident, $tt: ty) => {
            fn $func_name(
                name: &str,
                rep: &ReplicatedPlacement,
            ) -> Symbolic<AbstractReplicatedRingTensor<Symbolic<AbstractHostRingTensor<$tt>>>> {
                let (alice, bob, carole) = rep.host_placements();
                let symbolic_replicated = Symbolic::Concrete(AbstractReplicatedRingTensor {
                    shares: [
                        [
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"00"),
                                plc: alice.clone(),
                            }),
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"01"),
                                plc: alice.clone(),
                            }),
                        ],
                        [
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"10"),
                                plc: bob.clone(),
                            }),
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"11"),
                                plc: bob.clone(),
                            }),
                        ],
                        [
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"20"),
                                plc: carole.clone(),
                            }),
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"21"),
                                plc: carole.clone(),
                            }),
                        ],
                    ],
                });
                symbolic_replicated
            }
        };
    }

    new_symbolic_replicated_tensor!(new_symbolic_replicated_tensor64, u64);
    new_symbolic_replicated_tensor!(new_symbolic_replicated_tensor128, u128);

    macro_rules! rep_div_symbolic_test {
        ($func_name:ident, $new_symbolic_rep: ident) => {
            fn $func_name(i_precision: u32, f_precision: u32) {
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = Symbolic::Concrete(AbstractReplicatedFixedTensor {
                    fractional_precision: f_precision,
                    integral_precision: i_precision,
                    tensor: $new_symbolic_rep(&"x", &rep),
                });

                let y = Symbolic::Concrete(AbstractReplicatedFixedTensor {
                    fractional_precision: f_precision,
                    integral_precision: i_precision,
                    tensor: $new_symbolic_rep(&"y", &rep),
                });

                let sess = SymbolicSession::default();

                let result = rep.div(&sess, &x, &y);
                match result {
                    Symbolic::Concrete(AbstractReplicatedFixedTensor {
                        tensor: _,
                        fractional_precision,
                        integral_precision,
                    }) => {
                        assert_eq!(fractional_precision, f_precision);
                        assert_eq!(integral_precision, i_precision);
                    }
                    _ => {
                        panic!("Expected a concrete result from the symbolic division on a concrete value")
                    }
                }
            }
        }
    }

    rep_div_symbolic_test!(rep_div_symbolic_test64, new_symbolic_replicated_tensor64);
    rep_div_symbolic_test!(rep_div_symbolic_test128, new_symbolic_replicated_tensor128);

    #[test]
    fn test_fixed_rep_symbolic_div64() {
        rep_div_symbolic_test64(10, 20);
    }

    #[test]
    fn test_fixed_rep_symbolic_div128() {
        rep_div_symbolic_test128(10, 50);
    }

    macro_rules! rep_prefix_op_fixed_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $f_precision: expr) => {
            fn $func_name(x: Vec<ArrayD<$tt>>, y_target: Vec<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let sess = SyncSession::default();

                let encode = |item: &$tt| (2_i64.pow($f_precision) as $tt * item) as $tt;

                let x_fixed_vec = x
                    .into_iter()
                    .map(|x| {
                        let x_encode = x.map(encode);
                        let x_ring = AbstractHostRingTensor::from_raw_plc(x_encode, alice.clone());
                        let x_shared: AbstractReplicatedRingTensor<AbstractHostRingTensor<$tt>> =
                            rep.share(&sess, &x_ring);
                        new_replicated_fixed_tensor(x_shared)
                    })
                    .collect();

                let outputs = rep.prefix_mul(&sess, x_fixed_vec);

                for (i, output) in outputs.iter().enumerate() {
                    let output_reveal = alice.reveal(&sess, output);
                    let result =
                        Convert::decode(&output_reveal.tensor, (2 as $tt).pow($f_precision));
                    assert_eq!(result.0.as_slice().unwrap()[0] as $tt, y_target[i]);
                }
            }
        };
    }

    rep_prefix_op_fixed_test!(test_rep_prefix_mul_fixed64, prefix_mul_fixed<u64>, 15);
    rep_prefix_op_fixed_test!(test_rep_prefix_mul_fixed128, prefix_mul_fixed<u128>, 15);

    #[test]
    fn test_rep_prefix_mul_fixed_64() {
        let x = vec![
            array![1u64].into_dyn(),
            array![2u64].into_dyn(),
            array![3u64].into_dyn(),
            array![4u64].into_dyn(),
        ];
        let y_target = vec![1, 2, 6, 24];

        test_rep_prefix_mul_fixed64(x, y_target);
    }

    #[test]
    fn test_rep_prefix_mul_fixed_128() {
        let x = vec![
            array![1u128].into_dyn(),
            array![2u128].into_dyn(),
            array![3u128].into_dyn(),
            array![4u128].into_dyn(),
        ];
        let y_target = vec![1u128, 2, 6, 24];

        test_rep_prefix_mul_fixed128(x, y_target);
    }

    macro_rules! rep_poly_eval_fixed_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $f_precision: expr) => {
            fn $func_name(x: ArrayD<f64>, coeffs: Vec<f64>, y_target: Vec<f64>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let sess = SyncSession::default();

                let encode = |item: &f64| (2_i64.pow($f_precision) as f64 * item) as $tt;
                let x_encoded = x.map(encode);
                let x_ring = AbstractHostRingTensor::from_raw_plc(x_encoded, alice.clone());
                let x_shared: AbstractReplicatedRingTensor<AbstractHostRingTensor<$tt>> =
                    rep.share(&sess, &x_ring);
                let x_fixed_shared = new_replicated_fixed_tensor(x_shared.clone());

                let output = rep.polynomial_eval(&sess, coeffs, x_fixed_shared);
                let output_reveal = alice.reveal(&sess, &output);
                let result = Convert::decode(&output_reveal.tensor, (2 as $tt).pow($f_precision));

                for i in 0..y_target.len() {
                    let error = (result.0[i] - y_target[i]).abs();
                    assert!(error < f64::EPSILON);
                }
            }
        };
    }

    rep_poly_eval_fixed_test!(test_rep_poly_eval_fixed64, poly_eval<u64>, 15);
    rep_poly_eval_fixed_test!(test_rep_poly_eval_fixed128, poly_eval<u128>, 15);

    #[test]
    fn test_rep_poly_eval_64() {
        let x = array![1f64, 2., 3., 4.].into_dyn();
        let coeffs = vec![1f64, 2., 3.];
        let y_targets = vec![6f64, 17., 34., 57.];

        test_rep_poly_eval_fixed64(x, coeffs, y_targets);
    }

    #[test]
    fn test_rep_poly_eval_128() {
        let x = array![1f64, 2., 3., 4.].into_dyn();
        let coeffs = vec![1f64, 2., 3.];
        let y_targets = vec![6f64, 17., 34., 57.];

        test_rep_poly_eval_fixed128(x, coeffs, y_targets);
    }

    #[test]
    fn test_host_shape_op() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let x = AbstractHostRingTensor::from_raw_plc(
            array![1024u64, 5, 4]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            alice,
        );

        let shape = x.shape();
        let raw_shape: RawShape = shape.0;
        let underlying = vec![3];
        let expected: RawShape = RawShape(underlying);
        assert_eq!(expected, raw_shape);
    }

    macro_rules! rep_approx_unary_fixed_test {
        ($func_name:ident, $test_func: ident<$ti: ty, $tu: ty>, $i_precision: expr, $f_precision: expr, $err: expr) => {
            fn $func_name(x: ArrayD<f64>, y_target: Vec<f64>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let sess = SyncSession::default();
                let encode = |item: &f64| -> $tu {
                    let tmp: $ti = (2f64.powf($f_precision as f64) * item) as $ti;
                    tmp as $tu
                };
                let x_encoded = x.map(encode);

                let x = FixedTensor::Host(new_host_fixed_tensor_with_precision(
                    AbstractHostRingTensor::from_raw_plc(x_encoded.clone(), alice.clone()), $i_precision, $f_precision)
                );

                let exp_result = rep.$test_func(&sess, &x);

                let opened_exp = match exp_result {
                    FixedTensor::Replicated(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an non-replicated tensor on a replicated placement"),
                };

                let result = Convert::decode(&opened_exp.tensor, (2 as $tu).pow($f_precision));

                // operation precision is not as accurate as the fixed point precision
                for i in 0..y_target.len() {
                    let error = (result.0[i] - y_target[i]).abs();
                    assert!(error < $err, "failed at index {:?}, error is {:?}", i, error);
                }
            }
        };
    }

    rep_approx_unary_fixed_test!(test_rep_pow2_fixed64, pow2<i64, u64>, 10, 10, 0.001);
    rep_approx_unary_fixed_test!(test_rep_pow2_fixed128, pow2<i128, u128>, 30, 10, 0.001);

    rep_approx_unary_fixed_test!(test_rep_exp_fixed64, exp<i64, u64>, 10, 10, 0.1);
    rep_approx_unary_fixed_test!(test_rep_exp_fixed128, exp<i128, u128>, 20, 20, 0.001);

    #[test]
    fn test_exp2_64() {
        let x = array![1f64, 2.5, -3.0, 4.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| 2_f64.powf(*item)).collect();
        test_rep_pow2_fixed64(x, y_targets);
    }

    #[test]
    fn test_exp2_128() {
        let x = array![1f64, 2.5, -3.0, 4.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| 2_f64.powf(*item)).collect();
        test_rep_pow2_fixed128(x, y_targets);
    }

    #[test]
    fn test_exponential_64() {
        let x = array![1f64, 2.5, -3.0, 4.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| item.exp()).collect();
        test_rep_exp_fixed64(x, y_targets);
    }

    #[test]
    fn test_exponential_128() {
        let x = array![1f64, 2.5, -3.0, 4.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| item.exp()).collect();
        test_rep_exp_fixed128(x, y_targets);
    }

    rep_approx_unary_fixed_test!(test_rep_sigmoid_fixed64, sigmoid<i64, u64>, 10, 10, 0.1);
    rep_approx_unary_fixed_test!(test_rep_sigmoid_fixed128, sigmoid<i128, u128>, 20, 20, 0.001);

    #[test]
    fn test_sigmoid_64() {
        let x = array![1f64, 2.5, -3.0, 4.0, 40.0, -30.0, -4.0, -6.0, 6.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| 1.0 / (1.0 + (-item).exp())).collect();
        test_rep_sigmoid_fixed64(x, y_targets);
    }

    #[test]
    fn test_sigmoid_128() {
        let x = array![1f64, 2.5, -3.0, 4.0, -4.0, -6.0, 6.0, 1024.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| 1.0 / (1.0 + (-item).exp())).collect();
        test_rep_sigmoid_fixed128(x, y_targets);
    }

    macro_rules! rep_unary_symbolic_test {
        ($func_name:ident, $test_func:ident, $new_symbolic_rep: ident) => {
            fn $func_name(i_precision: u32, f_precision: u32) {
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = Symbolic::Concrete(AbstractReplicatedFixedTensor {
                    fractional_precision: f_precision,
                    integral_precision: i_precision,
                    tensor: $new_symbolic_rep(&"x", &rep),
                });

                let sess = SymbolicSession::default();

                let result = rep.$test_func(&sess, &x);
                match result {
                    Symbolic::Concrete(AbstractReplicatedFixedTensor {
                        tensor: _,
                        fractional_precision,
                        integral_precision,
                    }) => {
                        assert_eq!(fractional_precision, f_precision);
                        assert_eq!(integral_precision, i_precision);
                    }
                    _ => {
                        panic!("Expected a concrete result from the symbolic unary op on a concrete value")
                    }
                }
            }
        }
    }

    rep_unary_symbolic_test!(
        rep_exp_symbolic_test64,
        exp,
        new_symbolic_replicated_tensor64
    );

    #[test]
    fn test_fixed_rep_symbolic_exp64() {
        rep_exp_symbolic_test64(10, 10);
    }

    macro_rules! rep_signed_binary_func_test {
        ($func_name:ident, $test_func: ident<$ti: ty, $tu: ty>, $i_precision: expr, $f_precision: expr) => {
            fn $func_name(x: ArrayD<f64>, y: ArrayD<f64>, target: Vec<$tu>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let sess = SyncSession::default();
                let encode = |item: &f64| -> $tu {
                    let tmp: $ti = (2f64.powf($f_precision as f64) * item) as $ti;
                    tmp as $tu
                };
                let x_encoded = x.map(encode);
                let y_encoded = y.map(encode);

                let xf = new_host_fixed_tensor_with_precision(
                    AbstractHostRingTensor::from_raw_plc(x_encoded.clone(), alice.clone()),
                    $i_precision,
                    $f_precision,
                );

                let yf = new_host_fixed_tensor_with_precision(
                    AbstractHostRingTensor::from_raw_plc(y_encoded.clone(), alice.clone()),
                    $i_precision,
                    $f_precision,
                );

                let xs = rep.share(&sess, &xf);
                let ys = rep.share(&sess, &yf);

                let zs: ReplicatedBitTensor = rep.$test_func(&sess, &xs, &ys);
                let z = alice.reveal(&sess, &zs);

                for i in 0..target.len() {
                    assert_eq!(
                        target[i] as $tu, z.0[i] as $tu,
                        "failed comparing {:?} with {:?}",
                        x[i], y[i]
                    );
                }
            }
        };
    }

    rep_signed_binary_func_test!(test_rep_greater_than64, greater_than<i64, u64>, 10, 10);
    rep_signed_binary_func_test!(test_rep_greater_than128, greater_than<i128, u128>, 10, 10);

    rep_signed_binary_func_test!(test_rep_less_than64, less_than<i64, u64>, 10, 10);
    rep_signed_binary_func_test!(test_rep_less_than128, less_than<i128, u128>, 20, 20);

    #[test]
    fn test_fixed_rep_greater_than64() {
        let x = array![0f64, 2.7, -2.9, 4.1].into_dyn();
        let y = array![1f64, 2.5, -3.0, 4.0].into_dyn();
        let targets: Vec<u64> = vec![0_u64, 1, 1, 1];
        test_rep_greater_than64(x, y, targets);
    }

    #[test]
    fn test_fixed_rep_greater_than128() {
        let x = array![0f64, 2.7, -2.9, 4.1, -3.555].into_dyn();
        let y = array![1f64, 2.5, -3.0, 4.0, -3.354].into_dyn();
        let targets: Vec<u128> = vec![0_u128, 1, 1, 1, 0];
        test_rep_greater_than128(x, y, targets);
    }

    #[test]
    fn test_fixed_rep_less_than64() {
        let x = array![0f64, 2.7, -2.9, 4.1, -3.555].into_dyn();
        let y = array![1f64, 2.5, -3.0, 4.0, -3.354].into_dyn();
        let targets: Vec<u64> = vec![1_u64, 0, 0, 0, 1];
        test_rep_less_than64(x, y, targets);
    }

    #[test]
    fn test_fixed_rep_less_than128() {
        let x = array![0f64, 2.7, -2.9, 4.1, -3.555].into_dyn();
        let y = array![1f64, 2.5, -3.0, 4.0, -3.354].into_dyn();
        let targets: Vec<u128> = vec![1_u128, 0, 0, 0, 1];
        test_rep_less_than128(x, y, targets);
    }

    macro_rules! rep_index_axis_fixed_test {
        ($func_name:ident, $ti: ty, $tu: ty,$axis: expr, $index: expr, $i_precision: expr, $f_precision: expr) => {
            fn $func_name(x: ArrayD<f64>, y_target: ArrayD<f64>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let sess = SyncSession::default();
                let encode = |item: &f64| -> $tu {
                    let tmp: $ti = (2f64.powf($f_precision as f64) * item) as $ti;
                    tmp as $tu
                };
                let x_encoded = x.map(encode);

                let x = FixedTensor::Host(new_host_fixed_tensor_with_precision(
                    AbstractHostRingTensor::from_raw_plc(x_encoded.clone(), alice.clone()), $i_precision, $f_precision)
                );

                let exp_result = rep.index_axis(&sess, $axis, $index, &x);

                let opened_exp = match exp_result {
                    FixedTensor::Replicated(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an non-replicated tensor on a replicated placement"),
                };

                let result = Convert::decode(&opened_exp.tensor, (2 as $tu).pow($f_precision));
                assert_eq!(result.0, y_target);
            }
        };
    }

    rep_index_axis_fixed_test!(test_rep_index_axis_fixed64, i64, u64, 1, 0, 10, 10);
    rep_index_axis_fixed_test!(test_rep_index_axis_fixed128, i128, u128, 1, 0, 20, 20);

    #[test]
    fn test_index_axis_64() {
        let x = array![[1f64, 2.5], [-3.0, 4.0]].into_dyn();
        let y_targets = x.index_axis(Axis(1), 0).into_dyn().to_owned();
        test_rep_index_axis_fixed64(x, y_targets);
    }

    #[test]
    fn test_index_axis_128() {
        let x = array![[1f64, 2.5], [-3.0, 4.0]].into_dyn();
        let y_targets = x.index_axis(Axis(1), 0).into_dyn().to_owned();
        test_rep_index_axis_fixed128(x, y_targets);
    }
}
