use crate::computation::{
    FixedpointAddOp, FixedpointDotOp, FixedpointEncodeOp, FixedpointMulOp, FixedpointRingMeanOp,
    FixedpointSubOp, HostPlacement, KnownType, Placed, Placement, ReplicatedPlacement,
};
use crate::error::Result;
use crate::kernels::{
    PlacementAdd, PlacementDot, PlacementDotSetup, PlacementFixedpointEncode, PlacementMul,
    PlacementMulSetup, PlacementPlace, PlacementReveal, PlacementRingMean, PlacementShareSetup,
    PlacementSub, PlacementTruncPr, RuntimeSession, Session,
};
use crate::replicated::{Replicated128Tensor, Replicated64Tensor};
use crate::ring::{AbstractRingTensor, Ring128Tensor, Ring64Tensor};
use crate::standard::Float64Tensor;
use macros::with_context;
use ndarray::prelude::*;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::num::Wrapping;
use std::ops::Mul;

pub type Fixed64Tensor = FixedTensor<Ring64Tensor, Replicated64Tensor>;

pub type Fixed128Tensor = FixedTensor<Ring128Tensor, Replicated128Tensor>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FixedTensor<RingTensorT, ReplicatedTensorT> {
    RingTensor(RingTensorT),
    ReplicatedTensor(ReplicatedTensorT),
}

impl<RingTensorT, ReplicatedTensorT> Placed for FixedTensor<RingTensorT, ReplicatedTensorT>
where
    RingTensorT: Placed,
    RingTensorT::Placement: Into<Placement>,
    ReplicatedTensorT: Placed,
    ReplicatedTensorT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            FixedTensor::RingTensor(x) => Ok(x.placement()?.into()),
            FixedTensor::ReplicatedTensor(x) => Ok(x.placement()?.into()),
        }
    }
}

pub trait Convert<T> {
    type Scale: One + Clone;
    fn encode(x: &T, scaling_factor: Self::Scale) -> Self;
    fn decode(x: &Self, scaling_factor: Self::Scale) -> T;
}

impl Convert<Float64Tensor> for Ring64Tensor {
    type Scale = u64;
    fn encode(x: &Float64Tensor, scaling_factor: Self::Scale) -> Ring64Tensor {
        let x_upshifted = &x.0 * (scaling_factor as f64);
        let x_converted: ArrayD<u64> = x_upshifted.mapv(|el| (el as i64) as u64);
        Ring64Tensor::from(x_converted)
    }
    fn decode(x: &Self, scaling_factor: Self::Scale) -> Float64Tensor {
        let x_upshifted: ArrayD<i64> = x.into();
        let x_converted = x_upshifted.mapv(|el| el as f64);
        Float64Tensor::from(x_converted / scaling_factor as f64)
    }
}

impl Convert<Float64Tensor> for Ring128Tensor {
    type Scale = u128;
    fn encode(x: &Float64Tensor, scaling_factor: Self::Scale) -> Ring128Tensor {
        let x_upshifted = &x.0 * (scaling_factor as f64);
        let x_converted: ArrayD<u128> = x_upshifted.mapv(|el| (el as i128) as u128);
        Ring128Tensor::from(x_converted)
    }

    fn decode(x: &Self, scaling_factor: Self::Scale) -> Float64Tensor {
        let x_upshifted: ArrayD<i128> = x.into();
        let x_converted = x_upshifted.mapv(|el| el as f64);
        Float64Tensor::from(x_converted / scaling_factor as f64)
    }
}

impl<T> AbstractRingTensor<T>
where
    Wrapping<T>: Clone + Zero + Mul<Wrapping<T>, Output = Wrapping<T>>,
    AbstractRingTensor<T>: Convert<Float64Tensor>,
{
    pub fn ring_mean(
        x: Self,
        axis: Option<usize>,
        scaling_factor: <AbstractRingTensor<T> as Convert<Float64Tensor>>::Scale,
    ) -> AbstractRingTensor<T> {
        let mean_weight = Self::compute_mean_weight(&x, &axis);
        let encoded_weight = AbstractRingTensor::<T>::encode(&mean_weight, scaling_factor);
        let operand_sum = x.sum(axis);
        operand_sum.mul(encoded_weight)
    }

    fn compute_mean_weight(x: &Self, &axis: &Option<usize>) -> Float64Tensor {
        let shape: &[usize] = x.0.shape();
        if let Some(ax) = axis {
            let dim_len = shape[ax] as f64;
            Float64Tensor::from(
                Array::from_elem([], 1.0 / dim_len)
                    .into_dimensionality::<IxDyn>()
                    .unwrap(),
            )
        } else {
            let dim_prod: usize = std::iter::Product::product(shape.iter());
            let prod_inv = 1.0 / dim_prod as f64;
            Float64Tensor::from(
                Array::from_elem([], prod_inv)
                    .into_dimensionality::<IxDyn>()
                    .unwrap(),
            )
        }
    }
}

modelled!(PlacementRingMean::ring_mean, HostPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (Ring64Tensor) -> Ring64Tensor, FixedpointRingMeanOp);
modelled!(PlacementRingMean::ring_mean, HostPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (Ring128Tensor) -> Ring128Tensor, FixedpointRingMeanOp);

kernel! {
    FixedpointRingMeanOp,
    [
        (HostPlacement, (Ring64Tensor) -> Ring64Tensor => attributes[axis, scaling_base, scaling_exp] Self::kernel_ring64tensor),
        (HostPlacement, (Ring128Tensor) -> Ring128Tensor => attributes[axis, scaling_base, scaling_exp] Self::kernel_ring128tensor),
    ]
}

impl FixedpointRingMeanOp {
    fn kernel_ring64tensor<S: RuntimeSession>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: Ring64Tensor,
    ) -> Ring64Tensor
    where
        HostPlacement: PlacementPlace<S, Ring64Tensor>,
    {
        let scaling_factor = u64::pow(scaling_base, scaling_exp);
        let axis = axis.map(|a| a as usize);
        let mean = Ring64Tensor::ring_mean(x, axis, scaling_factor);
        plc.place(sess, mean)
    }

    fn kernel_ring128tensor<S: RuntimeSession>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: Ring128Tensor,
    ) -> Ring128Tensor
    where
        HostPlacement: PlacementPlace<S, Ring128Tensor>,
    {
        let scaling_factor = u128::pow(scaling_base as u128, scaling_exp);
        let axis = axis.map(|a| a as usize);
        let mean = Ring128Tensor::ring_mean(x, axis, scaling_factor);
        plc.place(sess, mean)
    }
}

modelled!(PlacementFixedpointEncode::fixedpoint_encode, HostPlacement, attributes[precision: u32] (Float64Tensor) -> Fixed128Tensor, FixedpointEncodeOp);

kernel! {
    FixedpointEncodeOp,
    [
        (HostPlacement, (Float64Tensor) -> Fixed128Tensor => attributes[precision] Self::kernel),
    ]
}

impl FixedpointEncodeOp {
    fn kernel<S: RuntimeSession>(
        _sess: &S,
        _plc: &HostPlacement,
        _precision: u32,
        _x: Float64Tensor,
    ) -> Fixed128Tensor {
        todo!("Implement runtime kernel for FixedpointEncodeOp")
    }
}

modelled!(PlacementAdd::add, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointAddOp);
modelled!(PlacementAdd::add, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointAddOp);

hybrid_kernel! {
    FixedpointAddOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::rep_kernel),
    ]
}

impl FixedpointAddOp {
    fn host_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<RingT, RepT>,
        y: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        HostPlacement: PlacementReveal<S, RepT, RingT>,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
    {
        let x = match x {
            FixedTensor::RingTensor(v) => v,
            FixedTensor::ReplicatedTensor(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::RingTensor(v) => v,
            FixedTensor::ReplicatedTensor(v) => plc.reveal(sess, &v),
        };

        let result = with_context!(plc, sess, x + y);
        FixedTensor::RingTensor(result)
    }

    fn rep_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<RingT, RepT>,
        y: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, RingT, RepT>,
        ReplicatedPlacement: PlacementAdd<S, RepT, RepT, RepT>,
    {
        let setup = sess.replicated_setup(plc);

        let x = match x {
            FixedTensor::RingTensor(v) => plc.share(sess, setup, &v),
            FixedTensor::ReplicatedTensor(v) => v,
        };
        let y = match y {
            FixedTensor::RingTensor(v) => plc.share(sess, setup, &v),
            FixedTensor::ReplicatedTensor(v) => v,
        };

        let result = with_context!(plc, sess, x + y);
        FixedTensor::ReplicatedTensor(result)
    }
}

modelled!(PlacementSub::sub, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointSubOp);

hybrid_kernel! {
    FixedpointSubOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::rep_kernel),
    ]
}

impl FixedpointSubOp {
    fn host_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<RingT, RepT>,
        y: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        HostPlacement: PlacementReveal<S, RepT, RingT>,
        HostPlacement: PlacementSub<S, RingT, RingT, RingT>,
    {
        let x = match x {
            FixedTensor::RingTensor(v) => v,
            FixedTensor::ReplicatedTensor(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::RingTensor(v) => v,
            FixedTensor::ReplicatedTensor(v) => plc.reveal(sess, &v),
        };

        let result = with_context!(plc, sess, x - y);
        FixedTensor::RingTensor(result)
    }

    fn rep_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<RingT, RepT>,
        y: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, RingT, RepT>,
        ReplicatedPlacement: PlacementSub<S, RepT, RepT, RepT>,
    {
        let setup = sess.replicated_setup(plc);

        let x = match x {
            FixedTensor::RingTensor(v) => plc.share(sess, setup, &v),
            FixedTensor::ReplicatedTensor(v) => v,
        };
        let y = match y {
            FixedTensor::RingTensor(v) => plc.share(sess, setup, &v),
            FixedTensor::ReplicatedTensor(v) => v,
        };

        let result = with_context!(plc, sess, x - y);
        FixedTensor::ReplicatedTensor(result)
    }
}

modelled!(PlacementMul::mul, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointMulOp);
modelled!(PlacementMul::mul, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointMulOp);

hybrid_kernel! {
    FixedpointMulOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::rep_kernel),
    ]
}

impl FixedpointMulOp {
    fn host_kernel<S: Session, RingT, RepT>(
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
            FixedTensor::RingTensor(v) => v,
            FixedTensor::ReplicatedTensor(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::RingTensor(v) => v,
            FixedTensor::ReplicatedTensor(v) => plc.reveal(sess, &v),
        };

        let result = with_context!(plc, sess, x * y);
        FixedTensor::RingTensor(result)
    }

    fn rep_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<RingT, RepT>,
        y: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, RingT, RepT>,
        ReplicatedPlacement: PlacementMulSetup<S, S::ReplicatedSetup, RepT, RepT, RepT>,
        ReplicatedPlacement: PlacementTruncPr<S, RepT, RepT>,
    {
        let setup = sess.replicated_setup(plc);

        let x = match x {
            FixedTensor::RingTensor(v) => plc.share(sess, setup, &v),
            FixedTensor::ReplicatedTensor(v) => v,
        };
        let y = match y {
            FixedTensor::RingTensor(v) => plc.share(sess, setup, &v),
            FixedTensor::ReplicatedTensor(v) => v,
        };

        let result = with_context!(plc, sess, mul_setup(setup, &x, &y));
        // TODO(lvorona): get the `amount` for the TruncPr from the FixedTensor
        let truncated = plc.trunc_pr(sess, 27, &result);
        FixedTensor::ReplicatedTensor(truncated)
    }
}
modelled!(PlacementDot::dot, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedpointDotOp);
modelled!(PlacementDot::dot, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedpointDotOp);

hybrid_kernel! {
    FixedpointDotOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::rep_kernel),
    ]
}

impl FixedpointDotOp {
    fn host_kernel<S: Session, RingT, RepT>(
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
            FixedTensor::RingTensor(x) => x,
            FixedTensor::ReplicatedTensor(x) => plc.reveal(sess, &x),
        };
        let y_revealed = match y {
            FixedTensor::RingTensor(x) => x,
            FixedTensor::ReplicatedTensor(x) => plc.reveal(sess, &x),
        };

        let result = plc.dot(sess, &x_revealed, &y_revealed);
        FixedTensor::RingTensor(result)
    }

    fn rep_kernel<S: Session, RingT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<RingT, RepT>,
        y: FixedTensor<RingT, RepT>,
    ) -> FixedTensor<RingT, RepT>
    where
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, RingT, RepT>,
        ReplicatedPlacement: PlacementDotSetup<S, S::ReplicatedSetup, RepT, RepT, RepT>,
        ReplicatedPlacement: PlacementTruncPr<S, RepT, RepT>,
    {
        let setup = sess.replicated_setup(plc);

        let x_shared = match x {
            FixedTensor::RingTensor(x) => plc.share(sess, setup, &x),
            FixedTensor::ReplicatedTensor(x) => x,
        };
        let y_shared = match y {
            FixedTensor::RingTensor(x) => plc.share(sess, setup, &x),
            FixedTensor::ReplicatedTensor(x) => x,
        };

        let result = plc.dot_setup(sess, setup, &x_shared, &y_shared);
        // TODO(lvorona): get the `amount` for the TruncPr from the FixedTensor
        let truncated = plc.trunc_pr(sess, 27, &result);

        FixedTensor::ReplicatedTensor(truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_fixedpoint() {
        let x = Float64Tensor::from(
            array![1.0, -2.0, 3.0, -4.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );

        let scaling_factor = 2u64.pow(16);
        let x_encoded = Ring64Tensor::encode(&x, scaling_factor);
        assert_eq!(
            x_encoded,
            Ring64Tensor::from(vec![
                65536,
                18446744073709420544,
                196608,
                18446744073709289472
            ])
        );

        let x_decoded = Ring64Tensor::decode(&x_encoded, scaling_factor);
        assert_eq!(x_decoded, x);

        let scaling_factor_long = 2u128.pow(80);
        let x_encoded = Ring128Tensor::encode(&x, scaling_factor_long);
        assert_eq!(
            x_encoded,
            Ring128Tensor::from(vec![
                1208925819614629174706176,
                340282366920936045611735378173418799104,
                3626777458843887524118528,
                340282366920933627760096148915069386752
            ])
        );

        let x_decoded_long = Ring128Tensor::decode(&x_encoded, scaling_factor_long);
        assert_eq!(x_decoded_long, x);
    }

    #[test]
    fn ring_mean_with_axis() {
        let x_backing: Float64Tensor = Float64Tensor::from(
            array![[1., 2.], [3., 4.]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let encoding_factor = 2u64.pow(16);
        let decoding_factor = 2u64.pow(32);
        let x = Ring64Tensor::encode(&x_backing, encoding_factor);
        let out = Ring64Tensor::ring_mean(x, Some(0), encoding_factor);
        let dec = Ring64Tensor::decode(&out, decoding_factor);
        assert_eq!(
            dec,
            Float64Tensor::from(array![2., 3.].into_dimensionality::<IxDyn>().unwrap())
        );
    }

    #[test]
    fn ring_mean_no_axis() {
        let x_backing: Float64Tensor = Float64Tensor::from(
            array![[1., 2.], [3., 4.]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let encoding_factor = 2u64.pow(16);
        let decoding_factor = 2u64.pow(32);
        let x = Ring64Tensor::encode(&x_backing, encoding_factor);
        let out = Ring64Tensor::ring_mean(x, None, encoding_factor);
        let dec = Ring64Tensor::decode(&out, decoding_factor);
        assert_eq!(
            dec.0.into_shape((1,)).unwrap(),
            array![2.5].into_shape((1,)).unwrap()
        );
    }
}
