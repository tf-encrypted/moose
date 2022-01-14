//! Abstraction layer for fixed-point values

use crate::computation::*;
use crate::error::{Error, Result};
use crate::host::*;
use crate::kernels::*;
use crate::replicated::*;
use crate::symbolic::Symbolic;
use crate::types::*;
use ndarray::prelude::*;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::num::Wrapping;
use std::ops::Mul;

mod ops;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FixedTensor<HostFixedT, MirFixedT, RepFixedT> {
    Host(HostFixedT),
    Mirrored3(MirFixedT),
    Replicated(RepFixedT),
}

impl<HostFixedT, MirFixedT, RepFixedT> Placed for FixedTensor<HostFixedT, MirFixedT, RepFixedT>
where
    HostFixedT: Placed,
    HostFixedT::Placement: Into<Placement>,
    MirFixedT: Placed,
    MirFixedT::Placement: Into<Placement>,
    RepFixedT: Placed,
    RepFixedT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            FixedTensor::Host(x) => Ok(x.placement()?.into()),
            FixedTensor::Mirrored3(x) => Ok(x.placement()?.into()),
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

pub trait FixedpointTensor {
    fn fractional_precision(&self) -> u32;
    fn integral_precision(&self) -> u32;
}

impl<RepRingT> FixedpointTensor for RepFixedTensor<RepRingT> {
    fn fractional_precision(&self) -> u32 {
        self.fractional_precision
    }

    fn integral_precision(&self) -> u32 {
        self.integral_precision
    }
}

impl<RepRingT: Placed> FixedpointTensor for Symbolic<RepFixedTensor<RepRingT>> {
    fn fractional_precision(&self) -> u32 {
        match self {
            Symbolic::Symbolic(_) => unimplemented!(), // TODO(Dragos) extract from underlying op signature
            Symbolic::Concrete(x) => x.fractional_precision,
        }
    }

    fn integral_precision(&self) -> u32 {
        match self {
            Symbolic::Symbolic(_) => unimplemented!(), // TODO(Dragos) extract from underlying op signature
            Symbolic::Concrete(x) => x.integral_precision,
        }
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
