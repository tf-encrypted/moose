//! Abstraction layer for fixed-point values.

use crate::computation::*;
use crate::error::Result;
#[cfg(feature = "compile")]
use crate::execution::symbolic::Symbolic;
use crate::execution::Session;
use crate::kernels::*;
use crate::replicated::*;
use serde::{Deserialize, Serialize};

mod ops;

/// Abstract fixed-point tensor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[cfg(feature = "compile")]
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
