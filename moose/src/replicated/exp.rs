use crate::fixedpoint::{FixedpointTensor, PolynomialEval};
use num::bigint::BigInt;
use num::rational::Ratio;
use num_traits::{FromPrimitive, One, ToPrimitive};

use super::*;
use lazy_static::lazy_static;

impl Pow2Op {
    pub(crate) fn rep_rep_kernel<
        S: Session,
        RepRingT,
        RepBitT,
        RepBitArrayT,
        RepShapeT,
        N: Const,
    >(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        RepRingT: Ring<BitLength = N>,
        RepRingT: Clone,
        ReplicatedPlacement: PlacementBitDec<S, RepRingT, RepBitArrayT>,
        ReplicatedPlacement: PlacementIndex<S, RepBitArrayT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
        ReplicatedPlacement: PlacementIfElse<S, RepRingT, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementNeg<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShape<S, RepRingT, RepShapeT>,
        ReplicatedPlacement: PlacementFill<S, RepShapeT, RepRingT>,
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
        ReplicatedPlacement: Pow2FromBits<S, RepRingT>,
        ReplicatedPlacement: ExpFromParts<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementDiv<
            S,
            AbstractReplicatedFixedTensor<RepRingT>,
            AbstractReplicatedFixedTensor<RepRingT>,
            AbstractReplicatedFixedTensor<RepRingT>,
        >,
    {
        let integral_precision = x.integral_precision as usize;
        let fractional_precision = x.fractional_precision as usize;
        let bit_len_prec = fractional_precision + integral_precision;

        let x_bits = rep.bit_decompose(sess, &x.tensor);

        let msb_bit = rep.index(sess, RepRingT::BitLength::VALUE - 1, &x_bits);
        let msb = rep.ring_inject(sess, 0, &msb_bit);

        let abs_x = rep.if_else(sess, &msb, &rep.neg(sess, &x.tensor), &x.tensor);

        let absolute_bits = rep.bit_decompose(sess, &abs_x);
        let x_bits_vec: Vec<_> = (0..RepRingT::BitLength::VALUE)
            .map(|i| rep.index(sess, i, &absolute_bits))
            .collect();

        // higher is the integral part of the exponent
        let higher_bits: Vec<_> = (fractional_precision..RepRingT::BitLength::VALUE)
            .map(|i| rep.ring_inject(sess, 0, &x_bits_vec[i]))
            .collect();
        let x_shape = rep.shape(sess, &x.tensor);
        let zero = rep.fill(sess, 0_u8.into(), &x_shape);
        let higher_composed = higher_bits
            .clone()
            .into_iter()
            .enumerate()
            .fold(zero, |acc, (i, item)| {
                rep.add(sess, &acc, &rep.shl(sess, fractional_precision + i, &item))
            });

        // fractional part of the exponent
        let fractional_part = rep.sub(sess, &abs_x, &higher_composed);

        // computes 2^{integral_part}
        let d = rep.pow2_from_bits(sess, &higher_bits[0..integral_precision]);

        // computes the 2^x from 2^{int(x)} and frac(x)
        let g = rep.exp_from_parts(
            sess,
            &d,
            &fractional_part,
            fractional_precision as u32,
            bit_len_prec as u32,
        );

        // if exponent is negative than compute the inverse of the result
        // since 2^-x = 1/2^x.
        let one = rep.fill(sess, 1_u8.into(), &x_shape);
        let one_fixed = AbstractReplicatedFixedTensor {
            tensor: rep.shl(sess, x.fractional_precision as usize, &one),
            integral_precision: x.integral_precision,
            fractional_precision: x.fractional_precision,
        };

        let g_fixed = AbstractReplicatedFixedTensor {
            tensor: g.clone(),
            integral_precision: x.integral_precision,
            fractional_precision: x.fractional_precision,
        };

        // compute 1/2^x
        let inverse = rep.div(sess, &one_fixed, &g_fixed);

        // oblivious branching depending on the exponent sign, choose 1/2^x or 2^x
        let switch = rep.if_else(sess, &msb, &inverse.tensor, &g);

        Ok(AbstractReplicatedFixedTensor {
            tensor: switch,
            integral_precision: x.integral_precision,
            fractional_precision: x.fractional_precision,
        })
    }
}

/// Computes 2^x given the bit representation of x: [b(0)]...[b(k-1)].
///
/// This is done by computing the product of p(i) = b(i) * 2^i + (1 - b(i)).
///
/// One can see that b(i) acts as selector here, i.e. if b(i) = 1 then p(i) = 2^i, o/w p(i) = 1.
///
/// The product of all p(i) yields [2^x] since 2^x = prod(2^b(i)) where b(i) = 1.
pub(crate) trait Pow2FromBits<S: Session, RepRingT> {
    fn pow2_from_bits(&self, sess: &S, x: &[RepRingT]) -> RepRingT;
}

impl<S: Session, RepRingT> Pow2FromBits<S, RepRingT> for ReplicatedPlacement
where
    ReplicatedShape: KnownType<S>,
    ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementFill<S, cs!(ReplicatedShape), RepRingT>,
    ReplicatedPlacement: PlacementShape<S, RepRingT, cs!(ReplicatedShape)>,
    ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
{
    fn pow2_from_bits(&self, sess: &S, x: &[RepRingT]) -> RepRingT {
        let rep = self;

        let ones = rep.fill(sess, 1_u8.into(), &rep.shape(sess, &x[0]));

        let selectors: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(i, bit)| {
                // compute b(i) << 2^i
                // (Dragos) aware that this can overflow but can't do much against
                // (unless we support larger rings than 128 bits)
                let pos = rep.shl(sess, 1 << i, bit);
                // compute 1 - b(i)
                let neg = rep.sub(sess, &ones, bit);
                // compute p(i) = b(i) << 2^i + (1 - b(i))
                rep.add(sess, &pos, &neg)
            })
            .collect();

        // TODO(Dragos) do tree multiplication here
        selectors.iter().fold(ones, |acc, y| rep.mul(sess, &acc, y))
    }
}

lazy_static! {
    static ref P_1045: Vec<f64> = (0..100).map(|coefficient_index| {
        // p_1405[i] = math.log(2) ** i / math.factorial(i)
        let ln2i: Ratio<BigInt> = Ratio::from_float(2_f64.ln().powf(coefficient_index as f64)).unwrap();
        let fact = (1..coefficient_index + 1).fold(One::one(), |acc: BigInt, i| {
            let i_th: BigInt = FromPrimitive::from_usize(i).unwrap();
            acc * i_th
        });
        let coefficient_value: Ratio<BigInt> = ln2i / fact;
        coefficient_value.to_f64().unwrap()
    }).collect();
}

pub(crate) trait ExpFromParts<S: Session, T, O> {
    fn exp_from_parts(&self, sess: &S, e_int: &T, e_frac: &T, f: u32, k: u32) -> O;
}

impl<S: Session, RepRingT> ExpFromParts<S, RepRingT, RepRingT> for ReplicatedPlacement
where
    ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    RepRingT: Clone,

    AbstractReplicatedFixedTensor<RepRingT>: CanonicalType,
    <AbstractReplicatedFixedTensor<RepRingT> as CanonicalType>::Type: KnownType<S>,

    m!(c!(AbstractReplicatedFixedTensor<RepRingT>)):
        TryInto<AbstractReplicatedFixedTensor<RepRingT>>,
    AbstractReplicatedFixedTensor<RepRingT>:
        Into<m!(c!(AbstractReplicatedFixedTensor<RepRingT>))>,

    // TODO(Morten) Good chance we can remove macros here after complete switch to modelled_kernel
    ReplicatedPlacement: PolynomialEval<S, m!(c!(AbstractReplicatedFixedTensor<RepRingT>))>,

    ReplicatedPlacement: PlacementTruncPr<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
{
    fn exp_from_parts(
        &self,
        sess: &S,
        e_int: &RepRingT,
        e_frac: &RepRingT,
        f: u32,
        k: u32,
    ) -> RepRingT {
        let amount = k - 2 - f;
        let x = AbstractReplicatedFixedTensor {
            tensor: self.shl(sess, amount as usize, e_frac),
            integral_precision: 2,
            fractional_precision: k - 2,
        };
        let e_approx = self.polynomial_eval(sess, P_1045.to_vec(), x.into());

        // convert replicated fixed tensor to concrete value in order to grab the replicated ring tensor
        let e_approx_f: AbstractReplicatedFixedTensor<RepRingT> = e_approx.try_into().ok().unwrap();
        let e_approx_ring = e_approx_f.tensor;

        // do replicated multiplication at the ring level
        let e_prod = self.mul(sess, e_int, &e_approx_ring);

        // truncate the result but keep the most significant f bits to preserve fixed point encoding
        self.trunc_pr(sess, amount, &e_prod)
    }
}

impl ExpOp {
    pub(crate) fn rep_rep_kernel<S: Session, RepFixedT, MirFixedT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepFixedT,
    ) -> Result<RepFixedT>
    where
        RepFixedT: FixedpointTensor,
        ReplicatedPlacement: ShapeFill<S, RepFixedT, Result = MirFixedT>,
        ReplicatedPlacement: PlacementMul<S, MirFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementTruncPr<S, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementPow2<S, RepFixedT, RepFixedT>,
    {
        let log2e = rep.shape_fill(
            sess,
            1.0_f64
                .exp()
                .log2()
                .as_fixedpoint(x.fractional_precision() as usize),
            &x,
        );
        let shifted_exponent = rep.mul(sess, &log2e, &x);
        let exponent = rep.trunc_pr(sess, x.fractional_precision(), &shifted_exponent);
        Ok(rep.pow2(sess, &exponent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::AbstractHostRingTensor;
    use crate::kernels::SyncSession;
    use ndarray::array;

    #[test]
    fn test_pow2_from_bits() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x = AbstractHostRingTensor::from_raw_plc(
            array![[0u64], [1], [1], [1]].into_dyn(),
            alice.clone(),
        );
        let target = AbstractHostRingTensor::from_raw_plc(array![16384u64], alice.clone());

        let sess = SyncSession::default();

        let x_shared = rep.share(&sess, &x);
        let x0 = rep.index_axis(&sess, 0, 0, &x_shared);
        let x1 = rep.index_axis(&sess, 0, 1, &x_shared);
        let x2 = rep.index_axis(&sess, 0, 2, &x_shared);
        let x3 = rep.index_axis(&sess, 0, 3, &x_shared);

        let x_vec = vec![x0, x1, x2, x3];
        // compute 2^(x0 * 2^0 + x1 * 2^1 + x2 * 2^2 + x3 * 2^3)
        let pow2_shared = rep.pow2_from_bits(&sess, &x_vec);

        assert_eq!(target, alice.reveal(&sess, &pow2_shared));
    }
}
