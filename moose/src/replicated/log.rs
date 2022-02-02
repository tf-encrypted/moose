use super::*;
use crate::computation::RepEqualOp;
use crate::error::Result;
use crate::execution::Session;
use crate::fixedpoint::PolynomialEval;
use crate::{Const, Ring};
use lazy_static::lazy_static;

lazy_static! {
    static ref P_2524: Vec<f64> = vec![-2.05466671951, -8.8626599391, 6.10585199015, 4.81147460989];
    static ref Q_2524: Vec<f64> = vec![0.353553425277, 4.54517087629, 6.42784209029, 1.0];
}

impl RepEqualOp {
    pub(crate) fn rep_kernel<S: Session, RepRingT, RepBitT, RepBitArrayT, ShapeT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepBitT>
    where
        RepRingT: Ring<BitLength = N>,

        ReplicatedPlacement: PlacementBitDec<S, RepRingT, RepBitArrayT>,
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepBitT>,
        ReplicatedPlacement: PlacementShape<S, RepRingT, ShapeT>,
        ReplicatedPlacement: PlacementIndex<S, RepBitArrayT, RepBitT>,
        ReplicatedPlacement: PlacementMul<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
    {
        let z = rep.sub(sess, &x, &y);
        let bits = rep.bit_decompose(sess, &z);

        let v: Vec<_> = (0..RepRingT::BitLength::VALUE)
            .map(|i| rep.index(sess, i, &bits))
            .collect();

        let ones = rep.fill(sess, 1u8.into(), &rep.shape(sess, &z));

        let v_not: Vec<_> = v.iter().map(|vi| rep.xor(sess, &ones, vi)).collect();

        // TODO we can optimize this by having a binary multipler like
        // we are doing with the binary adder in bit decompitision
        Ok(v_not.iter().fold(ones, |acc, y| rep.mul(sess, &acc, y)))
    }

    pub(crate) fn rep_ring_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepRingT>
    where
        ReplicatedPlacement: PlacementEqual<S, RepRingT, RepRingT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
    {
        let b = rep.equal(sess, &x, &y);
        Ok(rep.ring_inject(sess, 0, &b))
    }
}

impl Log2Op {
    pub(crate) fn rep_rep_kernel<S: Session, RepRingT, HostRingT>( // RepFixedT
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        RepFixedTensor<RepRingT>: CanonicalType,
        <RepFixedTensor<RepRingT> as CanonicalType>::Type: KnownType<S>,

        m!(c!(RepFixedTensor<RepRingT>)): TryInto<RepFixedTensor<RepRingT>>,
        m!(c!(RepFixedTensor<RepRingT>)): Clone,

        RepFixedTensor<RepRingT>: Into<m!(c!(RepFixedTensor<RepRingT>))>,

        ReplicatedPlacement: Int2FL<S, RepRingT>,
        ReplicatedPlacement: PolynomialEval<S, m!(c!(RepFixedTensor<RepRingT>))>,
        ReplicatedPlacement: PlacementDiv<
            S,
            m!(c!(RepFixedTensor<RepRingT>)),
            m!(c!(RepFixedTensor<RepRingT>)),
            m!(c!(RepFixedTensor<RepRingT>)),
        >,
        ReplicatedPlacement: PlacementAdd<
            S,
            m!(c!(RepFixedTensor<RepRingT>)),
            m!(c!(RepFixedTensor<RepRingT>)),
            m!(c!(RepFixedTensor<RepRingT>)),
        >,

        HostPlacement: PlacementReveal<S, RepRingT, HostRingT>,
        HostRingT: std::fmt::Debug,
    {

        let (player0, player1, player2) = rep.host_placements();

        let total_precision = x.fractional_precision + x.integral_precision;

        let (v, p, _s, _z) = rep.int2fl(
            sess,
            &x.tensor,
            total_precision as usize,
            x.fractional_precision as usize,
        );

        let v_fixed = RepFixedTensor {
            tensor: v,
            integral_precision: x.integral_precision,
            fractional_precision: x.fractional_precision,
        }
        .into();

        println!("v: {:?}", player0.reveal(sess, &v));
        println!("p: {:?}", player0.reveal(sess, &p));

        let p2524 = rep.polynomial_eval(sess, P_2524.to_vec(), v_fixed.clone());
        let q2524 = rep.polynomial_eval(sess, Q_2524.to_vec(), v_fixed);

        let quotient = rep.div(sess, &p2524, &q2524);
        let p_fixed = RepFixedTensor {
            tensor: p,
            integral_precision: x.integral_precision,
            fractional_precision: x.fractional_precision,
        }
        .into();

        let result = with_context!(rep, sess, p_fixed + quotient);
        Ok(result.try_into().ok().unwrap())
    }
}

/// Internal trait for converting int to float number generation
/// Computes v, p, s, z such that (1 − 2s) · (1 − z) · v · 2 p = x.
pub(crate) trait Int2FL<S: Session, RepRingT> {
    fn int2fl(
        &self,
        sess: &S,
        x: &RepRingT,
        k: usize,
        f: usize,
    ) -> (RepRingT, RepRingT, RepRingT, RepRingT);
}

impl<S: Session, RepRingT, MirRingT, N: Const> Int2FL<S, RepRingT> for ReplicatedPlacement
where
    RepRingT: Clone + Ring<BitLength = N>,
    RepBitArray<ReplicatedBitTensor, N>: KnownType<S>,
    HostRing64Tensor: KnownType<S>,

    ReplicatedBitTensor: KnownType<S>,

    ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementMsb<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementEqual<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementMux<S, RepRingT, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementNeg<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementBitDec<S, RepRingT, cs!(RepBitArray<ReplicatedBitTensor, N>)>,
    ReplicatedPlacement:
        PlacementIndex<S, cs!(RepBitArray<ReplicatedBitTensor, N>), cs!(ReplicatedBitTensor)>,

    ReplicatedPlacement: PlacementRingInject<S, cs!(ReplicatedBitTensor), RepRingT>,
    ReplicatedPlacement: PlacementNeg<S, cs!(ReplicatedBitTensor), cs!(ReplicatedBitTensor)>,
    ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: ShapeFill<S, RepRingT, Result = MirRingT>,
    ReplicatedPlacement: PlacementSub<S, MirRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementSub<S, RepRingT, MirRingT, RepRingT>,
    ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementAdd<S, MirRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementTruncPr<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,

    // this has to dissapear after prefixor is a trait
    ReplicatedPlacement: PlacementAnd<
        S,
        cs!(ReplicatedBitTensor),
        cs!(ReplicatedBitTensor),
        cs!(ReplicatedBitTensor),
    >,
    ReplicatedPlacement: PlacementXor<
        S,
        cs!(ReplicatedBitTensor),
        cs!(ReplicatedBitTensor),
        cs!(ReplicatedBitTensor),
    >,
{
    fn int2fl(
        &self,
        sess: &S,
        x: &RepRingT,
        k: usize,
        f: usize,
    ) -> (RepRingT, RepRingT, RepRingT, RepRingT) {
        let rep = self;

        let lambda = k - 1;

        // hack because fill op doesn't work without adding too many un-necessary type constrains
        let zero = rep.sub(sess, x, x);

        // TODO(Dragos) this can be optimized by performing a single bit-decomposition and use it
        // to compute the msb and equality to zero
        let sign = rep.msb(sess, x);
        let is_zero = rep.equal(sess, x, &zero);

        // x positive
        let x_pos = rep.mux(sess, &sign, &rep.neg(sess, x), x);
        let x_pos_bits = rep.bit_decompose(sess, &x_pos);
        let x_pos_bits_rev: Vec<_> = (0..lambda)
            .map(|i| rep.index(sess, i, &x_pos_bits))
            .rev()
            .collect();

        // from msb(x) every bit will be set to 1
        let b = rep.prefix_or(sess, x_pos_bits_rev);
        let b_ring: Vec<_> = (0..lambda)
            .map(|i| rep.ring_inject(sess, 0, &b[i]))
            .collect();

        let ones = rep.shape_fill(sess, 1_u8, x);
        // the following computes bit_compose(1 - bi), basically the amount that x needs to be
        // scaled up so that msb_index(x_upshifted) = lam-1
        let neg_b_sum = (0..lambda).fold(zero.clone(), |acc, i| {
            let neg = rep.sub(sess, &ones, &b_ring[i]);
            rep.add(sess, &acc, &rep.shl(sess, i, &neg))
        });

        // add 1 to multiply with a power of 2 and do a bit shift by log(neg_b_sum)
        let x_upshifted = with_context!(rep, sess, x_pos * (ones + neg_b_sum));

        // we truncate x_upshifted but be sure to leave out f bits
        // we have k-1 bits in total because the input x has signed k bits,
        let x_norm = rep.trunc_pr(sess, (k - 1 - f) as u32, &x_upshifted);

        let bit_count = (0..lambda).fold(zero, |acc, i| rep.add(sess, &acc, &b_ring[i]));
        let ften = rep.shape_fill(sess, f as u8, &bit_count);

        // we need to exclude f bits since we consider the number to be scaled by 2^f
        let p_res = with_context!(rep, sess, (bit_count - ften) * (ones - is_zero));

        (x_norm, p_res, sign, is_zero)
    }
}

#[cfg(test)]
mod tests {
    use crate::execution::SyncSession;
    use crate::host::{FromRaw, FromRawScaled, HostPlacement};
    use crate::kernels::*;
    use crate::replicated::log::Int2FL;
    use crate::replicated::{ReplicatedBitTensor, ReplicatedPlacement};
    use crate::types::{HostBitTensor, HostRing64Tensor, HostFixed128Tensor};
    use ndarray::array;

    #[test]
    fn test_equal() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };

        let bob = HostPlacement {
            owner: "bob".into(),
        };

        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let sess = SyncSession::default();

        let x: HostRing64Tensor = alice.from_raw(array![1024u64, 5, 4]);
        let y: HostRing64Tensor = bob.from_raw(array![1024u64, 4, 5]);

        let x_shared = rep.share(&sess, &x);
        let y_shared = rep.share(&sess, &y);

        let res: ReplicatedBitTensor = rep.equal(&sess, &x_shared, &y_shared);
        let expected: HostBitTensor = alice.from_raw(array![1, 0, 0]);

        let opened_result = alice.reveal(&sess, &res);
        assert_eq!(opened_result, expected);
    }

    #[test]
    fn test_int2fl() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };

        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let sess = SyncSession::default();

        let x: HostRing64Tensor = alice.from_raw(array![2u64, 10, 32, 0, -4_i64 as u64]);
        let x_shared = rep.share(&sess, &x);

        let (v, p, s, z) = rep.int2fl(&sess, &x_shared, 32, 5);
        let (vc, pc, sc, zc) = (
            alice.reveal(&sess, &v),
            alice.reveal(&sess, &p),
            alice.reveal(&sess, &s),
            alice.reveal(&sess, &z),
        );

        let twop = pc.0.mapv(|x| 2_f64.powf(x.0 as i64 as f64));
        let z_neg = zc.0.mapv(|x| 1_f64 - (x.0 as f64));
        let sign = sc.0.mapv(|x| 1_f64 - 2_f64 * (x.0 as f64));
        let vc_i64 = vc.0.mapv(|x| x.0 as f64);

        // (1-2s) * (1-z) * v * 2^p == x
        let expected =
            (sign * z_neg * vc_i64 * twop).mapv(|x| std::num::Wrapping((x as i64) as u64));

        assert_eq!(expected, x.0);
    }

    #[test]
    fn test_log2() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };

        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let sess = SyncSession::default();

        let x: HostFixed128Tensor = alice.from_raw_scaled(array![2.0, 4.0, 8.0, 4.5, 10.5], 10, 10);

        println!("encoded x: {:?}", x);
        let x_shared = rep.share(&sess, &x);

        println!("x_revealed: {:?}", alice.reveal(&sess, &x_shared));
        let log2x = rep.log2(&sess, &x_shared);
        println!("{:?}", alice.reveal(&sess, &log2x));
    }
}
