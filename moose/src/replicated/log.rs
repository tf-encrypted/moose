use super::*;
use crate::error::Result;
use crate::execution::Session;
use crate::fixedpoint::{FixedpointTensor, PolynomialEval};
use crate::{Const, Ring};
use lazy_static::lazy_static;
use std::convert::TryInto;

lazy_static! {
    static ref P_2524: Vec<f64> = vec![-2.05466671951, -8.8626599391, 6.10585199015, 4.81147460989];
    static ref Q_2524: Vec<f64> = vec![0.353553425277, 4.54517087629, 6.42784209029, 1.0];
}

impl Log2Op {
    pub(crate) fn rep_rep_kernel<S: Session, RepRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        RepFixedTensor<RepRingT>: Clone,

        RepFixedTensor<RepRingT>: CanonicalType,
        <RepFixedTensor<RepRingT> as CanonicalType>::Type: KnownType<S>,
        m!(c!(RepFixedTensor<RepRingT>)): From<RepFixedTensor<RepRingT>>,
        m!(c!(RepFixedTensor<RepRingT>)): TryInto<RepFixedTensor<RepRingT>>,

        ReplicatedPlacement: Int2FL<S, RepRingT>,
        ReplicatedPlacement: PolynomialEval<S, m!(c!(RepFixedTensor<RepRingT>))>,
        ReplicatedPlacement: PlacementDiv<
            S,
            RepFixedTensor<RepRingT>,
            RepFixedTensor<RepRingT>,
            RepFixedTensor<RepRingT>,
        >,
        ReplicatedPlacement: PlacementAdd<
            S,
            RepFixedTensor<RepRingT>,
            RepFixedTensor<RepRingT>,
            RepFixedTensor<RepRingT>,
        >,
        ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    {
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
        };

        // TODO(Morten) hopefully these will clean up nicely after making PolynomialEval concrete
        let p2524 = rep
            .polynomial_eval(sess, P_2524.to_vec(), v_fixed.clone().into())
            .try_into()
            .ok()
            .unwrap();
        let q2524 = rep
            .polynomial_eval(sess, Q_2524.to_vec(), v_fixed.into())
            .try_into()
            .ok()
            .unwrap();

        let quotient = rep.div(sess, &p2524, &q2524);
        let p_fixed = RepFixedTensor {
            tensor: rep.shl(sess, x.fractional_precision as usize, &p),
            integral_precision: x.integral_precision,
            fractional_precision: x.fractional_precision,
        };

        Ok(with_context!(rep, sess, p_fixed + quotient))
    }
}

impl LogOp {
    pub(crate) fn rep_rep_kernel<S: Session, RepFixedT, MirFixedT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepFixedT,
    ) -> Result<RepFixedT>
    where
        RepFixedT: FixedpointTensor,
        ReplicatedPlacement: ShapeFill<S, RepFixedT, Result = MirFixedT>,
        ReplicatedPlacement: PlacementLog2<S, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementMul<S, MirFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementTruncPr<S, RepFixedT, RepFixedT>,
    {
        let ln2 = rep.shape_fill(
            sess,
            2.0_f64
                .ln()
                .as_fixedpoint(x.fractional_precision() as usize),
            &x,
        );

        let log2 = rep.log2(sess, &x);
        let result = rep.mul(sess, &ln2, &log2);
        Ok(rep.trunc_pr(sess, x.fractional_precision(), &result))
    }
}

/// Internal trait for converting int to a floating point number
/// Computes v, p, s, z such that (1 − 2s) · (1 − z) · v · 2 p = x.
/// See for more details https://eprint.iacr.org/2012/405 and https://eprint.iacr.org/2019/354
pub(crate) trait Int2FL<S: Session, RepRingT> {
    fn int2fl(
        &self,
        sess: &S,
        x: &RepRingT,
        max_bit_len: usize,
        fractional_precision: usize,
    ) -> (RepRingT, RepRingT, RepRingT, RepRingT);
}

impl<S: Session, RepRingT, MirRingT, N: Const> Int2FL<S, RepRingT> for ReplicatedPlacement
where
    RepRingT: Clone + Ring<BitLength = N>,
    RepBitArray<ReplicatedBitTensor, N>: KnownType<S>,
    HostRing64Tensor: KnownType<S>,

    ReplicatedBitTensor: KnownType<S>,
    ReplicatedShape: KnownType<S>,

    ReplicatedPlacement: PlacementMsb<S, m!(RepBitArray<ReplicatedBitTensor, N>), RepRingT>,
    ReplicatedPlacement: PlacementEqualZero<S, m!(RepBitArray<ReplicatedBitTensor, N>), RepRingT>,
    ReplicatedPlacement:
        PlacementBitDecompose<S, RepRingT, m!(RepBitArray<ReplicatedBitTensor, N>)>,

    ReplicatedPlacement: PlacementMux<S, RepRingT, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementNeg<S, RepRingT, RepRingT>,
    ReplicatedPlacement:
        PlacementIndex<S, m!(RepBitArray<ReplicatedBitTensor, N>), m!(ReplicatedBitTensor)>,

    ReplicatedPlacement: PlacementRingInject<S, m!(ReplicatedBitTensor), RepRingT>,
    ReplicatedPlacement: PlacementNeg<S, m!(ReplicatedBitTensor), m!(ReplicatedBitTensor)>,
    ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: ShapeFill<S, RepRingT, Result = MirRingT>,
    ReplicatedPlacement: PlacementSub<S, MirRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementSub<S, RepRingT, MirRingT, RepRingT>,
    ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementAdd<S, MirRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementTruncPr<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,

    ReplicatedPlacement: PlacementFill<S, m!(ReplicatedShape), RepRingT>,
    ReplicatedPlacement: PlacementShape<S, RepRingT, m!(ReplicatedShape)>,

    // these are due to prefixor implementation
    ReplicatedPlacement:
        PlacementAnd<S, m!(ReplicatedBitTensor), m!(ReplicatedBitTensor), m!(ReplicatedBitTensor)>,
    ReplicatedPlacement:
        PlacementXor<S, m!(ReplicatedBitTensor), m!(ReplicatedBitTensor), m!(ReplicatedBitTensor)>,
{
    fn int2fl(
        &self,
        sess: &S,
        x: &RepRingT,
        max_bit_len: usize,
        fractional_precision: usize,
    ) -> (RepRingT, RepRingT, RepRingT, RepRingT) {
        let rep = self;

        let lambda = max_bit_len - 1;

        let x_bits = rep.bit_decompose(sess, &x.clone());
        let sign = rep.msb(sess, &x_bits);
        let is_zero = rep.equal_zero(sess, &x_bits);

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
        let zero = rep.fill(sess, 0_u8.into(), &rep.shape(sess, x));

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
        let x_norm = rep.trunc_pr(
            sess,
            (max_bit_len - 1 - fractional_precision) as u32,
            &x_upshifted,
        );

        let bit_count = (0..lambda).fold(zero, |acc, i| rep.add(sess, &acc, &b_ring[i]));
        let ften = rep.shape_fill(sess, fractional_precision as u8, &bit_count);

        // we need to exclude f bits since we consider the number to be scaled by 2^f
        let p_res = with_context!(rep, sess, (bit_count - ften) * (ones - is_zero));

        (x_norm, p_res, sign, is_zero)
    }
}

#[cfg(test)]
mod tests {
    use crate::fixedpoint::FixedTensor;
    use crate::host::{Convert, FromRaw};
    use crate::kernels::*;
    use crate::prelude::*;
    use crate::replicated::log::Int2FL;
    use ndarray::array;
    use ndarray::prelude::*;

    macro_rules! rep_approx_log_fixed_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $scalar_float_ty: ident, $tensor_float_ty:ident, $i_precision: expr, $f_precision: expr, $err: expr) => {
            fn $func_name(x: ArrayD<$scalar_float_ty>, y_target: Vec<$scalar_float_ty>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let y: $tensor_float_ty = alice.from_raw(x);
                let x = alice.fixedpoint_encode(&sess, $f_precision, $i_precision, &y);
                let x = FixedTensor::Host(x);

                let log_result = rep.$test_func(&sess, &x);

                let opened_log = match log_result {
                    FixedTensor::Replicated(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an non-replicated tensor on a replicated placement"),
                };

                let result = Convert::decode(&opened_log.tensor, (2 as $tt).pow($f_precision));
                let result: Vec<_> = result.0.mapv(|item| item as $scalar_float_ty).iter().copied().collect();
                // operation precision is not as accurate as the fixed point precision
                for i in 0..y_target.len() {
                    let error = (result[i] - y_target[i]).abs();
                    assert!(error < $err, "failed comparing {:?} against {:?}, error is {:?}", result[i], y_target[i], error);
                }
            }
        };
    }

    rep_approx_log_fixed_test!(
        test_rep_log2_fixed64,
        log2<u64>,
        f32,
        HostFloat32Tensor,
        8,
        20,
        0.01
    );
    rep_approx_log_fixed_test!(
        test_rep_ln_fixed64,
        log<u64>,
        f32,
        HostFloat32Tensor,
        8,
        20,
        0.01
    );
    rep_approx_log_fixed_test!(
        test_rep_log2_fixed128,
        log2<u128>,
        f64,
        HostFloat64Tensor,
        10,
        30,
        0.01
    );
    rep_approx_log_fixed_test!(
        test_rep_ln_fixed128,
        log<u128>,
        f64,
        HostFloat64Tensor,
        10,
        30,
        0.001
    );

    #[test]
    fn test_int2fl() {
        let alice = HostPlacement::from("alice");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

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
    fn test_log2_64() {
        let x = array![1.0_f32, 2.0, 4.0, 8.0, 4.5, 10.5].into_dyn();
        let expected = x.mapv(|item| item.log2()).iter().copied().collect();
        test_rep_log2_fixed64(x, expected);

        let x = array![[
            [1.0_f32, 2.0],
            [4.0, 23.3124],
            [42.954, 4.5],
            [10.5, 13.4219]
        ]]
        .into_dyn();
        let expected = x.mapv(|item| item.log2()).iter().copied().collect();
        test_rep_log2_fixed64(x, expected);

        let x = array![
            [1.0_f32, 2.0],
            [4.0, 23.3124],
            [42.954, 4.5],
            [10.5, 13.4219]
        ]
        .into_dyn();
        let expected = x.mapv(|item| item.log2()).iter().copied().collect();
        test_rep_log2_fixed64(x, expected);
    }

    #[test]
    fn test_log2_128() {
        let x = array![1.0_f64, 2.0, 4.0, 8.0, 4.5, 10.5].into_dyn();
        let expected = x.mapv(|item| item.log2()).iter().copied().collect();
        test_rep_log2_fixed128(x, expected);

        let x = array![[
            [1.0_f64, 2.0],
            [4.0, 23.3124],
            [42.954, 4.5],
            [10.5, 13.42190]
        ]]
        .into_dyn();
        let expected = x.mapv(|item| item.log2()).iter().copied().collect();
        test_rep_log2_fixed128(x, expected);

        let x = array![
            [1.0_f64, 2.0],
            [4.0, 23.3124],
            [42.954, 4.5],
            [10.5, 13.42190]
        ]
        .into_dyn();
        let expected = x.mapv(|item| item.log2()).iter().copied().collect();
        test_rep_log2_fixed128(x, expected);
    }

    #[test]
    fn test_ln64() {
        let x = array![1.0_f32, 2.5, 3.0, 4.0, 5.0].into_dyn();
        let expected = x.mapv(|item| item.ln()).iter().copied().collect();
        test_rep_ln_fixed64(x, expected);

        let x = array![
            [1.0_f32, 2.5, 3.0, 4.0, 5.0],
            [1.33, 4.123, 13.432, 10.33, 55.33]
        ]
        .into_dyn();
        let expected = x.mapv(|item| item.ln()).iter().copied().collect();
        test_rep_ln_fixed64(x, expected);

        let x = array![[
            [1.0_f32, 127.0],
            [10.3121, 123.025],
            [15.3213, 65.323],
            [126.9599, 74.98876]
        ]]
        .into_dyn();
        let expected = x.mapv(|item| item.ln()).iter().copied().collect();
        test_rep_ln_fixed64(x, expected);
    }

    #[test]
    fn test_ln128() {
        let x = array![1.0_f64, 2.5, 3.0, 4.0, 5.0].into_dyn();
        let expected = x.mapv(|item| item.ln()).iter().copied().collect();
        test_rep_ln_fixed128(x, expected);

        let x = array![
            [1.0_f64, 2.5, 3.0, 4.0, 5.0],
            [1.33, 4.123, 13.432, 10.33, 55.33]
        ]
        .into_dyn();
        let expected = x.mapv(|item| item.ln()).iter().copied().collect();
        test_rep_ln_fixed128(x, expected);

        let x = array![[
            [1.0_f64, 2.5],
            [10.3121, 123.025],
            [15.3213, 65.323],
            [128.321, 156.3214]
        ]]
        .into_dyn();
        let expected = x.mapv(|item| item.ln()).iter().copied().collect();
        test_rep_ln_fixed128(x, expected);
    }
}
