use super::*;

impl FixedpointDivOp {
    pub(crate) fn rep_rep_kernel<S: Session, RepRingT, MirroredT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        RepRingT: Ring,
        ReplicatedPlacement: ApproximateReciprocal<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementTruncPr<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, MirroredT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementSub<S, MirroredT, RepRingT, RepRingT>,
        ReplicatedPlacement: ShapeFill<S, RepRingT, Result = MirroredT>,
    {
        #![allow(clippy::many_single_char_names)]

        assert_eq!(x.integral_precision, y.integral_precision);
        assert_eq!(x.fractional_precision, y.fractional_precision);

        let int_precision = x.integral_precision;
        let frac_precision = x.fractional_precision;

        let k = int_precision + frac_precision;

        assert!(2 * k as usize <= RepRingT::BitLength::VALUE);

        let constant_quotient: f64 = 17_f64.log2();
        let theta = ((k as f64) / constant_quotient).log2().ceil() as u32;

        let x_st = x.tensor;
        let y_st = y.tensor;

        let w = rep.approximate_reciprocal(
            sess,
            int_precision as usize,
            frac_precision as usize,
            &y_st,
        );
        // max_bits(w) = k

        let alpha = Constant::Fixed(FixedpointConstant {
            value: 1.0,
            precision: 2 * frac_precision as usize,
        });
        let rep_alpha = rep.shape_fill(sess, alpha, &x_st);

        let mut a = with_context!(rep, sess, rep_alpha - &rep.mul(sess, &y_st, &w));
        // max_bits(a) = max(2f, k)

        let mut b = rep.mul(sess, &x_st, &w);

        // no need to truncate with 2f since w is already truncated
        b = rep.trunc_pr(sess, frac_precision, &b);

        for _i in 0..theta {
            let x = rep.mul(sess, &a, &a);
            let y = rep.mul(sess, &b, &rep.add(sess, &rep_alpha, &a));
            a = rep.trunc_pr(sess, 2 * frac_precision, &x);
            b = rep.trunc_pr(sess, 2 * frac_precision, &y);
        }
        b = rep.mul(sess, &b, &rep.add(sess, &rep_alpha, &a));
        b = rep.trunc_pr(sess, 2 * frac_precision, &b);

        Ok(AbstractReplicatedFixedTensor {
            tensor: b,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
            fractional_precision: x.fractional_precision,
        })
    }
}

pub(crate) trait SignFromMsb<S: Session, T, O> {
    fn sign_from_msb(&self, sess: &S, msb_ring: &T) -> O;
}

impl<S: Session, RepRingT, MirroredT> SignFromMsb<S, RepRingT, RepRingT> for ReplicatedPlacement
where
    ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementSub<S, MirroredT, RepRingT, RepRingT>,
    ReplicatedPlacement: ShapeFill<S, RepRingT, Result = MirroredT>,
{
    fn sign_from_msb(&self, sess: &S, msb_ring: &RepRingT) -> RepRingT {
        let rep = self;
        let double = rep.shl(sess, 1, msb_ring);
        let ones = rep.shape_fill(sess, 1_u8, msb_ring);
        rep.sub(sess, &ones, &double)
    }
}

pub(crate) trait DivNorm<S: Session, T, O> {
    fn norm(&self, sess: &S, max_bits: usize, x: &T) -> (O, O);
}

impl<S: Session, RepRingT, N> DivNorm<S, RepRingT, RepRingT> for ReplicatedPlacement
where
    RepRingT: Ring<BitLength = N>,
    AbstractReplicatedBitArray<ReplicatedBitTensor, N>: KnownType<S>,
    ReplicatedBitTensor: KnownType<S>,

    ReplicatedPlacement: PlacementMsb<S, RepRingT, RepRingT>,
    ReplicatedPlacement: SignFromMsb<S, RepRingT, RepRingT>,

    ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement:
        PlacementBitDec<S, RepRingT, cs!(AbstractReplicatedBitArray<ReplicatedBitTensor, N>)>,
    ReplicatedPlacement: TopMost<S, cs!(ReplicatedBitTensor), RepRingT>,
    ReplicatedPlacement: PlacementIndex<
        S,
        cs!(AbstractReplicatedBitArray<ReplicatedBitTensor, N>),
        cs!(ReplicatedBitTensor),
    >,
{
    fn norm(&self, sess: &S, max_bits: usize, x: &RepRingT) -> (RepRingT, RepRingT) {
        let rep = self;

        let msb = rep.msb(sess, x);
        let sign = rep.sign_from_msb(sess, &msb);
        let abs_x = rep.mul(sess, &sign, x);

        // Although we don't need all bits (only max_bits from the bit-decomposition)
        // this is going to be optimized when using the rust compiler since the extra operations
        // will be pruned away.
        let x_bits = rep.bit_decompose(sess, &abs_x);
        let x_bits_vec: Vec<_> = (0..max_bits).map(|i| rep.index(sess, i, &x_bits)).collect();

        let top_most = rep.top_most(sess, max_bits, x_bits_vec);
        let upshifted = rep.mul(sess, x, &top_most);

        let signed_topmost = rep.mul(sess, &sign, &top_most);
        (upshifted, signed_topmost)
    }
}

pub(crate) trait TopMost<S: Session, RepBitT, RepRingT> {
    fn top_most(&self, sess: &S, max_bits: usize, x: Vec<RepBitT>) -> RepRingT;
}

impl<S: Session, RepBitT, RepRingT> TopMost<S, RepBitT, RepRingT> for ReplicatedPlacement
where
    ReplicatedBitTensor: KnownType<S>,
    HostBitTensor: KnownType<S>,
    RepBitT: Clone,
    RepRingT: Clone,
    ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
    ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementAnd<S, RepBitT, RepBitT, RepBitT>,
    ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
{
    fn top_most(&self, sess: &S, max_bits: usize, x: Vec<RepBitT>) -> RepRingT {
        assert_eq!(max_bits, x.len());

        let rep = self;
        let x_rev: Vec<_> = x.into_iter().take(max_bits).rev().collect();

        let y = rep.prefix_or(sess, x_rev);

        let y_vec: Vec<_> = y
            .iter()
            .take(max_bits)
            .map(|item| rep.ring_inject(sess, 0, item))
            .rev()
            .collect();

        let mut z: Vec<_> = (0..max_bits - 1)
            .map(|i| rep.sub(sess, &y_vec[i], &y_vec[i + 1]))
            .collect();

        z.push(y_vec[max_bits - 1].clone());

        let s_vec: Vec<_> = (0..max_bits)
            .map(|i| rep.shl(sess, max_bits - i - 1, &z[i]))
            .collect();

        // note this can be replaced with a variadic kernel for replicated sum operation
        let mut res = rep.shl(sess, 0, &s_vec[max_bits - 1]);
        for item in s_vec.iter().take(max_bits).skip(1) {
            res = rep.add(sess, &res, item);
        }
        res
    }
}

pub(crate) trait ApproximateReciprocal<S: Session, T, O> {
    fn approximate_reciprocal(
        &self,
        sess: &S,
        int_precision: usize,
        frac_precision: usize,
        x: &T,
    ) -> O;
}

impl<S: Session, RepRingT, MirroredT> ApproximateReciprocal<S, RepRingT, RepRingT>
    for ReplicatedPlacement
where
    ReplicatedPlacement: DivNorm<S, RepRingT, RepRingT>,
    ReplicatedPlacement: ShapeFill<S, RepRingT, Result = MirroredT>,
    ReplicatedPlacement: PlacementSub<S, MirroredT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementTruncPr<S, RepRingT, RepRingT>,
{
    fn approximate_reciprocal(
        &self,
        sess: &S,
        int_precision: usize,
        frac_precision: usize,
        x: &RepRingT,
    ) -> RepRingT {
        let rep = self;
        let total_precision = int_precision + frac_precision;

        let (upshifted, signed_topmost) = rep.norm(sess, total_precision, x);

        // 2.9142 * 2^{total_precision}
        let alpha = Constant::Fixed(FixedpointConstant {
            value: 2.9142,
            precision: total_precision,
        });
        let alpha = rep.shape_fill(sess, alpha, x);

        let d = with_context!(rep, sess, alpha - rep.shl(sess, 1, &upshifted));
        let w = rep.mul(sess, &d, &signed_topmost);

        // truncate result
        rep.trunc_pr(sess, 2 * int_precision as u32, &w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::AbstractHostRingTensor;
    use crate::kernels::SyncSession;
    use ndarray::array;

    #[test]
    fn test_norm() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x = AbstractHostRingTensor::from_raw_plc(array![896u64], alice.clone());

        let sess = SyncSession::default();

        let x_shared = rep.share(&sess, &x);

        let (upshifted, topmost) = rep.norm(&sess, 12, &x_shared);

        let topmost_target = AbstractHostRingTensor::from_raw_plc(array![4u64], alice.clone());
        let upshifted_target = AbstractHostRingTensor::from_raw_plc(array![3584], alice.clone());

        assert_eq!(topmost_target, alice.reveal(&sess, &topmost));
        assert_eq!(upshifted_target, alice.reveal(&sess, &upshifted));
    }

    #[test]
    fn test_binary_adder() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x = AbstractHostRingTensor::from_raw_plc(array![3884509700957842751u64], alice.clone());
        let y =
            AbstractHostRingTensor::from_raw_plc(array![13611438098135434720u64], alice.clone());
        let expected_output = x.clone() + y.clone();

        let sess = SyncSession::default();

        let x_bit = alice.bit_decompose(&sess, &x);
        let y_bit = alice.bit_decompose(&sess, &y);
        let expected_output_bit: HostBitTensor = alice.bit_decompose(&sess, &expected_output);

        let x_shared = rep.share(&sess, &x_bit);
        let y_shared = rep.share(&sess, &y_bit);
        let binary_adder = rep.binary_adder(&sess, x_shared, y_shared, 64);
        let binary_adder_clear = alice.reveal(&sess, &binary_adder);

        assert_eq!(expected_output_bit, binary_adder_clear);
    }

    #[test]
    fn test_approximate_reciprocal() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        // 3.5 * 2^8
        let x = AbstractHostRingTensor::from_raw_plc(array![896u64], alice.clone());

        let sess = SyncSession::default();

        let expected_output = array![74i64];

        let x_shared = rep.share(&sess, &x);
        let approximation = rep.approximate_reciprocal(&sess, 4, 8, &x_shared);

        let out = alice.reveal(&sess, &approximation).0;
        for (i, item) in out.iter().enumerate() {
            match item {
                std::num::Wrapping(x) => {
                    let d = (*x as i64) - expected_output[i];
                    assert!(d * d <= 1);
                }
            }
        }
    }
}
