use super::*;

impl FixedpointDivOp {
    pub fn rep_rep_kernel<S: Session, RepRingT, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
        y: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<st!(AbstractReplicatedFixedTensor<RepRingT>)>
    where
        AbstractReplicatedFixedTensor<RepRingT>: CanonicalType,
        <AbstractReplicatedFixedTensor<RepRingT> as CanonicalType>::Type: KnownType<S>,
        AbstractReplicatedFixedTensor<RepRingT>: Into<st!(AbstractReplicatedFixedTensor<RepRingT>)>,
        ReplicatedPlacement: PlacementShape<S, RepRingT, cs!(ReplicatedShape)>,
        ReplicatedPlacement: PlacementFillPrecission<S, cs!(ReplicatedShape), RepRingT>,
        ReplicatedPlacement: ApproximateReciprocal<S, S::ReplicatedSetup, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMulSetup<S, S::ReplicatedSetup, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementTruncPr<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
        RepRingT: Ring,

        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        RepRingT: Clone,
        ReplicatedShape: KnownType<S>,

        HostPlacement: PlacementReveal<S, RepRingT, HostRingT>,
        HostRingT: std::fmt::Debug,
    {
        #![allow(clippy::many_single_char_names)]
        let setup = rep.gen_setup(sess);

        assert_eq!(x.integral_precision, y.integral_precision);
        assert_eq!(x.fractional_precision, y.fractional_precision);

        let int_precision = x.integral_precision;
        let frac_precision = x.fractional_precision;

        let k = int_precision + frac_precision;

        assert!(2 * k as usize <= RepRingT::BitLength::VALUE);

        let constant_quotient: f64 = 17_f64.log2();
        let theta = (((frac_precision + 1) as f64) / constant_quotient)
            .log2()
            .ceil() as u32;

        let x_st = x.tensor;
        let y_st = y.tensor;

        let x_shape = rep.shape(sess, &x_st);

        let w = rep.approximate_reciprocal(
            sess,
            &setup,
            int_precision as usize,
            frac_precision as usize,
            &y_st,
        );
        // max_bits(w) = k

        let alpha = Constant::Float64(1.0);
        let rep_alpha = rep.fill_precision(sess, alpha, Some(2 * frac_precision), &x_shape);

        let mut a = with_context!(
            rep,
            sess,
            rep_alpha - &rep.mul_setup(sess, &setup, &y_st, &w)
        );
        // max_bits(a) = max(2f, k)

        let mut b = rep.mul_setup(sess, &setup, &x_st, &w);

        // no need to truncate with 2f since w is already truncated
        b = rep.trunc_pr(sess, frac_precision, &b);

        // TODO [Yann] fix to return tuple (a, b)
        for _i in 0..theta {
            let x = rep.mul_setup(sess, &setup, &a, &a);
            let y = rep.mul_setup(sess, &setup, &b, &rep.add(sess, &rep_alpha, &a));
            a = rep.trunc_pr(sess, 2 * frac_precision, &x);
            b = rep.trunc_pr(sess, 2 * frac_precision, &y);
        }
        b = rep.mul_setup(sess, &setup, &b, &rep.add(sess, &rep_alpha, &a));
        b = rep.trunc_pr(sess, 2 * frac_precision, &b);

        Ok(AbstractReplicatedFixedTensor {
            tensor: b,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
            fractional_precision: x.fractional_precision,
        }
        .into())
    }
}
