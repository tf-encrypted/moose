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

pub trait PrefixOr<S: Session, SetupT, RepBitT> {
    fn prefix_or(&self, sess: &S, setup: &SetupT, x: Vec<RepBitT>) -> Vec<RepBitT>;
}

impl<S: Session, SetupT, RepBitT> PrefixOr<S, SetupT, RepBitT> for ReplicatedPlacement
where
    ReplicatedPlacement: PlacementAndSetup<S, SetupT, RepBitT, RepBitT, RepBitT>,
    ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
    RepBitT: Clone,
{
    /// Prefix Or protocol
    ///
    /// `x` is a replicated bit tensor.
    fn prefix_or(&self, sess: &S, setup: &SetupT, x: Vec<RepBitT>) -> Vec<RepBitT> {
        let v_len = x.len();

        let log_r = ((v_len as f64).log2().ceil()) as u32;
        let rep = self;

        let bitwise_or = |x: &RepBitT, y: &RepBitT| -> RepBitT {
            rep.xor(
                sess,
                &rep.xor(sess, x, y),
                &rep.and_setup(sess, setup, x, y),
            )
        };

        let mut res = x;
        for i in 0..log_r {
            for j in 0..(2_i32.pow(log_r) / 2_i32.pow(i + 1)) {
                let y = (2_i32.pow(i) + j * 2_i32.pow(i + 1) - 1) as usize;
                let k_bound = (2_i32.pow(i) + 1) as usize;
                for k in 1..k_bound {
                    if y + k < v_len {
                        res[y + k] = bitwise_or(&res[y], &res[y + k]);
                    }
                }
            }
        }
        res
    }
}

pub trait SignFromMsb<S: Session, T, O> {
    fn sign_from_msb(&self, sess: &S, msb_ring: &T) -> O;
}

impl<S: Session, RepRingT> SignFromMsb<S, RepRingT, RepRingT> for ReplicatedPlacement
where
    ReplicatedShape: KnownType<S>,

    ReplicatedPlacement: PlacementFillPrecission<S, cs!(ReplicatedShape), RepRingT>,
    ReplicatedPlacement: PlacementShape<S, RepRingT, cs!(ReplicatedShape)>,
    ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
{
    fn sign_from_msb(&self, sess: &S, msb_ring: &RepRingT) -> RepRingT {
        let rep = self;
        let double = rep.shl(sess, 1, msb_ring);

        // TODO(Dragos) use ones() from Morten's PR
        let one_value = Constant::Float64(1.0);

        let x_shape = rep.shape(sess, msb_ring);
        let ones = rep.fill_precision(sess, one_value, Some(0_u32), &x_shape);
        rep.sub(sess, &ones, &double)
    }
}

pub trait DivNorm<S: Session, SetupT, T, O> {
    fn norm(&self, sess: &S, setup: &SetupT, max_bits: usize, x: &T) -> (O, O);
}

impl<S: Session, SetupT, RepRingT, N> DivNorm<S, SetupT, RepRingT, RepRingT> for ReplicatedPlacement
where
    RepRingT: Ring<BitLength = N>,
    AbstractReplicatedBitArray<ReplicatedBitTensor, N>: KnownType<S>,
    ReplicatedBitTensor: KnownType<S>,

    ReplicatedPlacement: PlacementMsb<S, SetupT, RepRingT, RepRingT>,
    ReplicatedPlacement: SignFromMsb<S, RepRingT, RepRingT>,

    ReplicatedPlacement: PlacementMulSetup<S, SetupT, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementBitDecSetup<
        S,
        SetupT,
        RepRingT,
        cs!(AbstractReplicatedBitArray<ReplicatedBitTensor, N>),
    >,
    ReplicatedPlacement: TopMost<S, SetupT, cs!(ReplicatedBitTensor), RepRingT>,
    ReplicatedPlacement: PlacementIndex<
        S,
        cs!(AbstractReplicatedBitArray<ReplicatedBitTensor, N>),
        cs!(ReplicatedBitTensor),
    >,
    RepRingT: Clone,
{
    fn norm(
        &self,
        sess: &S,
        setup: &SetupT,
        max_bits: usize,
        x: &RepRingT,
    ) -> (RepRingT, RepRingT) {
        let rep = self;

        let msb = rep.msb(sess, setup, x);
        let sign = rep.sign_from_msb(sess, &msb);
        let abs_x = rep.mul_setup(sess, setup, &sign, x);

        // (Dragos) TODO: optimize this in the future, we don't need all bits (only max_bits from the bit-decomposition)
        let x_bits = rep.bit_decompose(sess, setup, &abs_x);

        let x_bits_vec: Vec<_> = (0..max_bits).map(|i| rep.index(sess, i, &x_bits)).collect();

        let top_most = rep.top_most(sess, setup, max_bits, x_bits_vec);
        let upshifted = rep.mul_setup(sess, setup, x, &top_most);

        let signed_topmost = rep.mul_setup(sess, setup, &sign, &top_most);
        (upshifted, signed_topmost)
    }
}

pub trait TopMost<S: Session, SetupT, RepBitT, RepRingT> {
    fn top_most(&self, sess: &S, setup: &SetupT, max_bits: usize, x: Vec<RepBitT>) -> RepRingT;
}

impl<S: Session, SetupT, RepBitT, RepRingT> TopMost<S, SetupT, RepBitT, RepRingT>
    for ReplicatedPlacement
where
    ReplicatedBitTensor: KnownType<S>,
    ReplicatedPlacement: PrefixOr<S, SetupT, RepBitT>,
    RepBitT: Clone,
    RepRingT: Clone,

    HostBitTensor: KnownType<S>,

    ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
    ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
{
    fn top_most(&self, sess: &S, setup: &SetupT, max_bits: usize, x: Vec<RepBitT>) -> RepRingT {
        assert_eq!(max_bits, x.len());

        let rep = self;
        let x_rev: Vec<_> = (0..max_bits).map(|i| x[max_bits - i - 1].clone()).collect();
        let y = rep.prefix_or(sess, setup, x_rev);

        let mut y_vec: Vec<_> = y
            .iter()
            .take(max_bits)
            .map(|item| rep.ring_inject(sess, 0, item))
            .collect();

        y_vec.reverse();
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

pub trait ApproximateReciprocal<S: Session, SetupT, T, O> {
    fn approximate_reciprocal(
        &self,
        sess: &S,
        setup: &SetupT,
        int_precision: usize,
        frac_precision: usize,
        x: &T,
    ) -> O;
}

impl<S: Session, SetupT, RepRingT> ApproximateReciprocal<S, SetupT, RepRingT, RepRingT>
    for ReplicatedPlacement
where
    ReplicatedShape: KnownType<S>,

    ReplicatedPlacement: DivNorm<S, SetupT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementShape<S, RepRingT, cs!(ReplicatedShape)>,
    ReplicatedPlacement: PlacementFillPrecission<S, cs!(ReplicatedShape), RepRingT>,
    ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementMulSetup<S, SetupT, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementTruncPr<S, RepRingT, RepRingT>,

    ReplicatedRing64Tensor: KnownType<S>,
{
    fn approximate_reciprocal(
        &self,
        sess: &S,
        setup: &SetupT,
        int_precision: usize,
        frac_precision: usize,
        x: &RepRingT,
    ) -> RepRingT {
        let rep = self;
        let total_precision = int_precision + frac_precision;

        let (upshifted, signed_topmost) = rep.norm(sess, setup, total_precision, x);

        let x_shape = rep.shape(sess, x);
        // 2.9142 * 2^{total_precision}
        let alpha = Constant::Float64(2.9142);
        let alpha = rep.fill_precision(sess, alpha, Some(total_precision as u32), &x_shape);
        let d = with_context!(rep, sess, alpha - rep.shl(sess, 1, &upshifted));
        let w = rep.mul_setup(sess, setup, &d, &signed_topmost);

        // truncate result
        rep.trunc_pr(sess, 2 * int_precision as u32, &w)
    }
}
