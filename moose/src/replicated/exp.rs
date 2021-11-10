use crate::fixedpoint::PolynomialEval;

use super::*;

impl Pow2Op {
    pub(crate) fn rep_rep_kernel<S: Session, RepRingT, RepBitT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        AbstractReplicatedFixedTensor<RepRingT>: CanonicalType,
        <AbstractReplicatedFixedTensor<RepRingT> as CanonicalType>::Type: KnownType<S>,

        RepRingT: Ring<BitLength = N>,
        RepRingT: Clone,

        RepBitT: Clone,
        AbstractReplicatedBitArray<RepBitT, N>: Into<st!(AbstractReplicatedBitArray<RepBitT, N>)>,
        AbstractReplicatedBitArray<RepBitT, N>: CanonicalType,
        <AbstractReplicatedBitArray<RepBitT, N> as CanonicalType>::Type: KnownType<S>,

        ReplicatedPlacement: PlacementShrRaw<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementSplit<S, RepRingT, RepBitT, RepBitT>,
        ReplicatedPlacement: BinaryAdder<S, RepBitT>,
        ReplicatedPlacement:
            PlacementIndex<S, st!(AbstractReplicatedBitArray<RepBitT, N>), RepBitT>,
        ReplicatedPlacement: PlacementAdd<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
        ReplicatedPlacement: Pow2FromBits<S, RepRingT>,

        ReplicatedPlacement:
            ExpFromParts<S, RepRingT, st!(AbstractReplicatedFixedTensor<RepRingT>)>,
    {
        // unimplemented!()
        let integral_precision = x.integral_precision as usize;
        let fractional_precision = x.fractional_precision as usize;
        let k = fractional_precision + integral_precision;

        let (x0, x1) = rep.split(sess, &x.tensor);
        let bits = rep.binary_adder(sess, &x0, &x1, RepRingT::BitLength::VALUE);

        let x_bits = AbstractReplicatedBitArray::<RepBitT, N>(bits, PhantomData);
        let x0_bits = AbstractReplicatedBitArray::<RepBitT, N>(x0, PhantomData);
        let x1_bits = AbstractReplicatedBitArray::<RepBitT, N>(x1, PhantomData);

        let x0_f = rep.index(sess, fractional_precision, &x0_bits.into());
        let x1_f = rep.index(sess, fractional_precision, &x1_bits.into());

        let x_bits_canonical = x_bits.into();
        let b_f = rep.index(sess, fractional_precision, &x_bits_canonical);

        let overflow_half1 = rep.ring_inject(sess, fractional_precision, &x0_f);
        let overflow_half2 = with_context!(rep, sess, x0_f + x1_f + b_f);
        let overflow_half2 = rep.ring_inject(sess, fractional_precision, &overflow_half2);
        let shifted_overflow = with_context!(rep, sess, overflow_half1 + overflow_half2);

        // compute RawMod2M
        let x_shifted_raw = rep.shr_raw(sess, fractional_precision, &x.tensor);
        let x_mod2m = with_context!(
            rep,
            sess,
            x.tensor - &rep.shl(sess, fractional_precision, &x_shifted_raw)
        );
        let lower = with_context!(rep, sess, x_mod2m - shifted_overflow);
        // convert lower to fixed point representation
        // let c = AbstractReplicatedFixedTensor {
        //     tensor: lower,
        //     fractional_precision: fractional_precision as u32,
        //     integral_precision: integral_precision as u32,
        // };

        let higher: Vec<_> = (fractional_precision..k)
            .map(|i| rep.ring_inject(sess, 0, &rep.index(sess, i, &x_bits_canonical)))
            .collect();

        let d = rep.pow2_from_bits(sess, higher.as_slice());
        unimplemented!()
        // let g = rep.exp_from_parts(sess, &d, &lower, fractional_precision as u32, k as u32);
        // Ok(g)
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
                // compute b(i) * 2^i
                let pos = rep.shl(sess, i, bit);
                // compute 1 - b(i)
                let neg = rep.sub(sess, &ones, bit);
                // compute p(i) = b(i) * 2^i + (1 - b(i))
                rep.add(sess, &pos, &neg)
            })
            .collect();

        // TODO(Dragos) do tree multiplication here
        selectors.iter().fold(ones, |acc, y| rep.mul(sess, &acc, y))
    }
}

pub(crate) trait ExpFromParts<S: Session, RepRingT, RepFixedT> {
    fn exp_from_parts(
        &self,
        sess: &S,
        e_int: &RepRingT,
        e_frac: &RepRingT,
        f: u32,
        k: u32,
    ) -> RepFixedT;
}
impl<S: Session, RepRingT, RepFixedT> ExpFromParts<S, RepRingT, RepFixedT> for ReplicatedPlacement
where
    ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
    RepRingT: Clone,
    ReplicatedPlacement: PolynomialEval<S, RepFixedT>,
    AbstractReplicatedFixedTensor<RepRingT>: TryInto<RepFixedT>,
    RepFixedT: TryInto<AbstractReplicatedFixedTensor<RepRingT>>,
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
    ) -> RepFixedT {
        let p_1045: Vec<_> = vec![
            1.0f64,
            0.6931471805599453,
            0.2402265069591007,
            0.055504108664821576,
            0.009618129107628477,
            0.0013333558146428441,
            0.00015403530393381606,
            1.5252733804059838e-05,
            1.3215486790144305e-06,
            1.0178086009239696e-07,
            7.054911620801121e-09,
            4.44553827187081e-10,
            2.5678435993488196e-11,
            1.3691488853904124e-12,
            6.778726354822543e-14,
            3.132436707088427e-15,
            1.357024794875514e-16,
            5.533046532458238e-18,
            2.1306753354891168e-19,
            7.77300842885735e-21,
        ];

        let amount = k - 2 - f;
        let x = AbstractReplicatedFixedTensor {
            tensor: self.shl(sess, amount as usize, e_frac),
            integral_precision: 2,
            fractional_precision: k - 2,
        };
        let e_approx = self.polynomial_eval(sess, p_1045, x.try_into().ok().unwrap());
        let e_approx_f: AbstractReplicatedFixedTensor<RepRingT> = e_approx.try_into().ok().unwrap();

        let e_approx_ring = e_approx_f.tensor;
        let e_prod = self.mul(sess, e_int, &e_approx_ring);
        let e_tr = self.trunc_pr(sess, amount, &e_prod);

        let e_tr_fixed: AbstractReplicatedFixedTensor<RepRingT> = AbstractReplicatedFixedTensor {
            tensor: e_tr,
            integral_precision: k - f,
            fractional_precision: f,
        };

        e_tr_fixed.try_into().ok().unwrap()
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
        let target = AbstractHostRingTensor::from_raw_plc(array![64u64], alice.clone());

        let sess = SyncSession::default();

        let x_shared = rep.share(&sess, &x);
        let x0 = rep.index_axis(&sess, 0, 0, &x_shared);
        let x1 = rep.index_axis(&sess, 0, 1, &x_shared);
        let x2 = rep.index_axis(&sess, 0, 2, &x_shared);
        let x3 = rep.index_axis(&sess, 0, 3, &x_shared);

        let x_vec = vec![x0, x1, x2, x3];
        let pow2_shared = rep.pow2_from_bits(&sess, &x_vec);

        assert_eq!(target, alice.reveal(&sess, &pow2_shared));
    }
}
