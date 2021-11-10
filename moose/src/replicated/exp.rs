use super::*;

impl ExpOp {
    pub(crate) fn rep_rep_kernel<S: Session, RepRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>> {
        let base_e = 1.0_f64.exp().log2();
        Ok(x)
    }
}

impl Pow2Op {
    pub(crate) fn rep_rep_kernel<S: Session, RepRingT, RepBitT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        RepRingT: Ring<BitLength = N>,

        RepBitT: Clone,
        AbstractReplicatedBitArray<RepBitT, N>: Into<st!(AbstractReplicatedBitArray<RepBitT, N>)>,
        AbstractReplicatedBitArray<RepBitT, N>: CanonicalType,
        <AbstractReplicatedBitArray<RepBitT, N> as CanonicalType>::Type: KnownType<S>,

        ReplicatedPlacement: PlacementShrRaw<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementSplit<S, RepRingT, RepBitT, RepBitT>,
        ReplicatedPlacement: BinaryAdder<S, RepBitT>,
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement:
            PlacementIndex<S, st!(AbstractReplicatedBitArray<RepBitT, N>), RepBitT>,
        ReplicatedPlacement: PlacementAdd<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
    {
        let integral_precision = x.integral_precision as usize;
        let fractional_precision = x.fractional_precision as usize;
        let _k = fractional_precision + integral_precision;

        let (x0, x1) = rep.split(sess, &x.tensor);
        let bits = rep.binary_adder(sess, &x0, &x1, RepRingT::BitLength::VALUE);

        let x_bits = AbstractReplicatedBitArray::<RepBitT, N>(bits, PhantomData);
        let x0_bits = AbstractReplicatedBitArray::<RepBitT, N>(x0, PhantomData);
        let x1_bits = AbstractReplicatedBitArray::<RepBitT, N>(x1, PhantomData);

        let x0_f = rep.index(sess, fractional_precision, &x0_bits.into());
        let x1_f = rep.index(sess, fractional_precision, &x1_bits.into());
        let b_f = rep.index(sess, fractional_precision, &x_bits.into());

        let overflow_half1 = rep.ring_inject(sess, fractional_precision, &x0_f);
        let overflow_half2 = with_context!(rep, sess, x0_f + x1_f + b_f);
        let overflow_half2 = rep.ring_inject(sess, fractional_precision, &overflow_half2);
        let overflow = with_context!(rep, sess, overflow_half1 + overflow_half2);

        let x_shifted_raw = rep.shr_raw(sess, fractional_precision, &x.tensor);
        let lower = with_context!(rep, sess, x_shifted_raw - overflow);

        Ok(AbstractReplicatedFixedTensor {
            tensor: lower,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
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
