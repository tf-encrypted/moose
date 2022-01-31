use super::*;
use crate::computation::RepEqualOp;
use crate::error::Result;
use crate::execution::Session;
use crate::{Const, Ring};

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
    pub(crate) fn rep_rep_kernel<S: Session, RepRingT>(
        _sess: &S,
        _rep: &ReplicatedPlacement,
        _x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>> {
        unimplemented!()
    }
}

/// Internal trait for converting int to float number generation
/// Computes v, p, s, z such that (1 − 2s) · (1 − z) · v · 2 p = x.
trait Int2FL<S: Session, RepRingT> {
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

        // hack because fill op doesn't work without adding too many un-necessary type constrains
        let zero = rep.sub(sess, x, x);

        // TODO(Dragos) this can be optimized by performing a single bit-decomposition and use it
        // to compute the msb and equality to zero
        let sign = rep.msb(sess, x);
        let is_zero = rep.equal(sess, x, &zero);

        let x_pos = rep.mux(sess, &sign, &rep.neg(sess, x), x);
        let x_pos_bits = rep.bit_decompose(sess, &x_pos);

        let x_pos_bits: Vec<_> = (0..k).map(|i| rep.index(sess, i, &x_pos_bits)).collect();

        let x_pos_rev: Vec<_> = x_pos_bits.into_iter().rev().collect();
        let b = rep.prefix_or(sess, x_pos_rev);
        let b_ring: Vec<_> = (0..k).map(|i| rep.ring_inject(sess, 0, &b[i])).collect();

        let ones = rep.shape_fill(sess, 1_u8, x);
        let neg_b_sum = (0..k).fold(zero.clone(), |acc, i| {
            let neg = rep.sub(sess, &ones, &b_ring[i]);
            rep.add(sess, &acc, &rep.shl(sess, i, &neg))
        });

        let x_shifted = with_context!(rep, sess, x_pos * (ones + neg_b_sum));
        let x_norm = rep.trunc_pr(sess, f as u32, &x_shifted);

        let bit_count = (0..k).fold(zero, |acc, i| rep.add(sess, &acc, &b_ring[i]));
        let ften = rep.shape_fill(sess, f as u8, &bit_count);
        let p_res = with_context!(rep, sess, (bit_count - ften) * (ones - is_zero));

        (x_norm, p_res, sign, is_zero)
    }
}

#[cfg(test)]
mod tests {
    use crate::execution::SyncSession;
    use crate::host::{FromRaw, HostPlacement};
    use crate::kernels::*;
    use crate::replicated::{ReplicatedBitTensor, ReplicatedPlacement};
    use crate::types::{HostBitTensor, HostRing64Tensor};
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
}
