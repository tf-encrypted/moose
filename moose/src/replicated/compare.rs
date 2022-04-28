//! Support for comparison operators

use super::*;
use crate::computation::EqualOp;
use crate::error::Result;
use crate::execution::Session;
use crate::Const;

pub(crate) trait TreeReduceMul<S: Session, T, O> {
    fn reduce_mul(&self, sess: &S, x: &[T]) -> O;
}

impl<S: Session, T: Clone> TreeReduceMul<S, T, T> for ReplicatedPlacement
where
    ReplicatedPlacement: PlacementMul<S, T, T, T>,
{
    fn reduce_mul(&self, sess: &S, x: &[T]) -> T {
        let elementwise_mul =
            |rep: &ReplicatedPlacement, sess: &S, x: &T, y: &T| -> T { rep.mul(sess, x, y) };
        self.tree_reduce(sess, x, elementwise_mul)
    }
}

impl EqualOp {
    pub(crate) fn rep_kernel<S: Session, RepRingT, RepBitT, RepBitArrayT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementBitDecompose<S, RepRingT, RepBitArrayT>,
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementEqualZero<S, RepBitArrayT, RepBitT>,
    {
        let z = rep.sub(sess, &x, &y);
        let bits = rep.bit_decompose(sess, &z);
        Ok(rep.equal_zero(sess, &bits))
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

impl EqualZeroOp {
    pub(crate) fn bitdec_bit_kernel<S: Session, RepBitArrayT, RepBitT, MirBitT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepBitArrayT,
    ) -> Result<RepBitT>
    where
        RepBitArrayT: BitArray<Len = N>,
        ReplicatedPlacement: PlacementIndex<S, RepBitArrayT, RepBitT>,
        ReplicatedPlacement: ShapeFill<S, RepBitT, Result = MirBitT>,
        ReplicatedPlacement: PlacementXor<S, MirBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: TreeReduceMul<S, RepBitT, RepBitT>,
    {
        let vx: Vec<_> = (0..N::VALUE).map(|i| rep.index(sess, i, &x)).collect();

        let ones = rep.shape_fill(sess, 1u8, &vx[0]);
        let v_not: Vec<_> = vx.iter().map(|vi| rep.xor(sess, &ones, vi)).collect();

        Ok(rep.reduce_mul(sess, &v_not))
    }

    pub(crate) fn bitdec_ring_kernel<S: Session, RepBitArrayT, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepBitArrayT,
    ) -> Result<RepRingT>
    where
        ReplicatedPlacement: PlacementEqualZero<S, RepBitArrayT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
    {
        let r_bit = rep.equal_zero(sess, &x);
        Ok(rep.ring_inject(sess, 0, &r_bit))
    }
}

impl LessOp {
    pub(crate) fn rep_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &x, &y);
        Ok(rep.msb(sess, &z))
    }

    pub(crate) fn rep_mir_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: MirRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, MirRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &x, &y);
        Ok(rep.msb(sess, &z))
    }

    pub(crate) fn mir_rep_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: MirRingT,
        y: RepRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, MirRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &x, &y);
        Ok(rep.msb(sess, &z))
    }
}

impl GreaterOp {
    pub(crate) fn rep_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &y, &x);
        Ok(rep.msb(sess, &z))
    }

    pub(crate) fn rep_mir_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: MirRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, MirRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &y, &x);
        Ok(rep.msb(sess, &z))
    }

    pub(crate) fn mir_rep_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: MirRingT,
        y: RepRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, MirRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &y, &x);
        Ok(rep.msb(sess, &z))
    }
}

#[cfg(feature = "sync_execute")]
#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use ndarray::prelude::*;

    #[test]
    fn test_equal() {
        let alice = HostPlacement::from("alice");
        let bob = HostPlacement::from("bob");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        let sess = SyncSession::default();

        let x: HostRing64Tensor = alice.from_raw(array![1024u64, 5, 4]);
        let y: HostRing64Tensor = bob.from_raw(array![1024u64, 4, 5]);

        let x_shared = rep.share(&sess, &x);
        let y_shared = rep.share(&sess, &y);

        let res: ReplicatedBitTensor = rep.equal(&sess, &x_shared, &y_shared);

        let opened_result = alice.reveal(&sess, &res);
        assert_eq!(opened_result, alice.from_raw(array![1, 0, 0]));
    }
}
