//! Support for comparison operators

use super::*;
use crate::computation::EqualOp;
use crate::error::Result;
use crate::execution::Session;
use crate::{Const, Ring};

impl EqualOp {
    pub(crate) fn rep_kernel<S: Session, RepRingT, RepBitT, RepBitArrayT, ShapeT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepBitT>
    where
        RepRingT: Ring<BitLength = N>,
        ReplicatedPlacement: PlacementBitDecompose<S, RepRingT, RepBitArrayT>,
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

        // TODO(Morten) can we use `neg` here instead? would it be more efficient?
        let v_not: Vec<_> = v.iter().map(|vi| rep.xor(sess, &ones, vi)).collect();

        // TODO we can optimize this by having a binary multiplier like
        // we are doing with the binary adder in bit decomposition
        Ok(v_not.iter().fold(ones, |acc, y| rep.mul(sess, &acc, y)))
    }
}

impl LessThanOp {
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

impl GreaterThanOp {
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
