use super::*;
use crate::computation::RepEqualOp;
use crate::error::Result;
use crate::replicated::{ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing64Tensor};
use crate::{Const, Ring};

modelled_kernel! {
    PlacementEqual::equal, RepEqualOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
    ]
}

impl RepEqualOp {
    fn rep_kernel<S: Session, RepRingT, RepBitT, RepBitArrayT, ShapeT, N: Const>(
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
}

#[cfg(test)]
mod tests {
    use crate::computation::HostPlacement;
    use crate::host::{AbstractHostRingTensor, HostBitTensor};
    use crate::kernels::*;
    use crate::replicated::{ReplicatedBitTensor, ReplicatedPlacement};
    use ndarray::{array, IxDyn};

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

        let x = AbstractHostRingTensor::from_raw_plc(
            array![1024u64, 5, 4]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            alice.clone(),
        );

        let y = AbstractHostRingTensor::from_raw_plc(
            array![1024u64, 4, 5]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            bob,
        );

        let x_shared = rep.share(&sess, &x);

        let y_shared = rep.share(&sess, &y);

        let res: ReplicatedBitTensor = rep.equal(&sess, &x_shared, &y_shared);

        let opened_result = alice.reveal(&sess, &res);
        assert_eq!(
            opened_result,
            HostBitTensor::from_raw_plc(
                array![1, 0, 0].into_dimensionality::<IxDyn>().unwrap(),
                alice
            )
        );
    }
}
