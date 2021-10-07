use crate::computation::{
    KnownType, Placed, RepEqualOp, RepIfElseOp, RepNegOp, ReplicatedPlacement, HostPlacement,
};
use crate::error::Result;
use crate::kernels::*;
use crate::replicated::{
    RepTen, ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing64Tensor,
};
use crate::{Const, Ring};

modelled!(PlacementEqual::equal, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor, RepEqualOp);
modelled!(PlacementEqual::equal, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor, RepEqualOp);

kernel! {
    RepEqualOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
    ]
}

impl RepEqualOp {
    fn rep_kernel<S: Session, HostRingT, RepBitT, RepBitArrayT, ShapeT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: HostRingT,
        y: HostRingT,
    ) -> Result<RepBitT>
    where
        HostRingT: Ring<BitLength = N>,

        ReplicatedPlacement: PlacementBitDecSetup<S, S::ReplicatedSetup, HostRingT, RepBitArrayT>,
        ReplicatedPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
        ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepBitT>,
        ReplicatedPlacement: PlacementShape<S, HostRingT, ShapeT>,
        ReplicatedPlacement: PlacementIndex<S, RepBitArrayT, RepBitT>,
        ReplicatedPlacement: PlacementMulSetup<S, S::ReplicatedSetup, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
    {
        let setup = rep.gen_setup(sess);

        let z = rep.sub(sess, &x, &y);
        let bits = rep.bit_decompose(sess, &setup, &z);

        let v: Vec<_> = (0..HostRingT::BitLength::VALUE)
            .map(|i| rep.index(sess, i, &bits))
            .collect();

        let ones = rep.fill(sess, 1u8.into(), &rep.shape(sess, &z));

        let v_not: Vec<_> = v.iter().map(|vi| rep.xor(sess, &ones, vi)).collect();

        // TODO we can optimize this by having a binary multipler like
        // we are doing with the binary adder in bit decompitision
        Ok(v_not
            .iter()
            .fold(ones, |acc, y| rep.mul_setup(sess, &setup, &acc, y)))
    }
}

modelled!(PlacementIfElse::if_else, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepIfElseOp);
modelled!(PlacementIfElse::if_else, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepIfElseOp);

kernel! {
    RepIfElseOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor  => [transparent] Self::rep_kernel),
    ]
}

impl RepIfElseOp {
    fn rep_kernel<S: Session, HostRingT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        s: HostRingT,
        x: HostRingT,
        y: HostRingT,
    ) -> Result<HostRingT>
    where
        ReplicatedPlacement: PlacementFill<S, ShapeT, HostRingT>,
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement:
            PlacementMulSetup<S, S::ReplicatedSetup, HostRingT, HostRingT, HostRingT>,
        ReplicatedPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
        ReplicatedPlacement: PlacementShape<S, HostRingT, ShapeT>,
        ReplicatedPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
    {
        let setup = rep.gen_setup(sess);
        let ones = rep.fill(sess, 1u64.into(), &rep.shape(sess, &x));

        // if else [s] * [x] + (1 - [s]) * [y]
        let s_x = rep.mul_setup(sess, &setup, &s, &x);

        let ones_minus_s = rep.sub(sess, &ones, &s);
        let ones_s_y = rep.mul_setup(sess, &setup, &ones_minus_s, &y);

        Ok(rep.add(sess, &s_x, &ones_s_y))
    }
}

modelled!(PlacementNeg::neg, ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor, RepNegOp);
modelled!(PlacementNeg::neg, ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepNegOp);
modelled!(PlacementNeg::neg, ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepNegOp);

kernel! {
    RepNegOp,
    [
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor => [runtime] Self::rep_bit_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [runtime] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor ) ->ReplicatedRing128Tensor  => [runtime] Self::rep_rep_kernel),
    ]
}

impl RepNegOp {
    fn rep_bit_kernel<S: Session, HostBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<HostBitT>,
    ) -> Result<RepTen<HostBitT>>
    where
        HostPlacement: PlacementNeg<S, HostBitT, HostBitT>,
    {
        let (player0, _player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        // TODO(Morten)
        // we could choose share to change at random
        // to more fairly distribute compute load
        let y00 = player0.neg(sess, &x00);
        let y10 = x10;
        let y11 = x11;
        let y21 = x21;
        let y22 = x22;
        let y02 = player2.neg(sess, &x02);

        Ok(RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        })
    }

    fn rep_rep_kernel<S: Session, HostRepT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<HostRepT>,
    ) -> Result<RepTen<HostRepT>>
    where
        HostPlacement: PlacementNeg<S, HostRepT, HostRepT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let y00 = player0.neg(sess, &x00);
        let y10 = player0.neg(sess, &x10);
        let y11 = player1.neg(sess, &x11);
        let y21 = player1.neg(sess, &x21);
        let y22 = player2.neg(sess, &x22);
        let y02 = player2.neg(sess, &x02);

        Ok(RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::computation::{HostPlacement, ReplicatedPlacement};
    use crate::host::{AbstractHostRingTensor, FromRawPlc, HostBitTensor};
    use crate::kernels::*;
    use crate::replicated::{AbstractReplicatedRingTensor, ReplicatedBitTensor};
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
        let setup = rep.gen_setup(&sess);

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

        let x_shared = rep.share(&sess, &setup, &x);

        let y_shared = rep.share(&sess, &setup, &y);

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

    #[test]
    fn test_if_else() {
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
        let setup = rep.gen_setup(&sess);

        let scaling_base = 2;
        let scaling_exp = 24;

        let a = crate::host::HostFloat64Tensor::from_raw_plc(
            array![1.0, 1.0, -1.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            alice.clone(),
        );

        let x = crate::host::HostFloat64Tensor::from_raw_plc(
            array![1.0, 2.0, 3.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            alice.clone(),
        );
        let y = crate::host::HostFloat64Tensor::from_raw_plc(
            array![4.0, 5.0, 6.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            bob.clone(),
        );

        let a = alice.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &a);
        let a_shared = rep.share(&sess, &setup, &a);

        let x = alice.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &x);
        let x_shared = rep.share(&sess, &setup, &x);

        let y = bob.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &y);
        let y_shared = rep.share(&sess, &setup, &y);

        // simulate to a less than zero calculation to get some good values
        // to pass into if else
        let msb: AbstractReplicatedRingTensor<_> = rep.msb(&sess, &setup, &a_shared);
        let ones: AbstractReplicatedRingTensor<_> =
            rep.fill(&sess, 1u64.into(), &rep.shape(&sess, &a_shared));
        let s = rep.sub(&sess, &ones, &msb);

        let res = rep.if_else(&sess, &s, &x_shared, &y_shared);

        let opened_result = alice.reveal(&sess, &res);
        let decoded_result =
            alice.fixedpoint_ring_decode(&sess, scaling_base, scaling_exp, &opened_result);

        assert_eq!(
            decoded_result,
            crate::host::HostFloat64Tensor::from_raw_plc(
                array![1.0, 2.0, 6.0]
                    .into_dimensionality::<IxDyn>()
                    .unwrap(),
                alice
            )
        );
    }
}
