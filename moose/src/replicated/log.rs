//use macros::with_context;
use crate::computation::{CanonicalType, KnownType, RepEqualOp, RepIfElseOp, ReplicatedPlacement, Constant};
use crate::kernels::*;
use crate::replicated::{RepTen, ReplicatedRing64Tensor, ReplicatedRing128Tensor, ReplicatedBitTensor};
use crate::{Const, Ring};
use crate::computation::{Placed, HostPlacement};

modelled!(PlacementEqual::equal, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor, RepEqualOp);
modelled!(PlacementEqual::equal, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor, RepEqualOp);

kernel! {
    RepEqualOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_kernel),
    ]
}

impl RepEqualOp {
    fn rep_kernel<S: Session, RingT, RingBitT, BitArrayT, ShapeT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<RingT>,
        y: RepTen<RingT>,
    ) -> RingBitT
    where
        RepTen<RingT>: Into<st!(RepTen<RingT>)>,
        RepTen<RingT>: CanonicalType,
        <RepTen<RingT> as CanonicalType>::Type: KnownType<S>,

        RingT: Ring<BitLength = N>,

        RingT: Tensor<S>,
        RingT::Scalar: Into<Constant>,
        RingT::Scalar: From<u8>,

        ReplicatedPlacement:
            PlacementBitDecSetup<S, S::ReplicatedSetup, st!(RepTen<RingT>), BitArrayT>,
        ReplicatedPlacement:
            PlacementSub<S, st!(RepTen<RingT>), st!(RepTen<RingT>), st!(RepTen<RingT>)>,
        ReplicatedPlacement:
            PlacementXor<S, RingBitT, RingBitT, RingBitT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RingBitT>,
            ReplicatedPlacement: PlacementShape<S, st!(RepTen<RingT>), ShapeT>,
        ReplicatedPlacement: PlacementIndex<S, BitArrayT, RingBitT>,
        ReplicatedPlacement: PlacementMulSetup<
            S,
            S::ReplicatedSetup,
            RingBitT,
            RingBitT,
            RingBitT,
        >,
        ReplicatedPlacement: PlacementXor<S, RingBitT, RingBitT, RingBitT>,
        BitArrayT: std::fmt::Debug,
        RingBitT: std::fmt::Debug,
        ShapeT: std::fmt::Debug,
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        HostPlacement: PlacementReveal<S, RingBitT, RingBitT>,
    {
        let setup = rep.gen_setup(sess);

        let z = rep.sub(sess, &x.into(), &y.into());

        let bits = rep.bit_decompose(sess, &setup, &z);

        let v: Vec<_> = (0..RingT::BitLength::VALUE).map(|i| rep.index(sess, i, &bits)).collect();

        let one_r: Constant = 1u8.into();
        let ones = rep.fill(sess, one_r, &rep.shape(sess, &z));

        let v_not: Vec<_> = v.iter().map(|vi| rep.xor(sess, &ones, vi)).collect();

        let res = v_not.iter().fold(ones, |acc, y| rep.mul_setup(sess, &setup, &acc, &y));

        let (alice, bob, carole) = rep.host_placements();

        alice.reveal(sess, &res);

        res

    }
}

impl RepIfElseOp {
    fn rep_kernel<S: Session, SetupT, RingT, RingBitT, BitArrayT, ShapeT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        setup: SetupT,
        v: RingBitT,
        x: RingT,
        y: RingT,
    ) {

    }

}

mod tests {
    use crate::computation::{HostPlacement, ReplicatedPlacement};
    use crate::host::{FromRawPlc, HostBitTensor};
    use ndarray::{array, IxDyn};
    use crate::kernels::*;
    use crate::replicated::ReplicatedBitTensor;

    #[test]
    fn test_equal() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };

        let bob= HostPlacement {
            owner: "bob".into(),
        };

        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let sess = SyncSession::default();
        let setup = rep.gen_setup(&sess);

        //sess.replicated_keys.insert(rep, setup);

        let scaling_base = 2;
        let scaling_exp = 24;

        let x = crate::host::HostFloat64Tensor::from_raw_plc(
            array![5.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            alice.clone(),
        );
        let y= crate::host::HostFloat64Tensor::from_raw_plc(
            array![4.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            bob.clone(),
        );
        let x = alice.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &x);
        let x_shared = rep.share(&sess, &setup, &x);

        let y = bob.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &y);
        let y_shared = rep.share(&sess, &setup, &y);

        let res: ReplicatedBitTensor = rep.equal(&sess, &x_shared, &y_shared);

        let opened_result = alice.reveal(&sess, &res);
        assert_eq!(opened_result, HostBitTensor::from_raw_plc(array![1].into_dimensionality::<IxDyn>().unwrap(), alice));
    }
}
