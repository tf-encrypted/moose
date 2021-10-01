//use macros::with_context;
use crate::computation::Placed;
use crate::computation::{CanonicalType, Constant, KnownType, RepEqualOp, ReplicatedPlacement};
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
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_kernel),
    ]
}

impl RepEqualOp {
    fn rep_kernel<S: Session, HostRingT, RepBitT, RepBitArrayT, ShapeT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<HostRingT>,
        y: RepTen<HostRingT>,
    ) -> RepBitT
    where
        RepTen<HostRingT>: Into<st!(RepTen<HostRingT>)>,
        RepTen<HostRingT>: CanonicalType,
        <RepTen<HostRingT> as CanonicalType>::Type: KnownType<S>,

        HostRingT: Ring<BitLength = N>,

        HostRingT: Tensor<S>,
        HostRingT::Scalar: Into<Constant>,
        HostRingT::Scalar: From<u8>,

        ReplicatedPlacement:
            PlacementBitDecSetup<S, S::ReplicatedSetup, st!(RepTen<HostRingT>), RepBitArrayT>,
        ReplicatedPlacement:
            PlacementSub<S, st!(RepTen<HostRingT>), st!(RepTen<HostRingT>), st!(RepTen<HostRingT>)>,
        ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepBitT>,
        ReplicatedPlacement: PlacementShape<S, st!(RepTen<HostRingT>), ShapeT>,
        ReplicatedPlacement: PlacementIndex<S, RepBitArrayT, RepBitT>,
        ReplicatedPlacement: PlacementMulSetup<S, S::ReplicatedSetup, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
    {
        let setup = rep.gen_setup(sess);

        let z = rep.sub(sess, &x.into(), &y.into());
        let bits = rep.bit_decompose(sess, &setup, &z);

        let v: Vec<_> = (0..HostRingT::BitLength::VALUE)
            .map(|i| rep.index(sess, i, &bits))
            .collect();

        let one_r: Constant = 1u8.into();
        let ones = rep.fill(sess, one_r, &rep.shape(sess, &z));

        let v_not: Vec<_> = v.iter().map(|vi| rep.xor(sess, &ones, vi)).collect();

        v_not
            .iter()
            .fold(ones, |acc, y| rep.mul_setup(sess, &setup, &acc, y))
    }
}

#[cfg(test)]
mod tests {
    use crate::computation::{HostPlacement, ReplicatedPlacement};
    use crate::host::{AbstractHostRingTensor, HostBitTensor};
    use crate::kernels::*;
    use crate::replicated::ReplicatedBitTensor;
    use ndarray::{array, IxDyn};

    #[test]
    fn test_equal() {
        let alice = HostPlacement {
            owner: "alice".into(),
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
            alice.clone(),
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
}
