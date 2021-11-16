use crate::computation::{CanonicalType, KnownType, Placed, RepIfElseOp, ReplicatedPlacement};
use crate::error::Result;
use crate::kernels::*;
use crate::replicated::{
    Mirrored3RingTensor, ReplicatedRing128Tensor, ReplicatedRing64Tensor, Underlying,
};

modelled!(PlacementIfElse::if_else, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepIfElseOp);
modelled!(PlacementIfElse::if_else, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepIfElseOp);
// modelled!(PlacementIfElse::if_else, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepIfElseOp);
// modelled!(PlacementIfElse::if_else, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepIfElseOp);

kernel! {
    RepIfElseOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor  => [transparent] Self::rep_kernel),
        // (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_kernel),
        // (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor  => [transparent] Self::rep_kernel),
    ]
}

impl RepIfElseOp {
    fn rep_kernel<S: Session, RepRingT, ShapeT, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        s: RepRingT,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepRingT>
    where
        RepRingT: Underlying<TensorType = HostRingT>,
        Mirrored3RingTensor<HostRingT>: Underlying<TensorType = HostRingT>,
        Mirrored3RingTensor<HostRingT>: CanonicalType,
        <Mirrored3RingTensor<HostRingT> as CanonicalType>::Type: KnownType<S>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, m!(c!(Mirrored3RingTensor<HostRingT>))>,
        ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShape<S, RepRingT, ShapeT>,
        ReplicatedPlacement:
            PlacementSub<S, m!(c!(Mirrored3RingTensor<HostRingT>)), RepRingT, RepRingT>,
    {
        let ones = rep.fill(sess, 1u64.into(), &rep.shape(sess, &x));

        // if else [s] * [x] + (1 - [s]) * [y]
        let s_x = rep.mul(sess, &s, &x);

        let ones_minus_s = rep.sub(sess, &ones, &s);
        let ones_s_y = rep.mul(sess, &ones_minus_s, &y);

        Ok(rep.add(sess, &s_x, &ones_s_y))
    }
}

#[cfg(test)]
mod tests {
    use crate::computation::{HostPlacement, ReplicatedPlacement};
    use crate::host::FromRawPlc;
    use crate::kernels::*;
    use crate::replicated::AbstractReplicatedRingTensor;
    use ndarray::{array, IxDyn};

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
        let a_shared = rep.share(&sess, &a);

        let x = alice.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &x);
        let x_shared = rep.share(&sess, &x);

        let y = bob.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &y);
        let y_shared = rep.share(&sess, &y);

        // simulate to a less than zero calculation to get some good values
        // to pass into if else
        let msb: AbstractReplicatedRingTensor<_> = rep.msb(&sess, &a_shared);
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
