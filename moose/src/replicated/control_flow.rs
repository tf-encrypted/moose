use crate::computation::{MuxOp, ReplicatedPlacement};
use crate::error::Result;
use crate::kernels::*;
use crate::replicated::{ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing64Tensor};

modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, MuxOp);
// modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, MuxOp);
// modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, MuxOp);

impl MuxOp {
    pub(crate) fn rep_kernel<S: Session, RepRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        s: RepRingT,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepRingT>
    where
        ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
    {
        // [s] * ([x] - [y]) + [y] <=> if s=1 choose x, otherwise y
        let diff = rep.sub(sess, &x, &y);
        let s_diff = rep.mul(sess, &s, &diff);

        Ok(rep.add(sess, &s_diff, &y))
    }

    pub(crate) fn rep_bit_selector_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        s: RepBitT,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepRingT>
    where
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
        ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
    {
        let s_ring = rep.ring_inject(sess, 0, &s);
        // [s] * ([x] - [y]) + [y] <=> if s=1 choose x, otherwise y
        let diff = rep.sub(sess, &x, &y);
        let s_diff = rep.mul(sess, &s_ring, &diff);

        Ok(rep.add(sess, &s_diff, &y))
    }
}

#[cfg(test)]
mod tests {
    use crate::computation::{HostPlacement, ReplicatedPlacement};
    use crate::host::FromRawPlc;
    use crate::kernels::*;
    use crate::replicated::ReplicatedRing128Tensor;
    use ndarray::{array, IxDyn};

    #[test]
    fn test_mux() {
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
        let a_shared: ReplicatedRing128Tensor = rep.share(&sess, &a);

        let x = alice.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &x);
        let x_shared = rep.share(&sess, &x);

        let y = bob.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &y);
        let y_shared = rep.share(&sess, &y);

        // simulate to a less than zero calculation to get some good values
        // to pass into if else
        let msb: ReplicatedRing128Tensor = rep.msb(&sess, &a_shared);
        let ones: ReplicatedRing128Tensor =
            rep.fill(&sess, 1u64.into(), &rep.shape(&sess, &a_shared));
        let s = rep.sub(&sess, &ones, &msb);

        let res = rep.mux(&sess, &s, &x_shared, &y_shared);

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

    #[test]
    fn test_bit_selector_mux() {
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

        let s = crate::host::HostBitTensor::from_raw_plc(
            array![1, 0, 1, 0].into_dimensionality::<IxDyn>().unwrap(),
            alice.clone(),
        );

        let x = crate::host::HostFloat64Tensor::from_raw_plc(
            array![1.0, 2.0, 3.0, 4.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            alice.clone(),
        );
        let y = crate::host::HostFloat64Tensor::from_raw_plc(
            array![4.0, 5.0, 6.0, -1.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            bob.clone(),
        );

        let s_shared: crate::replicated::ReplicatedBitTensor = rep.share(&sess, &s);

        let x = alice.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &x);
        let x_shared = rep.share(&sess, &x);

        let y = bob.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &y);
        let y_shared = rep.share(&sess, &y);

        let res = rep.mux(&sess, &s_shared, &x_shared, &y_shared);

        let opened_result = alice.reveal(&sess, &res);
        let decoded_result =
            alice.fixedpoint_ring_decode(&sess, scaling_base, scaling_exp, &opened_result);

        assert_eq!(
            decoded_result,
            crate::host::HostFloat64Tensor::from_raw_plc(
                array![1.0, 5.0, 3.0, -1.0]
                    .into_dimensionality::<IxDyn>()
                    .unwrap(),
                alice
            )
        );
    }
}
