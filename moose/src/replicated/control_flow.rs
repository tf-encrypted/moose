//! Support for control-flow operators

use super::*;
use crate::computation::MuxOp;
use crate::error::Result;
use crate::execution::Session;
use moose_macros::with_context;

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
        let res = with_context!(rep, sess, s * (x - y) + y);
        Ok(res)
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
        let res = with_context!(rep, sess, s_ring * (x - y) + y);
        Ok(res)
    }
}

#[cfg(feature = "sync_execute")]
#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use ndarray::prelude::*;

    #[test]
    fn test_mux() {
        let alice = HostPlacement::from("alice");
        let bob = HostPlacement::from("bob");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        let sess = SyncSession::default();

        let scaling_base = 2;
        let scaling_exp = 24;

        let a: HostFloat64Tensor = alice.from_raw(array![1.0, 1.0, -1.0]);
        let x: HostFloat64Tensor = alice.from_raw(array![1.0, 2.0, 3.0]);
        let y: HostFloat64Tensor = bob.from_raw(array![4.0, 5.0, 6.0]);

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

        assert_eq!(decoded_result, alice.from_raw(array![1.0, 2.0, 6.0]));
    }

    #[test]
    fn test_bit_selector_mux() {
        let alice = HostPlacement::from("alice");
        let bob = HostPlacement::from("bob");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        let sess = SyncSession::default();

        let scaling_base = 2;
        let scaling_exp = 24;

        let s: HostBitTensor = alice.from_raw(array![1, 0, 1, 0]);
        let x: HostFloat64Tensor = alice.from_raw(array![1.0, 2.0, 3.0, 4.0]);
        let y: HostFloat64Tensor = bob.from_raw(array![4.0, 5.0, 6.0, -1.0]);

        let s_shared: ReplicatedBitTensor = rep.share(&sess, &s);

        let x = alice.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &x);
        let x_shared = rep.share(&sess, &x);

        let y = bob.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &y);
        let y_shared = rep.share(&sess, &y);

        let res = rep.mux(&sess, &s_shared, &x_shared, &y_shared);

        let opened_result = alice.reveal(&sess, &res);
        let decoded_result =
            alice.fixedpoint_ring_decode(&sess, scaling_base, scaling_exp, &opened_result);

        assert_eq!(decoded_result, alice.from_raw(array![1.0, 5.0, 3.0, -1.0]));
    }
}
