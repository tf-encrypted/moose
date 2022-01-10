use super::*;
use crate::computation::{AdditivePlacement, HostPlacement, Placed, RepToAdtOp};
use crate::error::Result;
use crate::kernels::*;
use crate::replicated::{
    AbstractReplicatedRingTensor, ReplicatedBitTensor, ReplicatedRing128Tensor,
    ReplicatedRing64Tensor,
};
use macros::with_context;

modelled_kernel! {
    PlacementRepToAdt::rep_to_adt, RepToAdtOp,
    [
        (AdditivePlacement, (ReplicatedRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::rep_to_adt_kernel),
        (AdditivePlacement, (ReplicatedRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::rep_to_adt_kernel),
        (AdditivePlacement, (ReplicatedBitTensor) -> AdditiveBitTensor => [concrete] Self::rep_to_adt_kernel),
    ]
}

impl RepToAdtOp {
    fn rep_to_adt_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AbstractReplicatedRingTensor<HostRingT>,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
        AdditivePlacement: PlacementPlace<S, AdtTensor<HostRingT>>,
    {
        let (adt_player0, adt_player1) = adt.host_placements();
        let (rep_player0, rep_player1, rep_player2) = x.placement()?.host_placements();

        let AbstractReplicatedRingTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let shares = match () {
            _ if adt_player0 == rep_player0 => {
                let y0 = with_context!(rep_player0, sess, x00 + x10);
                let y1 = match () {
                    _ if adt_player1 == rep_player1 => x21,
                    _ if adt_player1 == rep_player2 => x22,
                    _ => x21,
                };
                [y0, y1]
            }
            _ if adt_player0 == rep_player1 => {
                let y0 = with_context!(rep_player1, sess, x11 + x21);
                let y1 = match () {
                    _ if adt_player1 == rep_player2 => x02,
                    _ if adt_player1 == rep_player0 => x00,
                    _ => x02,
                };
                [y0, y1]
            }
            _ if adt_player0 == rep_player2 => {
                let y0 = with_context!(rep_player2, sess, x22 + x02);
                let y1 = match () {
                    _ if adt_player1 == rep_player0 => x10,
                    _ if adt_player1 == rep_player1 => x11,
                    _ => x10,
                };
                [y0, y1]
            }
            _ if adt_player1 == rep_player0 => {
                let y0 = x21;
                let y1 = with_context!(rep_player0, sess, x00 + x10);
                [y0, y1]
            }
            _ if adt_player1 == rep_player1 => {
                let y0 = x02;
                let y1 = with_context!(rep_player1, sess, x11 + x21);
                [y0, y1]
            }
            _ if adt_player1 == rep_player2 => {
                let y0 = x10;
                let y1 = with_context!(rep_player2, sess, x22 + x02);
                [y0, y1]
            }
            _ => {
                let y0 = with_context!(rep_player0, sess, x00 + x10);
                let y1 = x21;
                [y0, y1]
            }
        };
        Ok(adt.place(sess, AdtTensor { shares }))
    }
}
