use crate::computation::*;
use crate::error::Result;
use crate::host::AbstractHostFixedTensor;
use crate::kernels::*;
use crate::replicated::AbstractMirroredFixedTensor;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mirrored3Tensor<HostRingT> {
    pub values: [HostRingT; 3],
}

impl<HostTenT> Placed for Mirrored3Tensor<HostTenT>
where
    HostTenT: Placed<Placement = HostPlacement>,
{
    type Placement = Mirrored3Placement;

    fn placement(&self) -> Result<Self::Placement> {
        // put a match in here depending on the placement
        //
        let Mirrored3Tensor {
            values: [x0, x1, x2],
        } = self;

        let owner0 = x0.placement()?.owner;
        let owner1 = x1.placement()?.owner;
        let owner2 = x2.placement()?.owner;

        let owners = [owner0, owner1, owner2];

        Ok(Mirrored3Placement { owners })
    }
}

impl IdentityOp {
    pub(crate) fn host_mir3_kernel<S: Session, HostT>(
        sess: &S,
        plc: &HostPlacement,
        x: Mirrored3Tensor<HostT>,
    ) -> Result<HostT>
    where
        HostT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementIdentity<S, HostT, HostT>,
    {
        let mir_plc = x.placement()?;
        let (player0, player1, _player2) = mir_plc.host_placements();

        let Mirrored3Tensor {
            values: [x0, x1, x2],
        } = &x;

        let x_plc = match () {
            _ if *plc == player0 => x0,
            _ if *plc == player1 => x1,
            _ => x2, // we send it to player2 in case there's no one else to place the value on
                     // last case we re
        };

        // TODO(Dragos)
        // remove identity op; call identity at default case
        Ok(plc.identity(sess, x_plc))
    }

    pub(crate) fn host_mir3_fixed_kernel<S: Session, MirRingT, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractMirroredFixedTensor<MirRingT>,
    ) -> Result<AbstractHostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementIdentity<S, MirRingT, HostRingT>,
    {
        let x_id = plc.identity(sess, &x.tensor);
        Ok(AbstractHostFixedTensor {
            tensor: x_id,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}
impl GatherOp {
    pub(crate) fn kernel<S: Session, R: Clone>(
        sess: &S,
        receiver: &HostPlacement,
        xe: Mirrored3Tensor<R>,
    ) -> Result<R>
    where
        HostPlacement: PlacementPlace<S, R>,
        R: Placed<Placement = HostPlacement>,
    {
        let Mirrored3Tensor {
            values: [x0, x1, x2],
        } = xe.clone();

        let mir_plc = xe.placement()?;
        let (player0, player1, _player2) = &mir_plc.host_placements();

        let res = match () {
            _ if receiver == player0 => x0,
            _ if receiver == player1 => x1,
            _ => x2,
            // we send it to player2 in case there's no one else to place the value on
        };

        Ok(receiver.place(sess, res))
    }

    pub(crate) fn fixed_kernel<S: Session, MirRingT, HostRingT>(
        sess: &S,
        receiver: &HostPlacement,
        xe: AbstractMirroredFixedTensor<MirRingT>,
    ) -> Result<AbstractHostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementGather<S, MirRingT, HostRingT>,
    {
        let x = receiver.gather(sess, &xe.tensor);
        Ok(AbstractHostFixedTensor {
            tensor: x,
            fractional_precision: xe.fractional_precision,
            integral_precision: xe.integral_precision,
        })
    }
}

// impl FixedpointEncodeOp {
//     pub(crate) fn mir_fixed_lower_kernel<S: Session, MirFloatT, MirRingT>(
//         sess: &S,
//         plc: &Mirrored3Placement,
//         fractional_precision: u32,
//         integral_precision: u32,
//         x: MirFloatT,
//     ) -> Result<AbstractMirroredFixedTensor<MirRingT>>
//     where
//         Mirrored3Placement: PlacementRingFixedpointEncode<S, MirFloatT, MirRingT>,
//     {
//         let y = plc.fixedpoint_ring_encode(sess, 2, fractional_precision, &x);
//         Ok(AbstractMirroredFixedTensor {
//             tensor: y,
//             fractional_precision,
//             integral_precision,
//         })
//     }
// }

impl RingFixedpointEncodeOp {
    pub(crate) fn mir_kernel<S: Session, HostFloatT, HostRingT>(
        sess: &S,
        plc: &Mirrored3Placement,
        scaling_base: u64,
        scaling_exp: u32,
        x: Mirrored3Tensor<HostFloatT>,
    ) -> Result<Mirrored3Tensor<HostRingT>>
    where
        HostPlacement: PlacementRingFixedpointEncode<S, HostFloatT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();

        let Mirrored3Tensor {
            values: [x0, x1, x2],
        } = &x;

        let y0 = player0.fixedpoint_ring_encode(sess, scaling_base, scaling_exp, x0);
        let y1 = player1.fixedpoint_ring_encode(sess, scaling_base, scaling_exp, x1);
        let y2 = player2.fixedpoint_ring_encode(sess, scaling_base, scaling_exp, x2);

        Ok(Mirrored3Tensor {
            values: [y0, y1, y2],
        })
    }
}

impl RingFixedpointDecodeOp {
    pub(crate) fn mir_kernel<S: Session, HostRingT, HostFloatT>(
        sess: &S,
        plc: &Mirrored3Placement,
        scaling_base: u64,
        scaling_exp: u32,
        x: Mirrored3Tensor<HostRingT>,
    ) -> Result<Mirrored3Tensor<HostFloatT>>
    where
        HostPlacement: PlacementRingFixedpointDecode<S, HostRingT, HostFloatT>,
    {
        let (player0, player1, player2) = plc.host_placements();

        let Mirrored3Tensor {
            values: [x0, x1, x2],
        } = &x;

        let y0 = player0.fixedpoint_ring_decode(sess, scaling_base, scaling_exp, x0);
        let y1 = player1.fixedpoint_ring_decode(sess, scaling_base, scaling_exp, x1);
        let y2 = player2.fixedpoint_ring_decode(sess, scaling_base, scaling_exp, x2);

        Ok(Mirrored3Tensor {
            values: [y0, y1, y2],
        })
    }
}
