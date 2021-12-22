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
        };

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

impl FixedpointEncodeOp {
    pub(crate) fn host_mir_fixed_kernel<S: Session, HostFloatT, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        fractional_precision: u32,
        integral_precision: u32,
        x: Mirrored3Tensor<HostFloatT>,
    ) -> Result<AbstractMirroredFixedTensor<HostRingT>>
    where
        HostFloatT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementRingFixedpointEncode<S, HostFloatT, HostRingT>,
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
        };

        let y = plc.fixedpoint_ring_encode(sess, 2, fractional_precision, &x_plc);
        Ok(AbstractMirroredFixedTensor {
            tensor: y,
            fractional_precision,
            integral_precision,
        })
    }
}
