use crate::computation::*;
use crate::error::Result;
use crate::kernels::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mirrored3Tensor<HostRingT> {
    pub values: [HostRingT; 3],
}

impl IdentityOp {
    pub(crate) fn host_mir3_float_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        x: Mirrored3Tensor<HostFloatT>,
    ) -> Result<HostFloatT>
    where
        HostFloatT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementIdentity<S, HostFloatT, HostFloatT>,
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
}
