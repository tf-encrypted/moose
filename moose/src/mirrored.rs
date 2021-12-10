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
        let plc3 = x.placement()?;
        let (player0, player1, player2) = plc3.host_placements();
        let Mirrored3Tensor {
            values: [x0, x1, x2],
        } = &x;

        Ok(plc.identity(sess, &x0))
    }
}
