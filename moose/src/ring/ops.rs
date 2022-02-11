use super::*;
use crate::execution::Session;
use crate::host::HostPlacement;

impl IdentityOp {
    pub(crate) fn ring64_host_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: Z64Tensor<HostT, RepT>,
    ) -> Result<Z64Tensor<HostT, RepT>>
    where
        HostPlacement: PlacementPlace<S, HostT>,
        HostPlacement: PlacementReveal<S, RepT, HostT>,
    {
        let x = match x {
            Z64Tensor::Host(v) => plc.place(sess, v),
            Z64Tensor::Replicated(v) => plc.reveal(sess, &v),
        };
        Ok(Z64Tensor::Host(x))
    }
}
