use super::*;
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::replicated::ReplicatedPlacement;

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

impl AddOp {
    pub(crate) fn ring64_host_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: Z64Tensor<HostT, RepT>,
        y: Z64Tensor<HostT, RepT>,
    ) -> Result<Z64Tensor<HostT, RepT>>
    where
        HostPlacement: PlacementPlace<S, HostT>,
        HostPlacement: PlacementReveal<S, RepT, HostT>,
        HostPlacement: PlacementAdd<S, HostT, HostT, HostT>,
    {
        let x = match x {
            Z64Tensor::Host(v) => plc.place(sess, v),
            Z64Tensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            Z64Tensor::Host(v) => plc.place(sess, v),
            Z64Tensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let z = plc.add(sess, &x, &y);
        Ok(Z64Tensor::Host(z))
    }

    pub(crate) fn ring64_rep_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: Z64Tensor<HostT, RepT>,
        y: Z64Tensor<HostT, RepT>,
    ) -> Result<Z64Tensor<HostT, RepT>>
    where
        ReplicatedPlacement: PlacementPlace<S, RepT>,
        ReplicatedPlacement: PlacementShare<S, HostT, RepT>,
        ReplicatedPlacement: PlacementAdd<S, RepT, RepT, RepT>,
    {
        let x = match x {
            Z64Tensor::Host(v) => plc.share(sess, &v),
            Z64Tensor::Replicated(v) => v,
        };
        let y = match y {
            Z64Tensor::Host(v) => plc.share(sess, &v),
            Z64Tensor::Replicated(v) => v,
        };

        let z = plc.add(sess, &x, &y);

        Ok(Z64Tensor::Replicated(z))
    }
}
