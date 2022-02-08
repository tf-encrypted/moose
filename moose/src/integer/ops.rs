use super::*;
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::mirrored::Mirrored3Placement;

impl ConstantOp {
    pub(crate) fn u64_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        value: Constant,
    ) -> Result<U64Tensor<HostT, RepT>>
    where
        HostPlacement: PlacementConstant<S, HostT>,
    {
        let z = plc.constant(sess, value);
        Ok(U64Tensor::Host(z))
    }
}

impl ConstantOp {
    pub(crate) fn mir3_u64_kernel<S: Session, HostT, RepT>(
        _sess: &S,
        _plc: &Mirrored3Placement,
        _value: Constant,
    ) -> Result<U64Tensor<HostT, RepT>>
    where
        HostPlacement: PlacementConstant<S, HostT>,
    {
        unimplemented!()
    }
}
