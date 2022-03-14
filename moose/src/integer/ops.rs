use super::*;
use crate::error::{Error, Result};
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::types::HostString;

impl ConstantOp {
    pub(crate) fn u64_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        value: Constant,
    ) -> Result<AbstractUint64Tensor<HostT, RepT>>
    where
        HostPlacement: PlacementConstant<S, HostT>,
    {
        let z = plc.constant(sess, value);
        Ok(AbstractUint64Tensor::Host(z))
    }
}

impl SaveOp {
    pub fn u64_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        key: m!(HostString),
        x: AbstractUint64Tensor<HostT, RepT>,
    ) -> Result<m!(HostUnit)>
    where
        HostString: KnownType<S>,
        HostUnit: KnownType<S>,
        HostPlacement: PlacementSave<S, m!(HostString), HostT, m!(HostUnit)>,
    {
        let x = match x {
            AbstractUint64Tensor::Replicated(_v) => {
                return Err(Error::UnimplementedOperator(
                    "SaveOp not implemented for ReplicatedUint64Tensor on a host placement"
                        .to_string(),
                ));
            }
            AbstractUint64Tensor::Host(v) => v,
        };
        Ok(plc.save(sess, &key, &x))
    }
}

impl IdentityOp {
    pub(crate) fn u64_host_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractUint64Tensor<HostT, RepT>,
    ) -> Result<AbstractUint64Tensor<HostT, RepT>>
    where
        HostPlacement: PlacementPlace<S, HostT>,
        HostPlacement: PlacementReveal<S, RepT, HostT>,
    {
        let x = match x {
            AbstractUint64Tensor::Host(v) => plc.place(sess, v),
            AbstractUint64Tensor::Replicated(v) => plc.reveal(sess, &v),
        };
        Ok(AbstractUint64Tensor::Host(x))
    }
}
