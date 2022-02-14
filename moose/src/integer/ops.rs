use super::*;
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::mirrored::Mirrored3Placement;
use crate::ring::Z64Tensor;
use crate::types::HostString;

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

impl SaveOp {
    pub fn u64_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        key: m!(HostString),
        x: U64Tensor<HostT, RepT>,
    ) -> Result<m!(Unit)>
    where
        HostString: KnownType<S>,
        Unit: KnownType<S>,
        HostPlacement: PlacementSave<S, m!(HostString), HostT, m!(Unit)>,
    {
        let x = match x {
            U64Tensor::Replicated(_v) => unimplemented!(),
            U64Tensor::Host(v) => v,
        };
        Ok(plc.save(sess, &key, &x))
    }
}

impl IdentityOp {
    pub(crate) fn u64_host_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: U64Tensor<HostT, RepT>,
    ) -> Result<U64Tensor<HostT, RepT>>
    where
        HostPlacement: PlacementPlace<S, HostT>,
        HostPlacement: PlacementReveal<S, RepT, HostT>,
    {
        let x = match x {
            U64Tensor::Host(v) => plc.place(sess, v),
            U64Tensor::Replicated(v) => plc.reveal(sess, &v),
        };
        Ok(U64Tensor::Host(x))
    }
}

impl CastOp {
    pub(crate) fn uint64_ring64_host_kernel<
        S: Session,
        HostUint64T,
        RepUintT,
        HostRing64T,
        RepRingT,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: U64Tensor<HostUint64T, RepUintT>,
    ) -> Result<Z64Tensor<HostRing64T, RepRingT>>
    where
        HostPlacement: PlacementReveal<S, RepUintT, HostUint64T>,
        HostPlacement: PlacementCast<S, HostUint64T, HostRing64T>,
    {
        let x = match x {
            U64Tensor::Host(v) => v,
            U64Tensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let z = plc.cast(sess, &x);
        Ok(Z64Tensor::Host(z))
    }

    pub(crate) fn ring64_uint64_host_kernel<
        S: Session,
        HostUint64T,
        RepUintT,
        HostRing64T,
        RepRingT,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: Z64Tensor<HostRing64T, RepRingT>,
    ) -> Result<U64Tensor<HostUint64T, RepUintT>>
    where
        HostPlacement: PlacementReveal<S, RepRingT, HostRing64T>,
        HostPlacement: PlacementCast<S, HostRing64T, HostUint64T>,
    {
        let x = match x {
            Z64Tensor::Host(v) => v,
            Z64Tensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let z = plc.cast(sess, &x);
        Ok(U64Tensor::Host(z))
    }
}
