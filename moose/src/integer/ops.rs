use super::*;
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::mirrored::Mirrored3Placement;
use crate::replicated::*;
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

impl AddOp {
    pub(crate) fn u64_host_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: U64Tensor<HostT, RepT>,
        y: U64Tensor<HostT, RepT>,
    ) -> Result<U64Tensor<HostT, RepT>>
    where
        HostPlacement: PlacementReveal<S, RepT, HostT>,
        HostPlacement: PlacementAdd<S, HostT, HostT, HostT>,
    {
        let x = match x {
            U64Tensor::Host(v) => v,
            U64Tensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            U64Tensor::Host(v) => v,
            U64Tensor::Replicated(v) => plc.reveal(sess, &v),
        };

        let z = plc.add(sess, &x, &y);
        Ok(U64Tensor::Host(z))
    }

    pub(crate) fn u64_rep_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: U64Tensor<HostT, RepT>,
        y: U64Tensor<HostT, RepT>,
    ) -> Result<U64Tensor<HostT, RepT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostT, RepT>,
        ReplicatedPlacement: PlacementAdd<S, RepT, RepT, RepT>,
    {
        let x = match x {
            U64Tensor::Host(v) => plc.share(sess, &v),
            U64Tensor::Replicated(v) => v,
        };
        let y = match y {
            U64Tensor::Host(v) => plc.share(sess, &v),
            U64Tensor::Replicated(v) => v,
        };
        let z = plc.add(sess, &x, &y);
        Ok(U64Tensor::Replicated(z))
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
    pub(crate) fn u64_host_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: U64Tensor<HostT, RepT>,
    ) -> Result<U64Tensor<HostT, RepT>>
    where
        HostPlacement: PlacementReveal<S, RepT, HostT>,
        HostPlacement: PlacementCast<S, HostT, HostT>,
    {
        let x = match x {
            U64Tensor::Host(v) => v,
            U64Tensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let z = plc.cast(sess, &x);
        Ok(U64Tensor::Host(z))
    }
}
