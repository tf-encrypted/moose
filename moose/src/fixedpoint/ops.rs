//! Support for fixed-point arithmetic

use super::FixedTensor;
use crate::boolean::BoolTensor;
use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::Session;
use crate::floatingpoint::FloatTensor;
use crate::host::*;
use crate::integer::AbstractUint64Tensor;
use crate::kernels::*;
use crate::mirrored::*;
use crate::replicated::*;
use crate::types::*;
use macros::with_context;

impl IdentityOp {
    pub(crate) fn fixed_host_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementIdentity<S, HostFixedT, HostFixedT>,
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
        HostPlacement: PlacementPlace<S, HostFixedT>,
    {
        let v = match x {
            FixedTensor::Host(x) => plc.place(sess, x),
            FixedTensor::Mirrored3(x) => plc.demirror(sess, &x),
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };
        Ok(FixedTensor::Host(v))
    }

    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementPlace<S, RepFixedT>,
    {
        let v = match x {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Mirrored3(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => plc.place(sess, x),
        };

        Ok(FixedTensor::Replicated(v))
    }
}

impl FixedpointEncodeOp {
    pub(crate) fn fixed_kernel<
        S: Session,
        HostFloatT,
        HostFixedT,
        RepFixedT,
        MirFloatT,
        MirFixedT,
    >(
        sess: &S,
        plc: &HostPlacement,
        fractional_precision: u32,
        integral_precision: u32,
        x: FloatTensor<HostFloatT, MirFloatT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementFixedpointEncode<S, HostFloatT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFloatT, HostFloatT>,
    {
        let v = match x {
            FloatTensor::Host(x) => x,
            FloatTensor::Mirrored3(x) => plc.demirror(sess, &x),
        };

        Ok(FixedTensor::Host(plc.fixedpoint_encode(
            sess,
            fractional_precision,
            integral_precision,
            &v,
        )))
    }

    pub(crate) fn mir_fixed_kernel<
        S: Session,
        HostFloatT,
        HostFixedT,
        RepFixedT,
        MirFloatT,
        MirFixedT,
    >(
        sess: &S,
        plc: &Mirrored3Placement,
        fractional_precision: u32,
        integral_precision: u32,
        x: FloatTensor<HostFloatT, MirFloatT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        Mirrored3Placement: PlacementFixedpointEncode<S, MirFloatT, MirFixedT>,
        Mirrored3Placement: PlacementMirror<S, HostFloatT, MirFloatT>,
    {
        let v = match x {
            FloatTensor::Mirrored3(x) => x,
            FloatTensor::Host(x) => plc.mirror(sess, &x),
        };

        Ok(FixedTensor::Mirrored3(plc.fixedpoint_encode(
            sess,
            fractional_precision,
            integral_precision,
            &v,
        )))
    }

    pub(crate) fn mir_fixed_lower_kernel<S: Session, MirFloatT, MirRingT>(
        sess: &S,
        plc: &Mirrored3Placement,
        fractional_precision: u32,
        integral_precision: u32,
        x: MirFloatT,
    ) -> Result<MirFixedTensor<MirRingT>>
    where
        Mirrored3Placement: PlacementRingFixedpointEncode<S, MirFloatT, MirRingT>,
    {
        let tensor = plc.fixedpoint_ring_encode(sess, 2, fractional_precision, &x);
        Ok(MirFixedTensor {
            tensor,
            fractional_precision,
            integral_precision,
        })
    }

    pub(crate) fn hostfixed_kernel<S: Session, HostFloatT, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        fractional_precision: u32,
        integral_precision: u32,
        x: HostFloatT,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementRingFixedpointEncode<S, HostFloatT, HostRingT>,
    {
        let y = plc.fixedpoint_ring_encode(sess, 2, fractional_precision, &x);
        Ok(HostFixedTensor {
            tensor: y,
            fractional_precision,
            integral_precision,
        })
    }
}

impl FixedpointDecodeOp {
    pub(crate) fn fixed_kernel<
        S: Session,
        HostFixedT,
        RepFixedT,
        HostFloatT,
        MirFixedT,
        MirFloatT,
    >(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FloatTensor<HostFloatT, MirFloatT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementFixedpointDecode<S, HostFixedT, HostFloatT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
    {
        let v = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        Ok(FloatTensor::Host(
            plc.fixedpoint_decode(sess, precision, &v),
        ))
    }

    pub(crate) fn mir_fixed_kernel<
        S: Session,
        HostFixedT,
        MirFixedT,
        RepFixedT,
        HostFloatT,
        MirFloatT,
    >(
        sess: &S,
        plc: &Mirrored3Placement,
        precision: u32,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FloatTensor<HostFloatT, MirFloatT>>
    where
        Mirrored3Placement: PlacementFixedpointDecode<S, MirFixedT, MirFloatT>,
        Mirrored3Placement: PlacementMirror<S, HostFixedT, MirFixedT>,
        Mirrored3Placement: PlacementReveal<S, RepFixedT, MirFixedT>,
    {
        let v = match x {
            FixedTensor::Mirrored3(v) => v,
            FixedTensor::Host(v) => plc.mirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        Ok(FloatTensor::Mirrored3(
            plc.fixedpoint_decode(sess, precision, &v),
        ))
    }

    pub(crate) fn mir_fixed_lower_kernel<S: Session, MirFloatT, MirRingT>(
        sess: &S,
        plc: &Mirrored3Placement,
        precision: u32,
        x: MirFixedTensor<MirRingT>,
    ) -> Result<MirFloatT>
    where
        Mirrored3Placement: PlacementRingFixedpointDecode<S, MirRingT, MirFloatT>,
    {
        let tensor = plc.fixedpoint_ring_decode(sess, 2, precision, &x.tensor);
        Ok(tensor)
    }

    pub(crate) fn hostfixed_kernel<S: Session, HostRingT, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        precision: u32,
        x: HostFixedTensor<HostRingT>,
    ) -> Result<HostFloatT>
    where
        HostPlacement: PlacementRingFixedpointDecode<S, HostRingT, HostFloatT>,
    {
        assert_eq!(x.fractional_precision, precision);
        Ok(plc.fixedpoint_ring_decode(sess, 2, precision, &x.tensor))
    }
}

impl AbsOp {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementAbs<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.abs(sess, &x);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementAbsAsFixedpoint<S, RepRingT, RepRingT>,
    {
        Ok(RepFixedTensor {
            tensor: plc.abs_as_fixedpoint(sess, &x.tensor),
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}

impl AddOp {
    pub(crate) fn fixed_host_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementAdd<S, HostFixedT, HostFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        let z = plc.add(sess, &x, &y);
        Ok(FixedTensor::Host(z))
    }

    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementAdd<S, RepFixedT, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.add(sess, &x, &y);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: HostFixedTensor<HostRingT>,
        y: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.add(sess, &x.tensor, &y.tensor);
        Ok(HostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    pub(crate) fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.add(sess, &x.tensor, &y.tensor);
        Ok(RepFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    pub(crate) fn repfixed_mirfixed_kernel<S: Session, RepRingT, MirRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
        y: MirFixedTensor<MirRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementAdd<S, RepRingT, MirRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.add(sess, &x.tensor, &y.tensor);
        Ok(RepFixedTensor {
            tensor: z,
            fractional_precision: u32::max(x.fractional_precision, y.fractional_precision),
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    pub(crate) fn mirfixed_repfixed_kernel<S: Session, RepRingT, MirRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: MirFixedTensor<MirRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementAdd<S, MirRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.add(sess, &x.tensor, &y.tensor);
        Ok(RepFixedTensor {
            tensor: z,
            fractional_precision: u32::max(x.fractional_precision, y.fractional_precision),
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }
}

impl SubOp {
    pub(crate) fn fixed_host_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementSub<S, HostFixedT, HostFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        let z = plc.sub(sess, &x, &y);
        Ok(FixedTensor::Host(z))
    }

    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementSub<S, RepFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.sub(sess, &x, &y);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: HostFixedTensor<HostRingT>,
        y: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.sub(sess, &x.tensor, &y.tensor);
        Ok(HostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    pub(crate) fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.sub(sess, &x.tensor, &y.tensor);
        Ok(RepFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}

impl MulOp {
    pub(crate) fn fixed_host_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementMul<S, HostFixedT, HostFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        let z = with_context!(plc, sess, x * y);
        Ok(FixedTensor::Host(z))
    }

    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementMul<S, RepFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = with_context!(plc, sess, x * y);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: HostFixedTensor<HostRingT>,
        y: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementMul<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.mul(sess, &x.tensor, &y.tensor);
        Ok(HostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    pub(crate) fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.mul(sess, &x.tensor, &y.tensor);
        Ok(RepFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    pub(crate) fn repfixed_mirfixed_kernel<S: Session, RepRingT, MirRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
        y: MirFixedTensor<MirRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementMul<S, RepRingT, MirRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.mul(sess, &x.tensor, &y.tensor);
        Ok(RepFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    pub(crate) fn mirfixed_repfixed_kernel<S: Session, RepRingT, MirRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: MirFixedTensor<MirRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementMul<S, MirRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.mul(sess, &x.tensor, &y.tensor);
        Ok(RepFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }
}

impl DivOp {
    pub(crate) fn fixed_host_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDiv<S, HostFixedT, HostFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        let z = plc.div(sess, &x, &y);
        Ok(FixedTensor::Host(z))
    }

    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementDiv<S, RepFixedT, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.div(sess, &x, &y);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: HostFixedTensor<HostRingT>,
        y: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementDiv<S, HostRingT, HostRingT, HostRingT>,
        HostPlacement: PlacementShl<S, HostRingT, HostRingT>,
        HostPlacement: PlacementSign<S, HostRingT, HostRingT>,
        HostPlacement: PlacementMul<S, HostRingT, HostRingT, HostRingT>,
    {
        // Fx(x) / Fx(y) = ((x*2^f)*2^f)/ (y*2^f)
        assert_eq!(x.fractional_precision, y.fractional_precision);

        let sgn_x = plc.sign(sess, &x.tensor);
        let sgn_y = plc.sign(sess, &y.tensor);

        let abs_x = plc.mul(sess, &x.tensor, &sgn_x);
        let abs_y = plc.mul(sess, &y.tensor, &sgn_y);

        let x_upshifted = plc.shl(sess, x.fractional_precision as usize, &abs_x);

        let abs_z: HostRingT = plc.div(sess, &x_upshifted, &abs_y);
        let sgn_z = plc.mul(sess, &sgn_x, &sgn_y);

        Ok(HostFixedTensor {
            tensor: plc.mul(sess, &abs_z, &sgn_z),
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}

impl DotOp {
    pub(crate) fn fixed_on_host_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDot<S, HostFixedT, HostFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
    {
        let x_revealed = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Mirrored3(x) => plc.demirror(sess, &x),
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };
        let y_revealed = match y {
            FixedTensor::Host(x) => x,
            FixedTensor::Mirrored3(x) => plc.demirror(sess, &x),
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let z = plc.dot(sess, &x_revealed, &y_revealed);
        Ok(FixedTensor::Host(z))
    }

    pub(crate) fn fixed_on_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementDot<S, RepFixedT, RepFixedT, RepFixedT>,
    {
        let x_shared = match x {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Mirrored3(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };
        let y_shared = match y {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Mirrored3(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };

        let z = plc.dot(sess, &x_shared, &y_shared);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        x: HostFixedTensor<HostRingT>,
        y: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementDot<S, HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.dot(sess, &x.tensor, &y.tensor);
        Ok(HostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    pub(crate) fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementDot<S, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        let z = plc.dot(sess, &x.tensor, &y.tensor);
        Ok(RepFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision + y.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }
}

impl TruncPrOp {
    pub(crate) fn fixed_host_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        amount: u32,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
        HostPlacement: PlacementTruncPr<S, HostFixedT, HostFixedT>,
    {
        let v = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Mirrored3(x) => plc.demirror(sess, &x),
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let z = plc.trunc_pr(sess, amount, &v);
        Ok(FixedTensor::Host(z))
    }

    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        amount: u32,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementTruncPr<S, RepFixedT, RepFixedT>,
    {
        let v = match x {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Mirrored3(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };

        let z = plc.trunc_pr(sess, amount, &v);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        amount: u32,
        x: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementShr<S, HostRingT, HostRingT>,
    {
        // NOTE(Morten) we assume fixedpoint base is 2 so that truncation becomes (integer) division by 2**precision
        let z = plc.shr(sess, amount as usize, &x.tensor);
        Ok(HostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision - amount,
            integral_precision: x.integral_precision,
        })
    }

    pub(crate) fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        amount: u32,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementTruncPr<S, RepRingT, RepRingT>,
    {
        let z = plc.trunc_pr(sess, amount, &x.tensor);
        Ok(RepFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision - amount,
            integral_precision: x.integral_precision,
        })
    }
}

impl SumOp {
    pub(crate) fn fixed_host_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<usize>,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
        HostPlacement: PlacementSum<S, HostFixedT, HostFixedT>,
    {
        let v = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Mirrored3(x) => plc.demirror(sess, &x),
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let result = plc.sum(sess, axis, &v);
        Ok(FixedTensor::Host(result))
    }

    pub(crate) fn fixed_rep_kernel<S: Session, RingT, MirT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<usize>,
        x: FixedTensor<RingT, MirT, RepT>,
    ) -> Result<FixedTensor<RingT, MirT, RepT>>
    where
        ReplicatedPlacement: PlacementShare<S, RingT, RepT>,
        ReplicatedPlacement: PlacementShare<S, MirT, RepT>,
        ReplicatedPlacement: PlacementSum<S, RepT, RepT>,
    {
        let x_shared = match x {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Mirrored3(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };

        let result = plc.sum(sess, axis, &x_shared);
        Ok(FixedTensor::Replicated(result))
    }

    pub(crate) fn fixed_hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<usize>,
        x: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementSum<S, HostRingT, HostRingT>,
    {
        let z = plc.sum(sess, axis, &x.tensor);
        Ok(HostFixedTensor {
            tensor: z,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    pub(crate) fn fixed_repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<usize>,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementSum<S, RepRingT, RepRingT>,
    {
        let z = plc.sum(sess, axis, &x.tensor);
        Ok(RepFixedTensor {
            tensor: z,
            integral_precision: x.integral_precision,
            fractional_precision: x.fractional_precision,
        })
    }
}

impl ExpandDimsOp {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Vec<usize>,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementExpandDims<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.expand_dims(sess, axis, &x);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn fixed_host_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<usize>,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
        HostPlacement: PlacementExpandDims<S, HostFixedT, HostFixedT>,
    {
        let x = match x {
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Host(v) => v,
        };
        let z = plc.expand_dims(sess, axis, &x);
        Ok(FixedTensor::Host(z))
    }

    pub(crate) fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Vec<usize>,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementExpandDims<S, RepRingT, RepRingT>,
    {
        let y = plc.expand_dims(sess, axis, &x.tensor);
        Ok(RepFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    pub(crate) fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<usize>,
        x: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementExpandDims<S, HostRingT, HostRingT>,
    {
        let y = plc.expand_dims(sess, axis, &x.tensor);
        Ok(HostFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}

impl IndexAxisOp {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        index: usize,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementIndexAxis<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.index_axis(sess, axis, index, &x);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn fixed_host_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
        HostPlacement: PlacementIndexAxis<S, HostFixedT, HostFixedT>,
    {
        let x = match x {
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Host(v) => v,
        };
        let z = plc.index_axis(sess, axis, index, &x);
        Ok(FixedTensor::Host(z))
    }

    pub(crate) fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        index: usize,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementIndexAxis<S, RepRingT, RepRingT>,
    {
        let y = plc.index_axis(sess, axis, index, &x.tensor);
        Ok(RepFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    pub(crate) fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementIndexAxis<S, HostRingT, HostRingT>,
    {
        let y = plc.index_axis(sess, axis, index, &x.tensor);
        Ok(HostFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}

impl ShapeOp {
    pub(crate) fn host_fixed_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT, HostShapeT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<HostShapeT>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
        HostPlacement: PlacementShape<S, HostFixedT, HostShapeT>,
    {
        let x_revealed = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Mirrored3(x) => plc.demirror(sess, &x),
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        Ok(plc.shape(sess, &x_revealed))
    }

    pub(crate) fn rep_fixed_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT, RepShapeT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<RepShapeT>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShape<S, RepFixedT, RepShapeT>,
    {
        let x_shared = match x {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Mirrored3(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };

        Ok(plc.shape(sess, &x_shared))
    }

    pub(crate) fn host_hostfixed_kernel<S: Session, HostRingT, HostShapeT>(
        sess: &S,
        plc: &HostPlacement,
        x: HostFixedTensor<HostRingT>,
    ) -> Result<HostShapeT>
    where
        HostPlacement: PlacementShape<S, HostRingT, HostShapeT>,
    {
        Ok(plc.shape(sess, &x.tensor))
    }
}

impl MeanOp {
    pub(crate) fn fixed_host_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
        HostPlacement: PlacementMean<S, HostFixedT, HostFixedT>,
    {
        let x_revealed = match x {
            FixedTensor::Host(x) => x,
            FixedTensor::Mirrored3(x) => plc.demirror(sess, &x),
            FixedTensor::Replicated(x) => plc.reveal(sess, &x),
        };

        let result = plc.mean(sess, axis, &x_revealed);
        Ok(FixedTensor::Host(result))
    }

    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementMean<S, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
    {
        let x_shared = match x {
            FixedTensor::Host(x) => plc.share(sess, &x),
            FixedTensor::Mirrored3(x) => plc.share(sess, &x),
            FixedTensor::Replicated(x) => x,
        };

        let result = plc.mean(sess, axis, &x_shared);
        Ok(FixedTensor::Replicated(result))
    }

    pub(crate) fn hostfixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementMeanAsFixedpoint<S, HostRingT, HostRingT>,
    {
        let y = plc.mean_as_fixedpoint(sess, axis, 2, x.fractional_precision, &x.tensor);
        Ok(HostFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision * 2,
            integral_precision: x.integral_precision,
        })
    }

    pub(crate) fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementMeanAsFixedpoint<S, RepRingT, RepRingT>,
    {
        let y = plc.mean_as_fixedpoint(sess, axis, 2, x.fractional_precision, &x.tensor);
        Ok(RepFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision * 2,
            integral_precision: x.integral_precision,
        })
    }
}

impl NegOp {
    pub(crate) fn repfixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementNeg<S, RepRingT, RepRingT>,
    {
        let y = plc.neg(sess, &x.tensor);
        Ok(RepFixedTensor {
            tensor: y,
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }
}

impl AddNOp {
    pub(crate) fn fixed_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &HostPlacement,
        xs: &[FixedTensor<HostFixedT, MirFixedT, RepFixedT>],
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementAddN<S, HostFixedT, HostFixedT>,
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostFixedT: Clone,
    {
        if xs.is_empty() {
            Err(Error::InvalidArgument(
                "cannot add_n on empty array of tensors".to_string(),
            ))
        } else {
            let first = &xs[0];
            match first {
                FixedTensor::Host(_) => {
                    let vec: Vec<HostFixedT> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            FixedTensor::Host(x) => (*x).clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(FixedTensor::Host(result))
                }
                FixedTensor::Replicated(_) => {
                    let vec: Vec<HostFixedT> = xs
                        .iter()
                        .map(|t| match t {
                            FixedTensor::Replicated(x) => plc.reveal(sess, x),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    Ok(FixedTensor::Host(plc.add_n(sess, &vec)))
                }
                FixedTensor::Mirrored3(_) => unimplemented!("add_n does not yet support mirrored"),
            }
        }
    }

    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        xs: &[FixedTensor<HostFixedT, MirFixedT, RepFixedT>],
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementAddN<S, RepFixedT, RepFixedT>,
        RepFixedT: Clone,
    {
        let first = &xs[0];
        match first {
            FixedTensor::Host(_) => {
                let vec: Vec<RepFixedT> = xs
                    .iter()
                    .map(|t| match t {
                        FixedTensor::Host(x) => plc.share(sess, x),
                        _ => unimplemented!("mixed types in tensor"),
                    })
                    .collect();
                Ok(FixedTensor::Replicated(plc.add_n(sess, &vec)))
            }
            FixedTensor::Replicated(_) => {
                let vec: Vec<RepFixedT> = xs
                    .iter()
                    .map(|t| match t {
                        FixedTensor::Replicated(x) => (*x).clone(),
                        _ => unimplemented!("mixed types in tensor"),
                    })
                    .collect();
                Ok(FixedTensor::Replicated(plc.add_n(sess, &vec)))
            }
            FixedTensor::Mirrored3(_) => unimplemented!("add_n does not yet support mirrored"),
        }
    }
    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        xs: &[RepFixedTensor<RepRingT>],
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementAddN<S, RepRingT, RepRingT>,
        RepRingT: Clone,
    {
        let fractional_precision = xs[0].fractional_precision;
        let integral_precision = xs
            .iter()
            .fold(xs[0].integral_precision, |a, b| a.max(b.integral_precision));

        assert!(xs
            .iter()
            .all(|x| x.fractional_precision == fractional_precision));

        let zs: Vec<RepRingT> = xs.iter().map(|item| item.tensor.clone()).collect();

        Ok(RepFixedTensor {
            tensor: rep.add_n(sess, &zs),
            fractional_precision,
            integral_precision,
        })
    }

    pub(crate) fn host_fixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        xs: &[HostFixedTensor<HostRingT>],
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementAddN<S, HostRingT, HostRingT>,
        HostRingT: Clone,
    {
        if xs.is_empty() {
            Err(Error::InvalidArgument(
                "cannot add_n on empty array of tensors".to_string(),
            ))
        } else {
            let mut tensors = Vec::new();
            let fractional_precision = xs[0].fractional_precision;
            let integral_precision = xs[0].integral_precision;
            for x in xs.iter() {
                if (x.integral_precision != integral_precision)
                    || (x.fractional_precision != fractional_precision)
                {
                    return Err(Error::InvalidArgument(
                        "precisions of tensors must match for add_n".to_string(),
                    ));
                }
                tensors.push(x.tensor.clone());
            }
            let tensor = plc.add_n(sess, &tensors);
            Ok(HostFixedTensor {
                tensor,
                fractional_precision,
                integral_precision,
            })
        }
    }
}

impl ConcatOp {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: u32,
        xs: &[FixedTensor<HostFixedT, MirFixedT, RepFixedT>],
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementConcatenate<S, RepFixedT, RepFixedT>,
        RepFixedT: Clone,
    {
        let vec: Result<Vec<RepFixedT>> = xs
            .iter()
            .map(|t| match t {
                FixedTensor::Host(x) => Ok(plc.share(sess, x)),
                FixedTensor::Replicated(x) => Ok(x.clone()),
                FixedTensor::Mirrored3(_) => Err(Error::InvalidArgument(
                    "concat does not support mirrored tensors".to_string(),
                )),
            })
            .collect();
        Ok(FixedTensor::Replicated(plc.concatenate(sess, axis, &vec?)))
    }

    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: u32,
        xs: &[RepFixedTensor<RepRingT>],
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementConcatenate<S, RepRingT, RepRingT>,
        RepRingT: Clone,
    {
        if xs.is_empty() {
            Err(Error::InvalidArgument(
                "cannot concat on empty array of tensors".to_string(),
            ))
        } else {
            let mut tensors = Vec::new();
            let fractional_precision = xs[0].fractional_precision;
            let integral_precision = xs[0].integral_precision;
            for x in xs.iter() {
                if x.fractional_precision != fractional_precision {
                    return Err(Error::InvalidArgument(
                        "precisions of tensors must match when concatenating".to_string(),
                    ));
                }
                tensors.push(x.tensor.clone());
            }
            let tensor = plc.concatenate(sess, axis, &tensors);
            Ok(RepFixedTensor {
                tensor,
                fractional_precision,
                integral_precision,
            })
        }
    }
}

impl Pow2Op {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementPow2<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.pow2(sess, &x);
        Ok(FixedTensor::Replicated(z))
    }
}

impl ExpOp {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementExp<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.exp(sess, &x);
        Ok(FixedTensor::Replicated(z))
    }
}

impl SqrtOp {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementSqrt<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.sqrt(sess, &x);
        Ok(FixedTensor::Replicated(z))
    }
}

impl SigmoidOp {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementSigmoid<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.sigmoid(sess, &x);
        Ok(FixedTensor::Replicated(z))
    }
}

impl LessThanOp {
    pub(crate) fn fixed_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT, HostBitT, RepBitT>(
        sess: &S,
        plc: &HostPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<BoolTensor<HostBitT, RepBitT>>
    where
        HostPlacement: PlacementLessThan<S, HostFixedT, HostFixedT, HostBitT>,
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let z = plc.less(sess, &x, &y);
        Ok(BoolTensor::Host(z))
    }

    pub(crate) fn fixed_rep_kernel<
        S: Session,
        HostFixedT,
        MirFixedT,
        RepFixedT,
        HostBitT,
        RepBitT,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<BoolTensor<HostBitT, RepBitT>>
    where
        ReplicatedPlacement: PlacementLessThan<S, RepFixedT, RepFixedT, RepBitT>,
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let z = plc.less(sess, &x, &y);
        Ok(BoolTensor::Replicated(z))
    }

    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementLessThan<S, RepRingT, RepRingT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.less(sess, &x.tensor, &y.tensor))
    }

    pub(crate) fn rep_mir_fixed_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: MirFixedTensor<MirRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementLessThan<S, MirRingT, RepRingT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.less(sess, &x.tensor, &y.tensor))
    }

    pub(crate) fn rep_fixed_mir_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
        y: MirFixedTensor<MirRingT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementLessThan<S, RepRingT, MirRingT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.less(sess, &x.tensor, &y.tensor))
    }
}

impl GreaterThanOp {
    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementGreaterThan<S, RepRingT, RepRingT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.greater_than(sess, &x.tensor, &y.tensor))
    }

    pub(crate) fn rep_mir_fixed_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: MirFixedTensor<MirRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementGreaterThan<S, MirRingT, RepRingT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.greater_than(sess, &x.tensor, &y.tensor))
    }

    pub(crate) fn rep_fixed_mir_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
        y: MirFixedTensor<MirRingT>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementGreaterThan<S, RepRingT, MirRingT, RepBitT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(plc.greater_than(sess, &x.tensor, &y.tensor))
    }
}

impl FillOp {
    pub(crate) fn mir_fixed_kernel<S: Session, MirRingT, ShapeT>(
        sess: &S,
        plc: &Mirrored3Placement,
        value: Constant,
        shape: ShapeT,
        fractional_precision: u32,
        integral_precision: u32,
    ) -> Result<MirFixedTensor<MirRingT>>
    where
        Mirrored3Placement: PlacementFill<S, ShapeT, MirRingT>,
    {
        let filled = plc.fill(sess, value, &shape);
        Ok(MirFixedTensor {
            tensor: filled,
            integral_precision,
            fractional_precision,
        })
    }
}

impl MuxOp {
    pub(crate) fn fixed_rep_kernel<
        S: Session,
        HostFixedT,
        MirFixedT,
        RepFixedT,
        HostBitT,
        RepBitT,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        s: BoolTensor<HostBitT, RepBitT>,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementMux<S, RepBitT, RepFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, HostBitT, RepBitT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
    {
        let s = match s {
            BoolTensor::Host(v) => plc.share(sess, &v),
            BoolTensor::Replicated(v) => v,
        };
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };
        let y = match y {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.mux(sess, &s, &x, &y);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        s: RepRingT,
        x: RepFixedTensor<RepRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementMux<S, RepRingT, RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(RepFixedTensor {
            tensor: plc.mux(sess, &s, &x.tensor, &y.tensor),
            fractional_precision: x.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    pub(crate) fn rep_bit_selector_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        s: m!(ReplicatedBitTensor),
        x: RepFixedTensor<RepRingT>,
        y: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedBitTensor: KnownType<S>,
        ReplicatedPlacement: PlacementMux<S, m!(ReplicatedBitTensor), RepRingT, RepRingT, RepRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(RepFixedTensor {
            tensor: plc.mux(sess, &s, &x.tensor, &y.tensor),
            fractional_precision: x.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }

    pub(crate) fn fixed_host_kernel<
        S: Session,
        HostFixedT,
        MirFixedT,
        RepFixedT,
        HostBitT,
        RepBitT,
    >(
        sess: &S,
        plc: &HostPlacement,
        s: BoolTensor<HostBitT, RepBitT>,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
        y: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        HostPlacement: PlacementReveal<S, RepBitT, HostBitT>,
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
        HostPlacement: PlacementMux<S, HostBitT, HostFixedT, HostFixedT, HostFixedT>,
    {
        let s = match s {
            BoolTensor::Replicated(v) => plc.reveal(sess, &v),
            BoolTensor::Host(v) => v,
        };
        let x = match x {
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Host(v) => v,
        };
        let y = match y {
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Host(v) => v,
        };
        let z = plc.mux(sess, &s, &x, &y);
        Ok(FixedTensor::Host(z))
    }

    pub(crate) fn host_bit_fixed_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &HostPlacement,
        s: m!(HostBitTensor),
        x: HostFixedTensor<HostRingT>,
        y: HostFixedTensor<HostRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostBitTensor: KnownType<S>,
        HostPlacement: PlacementMux<S, m!(HostBitTensor), HostRingT, HostRingT, HostRingT>,
    {
        assert_eq!(x.fractional_precision, y.fractional_precision);
        Ok(HostFixedTensor {
            tensor: plc.mux(sess, &s, &x.tensor, &y.tensor),
            fractional_precision: x.fractional_precision,
            integral_precision: u32::max(x.integral_precision, y.integral_precision),
        })
    }
}

impl MaximumOp {
    pub(crate) fn fixed_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: &[FixedTensor<HostFixedT, MirFixedT, RepFixedT>],
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementMaximum<S, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        RepFixedT: Clone,
    {
        let xv: Vec<RepFixedT> = x
            .iter()
            .map(|item| match item {
                FixedTensor::Host(v) => plc.share(sess, v),
                FixedTensor::Mirrored3(v) => plc.share(sess, v),
                FixedTensor::Replicated(v) => v.clone(),
            })
            .collect();
        let z = plc.maximum(sess, &xv);
        Ok(FixedTensor::Replicated(z))
    }

    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: &[RepFixedTensor<RepRingT>],
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementMaximum<S, RepRingT, RepRingT>,
        RepRingT: Clone,
    {
        // leave it up to the reduce op to identify whethere x is empty.
        let integral_precision = x
            .iter()
            .map(|item| item.integral_precision)
            .reduce(u32::max);
        let integral_precision = match integral_precision {
            Some(v) => v,
            None => {
                return Err(Error::Unexpected(Some(
                    "maximum op had no inputs".to_string(),
                )))
            }
        };

        let fractional_precision = x[0].fractional_precision;
        for item in x.iter() {
            if item.fractional_precision != fractional_precision {
                return Err(Error::InvalidArgument(
                    "maximum op needs all array entries to have same precision".to_string(),
                ));
            };
        }

        let xv: Vec<_> = x
            .iter()
            .map(|item| {
                // TODO(Dragos) can we get rid of this cloning?
                item.tensor.clone()
            })
            .collect();

        Ok(RepFixedTensor {
            tensor: plc.maximum(sess, &xv),
            fractional_precision,
            integral_precision,
        })
    }
}

impl SoftmaxOp {
    pub(crate) fn fixed_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        upmost_index: usize,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementSoftmax<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.softmax(sess, axis, upmost_index, &x);
        Ok(FixedTensor::Replicated(z))
    }
}

impl ArgmaxOp {
    pub(crate) fn fixed_rep_kernel<
        S: Session,
        HostFixedT,
        MirFixedT,
        RepFixedT,
        HostUintT,
        RepUintT,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        upmost_index: usize,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<AbstractUint64Tensor<HostUintT, RepUintT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementArgmax<S, RepFixedT, RepUintT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.argmax(sess, axis, upmost_index, &x);
        Ok(AbstractUint64Tensor::Replicated(z))
    }

    pub(crate) fn fixed_host_kernel<
        S: Session,
        HostFixedT,
        MirFixedT,
        RepFixedT,
        HostUintT,
        RepUintT,
    >(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        upmost_index: usize,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<AbstractUint64Tensor<HostUintT, RepUintT>>
    where
        HostRing64Tensor: KnownType<S>,
        HostPlacement: PlacementReveal<S, RepFixedT, HostFixedT>,
        HostPlacement: PlacementDemirror<S, MirFixedT, HostFixedT>,
        HostPlacement: PlacementArgmax<S, HostFixedT, HostUintT>,
    {
        let x = match x {
            FixedTensor::Host(v) => v,
            FixedTensor::Mirrored3(v) => plc.demirror(sess, &v),
            FixedTensor::Replicated(v) => plc.reveal(sess, &v),
        };

        let z = plc.argmax(sess, axis, upmost_index, &x);
        Ok(AbstractUint64Tensor::Host(z))
    }
}

impl LogOp {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementLog<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.log(sess, &x);
        Ok(FixedTensor::Replicated(z))
    }
}

impl Log2Op {
    pub(crate) fn fixed_rep_kernel<S: Session, HostFixedT, MirFixedT, RepFixedT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: FixedTensor<HostFixedT, MirFixedT, RepFixedT>,
    ) -> Result<FixedTensor<HostFixedT, MirFixedT, RepFixedT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementShare<S, MirFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementLog2<S, RepFixedT, RepFixedT>,
    {
        let x = match x {
            FixedTensor::Host(v) => plc.share(sess, &v),
            FixedTensor::Mirrored3(v) => plc.share(sess, &v),
            FixedTensor::Replicated(v) => v,
        };

        let z = plc.log2(sess, &x);
        Ok(FixedTensor::Replicated(z))
    }
}

#[cfg(feature = "sync_execute")]
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "compile")]
    use crate::execution::symbolic::{Symbolic, SymbolicHandle, SymbolicSession};
    use crate::fixedpoint::PrefixMul;
    use crate::prelude::*;
    use crate::replicated::RepTensor;
    use ndarray::prelude::*;
    use proptest::prelude::*;
    use std::num::Wrapping;

    #[test]
    fn ring_fixedpoint() {
        let plc = HostPlacement::from("TODO");

        let x = plc.from_raw(array![1.0, -2.0, 3.0, -4.0]);

        let scaling_factor = 2u64.pow(16);
        let x_encoded = HostFixed64Tensor {
            tensor: HostRing64Tensor::encode(&x, scaling_factor),
            fractional_precision: 16,
            integral_precision: 5,
        };

        assert_eq!(
            x_encoded,
            HostFixed64Tensor {
                tensor: plc.from_raw(array![
                    65536,
                    18446744073709420544,
                    196608,
                    18446744073709289472
                ]),
                fractional_precision: 16,
                integral_precision: 5,
            }
        );
        let x_decoded = HostRing64Tensor::decode(&x_encoded.tensor, scaling_factor);
        assert_eq!(x_decoded, x);

        let scaling_factor_long = 2u128.pow(80);
        let x_encoded = HostFixed128Tensor {
            tensor: HostRing128Tensor::encode(&x, scaling_factor_long),
            fractional_precision: 80,
            integral_precision: 5,
        };

        assert_eq!(
            x_encoded,
            HostFixed128Tensor {
                tensor: plc.from_raw(array![
                    1208925819614629174706176,
                    340282366920936045611735378173418799104,
                    3626777458843887524118528,
                    340282366920933627760096148915069386752
                ]),
                fractional_precision: 80,
                integral_precision: 5,
            }
        );

        let x_decoded_long = HostRing128Tensor::decode(&x_encoded.tensor, scaling_factor_long);
        assert_eq!(x_decoded_long, x);
    }

    fn new_host_fixed_tensor<HostRingT>(x: HostRingT) -> HostFixedTensor<HostRingT> {
        HostFixedTensor {
            tensor: x,
            fractional_precision: 15,
            integral_precision: 8,
        }
    }

    fn new_replicated_fixed_tensor<RepRingT>(x: RepRingT) -> RepFixedTensor<RepRingT> {
        RepFixedTensor {
            tensor: x,
            fractional_precision: 15,
            integral_precision: 8,
        }
    }

    macro_rules! host_binary_approx_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $factor: expr, $error: expr) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<f64>,
                integral_precision: u32, fractional_precision: u32
            ) {
                let alice = HostPlacement::from("alice");

                let sess = SyncSession::default();

                let one_r: $tt = 1;
                let encode = |item: &$tt| (Wrapping(one_r << fractional_precision) * Wrapping(*item)).0;
                let xs = xs.clone().map(encode);
                let ys = ys.clone().map(encode);

                let x = FixedTensor::Host(HostFixedTensor {
                    tensor: HostRingTensor::from_raw_plc(xs.clone(), alice.clone()),
                    integral_precision,
                    fractional_precision
                });
                let y = FixedTensor::Host(HostFixedTensor {
                    tensor: HostRingTensor::from_raw_plc(ys.clone(), alice.clone()),
                    integral_precision,
                    fractional_precision,
                });

                let sum = alice.$test_func(&sess, &x, &y);
                let opened_product = match sum {
                    FixedTensor::Host(r) => r,
                    _ => panic!("Should not produce a replicated tensor on a host placement"),
                };
                let r64 = Convert::decode(&opened_product.tensor, one_r << (opened_product.fractional_precision));
                let distance = squared_distance(&r64, &zs);

                let _: Vec<_> = distance.iter().enumerate().map(|(i, d)| {
                    assert!(*d < $error, "failed at index {:?} when dividing {:?} / {:?}, result is {:?}, should have been {:?}", i, xs[i], ys[i], r64.0[i], zs[i]);
                }).collect();

                let expected_precision = match x {
                    FixedTensor::Host(x) => x.fractional_precision * $factor,
                    _ => unreachable!(),
                };
                assert_eq!(opened_product.fractional_precision, expected_precision);
            }
        };
    }

    host_binary_approx_func_test!(test_host_div64, div<u64>, 1, 0.00000001);
    host_binary_approx_func_test!(test_host_div128, div<u128>, 1, 0.00000001);

    macro_rules! host_binary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $factor: expr) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement::from("alice");

                let sess = SyncSession::default();

                let x = FixedTensor::Host(new_host_fixed_tensor(HostRingTensor::from_raw_plc(
                    xs,
                    alice.clone(),
                )));
                let y = FixedTensor::Host(new_host_fixed_tensor(HostRingTensor::from_raw_plc(
                    ys,
                    alice.clone(),
                )));

                let sum = alice.$test_func(&sess, &x, &y);
                let opened_product = match sum {
                    FixedTensor::Host(r) => r,
                    _ => panic!("Should not produce a replicated tensor on a host placement"),
                };
                assert_eq!(opened_product.tensor, alice.from_raw(zs));
                let expected_precision = match x {
                    FixedTensor::Host(x) => x.fractional_precision * $factor,
                    _ => unreachable!(),
                };
                assert_eq!(opened_product.fractional_precision, expected_precision);
            }
        };
    }

    host_binary_func_test!(test_host_add64, add<u64>, 1);
    host_binary_func_test!(test_host_add128, add<u128>, 1);
    host_binary_func_test!(test_host_sub64, sub<u64>, 1);
    host_binary_func_test!(test_host_sub128, sub<u128>, 1);
    host_binary_func_test!(test_host_mul64, mul<u64>, 2);
    host_binary_func_test!(test_host_mul128, mul<u128>, 2);
    host_binary_func_test!(test_host_dot64, dot<u64>, 2);
    host_binary_func_test!(test_host_dot128, dot<u128>, 2);

    #[test]
    fn test_fixed_host_mul64() {
        let a = vec![1u64, 2];
        let b = vec![3u64, 4];
        let a = Array::from_shape_vec(IxDyn(&[a.len()]), a).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[b.len()]), b).unwrap();
        let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
        for i in 0..a.len() {
            target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
        }
        test_host_mul64(a, b, target);
    }

    macro_rules! rep_binary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $factor: expr) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let x = FixedTensor::Host(new_host_fixed_tensor(HostRingTensor::from_raw_plc(xs, alice.clone())));
                let y = FixedTensor::Host(new_host_fixed_tensor(HostRingTensor::from_raw_plc(ys, alice.clone())));

                let sess = SyncSession::default();

                let sum = rep.$test_func(&sess, &x, &y);
                let opened_product = match sum {
                    FixedTensor::Replicated(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an unreplicated tensor on a replicated placement"),
                };
                assert_eq!(
                    opened_product.tensor,
                    alice.from_raw(zs)
                );
                let expected_precision = match x {
                    FixedTensor::Host(x) => x.fractional_precision * $factor,
                    _ => unreachable!(),
                };
                assert_eq!(
                    opened_product.fractional_precision,
                    expected_precision
                );
            }
        };
    }

    rep_binary_func_test!(test_rep_add64, add<u64>, 1);
    rep_binary_func_test!(test_rep_add128, add<u128>, 1);
    rep_binary_func_test!(test_rep_sub64, sub<u64>, 1);
    rep_binary_func_test!(test_rep_sub128, sub<u128>, 1);
    rep_binary_func_test!(test_rep_mul64, mul<u64>, 2);
    rep_binary_func_test!(test_rep_mul128, mul<u128>, 2);
    rep_binary_func_test!(test_rep_dot64, dot<u64>, 2);
    rep_binary_func_test!(test_rep_dot128, dot<u128>, 2);

    macro_rules! pairwise_same_length {
        ($func_name:ident, $tt: ident) => {
            fn $func_name() -> impl Strategy<Value = (ArrayD<$tt>, ArrayD<$tt>)> {
                (1usize..25)
                    .prop_flat_map(|length| {
                        (
                            proptest::collection::vec(any::<$tt>(), length),
                            proptest::collection::vec(any::<$tt>(), length),
                        )
                    })
                    .prop_map(|(x, y)| {
                        let a = Array::from_shape_vec(IxDyn(&[x.len()]), x).unwrap();
                        let b = Array::from_shape_vec(IxDyn(&[y.len()]), y).unwrap();
                        (a, b)
                    })
                    .boxed()
            }
        };
    }

    pairwise_same_length!(pairwise_same_length64, u64);
    pairwise_same_length!(pairwise_same_length128, u128);

    macro_rules! pairwise_bounded_same_length {
        ($func_name:ident, $tt: ident) => {
            fn $func_name(bit_room: usize) -> impl Strategy<Value = (ArrayD<$tt>, ArrayD<$tt>)> {
                (1usize..25)
                    .prop_flat_map(move |length: usize| {
                        let foo1 = bit_room;
                        let foo2 = bit_room;
                        (
                            proptest::collection::vec(
                                any::<$tt>().prop_map(move |x| x >> foo1),
                                length,
                            ),
                            proptest::collection::vec(
                                any::<$tt>().prop_map(move |x| {
                                    let y = x >> foo2;
                                    if y == 0 {
                                        1
                                    } else {
                                        y
                                    }
                                }),
                                length,
                            ),
                        )
                    })
                    .prop_map(|(x, y)| {
                        let a = Array::from_shape_vec(IxDyn(&[x.len()]), x).unwrap();
                        let b = Array::from_shape_vec(IxDyn(&[y.len()]), y).unwrap();
                        (a, b)
                    })
                    .boxed()
            }
        };
    }

    pairwise_bounded_same_length!(pairwise_bounded_same_length64, i64);
    pairwise_bounded_same_length!(pairwise_bounded_same_length128, i128);

    #[test]
    fn test_fixed_rep_mul64() {
        let a = vec![1u64, 2];
        let b = vec![3u64, 4];
        let a = Array::from_shape_vec(IxDyn(&[a.len()]), a).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[b.len()]), b).unwrap();
        let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
        for i in 0..a.len() {
            target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
        }
        test_rep_mul64(a, b, target);
    }

    proptest! {
        #[test]
        fn test_fuzzy_host_add64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) + std::num::Wrapping(b[i])).0;
            }
            test_host_add64(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_add128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) + std::num::Wrapping(b[i])).0;
            }
            test_host_add128(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_sub64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) - std::num::Wrapping(b[i])).0;
            }
            test_host_sub64(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_sub128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) - std::num::Wrapping(b[i])).0;
            }
            test_host_sub128(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_mul64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_host_mul64(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_mul128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_host_mul128(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_dot64((a,b) in pairwise_same_length64())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_host_dot64(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_dot128((a,b) in pairwise_same_length128())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_host_dot128(a, b, target);
        }

        #[test]
        fn test_fuzzy_host_div64((a,b) in pairwise_bounded_same_length64(2 * 15))
        {
            let fractional_precision = 15;
            let integral_precision = 49;
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0f64; a.len()]).unwrap();
            for i in 0..a.len() {
                let d = (a[i] as f64) / (b[i] as f64);
                target[i] = d;
            }
            test_host_div64(a.map(|x| *x as u64), b.map(|x| *x as u64), target, integral_precision, fractional_precision);
        }

        #[test]
        fn test_fuzzy_host_div128((a,b) in pairwise_bounded_same_length64(2 * 15))
        {
            let fractional_precision = 15;
            let integral_precision = 49;
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0f64; a.len()]).unwrap();
            for i in 0..a.len() {
                let d = (a[i] as f64) / (b[i] as f64);
                target[i] = d;
            }
            test_host_div128(a.map(|x| *x as u128), b.map(|x| *x as u128), target, integral_precision, fractional_precision);
        }

        #[test]
        fn test_fuzzy_rep_add64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) + std::num::Wrapping(b[i])).0;
            }
            test_rep_add64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_add128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) + std::num::Wrapping(b[i])).0;
            }
            test_rep_add128(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_sub64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) - std::num::Wrapping(b[i])).0;
            }
            test_rep_sub64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_sub128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) - std::num::Wrapping(b[i])).0;
            }
            test_rep_sub128(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_mul64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_rep_mul64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_mul128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_rep_mul128(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_dot64((a,b) in pairwise_same_length64())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_rep_dot64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_dot128((a,b) in pairwise_same_length128())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_rep_dot128(a, b, target);
        }

        #[test]
        fn test_fuzzy_fixed_rep_greater_than64((a,b) in pairwise_bounded_same_length64(10 + 1))
        {
            let mut target: Vec<u64> = vec![0_u64; a.len()];
            for i in 0..a.len() {
                target[i] = (a[i] > b[i]) as u64;
            }
            test_rep_greater_than64(a.map(|x| *x as f64), b.map(|x| *x as f64), target);
        }

        #[test]
        fn test_fuzzy_fixed_rep_greater_than128((a,b) in pairwise_bounded_same_length128(10 + 1))
        {
            let mut target: Vec<u128> = vec![0_u128; a.len()];
            for i in 0..a.len() {
                target[i] = (a[i] > b[i]) as u128;
            }
            test_rep_greater_than128(a.map(|x| *x as f64), b.map(|x| *x as f64), target);
        }
    }

    fn squared_distance(x: &HostFloat64Tensor, target: &ArrayD<f64>) -> ArcArrayD<f64> {
        assert_eq!(x.shape().0 .0, target.shape());
        (x.0.clone() - target) * (x.0.clone() - target)
    }

    macro_rules! rep_div_func_concrete_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $i_precision: expr, $f_precision: expr) => {
            fn $func_name(xs: ArrayD<f64>, ys: ArrayD<f64>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let encode = |item: &f64| (2_i64.pow($f_precision) as f64 * item) as $tt;

                let xs = xs.clone().map(encode);
                let ys = ys.clone().map(encode);
                let x = FixedTensor::Host(HostFixedTensor{
                    tensor: HostRingTensor::from_raw_plc(xs.clone(), alice.clone()),
                    integral_precision: $i_precision,
                    fractional_precision: $f_precision,
                });
                let y = FixedTensor::Host(HostFixedTensor {
                    tensor: HostRingTensor::from_raw_plc(ys.clone(), alice.clone()),
                    integral_precision: $i_precision,
                    fractional_precision: $f_precision,
                });

                let sess = SyncSession::default();

                let sum = rep.$test_func(&sess, &x, &y);
                let opened_product = match sum {
                    FixedTensor::Replicated(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an non-replicated tensor on a replicated placement"),
                };

                let mut expected_result = Array::from_shape_vec(IxDyn(&[xs.clone().len()]), vec![0 as f64; xs.clone().len()]).unwrap();
                for i in 0..xs.len() {
                    expected_result[i] = (xs[i] as f64) / (ys[i] as f64);
                }
                let result = Convert::decode(&opened_product.tensor, (2 as $tt).pow($f_precision));
                let distance = squared_distance(&result, &expected_result);
                let error: f64 = (1_f64) / ((2 as $tt).pow($f_precision) as f64);

                let _: Vec<_> = distance.iter().enumerate().map(|(i, d)| {
                    assert!(*d < error, "failed at index {:?} when dividing {:?} / {:?}, result is {:?}, should have been {:?}", i, xs[i], ys[i], result.0[i], expected_result[i]);
                }).collect();


            }
        };
    }

    #[test]
    fn test_fixed_rep_div64() {
        let a: Vec<f64> = vec![1.0, 2.0, 2.0];
        let b: Vec<f64> = vec![3.0, 7.0, 1.41];
        let a = Array::from_shape_vec(IxDyn(&[a.len()]), a).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[b.len()]), b).unwrap();

        rep_div_func_concrete_test!(test_rep_div64, div<u64>, 10, 15);
        test_rep_div64(a, b);
    }

    #[test]
    fn test_fixed_rep_div128() {
        let a: Vec<f64> = vec![1.0, 2.0, 2.0];
        let b: Vec<f64> = vec![3.0, 7.0, 1.41];
        let a = Array::from_shape_vec(IxDyn(&[a.len()]), a).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[b.len()]), b).unwrap();

        rep_div_func_concrete_test!(test_rep_div128, div<u128>, 10, 15);
        test_rep_div128(a, b);
    }

    macro_rules! new_symbolic_replicated_tensor {
        ($func_name:ident, $tt: ty) => {
            #[cfg(feature = "compile")]
            fn $func_name(
                name: &str,
                rep: &ReplicatedPlacement,
            ) -> Symbolic<RepTensor<Symbolic<HostRingTensor<$tt>>>> {
                let (alice, bob, carole) = rep.host_placements();
                let symbolic_replicated = Symbolic::Concrete(RepTensor {
                    shares: [
                        [
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"00"),
                                plc: alice.clone(),
                            }),
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"01"),
                                plc: alice.clone(),
                            }),
                        ],
                        [
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"10"),
                                plc: bob.clone(),
                            }),
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"11"),
                                plc: bob.clone(),
                            }),
                        ],
                        [
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"20"),
                                plc: carole.clone(),
                            }),
                            Symbolic::Symbolic(SymbolicHandle {
                                op: format!("{}{}", name, &"21"),
                                plc: carole.clone(),
                            }),
                        ],
                    ],
                });
                symbolic_replicated
            }
        };
    }

    new_symbolic_replicated_tensor!(new_symbolic_replicated_tensor64, u64);
    new_symbolic_replicated_tensor!(new_symbolic_replicated_tensor128, u128);

    macro_rules! rep_div_symbolic_test {
        ($func_name:ident, $new_symbolic_rep: ident) => {
            #[cfg(feature = "compile")]
            fn $func_name(i_precision: u32, f_precision: u32) {
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let x = Symbolic::Concrete(RepFixedTensor {
                    fractional_precision: f_precision,
                    integral_precision: i_precision,
                    tensor: $new_symbolic_rep(&"x", &rep),
                });

                let y = Symbolic::Concrete(RepFixedTensor {
                    fractional_precision: f_precision,
                    integral_precision: i_precision,
                    tensor: $new_symbolic_rep(&"y", &rep),
                });

                let sess = SymbolicSession::default();

                let result = rep.div(&sess, &x, &y);
                match result {
                    Symbolic::Concrete(RepFixedTensor {
                        tensor: _,
                        fractional_precision,
                        integral_precision,
                    }) => {
                        assert_eq!(fractional_precision, f_precision);
                        assert_eq!(integral_precision, i_precision);
                    }
                    _ => {
                        panic!("Expected a concrete result from the symbolic division on a concrete value")
                    }
                }
            }
        }
    }

    rep_div_symbolic_test!(rep_div_symbolic_test64, new_symbolic_replicated_tensor64);
    rep_div_symbolic_test!(rep_div_symbolic_test128, new_symbolic_replicated_tensor128);

    #[cfg(feature = "compile")]
    #[test]
    fn test_fixed_rep_symbolic_div64() {
        rep_div_symbolic_test64(10, 20);
    }

    #[cfg(feature = "compile")]
    #[test]
    fn test_fixed_rep_symbolic_div128() {
        rep_div_symbolic_test128(10, 50);
    }

    macro_rules! rep_prefix_op_fixed_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $f_precision: expr) => {
            fn $func_name(x: Vec<ArrayD<$tt>>, y_target: Vec<$tt>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let encode = |item: &$tt| (2_i64.pow($f_precision) as $tt * item) as $tt;

                let x_fixed_vec = x
                    .into_iter()
                    .map(|x| {
                        let x_encode = x.map(encode);
                        let x_ring: HostRingTensor<_> = alice.from_raw(x_encode);
                        let x_shared: RepTensor<_> = rep.share(&sess, &x_ring);
                        new_replicated_fixed_tensor(x_shared)
                    })
                    .collect();

                let outputs = rep.prefix_mul(&sess, x_fixed_vec);

                for (i, output) in outputs.iter().enumerate() {
                    let output_reveal = alice.reveal(&sess, output);
                    let result =
                        Convert::decode(&output_reveal.tensor, (2 as $tt).pow($f_precision));
                    assert_eq!(result.0.as_slice().unwrap()[0] as $tt, y_target[i]);
                }
            }
        };
    }

    rep_prefix_op_fixed_test!(test_rep_prefix_mul_fixed64, prefix_mul_fixed<u64>, 15);
    rep_prefix_op_fixed_test!(test_rep_prefix_mul_fixed128, prefix_mul_fixed<u128>, 15);

    #[test]
    fn test_rep_prefix_mul_fixed_64() {
        let x = vec![
            array![1u64].into_dyn(),
            array![2u64].into_dyn(),
            array![3u64].into_dyn(),
            array![4u64].into_dyn(),
        ];
        let y_target = vec![1, 2, 6, 24];

        test_rep_prefix_mul_fixed64(x, y_target);
    }

    #[test]
    fn test_rep_prefix_mul_fixed_128() {
        let x = vec![
            array![1u128].into_dyn(),
            array![2u128].into_dyn(),
            array![3u128].into_dyn(),
            array![4u128].into_dyn(),
        ];
        let y_target = vec![1u128, 2, 6, 24];

        test_rep_prefix_mul_fixed128(x, y_target);
    }

    macro_rules! rep_poly_eval_fixed_test {
        ($func_name:ident, $test_func: ident<$tt: ty>, $f_precision: expr) => {
            fn $func_name(x: ArrayD<f64>, coeffs: Vec<f64>, y_target: Vec<f64>) {
                use crate::fixedpoint::PolynomialEval;

                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let encode = |item: &f64| (2_i64.pow($f_precision) as f64 * item) as $tt;
                let x_encoded = x.map(encode);
                let x_ring: HostRingTensor<_> = alice.from_raw(x_encoded);
                let x_shared: RepTensor<_> = rep.share(&sess, &x_ring);
                let x_fixed_shared = new_replicated_fixed_tensor(x_shared.clone());

                let output = rep.polynomial_eval(&sess, coeffs, x_fixed_shared);
                let output_reveal = alice.reveal(&sess, &output);
                let result = Convert::decode(&output_reveal.tensor, (2 as $tt).pow($f_precision));

                for i in 0..y_target.len() {
                    let error = (result.0[i] - y_target[i]).abs();
                    assert!(error < f64::EPSILON);
                }
            }
        };
    }

    rep_poly_eval_fixed_test!(test_rep_poly_eval_fixed64, poly_eval<u64>, 15);
    rep_poly_eval_fixed_test!(test_rep_poly_eval_fixed128, poly_eval<u128>, 15);

    #[test]
    fn test_rep_poly_eval_64() {
        let x = array![1f64, 2., 3., 4.].into_dyn();
        let coeffs = vec![1f64, 2., 3.];
        let y_targets = vec![6f64, 17., 34., 57.];

        test_rep_poly_eval_fixed64(x, coeffs, y_targets);
    }

    #[test]
    fn test_rep_poly_eval_128() {
        let x = array![1f64, 2., 3., 4.].into_dyn();
        let coeffs = vec![1f64, 2., 3.];
        let y_targets = vec![6f64, 17., 34., 57.];

        test_rep_poly_eval_fixed128(x, coeffs, y_targets);
    }

    macro_rules! rep_approx_unary_fixed_test {
        ($func_name:ident, $test_func: ident<$ti: ty, $tu: ty>, $i_precision: expr, $f_precision: expr, $err: expr) => {
            fn $func_name(x: ArrayD<f64>, y_target: Vec<f64>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let encode = |item: &f64| -> $tu {
                    let tmp: $ti = (2f64.powf($f_precision as f64) * item) as $ti;
                    tmp as $tu
                };
                let x_encoded = x.map(encode);

                let x = FixedTensor::Host(HostFixedTensor {
                    tensor: HostRingTensor::from_raw_plc(x_encoded.clone(), alice.clone()),
                    integral_precision: $i_precision,
                    fractional_precision: $f_precision,
                });

                let exp_result = rep.$test_func(&sess, &x);

                let opened_exp = match exp_result {
                    FixedTensor::Replicated(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an non-replicated tensor on a replicated placement"),
                };

                let result = Convert::decode(&opened_exp.tensor, (2 as $tu).pow($f_precision));

                // operation precision is not as accurate as the fixed point precision
                for i in 0..y_target.len() {
                    let error = (result.0[i] - y_target[i]).abs();
                    assert!(error < $err, "failed at index {:?}, error is {:?}", i, error);
                }
            }
        };
    }

    rep_approx_unary_fixed_test!(test_rep_pow2_fixed64, pow2<i64, u64>, 10, 10, 0.001);
    rep_approx_unary_fixed_test!(test_rep_pow2_fixed128, pow2<i128, u128>, 30, 10, 0.001);

    rep_approx_unary_fixed_test!(test_rep_exp_fixed64, exp<i64, u64>, 10, 10, 0.1);
    rep_approx_unary_fixed_test!(test_rep_exp_fixed128, exp<i128, u128>, 20, 20, 0.001);

    rep_approx_unary_fixed_test!(test_rep_sqrt_fixed64, sqrt<i64, u64>, 10, 10, 0.1);
    rep_approx_unary_fixed_test!(test_rep_sqrt_fixed128, sqrt<i128, u128>, 20, 20, 0.001);

    #[test]
    fn test_exp2_64() {
        let x = array![1f64, 2.5, -3.0, 4.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| 2_f64.powf(*item)).collect();
        test_rep_pow2_fixed64(x, y_targets);
    }

    #[test]
    fn test_exp2_128() {
        let x = array![1f64, 2.5, -3.0, 4.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| 2_f64.powf(*item)).collect();
        test_rep_pow2_fixed128(x, y_targets);
    }

    #[test]
    fn test_exponential_64() {
        let x = array![1f64, 2.5, -3.0, 4.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| item.exp()).collect();
        test_rep_exp_fixed64(x, y_targets);
    }

    #[test]
    fn test_exponential_128() {
        let x = array![1f64, 2.5, -3.0, 4.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| item.exp()).collect();
        test_rep_exp_fixed128(x, y_targets);
    }

    #[test]
    fn test_sqrt_64() {
        let x = array![0.001, 0.01, 0.1, 1f64, 2., 3., 4., 10., 20., 30., 40., 50., 100., 1000.]
            .into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| item.sqrt()).collect();
        test_rep_sqrt_fixed64(x, y_targets);
    }

    #[test]
    fn test_sqrt_128() {
        let x = array![
            0.001, 0.01, 0.1, 1f64, 2., 3., 4., 10., 50., 100., 1000., 10000., 100000., 500000.
        ]
        .into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| item.sqrt()).collect();
        test_rep_sqrt_fixed128(x, y_targets);
    }

    rep_approx_unary_fixed_test!(test_rep_sigmoid_fixed64, sigmoid<i64, u64>, 10, 10, 0.1);
    rep_approx_unary_fixed_test!(test_rep_sigmoid_fixed128, sigmoid<i128, u128>, 20, 20, 0.001);

    #[test]
    fn test_sigmoid_64() {
        let x = array![1f64, 2.5, -3.0, 4.0, 40.0, -30.0, -4.0, -6.0, 6.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| 1.0 / (1.0 + (-item).exp())).collect();
        test_rep_sigmoid_fixed64(x, y_targets);
    }

    #[test]
    fn test_sigmoid_128() {
        let x = array![1f64, 2.5, -3.0, 4.0, -4.0, -6.0, 6.0, 1024.0].into_dyn();
        let y_targets: Vec<_> = x.iter().map(|item| 1.0 / (1.0 + (-item).exp())).collect();
        test_rep_sigmoid_fixed128(x, y_targets);
    }

    macro_rules! rep_unary_symbolic_test {
        ($func_name:ident, $test_func:ident, $new_symbolic_rep: ident) => {
            #[cfg(feature = "compile")]
            fn $func_name(i_precision: u32, f_precision: u32) {
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let x = Symbolic::Concrete(RepFixedTensor {
                    fractional_precision: f_precision,
                    integral_precision: i_precision,
                    tensor: $new_symbolic_rep(&"x", &rep),
                });

                let sess = SymbolicSession::default();

                let result = rep.$test_func(&sess, &x);
                match result {
                    Symbolic::Concrete(RepFixedTensor {
                        tensor: _,
                        fractional_precision,
                        integral_precision,
                    }) => {
                        assert_eq!(fractional_precision, f_precision);
                        assert_eq!(integral_precision, i_precision);
                    }
                    _ => {
                        panic!("Expected a concrete result from the symbolic unary op on a concrete value")
                    }
                }
            }
        }
    }

    rep_unary_symbolic_test!(
        rep_exp_symbolic_test64,
        exp,
        new_symbolic_replicated_tensor64
    );

    #[cfg(feature = "compile")]
    #[test]
    fn test_fixed_rep_symbolic_exp64() {
        rep_exp_symbolic_test64(10, 10);
    }

    macro_rules! rep_signed_binary_func_test {
        ($func_name:ident, $test_func: ident<$ti: ty, $tu: ty>, $i_precision: expr, $f_precision: expr) => {
            fn $func_name(x: ArrayD<f64>, y: ArrayD<f64>, target: Vec<$tu>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();
                let encode = |item: &f64| -> $tu {
                    let tmp: $ti = (2f64.powf($f_precision as f64) * item) as $ti;
                    tmp as $tu
                };
                let x_encoded = x.map(encode);
                let y_encoded = y.map(encode);

                let xf = HostFixedTensor {
                    tensor: HostRingTensor::from_raw_plc(x_encoded.clone(), alice.clone()),
                    integral_precision: $i_precision,
                    fractional_precision: $f_precision,
                };

                let yf = HostFixedTensor {
                    tensor: HostRingTensor::from_raw_plc(y_encoded.clone(), alice.clone()),
                    integral_precision: $i_precision,
                    fractional_precision: $f_precision,
                };

                let xs = rep.share(&sess, &xf);
                let ys = rep.share(&sess, &yf);

                let zs: ReplicatedBitTensor = rep.$test_func(&sess, &xs, &ys);
                let z = alice.reveal(&sess, &zs);

                for i in 0..target.len() {
                    assert_eq!(
                        target[i] as $tu, z.0.data[i] as $tu,
                        "failed comparing {:?} with {:?}",
                        x[i], y[i]
                    );
                }
            }
        };
    }

    rep_signed_binary_func_test!(test_rep_greater_than64, greater_than<i64, u64>, 10, 10);
    rep_signed_binary_func_test!(test_rep_greater_than128, greater_than<i128, u128>, 10, 10);

    rep_signed_binary_func_test!(test_rep_less_than64, less<i64, u64>, 10, 10);
    rep_signed_binary_func_test!(test_rep_less_than128, less<i128, u128>, 20, 20);

    #[test]
    fn test_fixed_rep_greater_than64() {
        let x = array![0f64, 2.7, -2.9, 4.1].into_dyn();
        let y = array![1f64, 2.5, -3.0, 4.0].into_dyn();
        let targets: Vec<u64> = vec![0_u64, 1, 1, 1];
        test_rep_greater_than64(x, y, targets);
    }

    #[test]
    fn test_fixed_rep_greater_than128() {
        let x = array![0f64, 2.7, -2.9, 4.1, -3.555].into_dyn();
        let y = array![1f64, 2.5, -3.0, 4.0, -3.354].into_dyn();
        let targets: Vec<u128> = vec![0_u128, 1, 1, 1, 0];
        test_rep_greater_than128(x, y, targets);
    }

    #[test]
    fn test_fixed_rep_less_than64() {
        let x = array![0f64, 2.7, -2.9, 4.1, -3.555].into_dyn();
        let y = array![1f64, 2.5, -3.0, 4.0, -3.354].into_dyn();
        let targets: Vec<u64> = vec![1_u64, 0, 0, 0, 1];
        test_rep_less_than64(x, y, targets);
    }

    #[test]
    fn test_fixed_rep_less_than128() {
        let x = array![0f64, 2.7, -2.9, 4.1, -3.555].into_dyn();
        let y = array![1f64, 2.5, -3.0, 4.0, -3.354].into_dyn();
        let targets: Vec<u128> = vec![1_u128, 0, 0, 0, 1];
        test_rep_less_than128(x, y, targets);
    }

    macro_rules! rep_index_axis_fixed_test {
        ($func_name:ident, $ti: ty, $tu: ty,$axis: expr, $index: expr, $i_precision: expr, $f_precision: expr) => {
            fn $func_name(x: ArrayD<f64>, y_target: ArrayD<f64>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();
                let encode = |item: &f64| -> $tu {
                    let tmp: $ti = (2f64.powf($f_precision as f64) * item) as $ti;
                    tmp as $tu
                };
                let x_encoded = x.map(encode);

                let x = FixedTensor::Host(HostFixedTensor {
                    tensor: HostRingTensor::from_raw_plc(x_encoded.clone(), alice.clone()),
                    integral_precision: $i_precision,
                    fractional_precision: $f_precision,
                });

                let exp_result = rep.index_axis(&sess, $axis, $index, &x);

                let opened_exp = match exp_result {
                    FixedTensor::Replicated(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an non-replicated tensor on a replicated placement"),
                };

                let result = Convert::decode(&opened_exp.tensor, (2 as $tu).pow($f_precision));
                assert_eq!(result.0, y_target);
            }
        };
    }

    rep_index_axis_fixed_test!(test_rep_index_axis_fixed64, i64, u64, 1, 0, 10, 10);
    rep_index_axis_fixed_test!(test_rep_index_axis_fixed128, i128, u128, 1, 0, 20, 20);

    #[test]
    fn test_index_axis_64() {
        let x = array![[1f64, 2.5], [-3.0, 4.0]].into_dyn();
        let y_targets = x.index_axis(Axis(1), 0).into_dyn().to_owned();
        test_rep_index_axis_fixed64(x, y_targets);
    }

    #[test]
    fn test_index_axis_128() {
        let x = array![[1f64, 2.5], [-3.0, 4.0]].into_dyn();
        let y_targets = x.index_axis(Axis(1), 0).into_dyn().to_owned();
        test_rep_index_axis_fixed128(x, y_targets);
    }

    macro_rules! rep_expand_dims_fixed_test {
        ($func_name:ident, $ti: ty, $tu: ty,$axis: expr, $i_precision: expr, $f_precision: expr) => {
            fn $func_name(x: ArrayD<f64>, y_target: ArrayD<f64>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();
                let encode = |item: &f64| -> $tu {
                    let tmp: $ti = (2f64.powf($f_precision as f64) * item) as $ti;
                    tmp as $tu
                };
                let x_encoded = x.map(encode);

                let x = FixedTensor::Host(HostFixedTensor {
                    tensor: HostRingTensor::from_raw_plc(x_encoded.clone(), alice.clone()),
                    integral_precision: $i_precision,
                    fractional_precision: $f_precision,
                });

                let exp_result = rep.expand_dims(&sess, $axis, &x);

                let opened_exp = match exp_result {
                    FixedTensor::Replicated(r) => alice.reveal(&sess, &r),
                    _ => panic!("Should not produce an non-replicated tensor on a replicated placement"),
                };

                let result = Convert::decode(&opened_exp.tensor, (2 as $tu).pow($f_precision));
                assert_eq!(result.0, y_target);
            }
        };
    }

    rep_expand_dims_fixed_test!(test_rep_expand_dim_fixed64, i64, u64, vec![0], 10, 10);
    rep_expand_dims_fixed_test!(test_rep_expand_dim_fixed128, i128, u128, vec![0], 20, 20);

    #[test]
    fn test_rep_expand_dim_fixed_64() {
        let x = array![1f64, 2.].into_dyn();
        let y_targets = array![[1f64, 2.0]].into_dyn();
        test_rep_expand_dim_fixed64(x, y_targets);
    }

    #[test]
    fn test_rep_expand_dim_fixed_128() {
        let x = array![1f64, 2.0].into_dyn();
        let y_targets = array![[1f64, 2.0]].into_dyn();
        test_rep_expand_dim_fixed128(x, y_targets);
    }
}
