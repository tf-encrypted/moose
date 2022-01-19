//! Abstraction layer for boolean values

use crate::computation::*;
use crate::error::{Error, Result};
use crate::kernels::*;
use crate::replicated::ReplicatedPlacement;
use crate::types::*;
use serde::{Deserialize, Serialize};

/// Boolean tensor abstracting over host and replicated values
// TODO(Dragos) perhaps we can unify BoolTensor with FixedTensor
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BoolTensor<HostT, RepT> {
    Host(HostT),
    Replicated(RepT),
}

impl<HostT, RepT> Placed for BoolTensor<HostT, RepT>
where
    HostT: Placed,
    HostT::Placement: Into<Placement>,
    RepT: Placed,
    RepT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            BoolTensor::Host(x) => Ok(x.placement()?.into()),
            BoolTensor::Replicated(x) => Ok(x.placement()?.into()),
        }
    }
}

impl IdentityOp {
    pub(crate) fn boolean_host_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: BoolTensor<HostT, RepT>,
    ) -> Result<BoolTensor<HostT, RepT>>
    where
        HostPlacement: PlacementIdentity<S, HostT, HostT>,
        HostPlacement: PlacementReveal<S, RepT, HostT>,
    {
        match x {
            BoolTensor::Host(v) => Ok(BoolTensor::Host(plc.identity(sess, &v))),
            BoolTensor::Replicated(v) => {
                let v = plc.reveal(sess, &v);
                Ok(BoolTensor::Host(plc.identity(sess, &v)))
            }
        }
    }

    pub(crate) fn boolean_rep_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: BoolTensor<HostT, RepT>,
    ) -> Result<BoolTensor<HostT, RepT>>
    where
        ReplicatedPlacement: PlacementIdentity<S, RepT, RepT>,
        ReplicatedPlacement: PlacementShare<S, HostT, RepT>,
    {
        match x {
            BoolTensor::Host(v) => {
                let v = plc.share(sess, &v);
                Ok(BoolTensor::Replicated(plc.identity(sess, &v)))
            }
            BoolTensor::Replicated(v) => Ok(BoolTensor::Replicated(plc.identity(sess, &v))),
        }
    }
}

modelled!(PlacementOutput::output, HostPlacement, (BooleanTensor) -> BooleanTensor, OutputOp);

impl OutputOp {
    pub(crate) fn bool_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: BoolTensor<HostT, RepT>,
    ) -> Result<BoolTensor<HostT, RepT>>
    where
        HostPlacement: PlacementOutput<S, HostT, HostT>,
    {
        match x {
            BoolTensor::Host(v) => Ok(BoolTensor::Host(plc.output(sess, &v))),
            BoolTensor::Replicated(_) => Err(Error::UnimplementedOperator(
                "OutputOp missing a replicated boolean tensor implementation.".to_string(),
            )),
        }
    }
}

modelled!(PlacementOr::or, HostPlacement, (BooleanTensor, BooleanTensor) -> BooleanTensor, BitOrOp);

impl BitOrOp {
    pub(crate) fn bool_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        x: BoolTensor<HostT, RepT>,
        y: BoolTensor<HostT, RepT>,
    ) -> Result<BoolTensor<HostT, RepT>>
    where
        HostPlacement: PlacementOr<S, HostT, HostT, HostT>,
        HostPlacement: PlacementReveal<S, RepT, HostT>,
    {
        let x = match x {
            BoolTensor::Host(v) => v,
            BoolTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let y = match y {
            BoolTensor::Host(v) => v,
            BoolTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        Ok(BoolTensor::Host(plc.or(sess, &x, &y)))
    }
}

impl ExpandDimsOp {
    pub(crate) fn bool_rep_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Vec<u32>,
        x: BoolTensor<HostT, RepT>,
    ) -> Result<BoolTensor<HostT, RepT>>
    where
        ReplicatedPlacement: PlacementExpandDims<S, RepT, RepT>,
        ReplicatedPlacement: PlacementShare<S, HostT, RepT>,
    {
        let x = match x {
            BoolTensor::Host(v) => plc.share(sess, &v),
            BoolTensor::Replicated(v) => v,
        };
        let result = plc.expand_dims(sess, axis, &x);
        Ok(BoolTensor::Replicated(result))
    }

    pub(crate) fn bool_host_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<u32>,
        x: BoolTensor<HostT, RepT>,
    ) -> Result<BoolTensor<HostT, RepT>>
    where
        HostPlacement: PlacementExpandDims<S, HostT, HostT>,
        HostPlacement: PlacementReveal<S, RepT, HostT>,
    {
        let x = match x {
            BoolTensor::Replicated(v) => plc.reveal(sess, &v),
            BoolTensor::Host(v) => v,
        };
        let result = plc.expand_dims(sess, axis, &x);
        Ok(BoolTensor::Host(result))
    }
}

impl IndexAxisOp {
    pub(crate) fn bool_rep_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        index: usize,
        x: BoolTensor<HostT, RepT>,
    ) -> Result<BoolTensor<HostT, RepT>>
    where
        ReplicatedPlacement: PlacementIndexAxis<S, RepT, RepT>,
        ReplicatedPlacement: PlacementShare<S, HostT, RepT>,
    {
        let x = match x {
            BoolTensor::Host(v) => plc.share(sess, &v),
            BoolTensor::Replicated(v) => v,
        };
        let result = plc.index_axis(sess, axis, index, &x);
        Ok(BoolTensor::Replicated(result))
    }

    pub(crate) fn bool_host_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: BoolTensor<HostT, RepT>,
    ) -> Result<BoolTensor<HostT, RepT>>
    where
        HostPlacement: PlacementIndexAxis<S, HostT, HostT>,
        HostPlacement: PlacementReveal<S, RepT, HostT>,
    {
        let x = match x {
            BoolTensor::Replicated(v) => plc.reveal(sess, &v),
            BoolTensor::Host(v) => v,
        };
        let result = plc.index_axis(sess, axis, index, &x);
        Ok(BoolTensor::Host(result))
    }
}

impl SaveOp {
    pub fn bool_kernel<S: Session, HostT, RepT>(
        sess: &S,
        plc: &HostPlacement,
        key: cs!(HostString),
        x: BoolTensor<HostT, RepT>,
    ) -> Result<cs!(Unit)>
    where
        HostString: KnownType<S>,
        Unit: KnownType<S>,
        HostPlacement: PlacementSave<S, cs!(HostString), HostT, cs!(Unit)>,
    {
        let x = match x {
            BoolTensor::Replicated(_v) => unimplemented!(),
            BoolTensor::Host(v) => v,
        };
        Ok(plc.save(sess, &key, &x))
    }
}
