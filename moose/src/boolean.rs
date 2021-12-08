use crate::computation::*;
use crate::error::{Error, Result};
use crate::host::HostBitTensor;
use crate::kernels::*;
use crate::replicated::ReplicatedBitTensor;
use serde::{Deserialize, Serialize};

/// TODO(Dragos) perhaps we can unify BoolTensor with FixedTensor
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BoolTensor<HostT, RepT> {
    Host(HostT),
    Replicated(RepT),
}

moose_type!(BooleanTensor = BoolTensor<HostBitTensor, ReplicatedBitTensor>);

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
}
