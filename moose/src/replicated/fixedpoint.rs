use super::*;
use crate::additive::{AdditivePlacement, AdtTensor, TruncPrProvider};
use crate::mirrored::MirFixedTensor;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct RepFixedTensor<RepRingT> {
    pub tensor: RepRingT,
    pub fractional_precision: u32,
    pub integral_precision: u32,
}

impl<RepRingT: Underlying> Underlying for RepFixedTensor<RepRingT> {
    type TensorType = RepRingT::TensorType;
}

impl<RepRingT: MirroredCounterpart> MirroredCounterpart for RepFixedTensor<RepRingT> {
    type MirroredType = MirFixedTensor<RepRingT::MirroredType>;
}

impl<RepRingT: Placed> Placed for RepFixedTensor<RepRingT> {
    type Placement = RepRingT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.tensor.placement()
    }
}

impl<S: Session, RepRingT> PlacementPlace<S, RepFixedTensor<RepRingT>> for ReplicatedPlacement
where
    RepFixedTensor<RepRingT>: Placed<Placement = ReplicatedPlacement>,
    ReplicatedPlacement: PlacementPlace<S, RepRingT>,
{
    fn place(&self, sess: &S, x: RepFixedTensor<RepRingT>) -> RepFixedTensor<RepRingT> {
        match x.placement() {
            Ok(place) if self == &place => x,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                RepFixedTensor {
                    tensor: self.place(sess, x.tensor),
                    integral_precision: x.integral_precision,
                    fractional_precision: x.fractional_precision,
                }
            }
        }
    }
}

impl RingFixedpointMeanOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: RepTensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementMeanAsFixedpoint<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x00);
        let z10 = player0.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x10);
        let z11 = player1.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x11);
        let z21 = player1.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x21);
        let z22 = player2.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x22);
        let z02 = player2.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl TruncPrOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        amount: u32,
        xe: RepTensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        AdditivePlacement: PlacementRepToAdt<S, RepTensor<HostRingT>, AdtTensor<HostRingT>>,
        AdditivePlacement: TruncPrProvider<S, AdtTensor<HostRingT>, AdtTensor<HostRingT>>,
        ReplicatedPlacement: PlacementAdtToRep<S, AdtTensor<HostRingT>, RepTensor<HostRingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let adt = AdditivePlacement {
            owners: [player0.owner, player1.owner],
        };
        let provider = player2;

        let x_adt = adt.rep_to_adt(sess, &xe);
        let y_adt = adt.trunc_pr(sess, amount as usize, &provider, &x_adt);
        Ok(rep.adt_to_rep(sess, &y_adt))
    }
}
