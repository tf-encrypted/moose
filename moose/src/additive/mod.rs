//! Placement backed by additive secret sharing
use crate::computation::{AdditivePlacement, HostPlacement, Placed};
use crate::error::Result;
use crate::host::{HostBitTensor, HostRing128Tensor, HostRing64Tensor, HostShape};
use crate::kernels::{PlacementAdd, PlacementPlace, PlacementShl, Session};
use serde::{Deserialize, Serialize};

mod convert;
mod dabit;
mod misc;
mod ops;
mod trunc;
pub(crate) use convert::*;
pub use dabit::*;
// pub use misc::*;
pub(crate) use ops::*;
pub(crate) use trunc::*;

/// Secret tensor used by additive placements
///
/// Values are shared using additive secret sharing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdtTensor<HostTensorT> {
    pub(crate) shares: [HostTensorT; 2],
}

moose_type!(AdditiveRing64Tensor = AdtTensor<HostRing64Tensor>);
moose_type!(AdditiveRing128Tensor = AdtTensor<HostRing128Tensor>);
moose_type!(AdditiveBitTensor = AdtTensor<HostBitTensor>);

impl<HostT> Placed for AdtTensor<HostT>
where
    HostT: Placed<Placement = HostPlacement>,
{
    type Placement = AdditivePlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let AdtTensor { shares: [x0, x1] } = self;

        let owner0 = x0.placement()?.owner;
        let owner1 = x1.placement()?.owner;

        let owners = [owner0, owner1];
        Ok(AdditivePlacement { owners })
    }
}

impl<S: Session, HostT> PlacementPlace<S, AdtTensor<HostT>> for AdditivePlacement
where
    AdtTensor<HostT>: Placed<Placement = AdditivePlacement>,
    HostPlacement: PlacementPlace<S, HostT>,
{
    fn place(&self, sess: &S, x: AdtTensor<HostT>) -> AdtTensor<HostT> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                let AdtTensor { shares: [x0, x1] } = x;
                let (player0, player1) = self.host_placements();
                AdtTensor {
                    shares: [player0.place(sess, x0), player1.place(sess, x1)],
                }
            }
        }
    }
}

/// Plaintext shape used by additive placements
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdtShape<HostShapeT> {
    pub(crate) shapes: [HostShapeT; 2],
}

moose_type!(AdditiveShape = AdtShape<HostShape>);

impl<HostT> Placed for AdtShape<HostT>
where
    HostT: Placed<Placement = HostPlacement>,
{
    type Placement = AdditivePlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let AdtShape { shapes: [s0, s1] } = self;

        let owner0 = s0.placement()?.owner;
        let owner1 = s1.placement()?.owner;

        let owners = [owner0, owner1];
        Ok(AdditivePlacement { owners })
    }
}
