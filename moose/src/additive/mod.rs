//! Placement backed by two-party additive secret sharing.

use crate::computation::{Placed, Role};
use crate::error::Result;
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::kernels::PlacementPlace;
use serde::{Deserialize, Serialize};

mod convert;
mod dabit;
mod ops;
mod trunc;
pub(crate) use dabit::DaBitProvider;
pub(crate) use trunc::TruncPrProvider;

/// Placement type for two-party additive secret sharing
#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Debug)]
pub struct AdditivePlacement {
    pub owners: [Role; 2],
}

impl<R: Into<Role>> From<[R; 2]> for AdditivePlacement {
    fn from(roles: [R; 2]) -> AdditivePlacement {
        let [role0, role1] = roles;
        AdditivePlacement {
            owners: [role0.into(), role1.into()],
        }
    }
}

impl AdditivePlacement {
    pub(crate) fn host_placements(&self) -> (HostPlacement, HostPlacement) {
        let player0 = HostPlacement {
            owner: self.owners[0].clone(),
        };
        let player1 = HostPlacement {
            owner: self.owners[1].clone(),
        };
        (player0, player1)
    }
}

/// Secret tensor used by additive placements
///
/// Values are shared using additive secret sharing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdtTensor<HostTensorT> {
    pub(crate) shares: [HostTensorT; 2],
}

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
