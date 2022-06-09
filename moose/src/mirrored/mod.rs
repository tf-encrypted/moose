//! Placement for mirroring operations across multiple hosts.

use crate::computation::{Placed, Role};
use crate::error::Result;
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::kernels::PlacementPlace;
use crate::Underlying;
use serde::{Deserialize, Serialize};

mod ops;

/// Placement for mirroring operations across three hosts
///
/// All values are plaintext and kept in lockstep across three host placements.
#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Debug)]
pub struct Mirrored3Placement {
    pub owners: [Role; 3],
}

impl<R: Into<Role>> From<[R; 3]> for Mirrored3Placement {
    fn from(roles: [R; 3]) -> Mirrored3Placement {
        let [role0, role1, role2] = roles;
        Mirrored3Placement {
            owners: [role0.into(), role1.into(), role2.into()],
        }
    }
}

impl Mirrored3Placement {
    pub fn host_placements(&self) -> (HostPlacement, HostPlacement, HostPlacement) {
        let player0 = HostPlacement {
            owner: self.owners[0].clone(),
        };
        let player1 = HostPlacement {
            owner: self.owners[1].clone(),
        };
        let player2 = HostPlacement {
            owner: self.owners[2].clone(),
        };
        (player0, player1, player2)
    }
}

/// Base tensor for mirroring across three hosts
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mir3Tensor<HostTenT> {
    pub values: [HostTenT; 3],
}

impl<HostTenT> Placed for Mir3Tensor<HostTenT>
where
    HostTenT: Placed<Placement = HostPlacement>,
{
    type Placement = Mirrored3Placement;

    fn placement(&self) -> Result<Self::Placement> {
        let Mir3Tensor {
            values: [x0, x1, x2],
        } = self;

        let owner0 = x0.placement()?.owner;
        let owner1 = x1.placement()?.owner;
        let owner2 = x2.placement()?.owner;

        let owners = [owner0, owner1, owner2];

        Ok(Mirrored3Placement { owners })
    }
}

impl<S: Session, HostT> PlacementPlace<S, Mir3Tensor<HostT>> for Mirrored3Placement
where
    Mir3Tensor<HostT>: Placed<Placement = Mirrored3Placement>,
    HostPlacement: PlacementPlace<S, HostT>,
{
    fn place(&self, sess: &S, x: Mir3Tensor<HostT>) -> Mir3Tensor<HostT> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                let Mir3Tensor {
                    values: [x0, x1, x2],
                } = x;
                let (player0, player1, player2) = self.host_placements();
                Mir3Tensor {
                    values: [
                        player0.place(sess, x0),
                        player1.place(sess, x1),
                        player2.place(sess, x2),
                    ],
                }
            }
        }
    }
}

impl<HostRingT> Underlying for Mir3Tensor<HostRingT> {
    type TensorType = HostRingT;
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MirFixedTensor<MirRingT> {
    pub tensor: MirRingT,
    pub fractional_precision: u32,
    pub integral_precision: u32,
}

impl<RepRingT: Placed> Placed for MirFixedTensor<RepRingT> {
    type Placement = RepRingT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.tensor.placement()
    }
}
