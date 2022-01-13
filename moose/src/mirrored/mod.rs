//! Placement for mirroring operations across multiple hosts
use crate::computation::{HostPlacement, Placed, Role};
use crate::error::Result;
use serde::{Deserialize, Serialize};

mod ops;

/// Placement for mirroring operations across three hosts
///
/// All values are plaintext and kept in lockstep across three host placements.
#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Mirrored3Placement {
    pub owners: [Role; 3],
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
pub struct Mirrored3Tensor<HostTenT> {
    pub values: [HostTenT; 3],
}

impl<HostTenT> Placed for Mirrored3Tensor<HostTenT>
where
    HostTenT: Placed<Placement = HostPlacement>,
{
    type Placement = Mirrored3Placement;

    fn placement(&self) -> Result<Self::Placement> {
        let Mirrored3Tensor {
            values: [x0, x1, x2],
        } = self;

        let owner0 = x0.placement()?.owner;
        let owner1 = x1.placement()?.owner;
        let owner2 = x2.placement()?.owner;

        let owners = [owner0, owner1, owner2];

        Ok(Mirrored3Placement { owners })
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AbstractMirroredFixedTensor<MirRingT> {
    pub tensor: MirRingT,
    pub fractional_precision: u32,
    pub integral_precision: u32,
}

impl<RepRingT: Placed> Placed for AbstractMirroredFixedTensor<RepRingT> {
    type Placement = RepRingT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.tensor.placement()
    }
}
