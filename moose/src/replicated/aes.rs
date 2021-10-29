use serde::{Deserialize, Serialize};

use crate::computation::{HostPlacement, Placed, ReplicatedPlacement};
use crate::error::{Error, Result};
use crate::host::HostBitArray128;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AbstractReplicatedAesKey<HostBitArrayT> {
    keys: [[HostBitArrayT; 2]; 3],
}

impl<HostBitArrayT> Placed for AbstractReplicatedAesKey<HostBitArrayT>
where
    HostBitArrayT: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let AbstractReplicatedAesKey {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = self;

        let owner0 = k00.placement()?.owner;
        let owner1 = k11.placement()?.owner;
        let owner2 = k22.placement()?.owner;

        if k10.placement()?.owner == owner0
            && k21.placement()?.owner == owner1
            && k02.placement()?.owner == owner2
        {
            let owners = [owner0, owner1, owner2];
            Ok(ReplicatedPlacement { owners })
        } else {
            Err(Error::MalformedPlacement)
        }
    }
}

moose_type!(ReplicatedAesKey = AbstractReplicatedAesKey<HostBitArray128>);
