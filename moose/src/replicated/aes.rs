use crate::computation::{Placed, ReplicatedPlacement};
use crate::error::Result;
use crate::replicated::ReplicatedBitArray128;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AbstractReplicatedAesKey<RepBitArrayT>(pub(crate) RepBitArrayT);

impl<RepBitArrayT> Placed for AbstractReplicatedAesKey<RepBitArrayT>
where
    RepBitArrayT: Placed<Placement = ReplicatedPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        self.0.placement()
    }
}

moose_type!(ReplicatedAesKey = AbstractReplicatedAesKey<ReplicatedBitArray128>);
