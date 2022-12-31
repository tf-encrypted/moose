use super::*;
use crate::computation::Placed;
use crate::error::Result;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct RepAesKey<RepBitArrayT>(pub(crate) RepBitArrayT);

impl<RepBitArrayT> Placed for RepAesKey<RepBitArrayT>
where
    RepBitArrayT: Placed<Placement = ReplicatedPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        self.0.placement()
    }
}
