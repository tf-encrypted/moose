use crate::computation::{AdditivePlacement, HostPlacement, Placed};
use crate::ring::{Ring128Tensor, Ring64Tensor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractAdditiveTensor<R> {
    shares: [R; 2],
}

pub type Additive64Tensor = AbstractAdditiveTensor<Ring64Tensor>;

pub type Additive128Tensor = AbstractAdditiveTensor<Ring128Tensor>;

impl<R> Placed for AbstractAdditiveTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Placement = AdditivePlacement;

    fn placement(&self) -> Self::Placement {
        let AbstractAdditiveTensor { shares: [x0, x1] } = self;

        let owner0 = x0.placement().owner;
        let owner1 = x1.placement().owner;

        let owners = [owner0, owner1];
        AdditivePlacement { owners }
    }
}
