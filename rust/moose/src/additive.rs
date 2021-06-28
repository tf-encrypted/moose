use crate::computation::{AdditivePlacement, HostPlacement, Placed};
use crate::ring::{PlacedRing128Tensor, PlacedRing64Tensor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractAdditiveTensor<R> {
    shares: [R; 2],
}

pub type Additive64Tensor = AbstractAdditiveTensor<PlacedRing64Tensor>;

pub type Additive128Tensor = AbstractAdditiveTensor<PlacedRing128Tensor>;

impl<R> Placed for AbstractAdditiveTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Placement = AdditivePlacement;

    fn placement(&self) -> Self::Placement {
        let AbstractAdditiveTensor { shares: [x0, x1] } = self;

        let player0 = x0.placement();
        let player1 = x1.placement();

        let owners = [player0.owner, player1.owner];
        AdditivePlacement { owners }
    }
}
