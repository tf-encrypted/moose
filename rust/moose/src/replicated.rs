use crate::bit::PlacedBitTensor;
use crate::computation::{HostPlacement, Placed, ReplicatedPlacement};
use crate::prim::PlacedPrfKey;
use crate::ring::{PlacedRing128Tensor, PlacedRing64Tensor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractReplicatedTensor<R> {
    shares: [[R; 2]; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractReplicatedSetup<K> {
    keys: [[K; 2]; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct AbstractReplicatedZeroShare<R> {
    alphas: [R; 3],
}

pub type Replicated64Tensor = AbstractReplicatedTensor<PlacedRing64Tensor>;

pub type Replicated128Tensor = AbstractReplicatedTensor<PlacedRing128Tensor>;

pub type ReplicatedBitTensor = AbstractReplicatedTensor<PlacedBitTensor>;

pub type ReplicatedSetup = AbstractReplicatedSetup<PlacedPrfKey>;

impl<R> Placed for AbstractReplicatedTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Self::Placement {
        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = self;

        let player0 = x00.placement();
        assert_eq!(x10.placement(), player0);

        let player1 = x11.placement();
        assert_eq!(x21.placement(), player1);

        let player2 = x22.placement();
        assert_eq!(x02.placement(), player2);

        let owners = [player0.owner, player1.owner, player2.owner];
        ReplicatedPlacement { owners }
    }
}

impl<K> Placed for AbstractReplicatedSetup<K>
where
    K: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Self::Placement {
        let AbstractReplicatedSetup {
            keys: [[x00, x10], [x11, x21], [x22, x02]],
        } = self;

        let player0 = x00.placement();
        assert_eq!(x10.placement(), player0);

        let player1 = x11.placement();
        assert_eq!(x21.placement(), player1);

        let player2 = x22.placement();
        assert_eq!(x02.placement(), player2);

        let owners = [player0.owner, player1.owner, player2.owner];
        ReplicatedPlacement { owners }
    }
}
