use crate::bit::BitTensor;
use crate::computation::{HostPlacement, Placed, ReplicatedPlacement};
use crate::prim::PrfKey;
use crate::ring::{Ring128Tensor, Ring64Tensor};
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

pub type Replicated64Tensor = AbstractReplicatedTensor<Ring64Tensor>;

pub type Replicated128Tensor = AbstractReplicatedTensor<Ring128Tensor>;

pub type ReplicatedBitTensor = AbstractReplicatedTensor<BitTensor>;

pub type ReplicatedSetup = AbstractReplicatedSetup<PrfKey>;

impl<R> Placed for AbstractReplicatedTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Self::Placement {
        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = self;

        let owner0 = x00.placement().owner;
        assert_eq!(x10.placement().owner, owner0);

        let owner1 = x11.placement().owner;
        assert_eq!(x21.placement().owner, owner1);

        let owner2 = x22.placement().owner;
        assert_eq!(x02.placement().owner, owner2);

        let owners = [owner0, owner1, owner2];
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
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = self;

        let owner0 = k00.placement().owner;
        assert_eq!(k10.placement().owner, owner0);

        let owner1 = k11.placement().owner;
        assert_eq!(k21.placement().owner, owner1);

        let owner2 = k22.placement().owner;
        assert_eq!(k02.placement().owner, owner2);

        let owners = [owner0, owner1, owner2];
        ReplicatedPlacement { owners }
    }
}
