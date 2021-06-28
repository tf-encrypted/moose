use crate::bit::BitTensor;
use crate::prim::PrfKey;
use crate::ring::{Ring128Tensor, Ring64Tensor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplicatedTensor<R> {
    shares: [[R; 2]; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractReplicatedSetup<K> {
    keys: [[K; 2]; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ReplicatedZeroShare<R> {
    alphas: [R; 3],
}

pub type Replicated64Tensor = ReplicatedTensor<Ring64Tensor>;

pub type Replicated128Tensor = ReplicatedTensor<Ring128Tensor>;

pub type ReplicatedBitTensor = ReplicatedTensor<BitTensor>;

pub type ReplicatedSetup = AbstractReplicatedSetup<PrfKey>;
