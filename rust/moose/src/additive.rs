use crate::ring::{Ring128Tensor, Ring64Tensor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractAdditiveTensor<R> {
    shares: [R; 2],
}

pub type Additive64Tensor = AbstractAdditiveTensor<Ring64Tensor>;

pub type Additive128Tensor = AbstractAdditiveTensor<Ring128Tensor>;
