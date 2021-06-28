use crate::ring::{Ring128Tensor, Ring64Tensor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdditiveTensor<R> {
    shares: [R; 2],
}

pub type Additive64Tensor = AdditiveTensor<Ring64Tensor>;

pub type Additive128Tensor = AdditiveTensor<Ring128Tensor>;
