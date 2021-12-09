use serde::{Deserialize, Serialize};


#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mirrored3Tensor<HostRingT> {
    pub values: [HostRingT; 3],
}

