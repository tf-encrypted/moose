//! Abstraction layer for integer values.

use crate::computation::*;
use crate::error::Result;
use crate::kernels::*;
use serde::{Deserialize, Serialize};

mod ops;

/// Uint64 Tensor abstracting over host and replicated values
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbstractUint64Tensor<HostT, RepT> {
    Host(HostT),
    Replicated(RepT),
}

impl<HostT, RepT> Placed for AbstractUint64Tensor<HostT, RepT>
where
    HostT: Placed,
    HostT::Placement: Into<Placement>,
    RepT: Placed,
    RepT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            AbstractUint64Tensor::Host(x) => Ok(x.placement()?.into()),
            AbstractUint64Tensor::Replicated(x) => Ok(x.placement()?.into()),
        }
    }
}
