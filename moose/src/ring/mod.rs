//! Abstraction layer for Ring values
use crate::computation::*;
use crate::error::Result;
use crate::kernels::*;
use serde::{Deserialize, Serialize};

mod ops;

/// Ring64 Tensor abstracting over host and replicated values
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Z64Tensor<HostT, RepT> {
    Host(HostT),
    Replicated(RepT),
}

impl<HostT, RepT> Placed for Z64Tensor<HostT, RepT>
where
    HostT: Placed,
    HostT::Placement: Into<Placement>,
    RepT: Placed,
    RepT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            Z64Tensor::Host(x) => Ok(x.placement()?.into()),
            Z64Tensor::Replicated(x) => Ok(x.placement()?.into()),
        }
    }
}
