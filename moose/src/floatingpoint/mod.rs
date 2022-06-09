//! Abstraction layer for floating-point values.

use crate::computation::{Placed, Placement};
use crate::error::Result;
use serde::{Deserialize, Serialize};

mod ops;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FloatTensor<HostTenT, Mir3TenT> {
    Host(HostTenT),
    Mirrored3(Mir3TenT),
}

impl<HostTenT, Mir3TenT> Placed for FloatTensor<HostTenT, Mir3TenT>
where
    HostTenT: Placed,
    HostTenT::Placement: Into<Placement>,
    Mir3TenT: Placed,
    Mir3TenT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            FloatTensor::Host(x) => Ok(x.placement()?.into()),
            FloatTensor::Mirrored3(x) => Ok(x.placement()?.into()),
        }
    }
}
