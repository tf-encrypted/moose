//! Abstraction layer for AES encrypted values

use crate::computation::{Placed, Placement};
use crate::error::Result;
use crate::host::{HostAesKey, HostFixed128AesTensor};
use crate::replicated::ReplicatedAesKey;
use serde::{Deserialize, Serialize};

mod ops;

/// Logical AES key
///
/// This abstracts over an AES key that either lives on a host or
/// replicated placement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AbstractAesKey<HostKeyT, RepKeyT> {
    Host(HostKeyT),
    Replicated(RepKeyT),
}

moose_type!(AesKey = AbstractAesKey<HostAesKey, ReplicatedAesKey>);

impl<HostKeyT, RepKeyT> Placed for AbstractAesKey<HostKeyT, RepKeyT>
where
    HostKeyT: Placed,
    HostKeyT::Placement: Into<Placement>,
    RepKeyT: Placed,
    RepKeyT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            AbstractAesKey::Host(x) => Ok(x.placement()?.into()),
            AbstractAesKey::Replicated(x) => Ok(x.placement()?.into()),
        }
    }
}

/// AES encrypted logical tensor
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AbstractAesTensor<Fixed128AesT> {
    Fixed128(Fixed128AesT),
}

moose_type!(AesTensor = AbstractAesTensor<Fixed128AesTensor>);

impl<Fixed128T> Placed for AbstractAesTensor<Fixed128T>
where
    Fixed128T: Placed,
    Fixed128T::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            AbstractAesTensor::Fixed128(x) => Ok(x.placement()?.into()),
        }
    }
}

/// AES encrypted fixed-point tensor
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FixedAesTensor<HostFixedAesT> {
    Host(HostFixedAesT),
}

moose_type!(Fixed128AesTensor = FixedAesTensor<HostFixed128AesTensor>);

impl<HostFixedAesT> Placed for FixedAesTensor<HostFixedAesT>
where
    HostFixedAesT: Placed,
    HostFixedAesT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            FixedAesTensor::Host(x) => Ok(x.placement()?.into()),
        }
    }
}
