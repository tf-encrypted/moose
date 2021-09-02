use crate::computation::{HostPlacement, Placed, Placement, SymbolicType};
use crate::error::Result;
use crate::host::{HostFloat32Tensor, HostFloat64Tensor};
use crate::symbolic::Symbolic;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FloatTensor<HostT> {
    Host(HostT),
}

pub type Float32Tensor = FloatTensor<HostFloat32Tensor>;

pub type Float64Tensor = FloatTensor<HostFloat64Tensor>;

impl<T> Placed for FloatTensor<T>
where
    T: Placed<Placement = HostPlacement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            FloatTensor::Host(x) => Ok(Placement::Host(x.placement()?)),
        }
    }
}

impl<HostT> SymbolicType for FloatTensor<HostT>
where
    HostT: SymbolicType,
    <HostT as SymbolicType>::Type: Placed<Placement = HostPlacement>,
{
    type Type = Symbolic<FloatTensor<<HostT as SymbolicType>::Type>>;
}

// TODO(lvorona): Not sure why we need this one separately... But the moose_type macro is coming!
impl<T: Placed<Placement = HostPlacement>> From<FloatTensor<T>> for Symbolic<FloatTensor<T>> {
    fn from(x: FloatTensor<T>) -> Self {
        Symbolic::Concrete(x)
    }
}
