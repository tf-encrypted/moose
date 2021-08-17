use crate::fixedpoint::{Fixed64Tensor, Fixed128Tensor};
use crate::floatingpoint::{Float32Tensor, Float64Tensor};
use serde::{Deserialize, Serialize};
use macros::ShortName;
use crate::computation::{Signature, HasShortName, Placed, Placement};
use crate::kernels::{PlacementAdd, Session};
use crate::error::Result;
use macros::with_context;

pub enum Tensor {
    Fixed64(Fixed64Tensor),
    Fixed128(Fixed128Tensor),
    Float32(Float32Tensor),
    Float64(Float64Tensor),
}

// impl Placed for Tensor
// {
//     type Placement = Placement;

//     fn placement(&self) -> Result<Self::Placement> {
//         match self {
//             Tensor::Fixed64(x) => Ok(x.placement()?.into()),
//             Tensor::Fixed128(x) => Ok(x.placement()?.into()),
//             Tensor::Float32(x) => Ok(x.placement()?.into()),
//             Tensor::Float64(x) => Ok(x.placement()?.into()),
//         }
//     }
// }

// NOTE(Morten) trying something new here by keeping op structs in dialect file

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct LogicalAddOp {
    pub sig: Signature,
}

// modelled!(PlacementAdd::add, Placement, (Tensor, Tensor) -> Tensor, LogicalAddOp);

// hybrid_kernel! {
//     LogicalAddOp,
//     [
//         (Placement, (Tensor, Tensor) -> Tensor => Self::kernel),
//     ]
// }

impl LogicalAddOp {
    fn kernel<S: Session>(
        sess: &S,
        plc: &Placement,
        x: Tensor,
        y: Tensor,
    ) -> Tensor
    {
        // TODO(Morten) would like to avoid with matching (since one is already done by `hybrid_kernel!`)
        match (x, y) {
            (Tensor::Fixed64(v), Tensor::Fixed64(w)) => {
                let result = with_context!(plc, sess, x + y);
                Tensor::Fixed64(result)
            },
            (Tensor::Fixed128(v), Tensor::Fixed128(w)) => {
                let result = with_context!(plc, sess, x + y);
                Tensor::Fixed128(result)
            },
            _ => unimplemented!() // TODO
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct LogicalSubOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct LogicalMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct LogicalDotOp {
    pub sig: Signature,
}
