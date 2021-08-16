use crate::bit::BitTensor;
use crate::computation::{HostPlacement, Placed, ReplicatedPlacement, ShapeOp};
use crate::kernels::{PlacementShape};
use crate::replicated::{
    Replicated128Tensor, Replicated64Tensor, ReplicatedBitTensor, ReplicatedShape,
};
use crate::host::{HostShape, HostFloat64Tensor};
use crate::ring::{Ring128Tensor, Ring64Tensor};

modelled!(PlacementShape::shape, HostPlacement, (Ring64Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (Ring128Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (BitTensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (HostFloat64Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (Replicated64Tensor) -> ReplicatedShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (Replicated128Tensor) -> ReplicatedShape, ShapeOp);

kernel! {
    ShapeOp,
    [
        (HostPlacement, (Ring64Tensor) -> HostShape => Self::ring_kernel),
        (HostPlacement, (Ring128Tensor) -> HostShape => Self::ring_kernel),
        (HostPlacement, (BitTensor) -> HostShape => Self::bit_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostShape => Self::host_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedShape => Self::rep_kernel),
        (ReplicatedPlacement, (Replicated64Tensor) -> ReplicatedShape => Self::rep_kernel),
        (ReplicatedPlacement, (Replicated128Tensor) -> ReplicatedShape => Self::rep_kernel),
    ]
}
