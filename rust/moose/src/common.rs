use crate::computation::{HostPlacement, Placed, ReplicatedPlacement, ShapeOp};
use crate::host::{
    HostBitTensor, HostFloat64Tensor, HostRing128Tensor, HostRing64Tensor, HostShape,
};
use crate::kernels::PlacementShape;
use crate::replicated::{
    ReplicatedRing128Tensor, ReplicatedRing64Tensor, ReplicatedBitTensor, ReplicatedShape,
};

modelled!(PlacementShape::shape, HostPlacement, (HostRing64Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (HostRing128Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (HostBitTensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (HostFloat64Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedShape, ShapeOp);

kernel! {
    ShapeOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostShape => Self::ring_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostShape => Self::ring_kernel),
        (HostPlacement, (HostBitTensor) -> HostShape => Self::bit_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostShape => Self::host_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedShape => Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedShape => Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedShape => Self::rep_kernel),
    ]
}
