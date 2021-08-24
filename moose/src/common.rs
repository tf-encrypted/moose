use crate::additive::{AdditiveRing128Tensor, AdditiveRing64Tensor, AdditiveShape};
use crate::computation::{
    AdditivePlacement, HostPlacement, HostReshapeOp, Placed, ReplicatedPlacement, RingInjectOp,
    ShapeOp,
};
use crate::host::{
    HostBitTensor, HostFloat64Tensor, HostRing128Tensor, HostRing64Tensor, HostShape,
};
use crate::kernels::PlacementShape;
use crate::kernels::{PlacementReshape, PlacementRingInject};
use crate::replicated::{
    ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing64Tensor, ReplicatedShape,
};

modelled!(PlacementShape::shape, HostPlacement, (HostRing64Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (HostRing128Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (HostBitTensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, HostPlacement, (HostFloat64Tensor) -> HostShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedShape, ShapeOp);
modelled!(PlacementShape::shape, ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedShape, ShapeOp);
modelled!(PlacementShape::shape, AdditivePlacement, (AdditiveRing64Tensor) -> AdditiveShape, ShapeOp);
modelled!(PlacementShape::shape, AdditivePlacement, (AdditiveRing128Tensor) -> AdditiveShape, ShapeOp);

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
        (AdditivePlacement, (AdditiveRing64Tensor) -> AdditiveShape => Self::adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor) -> AdditiveShape => Self::adt_kernel),
    ]
}

modelled!(PlacementReshape::reshape, HostPlacement, (HostFloat64Tensor, HostShape) -> HostFloat64Tensor, HostReshapeOp);

kernel! {
    HostReshapeOp, [
        (HostPlacement, (HostFloat64Tensor, HostShape) -> HostFloat64Tensor => Self::host_kernel),
    ]
}

modelled!(PlacementRingInject::ring_inject, HostPlacement, attributes[bit_idx: usize] (HostBitTensor) -> HostRing64Tensor, RingInjectOp);
modelled!(PlacementRingInject::ring_inject, HostPlacement, attributes[bit_idx: usize] (HostBitTensor) -> HostRing128Tensor, RingInjectOp);
modelled!(PlacementRingInject::ring_inject, ReplicatedPlacement, attributes[bit_idx: usize] (ReplicatedBitTensor) -> ReplicatedRing64Tensor, RingInjectOp);
modelled!(PlacementRingInject::ring_inject, ReplicatedPlacement, attributes[bit_idx: usize] (ReplicatedBitTensor) -> ReplicatedRing128Tensor, RingInjectOp);

kernel! {
    RingInjectOp,
    [
        (HostPlacement, (HostBitTensor) -> HostRing64Tensor => attributes[bit_idx] Self::host_kernel),
        (HostPlacement, (HostBitTensor) -> HostRing128Tensor => attributes[bit_idx] Self::host_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedRing64Tensor => attributes[bit_idx] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedRing128Tensor => attributes[bit_idx] Self::rep_kernel),
    ]
}
