use crate::additive::{AdditiveRing128Tensor, AdditiveRing64Tensor, AdditiveShape};
use crate::computation::{
    AdditivePlacement, HostPlacement, HostReshapeOp, Placed, ReplicatedPlacement, RingInjectOp,
    ShapeOp,
};
use crate::host::{
    HostBitTensor, HostFloat32Tensor, HostFloat64Tensor, HostInt16Tensor, HostInt32Tensor,
    HostInt64Tensor, HostInt8Tensor, HostRing128Tensor, HostRing64Tensor, HostShape,
    HostUint16Tensor, HostUint32Tensor, HostUint64Tensor, HostUint8Tensor,
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
        (HostPlacement, (HostRing64Tensor) -> HostShape => [runtime] Self::ring_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostShape => [runtime] Self::ring_kernel),
        (HostPlacement, (HostBitTensor) -> HostShape => [runtime] Self::bit_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostShape => [runtime] Self::host_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedShape => [runtime] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedShape => [runtime] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedShape => [runtime] Self::rep_kernel),
        (AdditivePlacement, (AdditiveRing64Tensor) -> AdditiveShape => [runtime] Self::adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor) -> AdditiveShape => [runtime] Self::adt_kernel),
    ]
}

modelled!(PlacementReshape::reshape, HostPlacement, (HostRing64Tensor, HostShape) -> HostRing64Tensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostRing128Tensor, HostShape) -> HostRing128Tensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostBitTensor, HostShape) -> HostBitTensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostFloat32Tensor, HostShape) -> HostFloat32Tensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostFloat64Tensor, HostShape) -> HostFloat64Tensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostInt8Tensor, HostShape) -> HostInt8Tensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostInt16Tensor, HostShape) -> HostInt16Tensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostInt32Tensor, HostShape) -> HostInt32Tensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostInt64Tensor, HostShape) -> HostInt64Tensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostUint8Tensor, HostShape) -> HostUint8Tensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostUint16Tensor, HostShape) -> HostUint16Tensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostUint32Tensor, HostShape) -> HostUint32Tensor, HostReshapeOp);
modelled!(PlacementReshape::reshape, HostPlacement, (HostUint64Tensor, HostShape) -> HostUint64Tensor, HostReshapeOp);

kernel! {
    HostReshapeOp, [
        (HostPlacement, (HostRing64Tensor, HostShape) -> HostRing64Tensor => [runtime] Self::ring_kernel),
        (HostPlacement, (HostRing128Tensor, HostShape) -> HostRing128Tensor => [runtime] Self::ring_kernel),
        (HostPlacement, (HostBitTensor, HostShape) -> HostBitTensor => [runtime] Self::bit_kernel),
        (HostPlacement, (HostFloat32Tensor, HostShape) -> HostFloat32Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostFloat64Tensor, HostShape) -> HostFloat64Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostInt8Tensor, HostShape) -> HostInt8Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostInt16Tensor, HostShape) -> HostInt16Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostInt32Tensor, HostShape) -> HostInt32Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostInt64Tensor, HostShape) -> HostInt64Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostUint8Tensor, HostShape) -> HostUint8Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostUint16Tensor, HostShape) -> HostUint16Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostUint32Tensor, HostShape) -> HostUint32Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostUint64Tensor, HostShape) -> HostUint64Tensor => [runtime] Self::host_kernel),
    ]
}

modelled!(PlacementRingInject::ring_inject, HostPlacement, attributes[bit_idx: usize] (HostBitTensor) -> HostRing64Tensor, RingInjectOp);
modelled!(PlacementRingInject::ring_inject, HostPlacement, attributes[bit_idx: usize] (HostBitTensor) -> HostRing128Tensor, RingInjectOp);
modelled!(PlacementRingInject::ring_inject, ReplicatedPlacement, attributes[bit_idx: usize] (ReplicatedBitTensor) -> ReplicatedRing64Tensor, RingInjectOp);
modelled!(PlacementRingInject::ring_inject, ReplicatedPlacement, attributes[bit_idx: usize] (ReplicatedBitTensor) -> ReplicatedRing128Tensor, RingInjectOp);

kernel! {
    RingInjectOp,
    [
        (HostPlacement, (HostBitTensor) -> HostRing64Tensor => [runtime] attributes[bit_idx] Self::host_kernel),
        (HostPlacement, (HostBitTensor) -> HostRing128Tensor => [runtime] attributes[bit_idx] Self::host_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedRing64Tensor => [runtime] attributes[bit_idx] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedRing128Tensor => [runtime] attributes[bit_idx] Self::rep_kernel),
    ]
}
