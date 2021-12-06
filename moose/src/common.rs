use crate::additive::{AdditiveRing128Tensor, AdditiveRing64Tensor, AdditiveShape};
use crate::computation::SymbolicValue;
use crate::computation::Value;
use crate::computation::{
    AddNOp, AdditivePlacement, HostPlacement, HostReshapeOp, KnownType, Placed,
    ReplicatedPlacement, RingInjectOp, ShapeOp,
};
use crate::fixedpoint::{Fixed128Tensor, Fixed64Tensor};
use crate::floatingpoint::{Float32Tensor, Float64Tensor};
use crate::host::{
    HostBitTensor, HostFixed128Tensor, HostFixed64Tensor, HostFloat32Tensor, HostFloat64Tensor,
    HostInt16Tensor, HostInt32Tensor, HostInt64Tensor, HostInt8Tensor, HostRing128Tensor,
    HostRing64Tensor, HostShape, HostUint16Tensor, HostUint32Tensor, HostUint64Tensor,
    HostUint8Tensor,
};
use crate::kernels::{PlacementAddN, PlacementReshape, PlacementRingInject, PlacementShape};
use crate::logical::Tensor;
use crate::replicated::{
    ReplicatedBitTensor, ReplicatedFixed128Tensor, ReplicatedFixed64Tensor,
    ReplicatedRing128Tensor, ReplicatedRing64Tensor, ReplicatedShape,
};

modelled_kernel! {
    PlacementShape::shape, ShapeOp,
    [
        (HostPlacement, (Tensor) -> HostShape => [hybrid] Self::host_logical_kernel),
        (HostPlacement, (Float32Tensor) -> HostShape => [hybrid] Self::float_kernel),
        (HostPlacement, (Float64Tensor) -> HostShape => [hybrid] Self::float_kernel),
        (HostPlacement, (Fixed64Tensor) -> HostShape => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (Fixed128Tensor) -> HostShape => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostShape => [hybrid] Self::host_hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostShape => [hybrid] Self::host_hostfixed_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostShape => [runtime] Self::ring_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostShape => [runtime] Self::ring_kernel),
        (HostPlacement, (HostBitTensor) -> HostShape => [runtime] Self::bit_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostShape => [runtime] Self::host_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostShape => [runtime] Self::host_kernel),
        (ReplicatedPlacement, (Tensor) -> ReplicatedShape => [hybrid] Self::rep_logical_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> ReplicatedShape => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> ReplicatedShape => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedShape => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedShape => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedShape => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedShape => [hybrid] Self::rep_repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedShape => [hybrid] Self::rep_repfixed_kernel),
        (AdditivePlacement, (AdditiveRing64Tensor) -> AdditiveShape => [hybrid] Self::adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor) -> AdditiveShape => [hybrid] Self::adt_kernel),
    ]
}

modelled_kernel! {
    PlacementReshape::reshape, HostReshapeOp,
    [
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

modelled_kernel! {
    PlacementRingInject::ring_inject, RingInjectOp{bit_idx: usize},
    [
        (HostPlacement, (HostBitTensor) -> HostRing64Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostBitTensor) -> HostRing128Tensor => [runtime] Self::host_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_kernel),
    ]
}

modelled!(PlacementAddN::add_n, HostPlacement, vec[HostRing64Tensor] -> HostRing64Tensor, AddNOp);
modelled!(PlacementAddN::add_n, HostPlacement, vec[HostRing128Tensor] -> HostRing128Tensor, AddNOp);
modelled!(PlacementAddN::add_n, ReplicatedPlacement, vec[ReplicatedRing64Tensor] -> ReplicatedRing64Tensor, AddNOp);
modelled!(PlacementAddN::add_n, ReplicatedPlacement, vec[ReplicatedRing128Tensor] -> ReplicatedRing128Tensor, AddNOp);
modelled!(PlacementAddN::add_n, ReplicatedPlacement, vec[ReplicatedFixed64Tensor] -> ReplicatedFixed64Tensor, AddNOp);
modelled!(PlacementAddN::add_n, ReplicatedPlacement, vec[ReplicatedFixed128Tensor] -> ReplicatedFixed128Tensor, AddNOp);

kernel! {
    AddNOp,
    [
        (HostPlacement, vec[HostRing64Tensor] -> HostRing64Tensor => [runtime] Self::host_kernel),
        (HostPlacement, vec[HostRing128Tensor] -> HostRing128Tensor => [runtime] Self::host_kernel),
        (ReplicatedPlacement, vec[ReplicatedRing64Tensor] -> ReplicatedRing64Tensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, vec[ReplicatedRing128Tensor] -> ReplicatedRing128Tensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed64Tensor] -> ReplicatedFixed64Tensor => [concrete] Self::rep_fixed_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed128Tensor] -> ReplicatedFixed128Tensor => [concrete] Self::rep_fixed_kernel),
    ]
}
