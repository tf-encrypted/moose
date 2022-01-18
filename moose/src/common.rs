use crate::additive::AdditivePlacement;
use crate::computation::SymbolicValue;
use crate::computation::Value;
use crate::computation::{
    AddNOp, ConcatOp, HostPlacement, HostReshapeOp, KnownType, Placed, ShapeOp,
};
use crate::kernels::{PlacementAddN, PlacementConcatenate, PlacementReshape, PlacementShape};
use crate::replicated::ReplicatedPlacement;
use crate::types::*;

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
        (AdditivePlacement, (AdditiveRing64Tensor) -> AdditiveShape => [concrete] Self::adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor) -> AdditiveShape => [concrete] Self::adt_kernel),
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

modelled!(PlacementAddN::add_n, HostPlacement, vec[Tensor] -> Tensor, AddNOp);
modelled!(PlacementAddN::add_n, HostPlacement, vec[Float32Tensor] -> Float32Tensor, AddNOp);
modelled!(PlacementAddN::add_n, HostPlacement, vec[Float64Tensor] -> Float64Tensor, AddNOp);
modelled!(PlacementAddN::add_n, HostPlacement, vec[HostFloat32Tensor] -> HostFloat32Tensor, AddNOp);
modelled!(PlacementAddN::add_n, HostPlacement, vec[HostFloat64Tensor] -> HostFloat64Tensor, AddNOp);
modelled!(PlacementAddN::add_n, HostPlacement, vec[Fixed64Tensor] -> Fixed64Tensor, AddNOp);
modelled!(PlacementAddN::add_n, HostPlacement, vec[Fixed128Tensor] -> Fixed128Tensor, AddNOp);
modelled!(PlacementAddN::add_n, HostPlacement, vec[HostFixed64Tensor] -> HostFixed64Tensor, AddNOp);
modelled!(PlacementAddN::add_n, HostPlacement, vec[HostFixed128Tensor] -> HostFixed128Tensor, AddNOp);
modelled!(PlacementAddN::add_n, HostPlacement, vec[HostRing64Tensor] -> HostRing64Tensor, AddNOp);
modelled!(PlacementAddN::add_n, HostPlacement, vec[HostRing128Tensor] -> HostRing128Tensor, AddNOp);
modelled!(PlacementAddN::add_n, ReplicatedPlacement, vec[ReplicatedRing64Tensor] -> ReplicatedRing64Tensor, AddNOp);
modelled!(PlacementAddN::add_n, ReplicatedPlacement, vec[ReplicatedRing128Tensor] -> ReplicatedRing128Tensor, AddNOp);
modelled!(PlacementAddN::add_n, ReplicatedPlacement, vec[ReplicatedFixed64Tensor] -> ReplicatedFixed64Tensor, AddNOp);
modelled!(PlacementAddN::add_n, ReplicatedPlacement, vec[ReplicatedFixed128Tensor] -> ReplicatedFixed128Tensor, AddNOp);
modelled!(PlacementAddN::add_n, ReplicatedPlacement, vec[Fixed64Tensor] -> Fixed64Tensor, AddNOp);
modelled!(PlacementAddN::add_n, ReplicatedPlacement, vec[Fixed128Tensor] -> Fixed128Tensor, AddNOp);
modelled!(PlacementAddN::add_n, ReplicatedPlacement, vec[Tensor] -> Tensor, AddNOp);

kernel! {
    AddNOp,
    [
        (HostPlacement, vec[Tensor] -> Tensor => [concrete] Self::host_logical_kernel),
        (HostPlacement, vec[Float32Tensor] -> Float32Tensor => [concrete] Self::float_kernel),
        (HostPlacement, vec[Float64Tensor] -> Float64Tensor => [concrete] Self::float_kernel),
        (HostPlacement, vec[HostFloat32Tensor] -> HostFloat32Tensor => [runtime] Self::host_float_kernel),
        (HostPlacement, vec[HostFloat64Tensor] -> HostFloat64Tensor => [runtime] Self::host_float_kernel),
        (HostPlacement, vec[Fixed64Tensor] -> Fixed64Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, vec[Fixed128Tensor] -> Fixed128Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, vec[HostFixed64Tensor] -> HostFixed64Tensor => [concrete] Self::host_fixed_kernel),
        (HostPlacement, vec[HostFixed128Tensor] -> HostFixed128Tensor => [concrete] Self::host_fixed_kernel),
        (HostPlacement, vec[HostRing64Tensor] -> HostRing64Tensor => [runtime] Self::host_kernel),
        (HostPlacement, vec[HostRing128Tensor] -> HostRing128Tensor => [runtime] Self::host_kernel),
        (ReplicatedPlacement, vec[ReplicatedRing64Tensor] -> ReplicatedRing64Tensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, vec[ReplicatedRing128Tensor] -> ReplicatedRing128Tensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed64Tensor] -> ReplicatedFixed64Tensor => [concrete] Self::rep_fixed_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed128Tensor] -> ReplicatedFixed128Tensor => [concrete] Self::rep_fixed_kernel),
        (ReplicatedPlacement, vec[Fixed64Tensor] -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, vec[Fixed128Tensor] -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, vec[Tensor] -> Tensor => [concrete] Self::logical_rep_kernel),
    ]
}

modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[Tensor] -> Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[HostFloat32Tensor] -> HostFloat32Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[HostFloat64Tensor] -> HostFloat64Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[HostInt8Tensor] -> HostInt8Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[HostInt16Tensor] -> HostInt16Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[HostInt32Tensor] -> HostInt32Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[HostInt64Tensor] -> HostInt64Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[HostRing64Tensor] -> HostRing64Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[HostRing128Tensor] -> HostRing128Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[ReplicatedRing64Tensor] -> ReplicatedRing64Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[ReplicatedRing128Tensor] -> ReplicatedRing128Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[ReplicatedFixed64Tensor] -> ReplicatedFixed64Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[ReplicatedFixed128Tensor] -> ReplicatedFixed128Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[Fixed64Tensor] -> Fixed64Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[Fixed128Tensor] -> Fixed128Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[Tensor] -> Tensor, ConcatOp);

kernel! {
    ConcatOp, [
        (HostPlacement, vec[Tensor] -> Tensor => [concrete] attributes[axis] Self::host_kernel),
        (HostPlacement, vec[HostFloat32Tensor] -> HostFloat32Tensor => [runtime] attributes[axis] Self::kernel),
        (HostPlacement, vec[HostFloat64Tensor] -> HostFloat64Tensor => [runtime] attributes[axis] Self::kernel),
        (HostPlacement, vec[HostInt8Tensor] -> HostInt8Tensor => [runtime] attributes[axis] Self::kernel),
        (HostPlacement, vec[HostInt16Tensor] -> HostInt16Tensor => [runtime] attributes[axis] Self::kernel),
        (HostPlacement, vec[HostInt32Tensor] -> HostInt32Tensor => [runtime] attributes[axis] Self::kernel),
        (HostPlacement, vec[HostInt64Tensor] -> HostInt64Tensor => [runtime] attributes[axis] Self::kernel),
        (HostPlacement, vec[HostRing64Tensor] -> HostRing64Tensor => [runtime] attributes[axis] Self::ring_kernel),
        (HostPlacement, vec[HostRing128Tensor] -> HostRing128Tensor => [runtime] attributes[axis] Self::ring_kernel),
        (ReplicatedPlacement, vec[ReplicatedRing64Tensor] -> ReplicatedRing64Tensor => [concrete] attributes[axis] Self::rep_rep_kernel),
        (ReplicatedPlacement, vec[ReplicatedRing128Tensor] -> ReplicatedRing128Tensor => [concrete] attributes[axis] Self::rep_rep_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed64Tensor] -> ReplicatedFixed64Tensor => [concrete] attributes[axis] Self::rep_fixed_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed128Tensor] -> ReplicatedFixed128Tensor => [concrete] attributes[axis] Self::rep_fixed_kernel),
        (ReplicatedPlacement, vec[Fixed64Tensor] -> Fixed64Tensor => [concrete] attributes[axis] Self::fixed_rep_kernel),
        (ReplicatedPlacement, vec[Fixed128Tensor] -> Fixed128Tensor => [concrete] attributes[axis] Self::fixed_rep_kernel),
        (ReplicatedPlacement, vec[Tensor] -> Tensor => [concrete] attributes[axis] Self::logical_rep_kernel),
    ]
}
