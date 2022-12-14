use super::*;
use crate::host::SliceInfo;

pub trait PlacementIndexAxis<S: Session, T, O> {
    fn index_axis(&self, sess: &S, axis: usize, index: usize, x: &T) -> O;
}

modelled_kernel! {
    PlacementIndexAxis::index_axis, IndexAxisOp{axis: usize, index: usize},
    [
        (HostPlacement, (BooleanTensor) -> BooleanTensor => [concrete] Self::bool_host_kernel),
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::host_bit_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
        (ReplicatedPlacement, (BooleanTensor) -> BooleanTensor => [concrete]  Self::bool_rep_kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete]  Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete]  Self::rep_kernel),
    ]
}

pub trait PlacementIndex<S: Session, T, O> {
    fn index(&self, sess: &S, index: usize, x: &T) -> O;
}

modelled_kernel! {
    PlacementIndex::index, IndexOp{index: usize},
    [
        (ReplicatedPlacement, (ReplicatedBitArray64) -> ReplicatedBitTensor => [hybrid] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitArray128) -> ReplicatedBitTensor => [hybrid] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitArray224) -> ReplicatedBitTensor => [hybrid] Self::rep_kernel),
        (HostPlacement, (HostBitArray64) -> HostBitTensor => [hybrid] Self::host_kernel),
        (HostPlacement, (HostBitArray128) -> HostBitTensor => [hybrid] Self::host_kernel),
        (HostPlacement, (HostBitArray224) -> HostBitTensor => [hybrid] Self::host_kernel),
        (HostPlacement, (HostBitArray256) -> HostBitTensor => [hybrid] Self::host_kernel),
    ]
}

pub trait PlacementSelect<S: Session, I, T, O> {
    fn select(&self, sess: &S, axis: usize, index: &I, x: &T) -> O;
}

modelled_kernel! {
    PlacementSelect::select, SelectOp{axis: usize},
    [
        // (HostPlacement, (BooleanTensor,BooleanTensor) -> BooleanTensor => [concrete] Self::bool_host_kernel),
        (HostPlacement, (BooleanTensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (BooleanTensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (BooleanTensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (BooleanTensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (HostBitTensor, HostFixed64Tensor) -> HostFixed64Tensor => [hybrid]  Self::hostfixed_kernel),
        (HostPlacement, (HostBitTensor, HostFixed128Tensor) -> HostFixed128Tensor => [hybrid] Self::hostfixed_kernel),
        // (HostPlacement, (BooleanTensor, HostBitTensor) -> HostBitTensor => [runtime] Self::host_bit_kernel),
        (HostPlacement, (HostBitTensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostBitTensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostBitTensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (HostBitTensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
    ]
}

pub trait PlacementSlice<S: Session, T, O> {
    fn slice(&self, sess: &S, slice_info: SliceInfo, x: &T) -> O;
}

modelled_kernel! {
    PlacementSlice::slice, SliceOp{slice: SliceInfo},
    [
        // runtime kernels
        (HostPlacement, (HostShape) -> HostShape => [runtime] Self::shape_kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::host_bit_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::host_generic_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::host_generic_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (HostUint64Tensor) -> HostUint64Tensor => [runtime] Self::host_generic_kernel),
        // host lowering kernels
        (HostPlacement, (BooleanTensor) -> BooleanTensor => [concrete] Self::bool_host_kernel),
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::host_fixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::host_fixed_kernel),
        (HostPlacement, (Shape) -> Shape => [concrete]  Self::logical_host_shape),
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
        (HostPlacement, (Uint64Tensor) -> Uint64Tensor => [concrete] Self::u64_host_kernel),
        // replicated kernels
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedShape => [concrete] Self::rep_shape_kernel),
        // replicated lowering kernels
        (ReplicatedPlacement, (BooleanTensor) -> BooleanTensor => [concrete] Self::bool_rep_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedUint64Tensor) -> ReplicatedUint64Tensor => [concrete] Self::rep_uint_kernel),
        (ReplicatedPlacement, (Shape) -> Shape => [concrete] Self::logical_rep_shape),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (Uint64Tensor) -> Uint64Tensor => [concrete] Self::u64_rep_kernel),
    ]
}
