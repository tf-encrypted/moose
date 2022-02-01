use super::*;

/// Mux
pub trait PlacementMux<S: Session, T, U, V, O> {
    fn mux(&self, sess: &S, s: &T, x: &U, y: &V) -> O;
}

modelled_kernel! {
    PlacementMux::mux, MuxOp,
    [
        (HostPlacement, (Tensor, Tensor, Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
        (HostPlacement, (BooleanTensor, Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (BooleanTensor, Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (BooleanTensor, Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (BooleanTensor, Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (HostBitTensor, HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [hybrid] Self::host_bit_fixed_kernel),
        (HostPlacement, (HostBitTensor, HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [hybrid] Self::host_bit_fixed_kernel),
        (HostPlacement, (HostBitTensor, HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (HostBitTensor, HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (HostBitTensor, HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::host_float_int_kernel),
        (HostPlacement, (HostBitTensor, HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::host_float_int_kernel),
        (HostPlacement, (HostBitTensor, HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::host_float_int_kernel),
        (HostPlacement, (HostBitTensor, HostInt32Tensor, HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::host_float_int_kernel),
        (HostPlacement, (HostBitTensor, HostInt16Tensor, HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::host_float_int_kernel),
        (ReplicatedPlacement, (Tensor, Tensor, Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (BooleanTensor, Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (BooleanTensor, Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::rep_bit_selector_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_bit_selector_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor  => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor  => [concrete] Self::rep_bit_selector_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_bit_selector_kernel),
    ]
}
