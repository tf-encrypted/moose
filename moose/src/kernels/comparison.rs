use super::*;

pub trait PlacementLess<S: Session, T, U, O> {
    fn less(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementLess::less, LessOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> BooleanTensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> BooleanTensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (Float32Tensor, Float32Tensor) -> BooleanTensor => [concrete] Self::float_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> BooleanTensor => [concrete] Self::float_kernel),
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostBitTensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostBitTensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostBitTensor => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostBitTensor => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostBitTensor => [runtime] Self::host_ring64_kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostBitTensor => [runtime] Self::host_ring128_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> BooleanTensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> BooleanTensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_mir_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_mir_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, Mirrored3Ring64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, Mirrored3Ring128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
    ]
}

pub trait PlacementGreater<S: Session, T, U, O> {
    fn greater(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementGreater::greater, GreaterOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> BooleanTensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> BooleanTensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Float32Tensor, Float32Tensor) -> BooleanTensor => [concrete] Self::float_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> BooleanTensor => [concrete] Self::float_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostBitTensor => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostBitTensor => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostBitTensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostBitTensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostBitTensor => [runtime] Self::host_ring64_kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostBitTensor => [runtime] Self::host_ring128_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> BooleanTensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> BooleanTensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_mir_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_mir_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, Mirrored3Ring64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, Mirrored3Ring128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
    ]
}

/// Equality
pub trait PlacementEqual<S: Session, T, U, O> {
    fn equal(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementEqual::equal, EqualOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [transparent] Self::rep_ring_kernel),
    ]
}

pub trait PlacementEqualZero<S: Session, T, O> {
    fn equal_zero(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementEqualZero::equal_zero, EqualZeroOp,
    [
        (ReplicatedPlacement, (ReplicatedBitArray64) -> ReplicatedRing64Tensor => [transparent] Self::bitdec_ring_kernel),
        (ReplicatedPlacement, (ReplicatedBitArray128) -> ReplicatedRing128Tensor => [transparent] Self::bitdec_ring_kernel),
        (ReplicatedPlacement, (ReplicatedBitArray64) -> ReplicatedBitTensor => [transparent] Self::bitdec_bit_kernel),
        (ReplicatedPlacement, (ReplicatedBitArray128) -> ReplicatedBitTensor => [transparent] Self::bitdec_bit_kernel),

    ]
}
