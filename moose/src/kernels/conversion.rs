use super::*;

pub trait PlacementCast<S: Session, T, O> {
    fn cast(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementCast::cast, CastOp,
    [
        // standard dtype conversions
        // lowering kernels
        (HostPlacement, (Tensor) -> Tensor => [concrete] attributes[sig] Self::kernel),
        (HostPlacement, (BooleanTensor) -> Float32Tensor => [concrete] Self::bool_float_kernel),
        (HostPlacement, (BooleanTensor) -> Float64Tensor => [concrete] Self::bool_float_kernel),
        (HostPlacement, (BooleanTensor) -> Uint64Tensor => [concrete] Self::bool_u64_kernel),
        (HostPlacement, (Float32Tensor) -> BooleanTensor => [concrete] Self::float_bool_kernel),
        (HostPlacement, (Float32Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float32Tensor) -> Uint64Tensor => [concrete] Self::float_u64_kernel),
        (HostPlacement, (Float64Tensor) -> BooleanTensor => [concrete] Self::float_bool_kernel),
        (HostPlacement, (Float64Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Uint64Tensor => [concrete] Self::float_u64_kernel),
        (HostPlacement, (Uint64Tensor) -> BooleanTensor => [concrete] Self::u64_bool_kernel),
        (HostPlacement, (Uint64Tensor) -> Float32Tensor => [concrete] Self::u64_float_kernel),
        (HostPlacement, (Uint64Tensor) -> Float64Tensor => [concrete] Self::u64_float_kernel),
        // runtime kernels
        (HostPlacement, (HostBitTensor) -> HostFloat32Tensor => [runtime] Self::from_bool_host_kernel),
        (HostPlacement, (HostBitTensor) -> HostFloat64Tensor => [runtime] Self::from_bool_host_kernel),
        (HostPlacement, (HostBitTensor) -> HostUint64Tensor => [runtime] Self::from_bool_host_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostBitTensor => [runtime] Self::f32_bool_host_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFloat64Tensor => [runtime] Self::standard_host_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostUint64Tensor => [runtime] Self::standard_host_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostBitTensor => [runtime] Self::f64_bool_host_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat32Tensor => [runtime] Self::standard_host_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostUint64Tensor => [runtime] Self::standard_host_kernel),
        (HostPlacement, (HostUint64Tensor) -> HostBitTensor => [runtime] Self::u64_bool_host_kernel),
        (HostPlacement, (HostUint64Tensor) -> HostFloat32Tensor => [runtime] Self::standard_host_kernel),
        (HostPlacement, (HostUint64Tensor) -> HostFloat64Tensor => [runtime] Self::standard_host_kernel),
        // mirrored casting
        (Mirrored3Placement, (Tensor) -> Tensor => [concrete] attributes[sig] Self::mir_kernel),
        // ring conversions
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::no_op_reduction_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostUint64Tensor => [runtime] Self::hr64_hu64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing64Tensor => [runtime] Self::ring_reduction_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_reduction_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_reduction_kernel),
    ]
}

/// Secret share value
pub trait PlacementShare<S: Session, T, O> {
    fn share(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementShare::share, ShareOp,
    [
        (ReplicatedPlacement, (HostFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::fixed_kernel),
        (ReplicatedPlacement, (HostFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::fixed_kernel),
        (ReplicatedPlacement, (HostRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::ring_kernel),
        (ReplicatedPlacement, (HostRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::ring_kernel),
        (ReplicatedPlacement, (HostBitTensor) -> ReplicatedBitTensor => [hybrid] Self::ring_kernel),
        (ReplicatedPlacement, (HostBitArray64) -> ReplicatedBitArray64 => [concrete] Self::array_kernel),
        (ReplicatedPlacement, (HostBitArray128) -> ReplicatedBitArray128 => [concrete] Self::array_kernel),
        (ReplicatedPlacement, (HostBitArray224) -> ReplicatedBitArray224 => [concrete] Self::array_kernel),
        (ReplicatedPlacement, (HostAesKey) -> ReplicatedAesKey => [concrete] Self::aeskey_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::fixed_mir_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::fixed_mir_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::ring_mir_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::ring_mir_kernel),
        (ReplicatedPlacement, (HostShape) -> ReplicatedShape => [hybrid] Self::shape_kernel),
    ]
}

/// Reveal secret shared value
pub trait PlacementReveal<S: Session, T, O> {
    fn reveal(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementReveal::reveal, RevealOp,
    [
        (HostPlacement, (ReplicatedFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::host_fixed_kernel),
        (HostPlacement, (ReplicatedFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::host_fixed_kernel),
        (HostPlacement, (ReplicatedRing64Tensor) -> HostRing64Tensor => [hybrid] Self::host_ring_kernel),
        (HostPlacement, (ReplicatedRing128Tensor) -> HostRing128Tensor => [hybrid] Self::host_ring_kernel),
        (HostPlacement, (ReplicatedUint64Tensor) -> HostUint64Tensor => [hybrid] Self::host_uint64_kernel),
        (HostPlacement, (ReplicatedBitTensor) -> HostBitTensor => [hybrid] Self::host_ring_kernel),
        (HostPlacement, (ReplicatedBitArray64) -> HostBitArray64 => [concrete] Self::host_bit_array_kernel),
        (HostPlacement, (ReplicatedBitArray128) -> HostBitArray128 => [concrete] Self::host_bit_array_kernel),
        (HostPlacement, (ReplicatedBitArray224) -> HostBitArray224 => [concrete] Self::host_bit_array_kernel),
        (HostPlacement, (ReplicatedAesKey) -> HostAesKey => [concrete] Self::host_aeskey_kernel),
        (Mirrored3Placement, (ReplicatedBitTensor) -> Mirrored3BitTensor => [concrete] Self::mir_ring_kernel),
        (Mirrored3Placement, (ReplicatedRing64Tensor) -> Mirrored3Ring64Tensor => [concrete] Self::mir_ring_kernel),
        (Mirrored3Placement, (ReplicatedRing128Tensor) -> Mirrored3Ring128Tensor => [concrete] Self::mir_ring_kernel),
        (Mirrored3Placement, (ReplicatedFixed64Tensor) -> Mirrored3Fixed64Tensor => [concrete] Self::mir_fixed_kernel),
        (Mirrored3Placement, (ReplicatedFixed128Tensor) -> Mirrored3Fixed128Tensor => [concrete] Self::mir_fixed_kernel),
        (HostPlacement, (AdditiveRing64Tensor) -> HostRing64Tensor => [hybrid] Self::host_adt_kernel),
        (HostPlacement, (AdditiveRing128Tensor) -> HostRing128Tensor => [hybrid] Self::host_adt_kernel),
        (HostPlacement, (AdditiveBitTensor) -> HostBitTensor => [hybrid] Self::host_adt_kernel),
        (HostPlacement, (ReplicatedShape) -> HostShape => [hybrid] Self::shape_kernel),
    ]
}

pub trait PlacementMirror<S: Session, T, O> {
    fn mirror(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementMirror::mirror, MirrorOp,
    [
        (Mirrored3Placement, (HostFixed64Tensor) -> Mirrored3Fixed64Tensor => [concrete] Self::fixed_kernel),
        (Mirrored3Placement, (HostFixed128Tensor) -> Mirrored3Fixed128Tensor => [concrete] Self::fixed_kernel),
        (Mirrored3Placement, (HostFloat32Tensor) -> Mirrored3Float32 => [hybrid] Self::kernel),
        (Mirrored3Placement, (HostFloat64Tensor) -> Mirrored3Float64 => [hybrid] Self::kernel),
        (Mirrored3Placement, (HostRing64Tensor) -> Mirrored3Ring64Tensor => [hybrid] Self::kernel),
        (Mirrored3Placement, (HostRing128Tensor) -> Mirrored3Ring128Tensor => [hybrid] Self::kernel),
    ]
}

pub trait PlacementDemirror<S: Session, T, O> {
    fn demirror(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementDemirror::demirror, DemirrorOp,
    [
        (HostPlacement, (Mirrored3BitTensor) -> HostBitTensor => [hybrid] Self::kernel),
        (HostPlacement, (Mirrored3Fixed64Tensor) -> HostFixed64Tensor => [hybrid] Self::fixed_kernel),
        (HostPlacement, (Mirrored3Fixed128Tensor) -> HostFixed128Tensor => [hybrid] Self::fixed_kernel),
        (HostPlacement, (Mirrored3Float32) -> HostFloat32Tensor => [hybrid] Self::kernel),
        (HostPlacement, (Mirrored3Float64) -> HostFloat64Tensor => [hybrid] Self::kernel),
        (HostPlacement, (Mirrored3Ring64Tensor) -> HostRing64Tensor => [hybrid] Self::kernel),
        (HostPlacement, (Mirrored3Ring128Tensor) -> HostRing128Tensor => [hybrid] Self::kernel),
    ]
}

pub trait PlacementRepToAdt<S: Session, T, O> {
    fn rep_to_adt(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementRepToAdt::rep_to_adt, RepToAdtOp,
    [
        (AdditivePlacement, (ReplicatedRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::rep_to_adt_kernel),
        (AdditivePlacement, (ReplicatedRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::rep_to_adt_kernel),
        (AdditivePlacement, (ReplicatedBitTensor) -> AdditiveBitTensor => [concrete] Self::rep_to_adt_kernel),
    ]
}

pub trait PlacementAdtToRep<S: Session, T, O> {
    fn adt_to_rep(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementAdtToRep::adt_to_rep, AdtToRepOp,
    [
        (ReplicatedPlacement, (AdditiveRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::kernel),
        (ReplicatedPlacement, (AdditiveRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::kernel),
    ]
}

pub trait PlacementDecrypt<S: Session, KeyT, C, O> {
    fn decrypt(&self, sess: &S, key: &KeyT, ciphertext: &C) -> O;
}

modelled_kernel! {
    PlacementDecrypt::decrypt, DecryptOp,
    [
        (HostPlacement, (AesKey, AesTensor) -> Tensor => [hybrid] Self::host_kernel),
        (HostPlacement, (HostAesKey, AesTensor) -> Tensor => [hybrid] Self::host_key_kernel),
        (HostPlacement, (HostAesKey, Fixed128AesTensor) -> Fixed128Tensor => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostAesKey, HostFixed128AesTensor) -> HostFixed128Tensor => [concrete] Self::host_fixed_aes_kernel),
        (ReplicatedPlacement, (AesKey, AesTensor) -> Tensor => [hybrid] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, AesTensor) -> Tensor => [hybrid] Self::rep_key_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, Fixed128AesTensor) -> Fixed128Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, HostFixed128AesTensor) -> ReplicatedFixed128Tensor => [concrete] Self::rep_fixed_aes_kernel),
    ]
}

pub trait PlacementRingFixedpointEncode<S: Session, T, O> {
    fn fixedpoint_ring_encode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
}

modelled_kernel! {
    PlacementRingFixedpointEncode::fixedpoint_ring_encode, RingFixedpointEncodeOp{scaling_base: u64, scaling_exp: u32},
    [
        (HostPlacement, (HostFloat32Tensor) -> HostRing64Tensor => [runtime] Self::float32_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostRing128Tensor => [runtime] Self::float64_kernel),
        (Mirrored3Placement, (Mirrored3Float32) -> Mirrored3Ring64Tensor => [concrete] Self::mir_kernel),
        (Mirrored3Placement, (Mirrored3Float64) -> Mirrored3Ring128Tensor => [concrete] Self::mir_kernel),
    ]
}

pub trait PlacementRingFixedpointDecode<S: Session, T, O> {
    fn fixedpoint_ring_decode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
}

modelled_kernel! {
    PlacementRingFixedpointDecode::fixedpoint_ring_decode, RingFixedpointDecodeOp{scaling_base: u64, scaling_exp: u32},
    [
        (HostPlacement, (HostRing64Tensor) -> HostFloat32Tensor => [runtime] Self::float32_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostFloat64Tensor => [runtime] Self::float64_kernel),
        (Mirrored3Placement, (Mirrored3Ring64Tensor) -> Mirrored3Float32 => [concrete] Self::mir_kernel),
        (Mirrored3Placement, (Mirrored3Ring128Tensor) -> Mirrored3Float64 => [concrete] Self::mir_kernel),
    ]
}

pub trait PlacementFixedpointEncode<S: Session, T, O> {
    fn fixedpoint_encode(
        &self,
        sess: &S,
        fractional_precision: u32,
        integral_precision: u32,
        x: &T,
    ) -> O;
}

modelled_kernel! {
    PlacementFixedpointEncode::fixedpoint_encode, FixedpointEncodeOp{fractional_precision: u32, integral_precision: u32},
    [
        (HostPlacement, (Float32Tensor) -> Fixed64Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (Float64Tensor) -> Fixed128Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFixed64Tensor => [hybrid] Self::hostfixed_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFixed128Tensor => [hybrid] Self::hostfixed_kernel),
        (Mirrored3Placement, (Float32Tensor) -> Fixed64Tensor => [concrete] Self::mir_fixed_kernel),
        (Mirrored3Placement, (Float64Tensor) -> Fixed128Tensor => [concrete] Self::mir_fixed_kernel),
        (Mirrored3Placement, (Mirrored3Float32) -> Mirrored3Fixed64Tensor => [hybrid] Self::mir_fixed_lower_kernel),
        (Mirrored3Placement, (Mirrored3Float64) -> Mirrored3Fixed128Tensor => [hybrid] Self::mir_fixed_lower_kernel),
    ]
}

pub trait PlacementFixedpointDecode<S: Session, T, O> {
    fn fixedpoint_decode(&self, sess: &S, precision: u32, x: &T) -> O;
}

modelled_kernel! {
    PlacementFixedpointDecode::fixedpoint_decode, FixedpointDecodeOp{fractional_precision: u32},
    [
        (HostPlacement, (Fixed64Tensor) -> Float32Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (Fixed128Tensor) -> Float64Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFloat32Tensor => [hybrid] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFloat64Tensor => [hybrid] Self::hostfixed_kernel),
        (Mirrored3Placement, (Fixed64Tensor) -> Float32Tensor => [concrete] Self::mir_fixed_kernel),
        (Mirrored3Placement, (Fixed128Tensor) -> Float64Tensor => [concrete] Self::mir_fixed_kernel),
        (Mirrored3Placement, (Mirrored3Fixed64Tensor) -> Mirrored3Float32 => [hybrid] Self::mir_fixed_lower_kernel),
        (Mirrored3Placement, (Mirrored3Fixed128Tensor) -> Mirrored3Float64 => [hybrid] Self::mir_fixed_lower_kernel),
    ]
}
