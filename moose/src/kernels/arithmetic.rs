use super::*;

/// Addition
pub trait PlacementAdd<S: Session, T, U, O> {
    fn add(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementAdd::add, FixedpointAddOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_mirfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_mirfixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::mirfixed_repfixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::mirfixed_repfixed_kernel),
    ]
}

modelled_kernel! {
    PlacementAdd::add, RepAddOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::mir_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, Mirrored3Ring64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, Mirrored3Ring128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_mir_kernel),
        (ReplicatedPlacement, (Mirrored3BitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, Mirrored3BitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_mir_kernel),
    ]
}

modelled_kernel! {
    PlacementAdd::add, HostAddOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor, HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor, HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor, HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementAdd::add, AddOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::rep_kernel),
    ]
}

modelled_kernel! {
    PlacementAdd::add, AdtAddOp,
    [
        (AdditivePlacement, (AdditiveRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [concrete] Self::adt_adt_kernel),
        // TODO(Morten) replace host tensors with mirrored tensors in the below
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (AdditiveBitTensor, HostBitTensor) -> AdditiveBitTensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::host_adt_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::host_adt_kernel),
        (AdditivePlacement, (HostBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [hybrid] Self::host_adt_kernel),
    ]
}

modelled_kernel! {
    PlacementAdd::add, FloatingpointAddOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

modelled_kernel! {
    PlacementAdd::add, RingAddOp,
    [
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

/// Subtraction
pub trait PlacementSub<S: Session, T, U, O> {
    fn sub(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementSub::sub, FixedpointSubOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
    ]
}

modelled_kernel! {
    PlacementSub::sub, RepSubOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::mir_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, Mirrored3Ring64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, Mirrored3Ring128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_mir_kernel),
    ]
}

modelled_kernel! {
    PlacementSub::sub, SubOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::rep_kernel),
    ]
}

modelled_kernel! {
    PlacementSub::sub, HostSubOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor, HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor, HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor, HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementSub::sub, AdtSubOp,
    [
        (AdditivePlacement, (AdditiveRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [concrete] Self::adt_adt_kernel),
        // TODO(Morten) replace host tensors with mirrored tensors in the below
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::host_adt_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::host_adt_kernel),
    ]
}

modelled_kernel! {
    PlacementSub::sub, FloatingpointSubOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

modelled_kernel! {
    PlacementSub::sub, RingSubOp,
    [
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

pub trait PlacementNeg<S: Session, T, O> {
    fn neg(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementNeg::neg, BitNegOp,
    [
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementNeg::neg, RingNegOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementNeg::neg, NegOp,
    [
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
    ]
}

modelled_kernel! {
    PlacementNeg::neg, RepNegOp,
    [
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_bit_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_rep_kernel),
    ]
}

/// Multiplication
pub trait PlacementMul<S: Session, T, U, O> {
    fn mul(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementMul::mul, FixedpointMulOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_mirfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_mirfixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::mirfixed_repfixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::mirfixed_repfixed_kernel),
    ]
}

modelled_kernel! {
    PlacementMul::mul, RepMulOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, Mirrored3Ring128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_mir_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, Mirrored3Ring64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_mir_kernel),
    ]
}

modelled_kernel! {
    PlacementMul::mul, MulOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] attributes[sig] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] attributes[sig] Self::rep_kernel),
    ]
}

modelled_kernel! {
    PlacementMul::mul, HostMulOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor, HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor, HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor, HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementMul::mul, AdtMulOp,
    [
        // TODO(Morten) replace host tensors with mirrored tensors in the below
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::host_adt_kernel),
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::host_adt_kernel),
        (AdditivePlacement, (AdditiveBitTensor, HostBitTensor) -> AdditiveBitTensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (HostBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [hybrid] Self::host_adt_kernel),

    ]
}

modelled_kernel! {
    PlacementMul::mul, FloatingpointMulOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

modelled_kernel! {
    PlacementMul::mul, RingMulOp,
    [
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

/// Division
pub trait PlacementDiv<S: Session, T, U, O> {
    fn div(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementDiv::div, FixedpointDivOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::rep_rep_kernel),
    ]
}

modelled_kernel! {
    PlacementDiv::div, FloatingpointDivOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

modelled_kernel! {
    PlacementDiv::div, DivOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::rep_kernel),
    ]
}

modelled_kernel! {
    PlacementDiv::div, HostDivOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring_kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring_kernel),
        (HostPlacement, (HostInt8Tensor, HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor, HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor, HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

/// Dot product
pub trait PlacementDot<S: Session, T, U, O> {
    fn dot(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementDot::dot, FixedpointDotOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_on_host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_on_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_on_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_on_rep_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
    ]
}

modelled_kernel! {
    PlacementDot::dot, RepDotOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_rep_kernel),
    ]
}

modelled_kernel! {
    PlacementDot::dot, FloatingpointDotOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

modelled_kernel! {
    PlacementDot::dot, HostDotOp,
    [
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementDot::dot, RingDotOp,
    [
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementDot::dot, DotOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] attributes[sig] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] attributes[sig] Self::rep_kernel),
    ]
}

/// Shift left
pub trait PlacementShl<S: Session, T, O> {
    fn shl(&self, sess: &S, amount: usize, x: &T) -> O;
}

modelled_kernel! {
    PlacementShl::shl, RepShlOp{amount: usize},
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementShl::shl, RingShlOp{amount: usize},
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementShl::shl, AdtShlOp{amount: usize},
    [
        (AdditivePlacement, (AdditiveRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::kernel),
        (AdditivePlacement, (AdditiveRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::kernel),
    ]
}

/// Shift right
pub trait PlacementShr<S: Session, T, O> {
    fn shr(&self, sess: &S, amount: usize, x: &T) -> O;
}

modelled_kernel! {
    PlacementShr::shr, RingShrOp{amount: usize},
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

/// Square root
pub trait PlacementSqrt<S: Session, T, O> {
    fn sqrt(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementSqrt::sqrt, HostSqrtOp,
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
    ]
}

/// Variadic addition
pub trait PlacementAddN<S: Session, T, O> {
    fn add_n(&self, sess: &S, x: &[T]) -> O;
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

/// Sum along axis
pub trait PlacementSum<S: Session, T, O> {
    fn sum(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

modelled_kernel! {
    PlacementSum::sum, FixedpointSumOp{axis: Option<u32>},
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::rep_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
    ]
}

modelled_kernel! {
    PlacementSum::sum, RingSumOp{axis: Option<u32>},
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementSum::sum, HostSumOp{axis: Option<u32>}, [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementSum::sum, FloatingpointSumOp{axis: Option<u32>},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

modelled_kernel! {
    PlacementSum::sum, RepSumOp{axis: Option<u32>},
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementSum::sum, SumOp{axis: Option<u32>},
    [
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::host_kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] Self::rep_kernel),
    ]
}

pub trait PlacementPow2<S: Session, T, O> {
    fn pow2(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementPow2::pow2, Pow2Op,
    [
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::rep_rep_kernel),
    ]
}

pub trait PlacementExp<S: Session, T, O> {
    fn exp(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementExp::exp, ExpOp,
    [
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [transparent] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [transparent] Self::rep_rep_kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] Self::logical_kernel),
    ]
}

pub trait PlacementSigmoid<S: Session, T, O> {
    fn sigmoid(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementSigmoid::sigmoid, SigmoidOp,
    [
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [transparent] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [transparent] Self::rep_rep_kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] Self::logical_kernel),
    ]
}

/// Mean
pub trait PlacementMean<S: Session, T, O> {
    fn mean(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

kernel! {
    MeanOp, [
        (HostPlacement, (Tensor) -> Tensor => [concrete] attributes[sig, axis] Self::host_kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] attributes[sig, axis] Self::rep_kernel),
    ]
}

modelled_kernel! {
    PlacementMean::mean, HostMeanOp{axis: Option<u32>},
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementMean::mean, FloatingpointMeanOp{axis: Option<u32>},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

modelled_kernel! {
    PlacementMean::mean, FixedpointMeanOp{axis: Option<u32>},
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
    ]
}

pub trait PlacementMeanAsFixedpoint<S: Session, T, O> {
    fn mean_as_fixedpoint(
        &self,
        sess: &S,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: &T,
    ) -> O;
}

modelled_kernel! {
    PlacementMeanAsFixedpoint::mean_as_fixedpoint, RepFixedpointMeanOp{axis: Option<u32>, scaling_base: u64, scaling_exp: u32},
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementMeanAsFixedpoint::mean_as_fixedpoint, RingFixedpointMeanOp{axis: Option<u32>, scaling_base: u64, scaling_exp: u32},
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring128_kernel),
    ]
}

pub trait PlacementMaximum<S: Session, TS, O> {
    fn maximum(&self, sess: &S, x: &[TS]) -> O;
}

modelled_kernel! {
    PlacementMaximum::maximum, MaximumOp,
    [
        (ReplicatedPlacement, vec[Tensor] -> Tensor => [concrete] Self::rep_logical_kernel),
        (ReplicatedPlacement, vec[Fixed64Tensor] -> Fixed64Tensor => [concrete] Self::fixed_kernel),
        (ReplicatedPlacement, vec[Fixed128Tensor] -> Fixed128Tensor => [concrete] Self::fixed_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed64Tensor] -> ReplicatedFixed64Tensor => [concrete] Self::rep_fixed_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed128Tensor] -> ReplicatedFixed128Tensor => [concrete] Self::rep_fixed_kernel),
        (ReplicatedPlacement, vec[ReplicatedRing64Tensor] -> ReplicatedRing64Tensor => [transparent] Self::kernel),
        (ReplicatedPlacement, vec[ReplicatedRing128Tensor] -> ReplicatedRing128Tensor => [transparent] Self::kernel),
    ]
}

pub trait PlacementAbs<S: Session, T, O> {
    fn abs(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementAbs::abs, RepAbsOp,
    [
        (ReplicatedPlacement,  (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::kernel),
        (ReplicatedPlacement,  (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [transparent] Self::kernel),
    ]
}

pub trait PlacementSign<S: Session, T, O> {
    fn sign(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementSign::sign, SignOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring128_kernel),
    ]
}

pub trait PlacementInverse<S: Session, T, O> {
    fn inverse(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementInverse::inverse, FloatingpointInverseOp,
    [
        // (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementInverse::inverse, HostInverseOp,
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementInverse::inverse, InverseOp,
    [
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::kernel),
    ]
}

pub trait PlacementDiag<S: Session, T, O> {
    fn diag(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementDiag::diag, HostDiagOp,
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint16Tensor) -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::bit_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring_kernel),
    ]
}

modelled_kernel! {
    PlacementDiag::diag, RepDiagOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::kernel),
    ]
}