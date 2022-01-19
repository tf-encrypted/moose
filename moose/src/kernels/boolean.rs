use super::*;

/// Exclusive-or
pub trait PlacementXor<S: Session, T, U, O> {
    fn xor(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementXor::xor, RepXorOp,
    [
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [transparent] Self::bit_kernel),
        (ReplicatedPlacement, (Mirrored3BitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [transparent] Self::bit_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, Mirrored3BitTensor) -> ReplicatedBitTensor => [transparent] Self::bit_kernel),
    ]
}

modelled_kernel! {
    PlacementXor::xor, BitXorOp,
    [
        (HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => [runtime] Self::kernel),
    ]
}

modelled_alias!(PlacementAdd::add, HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => PlacementXor::xor); // add = xor in Z2
modelled_alias!(PlacementSub::sub, HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => PlacementXor::xor); // sub = xor in Z2

/// Logical-and
pub trait PlacementAnd<S: Session, T, U, O> {
    fn and(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementAnd::and, RepAndOp,
    [
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [transparent] Self::bit_kernel),
    ]
}

modelled_kernel! {
    PlacementAnd::and, BitAndOp,
    [
        (HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => [runtime] Self::bit_kernel),
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring_kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring_kernel),
    ]
}

modelled_alias!(PlacementMul::mul, HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => PlacementAnd::and); // mul = and in Z2

/// Logical-or
pub trait PlacementOr<S: Session, T, U, O> {
    fn or(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementOr::or, BitOrOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
        (HostPlacement, (BooleanTensor, BooleanTensor) -> BooleanTensor => [concrete] Self::bool_kernel),
        (HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => [runtime] Self::host_kernel),
    ]
}

pub trait PlacementMsb<S: Session, T, O> {
    fn msb(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementMsb::msb, RepMsbOp,
    [
        (ReplicatedPlacement,  (ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::bit_kernel),
        (ReplicatedPlacement,  (ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::bit_kernel),
        (ReplicatedPlacement,  (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::ring_kernel),
        (ReplicatedPlacement,  (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [transparent] Self::ring_kernel),
    ]
}

pub trait PlacementBitExtract<S: Session, T, O> {
    fn bit_extract(&self, sess: &S, bit_idx: usize, x: &T) -> O;
}

modelled_kernel! {
    PlacementBitExtract::bit_extract, BitExtractOp{bit_idx: usize},
    [
        (HostPlacement, (HostRing64Tensor) -> HostBitTensor => [runtime] Self::kernel64),
        (HostPlacement, (HostRing128Tensor) -> HostBitTensor => [runtime] Self::kernel128),
    ]
}

pub trait PlacementBitDec<S: Session, T, O> {
    fn bit_decompose(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementBitDec::bit_decompose, HostBitDecOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::ring64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::ring128_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostBitTensor => [runtime] Self::bit64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostBitTensor => [runtime] Self::bit128_kernel),
    ]
}

modelled_kernel! {
    PlacementBitDec::bit_decompose, RepBitDecOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedBitArray64 => [hybrid] Self::ring_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedBitArray128 => [hybrid] Self::ring_kernel),
    ]
}

pub trait PlacementBitCompose<S: Session, T, O> {
    fn bit_compose(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementBitCompose::bit_compose, RepBitComposeOp,
    [
        (ReplicatedPlacement, (ReplicatedBitArray64) -> ReplicatedRing64Tensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitArray128) -> ReplicatedRing128Tensor => [transparent] Self::rep_kernel),
    ]
}

pub trait PlacementRingInject<S: Session, T, O> {
    fn ring_inject(&self, sess: &S, bit_idx: usize, x: &T) -> O;
}

modelled_kernel! {
    PlacementRingInject::ring_inject, RingInjectOp{bit_idx: usize},
    [
        (HostPlacement, (HostBitTensor) -> HostRing64Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostBitTensor) -> HostRing128Tensor => [runtime] Self::host_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_kernel),
    ]
}