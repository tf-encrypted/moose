use super::*;

/// Exclusive-or
pub trait PlacementXor<S: Session, T, U, O> {
    fn xor(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementXor::xor, XorOp,
    [
        (HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => [runtime] Self::host_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (Mirrored3BitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, Mirrored3BitTensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
    ]
}

// add = xor in Z2
impl<S: Session> PlacementAdd<S, m!(HostBitTensor), m!(HostBitTensor), m!(HostBitTensor)>
    for HostPlacement
where
    HostBitTensor: KnownType<S>,
    HostPlacement: PlacementXor<S, m!(HostBitTensor), m!(HostBitTensor), m!(HostBitTensor)>,
{
    fn add(&self, sess: &S, x0: &m!(HostBitTensor), x1: &m!(HostBitTensor)) -> m!(HostBitTensor) {
        self.xor(sess, x0, x1)
    }
}

// sub = xor in Z2
impl<S: Session> PlacementSub<S, m!(HostBitTensor), m!(HostBitTensor), m!(HostBitTensor)>
    for HostPlacement
where
    HostBitTensor: KnownType<S>,
    HostPlacement: PlacementXor<S, m!(HostBitTensor), m!(HostBitTensor), m!(HostBitTensor)>,
{
    fn sub(&self, sess: &S, x0: &m!(HostBitTensor), x1: &m!(HostBitTensor)) -> m!(HostBitTensor) {
        self.xor(sess, x0, x1)
    }
}

/// Logical-and
pub trait PlacementAnd<S: Session, T, U, O> {
    fn and(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementAnd::and, AndOp,
    [
        (HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_ring_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
    ]
}

// mul = and in Z2
impl<S: Session> PlacementMul<S, m!(HostBitTensor), m!(HostBitTensor), m!(HostBitTensor)>
    for HostPlacement
where
    HostBitTensor: KnownType<S>,
    HostPlacement: PlacementAnd<S, m!(HostBitTensor), m!(HostBitTensor), m!(HostBitTensor)>,
{
    fn mul(&self, sess: &S, x0: &m!(HostBitTensor), x1: &m!(HostBitTensor)) -> m!(HostBitTensor) {
        self.and(sess, x0, x1)
    }
}

/// Logical-or
pub trait PlacementOr<S: Session, T, U, O> {
    fn or(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementOr::or, OrOp,
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
    PlacementMsb::msb, MsbOp,
    [
        (ReplicatedPlacement,  (ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_bit_kernel),
        (ReplicatedPlacement,  (ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_bit_kernel),
        (ReplicatedPlacement,  (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_ring_kernel),
        (ReplicatedPlacement,  (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [transparent] Self::rep_ring_kernel),
        (ReplicatedPlacement,  (ReplicatedBitArray64) -> ReplicatedRing64Tensor => [transparent] Self::rep_bit_dec_kernel),
        (ReplicatedPlacement,  (ReplicatedBitArray128) -> ReplicatedRing128Tensor => [transparent] Self::rep_bit_dec_kernel),

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

pub trait PlacementBitDecompose<S: Session, T, O> {
    fn bit_decompose(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementBitDecompose::bit_decompose, BitDecomposeOp,
    [
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_ring64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_ring128_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostBitTensor => [runtime] Self::host_bit64_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostBitTensor => [runtime] Self::host_bit128_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedBitArray64 => [hybrid] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedBitArray128 => [hybrid] Self::rep_ring_kernel),
    ]
}

pub trait PlacementBitCompose<S: Session, T, O> {
    fn bit_compose(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementBitCompose::bit_compose, BitComposeOp,
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
