use crate::additive::*;
use crate::computation::*;
use crate::error::{Error, Result};
use crate::for_all_values;
use crate::host::*;
use crate::mirrored::*;
use crate::replicated::*;
use crate::types::*;

mod io;
mod misc;
pub use misc::*;

pub trait DispatchKernel<S: Session> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(
        &self,
        plc: &Placement,
    ) -> Result<Box<dyn Fn(&S, Vec<S::Value>) -> Result<S::Value> + Send>>;
}

// TODO if rustc can't figure out how to optimize Box<dyn Fn...> for
// function kernels then we could consider returning an enum over
// fn.. and Box<dyn Fn...> in the traits below instead

pub(crate) trait NullaryKernel<S: Session, P, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P) -> Result<Y> + Send>>;
}

pub(crate) trait UnaryKernel<S: Session, P, X0, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, X0) -> Result<Y> + Send>>;
}

pub(crate) trait BinaryKernel<S: Session, P, X0, X1, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, X0, X1) -> Result<Y> + Send>>;
}

pub(crate) trait TernaryKernel<S: Session, P, X0, X1, X2, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, X0, X1, X2) -> Result<Y> + Send>>;
}

pub(crate) trait VariadicKernel<S: Session, P, XS, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, Vec<XS>) -> Result<Y> + Send>>;
}

pub(crate) trait NullaryKernelCheck<S: Session, P, Y>
where
    Self: NullaryKernel<S, P, Y>,
{
}

pub(crate) trait UnaryKernelCheck<S: Session, P, X0, Y>
where
    Self: UnaryKernel<S, P, X0, Y>,
{
}

pub(crate) trait BinaryKernelCheck<S: Session, P, X0, X1, Y>
where
    Self: BinaryKernel<S, P, X0, X1, Y>,
{
}

pub(crate) trait TernaryKernelCheck<S: Session, P, X0, X1, X2, Y>
where
    Self: TernaryKernel<S, P, X0, X1, X2, Y>,
{
}

pub(crate) trait VariadicKernelCheck<S: Session, P, XS, Y>
where
    Self: VariadicKernel<S, P, XS, Y>,
{
}

pub trait TensorLike<S: Session> {
    type Scalar;
}

pub trait PlacementShape<S: Session, T, ShapeT> {
    fn shape(&self, sess: &S, x: &T) -> ShapeT;
}

pub trait PlacementReshape<S: Session, T, ShapeT, O> {
    fn reshape(&self, sess: &S, x: &T, shape: &ShapeT) -> O;
}

pub trait PlacementDecrypt<S: Session, KeyT, C, O> {
    fn decrypt(&self, sess: &S, key: &KeyT, ciphertext: &C) -> O;
}

modelled_kernel! {
    PlacementDecrypt::decrypt, AesDecryptOp,
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

pub trait PlacementKeyGen<S: Session, KeyT> {
    fn gen_key(&self, sess: &S) -> KeyT;
}

modelled_kernel! {
    PlacementKeyGen::gen_key, PrimPrfKeyGenOp,
    [
        (HostPlacement, () -> PrfKey => [runtime] Self::kernel),
    ]
}

pub trait PlacementSetupGen<S: Session, SetupT> {
    fn gen_setup(&self, sess: &S) -> SetupT;
}

modelled_kernel! {
    PlacementSetupGen::gen_setup, RepSetupOp,
    [
        (ReplicatedPlacement, () -> ReplicatedSetup => [concrete] Self::kernel),
    ]
}

pub trait PlacementDeriveSeed<S: Session, KeyT, SeedT> {
    fn derive_seed(&self, sess: &S, sync_key: SyncKey, key: &KeyT) -> SeedT;
}

modelled_kernel! {
    PlacementDeriveSeed::derive_seed, PrimDeriveSeedOp{sync_key: SyncKey},
    [
        (HostPlacement, (PrfKey) -> Seed => [runtime] Self::kernel),
    ]
}

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

/// Secret share value
pub trait PlacementShare<S: Session, T, O> {
    fn share(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementShare::share, RepShareOp,
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
    ]
}

/// Reveal secret shared value
pub trait PlacementReveal<S: Session, T, O> {
    fn reveal(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementReveal::reveal, RepRevealOp,
    [
        (HostPlacement, (ReplicatedFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (ReplicatedFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (ReplicatedRing64Tensor) -> HostRing64Tensor => [hybrid] Self::ring_kernel),
        (HostPlacement, (ReplicatedRing128Tensor) -> HostRing128Tensor => [hybrid] Self::ring_kernel),
        (HostPlacement, (ReplicatedBitTensor) -> HostBitTensor => [hybrid] Self::ring_kernel),
        (HostPlacement, (ReplicatedBitArray64) -> HostBitArray64 => [concrete] Self::bit_array_kernel),
        (HostPlacement, (ReplicatedBitArray128) -> HostBitArray128 => [concrete] Self::bit_array_kernel),
        (HostPlacement, (ReplicatedBitArray224) -> HostBitArray224 => [concrete] Self::bit_array_kernel),
        (HostPlacement, (ReplicatedAesKey) -> HostAesKey => [concrete] Self::aeskey_kernel),
        (Mirrored3Placement, (ReplicatedBitTensor) -> Mirrored3BitTensor => [concrete] Self::mir_ring_kernel),
        (Mirrored3Placement, (ReplicatedRing64Tensor) -> Mirrored3Ring64Tensor => [concrete] Self::mir_ring_kernel),
        (Mirrored3Placement, (ReplicatedRing128Tensor) -> Mirrored3Ring128Tensor => [concrete] Self::mir_ring_kernel),
        (Mirrored3Placement, (ReplicatedFixed64Tensor) -> Mirrored3Fixed64Tensor => [concrete] Self::mir_fixed_kernel),
        (Mirrored3Placement, (ReplicatedFixed128Tensor) -> Mirrored3Fixed128Tensor => [concrete] Self::mir_fixed_kernel),
    ]
}

modelled_kernel! {
    PlacementReveal::reveal, AdtRevealOp,
    [
        (HostPlacement, (AdditiveRing64Tensor) -> HostRing64Tensor => [hybrid] Self::kernel),
        (HostPlacement, (AdditiveRing128Tensor) -> HostRing128Tensor => [hybrid] Self::kernel),
        (HostPlacement, (AdditiveBitTensor) -> HostBitTensor => [hybrid] Self::kernel),
    ]
}

pub trait PlacementFill<S: Session, ShapeT, O> {
    fn fill(&self, sess: &S, value: Constant, shape: &ShapeT) -> O;
}

modelled!(PlacementFill::fill, ReplicatedPlacement, attributes[value: Constant] (ReplicatedShape) -> ReplicatedRing64Tensor, FillOp);
modelled!(PlacementFill::fill, ReplicatedPlacement, attributes[value: Constant] (ReplicatedShape) -> ReplicatedRing128Tensor, FillOp);
modelled!(PlacementFill::fill, ReplicatedPlacement, attributes[value: Constant] (ReplicatedShape) -> ReplicatedBitTensor, FillOp);
modelled!(PlacementFill::fill, Mirrored3Placement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3Ring64Tensor, FillOp);
modelled!(PlacementFill::fill, Mirrored3Placement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3Ring128Tensor, FillOp);
modelled!(PlacementFill::fill, Mirrored3Placement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3BitTensor, FillOp);
modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> HostBitTensor, FillOp);
modelled!(PlacementFill::fill, Mirrored3Placement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3Fixed64Tensor, FillOp);
modelled!(PlacementFill::fill, Mirrored3Placement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3Fixed128Tensor, FillOp);

kernel! {
    FillOp,
    [
        (HostPlacement, (HostShape) -> HostBitTensor => [runtime] custom |op| {
            use std::convert::TryInto;
            let value: u8 = match op.value {
                Constant::Bit(v) => v,
                Constant::Ring64(v) => v.try_into().map_err(|_| {
                    Error::KernelError("Cannot fill HostBitTensor with non-binary value.".to_string())
                })?,
                Constant::Ring128(v) => v.try_into().map_err(|_| {
                    Error::KernelError("Cannot fill HostBitTensor with non-binary value.".to_string())
                })?,
                _ => {
                    return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a HostBitTensor", op.value.ty())))
                }
            };
            if !(value == 0 || value == 1) {
                return Err(Error::KernelError(
                    "Cannot fill HostBitTensor with non-binary value.".to_string(),
                ));
            }
            assert!(value == 0 || value == 1);
            Ok(Box::new(move |sess, host, host_shape| {
                Self::bit_kernel(sess, host, value, host_shape)
            }))
        }),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedRing64Tensor => [concrete] custom |op| {
                let value: u64 = match op.value {
                    Constant::Bit(v) => v as u64,
                    Constant::Ring64(v) => v,
                    Constant::Float64(v) => v as u64,
                    Constant::Fixed(FixedpointConstant {
                        value, precision
                    }) => {
                        (value * ((1u64 << precision) as f64)) as u64
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a ReplicatedRing64Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::ring64_kernel(sess, rep, value, rep_shape)
                }))
            }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Ring64Tensor => [concrete] custom |op| {
                let value: u64 = match op.value {
                    Constant::Bit(v) => v as u64,
                    Constant::Ring64(v) => v,
                    Constant::Float64(v) => v as u64,
                    Constant::Fixed(FixedpointConstant {
                        value, precision
                    }) => {
                        (value * ((1u64 << precision) as f64)) as u64
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Ring64Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_ring64_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedRing128Tensor => [concrete] custom |op| {
                let value: u128 = match op.value {
                    Constant::Bit(v) => v as u128,
                    Constant::Ring64(v) => v as u128,
                    Constant::Ring128(v) => v,
                    Constant::Float64(v) => v as u128,
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                            (value * ((1u128 << precision) as f64)) as u128
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a ReplicatedRing128Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::ring128_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Ring128Tensor => [concrete] custom |op| {
                let value: u128 = match op.value {
                    Constant::Bit(v) => v as u128,
                    Constant::Ring64(v) => v as u128,
                    Constant::Ring128(v) => v,
                    Constant::Float64(v) => v as u128,
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                            (value * ((1u128 << precision) as f64)) as u128
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Ring128Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_ring128_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedBitTensor => [concrete] custom |op| {
                let value: u8 = match op.value {
                    Constant::Bit(v) => v,
                    Constant::Ring64(v) => v as u8,
                    Constant::Ring128(v) => v as u8,
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a ReplicatedBitTensor", op.value.ty()))),
                };
                if value != 0 && value != 1 {
                    return Err(Error::InvalidArgument(format!("Could only support 0 and 1 for the bit tensor fill, got {}", value)));
                }
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::rep_bit_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3BitTensor => [concrete] custom |op| {
                let value: u8 = match op.value {
                    Constant::Bit(v) => v,
                    Constant::Ring64(v) => v as u8,
                    Constant::Ring128(v) => v as u8,
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3BitTensor", op.value.ty()))),
                };
                if value != 0 && value != 1 {
                    return Err(Error::InvalidArgument(format!("Could only support 0 and 1 for the bit tensor fill, got {}", value)));
                }
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_bit_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Fixed64Tensor => [hybrid] custom |op| {
                let (ring_value, fractional_precision, integral_precision) = match op.value {
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                        let ring_value: u64 = (value * ((1u64 << precision) as f64)) as u64;
                        let fractional_precision = precision as u32;
                        let integral_precision = value.log2().ceil() as u32;
                        (ring_value, fractional_precision, integral_precision)
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Fixed64Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_fixed_kernel(sess, rep, Constant::Ring64(ring_value), rep_shape, fractional_precision, integral_precision)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Fixed128Tensor => [hybrid] custom |op| {
                let (ring_value, fractional_precision, integral_precision) = match op.value {
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                        let ring_value: u128 = (value * ((1u128 << precision) as f64)) as u128;
                        let fractional_precision = precision as u32;
                        let integral_precision = value.log2().ceil() as u32;
                        (ring_value, fractional_precision, integral_precision)
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Fixed128Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_fixed_kernel(sess, rep, Constant::Ring128(ring_value), rep_shape, fractional_precision, integral_precision)
                }))
        }),
    ]
}

modelled_kernel! {
    PlacementFill::fill, AdtFillOp{value: Constant},
    [
        (AdditivePlacement, (HostShape) -> AdditiveRing64Tensor => [hybrid] Self::host_kernel),
        (AdditivePlacement, (HostShape) -> AdditiveRing128Tensor => [hybrid] Self::host_kernel),
        (AdditivePlacement, (AdditiveShape) -> AdditiveRing64Tensor => [concrete] Self::adt_kernel),
        (AdditivePlacement, (AdditiveShape) -> AdditiveRing128Tensor => [concrete] Self::adt_kernel),
    ]
}

modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> HostRing64Tensor, RingFillOp);
modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> HostRing128Tensor, RingFillOp);

kernel! {
    RingFillOp,
    [
        (HostPlacement, (HostShape) -> HostRing64Tensor => [runtime] attributes[value: Ring64] Self::ring64_kernel),
        (HostPlacement, (HostShape) -> HostRing128Tensor => [runtime] attributes[value: Ring128] Self::ring128_kernel),
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

/// Equality
pub trait PlacementEqual<S: Session, T, U, O> {
    fn equal(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementEqual::equal, RepEqualOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
    ]
}

/// Mux
pub trait PlacementMux<S: Session, T, U, V, O> {
    fn mux(&self, sess: &S, s: &T, x: &U, y: &V) -> O;
}

modelled!(PlacementMux::mux, ReplicatedPlacement, (BooleanTensor, Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (BooleanTensor, Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (Tensor, Tensor, Tensor) -> Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, MuxOp);

kernel! {
    MuxOp,
    [
        (ReplicatedPlacement, (Tensor, Tensor, Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (BooleanTensor, Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (BooleanTensor, Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor  => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor  => [transparent] Self::rep_bit_selector_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_bit_selector_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::rep_bit_selector_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_bit_selector_fixed_kernel),
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

pub trait PlacementLessThan<S: Session, T, U, O> {
    fn less(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementLessThan::less, LessOp,
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

pub trait PlacementGreaterThan<S: Session, T, U, O> {
    fn greater_than(&self, sess: &S, x: &T, y: &U) -> O;
}

modelled_kernel! {
    PlacementGreaterThan::greater_than, GreaterThanOp,
    [
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_mir_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_mir_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, Mirrored3Ring128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, Mirrored3Ring64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
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

pub trait PlacementZeros<S: Session, ShapeT, O> {
    fn zeros(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementZeros<S, ShapeT, O> for P
where
    P: PlacementFill<S, ShapeT, O>,
    O: TensorLike<S>,
    O::Scalar: Into<Constant>,
    O::Scalar: From<u8>,
{
    fn zeros(&self, sess: &S, shape: &ShapeT) -> O {
        let value = O::Scalar::from(0).into();
        self.fill(sess, value, shape)
    }
}

pub trait PlacementOnes<S: Session, ShapeT, O> {
    fn ones(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementOnes<S, ShapeT, O> for P
where
    P: PlacementFill<S, ShapeT, O>,
    O: TensorLike<S>,
    O::Scalar: Into<Constant>,
    O::Scalar: From<u8>,
{
    fn ones(&self, sess: &S, shape: &ShapeT) -> O {
        let value = O::Scalar::from(1).into();
        self.fill(sess, value, shape)
    }
}

modelled_kernel! {
    PlacementOnes::ones, FloatingpointOnesOp,
    [
        (HostPlacement, (HostShape) -> Float64Tensor => [hybrid] Self::float_host_kernel),
    ]
}

modelled_kernel! {
    PlacementOnes::ones, HostOnesOp,
    [
        (HostPlacement, (HostShape) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementOnes::ones, OnesOp,
    [
        (HostPlacement, (HostShape) -> Tensor => [hybrid] Self::host_kernel),
        // We do not support the ReplicatedPlacement: PlacementFill yet, hence we do not support Ones.
        // Also, logical Tensor can only hold Host tensors at the moment.
        // (ReplicatedPlacement, (HostShape) -> Tensor => [hybrid] Self::rep_kernel),
    ]
}

pub trait PlacementSample<S: Session, ShapeT, O> {
    fn sample(&self, sess: &S, max_value: Option<u64>, shape: &ShapeT) -> O;
}

modelled!(PlacementSample::sample, HostPlacement, attributes[max_value: Option<u64>] (HostShape) -> HostRing64Tensor, RingSampleOp);
modelled!(PlacementSample::sample, HostPlacement, attributes[max_value: Option<u64>] (HostShape) -> HostRing128Tensor, RingSampleOp);

kernel! {
    RingSampleOp,
    [
        (HostPlacement, (HostShape) -> HostRing64Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_uniform_u64(ctx, plc, shape)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_bits_u64(ctx, plc, shape)
                })),
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleOp with max_value != 1".to_string()
                )),
            }
        }),
        (HostPlacement, (HostShape) -> HostRing128Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_uniform_u128(ctx, plc, shape)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_bits_u128(ctx, plc, shape)
                })),
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleOp with max_value != 1".to_string()
                )),
            }
        }),
    ]
}

pub trait PlacementSampleUniform<S: Session, ShapeT, O> {
    fn sample_uniform(&self, sess: &S, shape: &ShapeT) -> O;
}

modelled_kernel! {
    PlacementSampleUniform::sample_uniform, BitSampleOp,
    [
        (HostPlacement, (HostShape) -> HostBitTensor => [runtime] Self::kernel),
    ]
}

impl<S: Session, ShapeT, O, P> PlacementSampleUniform<S, ShapeT, O> for P
where
    P: PlacementSample<S, ShapeT, O>,
{
    fn sample_uniform(&self, sess: &S, shape: &ShapeT) -> O {
        self.sample(sess, None, shape)
    }
}

pub trait PlacementSampleBits<S: Session, ShapeT, O> {
    fn sample_bits(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementSampleBits<S, ShapeT, O> for P
where
    P: PlacementSample<S, ShapeT, O>,
{
    fn sample_bits(&self, sess: &S, shape: &ShapeT) -> O {
        self.sample(sess, Some(1), shape)
    }
}

pub trait PlacementSampleSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_seeded(&self, sess: &S, max_value: Option<u64>, shape: &ShapeT, seed: &SeedT) -> O;
}

modelled!(PlacementSampleSeeded::sample_seeded, HostPlacement, attributes[max_value: Option<u64>] (HostShape, Seed) -> HostRing64Tensor, RingSampleSeededOp);
modelled!(PlacementSampleSeeded::sample_seeded, HostPlacement, attributes[max_value: Option<u64>] (HostShape, Seed) -> HostRing128Tensor, RingSampleSeededOp);

kernel! {
    RingSampleSeededOp,
    [
        (HostPlacement, (HostShape, Seed) -> HostRing64Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_uniform_u64(ctx, plc, shape, seed)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_bits_u64(ctx, plc, shape, seed)
                })),
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleSeededOp with max_value != 1".to_string()
                )),
            }
        }),
        (HostPlacement, (HostShape, Seed) -> HostRing128Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_uniform_u128(ctx, plc, shape, seed)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_bits_u128(ctx, plc, shape, seed)
                })),
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleSeededOp with max_value != 1".to_string()
                )),
            }
        }),
    ]
}

pub trait PlacementSampleUniformSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_uniform_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

modelled_kernel! {
    PlacementSampleUniformSeeded::sample_uniform_seeded, BitSampleSeededOp,
    [
        (HostPlacement, (HostShape, Seed) -> HostBitTensor => [runtime] Self::kernel),
    ]
}

impl<S: Session, ShapeT, SeedT, O, P> PlacementSampleUniformSeeded<S, ShapeT, SeedT, O> for P
where
    P: PlacementSampleSeeded<S, ShapeT, SeedT, O>,
{
    fn sample_uniform_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O {
        self.sample_seeded(sess, None, shape, seed)
    }
}

pub trait PlacementSampleBitsSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_bits_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

impl<S: Session, ShapeT, SeedT, O, P> PlacementSampleBitsSeeded<S, ShapeT, SeedT, O> for P
where
    P: PlacementSampleSeeded<S, ShapeT, SeedT, O>,
{
    fn sample_bits_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O {
        self.sample_seeded(sess, Some(1), shape, seed)
    }
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

pub trait PlacementTruncPr<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: u32, x: &T) -> O;
}

modelled_kernel! {
    PlacementTruncPr::trunc_pr, FixedpointTruncPrOp{precision: u32},
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

// TODO(Morten) should we rename this as a shift?
modelled_kernel! {
    PlacementTruncPr::trunc_pr, RepTruncPrOp{amount: u32},
    [
        (ReplicatedPlacement,  (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::kernel),
        (ReplicatedPlacement,  (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::kernel),
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

pub trait PlacementPlace<S: Session, T> {
    fn place(&self, sess: &S, x: T) -> T;
}

pub trait PlacementConstant<S: Session, O> {
    fn constant(&self, sess: &S, value: Constant) -> O;
}

macro_rules! constant_kernels {
    ($($val:ident),+) => {
        $(
            modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> $val, ConstantOp);
        )+
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> HostString, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> HostShape, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> PrfKey, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Seed, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Float32Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Float64Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, Mirrored3Placement, attributes[value: Constant] () -> Float32Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, Mirrored3Placement, attributes[value: Constant] () -> Float64Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, Mirrored3Placement, attributes[value: Constant] () -> Tensor, ConstantOp);


        kernel! {
            ConstantOp, [
                $(
                    (HostPlacement, () -> $val => [runtime] attributes[value: $val] Self::kernel),
                )+
                (HostPlacement, () -> HostString => [runtime] attributes[value: String] Self::string_kernel),
                (HostPlacement, () -> HostShape => [runtime] attributes[value: RawShape] Self::shape_kernel),
                (HostPlacement, () -> PrfKey => [runtime] attributes[value: RawPrfKey] Self::prf_key_kernel),
                (HostPlacement, () -> Seed => [runtime] attributes[value: RawSeed] Self::seed_kernel),
                (HostPlacement, () -> Tensor => [concrete] attributes[sig, value] Self::logical_kernel),
                (HostPlacement, () -> Float32Tensor => [concrete] attributes[value] Self::float_kernel),
                (HostPlacement, () -> Float64Tensor => [concrete] attributes[value] Self::float_kernel),
                (Mirrored3Placement, () -> Tensor => [concrete] attributes[sig, value] Self::mir3_logical_kernel),
                (Mirrored3Placement, () -> Float32Tensor => [concrete] attributes[value] Self::mir3_float_kernel),
                (Mirrored3Placement, () -> Float64Tensor => [concrete] attributes[value] Self::mir3_float_kernel),

            ]
        }
    };
}

constant_kernels![
    HostRing64Tensor,
    HostRing128Tensor,
    HostFloat32Tensor,
    HostFloat64Tensor,
    HostInt8Tensor,
    HostInt16Tensor,
    HostInt32Tensor,
    HostInt64Tensor,
    HostUint8Tensor,
    HostUint16Tensor,
    HostUint32Tensor,
    HostUint64Tensor
];

pub trait PlacementIdentity<S: Session, T, O> {
    fn identity(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementIdentity::identity, IdentityOp,
    [
        (HostPlacement, (BooleanTensor) -> BooleanTensor => [concrete] Self::boolean_host_kernel),
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::host_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::host_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_inner_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_inner_kernel),
        (ReplicatedPlacement, (BooleanTensor) -> BooleanTensor => [concrete] Self::boolean_rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_inner_kernel),
        // TODO higher-level kernels for these
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint8Tensor) -> HostUint8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint16Tensor) -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString) -> HostString => [runtime] Self::kernel),
        (HostPlacement, (Unit) -> Unit => [runtime] Self::missing_kernel),
        (HostPlacement, (HostShape) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (Seed) -> Seed => [runtime] Self::kernel),
        (HostPlacement, (PrfKey) -> PrfKey => [runtime] Self::kernel),
    ]
}

pub trait PlacementInput<S: Session, O> {
    fn input(&self, sess: &S, arg_name: String) -> O;
}

pub trait PlacementOutput<S: Session, T, O> {
    fn output(&self, sess: &S, x: &T) -> O;
}

modelled!(PlacementOutput::output, HostPlacement, (BooleanTensor) -> BooleanTensor, OutputOp);

pub trait PlacementLoad<S: Session, KeyT, QueryT, O> {
    fn load(&self, sess: &S, key: &KeyT, query: &QueryT) -> O;
}

pub trait PlacementSave<S: Session, KeyT, T, O> {
    fn save(&self, sess: &S, key: &KeyT, x: &T) -> O;
}

pub trait PlacementSend<S: Session, T, O> {
    fn send(&self, sess: &S, rendezvous_key: RendezvousKey, receiver: Role, x: &T) -> O;
}

pub trait PlacementReceive<S: Session, O> {
    fn receive(&self, sess: &S, rendezvous_key: RendezvousKey, sender: Role) -> O;
}

pub trait PlacementAtLeast2D<S: Session, T, O> {
    fn at_least_2d(&self, sess: &S, to_column_vector: bool, x: &T) -> O;
}

modelled_kernel! {
    PlacementAtLeast2D::at_least_2d, HostAtLeast2DOp{to_column_vector: bool},
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementAtLeast2D::at_least_2d, FloatingpointAtLeast2DOp{to_column_vector: bool},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

modelled_kernel! {
    PlacementAtLeast2D::at_least_2d, AtLeast2DOp{to_column_vector: bool},
    [
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::host_kernel),
        // (ReplicatedPlacement, (Tensor) -> Tensor => [hybrid] attributes[to_column_vector] Self::rep_kernel),
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

pub trait PlacementExpandDims<S: Session, T, O> {
    fn expand_dims(&self, sess: &S, axis: Vec<u32>, x: &T) -> O;
}

modelled_kernel! {
    PlacementExpandDims::expand_dims, ExpandDimsOp{axis: Vec<u32>},
    [
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::host_kernel),
    ]
}

modelled_kernel! {
    PlacementExpandDims::expand_dims, FloatingpointExpandDimsOp{axis: Vec<u32>},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

modelled_kernel! {
    PlacementExpandDims::expand_dims, HostExpandDimsOp{axis: Vec<u32>}, [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

pub trait PlacementSqueeze<S: Session, T, O> {
    fn squeeze(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

modelled_kernel! {
    PlacementSqueeze::squeeze, HostSqueezeOp{axis: Option<u32>}, [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> HostUint64Tensor => [runtime] Self::kernel),
    ]
}

pub trait PlacementConcatenate<S: Session, TS, O> {
    fn concatenate(&self, sess: &S, axis: u32, xs: &[TS]) -> O;
}

modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[Float32Tensor] -> Float32Tensor, FloatingpointConcatOp);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[Float64Tensor] -> Float64Tensor, FloatingpointConcatOp);

kernel! {
    FloatingpointConcatOp,
    [
        (HostPlacement, vec[Float32Tensor] -> Float32Tensor => [concrete] attributes[axis] Self::float_host_kernel),
        (HostPlacement, vec[Float64Tensor] -> Float64Tensor => [concrete] attributes[axis] Self::float_host_kernel),
    ]
}

pub trait PlacementTranspose<S: Session, T, O> {
    fn transpose(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementTranspose::transpose, FloatingpointTransposeOp,
    [
        // (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementTranspose::transpose, TransposeOp,
    [
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementTranspose::transpose, HostTransposeOp,
    [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
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

pub trait PlacementCast<S: Session, T, O> {
    fn cast(&self, sess: &S, x: &T) -> O;
}

modelled!(PlacementCast::cast, HostPlacement, (Tensor) -> Tensor, CastOp);
modelled!(PlacementCast::cast, Mirrored3Placement, (Tensor) -> Tensor, CastOp);

kernel! {
    CastOp,
    [
        (HostPlacement, (Tensor) -> Tensor => [concrete] attributes[sig] Self::kernel),
        (Mirrored3Placement, (Tensor) -> Tensor => [concrete] attributes[sig] Self::mir_kernel),
    ]
}

pub trait PlacementSlice<S: Session, T, O> {
    fn slice(&self, sess: &S, slice_info: SliceInfo, x: &T) -> O;
}

modelled_kernel! {
    PlacementSlice::slice, HostSliceOp{slice: SliceInfo},
    [
        (HostPlacement, (HostShape) -> HostShape => [runtime] Self::shape_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementSlice::slice, RepSliceOp{slice: SliceInfo},
    [
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedShape => [concrete] Self::shape_kernel),
    ]
}

unmodelled!(HostPlacement, attributes[slice: SliceInfo] (HostShape) -> HostShape, SliceOp);

kernel! {
    SliceOp,
    [
        (HostPlacement, (HostShape) -> HostShape => [hybrid] attributes[slice] Self::kernel),
        // (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [hybrid] attributes[slice] Self::kernel),
        // (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [hybrid] attributes[slice] Self::kernel),
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

pub trait PlacementIndexAxis<S: Session, T, O> {
    fn index_axis(&self, sess: &S, axis: usize, index: usize, x: &T) -> O;
}

modelled_kernel! {
    PlacementIndexAxis::index_axis, IndexAxisOp{axis: usize, index: usize},
    [
        (HostPlacement, (BooleanTensor) -> BooleanTensor => [concrete] Self::bool_host_kernel),
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
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

pub trait PlacementShlDim<S: Session, T, O> {
    fn shl_dim(&self, sess: &S, amount: usize, ring_size: usize, x: &T) -> O;
}

modelled_kernel! {
    PlacementShlDim::shl_dim, RepShlDimOp{amount: usize, bit_length: usize},
    [
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementShlDim::shl_dim, HostShlDimOp{amount: usize, bit_length: usize},
    [
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::bit_kernel),
    ]
}
