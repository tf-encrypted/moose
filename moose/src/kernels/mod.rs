//! Moose recognized kernels

use crate::additive::AdditivePlacement;
use crate::computation::*;
use crate::error::Error;
use crate::error::Result;
use crate::execution::{Operands, Session};
use crate::host::HostPlacement;
use crate::mirrored::Mirrored3Placement;
use crate::replicated::ReplicatedPlacement;
use crate::types::*;

mod arithmetic;
mod boolean;
mod comparison;
mod constants;
mod control_flow;
mod conversion;
mod indexing;
mod io;
mod sampling;
mod shapes;

pub use arithmetic::*;
pub use boolean::*;
pub use comparison::*;
pub use constants::*;
pub use control_flow::*;
pub use conversion::*;
pub use indexing::*;
pub use io::*;
pub use sampling::*;
pub use shapes::*;

pub type Kernel<S> =
    Box<dyn Fn(&S, Operands<<S as Session>::Value>) -> Result<<S as Session>::Value> + Send + Sync>;

pub trait DispatchKernel<S: Session> {
    fn compile(&self, plc: &Placement) -> Result<Kernel<S>>;

    fn execute(&self, plc: &Placement, sess: &S, operands: Operands<S::Value>) -> Result<S::Value> {
        let kernel = Self::compile(self, plc)?;
        kernel(sess, operands)
    }
}

pub type NgNullaryKernel<S, V> = Box<
    dyn Fn(
            &S,
            &Placement, // TODO get rid of this?
        ) -> Result<V>
        + Send
        + Sync,
>;

pub type NgUnaryKernel<S, V> = Box<
    dyn Fn(
            &S,
            &Placement, // TODO get rid of this?
            V,
        ) -> Result<V>
        + Send
        + Sync,
>;

pub type NgBinaryKernel<S, V> = Box<
    dyn Fn(
            &S,
            &Placement, // TODO get rid of this?
            V,
            V,
        ) -> Result<V>
        + Send
        + Sync,
>;

pub type NgTernaryKernel<S, V> = Box<
    dyn Fn(
            &S,
            &Placement, // TODO get rid of this?
            V,
            V,
            V,
        ) -> Result<V>
        + Send
        + Sync,
>;

pub type NgVariadicKernel<S, V> = Box<
    dyn Fn(
            &S,
            &Placement, // TODO get rid of this?
            Vec<V>,
        ) -> Result<V>
        + Send
        + Sync,
>;

pub enum NgKernel<S: Session, V> {
    Nullary { closure: NgNullaryKernel<S, V> },
    Unary { closure: NgUnaryKernel<S, V> },
    Binary { closure: NgBinaryKernel<S, V> },
    Ternary { closure: NgTernaryKernel<S, V> },
    Variadic { closure: NgVariadicKernel<S, V> },
}

pub trait NgDispatchKernel<S: Session, V> {
    fn compile(&self, plc: &Placement) -> Result<NgKernel<S, V>>;
}

// TODO if rustc can't figure out how to optimize Box<dyn Fn...> for
// function kernels then we could consider returning an enum over
// fn.. and Box<dyn Fn...> in the traits below instead

pub(crate) type TypedNullaryKernel<S, P, Y> = Box<dyn Fn(&S, &P) -> Result<Y> + Send + Sync>;

pub(crate) trait NullaryKernel<S: Session, P, Y> {
    fn compile(&self) -> Result<TypedNullaryKernel<S, P, Y>>;
}

pub(crate) type TypedUnaryKernel<S, P, X0, Y> = Box<dyn Fn(&S, &P, X0) -> Result<Y> + Send + Sync>;

pub(crate) trait UnaryKernel<S: Session, P, X0, Y> {
    fn compile(&self) -> Result<TypedUnaryKernel<S, P, X0, Y>>;
}

pub(crate) type TypedBinaryKernel<S, P, X0, X1, Y> =
    Box<dyn Fn(&S, &P, X0, X1) -> Result<Y> + Send + Sync>;

pub(crate) trait BinaryKernel<S: Session, P, X0, X1, Y> {
    fn compile(&self) -> Result<TypedBinaryKernel<S, P, X0, X1, Y>>;
}

pub(crate) type TypedTernaryKernel<S, P, X0, X1, X2, Y> =
    Box<dyn Fn(&S, &P, X0, X1, X2) -> Result<Y> + Send + Sync>;

pub(crate) trait TernaryKernel<S: Session, P, X0, X1, X2, Y> {
    fn compile(&self) -> Result<TypedTernaryKernel<S, P, X0, X1, X2, Y>>;
}

pub(crate) type TypedVariadicKernel<S, P, XS, Y> =
    Box<dyn Fn(&S, &P, Operands<XS>) -> Result<Y> + Send + Sync>;

pub(crate) type TypedVariadicKernelSlice<S, P, XS, Y> =
    Box<dyn Fn(&S, &P, &[XS]) -> Result<Y> + Send + Sync>;

pub(crate) trait VariadicKernel<S: Session, P, XS, Y> {
    fn compile(&self) -> Result<TypedVariadicKernel<S, P, XS, Y>>;
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

pub trait PlacementKeyGen<S: Session, KeyT> {
    fn gen_key(&self, sess: &S) -> KeyT;
}

modelled_kernel! {
    PlacementKeyGen::gen_key, PrfKeyGenOp,
    [
        (HostPlacement, () -> HostPrfKey => [runtime] Self::kernel),
    ]
}

pub trait PlacementTruncPr<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: u32, x: &T) -> O;
}

modelled_kernel! {
    PlacementTruncPr::trunc_pr, TruncPrOp{amount: u32},
    [
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
        // TODO(Morten) should we rename this as a shift?
        (ReplicatedPlacement,  (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement,  (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_kernel),
    ]
}

pub trait PlacementPlace<S: Session, T> {
    fn place(&self, sess: &S, x: T) -> T;
}

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
        (HostPlacement, (Uint64Tensor) -> Uint64Tensor => [concrete] Self::u64_host_kernel),
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
        (HostPlacement, (HostUnit) -> HostUnit => [runtime] Self::missing_kernel),
        (HostPlacement, (HostShape) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (HostSeed) -> HostSeed => [runtime] Self::kernel),
        (HostPlacement, (HostPrfKey) -> HostPrfKey => [runtime] Self::kernel),
    ]
}

pub trait PlacementShlDim<S: Session, T, O> {
    fn shl_dim(&self, sess: &S, amount: usize, ring_size: usize, x: &T) -> O;
}

modelled_kernel! {
    PlacementShlDim::shl_dim, ShlDimOp{amount: usize, bit_length: usize},
    [
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::host_bit_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_bit_kernel),
    ]
}

pub trait PlacementSoftmax<S: Session, T, O> {
    fn softmax(&self, sess: &S, axis: usize, upmost_index: usize, x: &T) -> O;
}

modelled_kernel! {
    PlacementSoftmax::softmax, SoftmaxOp{axis: usize, upmost_index: usize},
    [
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] Self::logical_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [transparent] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [transparent] Self::rep_fixed_kernel),
    ]
}

pub trait PlacementBroadcast<S: Session, ShapeT, T, O> {
    fn broadcast(&self, sess: &S, s: &ShapeT, x: &T) -> O;
}

modelled_kernel! {
    PlacementBroadcast::broadcast, BroadcastOp,
    [
        (HostPlacement, (HostShape, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (HostShape, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (HostShape, HostBitTensor) -> HostBitTensor => [runtime] Self::host_bit_kernel),
        (ReplicatedPlacement, (ReplicatedShape, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedShape, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedShape, ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_ring_kernel),
    ]
}
