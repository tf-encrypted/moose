use crate::bit::BitTensor;
use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::{
    map_receive_error, map_send_result, AsyncKernel, CompilationContext, Compile, Kernel,
    SyncKernel,
};
use crate::prim::{PrfKey, RawNonce, RawPrfKey, RawSeed, Seed};
use crate::replicated::ReplicatedSetup;
use crate::ring::AbstractRingTensor;
use crate::ring::{Ring128Tensor, Ring64Tensor};
use crate::standard::Int16Tensor;
use crate::standard::Int8Tensor;
use crate::standard::StandardTensor;
use crate::standard::Uint16Tensor;
use crate::standard::Uint8Tensor;
use crate::standard::{
    Float32Tensor, Float64Tensor, Int32Tensor, Int64Tensor, Shape, Uint32Tensor, Uint64Tensor,
};
use crate::{closure_kernel, function_kernel};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;

/// General session trait determining basic properties for session objects.
pub trait Session {
    type Value;
    fn execute(&self, op: Operator, plc: &Placement, operands: Vec<Self::Value>) -> Self::Value;

    type ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup;
}

/// Trait for sessions that are intended for run-time use only.
///
/// This trait is used to make a distinct between functionality that may
/// only be executed during run-time as opposed to at compile-time, such
/// as for instance key generation. Moreover, it also offers access to
/// information that is only known at run-time, such as the concrete
/// session id under which execution is happening.
pub trait RuntimeSession: Session {
    fn session_id(&self) -> &SessionId;
}

/// Session object for synchronous/eager execution (in new framework).
pub struct SyncSession {
    session_id: SessionId,
    replicated_keys: HashMap<ReplicatedPlacement, ReplicatedSetup>,
}

impl Default for SyncSession {
    fn default() -> Self {
        SyncSession {
            session_id: "abcde".into(), // TODO
            replicated_keys: Default::default(),
        }
    }
}

impl Session for SyncSession {
    type Value = Value;

    fn execute(&self, op: Operator, plc: &Placement, operands: Vec<Value>) -> Value {
        match op {
            Operator::Shape(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::BitFill(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::Convert(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingFill(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::PrimPrfKeyGen(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::BitSample(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::BitXor(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::BitAnd(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingSample(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingAdd(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingAnd(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingSub(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingMul(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingDot(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingNeg(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingShl(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingShr(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingSum(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepSetup(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepShare(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepReveal(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepAdd(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepMul(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepDot(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepTruncPr(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepAbs(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepToAdt(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepMean(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepSum(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::AdtAdd(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::AdtSub(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::AdtShl(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::AdtMul(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::AdtReveal(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::AdtToRep(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::PrimDeriveSeed(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::Constant(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::StdOnes(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointRingMean(op) => DispatchKernel::compile(&op, plc)(self, operands),
            op => unimplemented!("SyncSession implementation is missing for {:?}", op), // TODO
        }
    }

    type ReplicatedSetup = ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup {
        self.replicated_keys.get(plc).unwrap()
    }
}

impl RuntimeSession for SyncSession {
    fn session_id(&self) -> &SessionId {
        &self.session_id
    }
}

/// Session object for asynchronous execution (in new framework).
pub struct AsyncSession {
    session_id: SessionId,
    // replicated_keys: HashMap<ReplicatedPlacement, ReplicatedSetup>,
}

impl Session for AsyncSession {
    type Value = (); // TODO
    fn execute(&self, _op: Operator, _plc: &Placement, _operands: Vec<Self::Value>) -> Self::Value {
        // TODO
        unimplemented!()
    }

    type ReplicatedSetup = (); // TODO
    fn replicated_setup(&self, _plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup {
        // TODO
        unimplemented!()
    }
}

impl RuntimeSession for AsyncSession {
    fn session_id(&self) -> &SessionId {
        &self.session_id
    }
}

pub trait DispatchKernel<S: Session> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &Placement) -> Box<dyn Fn(&S, Vec<S::Value>) -> S::Value>;
}

// TODO if rustc can't figure out how to optimize Box<dyn Fn...> for
// function kernels then we could consider returning an enum over
// fn.. and Box<dyn Fn...> in the traits below instead

pub trait NullaryKernel<S: Session, P, Y> {
    fn compile(&self, plc: &P) -> Box<dyn Fn(&S, &P) -> Y>;
}

pub trait UnaryKernel<S: Session, P, X0, Y> {
    fn compile(&self, plc: &P) -> Box<dyn Fn(&S, &P, X0) -> Y>;
}

pub trait BinaryKernel<S: Session, P, X0, X1, Y> {
    fn compile(&self, plc: &P) -> Box<dyn Fn(&S, &P, X0, X1) -> Y>;
}

pub trait TernaryKernel<S: Session, P, X0, X1, X2, Y> {
    fn compile(&self, plc: &P) -> Box<dyn Fn(&S, &P, X0, X1, X2) -> Y>;
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

pub trait Tensor<S: Session> {
    type Scalar;
}

pub trait PlacementShape<S: Session, T, ShapeT> {
    fn shape(&self, sess: &S, x: &T) -> ShapeT;
}

pub trait PlacementKeyGen<S: Session, KeyT> {
    fn gen_key(&self, sess: &S) -> KeyT;
}

pub trait PlacementSetupGen<S: Session, SetupT> {
    fn gen_setup(&self, sess: &S) -> SetupT;
}

pub trait PlacementDeriveSeed<S: Session, KeyT, SeedT> {
    fn derive_seed(&self, sess: &S, sync_key: RawNonce, key: &KeyT) -> SeedT;
}

pub trait PlacementAdd<S: Session, T, U, O> {
    fn add(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementSub<S: Session, T, U, O> {
    fn sub(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementNeg<S: Session, T, O> {
    fn neg(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementMul<S: Session, T, U, O> {
    fn mul(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementDot<S: Session, T, U, O> {
    fn dot(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementShl<S: Session, T, O> {
    fn shl(&self, sess: &S, amount: usize, x: &T) -> O;
}

pub trait PlacementShr<S: Session, T, O> {
    fn shr(&self, sess: &S, amount: usize, x: &T) -> O;
}

pub trait PlacementXor<S: Session, T, U, O> {
    fn xor(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementAnd<S: Session, T, U, O> {
    fn and(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementConvert<S: Session, T, O> {
    fn convert(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementMulSetup<S: Session, SetupT, T, U, O> {
    fn mul(&self, sess: &S, setup: &SetupT, x: &T, y: &U) -> O;
}

pub trait PlacementDotSetup<S: Session, SetupT, T, U, O> {
    fn dot(&self, sess: &S, setup: &SetupT, x: &T, y: &U) -> O;
}

pub trait PlacementShare<S: Session, T, O> {
    fn share(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementShareSetup<S: Session, SetupT, T, O> {
    fn share(&self, sess: &S, setup: &SetupT, x: &T) -> O;
}

pub trait PlacementReveal<S: Session, T, O> {
    fn reveal(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementFill<S: Session, ShapeT, O> {
    fn fill(&self, sess: &S, value: Constant, shape: &ShapeT) -> O;
}

pub trait PlacementZeros<S: Session, ShapeT, O> {
    fn zeros(&self, sess: &S, shape: &ShapeT) -> O;
}

pub trait PlacementMean<S: Session, T, O> {
    fn mean(&self, sess: &S, axis: Option<u32>, precision: u64, x: &T) -> O;
}

pub trait PlacementRingMean<S: Session, T, O> {
    fn ring_mean(
        &self,
        sess: &S,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: &T,
    ) -> O;
}

pub trait PlacementSum<S: Session, T, O> {
    fn sum(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementZeros<S, ShapeT, O> for P
where
    P: PlacementFill<S, ShapeT, O>,
    O: Tensor<S>,
    O::Scalar: Into<Constant>,
    O::Scalar: From<u8>,
{
    fn zeros(&self, sess: &S, shape: &ShapeT) -> O {
        let value = O::Scalar::from(0).into();
        self.fill(sess, value, shape)
    }
}

pub trait PlacementStdOnes<S: Session, ShapeT, O> {
    fn std_ones(&self, sess: &S, shape: &ShapeT) -> O;
}

modelled!(PlacementStdOnes::std_ones, HostPlacement, (Shape) -> Float64Tensor, StdOnesOp);

kernel! {
    StdOnesOp, [
        (HostPlacement, (Shape) -> Float64Tensor => Self::kernel),
    ]
}

pub trait PlacementOnes<S: Session, ShapeT, O> {
    fn ones(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementOnes<S, ShapeT, O> for P
where
    P: PlacementFill<S, ShapeT, O>,
    O: Tensor<S>,
    O::Scalar: Into<Constant>,
    O::Scalar: From<u8>,
{
    fn ones(&self, sess: &S, shape: &ShapeT) -> O {
        let value = O::Scalar::from(1).into();
        self.fill(sess, value, shape)
    }
}

pub trait PlacementSample<S: Session, ShapeT, SeedT, O> {
    fn sample(&self, sess: &S, max_value: Option<u64>, shape: &ShapeT, seed: &SeedT) -> O;
}

pub trait PlacementSampleUniform<S: Session, ShapeT, SeedT, O> {
    fn sample_uniform(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

impl<S: Session, ShapeT, SeedT, O, P> PlacementSampleUniform<S, ShapeT, SeedT, O> for P
where
    P: PlacementSample<S, ShapeT, SeedT, O>,
{
    fn sample_uniform(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O {
        self.sample(sess, None, shape, seed)
    }
}

pub trait PlacementSampleBits<S: Session, ShapeT, SeedT, O> {
    fn sample_bits(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

impl<S: Session, ShapeT, SeedT, O, P> PlacementSampleBits<S, ShapeT, SeedT, O> for P
where
    P: PlacementSample<S, ShapeT, SeedT, O>,
{
    fn sample_bits(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O {
        self.sample(sess, Some(1), shape, seed)
    }
}

pub trait PlacementRepToAdt<S: Session, T, O> {
    fn rep_to_adt(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementAdtToRep<S: Session, T, O> {
    fn adt_to_rep(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementTruncPr<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: usize, x: &T) -> O;
}

pub trait PlacementTruncPrProvider<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: usize, provider: &HostPlacement, x: &T) -> O;
}

pub trait PlacementAbs<S: Session, T, O> {
    fn abs(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementPlace<S: Session, T> {
    fn place(&self, sess: &S, x: T) -> T;
}

pub trait PlacementConstant<S: Session, O> {
    fn constant(&self, sess: &S, value: Constant) -> O;
}

pub trait PlacementInput<S: Session, O> {
    fn input(&self, sess: &S, arg_name: String) -> O;
}

pub trait PlacementOutput<S: Session, T, O> {
    fn output(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementLoad<S: Session, KeyT, QueryT, O> {
    fn load(&self, sess: &S, key: &KeyT, query: &QueryT) -> O;
}

pub trait PlacementSave<S: Session, KeyT, T, O> {
    fn save(&self, sess: &S, key: &KeyT, x: &T) -> O;
}

pub trait PlacementStdAtLeast2D<S: Session, T, O> {
    fn std_at_least_2d(&self, sess: &S, to_column_vector: bool, x: &T) -> O;
}

pub trait PlacementFixedpointRingEncode<S: Session, T, O> {
    fn fixedpoint_ring_encode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
}

pub trait PlacementFixedpointRingDecode<S: Session, T, O> {
    fn fixedpoint_ring_decode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
}

pub trait PlacementStdMean<S: Session, T, O> {
    fn std_mean(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

pub trait PlacementStdSum<S: Session, T, O> {
    fn std_sum(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

pub trait PlacementStdExpandDims<S: Session, T, O> {
    fn std_expand_dims(&self, sess: &S, axis: u32, x: &T) -> O;
}

pub trait PlacementStdConcatenate<S: Session, T1, T2, O> {
    fn std_concatenate(&self, sess: &S, axis: u32, x: &T1, y: &T2) -> O;
}

pub trait PlacementStdAdd<S: Session, T1, T2, O> {
    fn std_add(&self, sess: &S, x: &T1, y: &T2) -> O;
}

pub trait PlacementStdSub<S: Session, T1, T2, O> {
    fn std_sub(&self, sess: &S, x: &T1, y: &T2) -> O;
}

pub trait PlacementStdMul<S: Session, T1, T2, O> {
    fn std_mul(&self, sess: &S, x: &T1, y: &T2) -> O;
}

pub trait PlacementStdDiv<S: Session, T1, T2, O> {
    fn std_div(&self, sess: &S, x: &T1, y: &T2) -> O;
}

pub trait PlacementStdDot<S: Session, T1, T2, O> {
    fn std_dot(&self, sess: &S, x: &T1, y: &T2) -> O;
}

pub trait PlacementStdTranspose<S: Session, T, O> {
    fn std_transpose(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementStdInverse<S: Session, T, O> {
    fn std_inverse(&self, sess: &S, x: &T) -> O;
}

pub trait EmptyTypeHolder<T> {}

// The `T` type parameter is required by the modelled!() macros, but we are enforcing that T = ShapeT.
pub trait PlacementSlice<S: Session, ShapeT, T>
where
    // Forces ShapeT = T
    dyn EmptyTypeHolder<ShapeT>: EmptyTypeHolder<T>,
{
    fn slice(&self, sess: &S, start: u32, end: u32, x: &ShapeT) -> ShapeT;
}

fn check_type(v: &Value, expected: Ty) -> Result<()> {
    if v.ty() == expected {
        Ok(())
    } else {
        Err(Error::TypeMismatch {
            expected: format!("{:?}", expected),
            found: v.ty(),
        })
    }
}

impl Compile<SyncKernel> for Operator {
    fn compile(&self, ctx: &CompilationContext) -> Result<SyncKernel> {
        use Operator::*;
        match self {
            Identity(op) => Compile::<SyncKernel>::compile(op, ctx),
            Load(op) => Compile::<SyncKernel>::compile(op, ctx),
            Save(op) => Compile::<SyncKernel>::compile(op, ctx),
            Send(op) => Compile::<SyncKernel>::compile(op, ctx),
            Receive(op) => Compile::<SyncKernel>::compile(op, ctx),
            Input(op) => Compile::<SyncKernel>::compile(op, ctx),
            Output(op) => Compile::<SyncKernel>::compile(op, ctx),
            Constant(op) => Compile::<SyncKernel>::compile(op, ctx),
            Shape(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitFill(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingFill(op) => Compile::<SyncKernel>::compile(op, ctx),
            Convert(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdAdd(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdSub(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdMul(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdDiv(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdDot(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdMean(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdOnes(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdConcatenate(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdExpandDims(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdReshape(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdAtLeast2D(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdSlice(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdSum(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdTranspose(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdInverse(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingAdd(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingAnd(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSub(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingMul(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingDot(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSum(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSample(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingShl(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingShr(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingInject(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitSample(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitXor(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitAnd(op) => Compile::<SyncKernel>::compile(op, ctx),
            PrimDeriveSeed(op) => Compile::<SyncKernel>::compile(op, ctx),
            PrimPrfKeyGen(op) => Compile::<SyncKernel>::compile(op, ctx),
            FixedpointRingEncode(op) => Compile::<SyncKernel>::compile(op, ctx),
            FixedpointRingDecode(op) => Compile::<SyncKernel>::compile(op, ctx),
            FixedpointRingMean(op) => Compile::<SyncKernel>::compile(op, ctx),
            op => unimplemented!("deprecated, should not impl for {:?}", op),
        }
    }
}

impl Compile<AsyncKernel> for Operator {
    fn compile(&self, ctx: &CompilationContext) -> Result<AsyncKernel> {
        use Operator::*;
        match self {
            Identity(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Load(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Save(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Send(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Receive(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Input(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Output(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Constant(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Shape(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitFill(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingFill(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdAdd(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdSub(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdMul(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdDiv(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdDot(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdMean(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdOnes(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdConcatenate(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdExpandDims(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdReshape(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdAtLeast2D(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdSlice(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdSum(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdTranspose(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdInverse(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingNeg(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingAdd(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingAnd(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSub(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingMul(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingDot(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSum(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSample(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingShl(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingShr(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingInject(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitSample(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitXor(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitAnd(op) => Compile::<AsyncKernel>::compile(op, ctx),
            PrimDeriveSeed(op) => Compile::<AsyncKernel>::compile(op, ctx),
            PrimPrfKeyGen(op) => Compile::<AsyncKernel>::compile(op, ctx),
            FixedpointRingEncode(op) => Compile::<AsyncKernel>::compile(op, ctx),
            FixedpointRingDecode(op) => Compile::<AsyncKernel>::compile(op, ctx),
            FixedpointRingMean(op) => Compile::<AsyncKernel>::compile(op, ctx),
            op => unimplemented!("deprecated, should not impl for {:?}", op),
        }
    }
}

macro_rules! signature {
    (() -> $ret: pat) => {
        Signature::Nullary(NullarySignature { ret: $ret })
    };
    (($t0: pat) -> $ret: pat) => {
        Signature::Unary(UnarySignature {
            arg0: $t0,
            ret: $ret,
        })
    };
    (($t0: pat, $t1: pat) -> $ret: pat) => {
        Signature::Binary(BinarySignature {
            arg0: $t0,
            arg1: $t1,
            ret: $ret,
        })
    };
    (($t0: pat, $t1: pat, $t2: pat) -> $ret: pat) => {
        Signature::Ternary(TernarySignature {
            arg0: $t0,
            arg1: $t1,
            arg2: $t2,
            ret: $ret,
        })
    };
}

macro_rules! std_unary_kernel {
    ($op:ty, $k:expr) => {
        impl Compile<Kernel> for $op {
            fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
                match self.sig {
                    signature![(Ty::Float32Tensor) -> _] => {
                        function_kernel!(Float32Tensor, $k)
                    }
                    signature![(Ty::Float64Tensor) -> _] => {
                        function_kernel!(Float64Tensor, $k)
                    }
                    signature![(Ty::Int32Tensor) -> _] => {
                        function_kernel!(Int32Tensor, $k)
                    }
                    signature![(Ty::Int64Tensor) -> _] => {
                        function_kernel!(Int64Tensor, $k)
                    }
                    signature![(Ty::Uint32Tensor) -> _] => {
                        function_kernel!(Uint32Tensor, $k)
                    }
                    signature![(Ty::Uint64Tensor) -> _] => {
                        function_kernel!(Uint64Tensor, $k)
                    }
                    _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
                }
            }
        }
    };
}

macro_rules! std_binary_kernel {
    ($op:ident, $t:ident::$f:ident, $k:expr) => {
        impl Compile<Kernel> for $op {
            fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
                match self.sig {
                    signature![(Ty::Float32Tensor, Ty::Float32Tensor) -> _] => {
                        function_kernel!(Float32Tensor, Float32Tensor, $k)
                    }
                    signature![(Ty::Float64Tensor, Ty::Float64Tensor) -> _] => {
                        function_kernel!(Float64Tensor, Float64Tensor, $k)
                    }
                    signature![(Ty::Int32Tensor, Ty::Int32Tensor) -> _] => {
                        function_kernel!(Int32Tensor, Int32Tensor, $k)
                    }
                    signature![(Ty::Int64Tensor, Ty::Int64Tensor) -> _] => {
                        function_kernel!(Int64Tensor, Int64Tensor, $k)
                    }
                    signature![(Ty::Uint32Tensor, Ty::Uint32Tensor) -> _] => {
                        function_kernel!(Uint32Tensor, Uint32Tensor, $k)
                    }
                    signature![(Ty::Uint64Tensor, Ty::Uint64Tensor) -> _] => {
                        function_kernel!(Uint64Tensor, Uint64Tensor, $k)
                    }
                    _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
                }
            }
        }


        modelled!($t::$f, HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor, $op);
        modelled!($t::$f, HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor, $op);
        modelled!($t::$f, HostPlacement, (Int32Tensor, Int32Tensor) -> Int32Tensor, $op);
        modelled!($t::$f, HostPlacement, (Int64Tensor, Int64Tensor) -> Int64Tensor, $op);
        modelled!($t::$f, HostPlacement, (Uint32Tensor, Uint32Tensor) -> Uint32Tensor, $op);
        modelled!($t::$f, HostPlacement, (Uint64Tensor, Uint64Tensor) -> Uint64Tensor, $op);

        kernel! {
            $op, [
                (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => Self::kernel),
                (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => Self::kernel),
                (HostPlacement, (Int32Tensor, Int32Tensor) -> Int32Tensor => Self::kernel),
                (HostPlacement, (Int64Tensor, Int64Tensor) -> Int64Tensor => Self::kernel),
                (HostPlacement, (Uint32Tensor, Uint32Tensor) -> Uint32Tensor => Self::kernel),
                (HostPlacement, (Uint64Tensor, Uint64Tensor) -> Uint64Tensor => Self::kernel),
            ]
        }
    };
}

std_binary_kernel!(StdAddOp, PlacementStdAdd::std_add, |x, y| x + y);
std_binary_kernel!(StdSubOp, PlacementStdSub::std_sub, |x, y| x - y);
std_binary_kernel!(StdMulOp, PlacementStdMul::std_mul, |x, y| x * y);
std_binary_kernel!(StdDivOp, PlacementStdDiv::std_div, |x, y| x / y);
std_binary_kernel!(StdDotOp, PlacementStdDot::std_dot, |x, y| x.dot(y));
std_unary_kernel!(StdTransposeOp, |x| x.transpose());

modelled!(PlacementStdTranspose::std_transpose, HostPlacement, (Float64Tensor) -> Float64Tensor, StdTransposeOp);

kernel! {
    StdTransposeOp, [
        (HostPlacement, (Float64Tensor) -> Float64Tensor => Self::kernel),
    ]
}

modelled!(PlacementStdInverse::std_inverse, HostPlacement, (Float64Tensor) -> Float64Tensor, StdInverseOp);

kernel! {
    StdInverseOp, [
        (HostPlacement, (Float64Tensor) -> Float64Tensor => Self::kernel),
    ]
}

impl Compile<Kernel> for StdInverseOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                closure_kernel!(Float32Tensor, |x| x.inv())
            }
            signature![(_) -> Ty::Float64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.inv())
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementStdMean::std_mean, HostPlacement, attributes[axis: Option<u32>] (Float64Tensor) -> Float64Tensor, StdMeanOp);

kernel! {
    StdMeanOp, [
        (HostPlacement, (Float64Tensor) -> Float64Tensor => attributes[axis] Self::kernel),
    ]
}

impl Compile<Kernel> for StdMeanOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|x| x as usize);
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                closure_kernel!(Float32Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::Float64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::Int32Tensor] => {
                closure_kernel!(Int32Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::Int64Tensor] => {
                closure_kernel!(Int64Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::Uint32Tensor] => {
                closure_kernel!(Uint32Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::Uint64Tensor] => {
                closure_kernel!(Uint64Tensor, |x| x.mean(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdOnesOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                function_kernel!(Shape, |shape| Float32Tensor::ones(shape))
            }
            signature![(_) -> Ty::Float64Tensor] => {
                function_kernel!(Shape, |shape| Float64Tensor::ones(shape))
            }
            signature![(_) -> Ty::Int32Tensor] => {
                function_kernel!(Shape, |shape| Int32Tensor::ones(shape))
            }
            signature![(_) -> Ty::Int64Tensor] => {
                function_kernel!(Shape, |shape| Int64Tensor::ones(shape))
            }
            signature![(_) -> Ty::Uint32Tensor] => {
                function_kernel!(Shape, |shape| Uint32Tensor::ones(shape))
            }
            signature![(_) -> Ty::Uint64Tensor] => {
                function_kernel!(Shape, |shape| Uint64Tensor::ones(shape))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementStdConcatenate::std_concatenate, HostPlacement, attributes[axis: u32] (Float64Tensor, Float64Tensor) -> Float64Tensor, StdConcatenateOp);

kernel! {
    StdConcatenateOp, [
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => attributes[axis] Self::kernel),
    ]
}

impl Compile<Kernel> for StdConcatenateOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        use crate::standard::concatenate;
        let axis = self.axis as usize;
        match self.sig {
            signature![(_, _) -> Ty::Float32Tensor] => {
                closure_kernel!(vec[Float32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::Float64Tensor] => {
                closure_kernel!(vec[Float64Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::Int32Tensor] => {
                closure_kernel!(vec[Int32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::Int64Tensor] => {
                closure_kernel!(vec[Int64Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::Uint32Tensor] => {
                closure_kernel!(vec[Uint32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::Uint64Tensor] => {
                closure_kernel!(vec[Uint64Tensor], |xs| concatenate(axis, &xs))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementStdExpandDims::std_expand_dims, HostPlacement, attributes[axis: u32] (Float64Tensor) -> Float64Tensor, StdExpandDimsOp);

kernel! {
    StdExpandDimsOp, [
        (HostPlacement, (Float64Tensor) -> Float64Tensor => attributes[axis] Self::kernel),
    ]
}

impl Compile<Kernel> for StdExpandDimsOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis as usize;
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                closure_kernel!(Float32Tensor, |x| x.expand_dims(axis))
            }
            signature![(_) -> Ty::Float64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.expand_dims(axis))
            }
            signature![(_) -> Ty::Int32Tensor] => {
                closure_kernel!(Int32Tensor, |x| x.expand_dims(axis))
            }
            signature![(_) -> Ty::Int64Tensor] => {
                closure_kernel!(Int64Tensor, |x| x.expand_dims(axis))
            }
            signature![(_) -> Ty::Uint32Tensor] => {
                closure_kernel!(Uint32Tensor, |x| x.expand_dims(axis))
            }
            signature![(_) -> Ty::Uint64Tensor] => {
                closure_kernel!(Uint64Tensor, |x| x.expand_dims(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdReshapeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(_, _) -> Ty::Float32Tensor] => {
                function_kernel!(Float32Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            signature![(_, _) -> Ty::Float64Tensor] => {
                function_kernel!(Float64Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            signature![(_, _) -> Ty::Int32Tensor] => {
                function_kernel!(Int32Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            signature![(_, _) -> Ty::Int64Tensor] => {
                function_kernel!(Int64Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            signature![(_, _) -> Ty::Uint32Tensor] => {
                function_kernel!(Uint32Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            signature![(_, _) -> Ty::Uint64Tensor] => {
                function_kernel!(Uint64Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementStdAtLeast2D::std_at_least_2d, HostPlacement, attributes[to_column_vector: bool] (Float32Tensor) -> Float32Tensor, StdAtLeast2DOp);
modelled!(PlacementStdAtLeast2D::std_at_least_2d, HostPlacement, attributes[to_column_vector: bool] (Float64Tensor) -> Float64Tensor, StdAtLeast2DOp);

kernel! {
    StdAtLeast2DOp, [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => attributes[to_column_vector] Self::kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => attributes[to_column_vector] Self::kernel),
    ]
}

impl StdAtLeast2DOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _to_column_vector: bool,
        _x: StandardTensor<T>,
    ) -> StandardTensor<T> {
        unimplemented!()
    }
}

impl Compile<Kernel> for StdAtLeast2DOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let tcv = self.to_column_vector;
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::Float64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::Int32Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::Int64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::Uint32Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::Uint64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdSliceOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let start = self.start as usize;
        let end = self.end as usize;
        match self.sig {
            signature![(_) -> Ty::Shape] => {
                closure_kernel!(Shape, |x| Shape(x.0.slice(start, end), x.1))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementStdSum::std_sum, HostPlacement, attributes[axis: Option<u32>] (Float64Tensor) -> Float64Tensor, StdSumOp);

kernel! {
    StdSumOp, [
        (HostPlacement, (Float64Tensor) -> Float64Tensor => attributes[axis] Self::kernel),
    ]
}

impl Compile<Kernel> for StdSumOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                closure_kernel!(Float32Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::Float64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::Int32Tensor] => {
                closure_kernel!(Int32Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::Int64Tensor] => {
                closure_kernel!(Int64Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::Uint32Tensor] => {
                closure_kernel!(Uint32Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::Uint64Tensor] => {
                closure_kernel!(Uint64Tensor, |x| x.sum(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for PrimDeriveSeedOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let nonce = self.sync_key.clone();
        closure_kernel!(PrfKey, |key| Seed(
            RawSeed::from_prf(&key.0, &nonce),
            HostPlacement {
                owner: "TODO".into()
            }
        ))
    }
}

impl Compile<Kernel> for PrimPrfKeyGenOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(|| PrfKey(
            RawPrfKey::generate(),
            HostPlacement {
                owner: "TODO".into()
            }
        ))
    }
}

impl Compile<Kernel> for RingAddOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::Ring64Tensor, Ty::Ring64Tensor) -> _] => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x + y)
            }
            signature![(Ty::Ring128Tensor, Ty::Ring128Tensor) -> _] => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x + y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingAndOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::Ring64Tensor, Ty::Ring64Tensor) -> _] => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x & y)
            }
            signature![(Ty::Ring128Tensor, Ty::Ring128Tensor) -> _] => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x & y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSubOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::Ring64Tensor, Ty::Ring64Tensor) -> _] => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x - y)
            }
            signature![(Ty::Ring128Tensor, Ty::Ring128Tensor) -> _] => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x - y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingMulOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::Ring64Tensor, Ty::Ring64Tensor) -> _] => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x * y)
            }
            signature![(Ty::Ring128Tensor, Ty::Ring128Tensor) -> _] => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x * y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingDotOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::Ring64Tensor, Ty::Ring64Tensor) -> _] => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x.dot(y))
            }
            signature![(Ty::Ring128Tensor, Ty::Ring128Tensor) -> _] => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x.dot(y))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSumOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.sig {
            signature![(_) -> Ty::Ring64Tensor] => closure_kernel!(Ring64Tensor, |x| x.sum(axis)),
            signature![(_) -> Ty::Ring128Tensor] => {
                closure_kernel!(Ring128Tensor, |x| x.sum(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for ShapeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::Float32Tensor) -> Ty::Shape] => {
                function_kernel!(Float32Tensor, |x| x.shape())
            }
            signature![(Ty::Float64Tensor) -> Ty::Shape] => {
                function_kernel!(Float64Tensor, |x| x.shape())
            }
            signature![(Ty::Ring64Tensor) -> Ty::Shape] => {
                function_kernel!(Ring64Tensor, |x| x.shape())
            }
            signature![(Ty::Ring128Tensor) -> Ty::Shape] => {
                function_kernel!(Ring128Tensor, |x| x.shape())
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for BitFillOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (&self.sig, self.value.clone()) {
            (signature![(_) -> Ty::BitTensor], Constant::Ring64(value)) => {
                closure_kernel!(Shape, |shape| {
                    assert!(value == 0 || value == 1);
                    BitTensor::fill(&shape.0, value as u8)
                })
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingFillOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (&self.sig, self.value.clone()) {
            (signature![(_) -> Ty::Ring64Tensor], Constant::Ring64(value)) => {
                closure_kernel!(Shape, |shape| Ring64Tensor::fill(&shape.0, value))
            }
            (signature![(_) -> Ty::Ring128Tensor], Constant::Ring64(value)) => {
                closure_kernel!(Shape, |shape| Ring128Tensor::fill(&shape.0, value as u128))
            }
            (signature![(_) -> Ty::Ring128Tensor], Constant::Ring128(value)) => {
                closure_kernel!(Shape, |shape| Ring128Tensor::fill(&shape.0, value))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSampleOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (&self.sig, self.max_value) {
            (signature![(_, _) -> Ty::Ring64Tensor], None) => {
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_uniform(
                    &shape.0, &seed.0
                ))
            }
            (signature!((_, _) -> Ty::Ring64Tensor), Some(max_value)) if max_value == 1 => {
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_bits(
                    &shape.0, &seed.0
                ))
            }
            (signature![(_, _) -> Ty::Ring128Tensor], None) => {
                function_kernel!(Shape, Seed, |shape, seed| Ring128Tensor::sample_uniform(
                    &shape.0, &seed.0
                ))
            }
            (signature![(_, _) -> Ty::Ring128Tensor], Some(max_value)) if max_value == 1 => {
                function_kernel!(Shape, Seed, |shape, seed| Ring128Tensor::sample_bits(
                    &shape.0, &seed.0
                ))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingNegOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(_) -> Ty::Ring64Tensor] => {
                closure_kernel!(Ring64Tensor, |x| AbstractRingTensor(-x.0, x.1))
            }
            signature![(_) -> Ty::Ring128Tensor] => {
                closure_kernel!(Ring128Tensor, |x| AbstractRingTensor(-x.0, x.1))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingShlOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let amount = self.amount;
        match self.sig {
            signature![(_) -> Ty::Ring64Tensor] => {
                closure_kernel!(Ring64Tensor, |x| x << amount)
            }
            signature![(_) -> Ty::Ring128Tensor] => {
                closure_kernel!(Ring128Tensor, |x| x << amount)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingShrOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let amount = self.amount;
        match self.sig {
            signature![(_) -> Ty::Ring64Tensor] => {
                closure_kernel!(Ring64Tensor, |x| x >> amount)
            }
            signature![(_) -> Ty::Ring128Tensor] => {
                closure_kernel!(Ring128Tensor, |x| x >> amount)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingInjectOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let bit_idx = self.bit_idx;
        match self.sig {
            signature![(_) -> Ty::Ring64Tensor] => {
                closure_kernel!(BitTensor, |x| Ring64Tensor::from(x) << bit_idx)
            }
            signature![(_) -> Ty::Ring128Tensor] => {
                closure_kernel!(BitTensor, |x| Ring128Tensor::from(x) << bit_idx)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for BitExtractOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let bit_idx = self.bit_idx;
        match self.sig {
            signature![(Ty::Ring64Tensor) -> _] => {
                closure_kernel!(Ring64Tensor, |x| x.bit_extract(bit_idx))
            }
            signature![(Ty::Ring128Tensor) -> _] => {
                closure_kernel!(Ring128Tensor, |x| x.bit_extract(bit_idx))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for BitSampleOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(Shape, Seed, |shape, seed| BitTensor::sample_uniform(
            &shape.0, &seed.0
        ))
    }
}

impl Compile<Kernel> for BitXorOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(BitTensor, BitTensor, |x, y| x ^ y)
    }
}

impl Compile<Kernel> for BitAndOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(BitTensor, BitTensor, |x, y| x & y)
    }
}

modelled!(PlacementFixedpointRingEncode::fixedpoint_ring_encode, HostPlacement, attributes[scaling_base: u64, scaling_exp: u32] (Float64Tensor) -> Ring128Tensor, FixedpointRingEncodeOp);
modelled!(PlacementFixedpointRingEncode::fixedpoint_ring_encode, HostPlacement, attributes[scaling_base: u64, scaling_exp: u32] (Float32Tensor) -> Ring64Tensor, FixedpointRingEncodeOp);

kernel! {
    FixedpointRingEncodeOp, [
        (HostPlacement, (Float64Tensor) -> Ring128Tensor => attributes[scaling_base, scaling_exp] Self::kernel),
        (HostPlacement, (Float32Tensor) -> Ring64Tensor => attributes[scaling_base, scaling_exp] Self::kernel),
    ]
}

impl FixedpointRingEncodeOp {
    fn kernel<S: RuntimeSession, ST, TT>(
        _sess: &S,
        _plc: &HostPlacement,
        _scaling_base: u64,
        _scaling_exp: u32,
        _x: StandardTensor<ST>,
    ) -> AbstractRingTensor<TT> {
        unimplemented!()
    }
}

impl Compile<Kernel> for FixedpointRingEncodeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        use crate::fixedpoint::Convert;
        match self.sig {
            signature![(Ty::Float64Tensor) -> Ty::Ring64Tensor] => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(Float64Tensor, |x| Ring64Tensor::encode(&x, scaling_factor))
            }
            signature![(Ty::Float64Tensor) -> Ty::Ring128Tensor] => {
                let scaling_factor = u128::pow(self.scaling_base as u128, self.scaling_exp);
                closure_kernel!(Float64Tensor, |x| Ring128Tensor::encode(&x, scaling_factor))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementFixedpointRingDecode::fixedpoint_ring_decode, HostPlacement, attributes[scaling_base: u64, scaling_exp: u32] (Ring128Tensor) -> Float64Tensor, FixedpointRingDecodeOp);
modelled!(PlacementFixedpointRingDecode::fixedpoint_ring_decode, HostPlacement, attributes[scaling_base: u64, scaling_exp: u32] (Ring64Tensor) -> Float32Tensor, FixedpointRingDecodeOp);

kernel! {
    FixedpointRingDecodeOp, [
        (HostPlacement, (Ring128Tensor) -> Float64Tensor => attributes[scaling_base, scaling_exp] Self::kernel),
        (HostPlacement, (Ring64Tensor) -> Float32Tensor => attributes[scaling_base, scaling_exp] Self::kernel),
    ]
}

impl FixedpointRingDecodeOp {
    fn kernel<S: RuntimeSession, ST, TT>(
        _sess: &S,
        _plc: &HostPlacement,
        _scaling_base: u64,
        _scaling_exp: u32,
        _x: AbstractRingTensor<ST>,
    ) -> StandardTensor<TT> {
        unimplemented!()
    }
}

impl Compile<Kernel> for FixedpointRingDecodeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        use crate::fixedpoint::Convert;
        match self.sig {
            signature![(Ty::Ring64Tensor) -> _] => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(Ring64Tensor, |x| Ring64Tensor::decode(&x, scaling_factor))
            }
            signature![(Ty::Ring128Tensor) -> _] => {
                let scaling_factor = u128::pow(self.scaling_base as u128, self.scaling_exp);
                closure_kernel!(Ring128Tensor, |x| Ring128Tensor::decode(&x, scaling_factor))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for FixedpointRingMeanOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.sig {
            signature![(_) -> Ty::Ring64Tensor] => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(Ring64Tensor, |x| Ring64Tensor::ring_mean(
                    x,
                    axis,
                    scaling_factor
                ))
            }
            signature![(_) -> Ty::Ring128Tensor] => {
                let scaling_factor = u128::pow(self.scaling_base as u128, self.scaling_exp);
                closure_kernel!(Ring128Tensor, |x| Ring128Tensor::ring_mean(
                    x,
                    axis,
                    scaling_factor
                ))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for ConstantOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let value = self.value.clone();
        Ok(Kernel::NullaryClosure(Arc::new(move || {
            Ok(value.place(&HostPlacement {
                owner: "TODO".into(),
            }))
        })))
    }
}

impl PlacementPlace<SyncSession, String> for HostPlacement {
    fn place(&self, _sess: &SyncSession, x: String) -> String {
        match x.placement() {
            Ok(Placement::Host(place)) if &place == self => x,
            _ => unimplemented!("Not yet able to place strings"),
        }
    }
}

macro_rules! constant_kernels {
    ($($val:ident),+) => {
        $(
            modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> $val, ConstantOp);
        )+

        kernel! {
            ConstantOp, [
                $(
                    (HostPlacement, () -> $val => attributes[value: $val] Self::kernel),
                )+
            ]
        }
    };
}

constant_kernels![
    String,
    Float32Tensor,
    Float64Tensor,
    Int8Tensor,
    Int16Tensor,
    Int32Tensor,
    Int64Tensor,
    Uint8Tensor,
    Uint16Tensor,
    Uint32Tensor,
    Uint64Tensor
];

impl ConstantOp {
    fn kernel<S: RuntimeSession, T: Placed>(sess: &S, plc: &HostPlacement, value: T) -> T
    where
        HostPlacement: PlacementPlace<S, T>,
    {
        plc.place(sess, value)
    }
}

impl Compile<SyncKernel> for SendOp {
    fn compile(&self, ctx: &CompilationContext) -> Result<SyncKernel> {
        let rendezvous_key = self.rendezvous_key.clone();
        let receiver_id = ctx
            .role_assignment
            .get(&self.receiver)
            .cloned()
            .ok_or_else(|| {
                Error::Compilation(format!(
                    "missing identity assignment for '{}'",
                    &self.receiver
                ))
            })?;

        Ok(SyncKernel::Unary(Box::new(move |sess, v| {
            sess.networking
                .send(&v, &receiver_id, &rendezvous_key, &sess.sid)?;
            Ok(Value::Unit(Unit(HostPlacement {
                owner: "TODO".into(),
            })))
        })))
    }
}

impl Compile<AsyncKernel> for SendOp {
    fn compile(&self, ctx: &CompilationContext) -> Result<AsyncKernel> {
        let rendezvous_key = Arc::new(self.rendezvous_key.clone());
        let receiver_id = Arc::new(
            ctx.role_assignment
                .get(&self.receiver)
                .cloned()
                .ok_or_else(|| {
                    Error::Compilation(format!(
                        "missing identity assignment for '{}'",
                        &self.receiver
                    ))
                })?,
        );

        Ok(AsyncKernel::Unary(Box::new(move |sess, v, sender| {
            let sess = Arc::clone(sess);
            let rendezvous_key = Arc::clone(&rendezvous_key);
            let receiver_id = Arc::clone(&receiver_id);

            tokio::spawn(async move {
                let v: Value = v.await.map_err(map_receive_error)?;
                sess.networking
                    .send(&v, &receiver_id, &rendezvous_key, &sess.sid)
                    .await?;
                map_send_result(sender.send(Value::Unit(Unit(HostPlacement {
                    owner: "TODO".into(),
                }))))
            })
        })))
    }
}

impl Compile<SyncKernel> for ReceiveOp {
    fn compile(&self, ctx: &CompilationContext) -> Result<SyncKernel> {
        let expected_ty = self.sig.ret();
        let rendezvous_key = self.rendezvous_key.clone();
        let sender_id = ctx
            .role_assignment
            .get(&self.sender)
            .cloned()
            .ok_or_else(|| {
                Error::Compilation(format!(
                    "missing identity assignment for '{}'",
                    &self.sender
                ))
            })?;

        Ok(SyncKernel::Nullary(Box::new(move |sess| {
            let v: Value = sess
                .networking
                .receive(&sender_id, &rendezvous_key, &sess.sid)?;
            if expected_ty != Ty::Unknown {
                check_type(&v, expected_ty)?;
            }
            Ok(v)
        })))
    }
}

impl Compile<AsyncKernel> for ReceiveOp {
    fn compile(&self, ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.sig.ret();
        let rendezvous_key = Arc::new(self.rendezvous_key.clone());
        let sender_id = Arc::new(ctx.role_assignment.get(&self.sender).cloned().ok_or_else(
            || {
                Error::Compilation(format!(
                    "missing identity assignment for '{}'",
                    &self.sender
                ))
            },
        )?);

        Ok(AsyncKernel::Nullary(Box::new(move |sess, sender| {
            let sess = Arc::clone(sess);
            let rendezvous_key = Arc::clone(&rendezvous_key);
            let sender_id = Arc::clone(&sender_id);

            tokio::spawn(async move {
                let v: Value = sess
                    .networking
                    .receive(&sender_id, &rendezvous_key, &sess.sid)
                    .await?;
                if expected_ty != Ty::Unknown {
                    check_type(&v, expected_ty)?;
                }
                map_send_result(sender.send(v))
            })
        })))
    }
}

impl Compile<SyncKernel> for IdentityOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        let expected_ty = self.sig.ret();

        Ok(SyncKernel::Unary(Box::new(move |_sess, v| {
            check_type(&v, expected_ty)?;
            Ok(v)
        })))
    }
}

impl Compile<AsyncKernel> for IdentityOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.sig.ret();

        Ok(AsyncKernel::Unary(Box::new(move |_sess, v, sender| {
            tokio::spawn(async move {
                let v: Value = v.await.map_err(map_receive_error)?;
                check_type(&v, expected_ty)?;
                map_send_result(sender.send(v))
            })
        })))
    }
}

modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> String, InputOp);
// TODO: (lvorona) all the other types. Perhaps a macros?

kernel! {
    InputOp, [
        (HostPlacement, () -> String => attributes[arg_name] Self::kernel_string),
    ]
}

impl InputOp {
    fn kernel_string<S: RuntimeSession>(
        _sess: &S,
        _plc: &HostPlacement,
        arg_name: String,
    ) -> String {
        // TODO: (lvorona) should we be placing the constant on the placement here?
        // TODO: (lvorona) this is only good for the symbolic session
        format!("Input value for {}", arg_name)
    }
}

impl Compile<SyncKernel> for InputOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        let arg_name = self.arg_name.clone();
        let expected_ty = self.sig.ret();

        Ok(SyncKernel::Nullary(Box::new(move |sess| {
            let arg = sess
                .arguments
                .get(&arg_name)
                .cloned()
                .ok_or_else(|| Error::MissingArgument(arg_name.clone()))?;
            check_type(&arg, expected_ty)?;
            Ok(arg)
        })))
    }
}

impl Compile<AsyncKernel> for InputOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.sig.ret();
        let arg_name = Arc::new(self.arg_name.clone());

        Ok(AsyncKernel::Nullary(Box::new(move |sess, sender| {
            let sess = Arc::clone(sess);
            let arg_name = Arc::clone(&arg_name);

            tokio::spawn(async move {
                let arg = sess
                    .arguments
                    .get(arg_name.as_ref())
                    .cloned()
                    .ok_or_else(|| Error::MissingArgument(arg_name.as_ref().clone()))?;
                check_type(&arg, expected_ty)?;
                map_send_result(sender.send(arg))
            })
        })))
    }
}

modelled!(PlacementOutput::output, HostPlacement, (Unit) -> Unit, OutputOp);

kernel! {
    OutputOp, [
        (HostPlacement, (Unit) -> Unit => Self::kernel),
    ]
}

impl OutputOp {
    fn kernel<S: RuntimeSession>(_sess: &S, _plc: &HostPlacement, _x: Unit) -> Unit {
        // TODO: (lvorona)
        unimplemented!()
    }
}

impl Compile<SyncKernel> for OutputOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        Ok(SyncKernel::Unary(Box::new(move |_sess, x0| Ok(x0))))
    }
}

impl Compile<AsyncKernel> for OutputOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        Ok(AsyncKernel::Unary(Box::new(move |_sess, x0, sender| {
            tokio::spawn(async move {
                let val = x0.await.map_err(map_receive_error)?;
                map_send_result(sender.send(val))
            })
        })))
    }
}

modelled!(PlacementSave::save, HostPlacement, (String, Float64Tensor) -> Unit, SaveOp);
// TODO: (lvorona) all the other types. Perhaps a macros?

kernel! {
    SaveOp, [
        (HostPlacement, (String, Float64Tensor) -> Unit => Self::kernel_float64tensor),
    ]
}

impl SaveOp {
    fn kernel_float64tensor<S: RuntimeSession>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: String,
        _x: Float64Tensor,
    ) -> Unit {
        // TODO: (lvorona)
        unimplemented!()
    }
}

impl Compile<SyncKernel> for SaveOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        let expected_ty = self.sig.arg(1)?;

        Ok(SyncKernel::Binary(Box::new(move |sess, key, val| {
            let key = String::try_from(key)?;
            check_type(&val, expected_ty)?;
            sess.storage.save(&key, &sess.sid, &val)?;
            Ok(Value::Unit(Unit(HostPlacement {
                owner: "TODO".into(),
            })))
        })))
    }
}

impl Compile<AsyncKernel> for SaveOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.sig.arg(1)?;

        Ok(AsyncKernel::Binary(Box::new(
            move |sess, key, val, sender| {
                let sess = Arc::clone(sess);

                tokio::spawn(async move {
                    let key = String::try_from(key.await.map_err(map_receive_error)?)?;
                    let val = val.await.map_err(map_receive_error)?;
                    check_type(&val, expected_ty)?;
                    sess.storage.save(&key, &sess.sid, &val).await?;
                    map_send_result(sender.send(Value::Unit(Unit(HostPlacement {
                        owner: "TODO".into(),
                    }))))
                })
            },
        )))
    }
}

modelled!(PlacementLoad::load, HostPlacement, (String, String) -> Float64Tensor, LoadOp);
// TODO: (lvorona) all the other types. Perhaps a macros?

kernel! {
    LoadOp, [
        (HostPlacement, (String, String) -> Float64Tensor => Self::kernel_float64tensor),
    ]
}

impl LoadOp {
    fn kernel_float64tensor<S: RuntimeSession>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: String,
        _query: String,
    ) -> Float64Tensor {
        // TODO: (lvorona)
        unimplemented!()
    }
}

impl Compile<SyncKernel> for LoadOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        let expected_ty = self.sig.ret();

        Ok(SyncKernel::Binary(Box::new(move |sess, key, query| {
            let key = String::try_from(key)?;
            let _query = String::try_from(query)?;
            let val = sess
                .storage
                .load(&key, &sess.sid, Some(expected_ty), &_query)?;
            check_type(&val, expected_ty)?;
            Ok(val)
        })))
    }
}

impl Compile<AsyncKernel> for LoadOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.sig.ret();

        Ok(AsyncKernel::Binary(Box::new(
            move |sess, key, query, sender| {
                let sess = Arc::clone(sess);

                tokio::spawn(async move {
                    let key = String::try_from(key.await.map_err(map_receive_error)?)?;
                    let _query = String::try_from(query.await.map_err(map_receive_error)?)?;
                    let val = sess
                        .storage
                        .load(&key, &sess.sid, Some(expected_ty), &_query)
                        .await?;
                    check_type(&val, expected_ty)?;
                    map_send_result(sender.send(val))
                })
            },
        )))
    }
}
