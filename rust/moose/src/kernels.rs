use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::{
    map_receive_error, map_send_result, AsyncKernel, CompilationContext, Compile, Kernel,
    SyncKernel,
};
use crate::fixedpoint::{Fixed128Tensor, Fixed64Tensor};
use crate::host::{
    AbstractHostRingTensor, HostBitTensor, HostFloat32Tensor, HostFloat64Tensor, HostInt16Tensor,
    HostInt32Tensor, HostInt64Tensor, HostInt8Tensor, HostRing128Tensor, HostRing64Tensor,
    HostShape, HostTensor, HostUint16Tensor, HostUint32Tensor, HostUint64Tensor, HostUint8Tensor,
};
use crate::prim::{PrfKey, RawNonce, RawPrfKey, RawSeed, Seed};
use crate::replicated::ReplicatedSetup;
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

impl SyncSession {
    // Currently used in a test
    pub fn new(replicated_keys: HashMap<ReplicatedPlacement, ReplicatedSetup>) -> Self {
        SyncSession {
            session_id: "abcde".into(),
            replicated_keys,
        }
    }
}
impl Default for SyncSession {
    fn default() -> Self {
        SyncSession {
            session_id: "abcde".into(), // TODO sync session is only used in tests currently, but it should get the session if from then env still.
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
            Operator::RingFill(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::PrimPrfKeyGen(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::BitSample(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::BitXor(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::BitAnd(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::BitExtract(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingSample(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingAdd(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingSub(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingMul(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingDot(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingNeg(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingShl(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingShr(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RingSum(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepFill(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepSetup(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepShare(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepReveal(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepAdd(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepSub(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepMul(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepDot(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepTruncPr(op) => DispatchKernel::compile(&op, plc)(self, operands),
            // Operator::RepAbs(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::RepMsb(op) => DispatchKernel::compile(&op, plc)(self, operands),
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
            Operator::HostOnes(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::Input(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::Output(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::Load(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::Save(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::StdAtLeast2D(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostMean(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostSum(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointRingEncode(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointRingDecode(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointRingMean(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointEncode(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointAdd(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointSub(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointMul(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointDot(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointTruncPr(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointSum(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::FixedpointMean(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostSlice(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostAdd(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostSub(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostMul(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostDiv(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostDot(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostExpandDims(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostConcatenate(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostTranspose(op) => DispatchKernel::compile(&op, plc)(self, operands),
            Operator::HostInverse(op) => DispatchKernel::compile(&op, plc)(self, operands),
            op => unimplemented!("SyncSession implementation is missing for {:?}", op), // TODO Remove the catch-all case once all the Ops have kernels.
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
    type Value = (); // TODO AsyncExecutor for the new framework is not ready yet
    fn execute(&self, _op: Operator, _plc: &Placement, _operands: Vec<Self::Value>) -> Self::Value {
        // TODO AsyncExecutor for the new framework is not ready yet
        unimplemented!()
    }

    type ReplicatedSetup = (); // TODO AsyncExecutor for the new framework is not ready yet
    fn replicated_setup(&self, _plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup {
        // TODO AsyncExecutor for the new framework is not ready yet
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

pub trait PlacementDiv<S: Session, T, U, O> {
    fn div(&self, sess: &S, x: &T, y: &U) -> O;
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

pub trait PlacementBitExtract<S: Session, T, O> {
    fn bit_extract(&self, sess: &S, bit_idx: usize, x: &T) -> O;
}

pub trait PlacementMulSetup<S: Session, SetupT, T, U, O> {
    fn mul_setup(&self, sess: &S, setup: &SetupT, x: &T, y: &U) -> O;
}

pub trait PlacementDotSetup<S: Session, SetupT, T, U, O> {
    fn dot_setup(&self, sess: &S, setup: &SetupT, x: &T, y: &U) -> O;
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

modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostFloat64Tensor, HostOnesOp);

kernel! {
    HostOnesOp, [
        (HostPlacement, (HostShape) -> HostFloat64Tensor => Self::kernel),
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
    fn trunc_pr(&self, sess: &S, amount: u32, x: &T) -> O;
}

pub trait PlacementTruncPrProvider<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: usize, provider: &HostPlacement, x: &T) -> O;
}

pub trait PlacementAbs<S: Session, SetupT, T, O> {
    fn abs(&self, sess: &S, setup: &SetupT, x: &T) -> O;
}

pub trait PlacementMsb<S: Session, SetupT, T, O> {
    fn msb(&self, sess: &S, setup: &SetupT, x: &T) -> O;
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

pub trait PlacementAtLeast2D<S: Session, T, O> {
    fn at_least_2d(&self, sess: &S, to_column_vector: bool, x: &T) -> O;
}

pub trait PlacementFixedpointRingEncode<S: Session, T, O> {
    fn fixedpoint_ring_encode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
}

pub trait PlacementFixedpointRingDecode<S: Session, T, O> {
    fn fixedpoint_ring_decode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
}

pub trait PlacementFixedpointEncode<S: Session, T, O> {
    fn fixedpoint_encode(&self, sess: &S, precision: u32, x: &T) -> O;
}

pub trait PlacementFixedpointDecode<S: Session, T, O> {
    fn fixedpoint_decode(&self, sess: &S, precision: u32, x: &T) -> O;
}

pub trait PlacementStdMean<S: Session, T, O> {
    fn std_mean(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

pub trait PlacementExpandDims<S: Session, T, O> {
    fn expand_dims(&self, sess: &S, axis: Vec<u32>, x: &T) -> O;
}

pub trait PlacementConcatenate<S: Session, T1, T2, O> {
    fn concatenate(&self, sess: &S, axis: u32, x: &T1, y: &T2) -> O;
}

pub trait PlacementTranspose<S: Session, T, O> {
    fn transpose(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementInverse<S: Session, T, O> {
    fn inverse(&self, sess: &S, x: &T) -> O;
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
            HostAdd(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostSub(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostMul(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostDiv(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostDot(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostMean(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostOnes(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostConcatenate(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostExpandDims(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostReshape(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdAtLeast2D(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostSlice(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostSum(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostTranspose(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostInverse(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingAdd(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSub(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingMul(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingDot(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSum(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSample(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingShl(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingShr(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingInject(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitExtract(op) => Compile::<SyncKernel>::compile(op, ctx),
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
            HostAdd(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostSub(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostMul(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostDiv(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostDot(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostMean(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostOnes(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostConcatenate(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostExpandDims(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostReshape(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdAtLeast2D(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostSlice(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostSum(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostTranspose(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostInverse(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingNeg(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingAdd(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSub(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingMul(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingDot(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSum(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSample(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingShl(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingShr(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingInject(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitExtract(op) => Compile::<AsyncKernel>::compile(op, ctx),
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

macro_rules! host_unary_kernel {
    ($op:ty, $k:expr) => {
        impl Compile<Kernel> for $op {
            fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
                match self.sig {
                    signature![(Ty::HostFloat32Tensor) -> _] => {
                        function_kernel!(HostFloat32Tensor, $k)
                    }
                    signature![(Ty::HostFloat64Tensor) -> _] => {
                        function_kernel!(HostFloat64Tensor, $k)
                    }
                    signature![(Ty::HostInt32Tensor) -> _] => {
                        function_kernel!(HostInt32Tensor, $k)
                    }
                    signature![(Ty::HostInt64Tensor) -> _] => {
                        function_kernel!(HostInt64Tensor, $k)
                    }
                    signature![(Ty::HostUint32Tensor) -> _] => {
                        function_kernel!(HostUint32Tensor, $k)
                    }
                    signature![(Ty::HostUint64Tensor) -> _] => {
                        function_kernel!(HostUint64Tensor, $k)
                    }
                    _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
                }
            }
        }
    };
}

macro_rules! host_binary_kernel {
    ($op:ident, $t:ident::$f:ident, $k:expr) => {
        impl Compile<Kernel> for $op {
            fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
                match self.sig {
                    signature![(Ty::HostFloat32Tensor, Ty::HostFloat32Tensor) -> _] => {
                        function_kernel!(HostFloat32Tensor, HostFloat32Tensor, $k)
                    }
                    signature![(Ty::HostFloat64Tensor, Ty::HostFloat64Tensor) -> _] => {
                        function_kernel!(HostFloat64Tensor, HostFloat64Tensor, $k)
                    }
                    signature![(Ty::HostInt32Tensor, Ty::HostInt32Tensor) -> _] => {
                        function_kernel!(HostInt32Tensor, HostInt32Tensor, $k)
                    }
                    signature![(Ty::HostInt64Tensor, Ty::HostInt64Tensor) -> _] => {
                        function_kernel!(HostInt64Tensor, HostInt64Tensor, $k)
                    }
                    signature![(Ty::HostUint32Tensor, Ty::HostUint32Tensor) -> _] => {
                        function_kernel!(HostUint32Tensor, HostUint32Tensor, $k)
                    }
                    signature![(Ty::HostUint64Tensor, Ty::HostUint64Tensor) -> _] => {
                        function_kernel!(HostUint64Tensor, HostUint64Tensor, $k)
                    }
                    _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
                }
            }
        }

        modelled!($t::$f, HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor, $op);
        modelled!($t::$f, HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor, $op);
        modelled!($t::$f, HostPlacement, (HostInt32Tensor, HostInt32Tensor) -> HostInt32Tensor, $op);
        modelled!($t::$f, HostPlacement, (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor, $op);
        modelled!($t::$f, HostPlacement, (HostUint32Tensor, HostUint32Tensor) -> HostUint32Tensor, $op);
        modelled!($t::$f, HostPlacement, (HostUint64Tensor, HostUint64Tensor) -> HostUint64Tensor, $op);

        kernel! {
            $op, [
                (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor => Self::kernel),
                (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => Self::kernel),
                (HostPlacement, (HostInt32Tensor, HostInt32Tensor) -> HostInt32Tensor => Self::kernel),
                (HostPlacement, (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor => Self::kernel),
                (HostPlacement, (HostUint32Tensor, HostUint32Tensor) -> HostUint32Tensor => Self::kernel),
                (HostPlacement, (HostUint64Tensor, HostUint64Tensor) -> HostUint64Tensor => Self::kernel),
            ]
        }
    };
}

host_binary_kernel!(HostAddOp, PlacementAdd::add, |x, y| x + y);
host_binary_kernel!(HostSubOp, PlacementSub::sub, |x, y| x - y);
host_binary_kernel!(HostMulOp, PlacementMul::mul, |x, y| x * y);
host_binary_kernel!(HostDivOp, PlacementDiv::div, |x, y| x / y);
host_binary_kernel!(HostDotOp, PlacementDot::dot, |x, y| x.dot(y));
host_unary_kernel!(HostTransposeOp, |x| x.transpose());

modelled!(PlacementTranspose::transpose, HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor, HostTransposeOp);

kernel! {
    HostTransposeOp, [
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => Self::kernel),
    ]
}

modelled!(PlacementInverse::inverse, HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor, HostInverseOp);

kernel! {
    HostInverseOp, [
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => Self::kernel),
    ]
}

impl Compile<Kernel> for HostInverseOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        // Using a fake owner for the old kernel
        match self.sig {
            signature![(_) -> Ty::HostFloat32Tensor] => {
                closure_kernel!(HostFloat32Tensor, |x| x.inv())
            }
            signature![(_) -> Ty::HostFloat64Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.inv())
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementStdMean::std_mean, HostPlacement, attributes[axis: Option<u32>] (HostFloat64Tensor) -> HostFloat64Tensor, HostMeanOp);

kernel! {
    HostMeanOp, [
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => attributes[axis] Self::kernel),
    ]
}

impl Compile<Kernel> for HostMeanOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|x| x as usize);
        match self.sig {
            signature![(_) -> Ty::HostFloat32Tensor] => {
                closure_kernel!(HostFloat32Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::HostFloat64Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::HostInt32Tensor] => {
                closure_kernel!(HostInt32Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::HostInt64Tensor] => {
                closure_kernel!(HostInt64Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::HostUint32Tensor] => {
                closure_kernel!(HostUint32Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::HostUint64Tensor] => {
                closure_kernel!(HostUint64Tensor, |x| x.mean(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for HostOnesOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(_) -> Ty::HostFloat32Tensor] => {
                function_kernel!(HostShape, |shape| HostFloat32Tensor::ones(shape))
            }
            signature![(_) -> Ty::HostFloat64Tensor] => {
                function_kernel!(HostShape, |shape| HostFloat64Tensor::ones(shape))
            }
            signature![(_) -> Ty::HostInt32Tensor] => {
                function_kernel!(HostShape, |shape| HostInt32Tensor::ones(shape))
            }
            signature![(_) -> Ty::HostInt64Tensor] => {
                function_kernel!(HostShape, |shape| HostInt64Tensor::ones(shape))
            }
            signature![(_) -> Ty::HostUint32Tensor] => {
                function_kernel!(HostShape, |shape| HostUint32Tensor::ones(shape))
            }
            signature![(_) -> Ty::HostUint64Tensor] => {
                function_kernel!(HostShape, |shape| HostUint64Tensor::ones(shape))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor, HostConcatenateOp);

kernel! {
    HostConcatenateOp, [
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor => attributes[axis] Self::kernel),
    ]
}

impl Compile<Kernel> for HostConcatenateOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        use crate::host::concatenate;
        let axis = self.axis as usize;
        match self.sig {
            signature![(_, _) -> Ty::HostFloat32Tensor] => {
                closure_kernel!(vec[HostFloat32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::HostFloat64Tensor] => {
                closure_kernel!(vec[HostFloat64Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::HostInt32Tensor] => {
                closure_kernel!(vec[HostInt32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::HostInt64Tensor] => {
                closure_kernel!(vec[HostInt64Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::HostUint32Tensor] => {
                closure_kernel!(vec[HostUint32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::HostUint64Tensor] => {
                closure_kernel!(vec[HostUint64Tensor], |xs| concatenate(axis, &xs))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementExpandDims::expand_dims, HostPlacement, attributes[axis: Vec<u32>] (HostFloat64Tensor) -> HostFloat64Tensor, HostExpandDimsOp);

kernel! {
    HostExpandDimsOp, [
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => attributes[axis] Self::kernel),
    ]
}

impl Compile<Kernel> for HostExpandDimsOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis: Vec<usize> = self.axis.iter().map(|a| *a as usize).collect();
        match self.sig {
            signature![(_) -> Ty::HostFloat32Tensor] => {
                closure_kernel!(HostFloat32Tensor, |x| x.expand_dims(axis.clone()))
            }
            signature![(_) -> Ty::HostFloat64Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.expand_dims(axis.clone()))
            }
            signature![(_) -> Ty::HostInt32Tensor] => {
                closure_kernel!(HostInt32Tensor, |x| x.expand_dims(axis.clone()))
            }
            signature![(_) -> Ty::HostInt64Tensor] => {
                closure_kernel!(HostInt64Tensor, |x| x.expand_dims(axis.clone()))
            }
            signature![(_) -> Ty::HostUint32Tensor] => {
                closure_kernel!(HostUint32Tensor, |x| x.expand_dims(axis.clone()))
            }
            signature![(_) -> Ty::HostUint64Tensor] => {
                closure_kernel!(HostUint64Tensor, |x| x.expand_dims(axis.clone()))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for HostReshapeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(_, _) -> Ty::HostFloat32Tensor] => {
                function_kernel!(HostFloat32Tensor, HostShape, |x, newshape| x
                    .reshape(newshape))
            }
            signature![(_, _) -> Ty::HostFloat64Tensor] => {
                function_kernel!(HostFloat64Tensor, HostShape, |x, newshape| x
                    .reshape(newshape))
            }
            signature![(_, _) -> Ty::HostInt32Tensor] => {
                function_kernel!(HostInt32Tensor, HostShape, |x, newshape| x
                    .reshape(newshape))
            }
            signature![(_, _) -> Ty::HostInt64Tensor] => {
                function_kernel!(HostInt64Tensor, HostShape, |x, newshape| x
                    .reshape(newshape))
            }
            signature![(_, _) -> Ty::HostUint32Tensor] => {
                function_kernel!(HostUint32Tensor, HostShape, |x, newshape| x
                    .reshape(newshape))
            }
            signature![(_, _) -> Ty::HostUint64Tensor] => {
                function_kernel!(HostUint64Tensor, HostShape, |x, newshape| x
                    .reshape(newshape))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementAtLeast2D::at_least_2d, HostPlacement, attributes[to_column_vector: bool] (HostFloat32Tensor) -> HostFloat32Tensor, StdAtLeast2DOp);
modelled!(PlacementAtLeast2D::at_least_2d, HostPlacement, attributes[to_column_vector: bool] (HostFloat64Tensor) -> HostFloat64Tensor, StdAtLeast2DOp);

kernel! {
    StdAtLeast2DOp, [
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => attributes[to_column_vector] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => attributes[to_column_vector] Self::kernel),
    ]
}

impl StdAtLeast2DOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _to_column_vector: bool,
        _x: HostTensor<T>,
    ) -> HostTensor<T> {
        unimplemented!()
    }
}

impl Compile<Kernel> for StdAtLeast2DOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let tcv = self.to_column_vector;
        match self.sig {
            signature![(_) -> Ty::HostFloat32Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::HostFloat64Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::HostInt32Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::HostInt64Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::HostUint32Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::HostUint64Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.atleast_2d(tcv))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for HostSliceOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let start = self.start as usize;
        let end = self.end as usize;
        match self.sig {
            signature![(_) -> Ty::HostShape] => {
                closure_kernel!(HostShape, |x| HostShape(x.0.slice(start, end), x.1))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (HostFloat64Tensor) -> HostFloat64Tensor, HostSumOp);

kernel! {
    HostSumOp, [
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => attributes[axis] Self::kernel),
    ]
}

impl Compile<Kernel> for HostSumOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.sig {
            signature![(_) -> Ty::HostFloat32Tensor] => {
                closure_kernel!(HostFloat32Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::HostFloat64Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::HostInt32Tensor] => {
                closure_kernel!(HostInt32Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::HostInt64Tensor] => {
                closure_kernel!(HostInt64Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::HostUint32Tensor] => {
                closure_kernel!(HostUint32Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::HostUint64Tensor] => {
                closure_kernel!(HostUint64Tensor, |x| x.sum(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

// This impl is only used by the old kernels, which are not aware of the placements. See PrimDeriveSeedOp::kernel for the new code
#[cfg(not(feature = "symbolic"))]
impl Compile<Kernel> for PrimDeriveSeedOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let nonce = self.sync_key.clone();
        closure_kernel!(PrfKey, |key| Seed(
            RawSeed::from_prf(&key.0, &nonce),
            HostPlacement {
                owner: "TODO".into() // Fake owner for the older kernels.
            }
        ))
    }
}

// This impl is only used by the old kernels, which are not aware of the placements. See PrimPrfKeyGenOp::kernel for the new code
#[cfg(not(feature = "symbolic"))]
impl Compile<Kernel> for PrimPrfKeyGenOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(|| PrfKey(
            RawPrfKey::generate(),
            HostPlacement {
                owner: "TODO".into() // Fake owner for the older kernels.
            }
        ))
    }
}

impl Compile<Kernel> for RingAddOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostRing64Tensor, Ty::HostRing64Tensor) -> _] => {
                function_kernel!(HostRing64Tensor, HostRing64Tensor, |x, y| x + y)
            }
            signature![(Ty::HostRing128Tensor, Ty::HostRing128Tensor) -> _] => {
                function_kernel!(HostRing128Tensor, HostRing128Tensor, |x, y| x + y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSubOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostRing64Tensor, Ty::HostRing64Tensor) -> _] => {
                function_kernel!(HostRing64Tensor, HostRing64Tensor, |x, y| x - y)
            }
            signature![(Ty::HostRing128Tensor, Ty::HostRing128Tensor) -> _] => {
                function_kernel!(HostRing128Tensor, HostRing128Tensor, |x, y| x - y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingMulOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostRing64Tensor, Ty::HostRing64Tensor) -> _] => {
                function_kernel!(HostRing64Tensor, HostRing64Tensor, |x, y| x * y)
            }
            signature![(Ty::HostRing128Tensor, Ty::HostRing128Tensor) -> _] => {
                function_kernel!(HostRing128Tensor, HostRing128Tensor, |x, y| x * y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingDotOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostRing64Tensor, Ty::HostRing64Tensor) -> _] => {
                function_kernel!(HostRing64Tensor, HostRing64Tensor, |x, y| x.dot(y))
            }
            signature![(Ty::HostRing128Tensor, Ty::HostRing128Tensor) -> _] => {
                function_kernel!(HostRing128Tensor, HostRing128Tensor, |x, y| x.dot(y))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSumOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.sig {
            signature![(_) -> Ty::HostRing64Tensor] => {
                closure_kernel!(HostRing64Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::HostRing128Tensor] => {
                closure_kernel!(HostRing128Tensor, |x| x.sum(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for ShapeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostFloat32Tensor) -> Ty::HostShape] => {
                function_kernel!(HostFloat32Tensor, |x| x.shape())
            }
            signature![(Ty::HostFloat64Tensor) -> Ty::HostShape] => {
                function_kernel!(HostFloat64Tensor, |x| x.shape())
            }
            signature![(Ty::HostRing64Tensor) -> Ty::HostShape] => {
                function_kernel!(HostRing64Tensor, |x| x.shape())
            }
            signature![(Ty::HostRing128Tensor) -> Ty::HostShape] => {
                function_kernel!(HostRing128Tensor, |x| x.shape())
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for BitFillOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (&self.sig, self.value.clone()) {
            (signature![(_) -> Ty::HostBitTensor], Constant::Ring64(value)) => {
                closure_kernel!(HostShape, |shape| {
                    assert!(value == 0 || value == 1);
                    HostBitTensor::fill(&shape.0, value as u8)
                })
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingFillOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (&self.sig, self.value.clone()) {
            (signature![(_) -> Ty::HostRing64Tensor], Constant::Ring64(value)) => {
                closure_kernel!(HostShape, |shape| HostRing64Tensor::fill(&shape.0, value))
            }
            (signature![(_) -> Ty::HostRing128Tensor], Constant::Ring64(value)) => {
                closure_kernel!(HostShape, |shape| HostRing128Tensor::fill(
                    &shape.0,
                    value as u128
                ))
            }
            (signature![(_) -> Ty::HostRing128Tensor], Constant::Ring128(value)) => {
                closure_kernel!(HostShape, |shape| HostRing128Tensor::fill(&shape.0, value))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSampleOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (&self.sig, self.max_value) {
            (signature![(_, _) -> Ty::HostRing64Tensor], None) => {
                function_kernel!(HostShape, Seed, |shape, seed| {
                    HostRing64Tensor::sample_uniform(&shape.0, &seed.0)
                })
            }
            (signature!((_, _) -> Ty::HostRing64Tensor), Some(max_value)) if max_value == 1 => {
                function_kernel!(
                    HostShape,
                    Seed,
                    |shape, seed| HostRing64Tensor::sample_bits(&shape.0, &seed.0)
                )
            }
            (signature![(_, _) -> Ty::HostRing128Tensor], None) => {
                function_kernel!(HostShape, Seed, |shape, seed| {
                    HostRing128Tensor::sample_uniform(&shape.0, &seed.0)
                })
            }
            (signature![(_, _) -> Ty::HostRing128Tensor], Some(max_value)) if max_value == 1 => {
                function_kernel!(HostShape, Seed, |shape, seed| {
                    HostRing128Tensor::sample_bits(&shape.0, &seed.0)
                })
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingNegOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(_) -> Ty::HostRing64Tensor] => {
                closure_kernel!(HostRing64Tensor, |x| AbstractHostRingTensor(-x.0, x.1))
            }
            signature![(_) -> Ty::HostRing128Tensor] => {
                closure_kernel!(HostRing128Tensor, |x| AbstractHostRingTensor(-x.0, x.1))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingShlOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let amount = self.amount;
        match self.sig {
            signature![(_) -> Ty::HostRing64Tensor] => {
                closure_kernel!(HostRing64Tensor, |x| x << amount)
            }
            signature![(_) -> Ty::HostRing128Tensor] => {
                closure_kernel!(HostRing128Tensor, |x| x << amount)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingShrOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let amount = self.amount;
        match self.sig {
            signature![(_) -> Ty::HostRing64Tensor] => {
                closure_kernel!(HostRing64Tensor, |x| x >> amount)
            }
            signature![(_) -> Ty::HostRing128Tensor] => {
                closure_kernel!(HostRing128Tensor, |x| x >> amount)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingInjectOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let bit_idx = self.bit_idx;
        match self.sig {
            signature![(_) -> Ty::HostRing64Tensor] => {
                closure_kernel!(HostBitTensor, |x| HostRing64Tensor::from(x) << bit_idx)
            }
            signature![(_) -> Ty::HostRing128Tensor] => {
                closure_kernel!(HostBitTensor, |x| HostRing128Tensor::from(x) << bit_idx)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for BitExtractOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let bit_idx = self.bit_idx;
        match self.sig {
            signature![(Ty::HostRing64Tensor) -> _] => {
                closure_kernel!(HostRing64Tensor, |x| x.bit_extract(bit_idx))
            }
            signature![(Ty::HostRing128Tensor) -> _] => {
                closure_kernel!(HostRing128Tensor, |x| x.bit_extract(bit_idx))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for BitSampleOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(
            HostShape,
            Seed,
            |shape, seed| HostBitTensor::sample_uniform(&shape.0, &seed.0)
        )
    }
}

impl Compile<Kernel> for BitXorOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(HostBitTensor, HostBitTensor, |x, y| x ^ y)
    }
}

impl Compile<Kernel> for BitAndOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostRing64Tensor, Ty::HostRing64Tensor) -> _] => {
                closure_kernel!(HostRing64Tensor, HostRing64Tensor, |x, y| x & y)
            }
            signature![(Ty::HostRing128Tensor, Ty::HostRing128Tensor) -> _] => {
                closure_kernel!(HostRing128Tensor, HostRing128Tensor, |x, y| x & y)
            }
            signature![(Ty::HostBitTensor, Ty::HostBitTensor) -> _] => {
                closure_kernel!(HostBitTensor, HostBitTensor, |x, y| x & y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementFixedpointRingEncode::fixedpoint_ring_encode, HostPlacement, attributes[scaling_base: u64, scaling_exp: u32] (HostFloat64Tensor) -> HostRing128Tensor, FixedpointRingEncodeOp);
modelled!(PlacementFixedpointRingEncode::fixedpoint_ring_encode, HostPlacement, attributes[scaling_base: u64, scaling_exp: u32] (HostFloat32Tensor) -> HostRing64Tensor, FixedpointRingEncodeOp);

kernel! {
    FixedpointRingEncodeOp, [
        (HostPlacement, (HostFloat64Tensor) -> HostRing128Tensor => attributes[scaling_base, scaling_exp] Self::kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostRing64Tensor => attributes[scaling_base, scaling_exp] Self::kernel),
    ]
}

impl FixedpointRingEncodeOp {
    fn kernel<S: RuntimeSession, ST, TT>(
        _sess: &S,
        _plc: &HostPlacement,
        _scaling_base: u64,
        _scaling_exp: u32,
        _x: HostTensor<ST>,
    ) -> AbstractHostRingTensor<TT> {
        unimplemented!()
    }
}

impl Compile<Kernel> for FixedpointRingEncodeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        use crate::fixedpoint::Convert;
        match self.sig {
            signature![(Ty::HostFloat64Tensor) -> Ty::HostRing64Tensor] => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(HostFloat64Tensor, |x| HostRing64Tensor::encode(
                    &x,
                    scaling_factor
                ))
            }
            signature![(Ty::HostFloat64Tensor) -> Ty::HostRing128Tensor] => {
                let scaling_factor = u128::pow(self.scaling_base as u128, self.scaling_exp);
                closure_kernel!(HostFloat64Tensor, |x| HostRing128Tensor::encode(
                    &x,
                    scaling_factor
                ))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

modelled!(PlacementFixedpointRingDecode::fixedpoint_ring_decode, HostPlacement, attributes[scaling_base: u64, scaling_exp: u32] (HostRing128Tensor) -> HostFloat64Tensor, FixedpointRingDecodeOp);
modelled!(PlacementFixedpointRingDecode::fixedpoint_ring_decode, HostPlacement, attributes[scaling_base: u64, scaling_exp: u32] (HostRing64Tensor) -> HostFloat32Tensor, FixedpointRingDecodeOp);

kernel! {
    FixedpointRingDecodeOp, [
        (HostPlacement, (HostRing128Tensor) -> HostFloat64Tensor => attributes[scaling_base, scaling_exp] Self::kernel),
        (HostPlacement, (HostRing64Tensor) -> HostFloat32Tensor => attributes[scaling_base, scaling_exp] Self::kernel),
    ]
}

impl FixedpointRingDecodeOp {
    fn kernel<S: RuntimeSession, ST, TT>(
        _sess: &S,
        _plc: &HostPlacement,
        _scaling_base: u64,
        _scaling_exp: u32,
        _x: AbstractHostRingTensor<ST>,
    ) -> HostTensor<TT> {
        unimplemented!()
    }
}

impl Compile<Kernel> for FixedpointRingDecodeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        use crate::fixedpoint::Convert;
        match self.sig {
            signature![(Ty::HostRing64Tensor) -> _] => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(HostRing64Tensor, |x| HostRing64Tensor::decode(
                    &x,
                    scaling_factor
                ))
            }
            signature![(Ty::HostRing128Tensor) -> _] => {
                let scaling_factor = u128::pow(self.scaling_base as u128, self.scaling_exp);
                closure_kernel!(HostRing128Tensor, |x| HostRing128Tensor::decode(
                    &x,
                    scaling_factor
                ))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for FixedpointRingMeanOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.sig {
            signature![(_) -> Ty::HostRing64Tensor] => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(HostRing64Tensor, |x| HostRing64Tensor::ring_mean(
                    x,
                    axis,
                    scaling_factor
                ))
            }
            signature![(_) -> Ty::HostRing128Tensor] => {
                let scaling_factor = u128::pow(self.scaling_base as u128, self.scaling_exp);
                closure_kernel!(HostRing128Tensor, |x| HostRing128Tensor::ring_mean(
                    x,
                    axis,
                    scaling_factor
                ))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

// This impl is only used by the old kernels, which are not aware of the placements. See ConstantOp::kernel for the new code
#[cfg(not(feature = "symbolic"))]
impl Compile<Kernel> for ConstantOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let value = self.value.clone();
        Ok(Kernel::NullaryClosure(Arc::new(move || {
            Ok(value.place(&HostPlacement {
                owner: "TODO".into(), // Fake owner for the older kernels.
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

impl ConstantOp {
    fn kernel<S: RuntimeSession, T: Placed>(sess: &S, plc: &HostPlacement, value: T) -> T
    where
        HostPlacement: PlacementPlace<S, T>,
    {
        plc.place(sess, value)
    }
}

// This impl is only used by the old kernels, which are not aware of the placements.
#[cfg(not(feature = "symbolic"))]
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
                owner: "TODO".into(), // Fake owner for the older kernels.
            })))
        })))
    }
}

// This impl is only used by the old kernels, which are not aware of the placements.
#[cfg(not(feature = "symbolic"))]
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
                    owner: "TODO".into(), // Fake owner for the older kernels.
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

// Not all the variants from the `values![]` list can be received as an input (nothing replicated, for instance).
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> String, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Unit, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostShape, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Seed, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> PrfKey, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostBitTensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostRing64Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostRing128Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostFloat32Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostFloat64Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostInt8Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostInt16Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostInt32Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostInt64Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostUint8Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostUint16Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostUint32Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostUint64Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Fixed64Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Fixed128Tensor, InputOp);
kernel! {
    InputOp, [
        (HostPlacement, () -> String => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> Unit => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostShape => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> Seed => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> PrfKey => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostBitTensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostRing64Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostRing128Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostFloat32Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostFloat64Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt8Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt16Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt32Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt64Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint8Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint16Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint32Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint64Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> Fixed64Tensor => attributes[arg_name] Self::kernel),
        (HostPlacement, () -> Fixed128Tensor => attributes[arg_name] Self::kernel),

    ]
}

impl InputOp {
    fn kernel<S: RuntimeSession, O>(_sess: &S, _plc: &HostPlacement, _arg_name: String) -> O {
        unimplemented!() // TODO: Read the value from the environment for the Async and Sync sessions to work.
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

// Not all the variants from the `values![]` list can be saved as an output (nothing replicated, for instance).
modelled!(PlacementOutput::output, HostPlacement, (Unit) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostShape) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (Seed) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (PrfKey) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (String) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostBitTensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostRing64Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostRing128Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostFloat32Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostFloat64Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostInt8Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostInt16Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostInt32Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostInt64Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostUint8Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostUint16Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostUint32Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (HostUint64Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (Fixed64Tensor) -> Unit, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (Fixed128Tensor) -> Unit, OutputOp);

kernel! {
    OutputOp, [
        (HostPlacement, (Unit) -> Unit => Self::kernel),
        (HostPlacement, (HostShape) -> Unit => Self::kernel),
        (HostPlacement, (Seed) -> Unit => Self::kernel),
        (HostPlacement, (PrfKey) -> Unit => Self::kernel),
        (HostPlacement, (String) -> Unit => Self::kernel),
        (HostPlacement, (HostBitTensor) -> Unit => Self::kernel),
        (HostPlacement, (HostRing64Tensor) -> Unit => Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> Unit => Self::kernel),
        (HostPlacement, (HostFloat32Tensor) -> Unit => Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> Unit => Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> Unit => Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> Unit => Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> Unit => Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> Unit => Self::kernel),
        (HostPlacement, (HostUint8Tensor) -> Unit => Self::kernel),
        (HostPlacement, (HostUint16Tensor) -> Unit => Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> Unit => Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> Unit => Self::kernel),
        (HostPlacement, (Fixed64Tensor) -> Unit => Self::kernel),
        (HostPlacement, (Fixed128Tensor) -> Unit => Self::kernel),
    ]
}

impl OutputOp {
    fn kernel<S: RuntimeSession, O>(_sess: &S, _plc: &HostPlacement, _x: O) -> Unit {
        unimplemented!() // TODO: Save to the environment for the Sync and Async sessions to work.
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

// Not all the variants from the `values![]` list can be saved (nothing replicated, for instance).
modelled!(PlacementSave::save, HostPlacement, (String, Unit) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostShape) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, Seed) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, PrfKey) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, String) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostBitTensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostRing64Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostRing128Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostFloat32Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostFloat64Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostInt8Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostInt16Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostInt32Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostInt64Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostUint8Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostUint16Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostUint32Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, HostUint64Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, Fixed64Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (String, Fixed128Tensor) -> Unit, SaveOp);

kernel! {
    SaveOp, [
        (HostPlacement, (String, Unit) -> Unit => Self::kernel),
        (HostPlacement, (String, HostShape) -> Unit => Self::kernel),
        (HostPlacement, (String, Seed) -> Unit => Self::kernel),
        (HostPlacement, (String, PrfKey) -> Unit => Self::kernel),
        (HostPlacement, (String, String) -> Unit => Self::kernel),
        (HostPlacement, (String, HostBitTensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostRing64Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostRing128Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostFloat32Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostFloat64Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostInt8Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostInt16Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostInt32Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostInt64Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostUint8Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostUint16Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostUint32Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, HostUint64Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, Fixed64Tensor) -> Unit => Self::kernel),
        (HostPlacement, (String, Fixed128Tensor) -> Unit => Self::kernel),
    ]
}

impl SaveOp {
    fn kernel<S: RuntimeSession, O>(_sess: &S, _plc: &HostPlacement, _key: String, _x: O) -> Unit {
        unimplemented!() // TODO: Save the value into storage for the Async and Sync sessions to work.
    }
}

// This implementation is the old kernel.
#[cfg(not(feature = "symbolic"))]
impl Compile<SyncKernel> for SaveOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        let expected_ty = self.sig.arg(1)?;

        Ok(SyncKernel::Binary(Box::new(move |sess, key, val| {
            let key = String::try_from(key)?;
            check_type(&val, expected_ty)?;
            sess.storage.save(&key, &sess.sid, &val)?;
            Ok(Value::Unit(Unit(HostPlacement {
                owner: "TODO".into(), // Fake owner for the old kernel
            })))
        })))
    }
}

// This implementation is the old kernel.
#[cfg(not(feature = "symbolic"))]
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
                        owner: "TODO".into(), // Fake owner for the old kernel
                    }))))
                })
            },
        )))
    }
}

modelled!(PlacementLoad::load, HostPlacement, (String, String) -> Unit, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostShape, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> Seed, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> PrfKey, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> String, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostBitTensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostRing64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostRing128Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostFloat32Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostFloat64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostInt8Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostInt16Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostInt32Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostInt64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostUint8Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostUint16Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostUint32Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> HostUint64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> Fixed64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (String, String) -> Fixed128Tensor, LoadOp);

kernel! {
    LoadOp, [
        (HostPlacement, (String, String) -> Unit => Self::kernel),
        (HostPlacement, (String, String) -> HostShape => Self::kernel),
        (HostPlacement, (String, String) -> Seed => Self::kernel),
        (HostPlacement, (String, String) -> PrfKey => Self::kernel),
        (HostPlacement, (String, String) -> String => Self::kernel),
        (HostPlacement, (String, String) -> HostBitTensor => Self::kernel),
        (HostPlacement, (String, String) -> HostRing64Tensor => Self::kernel),
        (HostPlacement, (String, String) -> HostRing128Tensor => Self::kernel),
        (HostPlacement, (String, String) -> HostFloat32Tensor => Self::kernel),
        (HostPlacement, (String, String) -> HostFloat64Tensor => Self::kernel),
        (HostPlacement, (String, String) -> HostInt8Tensor => Self::kernel),
        (HostPlacement, (String, String) -> HostInt16Tensor => Self::kernel),
        (HostPlacement, (String, String) -> HostInt32Tensor => Self::kernel),
        (HostPlacement, (String, String) -> HostInt64Tensor => Self::kernel),
        (HostPlacement, (String, String) -> HostUint8Tensor => Self::kernel),
        (HostPlacement, (String, String) -> HostUint16Tensor => Self::kernel),
        (HostPlacement, (String, String) -> HostUint32Tensor => Self::kernel),
        (HostPlacement, (String, String) -> HostUint64Tensor => Self::kernel),
        (HostPlacement, (String, String) -> Fixed64Tensor => Self::kernel),
        (HostPlacement, (String, String) -> Fixed128Tensor => Self::kernel),
    ]
}

impl LoadOp {
    fn kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: String,
        _query: String,
    ) -> O {
        unimplemented!() // TODO: Implement loading from storage for the Async and Sync sessions to work.
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
