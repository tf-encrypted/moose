use crate::encrypted::{AesKey, AesTensor, Fixed128AesTensor};
use crate::error::{Error, Result};
use crate::execution::{
    map_receive_error, map_send_result, AsyncKernel, CompilationContext, Compile, Identity, Kernel,
    SyncKernel,
};
use crate::fixedpoint::{Convert, Fixed128Tensor, Fixed64Tensor};
use crate::floatingpoint::{Float32Tensor, Float64Tensor};
use crate::host::*;
use crate::prim::{PrfKey, RawPrfKey, RawSeed, Seed, SyncKey};
use crate::replicated::*;
use crate::{closure_kernel, function_kernel};
use crate::{computation::*, for_all_values};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;

/// General session trait determining basic properties for session objects.
pub trait Session {
    type Value;
    fn execute(
        &self,
        op: Operator,
        plc: &Placement,
        operands: Vec<Self::Value>,
    ) -> Result<Self::Value>;

    type ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> Arc<Self::ReplicatedSetup>;
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
    fn find_argument(&self, key: &str) -> Option<Value>;
    fn find_role_assignment(&self, role: &Role) -> Result<&Identity>;
}

/// Session object for synchronous/eager execution (in new framework).
pub struct SyncSession {
    session_id: SessionId,
    replicated_keys: std::sync::RwLock<HashMap<ReplicatedPlacement, Arc<ReplicatedSetup>>>,
    arguments: HashMap<String, Value>,
    role_assignments: HashMap<Role, Identity>,
}

impl Default for SyncSession {
    /// Default session should only be used in tests.
    ///
    /// Use new() for the real sessions instead.
    fn default() -> Self {
        SyncSession {
            session_id: SessionId::random(),
            replicated_keys: Default::default(),
            arguments: Default::default(),
            role_assignments: Default::default(),
        }
    }
}

impl SyncSession {
    pub fn new(
        sid: SessionId,
        arguments: HashMap<String, Value>,
        role_assignments: HashMap<Role, Identity>,
    ) -> Self {
        SyncSession {
            session_id: sid,
            replicated_keys: Default::default(),
            arguments,
            role_assignments,
        }
    }
}

impl Session for SyncSession {
    type Value = Value;

    fn execute(&self, op: Operator, plc: &Placement, operands: Vec<Value>) -> Result<Value> {
        use Operator::*;
        let kernel_output = match op {
            Shape(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitFill(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingFill(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            PrimPrfKeyGen(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitSample(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitSampleSeeded(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitXor(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitAnd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitNeg(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitExtract(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingSample(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingSampleSeeded(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingNeg(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingShl(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingShr(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingSum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingFixedpointMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingFixedpointEncode(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingFixedpointDecode(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingInject(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepFill(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepSetup(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepShare(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepReveal(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepAnd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepXor(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepNeg(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepTruncPr(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepMsb(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepAbs(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepToAdt(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepFixedpointMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepSum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AddN(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepShl(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepIndexAxis(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Index(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepDiag(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepSlice(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepBitDec(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepBitCompose(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepShlDim(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtShl(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtFill(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtReveal(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtToRep(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            PrimDeriveSeed(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AesDecrypt(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Constant(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostOnes(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Input(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Output(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Load(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Save(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostAtLeast2D(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSqrt(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointEncode(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointDecode(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointDiv(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointTruncPr(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointSum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSlice(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostDiag(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostShlDim(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostIndexAxis(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostDiv(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostExpandDims(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSqueeze(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostConcat(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Sign(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointDiv(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointAtLeast2D(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointOnes(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointConcat(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointExpandDims(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointTranspose(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointInverse(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointSum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostTranspose(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostInverse(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostBitDec(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Identity(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Cast(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Send(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Receive(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostReshape(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AtLeast2D(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Slice(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Ones(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            ExpandDims(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Concat(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Transpose(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Dot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Inverse(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Add(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Sub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Mul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Mean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Neg(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Sum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Div(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepEqual(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepIfElse(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Pow2(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Exp(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Sigmoid(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            LessThan(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            GreaterThan(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
        };
        Ok(kernel_output)
    }

    type ReplicatedSetup = ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> Arc<Self::ReplicatedSetup> {
        let mut replicated_keys = self.replicated_keys.write().unwrap();
        let setup = replicated_keys
            .entry(plc.clone())
            .or_insert_with(|| Arc::new(plc.gen_setup(self)));
        Arc::clone(setup)
    }
}

impl RuntimeSession for SyncSession {
    fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    fn find_argument(&self, key: &str) -> Option<Value> {
        self.arguments.get(key).cloned()
    }

    fn find_role_assignment(&self, role: &Role) -> Result<&Identity> {
        self.role_assignments
            .get(role)
            .ok_or_else(|| Error::Networking(format!("Missing role assignemnt for {}", role)))
    }
}

/// Session object for asynchronous execution (in new framework).
pub struct AsyncSession {
    session_id: SessionId,
    // replicated_keys: HashMap<ReplicatedPlacement, ReplicatedSetup>,
}

impl Session for AsyncSession {
    type Value = (); // TODO AsyncExecutor for the new framework is not ready yet
    fn execute(
        &self,
        _op: Operator,
        _plc: &Placement,
        _operands: Vec<Self::Value>,
    ) -> Result<Self::Value> {
        // TODO AsyncExecutor for the new framework is not ready yet
        unimplemented!()
    }

    type ReplicatedSetup = (); // TODO AsyncExecutor for the new framework is not ready yet
    fn replicated_setup(&self, _plc: &ReplicatedPlacement) -> Arc<Self::ReplicatedSetup> {
        // TODO AsyncExecutor for the new framework is not ready yet
        unimplemented!()
    }
}

impl RuntimeSession for AsyncSession {
    fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    fn find_argument(&self, _key: &str) -> Option<Value> {
        todo!("Please implement find_argument for the new AsyncSession")
        // self.arguments.get(key)
    }

    fn find_role_assignment(&self, _role: &Role) -> Result<&Identity> {
        Err(Error::Networking(
            "new AsyncSession networking is not implemented yet".to_string(),
        ))
    }
}

pub trait DispatchKernel<S: Session> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(
        &self,
        plc: &Placement,
    ) -> Result<Box<dyn Fn(&S, Vec<S::Value>) -> Result<S::Value>>>;
}

// TODO if rustc can't figure out how to optimize Box<dyn Fn...> for
// function kernels then we could consider returning an enum over
// fn.. and Box<dyn Fn...> in the traits below instead

pub trait NullaryKernel<S: Session, P, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P) -> Result<Y>>>;
}

pub trait UnaryKernel<S: Session, P, X0, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, X0) -> Result<Y>>>;
}

pub trait BinaryKernel<S: Session, P, X0, X1, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, X0, X1) -> Result<Y>>>;
}

pub trait TernaryKernel<S: Session, P, X0, X1, X2, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, X0, X1, X2) -> Result<Y>>>;
}

pub trait VariadicKernel<S: Session, P, XS, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, Vec<XS>) -> Result<Y>>>;
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

pub trait Tensor<S: Session> {
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

pub trait PlacementKeyGen<S: Session, KeyT> {
    fn gen_key(&self, sess: &S) -> KeyT;
}

pub trait PlacementSetupGen<S: Session, SetupT> {
    fn gen_setup(&self, sess: &S) -> SetupT;
}

pub trait PlacementDeriveSeed<S: Session, KeyT, SeedT> {
    fn derive_seed(&self, sess: &S, sync_key: SyncKey, key: &KeyT) -> SeedT;
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

pub trait PlacementBitDec<S: Session, T, O> {
    fn bit_decompose(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementBitCompose<S: Session, T, O> {
    fn bit_compose(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementRingInject<S: Session, T, O> {
    fn ring_inject(&self, sess: &S, bit_idx: usize, x: &T) -> O;
}

pub trait PlacementShare<S: Session, T, O> {
    fn share(&self, sess: &S, x: &T) -> O;
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
    fn mean(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
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

pub trait PlacementSqrt<S: Session, T, O> {
    fn sqrt(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementAddN<S: Session, T, O> {
    fn add_n(&self, sess: &S, x: &[T]) -> O;
}

pub trait PlacementSum<S: Session, T, O> {
    fn sum(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

pub trait PlacementEqual<S: Session, T, U, O> {
    fn equal(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementIfElse<S: Session, T, U, V, O> {
    fn if_else(&self, sess: &S, s: &T, x: &U, y: &V) -> O;
}

pub trait PlacementPow2<S: Session, T, O> {
    fn pow2(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementExp<S: Session, T, O> {
    fn exp(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementSigmoid<S: Session, T, O> {
    fn sigmoid(&self, sess: &S, x: &T) -> O;
}
pub trait PlacementLessThan<S: Session, T, U, O> {
    fn less_than(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementGreaterThan<S: Session, T, U, O> {
    fn greater_than(&self, sess: &S, x: &T, y: &U) -> O;
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

modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostFloat32Tensor, HostOnesOp);
modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostFloat64Tensor, HostOnesOp);
modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostInt8Tensor, HostOnesOp);
modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostInt16Tensor, HostOnesOp);
modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostInt32Tensor, HostOnesOp);
modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostInt64Tensor, HostOnesOp);

kernel! {
    HostOnesOp, [
        (HostPlacement, (HostShape) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt64Tensor => [runtime] Self::kernel),
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

pub trait PlacementSample<S: Session, ShapeT, O> {
    fn sample(&self, sess: &S, max_value: Option<u64>, shape: &ShapeT) -> O;
}

pub trait PlacementSampleUniform<S: Session, ShapeT, O> {
    fn sample_uniform(&self, sess: &S, shape: &ShapeT) -> O;
}

pub trait PlacementSampleBits<S: Session, ShapeT, O> {
    fn sample_bits(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementSampleUniform<S, ShapeT, O> for P
where
    P: PlacementSample<S, ShapeT, O>,
{
    fn sample_uniform(&self, sess: &S, shape: &ShapeT) -> O {
        self.sample(sess, None, shape)
    }
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

pub trait PlacementSampleUniformSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_uniform_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

pub trait PlacementSampleBitsSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_bits_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

impl<S: Session, ShapeT, SeedT, O, P> PlacementSampleUniformSeeded<S, ShapeT, SeedT, O> for P
where
    P: PlacementSampleSeeded<S, ShapeT, SeedT, O>,
{
    fn sample_uniform_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O {
        self.sample_seeded(sess, None, shape, seed)
    }
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

pub trait PlacementAdtToRep<S: Session, T, O> {
    fn adt_to_rep(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementTruncPr<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: u32, x: &T) -> O;
}

pub trait PlacementTruncPrProvider<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: usize, provider: &HostPlacement, x: &T) -> O;
}

pub trait PlacementDaBitProvider<S: Session, ShapeT, O1, O2> {
    fn gen_dabit(
        &self,
        sess: &S,
        shape_provider: ShapeT,
        shape_a: ShapeT,
        provider: &HostPlacement,
    ) -> (O1, O2);
}

pub trait PlacementAbs<S: Session, T, O> {
    fn abs(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementMsb<S: Session, T, O> {
    fn msb(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementSign<S: Session, T, O> {
    fn sign(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementPlace<S: Session, T> {
    fn place(&self, sess: &S, x: T) -> T;
}

pub trait PlacementConstant<S: Session, O> {
    fn constant(&self, sess: &S, value: Constant) -> O;
}

pub trait PlacementIdentity<S: Session, T, O> {
    fn identity(&self, sess: &S, x: &T) -> O;
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

pub trait PlacementSend<S: Session, T, O> {
    fn send(&self, sess: &S, rendezvous_key: RendezvousKey, receiver: Role, x: &T) -> O;
}

pub trait PlacementReceive<S: Session, O> {
    fn receive(&self, sess: &S, rendezvous_key: RendezvousKey, sender: Role) -> O;
}

pub trait PlacementAtLeast2D<S: Session, T, O> {
    fn at_least_2d(&self, sess: &S, to_column_vector: bool, x: &T) -> O;
}

pub trait PlacementRingFixedpointEncode<S: Session, T, O> {
    fn fixedpoint_ring_encode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
}

pub trait PlacementRingFixedpointDecode<S: Session, T, O> {
    fn fixedpoint_ring_decode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
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

pub trait PlacementFixedpointDecode<S: Session, T, O> {
    fn fixedpoint_decode(&self, sess: &S, precision: u32, x: &T) -> O;
}

pub trait PlacementExpandDims<S: Session, T, O> {
    fn expand_dims(&self, sess: &S, axis: Vec<u32>, x: &T) -> O;
}

pub trait PlacementSqueeze<S: Session, T, O> {
    fn squeeze(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

pub trait PlacementConcatenate<S: Session, TS, O> {
    fn concatenate(&self, sess: &S, axis: u32, xs: &[TS]) -> O;
}

pub trait PlacementTranspose<S: Session, T, O> {
    fn transpose(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementInverse<S: Session, T, O> {
    fn inverse(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementCast<S: Session, T, O> {
    fn cast(&self, sess: &S, x: &T) -> O;
}

pub trait EmptyTypeHolder<T> {}

pub trait PlacementSlice<S: Session, T, O> {
    fn slice(&self, sess: &S, slice_info: SliceInfo, x: &T) -> O;
}

pub trait PlacementDiag<S: Session, T, O> {
    fn diag(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementIndexAxis<S: Session, T, O> {
    fn index_axis(&self, sess: &S, axis: usize, index: usize, x: &T) -> O;
}

pub trait PlacementIndex<S: Session, T, O> {
    fn index(&self, sess: &S, index: usize, x: &T) -> O;
}

pub trait PlacementShlDim<S: Session, T, O> {
    fn shl_dim(&self, sess: &S, amount: usize, ring_size: usize, x: &T) -> O;
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
            HostExpandDims(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostReshape(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostAtLeast2D(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostSlice(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostSum(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostTranspose(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostInverse(op) => Compile::<SyncKernel>::compile(op, ctx),
            HostConcat(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingNeg(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingAdd(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSub(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingMul(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingDot(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSum(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingFixedpointEncode(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingFixedpointDecode(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingFixedpointMean(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSample(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSampleSeeded(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingShl(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingShr(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingInject(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitExtract(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitSample(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitSampleSeeded(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitXor(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitAnd(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitNeg(_) => unimplemented!(),
            PrimDeriveSeed(op) => Compile::<SyncKernel>::compile(op, ctx),
            PrimPrfKeyGen(op) => Compile::<SyncKernel>::compile(op, ctx),
            FixedpointEncode(op) => Compile::<SyncKernel>::compile(op, ctx),
            FixedpointDecode(op) => Compile::<SyncKernel>::compile(op, ctx),
            FixedpointAdd(op) => Compile::<SyncKernel>::compile(op, ctx),
            FixedpointSub(op) => Compile::<SyncKernel>::compile(op, ctx),
            FloatingpointAdd(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointSub(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointMul(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointDiv(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointDot(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointAtLeast2D(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointOnes(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointConcat(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointExpandDims(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointTranspose(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointInverse(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointMean(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointSum(op) => unimplemented!("Not done yet: {:?}", op),
            AtLeast2D(op) => unimplemented!("Not done yet: {:?}", op),
            Slice(op) => unimplemented!("Not done yet: {:?}", op),
            Ones(op) => unimplemented!("Not done yet: {:?}", op),
            ExpandDims(op) => unimplemented!("Not done yet: {:?}", op),
            Concat(op) => unimplemented!("Not done yet: {:?}", op),
            Transpose(op) => unimplemented!("Not done yet: {:?}", op),
            Dot(op) => unimplemented!("Not done yet: {:?}", op),
            Inverse(op) => unimplemented!("Not done yet: {:?}", op),
            Add(op) => unimplemented!("Not done yet: {:?}", op),
            Sub(op) => unimplemented!("Not done yet: {:?}", op),
            Mul(op) => unimplemented!("Not done yet: {:?}", op),
            Mean(op) => unimplemented!("Not done yet: {:?}", op),
            Neg(op) => unimplemented!("Not done yet: {:?}", op),
            Sum(op) => unimplemented!("Not done yet: {:?}", op),
            Div(op) => unimplemented!("Not done yet: {:?}", op),
            Pow2(op) => unimplemented!("Not done yet: {:?}", op),
            Exp(op) => unimplemented!("Not done yet: {:?}", op),
            Sigmoid(op) => unimplemented!("Not done yet: {:?}", op),
            LessThan(op) => unimplemented!("Not done yet: {:?}", op),
            GreaterThan(op) => unimplemented!("Not done yet: {:?}", op),
            // TODO
            AesDecrypt(_) => unimplemented!(),
            HostIndexAxis(_) => unimplemented!(),
            HostBitDec(_) => unimplemented!(),
            HostShlDim(_) => unimplemented!(),
            HostSqrt(_) => unimplemented!(),
            HostDiag(_) => unimplemented!(),
            HostSqueeze(_) => unimplemented!(),
            Cast(_) => unimplemented!("No implementation of Cast for the old framework"),
            _ => {
                unimplemented!("Not supported {:?}", self)
            }
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
            HostExpandDims(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostReshape(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostAtLeast2D(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostSlice(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostSum(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostTranspose(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostInverse(op) => Compile::<AsyncKernel>::compile(op, ctx),
            HostConcat(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingNeg(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingAdd(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSub(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingMul(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingDot(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSum(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingFixedpointEncode(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingFixedpointDecode(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingFixedpointMean(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSample(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSampleSeeded(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingShl(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingShr(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingInject(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitExtract(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitSample(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitSampleSeeded(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitXor(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitAnd(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitNeg(_) => unimplemented!(),
            PrimDeriveSeed(op) => Compile::<AsyncKernel>::compile(op, ctx),
            PrimPrfKeyGen(op) => Compile::<AsyncKernel>::compile(op, ctx),
            AddN(op) => unimplemented!("Not done yet: {:?}", op),
            AtLeast2D(op) => unimplemented!("Not done yet: {:?}", op),
            Slice(op) => unimplemented!("Not done yet: {:?}", op),
            Ones(op) => unimplemented!("Not done yet: {:?}", op),
            ExpandDims(op) => unimplemented!("Not done yet: {:?}", op),
            Concat(op) => unimplemented!("Not done yet: {:?}", op),
            Transpose(op) => unimplemented!("Not done yet: {:?}", op),
            Dot(op) => unimplemented!("Not done yet: {:?}", op),
            Inverse(op) => unimplemented!("Not done yet: {:?}", op),
            Add(op) => unimplemented!("Not done yet: {:?}", op),
            Sub(op) => unimplemented!("Not done yet: {:?}", op),
            Mul(op) => unimplemented!("Not done yet: {:?}", op),
            Mean(op) => unimplemented!("Not done yet: {:?}", op),
            Neg(op) => unimplemented!("Not done yet: {:?}", op),
            Sum(op) => unimplemented!("Not done yet: {:?}", op),
            Div(op) => unimplemented!("Not done yet: {:?}", op),
            Pow2(op) => unimplemented!("Not done yet: {:?}", op),
            Exp(op) => unimplemented!("Not done yet: {:?}", op),
            Sigmoid(op) => unimplemented!("Not done yet: {:?}", op),
            LessThan(op) => unimplemented!("Not done yet: {:?}", op),
            GreaterThan(op) => unimplemented!("Not done yet: {:?}", op),
            // TODO implement below (needed until we switch to new framework for execution)
            AesDecrypt(_) => unimplemented!(),
            FixedpointEncode(_) | FixedpointDecode(_) | FixedpointAdd(_) | FixedpointSub(_)
            | FixedpointMul(_) | FixedpointDot(_) | FixedpointTruncPr(_) | FixedpointMean(_)
            | FixedpointSum(_) | HostBitDec(_) | HostIndexAxis(_) | HostShlDim(_) | HostSqrt(_)
            | HostSqueeze(_) | HostDiag(_) | Cast(_) => {
                unimplemented!("deprecated, not impl {:?}", self)
            }
            FloatingpointAdd(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointSub(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointMul(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointDiv(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointDot(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointAtLeast2D(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointOnes(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointConcat(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointExpandDims(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointTranspose(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointInverse(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointMean(op) => unimplemented!("Not done yet: {:?}", op),
            FloatingpointSum(op) => unimplemented!("Not done yet: {:?}", op),
            _ => {
                unimplemented!("Not supported {:?}", self)
            }
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
    (vec[$ts: pat] -> $ret: pat) => {
        Signature::Variadic(VariadicSignature {
            args: $ts,
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
    ($op:ident, $k:expr) => {
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
    };
}

host_binary_kernel!(HostAddOp, |x, y| x + y);
host_binary_kernel!(HostSubOp, |x, y| x - y);
host_binary_kernel!(HostMulOp, |x, y| x * y);
host_binary_kernel!(HostDivOp, |x, y| x / y);
host_binary_kernel!(HostDotOp, |x, y| x.dot(y));
host_unary_kernel!(HostTransposeOp, |x| x.transpose());

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

impl Compile<Kernel> for HostMeanOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|x| x as usize);
        match self.sig {
            signature![(_) -> Ty::HostFloat32Tensor] => {
                closure_kernel!(HostFloat32Tensor, |x| x.mean(axis).unwrap())
            }
            signature![(_) -> Ty::HostFloat64Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.mean(axis).unwrap())
            }
            signature![(_) -> Ty::HostInt32Tensor] => {
                closure_kernel!(HostInt32Tensor, |x| x.mean(axis).unwrap())
            }
            signature![(_) -> Ty::HostInt64Tensor] => {
                closure_kernel!(HostInt64Tensor, |x| x.mean(axis).unwrap())
            }
            signature![(_) -> Ty::HostUint32Tensor] => {
                closure_kernel!(HostUint32Tensor, |x| x.mean(axis).unwrap())
            }
            signature![(_) -> Ty::HostUint64Tensor] => {
                closure_kernel!(HostUint64Tensor, |x| x.mean(axis).unwrap())
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

impl Compile<Kernel> for HostConcatOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis as usize;
        match self.sig {
            signature![vec[_] -> Ty::HostFloat32Tensor] => {
                closure_kernel!(vec[HostFloat32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![vec[_] -> Ty::HostFloat64Tensor] => {
                closure_kernel!(vec[HostFloat64Tensor], |xs| concatenate(axis, &xs))
            }
            signature![vec[_]  -> Ty::HostInt32Tensor] => {
                closure_kernel!(vec[HostInt32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![vec[_]  -> Ty::HostInt64Tensor] => {
                closure_kernel!(vec[HostInt64Tensor], |xs| concatenate(axis, &xs))
            }
            signature![vec[_] -> Ty::HostUint32Tensor] => {
                closure_kernel!(vec[HostUint32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![vec[_]  -> Ty::HostUint64Tensor] => {
                closure_kernel!(vec[HostUint64Tensor], |xs| concatenate(axis, &xs))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
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

impl Compile<Kernel> for HostAtLeast2DOp {
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
        assert!(self.slice.0.len() == 1);
        let start = self.slice.0[0].start as usize;
        let end = self.slice.0[0].end;

        if let Some(end) = end {
            match self.sig {
                signature![(_) -> Ty::HostShape] => {
                    closure_kernel!(HostShape, |x| HostShape(
                        x.0.slice(start, end as usize),
                        x.1
                    ))
                }
                _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
            }
        } else {
            Err(Error::UnimplementedOperator(format!("{:?}", self)))
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for HostSumOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.sig {
            signature![(_) -> Ty::HostFloat32Tensor] => {
                closure_kernel!(HostFloat32Tensor, |x| x.sum(axis).unwrap())
            }
            signature![(_) -> Ty::HostFloat64Tensor] => {
                closure_kernel!(HostFloat64Tensor, |x| x.sum(axis).unwrap())
            }
            signature![(_) -> Ty::HostInt32Tensor] => {
                closure_kernel!(HostInt32Tensor, |x| x.sum(axis).unwrap())
            }
            signature![(_) -> Ty::HostInt64Tensor] => {
                closure_kernel!(HostInt64Tensor, |x| x.sum(axis).unwrap())
            }
            signature![(_) -> Ty::HostUint32Tensor] => {
                closure_kernel!(HostUint32Tensor, |x| x.sum(axis).unwrap())
            }
            signature![(_) -> Ty::HostUint64Tensor] => {
                closure_kernel!(HostUint64Tensor, |x| x.sum(axis).unwrap())
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

// This impl is only used by the old kernels, which are not aware of the placements. See PrimDeriveSeedOp::kernel for the new code
#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for PrimDeriveSeedOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let sync_key = self.sync_key.clone();
        closure_kernel!(PrfKey, |key| Seed(
            RawSeed::from_prf(&key.0, &sync_key),
            HostPlacement {
                owner: "TODO".into() // Fake owner for the older kernels.
            }
        ))
    }
}

// This impl is only used by the old kernels, which are not aware of the placements. See PrimPrfKeyGenOp::kernel for the new code
#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for RingDotOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostRing64Tensor, Ty::HostRing64Tensor) -> _] => {
                function_kernel!(HostRing64Tensor, HostRing64Tensor, |x, y| x.dot(y).unwrap())
            }
            signature![(Ty::HostRing128Tensor, Ty::HostRing128Tensor) -> _] => {
                function_kernel!(HostRing128Tensor, HostRing128Tensor, |x, y| x
                    .dot(y)
                    .unwrap())
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for RingSumOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.sig {
            signature![(_) -> Ty::HostRing64Tensor] => {
                closure_kernel!(HostRing64Tensor, |x| x.sum(axis).unwrap())
            }
            signature![(_) -> Ty::HostRing128Tensor] => {
                closure_kernel!(HostRing128Tensor, |x| x.sum(axis).unwrap())
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for RingSampleOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (&self.sig, self.max_value) {
            (signature![(_) -> Ty::HostRing64Tensor], None) => {
                function_kernel!(HostShape, |shape| {
                    HostRing64Tensor::sample_uniform(&shape.0)
                })
            }
            (signature!((_) -> Ty::HostRing64Tensor), Some(max_value)) if max_value == 1 => {
                function_kernel!(HostShape, |shape| HostRing64Tensor::sample_bits(&shape.0))
            }
            (signature![(_) -> Ty::HostRing128Tensor], None) => {
                function_kernel!(HostShape, |shape| {
                    HostRing128Tensor::sample_uniform(&shape.0)
                })
            }
            (signature![(_) -> Ty::HostRing128Tensor], Some(max_value)) if max_value == 1 => {
                function_kernel!(HostShape, |shape| {
                    HostRing128Tensor::sample_bits(&shape.0)
                })
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for RingSampleSeededOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (&self.sig, self.max_value) {
            (signature![(_, _) -> Ty::HostRing64Tensor], None) => {
                function_kernel!(HostShape, Seed, |shape, seed| {
                    HostRing64Tensor::sample_uniform_seeded(&shape.0, &seed.0)
                })
            }
            (signature!((_, _) -> Ty::HostRing64Tensor), Some(max_value)) if max_value == 1 => {
                function_kernel!(HostShape, Seed, |shape, seed| {
                    HostRing64Tensor::sample_bits_seeded(&shape.0, &seed.0)
                })
            }
            (signature![(_, _) -> Ty::HostRing128Tensor], None) => {
                function_kernel!(HostShape, Seed, |shape, seed| {
                    HostRing128Tensor::sample_uniform_seeded(&shape.0, &seed.0)
                })
            }
            (signature![(_, _) -> Ty::HostRing128Tensor], Some(max_value)) if max_value == 1 => {
                function_kernel!(HostShape, Seed, |shape, seed| {
                    HostRing128Tensor::sample_bits_seeded(&shape.0, &seed.0)
                })
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for BitSampleOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(HostShape, |shape| HostBitTensor::sample_uniform(&shape.0))
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for BitSampleSeededOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(HostShape, Seed, |shape, seed| {
            HostBitTensor::sample_uniform_seeded(&shape.0, &seed.0)
        })
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for BitXorOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(HostBitTensor, HostBitTensor, |x, y| x ^ y)
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
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

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for RingFixedpointEncodeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
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

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for RingFixedpointDecodeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
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

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for RingFixedpointMeanOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.sig {
            signature![(_) -> Ty::HostRing64Tensor] => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(HostRing64Tensor, |x| HostRing64Tensor::fixedpoint_mean(
                    x,
                    axis,
                    scaling_factor
                )
                .unwrap())
            }
            signature![(_) -> Ty::HostRing128Tensor] => {
                let scaling_factor = u128::pow(self.scaling_base as u128, self.scaling_exp);
                closure_kernel!(HostRing128Tensor, |x| HostRing128Tensor::fixedpoint_mean(
                    x,
                    axis,
                    scaling_factor
                )
                .unwrap())
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for FixedpointEncodeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostFloat64Tensor) -> Ty::HostRing64Tensor] => {
                let scaling_factor = u64::pow(2, self.fractional_precision);
                closure_kernel!(HostFloat64Tensor, |x| HostRing64Tensor::encode(
                    &x,
                    scaling_factor
                ))
            }
            signature![(Ty::HostFloat64Tensor) -> Ty::HostRing128Tensor] => {
                let scaling_factor = u128::pow(2, self.fractional_precision);
                closure_kernel!(HostFloat64Tensor, |x| HostRing128Tensor::encode(
                    &x,
                    scaling_factor
                ))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for FixedpointDecodeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostRing64Tensor) -> _] => {
                let scaling_factor = u64::pow(2, self.fractional_precision);
                closure_kernel!(HostRing64Tensor, |x| HostRing64Tensor::decode(
                    &x,
                    scaling_factor
                ))
            }
            signature![(Ty::HostRing128Tensor) -> _] => {
                let scaling_factor = u128::pow(2, self.fractional_precision);
                closure_kernel!(HostRing128Tensor, |x| HostRing128Tensor::decode(
                    &x,
                    scaling_factor
                ))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for FixedpointAddOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostFixed64Tensor, Ty::HostFixed64Tensor) -> _] => {
                function_kernel!(HostFixed64Tensor, HostFixed64Tensor, |x, y| {
                    assert_eq!(x.fractional_precision, y.fractional_precision);
                    AbstractHostFixedTensor {
                        tensor: x.tensor + y.tensor,
                        fractional_precision: x.fractional_precision,
                        integral_precision: u32::max(x.integral_precision, y.integral_precision),
                    }
                })
            }
            signature![(Ty::HostFixed128Tensor, Ty::HostFixed128Tensor) -> _] => {
                function_kernel!(HostFixed128Tensor, HostFixed128Tensor, |x, y| {
                    assert_eq!(x.fractional_precision, y.fractional_precision);
                    AbstractHostFixedTensor {
                        tensor: x.tensor + y.tensor,
                        fractional_precision: x.fractional_precision,
                        integral_precision: u32::max(x.integral_precision, y.integral_precision),
                    }
                })
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for FixedpointSubOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostFixed64Tensor, Ty::HostFixed64Tensor) -> _] => {
                function_kernel!(HostFixed64Tensor, HostFixed64Tensor, |x, y| {
                    assert_eq!(x.fractional_precision, y.fractional_precision);
                    AbstractHostFixedTensor {
                        tensor: x.tensor - y.tensor,
                        fractional_precision: x.fractional_precision,
                        integral_precision: u32::max(x.integral_precision, y.integral_precision),
                    }
                })
            }
            signature![(Ty::HostFixed128Tensor, Ty::HostFixed128Tensor) -> _] => {
                function_kernel!(HostFixed128Tensor, HostFixed128Tensor, |x, y| {
                    assert_eq!(x.fractional_precision, y.fractional_precision);
                    AbstractHostFixedTensor {
                        tensor: x.tensor - y.tensor,
                        fractional_precision: x.fractional_precision,
                        integral_precision: u32::max(x.integral_precision, y.integral_precision),
                    }
                })
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<Kernel> for FixedpointMulOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::HostFixed64Tensor, Ty::HostFixed64Tensor) -> _] => {
                function_kernel!(HostFixed64Tensor, HostFixed64Tensor, |x, y| {
                    assert_eq!(x.fractional_precision, y.fractional_precision);
                    AbstractHostFixedTensor {
                        tensor: x.tensor * y.tensor,
                        fractional_precision: x.fractional_precision + y.fractional_precision,
                        integral_precision: u32::max(x.integral_precision, y.integral_precision),
                    }
                })
            }
            signature![(Ty::HostFixed128Tensor, Ty::HostFixed128Tensor) -> _] => {
                function_kernel!(HostFixed128Tensor, HostFixed128Tensor, |x, y| {
                    assert_eq!(x.fractional_precision, y.fractional_precision);
                    AbstractHostFixedTensor {
                        tensor: x.tensor * y.tensor,
                        fractional_precision: x.fractional_precision + y.fractional_precision,
                        integral_precision: u32::max(x.integral_precision, y.integral_precision),
                    }
                })
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

// This impl is only used by the old kernels, which are not aware of the placements. See ConstantOp::kernel for the new code
#[cfg(not(feature = "exclude_old_framework"))]
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

macro_rules! constant_kernels {
    ($($val:ident),+) => {
        $(
            modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> $val, ConstantOp);
        )+
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> HostString, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> HostShape, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> PrfKey, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Seed, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> crate::logical::Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Float32Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Float64Tensor, ConstantOp);

        kernel! {
            ConstantOp, [
                $(
                    (HostPlacement, () -> $val => [runtime] attributes[value: $val] Self::kernel),
                )+
                (HostPlacement, () -> HostString => [runtime] attributes[value: String] Self::string_kernel),
                (HostPlacement, () -> HostShape => [runtime] attributes[value: RawShape] Self::shape_kernel),
                (HostPlacement, () -> PrfKey => [runtime] attributes[value: RawPrfKey] Self::prf_key_kernel),
                (HostPlacement, () -> Seed => [runtime] attributes[value: RawSeed] Self::seed_kernel),
                (HostPlacement, () -> crate::logical::Tensor => [hybrid] attributes[sig, value] Self::logical_kernel),
                (HostPlacement, () -> Float32Tensor => [hybrid] attributes[value] Self::float32_kernel),
                (HostPlacement, () -> Float64Tensor => [hybrid] attributes[value] Self::float64_kernel),
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

macro_rules! wrapping_constant_kernel {
    ($name:ident for $wrapping:tt($inner:ty)) => {
        impl ConstantOp {
            fn $name<S: RuntimeSession>(
                _sess: &S,
                plc: &HostPlacement,
                value: $inner,
            ) -> Result<$wrapping> {
                Ok($wrapping(value.clone(), plc.clone()))
            }
        }
    };
}

wrapping_constant_kernel!(string_kernel for HostString(String));
wrapping_constant_kernel!(shape_kernel for HostShape(RawShape));
wrapping_constant_kernel!(prf_key_kernel for PrfKey(RawPrfKey));
wrapping_constant_kernel!(seed_kernel for Seed(RawSeed));

impl ConstantOp {
    fn kernel<S: RuntimeSession, T: Placed>(sess: &S, plc: &HostPlacement, value: T) -> Result<T>
    where
        HostPlacement: PlacementPlace<S, T>,
    {
        Ok(plc.place(sess, value))
    }
}

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementSend::send, HostPlacement, attributes[rendezvous_key: RendezvousKey, receiver: Role] ($value) -> Unit, SendOp);
    )*
)}

kernel! {
    SendOp, [
        (HostPlacement, (HostString) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (Unit) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostShape) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (Seed) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (PrfKey) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostBitTensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostRing64Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostFloat32Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostUint8Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostUint16Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostFixed64Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostFixed128Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
    ]
}

impl SendOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _receiver: Role,
        _x: T,
    ) -> Result<Unit>
    where
        Value: From<T>,
    {
        // let x: Value = x.into();
        // sess.networking.send(&x, &receiver, &rendezvous_key)?;
        // Ok(Unit(plc.clone()))
        todo!()
    }
}

// This impl is only used by the old kernels, which are not aware of the placements.
#[cfg(not(feature = "exclude_old_framework"))]
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
            Ok(Value::Unit(Box::new(Unit(HostPlacement {
                owner: "TODO".into(), // Fake owner for the older kernels.
            }))))
        })))
    }
}

// This impl is only used by the old kernels, which are not aware of the placements.
#[cfg(not(feature = "exclude_old_framework"))]
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
                map_send_result(sender.send(Value::Unit(Box::new(Unit(HostPlacement {
                    owner: "TODO".into(), // Fake owner for the older kernels.
                })))))
            })
        })))
    }
}

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementReceive::receive, HostPlacement, attributes[rendezvous_key: RendezvousKey, sender: Role] () -> $value, ReceiveOp);
    )*
)}

kernel! {
    ReceiveOp, [
        (HostPlacement, () -> HostString => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> Unit => [runtime] attributes[rendezvous_key, sender] Self::missing_kernel),
        (HostPlacement, () -> HostShape => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> Seed => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> PrfKey => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostBitTensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostRing64Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostRing128Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostFloat32Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostFloat64Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostInt8Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostInt16Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostInt32Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostInt64Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostUint8Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostUint16Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostUint32Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostUint64Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostFixed64Tensor => [runtime] attributes[rendezvous_key, sender] Self::missing_kernel),
        (HostPlacement, () -> HostFixed128Tensor => [runtime] attributes[rendezvous_key, sender] Self::missing_kernel),

    ]
}

impl ReceiveOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _sender: Role,
    ) -> Result<T>
    where
        T: TryFrom<Value, Error = Error>,
        T: std::fmt::Debug,
        HostPlacement: PlacementPlace<S, T>,
    {
        // use std::convert::TryInto;
        // let value = sess.networking.receive(&sender, &rendezvous_key)?;
        // Ok(plc.place(sess, value.try_into()?))
        todo!()
    }

    fn missing_kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _sender: Role,
    ) -> Result<T>
    where
        T: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<T as KnownType<S>>::TY
        )))
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

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementIdentity::identity, HostPlacement, ($value) -> $value, IdentityOp);
    )*
)}

kernel! {
    IdentityOp, [
        (HostPlacement, (HostString) -> HostString => [runtime] Self::kernel),
        (HostPlacement, (Unit) -> Unit => [runtime] Self::missing_kernel),
        (HostPlacement, (HostShape) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (Seed) -> Seed => [runtime] Self::kernel),
        (HostPlacement, (PrfKey) -> PrfKey => [runtime] Self::kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint8Tensor) -> HostUint8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint16Tensor) -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [runtime] Self::missing_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [runtime] Self::missing_kernel),

    ]
}

impl IdentityOp {
    fn kernel<S: RuntimeSession, T>(sess: &S, plc: &HostPlacement, x: T) -> Result<T>
    where
        HostPlacement: PlacementPlace<S, T>,
    {
        let value = plc.place(sess, x);
        Ok(value)
    }

    fn missing_kernel<S: RuntimeSession, T>(_sess: &S, _plc: &HostPlacement, _x: T) -> Result<T>
    where
        T: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<T as KnownType<S>>::TY
        )))
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

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> $value, InputOp);
    )*
)}

modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> crate::logical::Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Float32Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Float64Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostBitArray64, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostBitArray128, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostBitArray224, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostAesKey, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitTensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedRing64Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedRing128Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedFixed64Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedFixed128Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitArray64, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitArray128, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitArray224, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedAesKey, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> AesKey, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> AesKey, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> AesTensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Fixed128AesTensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostFixed128AesTensor, InputOp);

kernel! {
    InputOp, [
        (HostPlacement, () -> HostString => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> Unit => [runtime] attributes[arg_name] Self::missing_kernel),
        (HostPlacement, () -> HostShape => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> Seed => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> PrfKey => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostBitArray64 => [hybrid] attributes[arg_name] Self::host_bitarray64),
        (HostPlacement, () -> HostBitArray128 => [hybrid] attributes[arg_name] Self::host_bitarray128),
        (HostPlacement, () -> HostBitArray224 => [hybrid] attributes[arg_name] Self::host_bitarray224),
        (HostPlacement, () -> HostBitTensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostRing64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostRing128Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostFloat32Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostFloat64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt8Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt16Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt32Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint8Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint16Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint32Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostFixed64Tensor => [runtime] attributes[arg_name] Self::missing_kernel),
        (HostPlacement, () -> HostFixed128Tensor => [runtime] attributes[arg_name] Self::missing_kernel),
        (HostPlacement, () -> crate::logical::Tensor => [hybrid] attributes[sig, arg_name] Self::logical_kernel),
        (HostPlacement, () -> Float32Tensor => [hybrid] attributes[arg_name] Self::float32_kernel),
        (HostPlacement, () -> Float64Tensor => [hybrid] attributes[arg_name] Self::float64_kernel),
        (HostPlacement, () -> AesKey => [hybrid] attributes[arg_name] Self::aes_kernel_on_host),
        (HostPlacement, () -> HostAesKey => [hybrid] attributes[arg_name] Self::host_aes_kernel),
        (HostPlacement, () -> AesTensor => [hybrid] attributes[arg_name] Self::aestensor),
        (HostPlacement, () -> Fixed128AesTensor => [hybrid] attributes[arg_name] Self::fixed_aestensor),
        (HostPlacement, () -> HostFixed128AesTensor => [hybrid] attributes[sig, arg_name] Self::host_fixed_aestensor),
        (ReplicatedPlacement, () -> ReplicatedBitTensor => [hybrid] attributes[arg_name] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedRing64Tensor => [hybrid] attributes[arg_name] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedRing128Tensor => [hybrid] attributes[arg_name] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedFixed64Tensor => [hybrid] attributes[sig, arg_name] Self::replicated_fixed_kernel),
        (ReplicatedPlacement, () -> ReplicatedFixed128Tensor => [hybrid] attributes[sig, arg_name] Self::replicated_fixed_kernel),
        (ReplicatedPlacement, () -> ReplicatedBitArray64 => [hybrid] attributes[arg_name] Self::replicated_bitarray64),
        (ReplicatedPlacement, () -> ReplicatedBitArray128 => [hybrid] attributes[arg_name] Self::replicated_bitarray128),
        (ReplicatedPlacement, () -> ReplicatedBitArray224 => [hybrid] attributes[arg_name] Self::replicated_bitarray224),
        (ReplicatedPlacement, () -> AesKey => [hybrid] attributes[arg_name] Self::aes_kernel_on_replicated),
        (ReplicatedPlacement, () -> ReplicatedAesKey => [hybrid] attributes[arg_name] Self::replicated_aes_kernel),
    ]
}

impl InputOp {
    fn kernel<S: RuntimeSession, O>(sess: &S, plc: &HostPlacement, arg_name: String) -> Result<O>
    where
        O: TryFrom<Value, Error = Error>,
        HostPlacement: PlacementPlace<S, O>,
    {
        use std::convert::TryInto;
        let value = sess
            .find_argument(&arg_name)
            .ok_or_else(|| Error::MissingArgument(arg_name.clone()))?;
        let value = plc.place(sess, value.try_into()?);
        Ok(value)
    }

    fn missing_kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _arg_name: String,
    ) -> Result<O>
    where
        O: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<O as KnownType<S>>::TY
        )))
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

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementOutput::output, HostPlacement, ($value) -> $value, OutputOp);
    )*
)}
modelled!(PlacementOutput::output, HostPlacement, (crate::logical::Tensor) -> crate::logical::Tensor, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (Float32Tensor) -> Float32Tensor, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (Float64Tensor) -> Float64Tensor, OutputOp);

kernel! {
    OutputOp, [
        (HostPlacement, (Unit) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (Seed) -> Seed => [runtime] Self::kernel),
        (HostPlacement, (PrfKey) -> PrfKey => [runtime] Self::kernel),
        (HostPlacement, (HostString) -> HostString => [runtime] Self::kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint8Tensor) -> HostUint8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint16Tensor) -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [runtime] Self::non_placing_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [runtime] Self::non_placing_kernel),
        (HostPlacement, (crate::logical::Tensor) -> crate::logical::Tensor => [hybrid] Self::logical_kernel),
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [hybrid] Self::float_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [hybrid] Self::float_kernel),
    ]
}

impl OutputOp {
    fn kernel<S: RuntimeSession, O>(sess: &S, plc: &HostPlacement, x: O) -> Result<O>
    where
        HostPlacement: PlacementPlace<S, O>,
    {
        // Output is not doing anything now, it is just a marker on the graph.
        // But it has to return a value because that's how we collect outputs in the old framework
        let x = plc.place(sess, x);
        Ok(x)
    }

    fn non_placing_kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        x: O,
    ) -> Result<O> {
        // Output is not doing anything now, it is just a marker on the graph.
        // But it has to return a value because that's how we collect outputs in the old framework
        Ok(x)
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

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementSave::save, HostPlacement, (HostString, $value) -> Unit, SaveOp);
    )*
)}

modelled!(PlacementSave::save, HostPlacement, (HostString, crate::logical::Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, Float32Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, Float64Tensor) -> Unit, SaveOp);

kernel! {
    SaveOp, [
        (HostPlacement, (HostString, Unit) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostShape) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, Seed) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, PrfKey) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostBitTensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostRing64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostRing128Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFloat32Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFloat64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt8Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt16Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt32Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint8Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint16Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint32Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFixed64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFixed128Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, crate::logical::Tensor) -> Unit => [hybrid] Self::logical_kernel),
        (HostPlacement, (HostString, Float32Tensor) -> Unit => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, Float64Tensor) -> Unit => [hybrid] Self::float_kernel),
    ]
}

impl SaveOp {
    fn kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: HostString,
        _x: O,
    ) -> Result<Unit>
    where
        Value: From<O>,
    {
        // let x: Value = x.into();
        // sess.storage.save(&key.0, &x)?;
        // Ok(Unit(plc.clone()))
        todo!()
    }
}

// This implementation is the old kernel.
#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<SyncKernel> for SaveOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        let expected_ty = self.sig.arg(1)?;

        Ok(SyncKernel::Binary(Box::new(move |sess, key, val| {
            let key = HostString::try_from(key)?;
            check_type(&val, expected_ty)?;
            sess.storage.save(&key.0, &sess.sid, &val)?;
            Ok(Value::Unit(Box::new(Unit(HostPlacement {
                owner: "TODO".into(), // Fake owner for the old kernel
            }))))
        })))
    }
}

// This implementation is the old kernel.
#[cfg(not(feature = "exclude_old_framework"))]
impl Compile<AsyncKernel> for SaveOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.sig.arg(1)?;

        Ok(AsyncKernel::Binary(Box::new(
            move |sess, key, val, sender| {
                let sess = Arc::clone(sess);

                tokio::spawn(async move {
                    let key = HostString::try_from(key.await.map_err(map_receive_error)?)?;
                    let val = val.await.map_err(map_receive_error)?;
                    check_type(&val, expected_ty)?;
                    sess.storage.save(&key.0, &sess.sid, &val).await?;
                    map_send_result(sender.send(Value::Unit(Box::new(Unit(HostPlacement {
                        owner: "TODO".into(), // Fake owner for the old kernel
                    })))))
                })
            },
        )))
    }
}

// for_all_values! {( $($value:ty),* ) => (
//     $(
//         modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> $value, LoadOp);
//     )*
// )}

modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> HostFloat64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> Float64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> crate::logical::Tensor, LoadOp);

kernel! {
    LoadOp, [
        (HostPlacement, (HostString, HostString) -> Unit => [runtime] Self::missing_kernel),
        (HostPlacement, (HostString, HostString) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> Seed => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> PrfKey => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostString => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostBitTensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostRing128Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostFixed64Tensor => [runtime] Self::missing_kernel),
        (HostPlacement, (HostString, HostString) -> HostFixed128Tensor => [runtime] Self::missing_kernel),
        (HostPlacement, (HostString, HostString) -> Float64Tensor => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, HostString) -> crate::logical::Tensor => [hybrid] Self::logical_kernel),
    ]
}

impl LoadOp {
    fn kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: HostString,
        _query: HostString,
    ) -> Result<O>
    where
        O: KnownType<S>,
        O: TryFrom<Value, Error = Error>,
        HostPlacement: PlacementPlace<S, O>,
    {
        // use std::convert::TryInto;
        // let value = sess.storage.load(&key.0, &query.0, Some(<O as KnownType<S>>::TY))?;
        // let value = plc.place(sess, value.try_into()?);
        // Ok(value)
        todo!()
    }

    fn missing_kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: HostString,
        _query: HostString,
    ) -> Result<O>
    where
        O: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<O as KnownType<S>>::TY
        )))
    }
}

kernel! {
    SigmoidOp,
    [
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [hybrid] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [hybrid] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [transparent] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [transparent] Self::rep_rep_kernel),
    ]
}

kernel! {
    LessThanOp,
    [
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, Mirrored3Ring64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, Mirrored3Ring128Tensor) -> ReplicatedRing128Tensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_fixed_mir_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_fixed_mir_kernel),
        // TODO(Dragos) these do not work now as they should output a boolean/ring type. makes no sense to output a fixed tensor
        // (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [transparent] Self::fixed_kernel),
        // instead it should be
        // (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> BitTensor => [transparent] Self::fixed_kernel),
    ]
}

kernel! {
    GreaterThanOp,
    [
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_kernel),

        (ReplicatedPlacement, (ReplicatedRing64Tensor, Mirrored3Ring64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::mir_rep_kernel),

        (ReplicatedPlacement, (ReplicatedRing128Tensor, Mirrored3Ring128Tensor) -> ReplicatedRing128Tensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [transparent] Self::mir_rep_kernel),

        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [transparent] Self::rep_kernel),

        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_fixed_mir_kernel),

        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_fixed_mir_kernel),

    ]
}

impl Compile<SyncKernel> for LoadOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        let expected_ty = self.sig.ret();

        Ok(SyncKernel::Binary(Box::new(move |sess, key, query| {
            let key = HostString::try_from(key)?;
            let query = HostString::try_from(query)?;
            let val = sess
                .storage
                .load(&key.0, &sess.sid, Some(expected_ty), &query.0)?;
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
                    let key = HostString::try_from(key.await.map_err(map_receive_error)?)?;
                    let query = HostString::try_from(query.await.map_err(map_receive_error)?)?;
                    let val = sess
                        .storage
                        .load(&key.0, &sess.sid, Some(expected_ty), &query.0)
                        .await?;
                    check_type(&val, expected_ty)?;
                    map_send_result(sender.send(val))
                })
            },
        )))
    }
}

pub struct TestSyncExecutor {
    // Placeholder for the future state we want to keep
}

impl Default for TestSyncExecutor {
    fn default() -> Self {
        TestSyncExecutor {}
    }
}

impl TestSyncExecutor {
    pub fn run_computation(
        &self,
        computation: &Computation,
        session: &SyncSession,
    ) -> anyhow::Result<HashMap<String, Value>> {
        let mut env: HashMap<String, Value> = HashMap::default();
        let computation = computation.toposort()?;

        let output_names: Vec<String> = computation
            .operations
            .iter() // guessing that par_iter won't help here
            .filter_map(|op| match op.kind {
                Operator::Output(_) => Some(op.name.clone()),
                _ => None,
            })
            .collect();

        for op in computation.operations.iter() {
            let operator = op.kind.clone();
            let operands = op
                .inputs
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();
            let value = session
                .execute(operator, &op.placement, operands)
                .map_err(|e| {
                    Error::Compilation(format!(
                        "SyncSession failed to execute computation due to an error: {:?}",
                        e,
                    ))
                })?;
            env.insert(op.name.clone(), value);
        }

        let outputs: HashMap<String, Value> = output_names
            .iter()
            .map(|op_name| (op_name.clone(), env.get(op_name).cloned().unwrap()))
            .collect();
        Ok(outputs)
    }
}
