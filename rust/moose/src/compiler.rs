#![allow(dead_code)]
#![allow(unused_variables)]

use std::convert::{TryFrom, TryInto};
use std::ops::{Add, Sub, Mul};

#[derive(Debug, Clone)]
enum Placement {
    HostPlacement(HostPlacement),
    ReplicatedPlacement(ReplicatedPlacement),
}

impl Placement {
    pub fn ty(&self) -> PlacementTy {
        match self {
            Placement::HostPlacement(plc) => plc.ty(),
            Placement::ReplicatedPlacement(plc) => plc.ty(),
        }
    }
}

#[derive(Clone, Debug)]
struct HostPlacement {
    player: String,
}

#[derive(Clone, Debug)]
struct ReplicatedPlacement {
    players: [String; 3],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlacementTy {
    HostTy,
    ReplicatedTy,
}

trait KnownPlacement {
    const TY: PlacementTy;

    fn ty(&self) -> PlacementTy {
        Self::TY
    }
}

impl KnownPlacement for HostPlacement {
    const TY: PlacementTy = PlacementTy::HostTy;
}

impl KnownPlacement for ReplicatedPlacement {
    const TY: PlacementTy = PlacementTy::ReplicatedTy;
}

macro_rules! placement {
    ($t:ident) => {
        impl From<$t> for Placement {
            fn from(x: $t) -> Placement {
                Placement::$t(x)
            }
        }
        
        impl From<&$t> for Placement {
            fn from(x: &$t) -> Placement {
                Placement::$t(x.clone())
            }
        }

        impl TryFrom<Placement> for $t {
            type Error = ();
        
            fn try_from(x: Placement) -> Result<Self, Self::Error> {
                match x {
                    Placement::$t(x) => Ok(x),
                    _ => Err(()),
                }
            }
        }
    }
}

placement!(HostPlacement);
placement!(ReplicatedPlacement);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    Ring32TensorTy,
    Ring64TensorTy,
    Ring128TensorTy,
    Replicated64TensorTy,
    Replicated128TensorTy,
    ReplicatedSetupTy,
    PrfKeyTy,
}

pub trait KnownType {
    type Hybrid;
    type Symbolic;
    const TY: Ty;
}

impl KnownType for Ring32Tensor {
    type Hybrid = Self;
    type Symbolic = Symbolic<Ring32Tensor>;
    const TY: Ty = Ty::Ring32TensorTy;
}

impl KnownType for Ring64Tensor {
    type Hybrid = Self;
    type Symbolic = Symbolic<Ring64Tensor>;
    const TY: Ty = Ty::Ring64TensorTy;
}

impl KnownType for Ring128Tensor {
    type Hybrid = Self;
    type Symbolic = Symbolic<Ring128Tensor>;
    const TY: Ty = Ty::Ring128TensorTy;
}

impl KnownType for Replicated64Tensor {
    type Hybrid = ReplicatedTensor<Symbolic<Ring64Tensor>>;
    type Symbolic = Symbolic<Self::Hybrid>;
    const TY: Ty = Ty::Replicated64TensorTy;
}

impl KnownType for Replicated128Tensor {
    type Hybrid = ReplicatedTensor<Symbolic<Ring128Tensor>>;
    type Symbolic = Symbolic<Self::Hybrid>;
    const TY: Ty = Ty::Replicated128TensorTy;
}

impl KnownType for ReplicatedSetup {
    type Hybrid = AbstractReplicatedSetup<Symbolic<PrfKey>>;
    type Symbolic = Symbolic<Self::Hybrid>;
    const TY: Ty = Ty::ReplicatedSetupTy;
}

impl KnownType for PrfKey {
    type Hybrid = Self;
    type Symbolic = Symbolic<PrfKey>;
    const TY: Ty = Ty::PrfKeyTy;
}

impl<T: KnownType> KnownType for Symbolic<T> {
    type Hybrid = Self;
    type Symbolic = Self;
    const TY: Ty = T::TY;
}

#[derive(Clone, Debug)]
pub enum Value {
    Ring32Tensor(Ring32Tensor),
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    Replicated64Tensor(Replicated64Tensor),
    Replicated128Tensor(Replicated128Tensor),
    ReplicatedSetup(ReplicatedSetup),
    PrfKey(PrfKey),
}

#[derive(Clone, Debug)]
pub enum SymbolicValue {
    Ring32Tensor(<Ring32Tensor as KnownType>::Symbolic),
    Ring64Tensor(<Ring64Tensor as KnownType>::Symbolic),
    Ring128Tensor(<Ring128Tensor as KnownType>::Symbolic),
    Replicated64Tensor(<Replicated64Tensor as KnownType>::Symbolic),
    Replicated128Tensor(<Replicated128Tensor as KnownType>::Symbolic),
    ReplicatedSetup(<ReplicatedSetup as KnownType>::Symbolic),
    PrfKey(<PrfKey as KnownType>::Symbolic),
}

macro_rules! value {
    ($t:ident) => {
        impl From<$t> for Value {
            fn from(x: $t) -> Value {
                Value::$t(x)
            }
        }
        
        impl From<&$t> for Value {
            fn from(x: &$t) -> Value {
                Value::$t(x.clone())
            }
        }

        impl TryFrom<Value> for $t {
            type Error = ();
        
            fn try_from(x: Value) -> Result<Self, Self::Error> {
                match x {
                    Value::$t(x) => Ok(x),
                    _ => Err(()),
                }
            }
        }

        impl From<<$t as KnownType>::Symbolic> for SymbolicValue {
            fn from(x: <$t as KnownType>::Symbolic) -> SymbolicValue {
                SymbolicValue::$t(x)
            }
        }

        impl TryFrom<SymbolicValue> for <$t as KnownType>::Symbolic {
            type Error = ();

            fn try_from(x: SymbolicValue) -> Result<Self, Self::Error> {
                match x {
                    SymbolicValue::$t(x) => Ok(x),
                    _ => Err(()),
                }
            }
        }

    };
}

// NOTE a future improvement might be to have a single `values!` macro
// that takes care of everything, including generating `enum Value` and
// `enum SymbolicValue` and maybe even `enum Ty`.
// one thing to be careful about here is to still make room for manual
// constructions during development.
value!(Ring32Tensor);
value!(Ring64Tensor);
value!(Ring128Tensor);
value!(Replicated64Tensor);
value!(Replicated128Tensor);
value!(ReplicatedSetup);
value!(PrfKey);

#[derive(Clone, Debug)]
pub enum Symbolic<T> {
    Symbolic(SymbolicHandle),
    Concrete(T),
}

#[derive(Clone, Debug)]
pub struct SymbolicHandle {
    op: String,
}

impl<T> From<SymbolicHandle> for Symbolic<T> {
    fn from(x: SymbolicHandle) -> Symbolic<T> {
        Symbolic::Symbolic(x)
    }
}

#[derive(Clone, Debug)]
pub enum Operator {
    PrfKeyGenOp(PrfKeyGenOp),
    RingAddOp(RingAddOp),
    RingSubOp(RingSubOp),
    RingMulOp(RingMulOp),
    RingSampleOp(RingSampleOp),
    RepSetupOp(RepSetupOp),
    RepAddOp(RepAddOp),
    RepMulOp(RepMulOp),
    RepShareOp(RepShareOp),
    // Constant(ConstantOp),
}

macro_rules! operator {
    ($t:ident) => {
        impl From<$t> for Operator {
            fn from(x: $t) -> Operator {
                Operator::$t(x)
            }
        }
    };
}

// NOTE a future improvement might be to have a single `operators!` macro
// that takes care of everything, including generating `enum Operator`.
operator!(PrfKeyGenOp);
operator!(RingAddOp);
operator!(RingSubOp);
operator!(RingMulOp);
operator!(RingSampleOp);
operator!(RepSetupOp);
operator!(RepAddOp);
operator!(RepMulOp);
operator!(RepShareOp);

#[derive(Clone, Debug)]
struct Operation {
    name: String,
    operator: Operator,
    operands: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct RingTensor<T>(T);

impl Add<RingTensor<u64>> for RingTensor<u64> {
    type Output = RingTensor<u64>;

    fn add(self, other: RingTensor<u64>) -> Self::Output {
        RingTensor(self.0.wrapping_add(other.0))
    }
}

impl Add<RingTensor<u128>> for RingTensor<u128> {
    type Output = RingTensor<u128>;

    fn add(self, other: RingTensor<u128>) -> Self::Output {
        RingTensor(self.0.wrapping_add(other.0))
    }
}

impl Sub<RingTensor<u64>> for RingTensor<u64> {
    type Output = RingTensor<u64>;

    fn sub(self, other: RingTensor<u64>) -> Self::Output {
        RingTensor(self.0.wrapping_sub(other.0))
    }
}

impl Sub<RingTensor<u128>> for RingTensor<u128> {
    type Output = RingTensor<u128>;

    fn sub(self, other: RingTensor<u128>) -> Self::Output {
        RingTensor(self.0.wrapping_sub(other.0))
    }
}

impl Mul<RingTensor<u64>> for RingTensor<u64> {
    type Output = RingTensor<u64>;

    fn mul(self, other: RingTensor<u64>) -> Self::Output {
        RingTensor(self.0.wrapping_mul(other.0))
    }
}

impl Mul<RingTensor<u128>> for RingTensor<u128> {
    type Output = RingTensor<u128>;

    fn mul(self, other: RingTensor<u128>) -> Self::Output {
        RingTensor(self.0.wrapping_mul(other.0))
    }
}

#[derive(Clone, Debug)]
pub struct ReplicatedTensor<R> {
    shares: [[R; 2]; 3],
}

#[derive(Clone, Debug)]
pub struct PrfKey([u8; 16]);

#[derive(Clone, Debug)]
pub struct AbstractReplicatedSetup<K> {
    keys: [[K; 2]; 3],
}

#[derive(Clone, Debug)]
struct ReplicatedZeroShare<R> {
    alphas: [R; 3],
}

pub type Ring32Tensor = RingTensor<u32>;

pub type Ring64Tensor = RingTensor<u64>;

pub type Ring128Tensor = RingTensor<u128>;

pub type Replicated64Tensor = ReplicatedTensor<Ring64Tensor>;

pub type Replicated128Tensor = ReplicatedTensor<Ring128Tensor>;

pub type ReplicatedSetup = AbstractReplicatedSetup<PrfKey>;

macro_rules! modelled {
    ($t:ident, $plc:ty, () -> $u:ty, $op:ident) => {

        impl NullaryKernelCheck<ConcreteContext, $plc, $u> for $op {
            fn check(ctx: &ConcreteContext, plc: &$plc) -> $u {
                // NOTE we shouldn't do anything here, the kernel call is simply to check
                
                // TODO not sure whether to add `unimplemented!`. it might be better to
                // simply make sure the Check traits are private.
                <Self as NullaryKernel<ConcreteContext, $plc, $u>>::kernel(ctx, plc)
            }
        }

        impl $t::<ConcreteContext, $u> for $plc {
            fn apply(&self, ctx: &ConcreteContext) -> $u {
                let op = $op {
                    plc: self.clone().into(),
                };
                ctx.execute(
                    op.into(),
                    vec![]
                )
                .try_into()
                .unwrap()
            }
        }

        impl $t::<SymbolicContext, <$u as KnownType>::Symbolic> for $plc {
            fn apply(&self, ctx: &SymbolicContext) -> <$u as KnownType>::Symbolic {
                let op = $op {
                    plc: self.clone().into(),
                };
                ctx.execute(
                    op.into(),
                    vec![]
                )
                .try_into()
                .unwrap()
            }
        }

    };

    ($t:ident, $plc:ty, ($t0:ty) -> $u:ty, $op:ident) => {
        
        impl UnaryKernelCheck<ConcreteContext, $plc, $t0, $u> for $op {
            fn check(ctx: &ConcreteContext, plc: &$plc, x0: $t0) -> $u {
                // NOTE we shouldn't do anything here, the kernel call is simply to check
                
                // TODO not sure whether to add `unimplemented!`. it might be better to
                // simply make sure the Check traits are private.
                <Self as UnaryKernel<ConcreteContext, $plc, $t0, $u>>::kernel(ctx, plc, x0)
            }
        }

        impl $t::<ConcreteContext, $t0> for $plc {
            type Output = $u;

            fn apply(&self, ctx: &ConcreteContext, x0: &$t0) -> Self::Output {
                let op = $op {
                    lhs: <$t0>::TY,
                    plc: self.clone().into(),
                };
                ctx.execute(
                    op.into(),
                    vec![x0.clone().into()]
                )
                .try_into()
                .unwrap()
            }
        }

        impl $t::<SymbolicContext, <$t0 as KnownType>::Symbolic> for $plc {
            type Output = <$u as KnownType>::Symbolic;

            fn apply(&self, ctx: &SymbolicContext, x0: &<$t0 as KnownType>::Symbolic) -> Self::Output {
                let op = $op {
                    lhs: <$t0>::TY,
                    plc: self.clone().into(),
                };
                ctx.execute(
                    op.into(),
                    vec![x0.clone().into()]
                )
                .try_into()
                .unwrap()
            }
        }

    };

    ($t:ident, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, $op:ident) => {

        impl BinaryKernelCheck<ConcreteContext, $plc, $t0, $t1, $u> for $op {
            fn check(ctx: &ConcreteContext, plc: &$plc, x0: $t0, x1: $t1) -> $u {
                // NOTE we shouldn't do anything here, the kernel call is simply to check
                
                // TODO not sure whether to add `unimplemented!`. it might be better to
                // simply make sure the Check traits are private.
                <Self as BinaryKernel<ConcreteContext, $plc, $t0, $t1, $u>>::kernel(ctx, plc, x0, x1)
            }
        }

        impl $t::<ConcreteContext, $t0, $t1> for $plc {
            type Output = $u;

            fn apply(&self, ctx: &ConcreteContext, x0: &$t0, x1: &$t1) -> Self::Output {
                let op = $op {
                    lhs: <$t0>::TY,
                    rhs: <$t1>::TY,
                    plc: self.clone().into(),
                };
                ctx.execute(
                    op.into(),
                    vec![x0.clone().into(), x1.clone().into()]
                )
                .try_into()
                .unwrap()
            }
        }

        impl $t::<SymbolicContext, <$t0 as KnownType>::Symbolic, <$t1 as KnownType>::Symbolic> for $plc {
            type Output = <$u as KnownType>::Symbolic;

            fn apply(&self, ctx: &SymbolicContext, x0: &<$t0 as KnownType>::Symbolic, x1: &<$t1 as KnownType>::Symbolic) -> Self::Output {
                let op = $op {
                    lhs: <$t0>::TY,
                    rhs: <$t1>::TY,
                    plc: self.clone().into(),
                };
                ctx.execute(
                    op.into(),
                    vec![x0.clone().into(), x1.clone().into()]
                )
                .try_into()
                .unwrap()
            }
        }

    };
}

trait PlacementAdd<C: Context, T, U> {
    type Output;

    fn apply(&self, ctx: &C, x: &T, y: &U) -> Self::Output;

    fn add(&self, ctx: &C, x: &T, y: &U) -> Self::Output {
        self.apply(ctx, x, y)
    }
}

// NOTE uncomment the next line to see the kernel check system in action
// modelled!(PlacementAdd, HostPlacement, (Ring32Tensor, Ring32Tensor) -> Ring32Tensor, RingAddOp);
modelled!(PlacementAdd, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingAddOp);
modelled!(PlacementAdd, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingAddOp);

modelled!(PlacementAdd, ReplicatedPlacement, (Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepAddOp);
modelled!(PlacementAdd, ReplicatedPlacement, (Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepAddOp);

trait PlacementSub<C: Context, T, U> {
    type Output;

    fn apply(&self, ctx: &C, x: &T, y: &U) -> Self::Output;

    fn sub(&self, ctx: &C, x: &T, y: &U) -> Self::Output {
        self.apply(ctx, x, y)
    }
}

modelled!(PlacementSub, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingSubOp);
modelled!(PlacementSub, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingSubOp);

trait PlacementMul<C: Context, T, U> {
    type Output;

    fn apply(&self, ctx: &C, x: &T, y: &U) -> Self::Output;

    fn mul(&self, ctx: &C, x: &T, y: &U) -> Self::Output {
        self.apply(ctx, x, y)
    }
}

modelled!(PlacementMul, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingMulOp);
modelled!(PlacementMul, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingMulOp);
// modelled_op!(RepMulOp, PlacementMul, ReplicatedPlacement, ReplicatedSetup, Replicated64Tensor, Replicated64Tensor, Replicated64Tensor);
// modelled_op!(RepMulOp, PlacementMul, ReplicatedPlacement, ReplicatedSetup, Replicated128Tensor, Replicated128Tensor, Replicated128Tensor);

trait PlacementShare<C: Context, T> {
    type Output;

    fn apply(&self, ctx: &C, x: &T) -> Self::Output;

    fn share(&self, ctx: &C, x: &T) -> Self::Output {
        self.apply(ctx, x)
    }
}

modelled!(PlacementShare, ReplicatedPlacement, (Ring64Tensor) -> Replicated64Tensor, RepShareOp);
modelled!(PlacementShare, ReplicatedPlacement, (Ring128Tensor) -> Replicated128Tensor, RepShareOp);

pub trait Context {
    type Value;
    fn execute(&self, op: Operator, operands: Vec<Self::Value>) -> Self::Value;
}

#[derive(Clone, Debug, Default)]
pub struct ConcreteContext {}

impl Context for ConcreteContext {
    type Value = Value;

    fn execute(&self, op: Operator, operands: Vec<Value>) -> Value {
        match op {
            Operator::PrfKeyGenOp(op) => op.compile(self)(operands).try_into().unwrap(),
            Operator::RingSampleOp(op) => op.compile(self)(operands).try_into().unwrap(),
            Operator::RingAddOp(op) => op.compile(self)(operands).try_into().unwrap(),
            Operator::RingSubOp(op) => op.compile(self)(operands).try_into().unwrap(),
            Operator::RingMulOp(op) => op.compile(self)(operands).try_into().unwrap(),
            Operator::RepSetupOp(op) => op.compile(self)(operands).try_into().unwrap(),
            Operator::RepShareOp(op) => op.compile(self)(operands).try_into().unwrap(),
            Operator::RepAddOp(op) => op.compile(self)(operands).try_into().unwrap(),
            Operator::RepMulOp(op) => op.compile(self)(operands).try_into().unwrap(),
        }
    }
}

use std::sync::{Arc, RwLock};

#[derive(Clone, Debug, Default)]
pub struct SymbolicContext {
    ops: Arc<RwLock<Vec<Operation>>>, // TODO use HashMap so we can do some consistency checks on the fly?
}

impl Context for SymbolicContext {
    type Value = SymbolicValue;

    fn execute(&self, op: Operator, operands: Vec<SymbolicValue>) -> SymbolicValue {
        match op {
            Operator::PrfKeyGenOp(op) => op.execute_symbolic(self, operands).try_into().unwrap(),
            Operator::RingSampleOp(op) => op.execute_symbolic(self, operands).try_into().unwrap(),
            Operator::RingAddOp(op) => op.execute_symbolic(self, operands).try_into().unwrap(),
            Operator::RingSubOp(op) => op.execute_symbolic(self, operands).try_into().unwrap(),
            Operator::RingMulOp(op) => op.execute_symbolic(self, operands).try_into().unwrap(),
            Operator::RepSetupOp(op) => op.execute_symbolic(self, operands).try_into().unwrap(),
            Operator::RepShareOp(op) => op.execute_symbolic(self, operands).try_into().unwrap(),
            Operator::RepAddOp(op) => op.execute_symbolic(self, operands).try_into().unwrap(),
            Operator::RepMulOp(op) => op.execute_symbolic(self, operands).try_into().unwrap(),
        }
    }
}

impl SymbolicContext {
    pub fn add_operation<'s, O: Into<Operator> + Clone>(&'s self, operator: &O, operands: &[&str]) -> String {
        let mut ops = self.ops.write().unwrap(); // TODO unwrap
        let op_name: String = format!("op_{}", ops.len());
        let op = Operation {
            name: op_name.clone(),
            operator: operator.clone().into(),
            operands: operands.iter().map(|op| op.to_string()).collect(),
        };
        ops.push(op);
        op_name
    }
}

macro_rules! kernel {

    ($op:ty, [$(($plc:ty, ($t0:ty, $t1:ty) -> $u:ty)),+], $k:expr) => {
        $(
            impl BinaryKernel<ConcreteContext, $plc, $t0, $t1, $u> for $op {
                fn kernel(ctx: &ConcreteContext, plc: &$plc, x0: $t0, x1: $t1) -> $u {
                    $k(ctx, plc, x0, x1)
                }
            }

            // TODO not always clear whether we should by ::Hybric or ::Symbolic here, seems to depend on kernel
            // impl BinaryKernel<SymbolicContext, $plc, <$t0 as KnownType>::Hybrid, <$t1 as KnownType>::Hybrid, <$u as KnownType>::Hybrid> for $op {
            //     fn kernel(ctx: &SymbolicContext, plc: &$plc, x0: <$t0 as KnownType>::Hybrid, x1: <$t1 as KnownType>::Hybrid) -> <$u as KnownType>::Hybrid {
            //         $k(ctx, plc, x0, x1)
            //     }
            // }
        )+

        impl $op {
            pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Vec<Value>) -> Value> {
                match (self.plc.ty(), self.lhs, self.rhs) {
                    $(
                        (<$plc>::TY, <$t0>::TY, <$t1>::TY) => {
                            let plc: $plc = self.plc.clone().try_into().unwrap();
                            let ctx = ctx.clone();
                            let op = self.clone();
                            Box::new(move |operands| -> Value {
                                let x0: $t0 = operands.get(0).unwrap().clone().try_into().unwrap();
                                let x1: $t1 = operands.get(1).unwrap().clone().try_into().unwrap();
                                Self::kernel(&ctx, &plc, x0, x1).into()
                            })
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }

            pub fn execute_symbolic(
                &self,
                ctx: &SymbolicContext,
                operands: Vec<SymbolicValue>,
            ) -> SymbolicValue {
                match (self.plc.ty(), self.lhs, self.rhs) {
                    $(
                        (<$plc>::TY, Symbolic::<$t0>::TY, Symbolic::<$t1>::TY) => {
                            let plc: $plc = self.plc.clone().try_into().unwrap();

                            let x0: <$t0 as KnownType>::Symbolic = operands.get(0).unwrap().clone().try_into().unwrap();
                            let x1: <$t1 as KnownType>::Symbolic = operands.get(1).unwrap().clone().try_into().unwrap();

                            match (x0, x1) {
                                (Symbolic::Concrete(x0), Symbolic::Concrete(x1)) => {
                                    Symbolic::Concrete($k(ctx, &plc, x0, x1))
                                }
                                (Symbolic::Symbolic(x0), Symbolic::Symbolic(x1)) => {
                                    let op_name = ctx.add_operation(self, &[&x0.op, &x1.op]);
                                    Symbolic::Symbolic(SymbolicHandle { op: op_name })
                                }
                                _ => unimplemented!(), // ok
                            }
                            .into()
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }
        
        }
    };

    // ($op:ty, [$(($plc:ty, $t0:ty, $t1:ty, $t2:ty)),+], $k:expr) => {
    //     impl $op {
    //         pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Value, Value, Value) -> Value> {
    //             match (self.plc.ty(), self.lhs, self.rhs) {
    //                 $(
    //                     (<$plc>::TY, <$t0>::TY, <$t1>::TY, <$t2>::TY) => {
    //                         let plc: $plc = self.plc.clone().try_into().unwrap();
    //                         let ctx = ctx.clone();
    //                         let op = self.clone();
    //                         Box::new(move |x0: Value, x1: Value, x2: Value| -> Value {
    //                             let x0: $t0 = x0.try_into().unwrap();
    //                             let x1: $t1 = x1.try_into().unwrap();
    //                             let x2: $t2 = x2.try_into().unwrap();
    //                             $k(&ctx, &plc, x0, x1, x2).into()
    //                         })
    //                     }
    //                 )+
    //                 _ => unimplemented!(), // ok
    //             }
    //         }

    //         pub fn execute_symbolic(
    //             &self,
    //             ctx: &SymbolicContext,
    //             x0: SymbolicValue,
    //             x1: SymbolicValue,
    //             x2: SymbolicValue,
    //         ) -> SymbolicValue {
    //             match (self.plc.ty(), self.lhs, self.rhs) {
    //                 $(
    //                     (<$plc>::TY, Symbolic::<$t0>::TY, Symbolic::<$t1>::TY, Symbolic::<$t2>::TY) => {
    //                         let plc: $plc = self.plc.clone().try_into().unwrap();

    //                         let x0: <$t0 as KnownType>::Symbolic = x0.try_into().unwrap();
    //                         let x1: <$t1 as KnownType>::Symbolic = x1.try_into().unwrap();
    //                         let x2: <$t2 as KnownType>::Symbolic = x2.try_into().unwrap();

    //                         match (x0, x1, x2) {
    //                             (Symbolic::Concrete(x0), Symbolic::Concrete(x1), Symbolic::Concrete(x2)) => {
    //                                 Symbolic::Concrete($k(ctx, &plc, x0, x1, x2))
    //                             }
    //                             (Symbolic::Symbolic(x0), Symbolic::Symbolic(x1), Symbolic::Symbolic(x2)) => {
    //                                 let op_name = ctx.add_binary_operation(self, &x0.op, &x1.op, &x2.op);
    //                                 Symbolic::Symbolic(SymbolicHandle { op: op_name })
    //                             }
    //                             _ => unimplemented!(), // ok
    //                         }
    //                         .into()
    //                     }
    //                 )+
    //                 _ => unimplemented!(), // ok
    //             }
    //         }
        
    //     }
    // };
}

#[derive(Clone, Debug)]
pub struct RepSetupOp {
    plc: Placement,
}

impl RepSetupOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Vec<Value>) -> Value> {
        match &self.plc {
            Placement::ReplicatedPlacement(plc) => {
                let ctx = ctx.clone();
                let plc = plc.clone();

                Box::new(move |_operands| {
                    Self::abstract_kernel(&ctx, &plc).into()
                })
            }

            _ => unimplemented!(), // ok
        }
    }

    pub fn execute_symbolic(
        &self,
        ctx: &SymbolicContext,
        operands: Vec<SymbolicValue>,
    ) -> SymbolicValue {
        match &self.plc {
            Placement::ReplicatedPlacement(plc) => {
                let plc = plc.clone();
                
                Symbolic::Concrete(Self::abstract_kernel(ctx, &plc)).into()
            }

            _ => unimplemented!(), // ok
        }
    }

    fn abstract_kernel<C: Context, K: Clone>(
        ctx: &C,
        rep: &ReplicatedPlacement,
    ) -> AbstractReplicatedSetup<K>
    where
        HostPlacement: PlacementKeyGen<C, K>,
    {
        let player0 = HostPlacement {
            player: rep.players[0].clone(),
        };
        let player1 = HostPlacement {
            player: rep.players[1].clone(),
        };
        let player2 = HostPlacement {
            player: rep.players[2].clone(),
        };

        let k0 = player0.keygen(ctx);
        let k1 = player1.keygen(ctx);
        let k2 = player2.keygen(ctx);

        AbstractReplicatedSetup {
            keys: [[k0.clone(), k1.clone()], [k1.clone(), k2.clone()], [k2.clone(), k0.clone()]],
        }
    }
}

#[derive(Clone, Debug)]
pub struct RepAddOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

kernel!{
    RepAddOp,
    [
        (ReplicatedPlacement, (Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor),
        (ReplicatedPlacement, (Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor)
    ],
    Self::abstract_kernel
}

impl RepAddOp {
    fn abstract_kernel<C: Context, R>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: ReplicatedTensor<R>,
        y: ReplicatedTensor<R>,
    ) -> ReplicatedTensor<R>
    where
        HostPlacement: PlacementAdd<C, R, R, Output = R>,
    {
        let player0 = HostPlacement {
            player: rep.players[0].clone(),
        };
        let player1 = HostPlacement {
            player: rep.players[1].clone(),
        };
        let player2 = HostPlacement {
            player: rep.players[2].clone(),
        };

        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let ReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        // this could be turned into something like `let z00 = player0.with(|x00, y00| { x00 + y00 }, x00, y00)`
        let z00 = player0.add(ctx, x00, y00);
        let z10 = player0.add(ctx, x10, y10);

        let z11 = player1.add(ctx, x11, y11);
        let z21 = player1.add(ctx, x21, y21);

        let z22 = player2.add(ctx, x22, y22);
        let z02 = player2.add(ctx, x02, y02);

        ReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
}

#[derive(Clone, Debug)]
pub struct RepMulOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

pub trait NullaryKernel<C: Context, P, Y> {
    fn kernel(ctx: &C, plc: &P) -> Y;
}

trait NullaryKernelCheck<C: Context, P, Y> {
    fn check(ctx: &C, plc: &P) -> Y;
}

pub trait UnaryKernel<C: Context, P, X0, Y> {
    fn kernel(ctx: &C, plc: &P, x0: X0) -> Y;
}

trait UnaryKernelCheck<C: Context, P, X0, Y> {
    fn check(ctx: &C, plc: &P, x0: X0) -> Y;
}

pub trait BinaryKernel<C: Context, P, X0, X1, Y> {
    fn kernel(ctx: &C, plc: &P, x0: X0, x1: X1) -> Y;
}

trait BinaryKernelCheck<C: Context, P, X0, X1, Y> {
    fn check(ctx: &C, plc: &P, x0: X0, x1: X1) -> Y;
}

pub trait TernaryKernel<C: Context, P, X0, X1, X2, Y> {
    fn kernel(ctx: &C, plc: &P, x0: X0, x1: X1, x2: X2) -> Y;
}

trait TernaryKernelCheck<C: Context, P, X0, X1, X2, Y> {
    fn check(ctx: &C, plc: &P, x0: X0, x1: X1, x2: X2) -> Y;
}

impl TernaryKernel<ConcreteContext, ReplicatedPlacement, ReplicatedSetup, Replicated64Tensor, Replicated64Tensor, Replicated64Tensor> for RepMulOp {
    fn kernel(ctx: &ConcreteContext, plc: &ReplicatedPlacement, x0: ReplicatedSetup, x1: Replicated64Tensor, x2: Replicated64Tensor) -> Replicated64Tensor {
        Self::abstract_kernel(ctx, plc, x0, x1, x2)
    }
}

impl TernaryKernel<ConcreteContext, ReplicatedPlacement, ReplicatedSetup, Replicated128Tensor, Replicated128Tensor, Replicated128Tensor> for RepMulOp {
    fn kernel(ctx: &ConcreteContext, plc: &ReplicatedPlacement, x0: ReplicatedSetup, x1: Replicated128Tensor, x2: Replicated128Tensor) -> Replicated128Tensor {
        Self::abstract_kernel(ctx, plc, x0, x1, x2)
    }
}

impl TernaryKernel<SymbolicContext, ReplicatedPlacement, <ReplicatedSetup as KnownType>::Hybrid, <Replicated64Tensor as KnownType>::Hybrid, <Replicated64Tensor as KnownType>::Hybrid, <Replicated64Tensor as KnownType>::Hybrid> for RepMulOp {
    fn kernel(ctx: &SymbolicContext, plc: &ReplicatedPlacement, x0: <ReplicatedSetup as KnownType>::Hybrid, x1: <Replicated64Tensor as KnownType>::Hybrid, x2: <Replicated64Tensor as KnownType>::Hybrid) -> <Replicated64Tensor as KnownType>::Hybrid {
        Self::abstract_kernel(ctx, plc, x0, x1, x2)
    }
}

impl TernaryKernel<SymbolicContext, ReplicatedPlacement, <ReplicatedSetup as KnownType>::Hybrid, <Replicated128Tensor as KnownType>::Hybrid, <Replicated128Tensor as KnownType>::Hybrid, <Replicated128Tensor as KnownType>::Hybrid> for RepMulOp {
    fn kernel(ctx: &SymbolicContext, plc: &ReplicatedPlacement, x0: <ReplicatedSetup as KnownType>::Hybrid, x1: <Replicated128Tensor as KnownType>::Hybrid, x2: <Replicated128Tensor as KnownType>::Hybrid) -> <Replicated128Tensor as KnownType>::Hybrid {
        Self::abstract_kernel(ctx, plc, x0, x1, x2)
    }
}

impl RepMulOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Vec<Value>) -> Value> {
        match (&self.plc, self.lhs, self.rhs) {
            (Placement::ReplicatedPlacement(plc), Replicated64Tensor::TY, Replicated64Tensor::TY) => {
                let ctx = ctx.clone();
                let plc = plc.clone();

                Box::new(move |operands| {
                    let s: ReplicatedSetup = operands.get(0).unwrap().clone().try_into().unwrap();
                    let x: Replicated64Tensor = operands.get(1).unwrap().clone().try_into().unwrap();
                    let y: Replicated64Tensor = operands.get(2).unwrap().clone().try_into().unwrap();

                    Self::kernel(&ctx, &plc, s, x, y).into()
                })
            }

            (Placement::ReplicatedPlacement(plc), Replicated128Tensor::TY, Replicated128Tensor::TY) => {
                let ctx = ctx.clone();
                let plc = plc.clone();

                Box::new(move |operands| {
                    let s: ReplicatedSetup = operands.get(0).unwrap().clone().try_into().unwrap();
                    let x: Replicated128Tensor = operands.get(1).unwrap().clone().try_into().unwrap();
                    let y: Replicated128Tensor = operands.get(2).unwrap().clone().try_into().unwrap();

                    Self::kernel(&ctx, &plc, s, x, y).into()
                })
            }

            _ => unimplemented!(), // ok
        }
    }

    pub fn execute_symbolic(
        &self,
        ctx: &SymbolicContext,
        operands: Vec<SymbolicValue>
    ) -> SymbolicValue {
        match (&self.plc, self.lhs, self.rhs) {
            (Placement::ReplicatedPlacement(plc), Symbolic::<Replicated64Tensor>::TY, Symbolic::<Replicated64Tensor>::TY) => {
                let plc = plc.clone();

                let s: <ReplicatedSetup as KnownType>::Symbolic = operands.get(0).unwrap().clone().try_into().unwrap();
                let x: <Replicated64Tensor as KnownType>::Symbolic = operands.get(1).unwrap().clone().try_into().unwrap();
                let y: <Replicated64Tensor as KnownType>::Symbolic = operands.get(2).unwrap().clone().try_into().unwrap();

                match (s, x, y) {
                    (Symbolic::Concrete(s), Symbolic::Concrete(x), Symbolic::Concrete(y)) => {
                        Symbolic::Concrete(Self::kernel(ctx, &plc, s, x, y))
                    }
                    (Symbolic::Symbolic(s), Symbolic::Symbolic(x), Symbolic::Symbolic(y)) => {
                        let op_name = ctx.add_operation(self, &[&s.op, &x.op, &y.op]);
                        Symbolic::Symbolic(SymbolicHandle { op: op_name })
                    }
                    _ => unimplemented!(), // ok
                }
                .into()
            }

            (Placement::ReplicatedPlacement(plc), Symbolic::<Replicated128Tensor>::TY, Symbolic::<Replicated128Tensor>::TY) => {
                let plc = plc.clone();

                let s: <ReplicatedSetup as KnownType>::Symbolic = operands.get(0).unwrap().clone().try_into().unwrap();
                let x: <Replicated128Tensor as KnownType>::Symbolic = operands.get(1).unwrap().clone().try_into().unwrap();
                let y: <Replicated128Tensor as KnownType>::Symbolic = operands.get(2).unwrap().clone().try_into().unwrap();

                match (s, x, y) {
                    (Symbolic::Concrete(s), Symbolic::Concrete(x), Symbolic::Concrete(y)) => {
                        Symbolic::Concrete(Self::kernel(ctx, &plc, s, x, y))
                    }
                    (Symbolic::Symbolic(s), Symbolic::Symbolic(x), Symbolic::Symbolic(y)) => {
                        let op_name = ctx.add_operation(self, &[&s.op, &x.op, &y.op]);
                        Symbolic::Symbolic(SymbolicHandle { op: op_name })
                    }
                    _ => unimplemented!(), // ok
                }
                .into()
            }

            foo => {
                println!("{:?}", foo);
                unimplemented!() // ok
            }
        }
    }

    fn abstract_kernel<C: Context, R, K>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        s: AbstractReplicatedSetup<K>,
        x: ReplicatedTensor<R>,
        y: ReplicatedTensor<R>,
    ) -> ReplicatedTensor<R>
    where
        R: Clone + Into<C::Value> + TryFrom<C::Value>,
        HostPlacement: PlacementSample<C, R>,
        HostPlacement: PlacementAdd<C, R, R, Output = R>,
        HostPlacement: PlacementMul<C, R, R, Output = R>,
        ReplicatedPlacement: PlacementZeroShare<C, K, R>,
    {
        let player0 = HostPlacement {
            player: rep.players[0].clone(),
        };
        let player1 = HostPlacement {
            player: rep.players[1].clone(),
        };
        let player2 = HostPlacement {
            player: rep.players[2].clone(),
        };

        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let ReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        // TODO improve syntax
        let t0 = player0.add(ctx, &player0.add(ctx, &player0.mul(ctx, x00, y00), &player0.mul(ctx, x00, y10)), &player0.mul(ctx, x10, y00));
        let t1 = player1.add(ctx, &player1.add(ctx, &player1.mul(ctx, x11, y11), &player1.mul(ctx, x11, y21)), &player1.mul(ctx, x21, y11));
        let t2 = player2.add(ctx, &player2.add(ctx, &player2.mul(ctx, x22, y22), &player2.mul(ctx, x22, y02)), &player2.mul(ctx, x02, y22));

        let ReplicatedZeroShare {
            alphas: [a0, a1, a2]
        } = rep.zero_share(ctx, &s);

        let z0 = player0.add(ctx, &t0, &a0);
        let z1 = player1.add(ctx, &t1, &a1);
        let z2 = player2.add(ctx, &t2, &a2);

        ReplicatedTensor {
            shares: [[z0.clone(), z1.clone()], [z1.clone(), z2.clone()], [z2.clone(), z0.clone()]],
        }
    }
}

// kernel!{
//     RepMulOp,
//     [
//         (ReplicatedPlacement, ReplicatedSetup, Replicated64Tensor, Replicated64Tensor),
//         (ReplicatedPlacement, ReplicatedSetup, Replicated128Tensor, Replicated128Tensor)
//     ],
//     Self::abstract_kernel
// }

trait PlacementZeroShare<C: Context, K, R> {
    fn zero_share(&self, ctx: &C, setup: &AbstractReplicatedSetup<K>) -> ReplicatedZeroShare<R>;
}

// NOTE this is an un-modelled operation (as opposed to the modelled operation that have a representation in computations)
// TODO should we have a macro for this as well?
impl<C: Context, K, R> PlacementZeroShare<C, K, R> for ReplicatedPlacement
where
    HostPlacement: PlacementSample<C, R>,
    HostPlacement: PlacementSub<C, R, R, Output = R>,
{
    fn zero_share(&self, ctx: &C, s: &AbstractReplicatedSetup<K>) -> ReplicatedZeroShare<R> {
        let AbstractReplicatedSetup { keys: [[k00, k10], [k11, k21], [k22, k02]] } = s;

        let player0 = HostPlacement {
            player: self.players[0].clone(),
        };
        let player1 = HostPlacement {
            player: self.players[1].clone(),
        };
        let player2 = HostPlacement {
            player: self.players[2].clone(),
        };

        // TODO use keys when sampling below!

        let r00 = player0.sample(ctx);
        let r10 = player0.sample(ctx);
        let alpha0 = player0.sub(ctx, &r00, &r10);

        let r11 = player1.sample(ctx);
        let r21 = player1.sample(ctx);
        let alpha1 = player0.sub(ctx, &r11, &r21);

        let r22 = player2.sample(ctx);
        let r02 = player2.sample(ctx);
        let alpha2 = player0.sub(ctx, &r22, &r02);

        ReplicatedZeroShare {
            alphas: [alpha0, alpha1, alpha2],
        }
    }
}



// struct Placed<T> {
//     plc: Placement,
//     val: T,
// }

// impl<T> Add<Placed<T>> for Placed<T>
// where
//     T: Add<T, Output=T>,
// {
//     type Output = Placed<T>;

//     fn add(self, other: Placed<T>) -> Placed<T> {
//         // assert_eq!(self.plc, other.plc); // TODO
//         self.plc.add()
//     }
// }

#[derive(Clone, Debug)]
pub struct RepShareOp {
    lhs: Ty,
    plc: Placement,
}

impl UnaryKernel<ConcreteContext, ReplicatedPlacement, Ring64Tensor, Replicated64Tensor> for RepShareOp {
    fn kernel(ctx: &ConcreteContext, plc: &ReplicatedPlacement, x: Ring64Tensor) -> Replicated64Tensor {
        Self::abstract_kernel(ctx, plc, x)
    }
}

impl UnaryKernel<ConcreteContext, ReplicatedPlacement, Ring128Tensor, Replicated128Tensor> for RepShareOp {
    fn kernel(ctx: &ConcreteContext, plc: &ReplicatedPlacement, x: Ring128Tensor) -> Replicated128Tensor {
        Self::abstract_kernel(ctx, plc, x)
    }
}

impl UnaryKernel<SymbolicContext, ReplicatedPlacement, Symbolic<Ring64Tensor>, ReplicatedTensor<Symbolic<Ring64Tensor>>> for RepShareOp {
    fn kernel(ctx: &SymbolicContext, plc: &ReplicatedPlacement, x: Symbolic<Ring64Tensor>) -> ReplicatedTensor<Symbolic<Ring64Tensor>> {
        Self::abstract_kernel(ctx, plc, x)
    }
}

impl UnaryKernel<SymbolicContext, ReplicatedPlacement, Symbolic<Ring128Tensor>, ReplicatedTensor<Symbolic<Ring128Tensor>>> for RepShareOp {
    fn kernel(ctx: &SymbolicContext, plc: &ReplicatedPlacement, x: Symbolic<Ring128Tensor>) -> ReplicatedTensor<Symbolic<Ring128Tensor>> {
        Self::abstract_kernel(ctx, plc, x)
    }
}

impl RepShareOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Vec<Value>) -> Value> {
        match (&self.plc, self.lhs) {
            (Placement::ReplicatedPlacement(rep_plc), Ring64Tensor::TY) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                Box::new(move |operands: Vec<Value>| {
                    let x: Ring64Tensor = operands.get(0).unwrap().clone().try_into().unwrap();
                    <Self as UnaryKernel<ConcreteContext, ReplicatedPlacement, Ring64Tensor, Replicated64Tensor>>::kernel(&ctx, &rep_plc, x).into()
                })
            }

            (Placement::ReplicatedPlacement(rep_plc), Ring128Tensor::TY) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                Box::new(move |operands: Vec<Value>| {
                    let x: Ring128Tensor = operands.get(0).unwrap().clone().try_into().unwrap();
                    <Self as UnaryKernel<ConcreteContext, ReplicatedPlacement, Ring128Tensor, Replicated128Tensor>>::kernel(&ctx, &rep_plc, x).into()
                })
            }

            _ => unimplemented!(), // ok
        }
    }

    pub fn execute_symbolic(&self, ctx: &SymbolicContext, operands: Vec<SymbolicValue>) -> SymbolicValue {
        match (&self.plc, self.lhs) {
            (Placement::ReplicatedPlacement(rep_plc), Ring64Tensor::TY) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                let x: Symbolic<Ring64Tensor> = operands.get(0).unwrap().clone().try_into().unwrap();
                Symbolic::Concrete(Self::kernel(&ctx, &rep_plc, x)).into()
            }

            (Placement::ReplicatedPlacement(rep_plc), Ring128Tensor::TY) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                let x: Symbolic<Ring128Tensor> = operands.get(0).unwrap().clone().try_into().unwrap();
                Symbolic::Concrete(Self::kernel(&ctx, &rep_plc, x)).into()
            }

            _ => unimplemented!(), // ok
        }
    }

    fn abstract_kernel<C: Context, R: Clone>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: R,
    ) -> ReplicatedTensor<R>
    where
        R: Into<C::Value> + TryFrom<C::Value>,
        HostPlacement: PlacementSample<C, R>,
        HostPlacement: PlacementAdd<C, R, R, Output = R>,
        HostPlacement: PlacementSub<C, R, R, Output = R>,
    {
        let player0 = HostPlacement {
            player: rep.players[0].clone(),
        };
        let player1 = HostPlacement {
            player: rep.players[1].clone(),
        };
        let player2 = HostPlacement {
            player: rep.players[2].clone(),
        };

        // TODO we should not use player0 here, but rather the placement of `x` (which is currently not implemented)
        let x0 = player0.sample(ctx);
        let x1 = player0.sample(ctx);
        let x2 = player0.sub(ctx, &x, &player0.add(ctx, &x0, &x1));

        ReplicatedTensor {
            shares: [
                [x0.clone(), x1.clone()],
                [x1.clone(), x2.clone()],
                [x2.clone(), x0.clone()],
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub struct RingAddOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

impl RingAddOp {
    fn abstract_kernel<C: Context, T>(_ctx: &C, _plc: &HostPlacement, x: RingTensor<T>, y: RingTensor<T>) -> RingTensor<T>
    where
        RingTensor<T>: Add<RingTensor<T>, Output = RingTensor<T>>,
    {
        x + y
    }
}

kernel!{
    RingAddOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor)
    ],
    Self::abstract_kernel
}

#[derive(Clone, Debug)]
pub struct RingSubOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

impl RingSubOp {
    fn abstract_kernel<C: Context, T>(_ctx: &C, _plc: &HostPlacement, x: RingTensor<T>, y: RingTensor<T>) -> RingTensor<T>
    where
        RingTensor<T>: Sub<RingTensor<T>, Output = RingTensor<T>>,
    {
        x - y
    }
}

kernel!{
    RingSubOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor)
    ],
    Self::abstract_kernel
}

#[derive(Clone, Debug)]
pub struct RingMulOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

impl RingMulOp {
    fn abstract_kernel<C: Context, T>(_ctx: &C, _plc: &HostPlacement, x: RingTensor<T>, y: RingTensor<T>) -> RingTensor<T>
    where
        RingTensor<T>: Mul<RingTensor<T>, Output = RingTensor<T>>,
    {
        x * y
    }
}

kernel!{
    RingMulOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor)
    ],
    Self::abstract_kernel
}

trait PlacementKeyGen<C: Context, K> {
    fn apply(&self, ctx: &C) -> K;

    fn keygen(&self, ctx: &C) -> K {
        self.apply(ctx)
    }
}

#[derive(Clone, Debug)]
pub struct PrfKeyGenOp {
    plc: Placement,
}

modelled!(PlacementKeyGen, HostPlacement, () -> PrfKey, PrfKeyGenOp);

// kernel!{
//     PrfKeyGenOp,
//     [
//         (HostPlacement),
//         (HostPlacement)
//     ],
//     Self::kernel
// }

impl NullaryKernel<ConcreteContext, HostPlacement, PrfKey> for PrfKeyGenOp {
    fn kernel(ctx: &ConcreteContext, plc: &HostPlacement) -> PrfKey {
        Self::abstract_kernel(ctx, plc)
    }
}

impl PrfKeyGenOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Vec<Value>) -> Value> {
        match &self.plc {
            Placement::HostPlacement(plc) => {
                let plc = plc.clone();
                let ctx = ctx.clone();
                let op = self.clone();

                Box::new(move |_operands| -> Value {
                    <Self as NullaryKernel<ConcreteContext, HostPlacement, PrfKey>>::kernel(&ctx, &plc).into()
                })
            }
            _ => unimplemented!(), // ok
        }
    }

    pub fn execute_symbolic(
        &self,
        ctx: &SymbolicContext,
        _operands: Vec<SymbolicValue>,
    ) -> SymbolicValue {
        match &self.plc {
            Placement::HostPlacement(plc) => {
                let op_name = ctx.add_operation(self, &[]);
                Symbolic::<PrfKey>::Symbolic(SymbolicHandle { op: op_name })
                .into()
            }
            _ => unimplemented!(), // ok
        }
    }
}

impl PrfKeyGenOp {
    fn abstract_kernel(ctx: &ConcreteContext, plc: &HostPlacement) -> PrfKey {
        PrfKey([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    }
}

trait PlacementSample<C: Context, O> {
    fn sample(&self, ctx: &C) -> O;
}

impl PlacementSample<ConcreteContext, Ring64Tensor> for HostPlacement {
    fn sample(&self, ctx: &ConcreteContext) -> Ring64Tensor {
        ctx.execute(
            RingSampleOp {
                ty: Ring64Tensor::TY,
                plc: Placement::HostPlacement(self.clone()),
            }
            .into(),
            vec![]
        )
        .try_into()
        .unwrap()
    }
}

impl PlacementSample<ConcreteContext, Ring128Tensor> for HostPlacement {
    fn sample(&self, ctx: &ConcreteContext) -> Ring128Tensor {
        ctx.execute(
            RingSampleOp {
                ty: Ring128Tensor::TY,
                plc: Placement::HostPlacement(self.clone()),
            }
            .into(),
            vec![]
        )
        .try_into()
        .unwrap()
    }
}

impl PlacementSample<SymbolicContext, Symbolic<Ring64Tensor>> for HostPlacement {
    fn sample(&self, ctx: &SymbolicContext) -> Symbolic<Ring64Tensor> {
        ctx.execute(
            RingSampleOp {
                ty: Ty::Ring64TensorTy,
                plc: Placement::HostPlacement(self.clone()),
            }
            .into(),
            vec![]
        )
        .try_into()
        .unwrap()
    }
}

impl PlacementSample<SymbolicContext, Symbolic<Ring128Tensor>> for HostPlacement {
    fn sample(&self, ctx: &SymbolicContext) -> Symbolic<Ring128Tensor> {
        ctx.execute(
            RingSampleOp {
                ty: Ty::Ring128TensorTy,
                plc: Placement::HostPlacement(self.clone()),
            }
            .into(),
            vec![]
        )
        .try_into()
        .unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct RingSampleOp {
    ty: Ty,
    plc: Placement,
}

impl RingSampleOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Vec<Value>) -> Value> {
        match (&self.plc, &self.ty) {
            (Placement::HostPlacement(_), Ty::Ring64TensorTy) => {
                Box::new(move |_operands| {
                    let y: Ring64Tensor = RingTensor(987654321);
                    y.into()
                })
            }
            (Placement::HostPlacement(_), Ty::Ring128TensorTy) => {
                Box::new(move |_operands| {
                    let y: Ring128Tensor = RingTensor(987654321);
                    y.into()
                })
            }
            _ => unimplemented!(),
        }
    }

    // TODO could we derive this from the return type of the closure returned by `compile`?
    // not sure this will work, seems like we need a Ty instead, which comes as part of
    // type checking.
    pub fn execute_symbolic(&self, ctx: &SymbolicContext, _operands: Vec<SymbolicValue>) -> SymbolicValue {
        match (&self.plc, &self.ty) {
            (Placement::HostPlacement(_), Ty::Ring64TensorTy) => {
                let op_name = ctx.add_operation(self, &[]);
                SymbolicValue::Ring64Tensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                }))
            }
            (Placement::HostPlacement(_), Ty::Ring128TensorTy) => {
                let op_name = ctx.add_operation(self, &[]);
                SymbolicValue::Ring128Tensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                }))
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Debug)]
struct ConstantOp {
    // val: Value,
    plc: Placement,
}

impl ConstantOp {
    //     pub fn compile(&self) -> Box<dyn Fn() -> Value> {
    //         let val = self.val.clone();

    //         match (&self.plc) {

    //             (Placement::Host(_)) => {
    //                 Box::new(move || -> Value {
    //                     val.clone()
    //                 })
    //             }

    //             _ => unimplemented!()
    //         }
    //     }

    //     fn abstract_kernel<T>(x: RingTensor<T>, y: RingTensor<T>) -> RingTensor<T>
    //     where
    //         RingTensor<T>: Add<RingTensor<T>, Output=RingTensor<T>>,
    //     {
    //         x + y
    //     }
}

// enum HostPlacementInst {
//     Symbolic,
//     Concrete,
// }

// impl HostPlacementInst {
//     fn placement(&self) -> Placement {
//         Placement::Host
//     }

//     fn sample_uniform(&self) -> RingTensor {
//         let op = RingSampleOp {
//             plc: self.placement(),
//         };

//         match self {
//             HostPlacementInst::Symbolic => {
//                 RingTensor::Symbolic(RingSampleOp{
//                     plc: self.placement(),
//                 })
//             }
//             HostPlacementInst::Concrete => {
//                 RingTensor::Concrete(ConcreteRingTensor(5))
//             }
//         }
//     }
// }

// fn rep_share(plc: HostPlacement, x: RingTensor) -> ReplicatedTensor {

// }

#[test]
fn test_rep_add() {
    // let x = ReplicatedTensor{
    //     shares: [
    //         [Ring64Tensor::Concrete(RingTensor(1)), Ring64Tensor::Concrete(RingTensor(2))],
    //         [Ring64Tensor::Concrete(RingTensor(2)), Ring64Tensor::Concrete(RingTensor(3))],
    //         [Ring64Tensor::Concrete(RingTensor(3)), Ring64Tensor::Concrete(RingTensor(1))],
    //     ]
    // };

    // let y = ReplicatedTensor{
    //     shares: [
    //         [Ring64Tensor::Concrete(RingTensor(1)), Ring64Tensor::Concrete(RingTensor(2))],
    //         [Ring64Tensor::Concrete(RingTensor(2)), Ring64Tensor::Concrete(RingTensor(3))],
    //         [Ring64Tensor::Concrete(RingTensor(3)), Ring64Tensor::Concrete(RingTensor(1))],
    //     ]
    // };

    // let x = ReplicatedTensor{
    //     shares: [
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //     ]
    // };

    // let y = ReplicatedTensor{
    //     shares: [
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //     ]
    // };

    let x = ReplicatedTensor {
        shares: [
            [RingTensor(1_u128), RingTensor(2_u128)],
            [RingTensor(2_u128), RingTensor(3_u128)],
            [RingTensor(3_u128), RingTensor(1_u128)],
        ],
    };

    let y = ReplicatedTensor {
        shares: [
            [RingTensor(1_u128), RingTensor(2_u128)],
            [RingTensor(2_u128), RingTensor(3_u128)],
            [RingTensor(3_u128), RingTensor(1_u128)],
        ],
    };

    // let x = ReplicatedTensor{
    //     shares: [
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //     ]
    // };

    // let y = ReplicatedTensor{
    //     shares: [
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //     ]
    // };

    let ctx = ConcreteContext::default();
    let rep_plc = ReplicatedPlacement {
        players: ["alice".into(), "bob".into(), "carole".into()],
    };
    let z: ReplicatedTensor<_> = rep_plc.add(&ctx, &x, &y);
    // println!("{:?}", z);

    let graph: Vec<Operator> = vec![];
    let ctx = SymbolicContext::default();
    let host_plc = HostPlacement {
        player: "alice".into(),
    };
    // let r = Ring64Tensor::Concrete(RingTensor(2));
    // let s = Ring64Tensor::Concrete(RingTensor(1));
    // let t = host_plc.sub(&ctx, &r, &s);
    // println!("{:?}", t);
    let r: Symbolic<Ring128Tensor> = host_plc.sample(&ctx);
    println!("{:?}", r);

    let a: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> =
        Symbolic::Concrete(ReplicatedTensor {
            shares: [
                [
                    SymbolicHandle { op: "a00".into() }.into(),
                    SymbolicHandle { op: "a10".into() }.into(),
                ],
                [
                    SymbolicHandle { op: "a11".into() }.into(),
                    SymbolicHandle { op: "a21".into() }.into(),
                ],
                [
                    SymbolicHandle { op: "a22".into() }.into(),
                    SymbolicHandle { op: "a02".into() }.into(),
                ],
            ],
        });
    let b: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> =
        Symbolic::Concrete(ReplicatedTensor {
            shares: [
                [
                    SymbolicHandle { op: "b00".into() }.into(),
                    SymbolicHandle { op: "b10".into() }.into(),
                ],
                [
                    SymbolicHandle { op: "b11".into() }.into(),
                    SymbolicHandle { op: "b21".into() }.into(),
                ],
                [
                    SymbolicHandle { op: "b22".into() }.into(),
                    SymbolicHandle { op: "b02".into() }.into(),
                ],
            ],
        });
    let c = rep_plc.add(&ctx, &a, &b);
    // println!("{:?}", c);
}

#[test]
fn test_rep_share() {
    let alice_plc = HostPlacement {
        player: "alice".into(),
    };
    let bob_plc = HostPlacement {
        player: "bob".into(),
    };
    let rep_plc = ReplicatedPlacement {
        players: ["alice".into(), "bob".into(), "carole".into()],
    };

    {
        let ctx = ConcreteContext::default();
        let x: Ring64Tensor = alice_plc.sample(&ctx);
        let y: Ring64Tensor = bob_plc.sample(&ctx);
        let xe = rep_plc.share(&ctx, &x);
        let ye = rep_plc.share(&ctx, &y);
        let ze = rep_plc.add(&ctx, &xe, &ye);
        println!("CONCRETE {:?}", ze);
    }

    {
        let ctx = SymbolicContext::default();
        let x: Symbolic<Ring64Tensor> = alice_plc.sample(&ctx);
        let y: Symbolic<Ring64Tensor> = bob_plc.sample(&ctx);
        let xe = rep_plc.share(&ctx, &x);
        let ye = rep_plc.share(&ctx, &y);
        let ze = rep_plc.add(&ctx, &xe, &ye);
        println!("SYMBOLIC {:?}", ze);

        // let ops = ctx.ops.read().unwrap();
        // for op in ops.iter() {
        //     println!("  {:?}", op);
        // }
    }

    // let xe = rep_plc.share(&ctx, &x);

    // let a: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> = Symbolic::Concrete(ReplicatedTensor {
    //     shares: [
    //         [Symbolic::Symbolic{ty: Ty::Ring64TensorTy}, Symbolic::Symbolic{ty: Ty::Ring64TensorTy}],
    //         [Symbolic::Symbolic{ty: Ty::Ring64TensorTy}, Symbolic::Symbolic{ty: Ty::Ring64TensorTy}],
    //         [Symbolic::Symbolic{ty: Ty::Ring64TensorTy}, Symbolic::Symbolic{ty: Ty::Ring64TensorTy}],
    //     ]
    // });
    // let b: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> = Symbolic::Concrete(ReplicatedTensor {
    //     shares: [
    //         [Symbolic::Symbolic{ty: Ty::Ring64TensorTy}, Symbolic::Symbolic{ty: Ty::Ring64TensorTy}],
    //         [Symbolic::Symbolic{ty: Ty::Ring64TensorTy}, Symbolic::Symbolic{ty: Ty::Ring64TensorTy}],
    //         [Symbolic::Symbolic{ty: Ty::Ring64TensorTy}, Symbolic::Symbolic{ty: Ty::Ring64TensorTy}],
    //     ]
    // });
    // let op = RepAddOp {
    //     lhs: Ty::Replicated64TensorTy,
    //     rhs: Ty::Replicated64TensorTy,
    //     plc: Placement::Replicated(rep_plc.clone()),
    // };
    // let c = ctx.execute_binary(op.into(), a.into(), b.into());
    // println!("{:?}", c);
    // let c = rep_plc.add(&ctx, &a, &b);
}

#[test]
fn test_rep_exec() {
    use std::collections::HashMap;

    let alice_plc = HostPlacement {
        player: "alice".into(),
    };
    let bob_plc = HostPlacement {
        player: "bob".into(),
    };
    let rep_plc = ReplicatedPlacement {
        players: ["alice".into(), "bob".into(), "carole".into()],
    };

    let ops: Vec<Operation> = vec![
        Operation {
            name: "x".into(),
            operator: RingSampleOp {
                ty: Ring128Tensor::TY,
                plc: alice_plc.clone().into(),
            }
            .into(),
            operands: vec![],
        },
        Operation {
            name: "xe".into(),
            operator: RepShareOp {
                lhs: Ring128Tensor::TY,
                plc: rep_plc.clone().into(),
            }
            .into(),
            operands: vec!["x".into()],
        },
        Operation {
            name: "y".into(),
            operator: RingSampleOp {
                ty: Ring128Tensor::TY,
                plc: bob_plc.clone().into(),
            }
            .into(),
            operands: vec![],
        },
        Operation {
            name: "ye".into(),
            operator: RepShareOp {
                lhs: Ring128Tensor::TY,
                plc: rep_plc.clone().into(),
            }
            .into(),
            operands: vec!["y".into()],
        },
        Operation {
            name: "s".into(),
            operator: RepSetupOp {
                plc: rep_plc.clone().into(),
            }
            .into(),
            operands: vec![],
        },
        Operation {
            name: "ze".into(),
            operator: RepMulOp {
                lhs: Replicated128Tensor::TY,
                rhs: Replicated128Tensor::TY,
                plc: rep_plc.clone().into(),
            }
            .into(),
            operands: vec!["s".into(), "xe".into(), "ye".into()],
        },
        Operation {
            name: "ve".into(),
            operator: RepMulOp {
                lhs: Replicated128Tensor::TY,
                rhs: Replicated128Tensor::TY,
                plc: rep_plc.clone().into(),
            }
            .into(),
            operands: vec!["s".into(), "xe".into(), "ye".into()],
        },
    ];

    let ctx = SymbolicContext::default();
    // let ctx = ConcreteContext::default();

    let mut env: HashMap<String, SymbolicValue> = HashMap::default();
    for op in ops {
        let operator = op.operator;
        let operands = op.operands.iter().map(|input_name| env.get(input_name).unwrap().clone()).collect();
        let res = ctx.execute(operator, operands);
        env.insert(op.name, res);
    }

    // println!("{:?}", env);

    let ops = ctx.ops.read().unwrap();
    for op in ops.iter() {
        println!("  {:?}", op);
    }

    // let comp = r#"

    // "#.try_into().unwrap();

    // let exec = SymbolicExecutor;
    // exec.eval(comp);
}
