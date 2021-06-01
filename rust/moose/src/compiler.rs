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
            Placement::HostPlacement(_) => PlacementTy::HostTy,
            Placement::ReplicatedPlacement(_) => PlacementTy::ReplicatedTy,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlacementTy {
    HostTy,
    ReplicatedTy,
}

trait KnownPlacement {
    const TY: PlacementTy;
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

#[derive(Clone, Debug)]
struct HostPlacement {
    player: String,
}
#[derive(Clone, Debug)]
struct ReplicatedPlacement {
    players: [String; 3],
}

// enum PlacementInstantiation {
//     Host,
//     Rep,
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ty {
    // HostFixedTy,
    Ring64TensorTy,
    Ring128TensorTy,
    Replicated64TensorTy,
    Replicated128TensorTy,
    ReplicatedSetupTy,
}

trait KnownType {
    type Symbolic;
    const TY: Ty;
}

impl KnownType for Ring64Tensor {
    type Symbolic = Symbolic<Ring64Tensor>;
    const TY: Ty = Ty::Ring64TensorTy;
}

impl KnownType for Ring128Tensor {
    type Symbolic = Symbolic<Ring128Tensor>;
    const TY: Ty = Ty::Ring128TensorTy;
}

impl KnownType for Replicated64Tensor {
    type Symbolic = Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>>;
    const TY: Ty = Ty::Replicated64TensorTy;
}

impl KnownType for Replicated128Tensor {
    type Symbolic = Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>>;
    const TY: Ty = Ty::Replicated128TensorTy;
}

impl KnownType for ReplicatedSetup<PrfKey> {
    type Symbolic = Symbolic<ReplicatedSetup<Symbolic<PrfKey>>>;
    const TY: Ty = Ty::ReplicatedSetupTy;
}

impl<T: KnownType> KnownType for Symbolic<T> {
    type Symbolic = Self;
    const TY: Ty = T::TY;
}

#[derive(Clone, Debug)]
enum Value {
    // HostFixed(HostFixed),
    // RepFixed(RepFixed),
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    Replicated64Tensor(Replicated64Tensor),
    Replicated128Tensor(Replicated128Tensor),
    ReplicatedSetup(ReplicatedSetup<PrfKey>),
}

#[derive(Clone, Debug)]
enum SymbolicValue {
    Ring64Tensor(<Ring64Tensor as KnownType>::Symbolic),
    Ring128Tensor(<Ring128Tensor as KnownType>::Symbolic),
    Replicated64Tensor(<Replicated64Tensor as KnownType>::Symbolic),
    Replicated128Tensor(<Replicated128Tensor as KnownType>::Symbolic),
    ReplicatedSetup(<ReplicatedSetup<PrfKey> as KnownType>::Symbolic),
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

value!(Ring64Tensor);
value!(Ring128Tensor);
value!(Replicated64Tensor);
value!(Replicated128Tensor);

impl From<ReplicatedSetup<PrfKey>> for Value {
    fn from(x: ReplicatedSetup<PrfKey>) -> Value {
        Value::ReplicatedSetup(x)
    }
}

impl TryFrom<Value> for ReplicatedSetup<PrfKey> {
    type Error = ();

    fn try_from(x: Value) -> Result<Self, Self::Error> {
        match x {
            Value::ReplicatedSetup(x) => Ok(x),
            _ => Err(()),
        }
    }
}

impl From<Symbolic<ReplicatedSetup<Symbolic<PrfKey>>>> for SymbolicValue {
    fn from(x: Symbolic<ReplicatedSetup<Symbolic<PrfKey>>>) -> SymbolicValue {
        SymbolicValue::ReplicatedSetup(x)
    }
}

impl TryFrom<SymbolicValue> for Symbolic<ReplicatedSetup<Symbolic<PrfKey>>> {
    type Error = ();

    fn try_from(x: SymbolicValue) -> Result<Self, Self::Error> {
        match x {
            SymbolicValue::ReplicatedSetup(x) => Ok(x),
            _ => Err(()),
        }
    }
}


#[derive(Clone, Debug)]
enum Symbolic<T> {
    Symbolic(SymbolicHandle),
    Concrete(T),
}

#[derive(Clone, Debug)]
struct SymbolicHandle {
    op: String,
}

impl<T> From<SymbolicHandle> for Symbolic<T> {
    fn from(x: SymbolicHandle) -> Symbolic<T> {
        Symbolic::Symbolic(x)
    }
}

#[derive(Clone, Debug)]
enum Operator {
    RingAddOp(RingAddOp),
    RingSubOp(RingSubOp),
    RingMulOp(RingMulOp),
    RingSampleOp(RingSampleOp),
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

operator!(RingAddOp);
operator!(RingSubOp);
operator!(RingMulOp);
operator!(RingSampleOp);
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
struct RingTensor<T>(T);

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
struct ReplicatedTensor<R> {
    shares: [[R; 2]; 3],
}

#[derive(Clone, Debug)]
struct PrfKey([u8; 16]);

#[derive(Clone, Debug)]
struct ReplicatedSetup<K> {
    keys: [[K; 2]; 3],
}

#[derive(Clone, Debug)]
struct ReplicatedZeroShare<R> {
    alphas: [R; 3],
}

type Ring64Tensor = RingTensor<u64>;

type Ring128Tensor = RingTensor<u128>;

type Replicated64Tensor = ReplicatedTensor<Ring64Tensor>;

type Replicated128Tensor = ReplicatedTensor<Ring128Tensor>;

// impl HostFixed {
//     fn placement(&self) -> Placement {
//         Placement::Host
//     }
// }

// impl From<HostFixed> for Value {
//     fn from(x: HostFixed) -> Value {
//         Value::HostFixed(x)
//     }
// }

// impl From<RepFixed> for Value {
//     fn from(x: RepFixed) -> Value {
//         Value::RepFixed(x)
//     }
// }

// struct HostFixed {}

// struct RepFixed {}

// struct Context {}

// impl Context {
//     fn placement_instantiation(&self, plc: Placement) -> PlacementInstantiation {
//         unimplemented!()
//     }
// }

// enum Symbolic<T> {
//     Concrete(T),
//     Symbolic,
// }

macro_rules! modelled {
    ($t:ident, $plc:ty, ($t0:ty) -> $u:ty, $op:ident) => {
        
        impl $t::<ConcreteContext, $t0> for $plc {
            type Output = $u;

            fn apply(&self, ctx: &ConcreteContext, x0: &$t0) -> Self::Output {
                let op = $op {
                    lhs: <$t0>::TY,
                    plc: self.clone().into(),
                };
                ctx.execute_unary(
                    op.into(),
                    x0.clone().into(),
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
                ctx.execute_unary(
                    op.into(),
                    x0.clone().into(),
                )
                .try_into()
                .unwrap()
            }
        }

    };

    ($t:ident, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, $op:ident) => {
        
        impl $t::<ConcreteContext, $t0, $t1> for $plc {
            type Output = $u;

            fn apply(&self, ctx: &ConcreteContext, x0: &$t0, x1: &$t1) -> Self::Output {
                let op = $op {
                    lhs: <$t0>::TY,
                    rhs: <$t1>::TY,
                    plc: self.clone().into(),
                };
                ctx.execute_binary(
                    op.into(),
                    x0.clone().into(),
                    x1.clone().into(),
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
                ctx.execute_binary(
                    op.into(),
                    x0.clone().into(),
                    x1.clone().into(),
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

trait Context {
    type Value;
    fn execute(&self, op: Operator, operands: Vec<Self::Value>) -> Self::Value;
    fn execute_nullary(&self, op: Operator) -> Self::Value;
    fn execute_unary(&self, op: Operator, x: Self::Value) -> Self::Value;
    fn execute_binary(&self, op: Operator, x: Self::Value, y: Self::Value) -> Self::Value;
    fn execute_tenary(&self, op: Operator, s: Self::Value, x: Self::Value, y: Self::Value) -> Self::Value;
}

#[derive(Clone, Debug, Default)]
struct ConcreteContext {}

impl Context for ConcreteContext {
    type Value = Value;

    fn execute(&self, op: Operator, operands: Vec<Value>) -> Value {
        match operands.len() {
            0 => self.execute_nullary(op),
            1 => self.execute_unary(op, operands.get(0).unwrap().clone()),
            2 => self.execute_binary(op, operands.get(0).unwrap().clone(), operands.get(1).unwrap().clone()),
            _ => unimplemented!()
        }
    }

    fn execute_nullary(&self, op: Operator) -> Value {
        match op {
            Operator::RingSampleOp(op) => op.compile(self)().try_into().unwrap(),
            _ => unimplemented!(),
        }
    }

    fn execute_unary(&self, op: Operator, x: Value) -> Value {
        match op {
            Operator::RepShareOp(op) => op.compile(self)(x).try_into().unwrap(),
            _ => unimplemented!(),
        }
    }

    fn execute_binary(&self, op: Operator, x: Value, y: Value) -> Value {
        match op {
            Operator::RepAddOp(op) => op.compile(self)(x, y).try_into().unwrap(),
            Operator::RingAddOp(op) => op.compile(self)(x, y).try_into().unwrap(),
            Operator::RingSubOp(op) => op.compile(self)(x, y).try_into().unwrap(),
            _ => unimplemented!(),
        }
    }

    fn execute_tenary(&self, op: Operator, s: Value, x: Value, y: Value) -> Value {
        match op {
            Operator::RepMulOp(op) => op.compile(self)(s, x, y).try_into().unwrap(),
            _ => unimplemented!(),
        }
    }
}

use std::sync::{Arc, RwLock};

#[derive(Clone, Debug, Default)]
struct SymbolicContext {
    ops: Arc<RwLock<Vec<Operation>>>, // TODO use HashMap so we can do some consistency checks on the fly?
}

impl Context for SymbolicContext {
    type Value = SymbolicValue;

    fn execute(&self, op: Operator, operands: Vec<SymbolicValue>) -> SymbolicValue {
        match operands.len() {
            0 => self.execute_nullary(op),
            1 => self.execute_unary(op, operands.get(0).unwrap().clone()),
            2 => self.execute_binary(op, operands.get(0).unwrap().clone(), operands.get(1).unwrap().clone()),
            _ => unimplemented!()
        }
    }

    fn execute_nullary(&self, op: Operator) -> SymbolicValue {
        match op {
            Operator::RingSampleOp(op) => op.execute_symbolic(self).try_into().unwrap(),
            op => {
                println!("Not implemented: {:?}", op);
                unimplemented!()   
            }
        }
    }

    fn execute_unary(&self, op: Operator, x: SymbolicValue) -> SymbolicValue {
        match op {
            Operator::RepShareOp(op) => op.execute_symbolic(self, x).try_into().unwrap(),
            _ => unimplemented!(),
        }
    }

    fn execute_binary(&self, op: Operator, x: SymbolicValue, y: SymbolicValue) -> SymbolicValue {
        match op {
            Operator::RepAddOp(op) => op.execute_symbolic(self, x, y).try_into().unwrap(),
            Operator::RingAddOp(op) => op.execute_symbolic(self, x, y).try_into().unwrap(),
            Operator::RingSubOp(op) => op.execute_symbolic(self, x, y).try_into().unwrap(),
            _ => unimplemented!(),
        }
    }

    fn execute_tenary(&self, op: Operator, s: SymbolicValue, x: SymbolicValue, y: SymbolicValue) -> SymbolicValue {
        match op {
            Operator::RepMulOp(op) => op.execute_symbolic(self, s, x, y).try_into().unwrap(),
            _ => unimplemented!(),
        }
    }
}

impl SymbolicContext {
    pub fn add_nullary_operation<'s, O: Into<Operator> + Clone>(&'s self, operator: &O) -> String {
        let mut ops = self.ops.write().unwrap(); // TODO
        let op_name: String = format!("op_{}", ops.len());
        let op = Operation {
            name: op_name.clone(),
            operator: operator.clone().into(),
            operands: vec![],
        };
        ops.push(op);
        op_name
    }

    pub fn add_unary_operation<'s, O: Into<Operator> + Clone, T>(
        &'s self,
        operator: &O,
        x: &SymbolicHandle,
    ) -> String {
        let mut ops = self.ops.write().unwrap(); // TODO
        let op_name: String = format!("op_{}", ops.len());
        let op = Operation {
            name: op_name.clone(),
            operator: operator.clone().into(),
            operands: vec![x.op.clone()],
        };
        ops.push(op);
        op_name
    }

    pub fn add_binary_operation<'s, O: Into<Operator> + Clone>(
        &'s self,
        operator: &O,
        x_op: &str,
        y_op: &str,
    ) -> String {
        let mut ops = self.ops.write().unwrap(); // TODO
        let op_name: String = format!("op_{}", ops.len());
        let op = Operation {
            name: op_name.clone(),
            operator: operator.clone().into(),
            operands: vec![x_op.into(), y_op.into()],
        };
        ops.push(op);
        op_name
    }
}

macro_rules! kernel {
    ($op:ty, [$(($plc:ty, $t0:ty, $t1:ty)),+], $k:expr) => {
        impl $op {
            pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Value, Value) -> Value> {
                match (self.plc.ty(), self.lhs, self.rhs) {
                    $(
                        (<$plc>::TY, <$t0>::TY, <$t1>::TY) => {
                            let plc: $plc = self.plc.clone().try_into().unwrap();
                            let ctx = ctx.clone();
                            let op = self.clone();
                            Box::new(move |x: Value, y: Value| -> Value {
                                let x0: $t0 = x.try_into().unwrap();
                                let x1: $t1 = y.try_into().unwrap();
                                $k(&ctx, &plc, x0, x1).into()
                            })
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }

            pub fn execute_symbolic(
                &self,
                ctx: &SymbolicContext,
                x: SymbolicValue,
                y: SymbolicValue,
            ) -> SymbolicValue {
                match (self.plc.ty(), self.lhs, self.rhs) {
                    $(
                        (<$plc>::TY, Symbolic::<$t0>::TY, Symbolic::<$t1>::TY) => {
                            let plc: $plc = self.plc.clone().try_into().unwrap();

                            let x0: <$t0 as KnownType>::Symbolic = x.try_into().unwrap();
                            let x1: <$t1 as KnownType>::Symbolic = y.try_into().unwrap();

                            match (x0, x1) {
                                (Symbolic::Concrete(x0), Symbolic::Concrete(x1)) => {
                                    Symbolic::Concrete($k(ctx, &plc, x0, x1))
                                }
                                (Symbolic::Symbolic(x0), Symbolic::Symbolic(x1)) => {
                                    let op_name = ctx.add_binary_operation(self, &x0.op, &x1.op);
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
struct RepAddOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

kernel!{
    RepAddOp,
    [
        (ReplicatedPlacement, Replicated64Tensor, Replicated64Tensor),
        (ReplicatedPlacement, Replicated128Tensor, Replicated128Tensor)
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
struct RepMulOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

impl RepMulOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Value, Value, Value) -> Value> {
        match (&self.plc, self.lhs, self.rhs) {
            (Placement::ReplicatedPlacement(plc), Replicated64Tensor::TY, Replicated64Tensor::TY) => {
                let ctx = ctx.clone();
                let plc = plc.clone();

                Box::new(move |s: Value, x: Value, y: Value| {
                    let s: ReplicatedSetup<PrfKey> = s.try_into().unwrap();
                    let x: Replicated64Tensor = x.try_into().unwrap();
                    let y: Replicated64Tensor = y.try_into().unwrap();

                    Self::abstract_kernel(&ctx, &plc, s, x, y).into()
                })
            }

            (Placement::ReplicatedPlacement(plc), Replicated128Tensor::TY, Replicated128Tensor::TY) => {
                let ctx = ctx.clone();
                let plc = plc.clone();

                Box::new(move |s: Value, x: Value, y: Value| {
                    let s: ReplicatedSetup<PrfKey> = s.try_into().unwrap();
                    let x: Replicated128Tensor = x.try_into().unwrap();
                    let y: Replicated128Tensor = y.try_into().unwrap();

                    Self::abstract_kernel(&ctx, &plc, s, x, y).into()
                })
            }

            _ => unimplemented!(), // ok
        }
    }

    pub fn execute_symbolic(
        &self,
        ctx: &SymbolicContext,
        s: SymbolicValue,
        x: SymbolicValue,
        y: SymbolicValue,
    ) -> SymbolicValue {
        match (&self.plc, self.lhs, self.rhs) {
            (Placement::ReplicatedPlacement(plc), Symbolic::<Replicated64Tensor>::TY, Symbolic::<Replicated64Tensor>::TY) => {
                let plc = plc.clone();

                let s: <ReplicatedSetup<PrfKey> as KnownType>::Symbolic = s.try_into().unwrap();
                let x: <Replicated64Tensor as KnownType>::Symbolic = x.try_into().unwrap();
                let y: <Replicated64Tensor as KnownType>::Symbolic = y.try_into().unwrap();

                match (s, x, y) {
                    (Symbolic::Concrete(s), Symbolic::Concrete(x), Symbolic::Concrete(y)) => {
                        Symbolic::Concrete(Self::abstract_kernel(ctx, &plc, s, x, y))
                    }
                    (Symbolic::Symbolic(s), Symbolic::Symbolic(x), Symbolic::Symbolic(y)) => {
                        // let op_name = ctx.add_binary_operation(self, &x0.op, &x1.op);
                        // Symbolic::Symbolic(SymbolicHandle { op: op_name })
                        unimplemented!() // TODO
                    }
                    _ => unimplemented!(), // ok
                }
                .into()
            }

            _ => unimplemented!(), // ok
        }
    }

    fn abstract_kernel<C: Context, R, K>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        s: ReplicatedSetup<K>,
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
    fn zero_share(&self, ctx: &C, setup: &ReplicatedSetup<K>) -> ReplicatedZeroShare<R>;
}

// NOTE this is an un-modelled operation (as opposed to the modelled operation that have a representation in computations)
impl<C: Context, K, R> PlacementZeroShare<C, K, R> for ReplicatedPlacement
where
    HostPlacement: PlacementSample<C, R>,
    HostPlacement: PlacementSub<C, R, R, Output = R>,
{
    fn zero_share(&self, ctx: &C, s: &ReplicatedSetup<K>) -> ReplicatedZeroShare<R> {
        let ReplicatedSetup { keys: [[k00, k10], [k11, k21], [k22, k02]] } = s;

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
struct RepShareOp {
    lhs: Ty,
    plc: Placement,
}

impl RepShareOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Value) -> Value> {
        match (&self.plc, self.lhs) {
            (Placement::ReplicatedPlacement(rep_plc), Ring64Tensor::TY) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                Box::new(move |x: Value| {
                    let x: Ring64Tensor = x.try_into().unwrap();
                    Self::abstract_kernel(&ctx, &rep_plc, x).into()
                })
            }

            (Placement::ReplicatedPlacement(rep_plc), Ring128Tensor::TY) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                Box::new(move |x: Value| {
                    let x: Ring128Tensor = x.try_into().unwrap();
                    Self::abstract_kernel(&ctx, &rep_plc, x).into()
                })
            }

            _ => unimplemented!(), // ok
        }
    }

    pub fn execute_symbolic(&self, ctx: &SymbolicContext, x: SymbolicValue) -> SymbolicValue {
        match (&self.plc, self.lhs) {
            (Placement::ReplicatedPlacement(rep_plc), Ring64Tensor::TY) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                let x: Symbolic<Ring64Tensor> = x.try_into().unwrap();
                Symbolic::Concrete(Self::abstract_kernel(&ctx, &rep_plc, x)).into()
            }

            (Placement::ReplicatedPlacement(rep_plc), Ring128Tensor::TY) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                let x: Symbolic<Ring128Tensor> = x.try_into().unwrap();
                Symbolic::Concrete(Self::abstract_kernel(&ctx, &rep_plc, x)).into()
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
struct RingAddOp {
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
        (HostPlacement, Ring64Tensor, Ring64Tensor),
        (HostPlacement, Ring128Tensor, Ring128Tensor)
    ],
    Self::abstract_kernel
}

#[derive(Clone, Debug)]
struct RingSubOp {
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
        (HostPlacement, Ring64Tensor, Ring64Tensor),
        (HostPlacement, Ring128Tensor, Ring128Tensor)
    ],
    Self::abstract_kernel
}

#[derive(Clone, Debug)]
struct RingMulOp {
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
        (HostPlacement, Ring64Tensor, Ring64Tensor),
        (HostPlacement, Ring128Tensor, Ring128Tensor)
    ],
    Self::abstract_kernel
}

trait PlacementSample<C: Context, O> {
    fn sample(&self, ctx: &C) -> O;
}

impl PlacementSample<ConcreteContext, Ring64Tensor> for HostPlacement {
    fn sample(&self, ctx: &ConcreteContext) -> Ring64Tensor {
        ctx.execute_nullary(
            RingSampleOp {
                ty: Ring64Tensor::TY,
                plc: Placement::HostPlacement(self.clone()),
            }
            .into(),
        )
        .try_into()
        .unwrap()
    }
}

impl PlacementSample<ConcreteContext, Ring128Tensor> for HostPlacement {
    fn sample(&self, ctx: &ConcreteContext) -> Ring128Tensor {
        ctx.execute_nullary(
            RingSampleOp {
                ty: Ring128Tensor::TY,
                plc: Placement::HostPlacement(self.clone()),
            }
            .into(),
        )
        .try_into()
        .unwrap()
    }
}

impl PlacementSample<SymbolicContext, Symbolic<Ring64Tensor>> for HostPlacement {
    fn sample(&self, ctx: &SymbolicContext) -> Symbolic<Ring64Tensor> {
        ctx.execute_nullary(
            RingSampleOp {
                ty: Ty::Ring64TensorTy,
                plc: Placement::HostPlacement(self.clone()),
            }
            .into(),
        )
        .try_into()
        .unwrap()
    }
}

impl PlacementSample<SymbolicContext, Symbolic<Ring128Tensor>> for HostPlacement {
    fn sample(&self, ctx: &SymbolicContext) -> Symbolic<Ring128Tensor> {
        ctx.execute_nullary(
            RingSampleOp {
                ty: Ty::Ring128TensorTy,
                plc: Placement::HostPlacement(self.clone()),
            }
            .into(),
        )
        .try_into()
        .unwrap()
    }
}

#[derive(Clone, Debug)]
struct RingSampleOp {
    ty: Ty,
    plc: Placement,
}

impl RingSampleOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn() -> Value> {
        match (&self.plc, &self.ty) {
            (Placement::HostPlacement(_), Ty::Ring64TensorTy) => {
                Box::new(move || RingTensor(987654321_u64).into())
            }
            (Placement::HostPlacement(_), Ty::Ring128TensorTy) => {
                Box::new(move || RingTensor(987654321_u128).into())
            }
            _ => unimplemented!(),
        }
    }

    // TODO could we derive this from the return type of the closure returned by `compile`?
    // not sure this will work, seems like we need a Ty instead, which comes as part of
    // type checking.
    pub fn execute_symbolic(&self, ctx: &SymbolicContext) -> SymbolicValue {
        match (&self.plc, &self.ty) {
            (Placement::HostPlacement(_), Ty::Ring64TensorTy) => {
                let op_name = ctx.add_nullary_operation(self);
                SymbolicValue::Ring64Tensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                }))
            }
            (Placement::HostPlacement(_), Ty::Ring128TensorTy) => {
                let op_name = ctx.add_nullary_operation(self);
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
            name: "ze".into(),
            operator: RepMulOp {
                lhs: Ring128Tensor::TY,
                rhs: Ring128Tensor::TY,
                plc: rep_plc.clone().into(),
            }
            .into(),
            operands: vec!["xe".into(), "ye".into()],
        },
    ];

    let ctx = SymbolicContext::default();

    let mut env: HashMap<String, SymbolicValue> = HashMap::default();
    for op in ops {
        let operator = op.operator;
        let operands = op.operands.iter().map(|input_name| env.get(input_name).unwrap().clone()).collect();
        let res = ctx.execute(operator, operands);
        env.insert(op.name, res);
    }

    let ops = ctx.ops.read().unwrap();
    for op in ops.iter() {
        println!("  {:?}", op);
    }

    // let comp = r#"

    // "#.try_into().unwrap();

    // let exec = SymbolicExecutor;
    // exec.eval(comp);
}
