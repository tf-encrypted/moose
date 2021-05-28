#![allow(dead_code)]
#![allow(unused_variables)]

use std::convert::{TryFrom, TryInto};
use std::ops::{Add, Sub};

#[derive(Debug, Clone)]
enum Placement {
    Host(HostPlacement),
    Replicated(ReplicatedPlacement),
}

impl From<HostPlacement> for Placement {
    fn from(plc: HostPlacement) -> Self {
        Placement::Host(plc)
    }
}

impl From<ReplicatedPlacement> for Placement {
    fn from(plc: ReplicatedPlacement) -> Self {
        Placement::Replicated(plc)
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

// enum PlacementInstantiation {
//     Host,
//     Rep,
// }

#[derive(Debug, Clone)]
enum Ty {
    // HostFixedTy,
    Ring64TensorTy,
    Ring128TensorTy,
    Replicated64TensorTy,
    Replicated128TensorTy,
}

trait KnownType {
    fn ty() -> Ty;
}

impl KnownType for Ring64Tensor {
    fn ty() -> Ty {
        Ty::Ring64TensorTy
    }
}

impl KnownType for Ring128Tensor {
    fn ty() -> Ty {
        Ty::Ring128TensorTy
    }
}

impl KnownType for ReplicatedTensor<Ring64Tensor> {
    fn ty() -> Ty {
        Ty::Replicated64TensorTy
    }
}

impl KnownType for ReplicatedTensor<Ring128Tensor> {
    fn ty() -> Ty {
        Ty::Replicated128TensorTy
    }
}

impl<T: KnownType> KnownType for Symbolic<T> {
    fn ty() -> Ty {
        T::ty()
    }
}

#[derive(Clone, Debug)]
enum Value {
    // HostFixed(HostFixed),
    // RepFixed(RepFixed),
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    Replicated64Tensor(ReplicatedTensor<Ring64Tensor>),
    Replicated128Tensor(ReplicatedTensor<Ring128Tensor>),
}

impl From<Ring64Tensor> for Value {
    fn from(x: Ring64Tensor) -> Value {
        Value::Ring64Tensor(x)
    }
}

impl From<Ring128Tensor> for Value {
    fn from(x: Ring128Tensor) -> Value {
        Value::Ring128Tensor(x)
    }
}

impl From<ReplicatedTensor<Ring64Tensor>> for Value {
    fn from(x: ReplicatedTensor<Ring64Tensor>) -> Value {
        Value::Replicated64Tensor(x)
    }
}

impl From<ReplicatedTensor<Ring128Tensor>> for Value {
    fn from(x: ReplicatedTensor<Ring128Tensor>) -> Value {
        Value::Replicated128Tensor(x)
    }
}

impl TryFrom<Value> for Ring64Tensor {
    type Error = ();

    fn try_from(x: Value) -> Result<Self, Self::Error> {
        match x {
            Value::Ring64Tensor(x) => Ok(x),
            _ => Err(()),
        }
    }
}

impl TryFrom<Value> for Ring128Tensor {
    type Error = ();

    fn try_from(x: Value) -> Result<Self, Self::Error> {
        match x {
            Value::Ring128Tensor(x) => Ok(x),
            _ => Err(()),
        }
    }
}

impl TryFrom<Value> for ReplicatedTensor<Ring64Tensor> {
    type Error = ();

    fn try_from(x: Value) -> Result<Self, Self::Error> {
        match x {
            Value::Replicated64Tensor(x) => Ok(x),
            _ => Err(()),
        }
    }
}

impl TryFrom<Value> for ReplicatedTensor<Ring128Tensor> {
    type Error = ();

    fn try_from(x: Value) -> Result<Self, Self::Error> {
        match x {
            Value::Replicated128Tensor(x) => Ok(x),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug)]
enum SymbolicValue {
    Ring64Tensor(Symbolic<Ring64Tensor>),
    Ring128Tensor(Symbolic<Ring128Tensor>),
    Replicated64Tensor(Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>>),
    Replicated128Tensor(Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>>),
}

impl From<Symbolic<Ring64Tensor>> for SymbolicValue {
    fn from(x: Symbolic<Ring64Tensor>) -> SymbolicValue {
        SymbolicValue::Ring64Tensor(x)
    }
}

impl From<Symbolic<Ring128Tensor>> for SymbolicValue {
    fn from(x: Symbolic<Ring128Tensor>) -> SymbolicValue {
        SymbolicValue::Ring128Tensor(x)
    }
}

impl From<Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>>> for SymbolicValue {
    fn from(x: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>>) -> SymbolicValue {
        SymbolicValue::Replicated64Tensor(x)
    }
}

impl From<Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>>> for SymbolicValue {
    fn from(x: Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>>) -> SymbolicValue {
        SymbolicValue::Replicated128Tensor(x)
    }
}

impl TryFrom<SymbolicValue> for Symbolic<Ring64Tensor> {
    type Error = ();

    fn try_from(x: SymbolicValue) -> Result<Self, Self::Error> {
        match x {
            SymbolicValue::Ring64Tensor(x) => Ok(x),
            _ => Err(()),
        }
    }
}

impl TryFrom<SymbolicValue> for Symbolic<Ring128Tensor> {
    type Error = ();

    fn try_from(x: SymbolicValue) -> Result<Self, Self::Error> {
        match x {
            SymbolicValue::Ring128Tensor(x) => Ok(x),
            _ => Err(()),
        }
    }
}

impl TryFrom<SymbolicValue> for Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> {
    type Error = ();

    fn try_from(x: SymbolicValue) -> Result<Self, Self::Error> {
        match x {
            SymbolicValue::Replicated64Tensor(x) => Ok(x),
            _ => Err(()),
        }
    }
}

impl TryFrom<SymbolicValue> for Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>> {
    type Error = ();

    fn try_from(x: SymbolicValue) -> Result<Self, Self::Error> {
        match x {
            SymbolicValue::Replicated128Tensor(x) => Ok(x),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug)]
enum Symbolic<T> {
    Symbolic { ty: Ty },
    Concrete(T),
}

#[derive(Clone, Debug)]
enum Operator {
    RingAdd(RingAddOp),
    RingSub(RingSubOp),
    RingSample(RingSampleOp),
    RepAdd(RepAddOp),
    RepShare(RepShareOp),
    // Constant(ConstantOp),
}

impl From<RingAddOp> for Operator {
    fn from(op: RingAddOp) -> Self {
        Operator::RingAdd(op)
    }
}

impl From<RingSubOp> for Operator {
    fn from(op: RingSubOp) -> Self {
        Operator::RingSub(op)
    }
}

impl From<RingSampleOp> for Operator {
    fn from(op: RingSampleOp) -> Self {
        Operator::RingSample(op)
    }
}

impl From<RepAddOp> for Operator {
    fn from(op: RepAddOp) -> Self {
        Operator::RepAdd(op)
    }
}

impl From<RepShareOp> for Operator {
    fn from(op: RepShareOp) -> Self {
        Operator::RepShare(op)
    }
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

#[derive(Clone, Debug)]
struct ReplicatedTensor<R> {
    shares: [[R; 2]; 3],
}

#[derive(Clone, Debug)]
enum Ring64Tensor {
    Symbolic(Operator),
    Concrete(RingTensor<u64>),
}

#[derive(Clone, Debug)]
enum Ring128Tensor {
    Symbolic(Operator),
    Concrete(RingTensor<u128>),
}

impl From<RingTensor<u64>> for Ring64Tensor {
    fn from(x: RingTensor<u64>) -> Ring64Tensor {
        Ring64Tensor::Concrete(x)
    }
}

impl From<RingTensor<u128>> for Ring128Tensor {
    fn from(x: RingTensor<u128>) -> Ring128Tensor {
        Ring128Tensor::Concrete(x)
    }
}

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

trait PlacementAdd<C: Context, T, U, O> {
    fn add(&self, ctx: &C, x: T, y: U) -> O;
}

impl PlacementAdd<ConcreteContext, &Ring64Tensor, &Ring64Tensor, Ring64Tensor> for HostPlacement {
    fn add(&self, ctx: &ConcreteContext, x: &Ring64Tensor, y: &Ring64Tensor) -> Ring64Tensor {
        ctx.execute_binary(
            RingAddOp {
                lhs: Ring64Tensor::ty(),
                rhs: Ring64Tensor::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl PlacementAdd<ConcreteContext, &Ring128Tensor, &Ring128Tensor, Ring128Tensor>
    for HostPlacement
{
    fn add(&self, ctx: &ConcreteContext, x: &Ring128Tensor, y: &Ring128Tensor) -> Ring128Tensor {
        ctx.execute_binary(
            RingAddOp {
                lhs: Ring128Tensor::ty(),
                rhs: Ring128Tensor::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

// TODO the following could probably be derived from the above
impl
    PlacementAdd<
        SymbolicContext,
        &Symbolic<Ring64Tensor>,
        &Symbolic<Ring64Tensor>,
        Symbolic<Ring64Tensor>,
    > for HostPlacement
{
    fn add(
        &self,
        ctx: &SymbolicContext,
        x: &Symbolic<Ring64Tensor>,
        y: &Symbolic<Ring64Tensor>,
    ) -> Symbolic<Ring64Tensor> {
        ctx.execute_binary(
            RingAddOp {
                lhs: Symbolic::<Ring64Tensor>::ty(),
                rhs: Symbolic::<Ring64Tensor>::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl
    PlacementAdd<
        SymbolicContext,
        &Symbolic<Ring128Tensor>,
        &Symbolic<Ring128Tensor>,
        Symbolic<Ring128Tensor>,
    > for HostPlacement
{
    fn add(
        &self,
        ctx: &SymbolicContext,
        x: &Symbolic<Ring128Tensor>,
        y: &Symbolic<Ring128Tensor>,
    ) -> Symbolic<Ring128Tensor> {
        ctx.execute_binary(
            RingAddOp {
                lhs: Symbolic::<Ring128Tensor>::ty(),
                rhs: Symbolic::<Ring128Tensor>::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl
    PlacementAdd<
        ConcreteContext,
        &ReplicatedTensor<Ring64Tensor>,
        &ReplicatedTensor<Ring64Tensor>,
        ReplicatedTensor<Ring64Tensor>,
    > for ReplicatedPlacement
{
    fn add(
        &self,
        ctx: &ConcreteContext,
        x: &ReplicatedTensor<Ring64Tensor>,
        y: &ReplicatedTensor<Ring64Tensor>,
    ) -> ReplicatedTensor<Ring64Tensor> {
        ctx.execute_binary(
            RepAddOp {
                lhs: ReplicatedTensor::<Ring64Tensor>::ty(),
                rhs: ReplicatedTensor::<Ring64Tensor>::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl
    PlacementAdd<
        ConcreteContext,
        &ReplicatedTensor<Ring128Tensor>,
        &ReplicatedTensor<Ring128Tensor>,
        ReplicatedTensor<Ring128Tensor>,
    > for ReplicatedPlacement
{
    fn add(
        &self,
        ctx: &ConcreteContext,
        x: &ReplicatedTensor<Ring128Tensor>,
        y: &ReplicatedTensor<Ring128Tensor>,
    ) -> ReplicatedTensor<Ring128Tensor> {
        ctx.execute_binary(
            RepAddOp {
                lhs: ReplicatedTensor::<Ring128Tensor>::ty(),
                rhs: ReplicatedTensor::<Ring128Tensor>::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl
    PlacementAdd<
        SymbolicContext,
        &Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>>,
        &Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>>,
        Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>>,
    > for ReplicatedPlacement
{
    fn add(
        &self,
        ctx: &SymbolicContext,
        x: &Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>>,
        y: &Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>>,
    ) -> Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> {
        ctx.execute_binary(
            RepAddOp {
                lhs: ReplicatedTensor::<Ring64Tensor>::ty(),
                rhs: ReplicatedTensor::<Ring64Tensor>::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl
    PlacementAdd<
        SymbolicContext,
        &Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>>,
        &Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>>,
        Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>>,
    > for ReplicatedPlacement
{
    fn add(
        &self,
        ctx: &SymbolicContext,
        x: &Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>>,
        y: &Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>>,
    ) -> Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>> {
        ctx.execute_binary(
            RepAddOp {
                lhs: ReplicatedTensor::<Ring128Tensor>::ty(),
                rhs: ReplicatedTensor::<Ring128Tensor>::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

trait PlacementSub<C: Context, T, U> {
    type Output;

    fn sub(&self, ctx: &C, x: T, y: U) -> Self::Output;
}

impl PlacementSub<ConcreteContext, &Ring64Tensor, &Ring64Tensor> for HostPlacement {
    type Output = Ring64Tensor;

    fn sub(&self, ctx: &ConcreteContext, x: &Ring64Tensor, y: &Ring64Tensor) -> Self::Output {
        ctx.execute_binary(
            RingSubOp {
                lhs: Ring64Tensor::ty(),
                rhs: Ring64Tensor::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl PlacementSub<ConcreteContext, &Ring128Tensor, &Ring128Tensor> for HostPlacement {
    type Output = Ring128Tensor;

    fn sub(&self, ctx: &ConcreteContext, x: &Ring128Tensor, y: &Ring128Tensor) -> Self::Output {
        ctx.execute_binary(
            RingSubOp {
                lhs: Ring128Tensor::ty(),
                rhs: Ring128Tensor::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl PlacementSub<SymbolicContext, &Symbolic<Ring64Tensor>, &Symbolic<Ring64Tensor>>
    for HostPlacement
{
    type Output = Symbolic<Ring64Tensor>;

    fn sub(
        &self,
        ctx: &SymbolicContext,
        x: &Symbolic<Ring64Tensor>,
        y: &Symbolic<Ring64Tensor>,
    ) -> Self::Output {
        ctx.execute_binary(
            RingSubOp {
                lhs: Symbolic::<Ring64Tensor>::ty(),
                rhs: Symbolic::<Ring64Tensor>::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl PlacementSub<SymbolicContext, &Symbolic<Ring128Tensor>, &Symbolic<Ring128Tensor>>
    for HostPlacement
{
    type Output = Symbolic<Ring128Tensor>;

    fn sub(
        &self,
        ctx: &SymbolicContext,
        x: &Symbolic<Ring128Tensor>,
        y: &Symbolic<Ring128Tensor>,
    ) -> Self::Output {
        ctx.execute_binary(
            RingSubOp {
                lhs: Symbolic::<Ring128Tensor>::ty(),
                rhs: Symbolic::<Ring128Tensor>::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
            y.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

trait PlacementShare<C: Context, T, O> {
    fn share(&self, ctx: &C, x: T) -> O;
}

impl PlacementShare<ConcreteContext, &Ring64Tensor, ReplicatedTensor<Ring64Tensor>>
    for ReplicatedPlacement
{
    fn share(&self, ctx: &ConcreteContext, x: &Ring64Tensor) -> ReplicatedTensor<Ring64Tensor> {
        ctx.execute_unary(
            RepShareOp {
                lhs: Ring64Tensor::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl PlacementShare<ConcreteContext, &Ring128Tensor, ReplicatedTensor<Ring128Tensor>>
    for ReplicatedPlacement
{
    fn share(&self, ctx: &ConcreteContext, x: &Ring128Tensor) -> ReplicatedTensor<Ring128Tensor> {
        ctx.execute_unary(
            RepShareOp {
                lhs: Ring128Tensor::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl
    PlacementShare<
        SymbolicContext,
        &Symbolic<Ring64Tensor>,
        Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>>,
    > for ReplicatedPlacement
{
    fn share(
        &self,
        ctx: &SymbolicContext,
        x: &Symbolic<Ring64Tensor>,
    ) -> Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> {
        ctx.execute_unary(
            RepShareOp {
                lhs: Symbolic::<Ring64Tensor>::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

impl
    PlacementShare<
        SymbolicContext,
        &Symbolic<Ring128Tensor>,
        Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>>,
    > for ReplicatedPlacement
{
    fn share(
        &self,
        ctx: &SymbolicContext,
        x: &Symbolic<Ring128Tensor>,
    ) -> Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>> {
        ctx.execute_unary(
            RepShareOp {
                lhs: Symbolic::<Ring128Tensor>::ty(),
                plc: self.clone().into(),
            }
            .into(),
            x.clone().into(),
        )
        .try_into()
        .unwrap()
    }
}

trait Context {
    type Value;
    fn execute_nullary(&self, op: Operator) -> Self::Value;
    fn execute_unary(&self, op: Operator, x: Self::Value) -> Self::Value;
    fn execute_binary(&self, op: Operator, x: Self::Value, y: Self::Value) -> Self::Value;
}

#[derive(Clone, Debug)]
struct ConcreteContext {}

impl Context for ConcreteContext {
    type Value = Value;

    fn execute_nullary(&self, op: Operator) -> Value {
        match op {
            Operator::RingSample(op) => op.compile(self)().try_into().unwrap(),
            _ => unimplemented!(),
        }
    }

    fn execute_unary(&self, op: Operator, x: Value) -> Value {
        match op {
            Operator::RepShare(op) => op.compile(self)(x).try_into().unwrap(),
            _ => unimplemented!(),
        }
    }

    fn execute_binary(&self, op: Operator, x: Value, y: Value) -> Value {
        match op {
            Operator::RepAdd(op) => op.compile(self)(x, y).try_into().unwrap(),
            Operator::RingAdd(op) => op.compile(self)(x, y).try_into().unwrap(),
            Operator::RingSub(op) => op.compile(self)(x, y).try_into().unwrap(),
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Debug)]
struct SymbolicContext {}

impl Context for SymbolicContext {
    type Value = SymbolicValue;

    fn execute_nullary(&self, op: Operator) -> SymbolicValue {
        match op {
            Operator::RingSample(op) => op.compile_symbolic(self)().try_into().unwrap(),
            _ => unimplemented!(),
        }
    }

    fn execute_unary(&self, op: Operator, x: SymbolicValue) -> SymbolicValue {
        match op {
            Operator::RepShare(op) => op.compile_symbolic(self)(x).try_into().unwrap(),
            _ => unimplemented!(),
        }
    }

    fn execute_binary(&self, op: Operator, x: SymbolicValue, y: SymbolicValue) -> SymbolicValue {
        match op {
            Operator::RepAdd(op) => op.compile_symbolic(self)(x, y).try_into().unwrap(),
            Operator::RingAdd(op) => op.compile_symbolic(self)(x, y).try_into().unwrap(),
            Operator::RingSub(op) => op.compile_symbolic(self)(x, y).try_into().unwrap(),
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Debug)]
struct RepAddOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

impl RepAddOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Value, Value) -> Value> {
        match (&self.plc, &self.lhs, &self.rhs) {
            (Placement::Replicated(plc), Ty::Replicated64TensorTy, Ty::Replicated64TensorTy) => {
                let ctx = ctx.clone();
                let plc = plc.clone();

                Box::new(move |x: Value, y: Value| {
                    let x: ReplicatedTensor<Ring64Tensor> = x.try_into().unwrap();
                    let y: ReplicatedTensor<Ring64Tensor> = y.try_into().unwrap();

                    Self::abstract_kernel(&ctx, &plc, x, y).into()
                })
            }

            (Placement::Replicated(plc), Ty::Replicated128TensorTy, Ty::Replicated128TensorTy) => {
                let ctx = ctx.clone();
                let plc = plc.clone();

                Box::new(move |x: Value, y: Value| {
                    let x: ReplicatedTensor<Ring128Tensor> = x.try_into().unwrap();
                    let y: ReplicatedTensor<Ring128Tensor> = y.try_into().unwrap();

                    Self::abstract_kernel(&ctx, &plc, x, y).into()
                })
            }

            // (Placement::Rep, Ty::HostFixedTy, Ty::HostFixedTy) => {
            //     |x: Value, y: Value| {
            //         // let x_owner = ctx.placement_instantiation(x.placement());
            //         // let y_owner = ctx.placement_instantiation(y.placement());

            //         // let xe = x_owner.share(x);
            //         // let ye = y_owner.share(y);
            //         // add(xe, ye)
            //         unimplemented!()
            //     }
            // }
            // (Placement::Rep, Ty::RepFixedTy, Ty::RepFixedTy) => {
            //     |x: Value, y: Value| {
            //         unimplemented!()
            //     }
            // }
            _ => unimplemented!(), // ok
        }
    }

    pub fn compile_symbolic(
        &self,
        ctx: &SymbolicContext,
    ) -> Box<dyn Fn(SymbolicValue, SymbolicValue) -> SymbolicValue> {
        match (&self.plc, &self.lhs, &self.rhs) {
            (Placement::Replicated(plc), Ty::Replicated64TensorTy, Ty::Replicated64TensorTy) => {
                let ctx = ctx.clone();
                let plc = plc.clone();

                Box::new(move |x: SymbolicValue, y: SymbolicValue| {
                    let x: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> =
                        x.try_into().unwrap();
                    let y: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> =
                        y.try_into().unwrap();

                    match (x, y) {
                        (Symbolic::Concrete(x), Symbolic::Concrete(y)) => {
                            Symbolic::Concrete(Self::abstract_kernel(&ctx, &plc, x, y))
                        }
                        (
                            Symbolic::Symbolic {
                                ty: Ty::Replicated64TensorTy,
                            },
                            Symbolic::Symbolic {
                                ty: Ty::Replicated64TensorTy,
                            },
                        ) => Symbolic::Symbolic {
                            ty: Ty::Replicated64TensorTy,
                        },
                        (Symbolic::Symbolic { .. }, _) => {
                            unimplemented!() // ok
                        }
                        (Symbolic::Concrete(_), _) => {
                            unimplemented!() // ok
                        }
                    }
                    .into()
                })
            }

            (Placement::Replicated(plc), Ty::Replicated128TensorTy, Ty::Replicated128TensorTy) => {
                let ctx = ctx.clone();
                let plc = plc.clone();

                Box::new(move |x: SymbolicValue, y: SymbolicValue| {
                    let x: Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>> =
                        x.try_into().unwrap();
                    let y: Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>> =
                        y.try_into().unwrap();

                    match (x, y) {
                        (Symbolic::Concrete(x), Symbolic::Concrete(y)) => {
                            Symbolic::Concrete(Self::abstract_kernel(&ctx, &plc, x, y))
                        }
                        (
                            Symbolic::Symbolic {
                                ty: Ty::Replicated128TensorTy,
                            },
                            Symbolic::Symbolic {
                                ty: Ty::Replicated128TensorTy,
                            },
                        ) => Symbolic::Symbolic {
                            ty: Ty::Replicated128TensorTy,
                        },
                        (Symbolic::Symbolic { .. }, _) => {
                            unimplemented!() // ok
                        }
                        (Symbolic::Concrete(_), _) => {
                            unimplemented!() // ok
                        }
                    }
                    .into()
                })
            }

            _ => unimplemented!(), // ok
        }
    }

    fn abstract_kernel<C: Context, R>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: ReplicatedTensor<R>,
        y: ReplicatedTensor<R>,
    ) -> ReplicatedTensor<R>
    where
        for<'x, 'y> HostPlacement: PlacementAdd<C, &'x R, &'y R, R>,
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
struct RepShareOp {
    lhs: Ty,
    plc: Placement,
}

impl RepShareOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Value) -> Value> {
        match (&self.plc, &self.lhs) {
            (Placement::Replicated(rep_plc), Ty::Ring64TensorTy) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                Box::new(move |x: Value| {
                    let x: Ring64Tensor = x.try_into().unwrap();
                    Self::abstract_kernel(&ctx, &rep_plc, x).into()
                })
            }

            (Placement::Replicated(rep_plc), Ty::Ring128TensorTy) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                Box::new(move |x: Value| {
                    let x: Ring128Tensor = x.try_into().unwrap();
                    Self::abstract_kernel(&ctx, &rep_plc, x).into()
                })
            }

            // (Placement::Rep, Ty::HostFixedTy, Ty::HostFixedTy) => {
            //     |x: Value, y: Value| {
            //         // let x_owner = ctx.placement_instantiation(x.placement());
            //         // let y_owner = ctx.placement_instantiation(y.placement());

            //         // let xe = x_owner.share(x);
            //         // let ye = y_owner.share(y);
            //         // add(xe, ye)
            //         unimplemented!()
            //     }
            // }
            // (Placement::Rep, Ty::RepFixedTy, Ty::RepFixedTy) => {
            //     |x: Value, y: Value| {
            //         unimplemented!()
            //     }
            // }
            _ => unimplemented!(),
        }
    }

    pub fn compile_symbolic(
        &self,
        ctx: &SymbolicContext,
    ) -> Box<dyn Fn(SymbolicValue) -> SymbolicValue> {
        match (&self.plc, &self.lhs) {
            (Placement::Replicated(rep_plc), Ty::Ring64TensorTy) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                Box::new(move |x: SymbolicValue| {
                    let x: Symbolic<Ring64Tensor> = x.try_into().unwrap();
                    Symbolic::Concrete(Self::abstract_kernel(&ctx, &rep_plc, x)).into()
                })
            }

            (Placement::Replicated(rep_plc), Ty::Ring128TensorTy) => {
                let rep_plc = rep_plc.clone();
                let ctx = ctx.clone();
                Box::new(move |x: SymbolicValue| {
                    let x: Symbolic<Ring128Tensor> = x.try_into().unwrap();
                    Symbolic::Concrete(Self::abstract_kernel(&ctx, &rep_plc, x)).into()
                })
            }

            // (Placement::Rep, Ty::HostFixedTy, Ty::HostFixedTy) => {
            //     |x: Value, y: Value| {
            //         // let x_owner = ctx.placement_instantiation(x.placement());
            //         // let y_owner = ctx.placement_instantiation(y.placement());

            //         // let xe = x_owner.share(x);
            //         // let ye = y_owner.share(y);
            //         // add(xe, ye)
            //         unimplemented!()
            //     }
            // }
            // (Placement::Rep, Ty::RepFixedTy, Ty::RepFixedTy) => {
            //     |x: Value, y: Value| {
            //         unimplemented!()
            //     }
            // }
            _ => unimplemented!(), // ok
        }
    }

    fn abstract_kernel<C: Context, R: Clone>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: R,
    ) -> ReplicatedTensor<R>
    where
        HostPlacement: PlacementSample<C, R>,
        for<'x, 'y> HostPlacement: PlacementAdd<C, &'x R, &'y R, R>,
        for<'x, 'y> HostPlacement: PlacementSub<C, &'x R, &'y R, Output = R>,
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

        // TODO we should not use player0 here, but rather the owner of `x`
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
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Value, Value) -> Value> {
        match (&self.plc, &self.lhs, &self.rhs) {
            (Placement::Host(_), Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                let op = self.clone();
                Box::new(move |x: Value, y: Value| -> Value {
                    match (x, y) {
                        (Value::Ring64Tensor(x), Value::Ring64Tensor(y)) => match (x, y) {
                            (Ring64Tensor::Symbolic(x_op), Ring64Tensor::Symbolic(y_op)) => {
                                Ring64Tensor::Symbolic(Operator::RingAdd(op.clone()))
                            }
                            (Ring64Tensor::Concrete(x), Ring64Tensor::Concrete(y)) => {
                                Self::abstract_kernel(x, y).into()
                            }
                            _ => unimplemented!(),
                        }
                        .into(),
                        _ => unimplemented!(),
                    }
                })
            }

            (Placement::Host(_), Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                let op = self.clone();
                Box::new(move |x: Value, y: Value| -> Value {
                    match (x, y) {
                        (Value::Ring128Tensor(x), Value::Ring128Tensor(y)) => match (x, y) {
                            (Ring128Tensor::Symbolic(x_op), Ring128Tensor::Symbolic(y_op)) => {
                                Ring128Tensor::Symbolic(Operator::RingAdd(op.clone()))
                            }
                            (Ring128Tensor::Concrete(x), Ring128Tensor::Concrete(y)) => {
                                Self::abstract_kernel(x, y).into()
                            }
                            _ => unimplemented!(),
                        }
                        .into(),
                        _ => unimplemented!(),
                    }
                })
            }

            _ => unimplemented!(),
        }
    }

    pub fn compile_symbolic(
        &self,
        ctx: &SymbolicContext,
    ) -> Box<dyn Fn(SymbolicValue, SymbolicValue) -> SymbolicValue> {
        match (&self.plc, &self.lhs, &self.rhs) {
            (Placement::Host(_), Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                let op = self.clone();
                Box::new(move |x: SymbolicValue, y: SymbolicValue| -> SymbolicValue {
                    match (x, y) {
                        (SymbolicValue::Ring64Tensor(x), SymbolicValue::Ring64Tensor(y)) => {
                            match (x, y) {
                                (
                                    Symbolic::Concrete(Ring64Tensor::Concrete(x)),
                                    Symbolic::Concrete(Ring64Tensor::Concrete(y)),
                                ) => Symbolic::Concrete(Ring64Tensor::Concrete(
                                    Self::abstract_kernel(x, y),
                                )),
                                (
                                    Symbolic::Symbolic {
                                        ty: Ty::Ring64TensorTy,
                                    },
                                    Symbolic::Symbolic {
                                        ty: Ty::Ring64TensorTy,
                                    },
                                ) => Symbolic::Symbolic {
                                    ty: Ty::Ring64TensorTy,
                                },
                                _ => unimplemented!(),
                            }
                            .into()
                        }
                        _ => unimplemented!(),
                    }
                })
            }

            (Placement::Host(_), Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                let op = self.clone();
                Box::new(move |x: SymbolicValue, y: SymbolicValue| -> SymbolicValue {
                    match (x, y) {
                        (SymbolicValue::Ring128Tensor(x), SymbolicValue::Ring128Tensor(y)) => {
                            match (x, y) {
                                (
                                    Symbolic::Concrete(Ring128Tensor::Concrete(x)),
                                    Symbolic::Concrete(Ring128Tensor::Concrete(y)),
                                ) => Symbolic::Concrete(Ring128Tensor::Concrete(
                                    Self::abstract_kernel(x, y),
                                )),
                                (
                                    Symbolic::Symbolic {
                                        ty: Ty::Ring128TensorTy,
                                    },
                                    Symbolic::Symbolic {
                                        ty: Ty::Ring128TensorTy,
                                    },
                                ) => Symbolic::Symbolic {
                                    ty: Ty::Ring128TensorTy,
                                },
                                _ => unimplemented!(),
                            }
                            .into()
                        }
                        _ => unimplemented!(),
                    }
                })
            }

            _ => unimplemented!(),
        }
    }

    fn abstract_kernel<T>(x: RingTensor<T>, y: RingTensor<T>) -> RingTensor<T>
    where
        RingTensor<T>: Add<RingTensor<T>, Output = RingTensor<T>>,
    {
        x + y
    }
}

#[derive(Clone, Debug)]
struct RingSubOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

impl RingSubOp {
    pub fn compile<C: Context>(&self, ctx: &C) -> Box<dyn Fn(Value, Value) -> Value> {
        match (&self.plc, &self.lhs, &self.rhs) {
            (Placement::Host(_), Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                let op = self.clone();
                Box::new(move |x: Value, y: Value| -> Value {
                    match (x, y) {
                        (Value::Ring64Tensor(x), Value::Ring64Tensor(y)) => match (x, y) {
                            (Ring64Tensor::Symbolic(x_op), Ring64Tensor::Symbolic(y_op)) => {
                                Ring64Tensor::Symbolic(Operator::RingSub(op.clone()))
                            }
                            (Ring64Tensor::Concrete(x), Ring64Tensor::Concrete(y)) => {
                                Self::abstract_kernel(x, y).into()
                            }
                            _ => unimplemented!(),
                        }
                        .into(),
                        _ => unimplemented!(),
                    }
                })
            }

            (Placement::Host(_), Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                let op = self.clone();
                Box::new(move |x: Value, y: Value| -> Value {
                    match (x, y) {
                        (Value::Ring128Tensor(x), Value::Ring128Tensor(y)) => match (x, y) {
                            (Ring128Tensor::Symbolic(x_op), Ring128Tensor::Symbolic(y_op)) => {
                                Ring128Tensor::Symbolic(Operator::RingSub(op.clone()))
                            }
                            (Ring128Tensor::Concrete(x), Ring128Tensor::Concrete(y)) => {
                                Self::abstract_kernel(x, y).into()
                            }
                            _ => unimplemented!(),
                        }
                        .into(),
                        _ => unimplemented!(),
                    }
                })
            }

            _ => unimplemented!(),
        }
    }

    pub fn compile_symbolic(
        &self,
        ctx: &SymbolicContext,
    ) -> Box<dyn Fn(SymbolicValue, SymbolicValue) -> SymbolicValue> {
        match (&self.plc, &self.lhs, &self.rhs) {
            (Placement::Host(_), Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                let op = self.clone();
                Box::new(move |x: SymbolicValue, y: SymbolicValue| -> SymbolicValue {
                    match (x, y) {
                        (SymbolicValue::Ring64Tensor(x), SymbolicValue::Ring64Tensor(y)) => {
                            match (x, y) {
                                (
                                    Symbolic::Concrete(Ring64Tensor::Concrete(x)),
                                    Symbolic::Concrete(Ring64Tensor::Concrete(y)),
                                ) => Symbolic::Concrete(Ring64Tensor::Concrete(
                                    Self::abstract_kernel(x, y),
                                )),
                                (
                                    Symbolic::Symbolic {
                                        ty: Ty::Ring64TensorTy,
                                    },
                                    Symbolic::Symbolic {
                                        ty: Ty::Ring64TensorTy,
                                    },
                                ) => Symbolic::Symbolic {
                                    ty: Ty::Ring64TensorTy,
                                },
                                _ => unimplemented!(),
                            }
                            .into()
                        }
                        _ => unimplemented!(),
                    }
                })
            }

            (Placement::Host(_), Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                let op = self.clone();
                Box::new(move |x: SymbolicValue, y: SymbolicValue| -> SymbolicValue {
                    match (x, y) {
                        (SymbolicValue::Ring128Tensor(x), SymbolicValue::Ring128Tensor(y)) => {
                            match (x, y) {
                                (
                                    Symbolic::Concrete(Ring128Tensor::Concrete(x)),
                                    Symbolic::Concrete(Ring128Tensor::Concrete(y)),
                                ) => Symbolic::Concrete(Ring128Tensor::Concrete(
                                    Self::abstract_kernel(x, y),
                                )),
                                (
                                    Symbolic::Symbolic {
                                        ty: Ty::Ring128TensorTy,
                                    },
                                    Symbolic::Symbolic {
                                        ty: Ty::Ring128TensorTy,
                                    },
                                ) => Symbolic::Symbolic {
                                    ty: Ty::Ring128TensorTy,
                                },
                                _ => unimplemented!(),
                            }
                            .into()
                        }
                        _ => unimplemented!(),
                    }
                })
            }

            _ => unimplemented!(),
        }
    }

    fn abstract_kernel<T>(x: RingTensor<T>, y: RingTensor<T>) -> RingTensor<T>
    where
        RingTensor<T>: Sub<RingTensor<T>, Output = RingTensor<T>>,
    {
        x - y
    }
}

trait PlacementSample<C: Context, O> {
    fn sample(&self, ctx: &C) -> O;
}

impl PlacementSample<ConcreteContext, Ring64Tensor> for HostPlacement {
    fn sample(&self, ctx: &ConcreteContext) -> Ring64Tensor {
        ctx.execute_nullary(
            RingSampleOp {
                ty: Ty::Ring64TensorTy,
                plc: Placement::Host(self.clone()),
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
                ty: Ty::Ring128TensorTy,
                plc: Placement::Host(self.clone()),
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
                plc: Placement::Host(self.clone()),
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
                plc: Placement::Host(self.clone()),
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
            (Placement::Host(_), Ty::Ring64TensorTy) => {
                Box::new(move || Ring64Tensor::Concrete(RingTensor(987654321)).into())
            }
            (Placement::Host(_), Ty::Ring128TensorTy) => {
                Box::new(move || Ring128Tensor::Concrete(RingTensor(987654321)).into())
            }
            _ => unimplemented!(),
        }
    }

    // TODO could we derive this from the return type of the closure returned by `compile`?
    // not sure this will work, seems like we need a Ty instead, which comes as part of
    // type checking.
    pub fn compile_symbolic(&self, ctx: &SymbolicContext) -> Box<dyn Fn() -> SymbolicValue> {
        match (&self.plc, &self.ty) {
            (Placement::Host(_), Ty::Ring64TensorTy) => {
                let op = self.clone();
                Box::new(move || {
                    // Ring64Tensor::Symbolic(op.clone().into()).into()
                    SymbolicValue::Ring64Tensor(Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    })
                })
            }
            (Placement::Host(_), Ty::Ring128TensorTy) => {
                let op = self.clone();
                Box::new(move || {
                    // Ring128Tensor::Symbolic(op.clone().into()).into()
                    SymbolicValue::Ring128Tensor(Symbolic::Symbolic {
                        ty: Ty::Ring128TensorTy,
                    })
                })
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
            [
                Ring128Tensor::Concrete(RingTensor(1)),
                Ring128Tensor::Concrete(RingTensor(2)),
            ],
            [
                Ring128Tensor::Concrete(RingTensor(2)),
                Ring128Tensor::Concrete(RingTensor(3)),
            ],
            [
                Ring128Tensor::Concrete(RingTensor(3)),
                Ring128Tensor::Concrete(RingTensor(1)),
            ],
        ],
    };

    let y = ReplicatedTensor {
        shares: [
            [
                Ring128Tensor::Concrete(RingTensor(1)),
                Ring128Tensor::Concrete(RingTensor(2)),
            ],
            [
                Ring128Tensor::Concrete(RingTensor(2)),
                Ring128Tensor::Concrete(RingTensor(3)),
            ],
            [
                Ring128Tensor::Concrete(RingTensor(3)),
                Ring128Tensor::Concrete(RingTensor(1)),
            ],
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

    let ctx = ConcreteContext {};
    let rep_plc = ReplicatedPlacement {
        players: ["alice".into(), "bob".into(), "carole".into()],
    };
    let z: ReplicatedTensor<_> = rep_plc.add(&ctx, &x, &y);
    println!("{:?}", z);

    let ctx = SymbolicContext {};
    let host_plc = HostPlacement {
        player: "alice".into(),
    };
    // let r = Ring64Tensor::Concrete(RingTensor(2));
    // let s = Ring64Tensor::Concrete(RingTensor(1));
    // let t = host_plc.sub(&ctx, &r, &s);
    // println!("{:?}", t);
    let r: Symbolic<Ring128Tensor> = host_plc.sample(&ctx);
    // println!("{:?}", r);

    let a: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> =
        Symbolic::Concrete(ReplicatedTensor {
            shares: [
                [
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                ],
                [
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                ],
                [
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                ],
            ],
        });
    let b: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> =
        Symbolic::Concrete(ReplicatedTensor {
            shares: [
                [
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                ],
                [
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                ],
                [
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                    Symbolic::Symbolic {
                        ty: Ty::Ring64TensorTy,
                    },
                ],
            ],
        });
    let c = rep_plc.add(&ctx, &a, &b);
    println!("{:?}", c);

    // assert!(false);
}

#[test]
fn test_rep_share() {
    let host_plc = HostPlacement {
        player: "alice".into(),
    };
    let rep_plc = ReplicatedPlacement {
        players: ["alice".into(), "bob".into(), "carole".into()],
    };

    {
        let ctx = ConcreteContext {};
        let x: Ring64Tensor = host_plc.sample(&ctx);
        let xe = rep_plc.share(&ctx, &x);
        println!("{:?}", xe);
    }

    {
        let ctx = SymbolicContext {};
        let x: Symbolic<Ring64Tensor> = host_plc.sample(&ctx);
        let xe = rep_plc.share(&ctx, &x);
        println!("{:?}", xe);
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

    assert!(false);
}

#[test]
fn test_rep_exec() {

    // let comp = r#"

    // "#.try_into().unwrap();

    // let exec = SymbolicExecutor;
    // exec.eval(comp);

    // assert!(false);
}
