#![allow(unused_macros)]

use crate::prng::AesRng;
use crate::ring::{Dot, Ring128Tensor, Ring64Tensor, Sample};
use anyhow::{anyhow, Result};
use enum_dispatch::enum_dispatch;
use futures::future::{Map, Shared};
use futures::prelude::*;
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{Add, Sub};
use std::sync::Arc;
use tokio::sync::oneshot;
use rayon::prelude::*;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Seed(pub Vec<u8>);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Shape(pub Vec<usize>);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrfKey(pub Vec<u8>);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Nonce(pub Vec<u8>);

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
pub enum Ty {
    Ring64TensorTy,
    Ring128TensorTy,
    ShapeTy,
    SeedTy,
    PrfKeyTy,
    NonceTy,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Value {
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    Shape(Shape),
    Seed(Seed),
    PrfKey(PrfKey),
    Nonce(Nonce),
}

impl From<Ring64Tensor> for Value {
    fn from(v: Ring64Tensor) -> Self {
        Value::Ring64Tensor(v)
    }
}

impl From<Ring128Tensor> for Value {
    fn from(v: Ring128Tensor) -> Self {
        Value::Ring128Tensor(v)
    }
}

impl From<Shape> for Value {
    fn from(v: Shape) -> Self {
        Value::Shape(v)
    }
}

impl From<Seed> for Value {
    fn from(v: Seed) -> Self {
        Value::Seed(v)
    }
}

impl From<PrfKey> for Value {
    fn from(v: PrfKey) -> Self {
        Value::PrfKey(v)
    }
}

impl From<Nonce> for Value {
    fn from(v: Nonce) -> Self {
        Value::Nonce(v)
    }
}

// TODO(Morten) all the From<Value> below should be TryFrom instead, with proper error handling

impl From<Value> for Ring64Tensor {
    fn from(v: Value) -> Self {
        match v {
            Value::Ring64Tensor(x) => x,
            _ => panic!("Cannot convert {:?} to Ring64Tensor", v),
        }
    }
}

impl<'v> From<&'v Value> for &'v Ring64Tensor {
    fn from(v: &'v Value) -> Self {
        match v {
            Value::Ring64Tensor(x) => x,
            _ => unimplemented!(),
        }
    }
}

impl From<Value> for Ring128Tensor {
    fn from(v: Value) -> Self {
        match v {
            Value::Ring128Tensor(x) => x,
            _ => unimplemented!(),
        }
    }
}

impl From<Value> for Shape {
    fn from(v: Value) -> Self {
        match v {
            Value::Shape(x) => x,
            _ => unimplemented!(),
        }
    }
}

impl From<Value> for Seed {
    fn from(v: Value) -> Self {
        match v {
            Value::Seed(x) => x,
            _ => unimplemented!(),
        }
    }
}

impl From<Value> for PrfKey {
    fn from(v: Value) -> Self {
        match v {
            Value::PrfKey(x) => x,
            _ => unimplemented!(),
        }
    }
}

impl From<Value> for Nonce {
    fn from(v: Value) -> Self {
        match v {
            Value::Nonce(x) => x,
            _ => unimplemented!(),
        }
    }
}

pub trait NullaryFunction {
    type Output;
    fn execute() -> Self::Output;
}

pub trait NullaryClosure
where
    Self: Clone,
{
    type Output;
    fn execute(&self) -> Self::Output;
}

pub trait UnaryFunction<X0> {
    type Output;
    fn execute(x0: X0) -> Self::Output;
}

pub trait UnaryClosure<X0>
where
    Self: Clone,
{
    type Output;
    fn execute(&self, x0: X0) -> Self::Output;
}

pub trait BinaryFunction<X0, X1> {
    type Output;
    fn execute(x0: X0, x1: X1) -> Self::Output;
}

pub trait BinaryClosure<X0, X1>
where
    Self: Clone,
{
    type Output;
    fn execute(&self, x0: X0, x1: X1) -> Self::Output;
}

macro_rules! function_kernel {
    ($t0:ty, $f:expr) => {
        SyncKernel::UnaryFunction(|x0| {
            let x0 = <$t0 as From<Value>>::from(x0);
            let g: fn($t0) -> _ = $f;
            let y = g(x0);
            Value::from(y)
        })
    };
}

macro_rules! closure_kernel {
    ($t0:ty, $f:expr) => {
        SyncKernel::UnaryClosure(Arc::new(move |x0| {
            let x0 = <$t0 as From<Value>>::from(x0);
            let y = $f(x0);
            Value::from(y)
        }))
    };
}

macro_rules! binary_kernel {
    ($t0:ty, $t1:ty, $f:expr) => {
        SyncKernel::BinaryFunction(|x0, x1| {
            let x0 = <$t0 as From<Value>>::from(x0);
            let x1 = <$t1 as From<Value>>::from(x1);
            let y = $f(x0, x1);
            Value::from(y)
        })
    };
    ($self:ident, $t0:ty, $t1:ty, $f:expr) => {
        let s: Self = $self.clone();
        SyncKernel::BinaryClosure(Arc::new(move |x0, x1| {
            let x0 = <$t0 as From<Value>>::from(x0);
            let x1 = <$t1 as From<Value>>::from(x1);
            let y = $f(s, x0, x1);
            // let y = <Self as BinaryFunction<$t0, $t1>>::execute(x0, x1);
            Value::from(y)
        }))
    };
}

macro_rules! ternary_kernel {
    ($t0:ty, $t1:ty, $t2:ty) => {
        TypedTernaryKernel::Function(<Self as TernaryFunction<$t0, $t1, $t2>>::execute).into()
    };
    ($self:ident, $t0:ty, $t1:ty, $t2:ty) => {
        let s = $self.clone();
        TypedTernaryKernel::Closure(Arc::new(move |x0, x1, x2| {
            <Self as TernaryClosure<$t0, $t1, $t2>>::execute(&s, x0, x1, x2)
        }))
        .into()
    };
}

#[derive(Clone, Debug)]
pub struct Session {
    pub id: u128,
}

pub enum SyncKernel {
    NullaryClosure(Arc<dyn Fn() -> Value + Send + Sync>),
    NullaryFunction(fn() -> Value),
    UnaryClosure(Arc<dyn Fn(Value) -> Value + Send + Sync>),
    UnaryFunction(fn(Value) -> Value),
    BinaryClosure(Arc<dyn Fn(Value, Value) -> Value + Send + Sync>),
    BinaryFunction(fn(Value, Value) -> Value),
    TernaryClosure(Arc<dyn Fn(Value, Value, Value) -> Value + Send + Sync>),
    TernaryFunction(fn(Value, Value, Value) -> Value),
    VariadicClosure(Arc<dyn Fn(&[Value]) -> Value + Send + Sync>),
    VariadicFunction(fn(&[Value]) -> Value),
}

pub type AsyncValue = Shared<
    Map<
        oneshot::Receiver<Value>,
        fn(Result<Value, oneshot::error::RecvError>) -> Result<Value, ()>,
    >,
>;

pub enum AsyncKernel {
    Nullary(Box<dyn Fn() -> AsyncValue>),
    Unary(Box<dyn Fn(AsyncValue) -> AsyncValue>),
    Binary(Box<dyn Fn(AsyncValue, AsyncValue) -> AsyncValue>),
    BinaryFunction(fn(AsyncValue, AsyncValue) -> AsyncValue),
    Ternary(Box<dyn Fn(AsyncValue, AsyncValue, AsyncValue) -> AsyncValue>),
    Variadic(Box<dyn Fn(&[AsyncValue]) -> AsyncValue>),
}

fn remove_err<T, E>(r: Result<T, E>) -> Result<T, ()> {
    r.map_err(|_| ())
}

impl From<SyncKernel> for AsyncKernel {
    fn from(sync_kernel: SyncKernel) -> AsyncKernel {
        match sync_kernel {
            SyncKernel::NullaryFunction(k) => AsyncKernel::Nullary(Box::new(move || {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let _task = tokio::spawn(async move {
                    let y = k();
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            SyncKernel::NullaryClosure(k) => AsyncKernel::Nullary(Box::new(move || {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let k = k.clone();
                let _task = tokio::spawn(async move {
                    let y = k();
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            SyncKernel::UnaryFunction(k) => AsyncKernel::Unary(Box::new(move |x0| {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let _task = tokio::spawn(async move {
                    let x0 = x0.await?;
                    let y = k(x0);
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            SyncKernel::UnaryClosure(k) => AsyncKernel::Unary(Box::new(move |x0| {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let k = k.clone();
                let _task = tokio::spawn(async move {
                    let x0 = x0.await?;
                    let y = k(x0);
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            SyncKernel::BinaryFunction(k) => AsyncKernel::Binary(Box::new(move |x0, x1| {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let _task = tokio::spawn(async move {
                    let x0 = x0.await?;
                    let x1 = x1.await?;
                    let y = k(x0, x1);
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            SyncKernel::BinaryClosure(k) => AsyncKernel::Binary(Box::new(move |x0, x1| {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let k = k.clone();
                let _task = tokio::spawn(async move {
                    let x0 = x0.await?;
                    let x1 = x1.await?;
                    let y = k(x0, x1);
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            SyncKernel::TernaryFunction(k) => AsyncKernel::Ternary(Box::new(move |x0, x1, x2| {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let _task = tokio::spawn(async move {
                    let x0 = x0.await?;
                    let x1 = x1.await?;
                    let x2 = x2.await?;
                    let y = k(x0, x1, x2);
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            SyncKernel::TernaryClosure(k) => AsyncKernel::Ternary(Box::new(move |x0, x1, x2| {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let k = k.clone();
                let _task = tokio::spawn(async move {
                    let x0 = x0.await?;
                    let x1 = x1.await?;
                    let x2 = x2.await?;
                    let y = k(x0, x1, x2);
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            SyncKernel::VariadicFunction(k) => unimplemented!(),
            SyncKernel::VariadicClosure(k) => unimplemented!(),
        }
    }
}

#[enum_dispatch]
pub trait Kernel {
    fn sync_kernel(&self) -> SyncKernel;
    fn async_kernel(&self) -> AsyncKernel {
        AsyncKernel::from(self.sync_kernel())
    }
}

#[enum_dispatch(Kernel)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Operator {
    Constant(ConstantOp),
    RingAdd(RingAddOp),
    RingSub(RingSubOp),
    RingMul(RingMulOp),
    RingDot(RingDotOp),
    RingSum(RingSumOp),
    RingShape(RingShapeOp),
    RingSample(RingSampleOp),
    RingFill(RingFillOp),
    RingShl(RingShlOp),
    RingShr(RingShrOp),
    PrimDeriveSeed(PrimDeriveSeedOp),
    PrimGenPrfKey(PrimGenPrfKeyOp),
    Send(SendOp),
    Recv(RecvOp),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SendOp {
    pub rendezvous_key: String,
}

impl Kernel for SendOp {
    fn sync_kernel(&self) -> SyncKernel {
        unimplemented!()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RecvOp {
    pub rendezvous_key: String,
}

impl Kernel for RecvOp {
    fn sync_kernel(&self) -> SyncKernel {
        unimplemented!()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConstantOp {
    pub value: Value,
}

impl Kernel for ConstantOp {
    fn sync_kernel(&self) -> SyncKernel {
        let value = self.value.clone(); // TODO(Morten) avoid clone here
        SyncKernel::NullaryClosure(Arc::new(move || value.clone()))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrimDeriveSeedOp {
    pub nonce: Nonce,
}

impl Kernel for PrimDeriveSeedOp {
    fn sync_kernel(&self) -> SyncKernel {
        let nonce = self.nonce.0.clone();
        closure_kernel!(PrfKey, |key: PrfKey| {
            let todo = crate::utils::derive_seed(&key.0, &nonce);
            Seed(todo.into())
        })
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrimGenPrfKeyOp;

impl Kernel for PrimGenPrfKeyOp {
    fn sync_kernel(&self) -> SyncKernel {
        SyncKernel::NullaryClosure(Arc::new(move || {
            // TODO(Morten) we shouldn't have core logic directly in kernels
            let raw_key = AesRng::generate_random_key();
            Value::PrfKey(PrfKey(raw_key.into()))
        }))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingAddOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

impl Kernel for RingAddOp {
    fn sync_kernel(&self) -> SyncKernel {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                binary_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x + y)
            }
            (Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                binary_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x + y)
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSubOp {
    lhs: Ty,
    rhs: Ty,
}

impl Kernel for RingSubOp {
    fn sync_kernel(&self) -> SyncKernel {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                binary_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x - y)
            }
            (Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                binary_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x - y)
            }
            _ => unimplemented!(),
        }
    }
}

impl<T: Sub<U>, U> BinaryFunction<T, U> for RingSubOp {
    type Output = <T as Sub<U>>::Output;
    fn execute(x: T, y: U) -> Self::Output {
        x - y
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingMulOp;

// TODO(Morten) rewrite
impl Kernel for RingMulOp {
    fn sync_kernel(&self) -> SyncKernel {
        SyncKernel::BinaryClosure(Arc::new(move |x, y| {
            let x: Ring64Tensor = x.into();
            let y: Ring64Tensor = y.into();
            let z = x * y;
            z.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingDotOp;

// TODO(Morten) rewrite
impl Kernel for RingDotOp {
    fn sync_kernel(&self) -> SyncKernel {
        SyncKernel::BinaryClosure(Arc::new(move |x, y| {
            let x: Ring64Tensor = x.into();
            let y: Ring64Tensor = y.into();
            let z = x.dot(y);
            z.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSumOp {
    axis: Option<usize>, // TODO(Morten) use platform independent type instead?
}

// TODO(Morten) rewrite
impl Kernel for RingSumOp {
    fn sync_kernel(&self) -> SyncKernel {
        let axis = self.axis;
        SyncKernel::UnaryClosure(Arc::new(move |x| {
            let x: Ring64Tensor = x.into();
            let y = x.sum(axis);
            Value::from(y)
        }))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShapeOp {
    ty: Ty,
}

trait TypeCheck {
    fn output_type(&self) -> Ty;
}

impl TypeCheck for RingShapeOp {
    fn output_type(&self) -> Ty {
        self.ty
    }
}

// TODO(Morten) rewrite
impl Kernel for RingShapeOp {
    fn sync_kernel(&self) -> SyncKernel {
        match self.ty {
            Ty::Ring64TensorTy => function_kernel!(Ring64Tensor, |x| {
                Shape(x.0.shape().into()) // TODO(Morten) wrapping should not happen here
            }),
            Ty::Ring128TensorTy => function_kernel!(Ring128Tensor, |x| {
                Shape(x.0.shape().into()) // TODO(Morten) wrapping should not happen here
            }),
            _ => unimplemented!(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingFillOp {
    value: u64,
}

// TODO(Morten) rewrite
impl Kernel for RingFillOp {
    fn sync_kernel(&self) -> SyncKernel {
        let value = self.value;
        SyncKernel::UnaryClosure(Arc::new(move |shape| match shape {
            Value::Shape(shape) => Value::Ring64Tensor(Ring64Tensor::fill(&shape.0, value)), // TODO(Morten) should not call .0 here
            _ => unimplemented!(),
        }))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSampleOp {
    pub output: Ty,
    pub max_value: Option<usize>,
}

// TODO(Morten) rewrite
impl Kernel for RingSampleOp {
    fn sync_kernel(&self) -> SyncKernel {
        match (self.output, self.max_value) {
            (Ty::Ring64TensorTy, None) => {
                binary_kernel!(Shape, Seed, |shape: Shape, seed: Seed| Ring64Tensor::sample_uniform(&shape.0, &seed.0))
            }
            (Ty::Ring64TensorTy, Some(max_value)) if max_value == 1 => {
                binary_kernel!(Shape, Seed, |shape: Shape, seed: Seed| Ring64Tensor::sample_bits(&shape.0, &seed.0))
            }
            _ => unimplemented!(), // TODO
        }
    }
}

// impl BinaryFunction<Shape, Seed>

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShlOp {
    pub amount: usize,
}

// TODO(Morten) rewrite
impl Kernel for RingShlOp {
    fn sync_kernel(&self) -> SyncKernel {
        let amount = self.amount;
        closure_kernel!(Ring64Tensor, |x| x << amount)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShrOp {
    amount: usize,
}

// TODO(Morten) rewrite
impl Kernel for RingShrOp {
    fn sync_kernel(&self) -> SyncKernel {
        let amount = self.amount;
        SyncKernel::UnaryClosure(Arc::new(move |x| {
            let x: Ring64Tensor = x.into();
            let y = x >> amount;
            y.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Placement {
    Host,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum OperatorRef {
    Reference(usize),
    Inlined(Operator),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Operation {
    pub name: String,
    // pub kind: OperatorRef,
    pub kind: Operator,
    pub inputs: Vec<String>,  // TODO(Morten) use indices instead of strings?
    pub placement: Placement,
}

pub struct CompiledOperation<V> {
    name: String,
    kernel: Box<dyn Fn(&Environment<V>) -> V>,
}

impl<V> Apply<V> for CompiledOperation<V> {
    fn apply(&self, inputs: &Environment<V>) -> V {
        (self.kernel)(inputs)
    }
}

fn check_arity<T>(operation_name: &str, inputs: &[T], arity: usize) -> Result<()> {
    if inputs.len() != arity {
        Err(anyhow!("Arity mismatch for operation '{}'; operator expects {} arguments but were given {}", operation_name, arity, inputs.len()))
    } else {
        Ok(())
    }
}

impl Compile<CompiledOperation<Value>> for Operation {
    fn compile(&self) -> Result<CompiledOperation<Value>> {
        let operator_kernel: SyncKernel = self.kind.sync_kernel();
        match operator_kernel {
            SyncKernel::NullaryFunction(k) => {
                check_arity(&self.name, &self.inputs, 0)?;
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |_: &Environment<Value>| k()),
                })
            }
            SyncKernel::NullaryClosure(k) => {
                check_arity(&self.name, &self.inputs, 0)?;
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |_: &Environment<Value>| k()),
                })
            }
            SyncKernel::UnaryFunction(k) => {
                check_arity(&self.name, &self.inputs, 1)?;
                let x0_name = self.inputs[0].clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<Value>| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).unwrap().clone();
                        k(x0)
                    }),
                })
            }
            SyncKernel::UnaryClosure(k) => {
                check_arity(&self.name, &self.inputs, 1)?;
                let x0_name = self.inputs[0].clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<Value>| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).unwrap().clone();
                        k(x0)
                    }),
                })
            }
            SyncKernel::BinaryFunction(k) => {
                check_arity(&self.name, &self.inputs, 2)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<Value>| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).unwrap().clone();
                        let x1 = env.get(&x1_name).unwrap().clone();
                        k(x0, x1)
                    }),
                })
            }
            SyncKernel::BinaryClosure(k) => {
                check_arity(&self.name, &self.inputs, 2)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<Value>| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).unwrap().clone();
                        let x1 = env.get(&x1_name).unwrap().clone();
                        k(x0, x1)
                    }),
                })
            }
            SyncKernel::TernaryFunction(k) => {
                check_arity(&self.name, &self.inputs, 3)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                let x2_name = self.inputs[2].clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).unwrap().clone();
                        let x1 = env.get(&x1_name).unwrap().clone();
                        let x2 = env.get(&x2_name).unwrap().clone();
                        k(x0, x1, x2)
                    }),
                })
            }
            SyncKernel::TernaryClosure(k) => {
                check_arity(&self.name, &self.inputs, 3)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                let x2_name = self.inputs[2].clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).unwrap().clone();
                        let x1 = env.get(&x1_name).unwrap().clone();
                        let x2 = env.get(&x2_name).unwrap().clone();
                        k(x0, x1, x2)
                    }),
                })
            }
            SyncKernel::VariadicFunction(k) => {
                let inputs = self.inputs.clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env| {
                        let xs: Vec<_> = inputs
                            .iter()
                            .map(|input| env.get(input).unwrap())
                            .cloned() // TODO(Morten) avoid cloning
                            .collect();
                        k(&xs)
                    }),
                })
            }
            SyncKernel::VariadicClosure(k) => {
                let inputs = self.inputs.clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env| {
                        let xs: Vec<_> = inputs
                            .iter()
                            .map(|input| env.get(input).unwrap())
                            .cloned() // TODO(Morten) avoid cloning
                            .collect();
                        k(&xs)
                    }),
                })
            }
        }
    }
}

impl Compile<CompiledOperation<AsyncValue>> for Operation {
    fn compile(&self) -> Result<CompiledOperation<AsyncValue>> {
        let operator_kernel: AsyncKernel = self.kind.async_kernel();
        match operator_kernel {
            AsyncKernel::Nullary(k) => {
                check_arity(&self.name, &self.inputs, 0)?;
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |_: &Environment<AsyncValue>| k()),
                })
            }
            AsyncKernel::Unary(k) => {
                check_arity(&self.name, &self.inputs, 1)?;
                let x0_name = self.inputs[0].clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<AsyncValue>| {
                        let x0 = env.get(&x0_name).unwrap().clone();
                        k(x0)
                    }),
                })
            }
            AsyncKernel::Binary(k) => {
                check_arity(&self.name, &self.inputs, 2)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<AsyncValue>| {
                        let x0 = env.get(&x0_name).unwrap().clone();
                        let x1 = env.get(&x1_name).unwrap().clone();
                        k(x0, x1)
                    }),
                })
            }
            AsyncKernel::BinaryFunction(k) => {
                check_arity(&self.name, &self.inputs, 2)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<AsyncValue>| {
                        let x0 = env.get(&x0_name).unwrap().clone();
                        let x1 = env.get(&x1_name).unwrap().clone();
                        k(x0, x1)
                    }),
                })
            }
            AsyncKernel::Ternary(k) => {
                check_arity(&self.name, &self.inputs, 3)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                let x2_name = self.inputs[2].clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env| {
                        let x0 = env.get(&x0_name).unwrap().clone();
                        let x1 = env.get(&x1_name).unwrap().clone();
                        let x2 = env.get(&x2_name).unwrap().clone();
                        k(x0, x1, x2)
                    }),
                })
            }
            AsyncKernel::Variadic(k) => {
                let inputs = self.inputs.clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env| {
                        let xs: Vec<_> = inputs
                            .iter()
                            .map(|input| env.get(input).unwrap())
                            .cloned()
                            .collect();
                        k(&xs)
                    }),
                })
            }
        }
    }
}

impl Operation {
    pub fn apply(&self, env: &Environment<Value>) -> Result<Value> {
        let compiled: CompiledOperation<Value> = self.compile()?;
        Ok(compiled.apply(env))
    }

    pub fn apply_and_insert(&self, env: &mut Environment<Value>) -> Result<()> {
        let value = self.apply(env)?;
        env.insert(self.name.clone(), value);
        Ok(())
    }

    pub fn type_check(&self, _env: &Environment<Ty>) -> Ty {
        unimplemented!()
    }
}

pub struct Computation {
    // pub constants: Vec<Value>,
    // pub operators: Vec<Operator>,
    pub operations: Vec<Operation>,
}

impl Computation {
    pub fn toposort(&self) -> Result<Computation> {
        let mut graph = Graph::<String, ()>::new();

        let mut vertex_map: HashMap<String, NodeIndex> = HashMap::new();
        let mut inv_map: HashMap<NodeIndex, usize> = HashMap::new();

        for (i, op) in self.operations.iter().enumerate() {
            let vertex = graph.add_node(op.name.clone());

            vertex_map.insert(op.name.clone(), vertex);
            inv_map.insert(vertex, i);
        }

        for op in self.operations.iter() {
            for ins in op.inputs.iter() {
                graph.add_edge(vertex_map[ins], vertex_map[&op.name], ());
            }
        }

        let toposort = toposort(&graph, None)
            .map_err(|_| anyhow!("There is a cycle detected in the runtime graph"))?;

        let operations = toposort
            .iter()
            .map(|node| self.operations[inv_map[node]].clone())
            .collect();

        Ok(Computation { operations })
    }
}

pub struct CompiledComputation<V>(Arc<dyn Fn(Session, Environment<V>) -> Environment<V>>);

trait Apply<V> {
    fn apply(&self, env: &Environment<V>) -> V;
}

pub trait Compile<C> {
    fn compile(&self) -> Result<C>;
}

impl<V: 'static> Compile<CompiledComputation<V>> for Computation
where
    Operation: Compile<CompiledOperation<V>>,
{
    fn compile(&self) -> Result<CompiledComputation<V>> {
        // TODO(Morten) type check computation
        let compiled_ops: Vec<CompiledOperation<V>> = self
            .operations
            .iter()
            // .par_iter()
            .map(|op| op.compile())
            .collect::<Result<Vec<_>>>()?;
        // TODO(Morten) we want to sort topologically here, outside the closure
        // TODO(Morten) do we want to insert instructions for when values can be dropped from the environment?
        Ok(CompiledComputation(Arc::new(
            move |sess: Session, mut env: Environment<V>| {
                for compiled_op in compiled_ops.iter() {
                    let value = compiled_op.apply(&env);
                    env.insert(compiled_op.name.clone(), value);
                }
                env
            },
        )))
    }
}

impl<V> CompiledComputation<V> {
    pub fn apply(&self, sess: Session, env: Environment<V>) -> Environment<V> {
        (self.0)(sess, env)
    }
}

pub type Environment<V> = HashMap<String, V>;

/// In-order single-threaded executor.
///
/// This executor evaluates the operations of computations in-order, raising an error
/// in case data dependencies are not respected. This executor is intended for debug
/// and development only due to its unforgiving but highly predictable behaviour.
pub struct EagerExecutor;

impl EagerExecutor {
    pub fn run_computation(&self, comp: &Computation, args: Environment<Value>) -> Result<()> {
        let compiled_comp: CompiledComputation<Value> = comp.compile()?;
        let sess = Session{ id: 12345 };
        let _env = compiled_comp.apply(sess, args);
        println!("Done");
        Ok(())
    }
}

pub struct AsyncExecutor;

impl AsyncExecutor {
    pub fn run_computation(&self, comp: &Computation, args: Environment<AsyncValue>) -> Result<()> {
        let compiled_comp: CompiledComputation<AsyncValue> = comp.compile()?;
        println!("Compiled");

        // let rt = tokio::runtime::Runtime::new()?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let sess = Session{ id: 12345 };

        println!("Running");
        rt.block_on(async {
            let _env = compiled_comp.apply(sess, args);
            // let vals = futures::future::join_all(
            //     env.values().map(|op| op.clone()).collect::<Vec<_>>()).await;
            println!("Done");
        });
        Ok(())
    }
}

#[test]
fn test_foo() {
    use maplit::hashmap;

    let mut env = hashmap![];

    let key_op = Operation {
        name: "key".into(),
        kind: Operator::PrimGenPrfKey(PrimGenPrfKeyOp),
        inputs: vec![],
        placement: Placement::Host,
    };

    let x_seed_op = Operation {
        name: "x_seed".into(),
        kind: Operator::PrimDeriveSeed(PrimDeriveSeedOp {
            nonce: Nonce(vec![1, 2, 3]),
        }),
        inputs: vec!["key".into()],
        placement: Placement::Host,
    };

    let x_shape_op = Operation {
        name: "x_shape".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Shape(Shape(vec![2, 3])),
        }),
        inputs: vec![],
        placement: Placement::Host,
    };

    let sample_ops: Vec<_> = (0..100000)
        // .into_par_iter()
        .map(|i| {
            // Operation {
            //     name: format!("x{}", i),
            //     kind: Operator::RingSample(RingSampleOp { output: Ty::Ring64TensorTy, max_value: None }),
            //     inputs: vec!["x_shape".into(), "x_seed".into()],
            //     placement: Placement::Host,
            // }
            Operation {
                name: "key".into(),
                kind: Operator::PrimGenPrfKey(PrimGenPrfKeyOp),
                inputs: vec![],
                placement: Placement::Host,
            }
        })
        .collect();

    let comp = Computation {
        // operations: [vec![key_op, x_seed_op, x_shape_op], sample_ops].concat(),
        operations: sample_ops,
    }
    .toposort()
    .unwrap();

    let exec = AsyncExecutor;
    exec.run_computation(&comp, env).ok();
    // assert!(false);
}
