#![allow(unused_macros)]

use crate::fixedpoint::{Float64Tensor, Convert};
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
    Float64TensorTy,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Value {
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    Shape(Shape),
    Seed(Seed),
    PrfKey(PrfKey),
    Nonce(Nonce),
    Float64Tensor(Float64Tensor),
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

impl From<Float64Tensor> for Value {
    fn from(v: Float64Tensor) -> Self {
        Value::Float64Tensor(v)
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

impl From<Value> for Float64Tensor {
    fn from(v: Value) -> Self {
        match v {
            Value::Float64Tensor(x) => x,
            _ => unimplemented!(),
        }
    }
}

pub enum TypedNullaryKernel<Y> {
    Function(fn() -> Y),
    Closure(Arc<dyn Fn() -> Y + Send + Sync>),
}

impl<Y> From<TypedNullaryKernel<Y>> for SyncKernel
where
    Y: 'static,
    Value: From<Y>,
{
    fn from(typed_kernel: TypedNullaryKernel<Y>) -> SyncKernel {
        match typed_kernel {
            TypedNullaryKernel::Function(k) => SyncKernel::Nullary(Arc::new(move || {
                let y = k();
                Value::from(y)
            })),
            TypedNullaryKernel::Closure(k) => {
                let k = k.clone();
                SyncKernel::Nullary(Arc::new(move || {
                    let y = k();
                    Value::from(y)
                }))
            }
        }
    }
}

pub enum TypedUnaryKernel<X0, Y> {
    Function(fn(X0) -> Y),
    Closure(Arc<dyn Fn(X0) -> Y + Send + Sync>),
}

impl<X0, Y> From<TypedUnaryKernel<X0, Y>> for SyncKernel
where
    X0: 'static + From<Value>,
    Y: 'static,
    Value: From<Y>,
{
    fn from(typed_kernel: TypedUnaryKernel<X0, Y>) -> SyncKernel {
        match typed_kernel {
            TypedUnaryKernel::Function(k) => SyncKernel::Unary(Arc::new(move |x0| {
                let x0 = X0::from(x0);
                let y = k(x0);
                Value::from(y)
            })),
            TypedUnaryKernel::Closure(k) => {
                let k = k.clone();
                SyncKernel::Unary(Arc::new(move |x0| {
                    let x0 = X0::from(x0);
                    let y = k(x0);
                    Value::from(y)
                }))
            }
        }
    }
}
pub enum TypedBinaryKernel<X0, X1, Y> {
    Function(fn(X0, X1) -> Y),
    Closure(Arc<dyn Fn(X0, X1) -> Y + Send + Sync>),
}

impl<X0, X1, Y> From<TypedBinaryKernel<X0, X1, Y>> for SyncKernel
where
    X0: 'static + From<Value>,
    X1: 'static + From<Value>,
    Y: 'static,
    Value: From<Y>,
{
    fn from(typed_kernel: TypedBinaryKernel<X0, X1, Y>) -> SyncKernel {
        match typed_kernel {
            TypedBinaryKernel::Function(k) => SyncKernel::Binary(Arc::new(move |x0, x1| {
                let x0 = X0::from(x0);
                let x1 = X1::from(x1);
                let y = k(x0, x1);
                Value::from(y)
            })),
            TypedBinaryKernel::Closure(k) => {
                let k = k.clone();
                SyncKernel::Binary(Arc::new(move |x0, x1| {
                    let x0 = X0::from(x0);
                    let x1 = X1::from(x1);
                    let y = k(x0, x1);
                    Value::from(y)
                }))
            }
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

macro_rules! unary_kernel {
    ($t0:ty) => {
        TypedUnaryKernel::Function(<Self as UnaryFunction<$t0>>::execute).into()
    };
    ($self:ident, $t0:ty) => {{
        let s = $self.clone();
        TypedUnaryKernel::Closure(Arc::new(move |x0| {
            <Self as UnaryClosure<$t0>>::execute(&s, x0)
        }))
        .into()
    }};
}

macro_rules! binary_kernel {
    ($t0:ty, $t1:ty) => {
        TypedBinaryKernel::Function(<Self as BinaryFunction<$t0, $t1>>::execute).into()
    };
    ($self:ident, $t0:ty, $t1:ty) => {
        let s = $self.clone();
        TypedBinaryKernel::Closure(Arc::new(move |x0, x1| {
            <Self as BinaryClosure<$t0, $t1>>::execute(&s, x0, x1)
        }))
        .into()
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

pub struct Session {
    pub id: u128,
}

pub enum SyncKernel {
    Nullary(Arc<dyn Fn() -> Value + Send + Sync>),
    Unary(Arc<dyn Fn(Value) -> Value + Send + Sync>),
    Binary(Arc<dyn Fn(Value, Value) -> Value + Send + Sync>),
    Ternary(Arc<dyn Fn(Value, Value, Value) -> Value + Send + Sync>),
    Variadic(Arc<dyn Fn(&[Value]) -> Value + Send + Sync>),
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
    Ternary(Box<dyn Fn(AsyncValue, AsyncValue, AsyncValue) -> AsyncValue>),
    Variadic(Box<dyn Fn(&[AsyncValue]) -> AsyncValue>),
}

fn remove_err<T, E>(r: Result<T, E>) -> Result<T, ()> {
    r.map_err(|_| ())
}

impl From<SyncKernel> for AsyncKernel {
    fn from(sync_kernel: SyncKernel) -> AsyncKernel {
        match sync_kernel {
            SyncKernel::Nullary(k) => AsyncKernel::Nullary(Box::new(move || {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let k = k.clone();
                let _task = tokio::spawn(async move {
                    let y = k();
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            SyncKernel::Unary(k) => AsyncKernel::Unary(Box::new(move |x0| {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let k = k.clone();
                let _task = tokio::spawn(async move {
                    let x0 = x0.await?;
                    let y = k(x0);
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            SyncKernel::Binary(k) => AsyncKernel::Binary(Box::new(move |x0, x1| {
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
            SyncKernel::Ternary(k) => AsyncKernel::Ternary(Box::new(move |x0, x1, x2| {
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
            // TODO
            _ => unimplemented!()
            // SyncKernel::Variadic(k) => AsyncKernel::Variadic(Box::new(move |mut xs| {
            //     let (sender, _) = tokio::sync::broadcast::channel(1);
            //     let subscriber = sender.clone();
            //     let k = k.clone();
            //     tokio::spawn(async move {
            //         let xs = futures::future::join_all(xs.iter().map(|xi| xi.recv())).await;
            //         let xs: Vec<Value> = xs.iter().map(|xi| xi.unwrap()).collect();
            //         let y = k(&xs);
            //         sender.send(y)
            //     });
            //     subscriber
            // })),
        }
    }
}

#[enum_dispatch]
trait Kernel {
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
    FixedpointRingEncode(FixedpointRingEncodeOp),
    FixedpointRingDecode(FixedpointRingDecodeOp),
    FixedpointRingMean(FixedpointRingMeanOp),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConstantOp {
    pub value: Value,
}

impl Kernel for ConstantOp {
    fn sync_kernel(&self) -> SyncKernel {
        let value = self.value.clone(); // TODO(Morten) avoid clone here
        SyncKernel::Nullary(Arc::new(move || value.clone()))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrimDeriveSeedOp {
    pub nonce: Nonce,
}

impl Kernel for PrimDeriveSeedOp {
    fn sync_kernel(&self) -> SyncKernel {
        unary_kernel!(self, PrfKey)
    }
}

impl UnaryClosure<PrfKey> for PrimDeriveSeedOp {
    type Output = Seed;
    fn execute(&self, key: PrfKey) -> Self::Output {
        let todo = crate::utils::derive_seed(&key.0, &self.nonce.0);
        Seed(todo.into())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrimGenPrfKeyOp;

impl Kernel for PrimGenPrfKeyOp {
    fn sync_kernel(&self) -> SyncKernel {
        SyncKernel::Nullary(Arc::new(move || {
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
                binary_kernel!(Ring64Tensor, Ring64Tensor)
            }
            (Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                binary_kernel!(Ring128Tensor, Ring128Tensor)
            }
            _ => unimplemented!(),
        }
    }
}

impl<T: Add<U>, U> BinaryFunction<T, U> for RingAddOp {
    type Output = <T as Add<U>>::Output;
    fn execute(x: T, y: U) -> Self::Output {
        x + y
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
                binary_kernel!(Ring64Tensor, Ring64Tensor)
            }
            (Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                binary_kernel!(Ring128Tensor, Ring128Tensor)
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
        SyncKernel::Binary(Arc::new(move |x, y| {
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
        SyncKernel::Binary(Arc::new(move |x, y| {
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
        SyncKernel::Unary(Arc::new(move |x| {
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
            Ty::Ring64TensorTy => unary_kernel!(Ring64Tensor),
            Ty::Ring128TensorTy => unary_kernel!(Ring128Tensor),
            _ => unimplemented!(),
        }
    }
}

impl UnaryFunction<Ring64Tensor> for RingShapeOp {
    type Output = Shape;
    fn execute(x: Ring64Tensor) -> Self::Output {
        Shape(x.0.shape().into()) // TODO(Morten) wrapping should not happen here
    }
}

impl UnaryFunction<Ring128Tensor> for RingShapeOp {
    type Output = Shape;
    fn execute(x: Ring128Tensor) -> Self::Output {
        Shape(x.0.shape().into()) // TODO(Morten) wrapping should not happen here
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
        SyncKernel::Unary(Arc::new(move |shape| match shape {
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
                TypedBinaryKernel::Function(move |shape: Shape, seed: Seed| {
                    Ring64Tensor::sample_uniform(&shape.0, &seed.0)
                })
                .into()
            }
            (Ty::Ring64TensorTy, Some(max_value)) if max_value == 1 => {
                TypedBinaryKernel::Function(move |shape: Shape, seed: Seed| {
                    Ring64Tensor::sample_bits(&shape.0, &seed.0)
                })
                .into()
            }
            _ => unimplemented!(), // TODO
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShlOp {
    amount: usize,
}

// TODO(Morten) rewrite
impl Kernel for RingShlOp {
    fn sync_kernel(&self) -> SyncKernel {
        let amount = self.amount;
        TypedUnaryKernel::Closure(Arc::new(move |x: Ring64Tensor| x << amount)).into()
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
        SyncKernel::Unary(Arc::new(move |x| {
            let x: Ring64Tensor = x.into();
            let y = x >> amount;
            y.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FixedpointRingEncodeOp {
    pub scaling_factor: u64,
}

impl Kernel for FixedpointRingEncodeOp {
    fn sync_kernel(&self) -> SyncKernel {
        unary_kernel!(self, Float64Tensor)
    }
}

impl UnaryClosure<Float64Tensor> for FixedpointRingEncodeOp {
    type Output = Ring64Tensor;
    fn execute(&self, x: Float64Tensor) -> Self::Output {
        Ring64Tensor::encode(&x, self.scaling_factor)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FixedpointRingDecodeOp {
    pub scaling_factor: u64,
}

impl Kernel for FixedpointRingDecodeOp {
    fn sync_kernel(&self) -> SyncKernel {
        unary_kernel!(self, Ring64Tensor)
    }
}

impl UnaryClosure<Ring64Tensor> for FixedpointRingDecodeOp {
    type Output = Float64Tensor;
    fn execute(&self, x: Ring64Tensor) -> Self::Output {
        Ring64Tensor::decode(&x, self.scaling_factor)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FixedpointRingMeanOp {
    pub axis: Option<usize>,
    pub scaling_factor: u64,
}

impl Kernel for FixedpointRingMeanOp {
    fn sync_kernel(&self) -> SyncKernel {
        unary_kernel!(self, Ring64Tensor)
    }
}

impl UnaryClosure<Ring64Tensor> for FixedpointRingMeanOp {
    type Output = Ring64Tensor;
    fn execute(&self, x: Ring64Tensor) -> Self::Output {
        crate::fixedpoint::ring_mean(x, self.axis, self.scaling_factor)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Placement {
    Host,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Operation {
    pub name: String,
    pub kind: Operator,
    pub inputs: Vec<String>,
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

impl Compile<CompiledOperation<Value>> for Operation {
    fn compile(&self) -> Result<CompiledOperation<Value>> {
        let operator_kernel: SyncKernel = self.kind.sync_kernel();
        match (operator_kernel, self.inputs.len()) {
            (SyncKernel::Nullary(k), 0) => {
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |_: &Environment<Value>| k()),
                })
            }
            (SyncKernel::Unary(k), 1) => {
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
            (SyncKernel::Binary(k), 2) => {
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
            (SyncKernel::Ternary(k), 3) => {
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
            (SyncKernel::Variadic(k), _) => {
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
            _ => Err(anyhow!("Failed to compile kernel for operation '{}' due to arity mismatch; {} inputs were given", self.name, self.inputs.len())),
        }
    }
}

impl Compile<CompiledOperation<AsyncValue>> for Operation {
    fn compile(&self) -> Result<CompiledOperation<AsyncValue>> {
        let operator_kernel: AsyncKernel = self.kind.async_kernel();
        match (operator_kernel, self.inputs.len()) {
            (AsyncKernel::Nullary(k), 0) => {
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |_: &Environment<AsyncValue>| k()),
                })
            }
            (AsyncKernel::Unary(k), 1) => {
                let x0_name = self.inputs[0].clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<AsyncValue>| {
                        let x0 = env.get(&x0_name).unwrap().clone();
                        k(x0)
                    }),
                })
            }
            (AsyncKernel::Binary(k), 2) => {
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
            (AsyncKernel::Ternary(k), 3) => {
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
            (AsyncKernel::Variadic(k), _) => {
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
            _ => Err(anyhow!("Failed to compile async kernel for operation '{}' due to arity mismatch; {} inputs were given", self.name, self.inputs.len())),
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

pub struct CompiledComputation<V>(Arc<dyn Fn(Environment<V>) -> Environment<V>>);

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
            .map(|op| op.compile())
            .collect::<Result<Vec<_>>>()?;
        // TODO(Morten) we want to sort topologically here, outside the closure
        // TODO(Morten) do we want to insert instructions for when values can be dropped from the environment?
        Ok(CompiledComputation(Arc::new(
            move |mut env: Environment<V>| {
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
    pub fn apply(&self, env: Environment<V>) -> Environment<V> {
        (self.0)(env)
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
        let _env = compiled_comp.apply(args);
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

        println!("Running");
        rt.block_on(async {
            let _env = compiled_comp.apply(args);
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
