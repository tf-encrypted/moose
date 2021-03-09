use crate::prng::AesRng;
use crate::ring::{Dot, Ring128Tensor, Ring64Tensor, Sample};
use anyhow::{anyhow, Result};
use enum_dispatch::enum_dispatch;
use futures::prelude::*;
use maplit::hashmap;
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub};
use std::sync::Arc;
use std::{collections::HashMap, convert::TryFrom, marker::PhantomData};
use tokio;
use tokio::sync::oneshot;
use futures::future::{Map, Shared};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Seed(Vec<u8>);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Shape(Vec<usize>);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrfKey(Vec<u8>);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Nonce(Vec<u8>);

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

enum TypedNullaryKernel<Y> {
    Function(fn() -> Y),
    Closure(Arc<dyn Fn() -> Y + Send + Sync>),
}

impl<Y> From<TypedNullaryKernel<Y>> for Kernel
where
    Y: 'static,
    Value: From<Y>,
{
    fn from(typed_kernel: TypedNullaryKernel<Y>) -> Kernel {
        match typed_kernel {
            TypedNullaryKernel::Function(k) => Kernel::Nullary(Arc::new(move || {
                let y = k();
                Value::from(y)
            })),
            TypedNullaryKernel::Closure(k) => {
                let k = k.clone();
                Kernel::Nullary(Arc::new(move || {
                    let y = k();
                    Value::from(y)
                }))
            }
        }
    }
}

enum TypedUnaryKernel<X0, Y> {
    Function(fn(X0) -> Y),
    Closure(Arc<dyn Fn(X0) -> Y + Send + Sync>),
}

impl<X0, Y> From<TypedUnaryKernel<X0, Y>> for Kernel
where
    X0: 'static + From<Value>,
    Y: 'static,
    Value: From<Y>,
{
    fn from(typed_kernel: TypedUnaryKernel<X0, Y>) -> Kernel {
        match typed_kernel {
            TypedUnaryKernel::Function(k) => Kernel::Unary(Arc::new(move |x0| {
                let x0 = X0::from(x0);
                let y = k(x0);
                Value::from(y)
            })),
            TypedUnaryKernel::Closure(k) => {
                let k = k.clone();
                Kernel::Unary(Arc::new(move |x0| {
                    let x0 = X0::from(x0);
                    let y = k(x0);
                    Value::from(y)
                }))
            }
        }
    }
}
enum TypedBinaryKernel<X0, X1, Y> {
    Function(fn(X0, X1) -> Y),
    Closure(Arc<dyn Fn(X0, X1) -> Y + Send + Sync>),
}

impl<X0, X1, Y> From<TypedBinaryKernel<X0, X1, Y>> for Kernel
where
    X0: 'static + From<Value>,
    X1: 'static + From<Value>,
    Y: 'static,
    Value: From<Y>,
{
    fn from(typed_kernel: TypedBinaryKernel<X0, X1, Y>) -> Kernel {
        match typed_kernel {
            TypedBinaryKernel::Function(k) => Kernel::Binary(Arc::new(move |x0, x1| {
                let x0 = X0::from(x0);
                let x1 = X1::from(x1);
                let y = k(x0, x1);
                Value::from(y)
            })),
            TypedBinaryKernel::Closure(k) => {
                let k = k.clone();
                Kernel::Binary(Arc::new(move |x0, x1| {
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

pub enum Kernel {
    Nullary(Arc<dyn Fn() -> Value + Send + Sync>),
    Unary(Arc<dyn Fn(Value) -> Value + Send + Sync>),
    Binary(Arc<dyn Fn(Value, Value) -> Value + Send + Sync>),
    Ternary(Arc<dyn Fn(Value, Value, Value) -> Value + Send + Sync>),
    Variadic(Arc<dyn Fn(&[Value]) -> Value + Send + Sync>),
}

pub type AsyncValue = Shared<Map<oneshot::Receiver<Value>, fn(Result<Value, oneshot::error::RecvError>) -> Result<Value, ()>>>;

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

impl From<Kernel> for AsyncKernel {
    fn from(sync_kernel: Kernel) -> AsyncKernel {
        match sync_kernel {
            Kernel::Nullary(k) => AsyncKernel::Nullary(Box::new(move || {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let k = k.clone();
                tokio::spawn(async move {
                    let y = k();
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            Kernel::Unary(k) => AsyncKernel::Unary(Box::new(move |x0| {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let k = k.clone();
                tokio::spawn(async move {
                    let x0 = x0.await?;
                    let y = k(x0);
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            Kernel::Binary(k) => AsyncKernel::Binary(Box::new(move |x0, x1| {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let k = k.clone();
                tokio::spawn(async move {
                    let x0 = x0.await?;
                    let x1 = x1.await?;
                    let y = k(x0, x1);
                    sender.send(y).map_err(|_| ())
                });
                receiver.map(remove_err as fn(_) -> _).shared()
            })),
            Kernel::Ternary(k) => AsyncKernel::Ternary(Box::new(move |x0, x1, x2| {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                let k = k.clone();
                tokio::spawn(async move {
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
            // Kernel::Variadic(k) => AsyncKernel::Variadic(Box::new(move |mut xs| {
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
trait Compile {
    fn compile(&self) -> Kernel;
    fn async_compile(&self) -> AsyncKernel {
        self.compile().into()
    }
}

#[enum_dispatch(Compile)]
#[derive(Serialize, Deserialize, Debug)]
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
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConstantOp {
    value: Value,
}

impl Compile for ConstantOp {
    fn compile(&self) -> Kernel {
        let value = self.value.clone(); // TODO(Morten) avoid clone here
        Kernel::Nullary(Arc::new(move || value.clone()))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrimDeriveSeedOp {
    nonce: Nonce,
}

impl Compile for PrimDeriveSeedOp {
    fn compile(&self) -> Kernel {
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

impl Compile for PrimGenPrfKeyOp {
    fn compile(&self) -> Kernel {
        Kernel::Nullary(Arc::new(move || {
            // TODO(Morten) we shouldn't have core logic directly in kernels
            let raw_key = AesRng::generate_random_key();
            Value::PrfKey(PrfKey(raw_key.into()))
        }))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingAddOp {
    lhs: Ty,
    rhs: Ty,
}

impl Compile for RingAddOp {
    fn compile(&self) -> Kernel {
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

impl Compile for RingSubOp {
    fn compile(&self) -> Kernel {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                binary_kernel!(Ring64Tensor, Ring64Tensor)
            }
            (Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                binary_kernel!(Ring128Tensor, Ring128Tensor)
            }
            _ => unimplemented!()
        }
    }
}

impl<T: Sub<U>, U> BinaryFunction<T, U> for RingSubOp {
    type Output = <T as Sub<U>>::Output;
    fn execute(x: T, y: U) -> Self::Output {
        x - y
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RingMulOp;

// TODO(Morten) rewrite
impl Compile for RingMulOp {
    fn compile(&self) -> Kernel {
        Kernel::Binary(Arc::new(move |x, y| {
            let x: Ring64Tensor = x.into();
            let y: Ring64Tensor = y.into();
            let z = x * y;
            z.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RingDotOp;

// TODO(Morten) rewrite
impl Compile for RingDotOp {
    fn compile(&self) -> Kernel {
        Kernel::Binary(Arc::new(move |x, y| {
            let x: Ring64Tensor = x.into();
            let y: Ring64Tensor = y.into();
            let z = x.dot(y);
            z.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RingSumOp {
    axis: Option<usize>, // TODO(Morten) use platform independent type instead?
}

// TODO(Morten) rewrite
impl Compile for RingSumOp {
    fn compile(&self) -> Kernel {
        let axis = self.axis;
        Kernel::Unary(Arc::new(move |x| {
            let x: Ring64Tensor = x.into();
            let y = x.sum(axis);
            Value::from(y)
        }))
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RingShapeOp {
    ty: Ty,
}

trait TypeCheck {
    fn output_type(&self) -> Ty;
}

impl TypeCheck for RingShapeOp {
    fn output_type(&self) -> Ty {
        self.ty.clone()
    }
}

// TODO(Morten) rewrite
impl Compile for RingShapeOp {
    fn compile(&self) -> Kernel {
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

#[derive(Serialize, Deserialize, Debug)]
pub struct RingFillOp {
    value: u64,
}

// TODO(Morten) rewrite
impl Compile for RingFillOp {
    fn compile(&self) -> Kernel {
        let value = self.value;
        Kernel::Unary(Arc::new(move |shape| match shape {
            Value::Shape(shape) => Value::Ring64Tensor(Ring64Tensor::fill(&shape.0, value)), // TODO(Morten) should not call .0 here
            _ => unimplemented!(),
        }))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSampleOp {
    output: Ty,
    max_value: Option<usize>,
}

// TODO(Morten) rewrite
impl Compile for RingSampleOp {
    fn compile(&self) -> Kernel {
        match (self.output, self.max_value) {
            (Ty::Ring64TensorTy, None) => {
                TypedBinaryKernel::Function(move |shape: Shape, seed: Seed| {
                    Ring64Tensor::sample_uniform(&shape.0, &seed.0)
                }).into()
            }
            (Ty::Ring64TensorTy, Some(max_value)) if max_value == 1 => {
                TypedBinaryKernel::Function(move |shape: Shape, seed: Seed| {
                    Ring64Tensor::sample_bits(&shape.0, &seed.0)
                }).into()
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
impl Compile for RingShlOp {
    fn compile(&self) -> Kernel {
        let amount = self.amount;
        TypedUnaryKernel::Closure(Arc::new(move |x: Ring64Tensor| x << amount)).into()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShrOp {
    amount: usize,
}

// TODO(Morten) rewrite
impl Compile for RingShrOp {
    fn compile(&self) -> Kernel {
        let amount = self.amount;
        Kernel::Unary(Arc::new(move |x| {
            let x: Ring64Tensor = x.into();
            let y = x >> amount;
            y.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Placement {
    Host,
}

#[derive(Serialize, Deserialize, Debug)]
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

impl<V> CompiledOperation<V> {
    pub fn apply(&self, inputs: &Environment<V>) -> V {
        (self.kernel)(inputs)
    }
}

impl Operation {
    pub fn compile(&self) -> Result<CompiledOperation<Value>> {
        let operator_kernel: Kernel = self.kind.compile();
        match (operator_kernel, self.inputs.len()) {
            (Kernel::Nullary(k), 0) => {
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |_: &Environment<Value>| k()),
                })
            }
            (Kernel::Unary(k), 1) => {
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
            (Kernel::Binary(k), 2) => {
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
            (Kernel::Ternary(k), 3) => {
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
            (Kernel::Variadic(k), _) => {
                let inputs = self.inputs.clone();
                Ok(CompiledOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |env| {
                        let xs: Vec<_> = inputs
                            .iter()
                            .map(|input| env.get(input).unwrap())
                            .map(|value| value.clone()) // TODO(Morten) avoid cloning
                            .collect();
                        k(&xs)
                    }),
                })
            }
            _ => Err(anyhow!("Failed to compile kernel for operation '{}' due to arity mismatch; {} inputs were given", self.name, self.inputs.len())),
        }
    }

    pub fn async_compile(&self) -> Result<CompiledOperation<AsyncValue>> {
        let operator_kernel: AsyncKernel = self.kind.async_compile();
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
                            .map(|value| value.clone())
                            .collect();
                        k(&xs)
                    }),
                })
            }
            _ => Err(anyhow!("Failed to compile async kernel for operation '{}' due to arity mismatch; {} inputs were given", self.name, self.inputs.len())),
        }
    }

    pub fn apply(&self, env: &Environment<Value>) -> Result<Value> {
        let compiled = self.compile()?;
        Ok(compiled.apply(env))
    }

    pub fn apply_and_insert(&self, env: &mut Environment<Value>) -> Result<()> {
        let value = self.apply(env)?;
        env.insert(self.name.clone(), value);
        Ok(())
    }

    pub fn type_check(&self, env: &Environment<Ty>) -> Ty {
        unimplemented!()
    }
}

pub struct Computation {
    pub operations: Vec<Operation>,
}

pub struct CompiledComputation<V>(Arc<dyn Fn(Environment<V>) -> Environment<V>>);

impl Computation {
    pub fn compile(&self) -> Result<CompiledComputation<Value>> {
        // TODO(Morten) type check computation
        let compiled_ops: Vec<CompiledOperation<_>> = self
            .operations
            .iter()
            .map(|op| op.compile())
            .collect::<Result<Vec<_>>>()?;
        // TODO(Morten) we want to sort topologically here, outside the closure
        // TODO(Morten) do we want to insert instructions for when values can be dropped from the environment?
        Ok(CompiledComputation(Arc::new(move |mut env| {
            for compiled_op in compiled_ops.iter() {
                let value = compiled_op.apply(&env);
                env.insert(compiled_op.name.clone(), value);
            }
            env
        })))
    }

    pub fn async_compile(&self) -> Result<CompiledComputation<AsyncValue>> {
        // TODO(Morten) type check computation
        let compiled_ops: Vec<CompiledOperation<_>> = self
            .operations
            .iter()
            .map(|op| op.async_compile())
            .collect::<Result<Vec<_>>>()?;
        // TODO(Morten) we want to sort topologically here, outside the closure
        // TODO(Morten) do we want to insert instructions for when values can be dropped from the environment?
        Ok(CompiledComputation(Arc::new(move |mut env| {
            for compiled_op in compiled_ops.iter() {
                let value = compiled_op.apply(&env);
                env.insert(compiled_op.name.clone(), value);
            }
            env
        })))
    }
}

impl<V> CompiledComputation<V> {
    pub fn apply(&self, env: Environment<V>) -> Environment<V> {
        (self.0)(env)
    }
}

pub type Environment<V> = HashMap<String, V>;

pub type EagerEnvironment = Environment<Value>;

/// In-order single-threaded executor.
///
/// This executor evaluates the operations of computations in-order, raising an error
/// in case data dependencies are not respected. This executor is intended for debug
/// and development only due to its unforgiving but highly predictable behaviour.
pub struct EagerExecutor;

impl EagerExecutor {
    pub fn run_computation(&self, comp: &Computation, args: EagerEnvironment) -> Result<()> {
        let compiled_comp = comp.compile()?;
        let env = compiled_comp.apply(args);
        println!("{:?}", env);
        Ok(())
    }
}

pub struct AsyncExecutor;

impl AsyncExecutor {
    pub fn run_computation(&self, comp: &Computation, args: Environment<AsyncValue>) -> Result<()> {
        let compiled_comp = comp.async_compile()?;

        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let env = compiled_comp.apply(args);
            let vals = futures::future::join_all(env.values().map(|op| op.clone()).collect::<Vec<_>>()).await;
            println!("{:?}", vals);
        });
        Ok(())
    }
}

#[test]
fn test_foo() {
    let mut env = hashmap![];

    let key_op = Operation {
        name: "key".into(),
        kind: Operator::PrimGenPrfKey(PrimGenPrfKeyOp),
        inputs: vec![],
        placement: Placement::Host,
    };
    // key_op.apply_and_insert(&mut env).ok();

    let x_seed_op = Operation {
        name: "x_seed".into(),
        kind: Operator::PrimDeriveSeed(PrimDeriveSeedOp {
            nonce: Nonce(vec![1, 2, 3]),
        }),
        inputs: vec!["key".into()],
        placement: Placement::Host,
    };
    // x_seed_op.apply_and_insert(&mut env).ok();

    let x_shape_op = Operation {
        name: "x_shape".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Shape(Shape(vec![2, 3])),
        }),
        inputs: vec![],
        placement: Placement::Host,
    };
    // x_shape_op.apply_and_insert(&mut env).ok();

    let x_op = Operation {
        name: "x".into(),
        kind: Operator::RingSample(RingSampleOp { output: Ty::Ring64TensorTy, max_value: None }),
        inputs: vec!["x_shape".into(), "x_seed".into()],
        placement: Placement::Host,
    };
    // x_op.apply_and_insert(&mut env).ok();

    let y_op = Operation {
        name: "y".into(),
        kind: Operator::RingSample(RingSampleOp { output: Ty::Ring64TensorTy, max_value: None }),
        inputs: vec!["x_shape".into(), "x_seed".into()],
        placement: Placement::Host,
    };

    let z_op = Operation {
        name: "z".into(),
        kind: Operator::RingMul(RingMulOp),
        inputs: vec!["x".into(), "y".into()],
        placement: Placement::Host,
    };
    // z_op.apply_and_insert(&mut env).ok();

    let comp = Computation {
        operations: vec![
            key_op, x_seed_op, x_shape_op, x_op, y_op, z_op
        ],
    };

    let exec = AsyncExecutor;
    exec.run_computation(&comp, env).ok();
    assert!(false);
}
