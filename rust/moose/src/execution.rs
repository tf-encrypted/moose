#![allow(unused_macros)]

use crate::prng::AesRng;
use crate::ring::{Dot, Ring128Tensor, Ring64Tensor, Sample};
use anyhow::anyhow;
use async_trait::async_trait;
use enum_dispatch::enum_dispatch;
use futures::future::{Map, Shared};
use futures::prelude::*;
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::{sync::oneshot, task::JoinHandle};
use std::convert::TryFrom;
use tracing::{debug};


#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("Unexpected error")]
    Unexpected,
    #[error("Input to kernel unavailable")]
    InputUnavailable,
    #[error("Type mismatch")]
    TypeMismatch,
    #[error("Operator instantiation not supported")]
    UnimplementedOperator,
    #[error("Malformed environment")]
    MalformedEnvironment,
    #[error("Malformed computation")]
    MalformedComputation(String),
    #[error("Compilation error: {0}")]
    Compilation(String),
}

pub type Result<T> = anyhow::Result<T, Error>;

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
    Unit,
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

impl TryFrom<Value> for Ring64Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Ring64Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl<'v> TryFrom<&'v Value> for &'v Ring64Tensor {
    type Error = Error;
    fn try_from(v: &'v Value) -> Result<Self> {
        match v {
            Value::Ring64Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Ring128Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Ring128Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Shape {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Shape(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Seed {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Seed(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for PrfKey {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::PrfKey(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Nonce {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Nonce(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

macro_rules! function_kernel {
    ($f:expr) => {
        Ok(Kernel::NullaryFunction(|| {
            let y = $f();
            Ok(Value::from(y))
        }))
    };
    ($t0:ty, $f:expr) => {
        Ok(Kernel::UnaryFunction(|x0| {
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let g: fn($t0) -> _ = $f;
            let y = g(x0);
            Ok(Value::from(y))
        }))
    };
    ($t0:ty, $t1:ty, $f:expr) => {
        Ok(Kernel::BinaryFunction(|x0, x1| {
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let x1 = <$t1 as TryFrom<Value>>::try_from(x1)?;
            let g: fn($t0, $t1) -> _ = $f;
            let y = g(x0, x1);
            Ok(Value::from(y))
        }))
    };
    ($t0:ty, $t1:ty, $t2:ty) => {
        Ok(Kernel::TernaryFunction(|x0, x1, x2| {
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let x1 = <$t1 as TryFrom<Value>>::try_from(x1)?;
            let x2 = <$t2 as TryFrom<Value>>::try_from(x2)?;
            let g: fn($t0, $t1, $t2) -> _ = $f;
            let y = g(x0, x1, x2);
            Ok(Value::from(y))
        }))
    };
}

macro_rules! closure_kernel {
    ($f:expr) => {
        Ok(Kernel::NullaryClosure(Arc::new(move || {
            let y = $f();
            Ok(Value::from(y))
        })))
    };
    ($t0:ty, $f:expr) => {
        Ok(Kernel::UnaryClosure(Arc::new(move |x0| {
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let y = $f(x0);
            Ok(Value::from(y))
        })))
    };
    ($t0:ty, $t1:ty, $f:expr) => {
        Ok(Kernel::BinaryClosure(Arc::new(move |x0, x1| {
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let x1 = <$t1 as TryFrom<Value>>::try_from(x1)?;
            let y = $f(x0, x1);
            Ok(Value::from(y))
        })))
    };
    ($t0:ty, $t1:ty, $t2:ty, $f:expr) => {
        Ok(Kernel::TernaryClosure(Arc::new(move |x0, x1, x2| {
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let x1 = <$t1 as TryFrom<Value>>::try_from(x1)?;
            let x2 = <$t2 as TryFrom<Value>>::try_from(x2)?;
            let y = $f(x0, x1, x2);
            Ok(Value::from(y))
        })))
    };
}

pub enum Kernel {
    NullaryClosure(Arc<dyn Fn() -> Result<Value> + Send + Sync>),
    UnaryClosure(Arc<dyn Fn(Value) -> Result<Value> + Send + Sync>),
    BinaryClosure(Arc<dyn Fn(Value, Value) -> Result<Value> + Send + Sync>),
    TernaryClosure(Arc<dyn Fn(Value, Value, Value) -> Result<Value> + Send + Sync>),
    VariadicClosure(Arc<dyn Fn(&[Value]) -> Result<Value> + Send + Sync>),

    NullaryFunction(fn() -> Result<Value>),
    UnaryFunction(fn(Value) -> Result<Value>),
    BinaryFunction(fn(Value, Value) -> Result<Value>),
    TernaryFunction(fn(Value, Value, Value) -> Result<Value>),
    VariadicFunction(fn(&[Value]) -> Result<Value>),
}

pub enum SyncKernel {
    Nullary(Box<dyn Fn(&SyncSession) -> Result<Value> + Send + Sync>),
    Unary(Box<dyn Fn(&SyncSession, Value) -> Result<Value> + Send + Sync>),
    Binary(Box<dyn Fn(&SyncSession, Value, Value) -> Result<Value> + Send + Sync>),
    Ternary(Box<dyn Fn(&SyncSession, Value, Value, Value) -> Result<Value> + Send + Sync>),
    Variadic(Box<dyn Fn(&SyncSession, &[Value]) -> Result<Value> + Send + Sync>),
}
pub enum AsyncKernel {
    Nullary(Box<dyn Fn(&Arc<AsyncSession>, AsyncSender) -> AsyncTask>),
    Unary(Box<dyn Fn(&Arc<AsyncSession>, AsyncValue, AsyncSender) -> AsyncTask>),
    Binary(Box<dyn Fn(&Arc<AsyncSession>, AsyncValue, AsyncValue, AsyncSender) -> AsyncTask>),
    Ternary(Box<dyn Fn(&Arc<AsyncSession>, AsyncValue, AsyncValue, AsyncValue, AsyncSender) -> AsyncTask>),
    Variadic(Box<dyn Fn(&Arc<AsyncSession>, &[AsyncValue], AsyncSender) -> AsyncTask>),
}

pub type AsyncSender = oneshot::Sender<Value>;

pub type AsyncReceiver = Shared<
    Map<
        oneshot::Receiver<Value>,
        fn(anyhow::Result<Value, oneshot::error::RecvError>) -> anyhow::Result<Value, ()>,
    >,
>;

pub type AsyncTask = tokio::task::JoinHandle<Result<()>>;

pub type AsyncValue = Shared<
    Map<
        oneshot::Receiver<Value>,
        fn(anyhow::Result<Value, oneshot::error::RecvError>) -> anyhow::Result<Value, ()>,
    >,
>;


pub trait Compile<C> {
    fn compile(&self) -> Result<C>;
}

impl<O> SyncCompile for O
where
    O: Compile<Kernel>,
{
    fn compile(&self) -> Result<SyncKernel> {
        let kernel: Kernel = self.compile()?;
        match kernel {
            Kernel::NullaryClosure(k) => Ok(SyncKernel::Nullary(Box::new(move |_| k()))),
            Kernel::UnaryClosure(k) => Ok(SyncKernel::Unary(Box::new(move |_, x0| k(x0)))),
            Kernel::BinaryClosure(k) => {
                Ok(SyncKernel::Binary(Box::new(move |_, x0, x1| k(x0, x1))))
            }
            Kernel::TernaryClosure(k) => Ok(SyncKernel::Ternary(Box::new(move |_, x0, x1, x2| {
                k(x0, x1, x2)
            }))),
            Kernel::VariadicClosure(k) => Ok(SyncKernel::Variadic(Box::new(move |_, xs| k(xs)))),
            Kernel::NullaryFunction(k) => Ok(SyncKernel::Nullary(Box::new(move |_| k()))),
            Kernel::UnaryFunction(k) => Ok(SyncKernel::Unary(Box::new(move |_, x0| k(x0)))),
            Kernel::BinaryFunction(k) => {
                Ok(SyncKernel::Binary(Box::new(move |_, x0, x1| k(x0, x1))))
            }
            Kernel::TernaryFunction(k) => {
                Ok(SyncKernel::Ternary(Box::new(move |_, x0, x1, x2| {
                    k(x0, x1, x2)
                })))
            }
            Kernel::VariadicFunction(k) => Ok(SyncKernel::Variadic(Box::new(move |_, xs| k(xs)))),
        }
    }
}

impl SyncCompile for SendOp {
    fn compile(&self) -> Result<SyncKernel> {
        let rdv = self.rendezvous_key.clone();
        Ok(SyncKernel::Unary(Box::new(move |sess, v| {
            sess.networking.send(&v, &rdv, &sess.id);
            Ok(Value::Unit)
        })))
    }
}

impl SyncCompile for ReceiveOp {
    fn compile(&self) -> Result<SyncKernel> {
        let rdv = self.rendezvous_key.clone();
        Ok(SyncKernel::Nullary(Box::new(move |sess| {
            sess.networking.receive(&rdv, &sess.id)
        })))
    }
}

fn map_send_error<T>(_: T) -> Error {
    debug!("Failed to send result on channel, receiver was dropped");
    Error::Unexpected
}

fn map_receive_error<T>(_: T) -> Error {
    Error::InputUnavailable
}

impl<O> AsyncCompile for O
where
    O: Compile<Kernel>,
{
    fn compile(&self) -> Result<AsyncKernel> {
        let kernel: Kernel = self.compile()?;
        match kernel {
            Kernel::NullaryClosure(k) => Ok(AsyncKernel::Nullary(Box::new(move |_, sender| {
                let k = Arc::clone(&k);
                tokio::spawn(async move {
                    let y: Value = k()?;
                    sender.send(y).map_err(map_send_error)
                })
            }))),
            Kernel::UnaryClosure(k) => Ok(AsyncKernel::Unary(Box::new(move |_, x0, sender| {
                let k = Arc::clone(&k);
                tokio::spawn(async move {
                    let x0: Value = x0.await.map_err(map_receive_error)?;
                    let y: Value = k(x0)?;
                    sender.send(y).map_err(map_send_error)
                })
            }))),
            Kernel::BinaryClosure(k) => Ok(AsyncKernel::Binary(Box::new(move |_, x0, x1, sender| {
                let k = Arc::clone(&k);
                tokio::spawn(async move {
                    let x0: Value = x0.await.map_err(map_receive_error)?;
                    let x1: Value = x1.await.map_err(map_receive_error)?;
                    let y: Value = k(x0, x1)?;
                    sender.send(y).map_err(map_send_error)
                })
            }))),
            Kernel::TernaryClosure(k) => {
                Ok(AsyncKernel::Ternary(Box::new(move |_, x0, x1, x2, sender| {
                    let k = Arc::clone(&k);
                    tokio::spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let x1: Value = x1.await.map_err(map_receive_error)?;
                        let x2: Value = x2.await.map_err(map_receive_error)?;
                        let y: Value = k(x0, x1, x2)?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }
            Kernel::VariadicClosure(_k) => unimplemented!(), // TODO

            Kernel::NullaryFunction(k) => Ok(AsyncKernel::Nullary(Box::new(move |_, sender| {
                tokio::spawn(async move {
                    let y = k()?;
                    sender.send(y).map_err(map_send_error)
                })
            }))),
            Kernel::UnaryFunction(k) => Ok(AsyncKernel::Unary(Box::new(move |_, x0, sender| {
                tokio::spawn(async move {
                    let x0: Value = x0.await.map_err(map_receive_error)?;
                    let y: Value = k(x0)?;
                    sender.send(y).map_err(map_send_error)
                })
            }))),
            Kernel::BinaryFunction(k) => Ok(AsyncKernel::Binary(Box::new(move |_, x0, x1, sender| {
                tokio::spawn(async move {
                    let x0: Value = x0.await.map_err(map_receive_error)?;
                    let x1: Value = x1.await.map_err(map_receive_error)?;
                    let y: Value = k(x0, x1)?;
                    sender.send(y).map_err(map_send_error)
                })
            }))),
            Kernel::TernaryFunction(k) => {
                Ok(AsyncKernel::Ternary(Box::new(move |_, x0, x1, x2, sender| {
                    tokio::spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let x1: Value = x1.await.map_err(map_receive_error)?;
                        let x2: Value = x2.await.map_err(map_receive_error)?;
                        let y: Value = k(x0, x1, x2)?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }
            Kernel::VariadicFunction(_k) => unimplemented!(), // TODO
        }
    }
}

impl AsyncCompile for SendOp {
    fn compile(&self) -> Result<AsyncKernel> {
        let rdv = Arc::new(self.rendezvous_key.clone());
        Ok(AsyncKernel::Unary(Box::new(move |sess, v, sender| {
            let sess = Arc::clone(sess);
            let rdv = Arc::clone(&rdv);
            tokio::spawn(async move {
                let v: Value = v.await.map_err(map_receive_error)?;
                sess.networking.send(&v, &rdv, &sess.id).await;
                sender.send(Value::Unit).map_err(map_send_error)
            })
        })))
    }
}

impl AsyncCompile for ReceiveOp {
    fn compile(&self) -> Result<AsyncKernel> {
        let rdv = Arc::new(self.rendezvous_key.clone());
        Ok(AsyncKernel::Nullary(Box::new(move |sess, sender| {
            let sess = Arc::clone(sess);
            let rdv = Arc::clone(&rdv);
            tokio::spawn(async move {
                let v: Value = sess.networking.receive(&rdv, &sess.id).await?;
                sender.send(v).map_err(map_send_error)
            })
        })))
    }
}

pub type RendezvousKey = str;
pub type SessionId = u128;
pub struct SyncSession {
    pub id: SessionId,
    networking: Arc<dyn Send + Sync + SyncNetworking>,
}

pub struct AsyncSession {
    pub id: SessionId,
    networking: Arc<dyn Send + Sync + AsyncNetworking>,
}

impl SyncSession {
    pub fn new_subsession(&self) -> SyncSession {
        let id = self.id; // TODO
        SyncSession {
            id,
            networking: Arc::clone(&self.networking), // TODO
        }
    }
}

pub trait SyncNetworking {
    fn send(&self, v: &Value, rendezvous_key: &RendezvousKey, session_id: &SessionId);
    fn receive(&self, rendezvous_key: &RendezvousKey, session_id: &SessionId) -> Result<Value>;
}

#[async_trait]
pub trait AsyncNetworking {
    async fn send(&self, v: &Value, rendezvous_key: &RendezvousKey, session_id: &SessionId);
    async fn receive(&self, rendezvous_key: &RendezvousKey, session_id: &SessionId) -> Result<Value>;
}

pub struct DummySyncNetworking;

impl SyncNetworking for DummySyncNetworking {
    fn send(&self, _v: &Value, rendezvous_key: &RendezvousKey, session_id: &SessionId) {
        println!("Sending; rdv:'{}' sid:{}", rendezvous_key, session_id);
    }

    fn receive(&self, rendezvous_key: &RendezvousKey, session_id: &SessionId) -> Result<Value> {
        println!("Receiving; rdv:'{}', sid:{}", rendezvous_key, session_id);
        Ok(Value::Shape(Shape(vec![0])))
    }
}

pub struct DummyAsyncNetworking;

#[async_trait]
impl AsyncNetworking for DummyAsyncNetworking {
    async fn send(&self, _v: &Value, rendezvous_key: &RendezvousKey, session_id: &SessionId) {
        println!("Async sending; rdv:'{}' sid:{}", rendezvous_key, session_id);
    }

    async fn receive(&self, rendezvous_key: &RendezvousKey, session_id: &SessionId) -> Result<Value> {
        println!(
            "Async receiving; rdv:'{}', sid:{}",
            rendezvous_key, session_id
        );
        Ok(Value::Shape(Shape(vec![0])))
    }
}

#[enum_dispatch]
pub trait SyncCompile {
    fn compile(&self) -> Result<SyncKernel>;
}

#[enum_dispatch]
pub trait AsyncCompile {
    fn compile(&self) -> Result<AsyncKernel>;
}

#[enum_dispatch(AsyncCompile, SyncCompile)]
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
    Receive(ReceiveOp),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SendOp {
    pub rendezvous_key: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReceiveOp {
    pub rendezvous_key: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConstantOp {
    pub value: Value,
}

impl Compile<Kernel> for ConstantOp {
    fn compile(&self) -> Result<Kernel> {
        let value = self.value.clone();
        Ok(Kernel::NullaryClosure(Arc::new(move || Ok(value.clone()))))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrimDeriveSeedOp {
    pub nonce: Nonce,
}

impl Compile<Kernel> for PrimDeriveSeedOp {
    fn compile(&self) -> Result<Kernel> {
        let nonce = self.nonce.0.clone();
        closure_kernel!(PrfKey, |key: PrfKey| {
            let todo = crate::utils::derive_seed(&key.0, &nonce);
            Seed(todo.into())
        })
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrimGenPrfKeyOp;

impl Compile<Kernel> for PrimGenPrfKeyOp {
    fn compile(&self) -> Result<Kernel> {
        function_kernel!(|| {
            // TODO(Morten) we shouldn't have core logic directly in kernels
            let raw_key = AesRng::generate_random_key();
            Value::PrfKey(PrfKey(raw_key.into()))
        })
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingAddOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

impl Compile<Kernel> for RingAddOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x + y)
            }
            (Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x + y)
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

impl Compile<Kernel> for RingSubOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x - y)
            }
            (Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x - y)
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingMulOp {
    lhs: Ty,
    rhs: Ty,
}

impl Compile<Kernel> for RingMulOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x * y),
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingDotOp {
    lhs: Ty,
    rhs: Ty,
}

impl Compile<Kernel> for RingDotOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x.dot(y)),
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSumOp {
    ty: Ty,
    axis: Option<u32>,
}

impl Compile<Kernel> for RingSumOp {
    fn compile(&self) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.ty {
            Ty::Ring64TensorTy => closure_kernel!(Ring64Tensor, |x: Ring64Tensor| x.sum(axis)),
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShapeOp {
    ty: Ty,
}

impl Compile<Kernel> for RingShapeOp {
    fn compile(&self) -> Result<Kernel> {
        match self.ty {
            Ty::Ring64TensorTy => function_kernel!(Ring64Tensor, |x| {
                Shape(x.0.shape().into()) // TODO(Morten) wrapping should not happen here
            }),
            Ty::Ring128TensorTy => function_kernel!(Ring128Tensor, |x| {
                Shape(x.0.shape().into()) // TODO(Morten) wrapping should not happen here
            }),
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingFillOp {
    value: u64,
}

impl Compile<Kernel> for RingFillOp {
    fn compile(&self) -> Result<Kernel> {
        let value = self.value;
        // TODO(Morten) should not call .0 here
        closure_kernel!(Shape, |shape: Shape| Ring64Tensor::fill(&shape.0, value))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSampleOp {
    pub output: Ty,
    pub max_value: Option<usize>,
}

impl Compile<Kernel> for RingSampleOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.output, self.max_value) {
            (Ty::Ring64TensorTy, None) => {
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_uniform(
                    &shape.0, &seed.0
                ))
            }
            (Ty::Ring64TensorTy, Some(max_value)) if max_value == 1 => {
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_bits(
                    &shape.0, &seed.0
                ))
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShlOp {
    pub amount: usize,
}

impl Compile<Kernel> for RingShlOp {
    fn compile(&self) -> Result<Kernel> {
        let amount = self.amount;
        closure_kernel!(Ring64Tensor, |x| x << amount)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShrOp {
    pub amount: usize,
}

impl Compile<Kernel> for RingShrOp {
    fn compile(&self) -> Result<Kernel> {
        let amount = self.amount;
        closure_kernel!(Ring64Tensor, |x| x >> amount)
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
    pub inputs: Vec<String>, // TODO(Morten) use indices instead of strings?
    pub placement: Placement,
}

pub struct CompiledSyncOperation {
    name: String,
    kernel: Box<dyn Fn(&SyncSession, &Environment<Value>) -> Result<Value>>,
}

impl CompiledSyncOperation {
    pub fn apply(&self, session: &SyncSession, inputs: &Environment<Value>) -> Result<Value> {
        (self.kernel)(session, inputs)
    }
}

pub struct CompiledAsyncOperation {
    name: String,
    kernel: Box<dyn Fn(&Arc<AsyncSession>, &Environment<AsyncReceiver>, AsyncSender) -> Result<AsyncTask>>,
}

impl CompiledAsyncOperation {
    pub fn apply(&self, sess: &Arc<AsyncSession>, env: &Environment<AsyncReceiver>, sender: AsyncSender) -> Result<AsyncTask> {
        (self.kernel)(sess, env, sender)
    }
}

fn check_arity<T>(operation_name: &str, inputs: &[T], arity: usize) -> Result<()> {
    if inputs.len() != arity {
        Err(Error::Compilation(format!(
            "Arity mismatch for operation '{}'; operator expects {} arguments but were given {}",
            operation_name,
            arity,
            inputs.len()
        )))
    } else {
        Ok(())
    }
}

impl Compile<CompiledSyncOperation> for Operation {
    fn compile(&self) -> Result<CompiledSyncOperation> {
        let operator_kernel = SyncCompile::compile(&self.kind)?;
        match operator_kernel {
            SyncKernel::Nullary(k) => {
                check_arity(&self.name, &self.inputs, 0)?;
                Ok(CompiledSyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, _| k(sess)),
                })
            }
            SyncKernel::Unary(k) => {
                check_arity(&self.name, &self.inputs, 1)?;
                let x0_name = self.inputs[0].clone();
                Ok(CompiledSyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, env| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).ok_or(Error::MalformedEnvironment)?.clone();
                        k(sess, x0)
                    }),
                })
            }
            SyncKernel::Binary(k) => {
                check_arity(&self.name, &self.inputs, 2)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                Ok(CompiledSyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, env| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).ok_or(Error::MalformedEnvironment)?.clone();
                        let x1 = env.get(&x1_name).ok_or(Error::MalformedEnvironment)?.clone();
                        k(sess, x0, x1)
                    }),
                })
            }
            SyncKernel::Ternary(k) => {
                check_arity(&self.name, &self.inputs, 3)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                let x2_name = self.inputs[2].clone();
                Ok(CompiledSyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, env| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).ok_or(Error::MalformedEnvironment)?.clone();
                        let x1 = env.get(&x1_name).ok_or(Error::MalformedEnvironment)?.clone();
                        let x2 = env.get(&x2_name).ok_or(Error::MalformedEnvironment)?.clone();
                        k(sess, x0, x1, x2)
                    }),
                })
            }
            SyncKernel::Variadic(k) => {
                let inputs = self.inputs.clone();
                Ok(CompiledSyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, env| {
                        let xs = inputs
                            .iter()
                            .map(|input| env.get(input).cloned().ok_or(Error::MalformedEnvironment))  // TODO(Morten avoid cloning
                            .collect::<Result<Vec<_>>>()?;
                        k(sess, &xs)
                    }),
                })
            }
        }
    }
}

impl Compile<CompiledAsyncOperation> for Operation {
    fn compile(&self) -> Result<CompiledAsyncOperation> {
        let operator_kernel = AsyncCompile::compile(&self.kind)?;
        match operator_kernel {
            AsyncKernel::Nullary(k) => {
                check_arity(&self.name, &self.inputs, 0)?;
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, _, sender| Ok(k(sess, sender))),
                })
            }
            AsyncKernel::Unary(k) => {
                check_arity(&self.name, &self.inputs, 1)?;
                let x0_name = self.inputs[0].clone();
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, env, sender| {
                        let x0 = env.get(&x0_name).ok_or(Error::MalformedEnvironment)?.clone();
                        Ok(k(sess, x0, sender))
                    }),
                })
            }
            AsyncKernel::Binary(k) => {
                check_arity(&self.name, &self.inputs, 2)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, env, sender| {
                        let x0 = env.get(&x0_name).ok_or(Error::MalformedEnvironment)?.clone();
                        let x1 = env.get(&x1_name).ok_or(Error::MalformedEnvironment)?.clone();
                        Ok(k(sess, x0, x1, sender))
                    }),
                })
            }
            AsyncKernel::Ternary(k) => {
                check_arity(&self.name, &self.inputs, 3)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                let x2_name = self.inputs[2].clone();
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, env, sender| {
                        let x0 = env.get(&x0_name).ok_or(Error::MalformedEnvironment)?.clone();
                        let x1 = env.get(&x1_name).ok_or(Error::MalformedEnvironment)?.clone();
                        let x2 = env.get(&x2_name).ok_or(Error::MalformedEnvironment)?.clone();
                        Ok(k(sess, x0, x1, x2, sender))
                    }),
                })
            }
            AsyncKernel::Variadic(k) => {
                let inputs = self.inputs.clone();
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, env, sender| {
                        let xs = inputs
                            .iter()
                            .map(|input| env.get(input).cloned().ok_or(Error::MalformedEnvironment))
                            .collect::<Result<Vec<_>>>()?;
                        Ok(k(sess, &xs, sender))
                    }),
                })
            }
        }
    }
}

impl Operation {
    // pub fn apply<V, S>(&self, sess: &S, env: &Environment<V>) -> Result<V>
    // where
    //     Self: Compile<CompiledOperation<V, S>>,
    //     // CompiledOperation<V, S>: Apply<V, S>,
    // {
    //     let compiled: CompiledOperation<V, S> = self.compile()?;
    //     Ok(compiled.apply(sess, env))
    // }

    // pub fn apply_and_insert<V, S>(&self, sess: &S, env: &mut Environment<V>) -> Result<()>
    // where
    //     Self: Compile<CompiledOperation<V, S>>,
    // {
    //     let value = self.apply(sess, env)?;
    //     env.insert(self.name.clone(), value);
    //     Ok(())
    // }
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
            .map_err(|_| Error::MalformedComputation("There is a cycle detected in the runtime graph".into()))?;

        let operations = toposort
            .iter()
            .map(|node| self.operations[inv_map[node]].clone())
            .collect();

        Ok(Computation { operations })
    }
}

pub struct CompiledSyncComputation(Box<dyn Fn(SyncSession, Environment<Value>) -> Result<Environment<Value>>>);

impl Compile<CompiledSyncComputation> for Computation
where
    Operation: Compile<CompiledSyncOperation>,
{
    fn compile(&self) -> Result<CompiledSyncComputation> {
        // TODO(Morten) type check computation
        let compiled_ops: Vec<CompiledSyncOperation> = self
            .operations
            .iter()
            // .par_iter()
            .map(|op| op.compile())
            .collect::<Result<Vec<_>>>()?;
        // TODO(Morten) we want to sort topologically here, outside the closure
        // TODO(Morten) do we want to insert instructions for when values can be dropped from the environment?
        Ok(CompiledSyncComputation(Box::new(
            move |sess: SyncSession, mut env: Environment<Value>| {
                for compiled_op in compiled_ops.iter() {
                    let value = compiled_op.apply(&sess, &env)?;
                    env.insert(compiled_op.name.clone(), value);
                }
                Ok(env)
            },
        )))
    }
}

impl CompiledSyncComputation {
    pub fn apply(&self, sess: SyncSession, env: Environment<Value>) -> Result<Environment<Value>> {
        (self.0)(sess, env)
    }
}

pub struct CompiledAsyncComputation(Box<dyn Fn(AsyncSession, Environment<AsyncValue>) -> Result<Environment<AsyncValue>>>);

fn remove_err<T, E>(r: std::result::Result<T, E>) -> std::result::Result<T, ()> {
    r.map_err(|_| ())
}

impl Compile<CompiledAsyncComputation> for Computation
where
    Operation: Compile<CompiledAsyncOperation>,
{
    fn compile(&self) -> Result<CompiledAsyncComputation> {
        // TODO(Morten) type check computation
        let compiled_ops: Vec<CompiledAsyncOperation> = self
            .operations
            .iter()
            // .par_iter()
            .map(|op| op.compile())
            .collect::<Result<Vec<_>>>()?;
            
        Ok(CompiledAsyncComputation(Box::new(
            move |sess: AsyncSession, mut env: Environment<AsyncValue>| {
                let sess = Arc::new(sess);
                let channels = compiled_ops.iter()
                    .map(|op| {
                        let (sender, receiver): (AsyncSender, _) = oneshot::channel();
                        let shared_receiver: AsyncReceiver = receiver.map(remove_err as fn(_) -> _).shared();
                        (op.name.clone(), (sender, shared_receiver))
                    })
                    .collect::<HashMap<_, _>>();
                let tasks = compiled_ops.iter()
                    .map(|op| op.apply(&sess, env, sender))
                    compiled_op.apply(&sess, &env)?
                };
                Ok(())
            },
        )))
    }
}

impl CompiledAsyncComputation {
    pub fn apply(&self, sess: AsyncSession, env: Environment<AsyncValue>) -> Result<Environment<AsyncValue>> {
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
        let compiled_comp: CompiledSyncComputation = comp.compile()?;
        let sess = SyncSession {
            id: 12345,
            networking: Arc::new(DummySyncNetworking),
        };
        let _env = compiled_comp.apply(sess, args)?;
        println!("Done");
        Ok(())
    }
}

pub struct AsyncExecutor;

impl AsyncExecutor {
    pub fn run_computation(&self, comp: &Computation, args: Environment<AsyncValue>) -> Result<()> {
        let compiled_comp: CompiledAsyncComputation = comp.compile()?;
        println!("Compiled");

        // let rt = tokio::runtime::Runtime::new()?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let sess = Arc::new(AsyncSession {
            id: 12345,
            networking: Arc::new(DummyAsyncNetworking),
        });

        println!("Running");
        rt.block_on(async {
            let env = compiled_comp.apply(sess, args);
            let vals =
                futures::future::join_all(env.values().map(|op| op.clone()).collect::<Vec<_>>())
                    .await;
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

    let ops = vec![
        key_op,
        x_seed_op,
        Operation {
            name: "send".into(),
            kind: Operator::Send(SendOp {
                rendezvous_key: "rdv0".into(),
            }),
            inputs: vec!["x_seed".into()],
            placement: Placement::Host,
        },
        Operation {
            name: "recv".into(),
            kind: Operator::Receive(ReceiveOp {
                rendezvous_key: "rdv0".into(),
            }),
            inputs: vec![],
            placement: Placement::Host,
        },
    ];

    let comp = Computation {
        // operations: [vec![key_op, x_seed_op, x_shape_op], sample_ops].concat(),
        operations: ops,
    }
    .toposort()
    .unwrap();

    let exec = AsyncExecutor;
    // let exec = EagerExecutor;
    exec.run_computation(&comp, env).ok();
    // assert!(false);
}
