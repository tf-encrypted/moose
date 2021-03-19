#![allow(unused_macros)]

use crate::fixedpoint::Convert;
use crate::prng::AesRng;
use crate::ring::{Dot, Ring128Tensor, Ring64Tensor, Sample};
use crate::standard::*;
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
use std::convert::TryFrom;
use std::sync::Arc;
use tokio::sync::oneshot;
use tracing::debug;

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
    Float32TensorTy,
    Float64TensorTy,
    Int8TensorTy,
    Int16TensorTy,
    Int32TensorTy,
    Int64TensorTy,
    Uint8TensorTy,
    Uint16TensorTy,
    Uint32TensorTy,
    Uint64TensorTy,
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
    Float32Tensor(Float32Tensor),
    Float64Tensor(Float64Tensor),
    Int8Tensor(Int8Tensor),
    Int16Tensor(Int16Tensor),
    Int32Tensor(Int32Tensor),
    Int64Tensor(Int64Tensor),
    Uint8Tensor(Uint8Tensor),
    Uint16Tensor(Uint16Tensor),
    Uint32Tensor(Uint32Tensor),
    Uint64Tensor(Uint64Tensor),
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

impl From<Float32Tensor> for Value {
    fn from(v: Float32Tensor) -> Self {
        Value::Float32Tensor(v)
    }
}

impl From<Float64Tensor> for Value {
    fn from(v: Float64Tensor) -> Self {
        Value::Float64Tensor(v)
    }
}

impl From<Int8Tensor> for Value {
    fn from(v: Int8Tensor) -> Self {
        Value::Int8Tensor(v)
    }
}

impl From<Int16Tensor> for Value {
    fn from(v: Int16Tensor) -> Self {
        Value::Int16Tensor(v)
    }
}

impl From<Int32Tensor> for Value {
    fn from(v: Int32Tensor) -> Self {
        Value::Int32Tensor(v)
    }
}

impl From<Int64Tensor> for Value {
    fn from(v: Int64Tensor) -> Self {
        Value::Int64Tensor(v)
    }
}

impl From<Uint8Tensor> for Value {
    fn from(v: Uint8Tensor) -> Self {
        Value::Uint8Tensor(v)
    }
}

impl From<Uint16Tensor> for Value {
    fn from(v: Uint16Tensor) -> Self {
        Value::Uint16Tensor(v)
    }
}

impl From<Uint32Tensor> for Value {
    fn from(v: Uint32Tensor) -> Self {
        Value::Uint32Tensor(v)
    }
}

impl From<Uint64Tensor> for Value {
    fn from(v: Uint64Tensor) -> Self {
        Value::Uint64Tensor(v)
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

impl TryFrom<Value> for Float32Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Float32Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Float64Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Float64Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Int8Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Int8Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Int16Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Int16Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Int32Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Int32Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Int64Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Int64Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Uint8Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Uint8Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Uint16Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Uint16Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Uint32Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Uint32Tensor(x) => Ok(x),
            _ => Err(Error::TypeMismatch),
        }
    }
}

impl TryFrom<Value> for Uint64Tensor {
    type Error = Error;
    fn try_from(v: Value) -> Result<Self> {
        match v {
            Value::Uint64Tensor(x) => Ok(x),
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
    Nullary(Box<dyn Fn(&SyncContext, &SessionId) -> Result<Value> + Send + Sync>),
    Unary(Box<dyn Fn(&SyncContext, &SessionId, Value) -> Result<Value> + Send + Sync>),
    Binary(Box<dyn Fn(&SyncContext, &SessionId, Value, Value) -> Result<Value> + Send + Sync>),
    Ternary(
        Box<dyn Fn(&SyncContext, &SessionId, Value, Value, Value) -> Result<Value> + Send + Sync>,
    ),
    Variadic(Box<dyn Fn(&SyncContext, &SessionId, &[Value]) -> Result<Value> + Send + Sync>),
}
pub enum AsyncKernel {
    Nullary(
        Box<dyn Fn(&Arc<AsyncContext>, &Arc<SessionId>, AsyncSender) -> AsyncTask + Send + Sync>,
    ),
    Unary(
        Box<
            dyn Fn(&Arc<AsyncContext>, &Arc<SessionId>, AsyncReceiver, AsyncSender) -> AsyncTask
                + Send
                + Sync,
        >,
    ),
    Binary(
        Box<
            dyn Fn(
                    &Arc<AsyncContext>,
                    &Arc<SessionId>,
                    AsyncReceiver,
                    AsyncReceiver,
                    AsyncSender,
                ) -> AsyncTask
                + Send
                + Sync,
        >,
    ),
    Ternary(
        Box<
            dyn Fn(
                    &Arc<AsyncContext>,
                    &Arc<SessionId>,
                    AsyncReceiver,
                    AsyncReceiver,
                    AsyncReceiver,
                    AsyncSender,
                ) -> AsyncTask
                + Send
                + Sync,
        >,
    ),
    Variadic(
        Box<
            dyn Fn(&Arc<AsyncContext>, &Arc<SessionId>, &[AsyncReceiver], AsyncSender) -> AsyncTask
                + Send
                + Sync,
        >,
    ),
}

pub type AsyncSender = oneshot::Sender<Value>;

pub type AsyncReceiver = Shared<
    Map<
        oneshot::Receiver<Value>,
        fn(anyhow::Result<Value, oneshot::error::RecvError>) -> anyhow::Result<Value, ()>,
    >,
>;

pub type AsyncTask = tokio::task::JoinHandle<Result<()>>;

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
            Kernel::NullaryClosure(k) => Ok(SyncKernel::Nullary(Box::new(move |_, _| k()))),
            Kernel::UnaryClosure(k) => Ok(SyncKernel::Unary(Box::new(move |_, _, x0| k(x0)))),
            Kernel::BinaryClosure(k) => {
                Ok(SyncKernel::Binary(Box::new(move |_, _, x0, x1| k(x0, x1))))
            }
            Kernel::TernaryClosure(k) => {
                Ok(SyncKernel::Ternary(Box::new(move |_, _, x0, x1, x2| {
                    k(x0, x1, x2)
                })))
            }
            Kernel::VariadicClosure(k) => Ok(SyncKernel::Variadic(Box::new(move |_, _, xs| k(xs)))),
            Kernel::NullaryFunction(k) => Ok(SyncKernel::Nullary(Box::new(move |_, _| k()))),
            Kernel::UnaryFunction(k) => Ok(SyncKernel::Unary(Box::new(move |_, _, x0| k(x0)))),
            Kernel::BinaryFunction(k) => {
                Ok(SyncKernel::Binary(Box::new(move |_, _, x0, x1| k(x0, x1))))
            }
            Kernel::TernaryFunction(k) => {
                Ok(SyncKernel::Ternary(Box::new(move |_, _, x0, x1, x2| {
                    k(x0, x1, x2)
                })))
            }
            Kernel::VariadicFunction(k) => {
                Ok(SyncKernel::Variadic(Box::new(move |_, _, xs| k(xs))))
            }
        }
    }
}

impl SyncCompile for SendOp {
    fn compile(&self) -> Result<SyncKernel> {
        let rdv = self.rendezvous_key.clone();
        Ok(SyncKernel::Unary(Box::new(move |ctx, sid, v| {
            ctx.networking.send(&v, &rdv, &sid);
            Ok(Value::Unit)
        })))
    }
}

impl SyncCompile for ReceiveOp {
    fn compile(&self) -> Result<SyncKernel> {
        let rdv = self.rendezvous_key.clone();
        Ok(SyncKernel::Nullary(Box::new(move |ctx, sid| {
            ctx.networking.receive(&rdv, sid)
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
            Kernel::NullaryClosure(k) => {
                Ok(AsyncKernel::Nullary(Box::new(move |ctx, _, sender| {
                    let k = Arc::clone(&k);
                    ctx.runtime.spawn(async move {
                        let y: Value = k()?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }
            Kernel::UnaryClosure(k) => {
                Ok(AsyncKernel::Unary(Box::new(move |ctx, _, x0, sender| {
                    let k = Arc::clone(&k);
                    ctx.runtime.spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let y: Value = k(x0)?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }
            Kernel::BinaryClosure(k) => Ok(AsyncKernel::Binary(Box::new(
                move |ctx, _, x0, x1, sender| {
                    let k = Arc::clone(&k);
                    ctx.runtime.spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let x1: Value = x1.await.map_err(map_receive_error)?;
                        let y: Value = k(x0, x1)?;
                        sender.send(y).map_err(map_send_error)
                    })
                },
            ))),
            Kernel::TernaryClosure(k) => Ok(AsyncKernel::Ternary(Box::new(
                move |ctx, _, x0, x1, x2, sender| {
                    let k = Arc::clone(&k);
                    ctx.runtime.spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let x1: Value = x1.await.map_err(map_receive_error)?;
                        let x2: Value = x2.await.map_err(map_receive_error)?;
                        let y: Value = k(x0, x1, x2)?;
                        sender.send(y).map_err(map_send_error)
                    })
                },
            ))),
            Kernel::VariadicClosure(_k) => unimplemented!(), // TODO

            Kernel::NullaryFunction(k) => {
                Ok(AsyncKernel::Nullary(Box::new(move |ctx, _, sender| {
                    ctx.runtime.spawn(async move {
                        let y = k()?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }
            Kernel::UnaryFunction(k) => {
                Ok(AsyncKernel::Unary(Box::new(move |ctx, _, x0, sender| {
                    ctx.runtime.spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let y: Value = k(x0)?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }
            Kernel::BinaryFunction(k) => Ok(AsyncKernel::Binary(Box::new(
                move |ctx, _, x0, x1, sender| {
                    ctx.runtime.spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let x1: Value = x1.await.map_err(map_receive_error)?;
                        let y: Value = k(x0, x1)?;
                        sender.send(y).map_err(map_send_error)
                    })
                },
            ))),
            Kernel::TernaryFunction(k) => Ok(AsyncKernel::Ternary(Box::new(
                move |ctx, _, x0, x1, x2, sender| {
                    ctx.runtime.spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let x1: Value = x1.await.map_err(map_receive_error)?;
                        let x2: Value = x2.await.map_err(map_receive_error)?;
                        let y: Value = k(x0, x1, x2)?;
                        sender.send(y).map_err(map_send_error)
                    })
                },
            ))),
            Kernel::VariadicFunction(_k) => unimplemented!(), // TODO
        }
    }
}

impl AsyncCompile for SendOp {
    fn compile(&self) -> Result<AsyncKernel> {
        let rdv = Arc::new(self.rendezvous_key.clone());
        Ok(AsyncKernel::Unary(Box::new(move |ctx, sid, v, sender| {
            let ctx = Arc::clone(ctx);
            let sid = Arc::clone(sid);
            let rdv = Arc::clone(&rdv);
            tokio::spawn(async move {
                let v: Value = v.await.map_err(map_receive_error)?;
                ctx.networking.send(&v, &rdv, &sid).await;
                sender.send(Value::Unit).map_err(map_send_error)
            })
        })))
    }
}

impl AsyncCompile for ReceiveOp {
    fn compile(&self) -> Result<AsyncKernel> {
        let rdv = Arc::new(self.rendezvous_key.clone());
        Ok(AsyncKernel::Nullary(Box::new(move |ctx, sid, sender| {
            let ctx = Arc::clone(ctx);
            let sid = Arc::clone(sid);
            let rdv = Arc::clone(&rdv);
            tokio::spawn(async move {
                let v: Value = ctx.networking.receive(&rdv, &sid).await?;
                sender.send(v).map_err(map_send_error)
            })
        })))
    }
}

pub type RendezvousKey = str;

pub type SessionId = u128;

pub struct SyncContext {
    pub networking: Box<dyn Send + Sync + SyncNetworking>,
}

pub trait SyncNetworking {
    fn send(&self, v: &Value, rendezvous_key: &RendezvousKey, session_id: &SessionId);
    fn receive(&self, rendezvous_key: &RendezvousKey, session_id: &SessionId) -> Result<Value>;
}

#[async_trait]
pub trait AsyncNetworking {
    async fn send(&self, v: &Value, rendezvous_key: &RendezvousKey, session_id: &SessionId);
    async fn receive(
        &self,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<Value>;
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

    async fn receive(
        &self,
        rendezvous_key: &RendezvousKey,
        session_id: &SessionId,
    ) -> Result<Value> {
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
    StdAdd(StdAddOp),
    StdSub(StdSubOp),
    StdMul(StdMulOp),
    StdDiv(StdDivOp),
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
    FixedpointRingEncode(FixedpointRingEncodeOp),
    FixedpointRingDecode(FixedpointRingDecodeOp),
    FixedpointRingMean(FixedpointRingMeanOp),
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
pub struct StdAddOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

impl Compile<Kernel> for StdAddOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Float32TensorTy, Ty::Float32TensorTy) => {
                function_kernel!(Float32Tensor, Float32Tensor, |x, y| x + y)
            }
            (Ty::Float64TensorTy, Ty::Float64TensorTy) => {
                function_kernel!(Float64Tensor, Float64Tensor, |x, y| x + y)
            }
            (Ty::Int32TensorTy, Ty::Int32TensorTy) => {
                function_kernel!(Int32Tensor, Int32Tensor, |x, y| x + y)
            }
            (Ty::Int64TensorTy, Ty::Int64TensorTy) => {
                function_kernel!(Int64Tensor, Int64Tensor, |x, y| x + y)
            }
            (Ty::Uint32TensorTy, Ty::Uint32TensorTy) => {
                function_kernel!(Uint32Tensor, Uint32Tensor, |x, y| x + y)
            }
            (Ty::Uint64TensorTy, Ty::Uint64TensorTy) => {
                function_kernel!(Uint64Tensor, Uint64Tensor, |x, y| x + y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdSubOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

impl Compile<Kernel> for StdSubOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Float32TensorTy, Ty::Float32TensorTy) => {
                function_kernel!(Float32Tensor, Float32Tensor, |x, y| x - y)
            }
            (Ty::Float64TensorTy, Ty::Float64TensorTy) => {
                function_kernel!(Float64Tensor, Float64Tensor, |x, y| x - y)
            }
            (Ty::Int32TensorTy, Ty::Int32TensorTy) => {
                function_kernel!(Int32Tensor, Int32Tensor, |x, y| x - y)
            }
            (Ty::Int64TensorTy, Ty::Int64TensorTy) => {
                function_kernel!(Int64Tensor, Int64Tensor, |x, y| x - y)
            }
            (Ty::Uint32TensorTy, Ty::Uint32TensorTy) => {
                function_kernel!(Uint32Tensor, Uint32Tensor, |x, y| x - y)
            }
            (Ty::Uint64TensorTy, Ty::Uint64TensorTy) => {
                function_kernel!(Uint64Tensor, Uint64Tensor, |x, y| x - y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdMulOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

impl Compile<Kernel> for StdMulOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Float32TensorTy, Ty::Float32TensorTy) => {
                function_kernel!(Float32Tensor, Float32Tensor, |x, y| x * y)
            }
            (Ty::Float64TensorTy, Ty::Float64TensorTy) => {
                function_kernel!(Float64Tensor, Float64Tensor, |x, y| x * y)
            }
            (Ty::Int32TensorTy, Ty::Int32TensorTy) => {
                function_kernel!(Int32Tensor, Int32Tensor, |x, y| x * y)
            }
            (Ty::Int64TensorTy, Ty::Int64TensorTy) => {
                function_kernel!(Int64Tensor, Int64Tensor, |x, y| x * y)
            }
            (Ty::Uint32TensorTy, Ty::Uint32TensorTy) => {
                function_kernel!(Uint32Tensor, Uint32Tensor, |x, y| x * y)
            }
            (Ty::Uint64TensorTy, Ty::Uint64TensorTy) => {
                function_kernel!(Uint64Tensor, Uint64Tensor, |x, y| x * y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdDivOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

impl Compile<Kernel> for StdDivOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Float32TensorTy, Ty::Float32TensorTy) => {
                function_kernel!(Float32Tensor, Float32Tensor, |x, y| x / y)
            }
            (Ty::Float64TensorTy, Ty::Float64TensorTy) => {
                function_kernel!(Float64Tensor, Float64Tensor, |x, y| x / y)
            }
            (Ty::Int32TensorTy, Ty::Int32TensorTy) => {
                function_kernel!(Int32Tensor, Int32Tensor, |x, y| x / y)
            }
            (Ty::Int64TensorTy, Ty::Int64TensorTy) => {
                function_kernel!(Int64Tensor, Int64Tensor, |x, y| x / y)
            }
            (Ty::Uint32TensorTy, Ty::Uint32TensorTy) => {
                function_kernel!(Uint32Tensor, Uint32Tensor, |x, y| x / y)
            }
            (Ty::Uint64TensorTy, Ty::Uint64TensorTy) => {
                function_kernel!(Uint64Tensor, Uint64Tensor, |x, y| x / y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
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
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSubOp {
    pub lhs: Ty,
    pub rhs: Ty,
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
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingMulOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

impl Compile<Kernel> for RingMulOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x * y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingDotOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

impl Compile<Kernel> for RingDotOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x.dot(y))
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSumOp {
    pub ty: Ty,
    pub axis: Option<u32>,
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
    pub ty: Ty,
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
    pub value: u64,
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
pub struct FixedpointRingEncodeOp {
    pub scaling_factor: u64,
}

impl Compile<Kernel> for FixedpointRingEncodeOp {
    fn compile(&self) -> Result<Kernel> {
        let scaling_factor = self.scaling_factor;
        closure_kernel!(Float64Tensor, |x| Ring64Tensor::encode(&x, scaling_factor))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FixedpointRingDecodeOp {
    pub scaling_factor: u64,
}

impl Compile<Kernel> for FixedpointRingDecodeOp {
    fn compile(&self) -> Result<Kernel> {
        let scaling_factor = self.scaling_factor;
        closure_kernel!(Ring64Tensor, |x| Ring64Tensor::decode(&x, scaling_factor))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FixedpointRingMeanOp {
    pub axis: Option<usize>,
    pub scaling_factor: u64,
}

impl Compile<Kernel> for FixedpointRingMeanOp {
    fn compile(&self) -> Result<Kernel> {
        let axis = self.axis;
        let scaling_factor = self.scaling_factor;
        closure_kernel!(Ring64Tensor, |x| Ring64Tensor::ring_mean(x, axis, scaling_factor))
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
    pub inputs: Vec<String>, // TODO(Morten) use indices instead of strings?
    pub placement: Placement,
}

pub struct CompiledSyncOperation {
    name: String,
    kernel:
        Box<dyn Fn(&SyncContext, &SessionId, &Environment<Value>) -> Result<Value> + Send + Sync>,
}

impl CompiledSyncOperation {
    pub fn apply(
        &self,
        ctx: &SyncContext,
        sid: &SessionId,
        args: &Environment<Value>,
    ) -> Result<Value> {
        (self.kernel)(ctx, sid, args)
    }
}

pub struct CompiledAsyncOperation {
    name: String,
    kernel: Box<
        dyn Fn(
                &Arc<AsyncContext>,
                &Arc<SessionId>,
                &Environment<AsyncReceiver>,
                AsyncSender,
            ) -> Result<AsyncTask>
            + Send
            + Sync,
    >,
}

impl CompiledAsyncOperation {
    pub fn apply(
        &self,
        ctx: &Arc<AsyncContext>,
        sid: &Arc<SessionId>,
        env: &Environment<AsyncReceiver>,
        sender: AsyncSender,
    ) -> Result<AsyncTask> {
        (self.kernel)(ctx, sid, env, sender)
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
                    kernel: Box::new(move |ctx, sid, _| k(ctx, sid)),
                })
            }
            SyncKernel::Unary(k) => {
                check_arity(&self.name, &self.inputs, 1)?;
                let x0_name = self.inputs[0].clone();
                Ok(CompiledSyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |ctx, sid, env| {
                        // TODO(Morten) avoid cloning
                        let x0 = env
                            .get(&x0_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        k(ctx, sid, x0)
                    }),
                })
            }
            SyncKernel::Binary(k) => {
                check_arity(&self.name, &self.inputs, 2)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                Ok(CompiledSyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |ctx, sid, env| {
                        // TODO(Morten) avoid cloning
                        let x0 = env
                            .get(&x0_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        let x1 = env
                            .get(&x1_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        k(ctx, sid, x0, x1)
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
                    kernel: Box::new(move |ctx, sid, env| {
                        // TODO(Morten) avoid cloning
                        let x0 = env
                            .get(&x0_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        let x1 = env
                            .get(&x1_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        let x2 = env
                            .get(&x2_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        k(ctx, sid, x0, x1, x2)
                    }),
                })
            }
            SyncKernel::Variadic(k) => {
                let inputs = self.inputs.clone();
                Ok(CompiledSyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |ctx, sid, env| {
                        let xs = inputs
                            .iter()
                            .map(|input| env.get(input).cloned().ok_or(Error::MalformedEnvironment)) // TODO(Morten avoid cloning
                            .collect::<Result<Vec<_>>>()?;
                        k(ctx, sid, &xs)
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
                    kernel: Box::new(move |ctx, sid, _, sender| Ok(k(ctx, sid, sender))),
                })
            }
            AsyncKernel::Unary(k) => {
                check_arity(&self.name, &self.inputs, 1)?;
                let x0_name = self.inputs[0].clone();
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |ctx, sid, env, sender| {
                        let x0 = env
                            .get(&x0_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        Ok(k(ctx, sid, x0, sender))
                    }),
                })
            }
            AsyncKernel::Binary(k) => {
                check_arity(&self.name, &self.inputs, 2)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |ctx, sid, env, sender| {
                        let x0 = env
                            .get(&x0_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        let x1 = env
                            .get(&x1_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        Ok(k(ctx, sid, x0, x1, sender))
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
                    kernel: Box::new(move |ctx, sid, env, sender| {
                        let x0 = env
                            .get(&x0_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        let x1 = env
                            .get(&x1_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        let x2 = env
                            .get(&x2_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        Ok(k(ctx, sid, x0, x1, x2, sender))
                    }),
                })
            }
            AsyncKernel::Variadic(k) => {
                let inputs = self.inputs.clone();
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |ctx, sid, env, sender| {
                        let xs = inputs
                            .iter()
                            .map(|input| env.get(input).cloned().ok_or(Error::MalformedEnvironment))
                            .collect::<Result<Vec<_>>>()?;
                        Ok(k(ctx, sid, &xs, sender))
                    }),
                })
            }
        }
    }
}

impl Operation {
    pub fn apply(
        &self,
        ctx: &SyncContext,
        sid: &SessionId,
        args: &Environment<Value>,
    ) -> Result<Value> {
        let compiled: CompiledSyncOperation = self.compile()?;
        Ok(compiled.apply(ctx, sid, args)?)
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

        let toposort = toposort(&graph, None).map_err(|_| {
            Error::MalformedComputation("There is a cycle detected in the runtime graph".into())
        })?;

        let operations = toposort
            .iter()
            .map(|node| self.operations[inv_map[node]].clone())
            .collect();

        Ok(Computation { operations })
    }

    pub fn apply(
        &self,
        ctx: &SyncContext,
        sid: &SessionId,
        args: Environment<Value>,
    ) -> Result<Environment<Value>> {
        let mut env = args;
        env.reserve(self.operations.len());
        for op in self.operations.iter() {
            let value = op.apply(ctx, sid, &env)?;
            env.insert(op.name.clone(), value);
        }
        Ok(env)
    }
}

pub struct CompiledSyncComputation(
    Box<dyn Fn(&SyncContext, &SessionId, Environment<Value>) -> Result<Environment<Value>>>,
);

impl Compile<CompiledSyncComputation> for Computation
where
    Operation: Compile<CompiledSyncOperation>,
{
    fn compile(&self) -> Result<CompiledSyncComputation> {
        // TODO(Morten) type check computation
        let compiled_ops: Vec<CompiledSyncOperation> = self
            .operations
            .par_iter() // par_iter seems to make sense here, see benches
            .map(|op| op.compile())
            .collect::<Result<Vec<_>>>()?;
        // TODO(Morten) we want to sort topologically here, outside the closure
        // TODO(Morten) do we want to insert instructions for when values can be dropped from the environment?
        Ok(CompiledSyncComputation(Box::new(
            move |ctx: &SyncContext, sid: &SessionId, args: Environment<Value>| {
                let mut env = args;
                env.reserve(compiled_ops.len());
                for compiled_op in compiled_ops.iter() {
                    let value = compiled_op.apply(ctx, sid, &env)?;
                    env.insert(compiled_op.name.clone(), value);
                }
                Ok(env)
            },
        )))
    }
}

impl CompiledSyncComputation {
    pub fn apply(
        &self,
        ctx: &SyncContext,
        sid: &SessionId,
        args: Environment<Value>,
    ) -> Result<Environment<Value>> {
        (self.0)(ctx, sid, args)
    }
}

pub struct CompiledAsyncComputation(
    Box<
        dyn Fn(
            &Arc<AsyncContext>,
            &Arc<SessionId>,
            Environment<AsyncReceiver>,
        ) -> Result<(AsyncSession, Environment<AsyncReceiver>)>,
    >,
);

impl Compile<CompiledAsyncComputation> for Computation
where
    Operation: Compile<CompiledAsyncOperation>,
{
    fn compile(&self) -> Result<CompiledAsyncComputation> {
        let compiled_ops: Vec<CompiledAsyncOperation> = self
            .operations
            .par_iter() // par_iter seems to make sense here, see benches
            .map(|op| op.compile())
            .collect::<Result<Vec<_>>>()?;

        fn remove_err<T, E>(r: std::result::Result<T, E>) -> std::result::Result<T, ()> {
            r.map_err(|_| ())
        }

        Ok(CompiledAsyncComputation(Box::new(
            move |ctx: &Arc<AsyncContext>,
                  sid: &Arc<SessionId>,
                  _args: Environment<AsyncReceiver>| {
                // TODO(Morten) args should be passed into the op.apply's
                let (senders, receivers): (Vec<AsyncSender>, HashMap<String, AsyncReceiver>) =
                    compiled_ops
                        .iter() // par_iter doesn't seem to improve performance here
                        .map(|op| {
                            let (sender, receiver) = oneshot::channel();
                            let shared_receiver: AsyncReceiver =
                                receiver.map(remove_err as fn(_) -> _).shared();
                            (sender, (op.name.clone(), shared_receiver))
                        })
                        .unzip();
                let tasks: Vec<AsyncTask> = senders
                    .into_iter() // into_par_iter seems to hurt performance here
                    .zip(&compiled_ops)
                    .map(|(sender, op)| op.apply(ctx, sid, &receivers, sender))
                    .collect::<Result<Vec<_>>>()?;
                let outputs = receivers; // TODO filter to Output nodes
                Ok((AsyncSession { tasks }, outputs))
            },
        )))
    }
}

impl CompiledAsyncComputation {
    pub fn apply(
        &self,
        ctx: &Arc<AsyncContext>,
        sid: &Arc<SessionId>,
        args: Environment<AsyncReceiver>,
    ) -> Result<(AsyncSession, Environment<AsyncReceiver>)> {
        (self.0)(ctx, sid, args)
    }
}

pub type Environment<V> = HashMap<String, V>;

/// In-order single-threaded executor.
///
/// This executor evaluates the operations of computations in-order, raising an error
/// in case data dependencies are not respected. This executor is intended for debug
/// and development only due to its unforgiving but highly predictable behaviour.
pub struct EagerExecutor {
    ctx: SyncContext,
}

impl EagerExecutor {
    pub fn new() -> EagerExecutor {
        let ctx = SyncContext {
            networking: Box::new(DummySyncNetworking),
        };
        EagerExecutor { ctx }
    }

    pub fn run_computation(
        &self,
        comp: &Computation,
        sid: SessionId,
        args: Environment<Value>,
    ) -> Result<()> {
        let compiled_comp: CompiledSyncComputation = comp.compile()?;
        let _env = compiled_comp.apply(&self.ctx, &sid, args)?;
        Ok(())
    }
}

pub struct AsyncContext {
    pub runtime: tokio::runtime::Runtime,
    pub networking: Box<dyn Send + Sync + AsyncNetworking>,
}

impl AsyncContext {
    pub fn join_session(&self, sess: AsyncSession) -> Result<()> {
        for task in sess.tasks {
            let res = self.runtime.block_on(task);
            if res.is_err() {
                unimplemented!() // TODO
            }
        }
        Ok(())
    }
}

pub struct AsyncSession {
    tasks: Vec<AsyncTask>,
}

pub struct AsyncExecutor {
    ctx: Arc<AsyncContext>,
}

impl AsyncExecutor {
    pub fn new() -> AsyncExecutor {
        let ctx = Arc::new(AsyncContext {
            runtime: tokio::runtime::Runtime::new().expect("Failed to build tokio runtime"),
            networking: Box::new(DummyAsyncNetworking),
        });
        AsyncExecutor { ctx }
    }

    pub fn run_computation(
        &self,
        comp: &Computation,
        sid: SessionId,
        args: Environment<AsyncReceiver>,
    ) -> Result<()> {
        let compiled_comp: CompiledAsyncComputation = comp.compile()?;
        let sid = Arc::new(sid);
        let (sess, _vals) = match compiled_comp.apply(&self.ctx, &sid, args) {
            Ok(res) => res,
            Err(_e) => {
                return Err(Error::Unexpected); // TODO
            }
        };
        self.ctx.join_session(sess)?;
        Ok(()) // TODO(Morten) return vals
    }
}

#[test]
fn test_eager_executor() {
    use maplit::hashmap;

    let env = hashmap![];

    let key_op = Operation {
        name: "key".into(),
        kind: Operator::PrimGenPrfKey(PrimGenPrfKeyOp),
        inputs: vec![],
        placement: Placement::Host,
    };

    let seed_op = Operation {
        name: "seed".into(),
        kind: Operator::PrimDeriveSeed(PrimDeriveSeedOp {
            nonce: Nonce(vec![1, 2, 3]),
        }),
        inputs: vec!["key".into()],
        placement: Placement::Host,
    };

    let shape_op = Operation {
        name: "shape".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Shape(Shape(vec![2, 3])),
        }),
        inputs: vec![],
        placement: Placement::Host,
    };

    let sample_ops: Vec<_> = (0..100)
        .map(|i| {
            Operation {
                name: format!("x{}", i),
                kind: Operator::RingSample(RingSampleOp { output: Ty::Ring64TensorTy, max_value: None }),
                inputs: vec!["shape".into(), "seed".into()],
                placement: Placement::Host,
            }
        })
        .collect();

    let mut operations = sample_ops;
    operations.extend(vec![key_op, seed_op, shape_op]);

    let comp = Computation { operations }
        .toposort()
        .unwrap();

    let exec = EagerExecutor::new();
    exec.run_computation(&comp, 12345, env).ok();
}
