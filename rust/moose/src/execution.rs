use crate::ring::{Dot, Ring128Tensor, Ring64Tensor, Sample};
use enum_dispatch::enum_dispatch;
use futures::prelude::*;
use maplit::hashmap;
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::ops::Add;
use std::rc::Rc;
use std::sync::Arc;
use std::{collections::HashMap, convert::TryFrom, marker::PhantomData};
use tokio;
use tokio::sync::broadcast::{Receiver, Sender};

#[derive(Serialize, Deserialize, Copy, Clone, Debug, Hash)]
pub enum Ty {
    Ring64TensorTy,
    Ring128TensorTy,
    ShapeTy,
    SeedTy,
}

#[derive(Clone, Debug)]
pub enum Value {
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    Shape(Vec<usize>),
    Seed(Vec<u8>),
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

// TODO(Morten) we'll want a newtype for shapes
impl From<Vec<usize>> for Value {
    fn from(v: Vec<usize>) -> Self {
        Value::Shape(v)
    }
}

// TODO(Morten) we'll want a newtype for seeds
impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Self {
        Value::Seed(v)
    }
}

// TODO(Morten) all the From<Value> below should be TryFrom instead, with proper error handling

impl From<Value> for Ring64Tensor {
    fn from(v: Value) -> Self {
        match v {
            Value::Ring64Tensor(x) => x,
            _ => unimplemented!(),
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

impl From<Value> for Vec<usize> {
    fn from(v: Value) -> Self {
        match v {
            Value::Shape(x) => x,
            _ => unimplemented!(),
        }
    }
}

impl From<Value> for Vec<u8> {
    fn from(v: Value) -> Self {
        match v {
            Value::Seed(x) => x,
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

pub trait UnaryFunction<X0> {
    type Output;
    fn execute(x0: X0) -> Self::Output;
}

pub trait UnaryClosure<X0> {
    type Output;
    fn execute(&self, x0: X0) -> Self::Output;
}

pub trait BinaryFunction<X0, X1> {
    type Output;
    fn execute(x0: X0, x1: X1) -> Self::Output;
}

pub trait BinaryClosure<X0, X1> {
    type Output;
    fn execute(&self, x0: X0, x1: X1) -> Self::Output;
}

macro_rules! nullary_kernel {
    () => {
        {
            NullarayKernel::Function(<Self as NullarayFunction::execute).into()
        }
    };
    ($self:ident) => {
        {
            let s = $self.clone();
            NullarayKernel::Closure(Rc::new(move || <Self as NullarayClosure::execute(&s))).into()
        }
    };
}

macro_rules! unary_kernel {
    ($t0:ty) => {{
        TypedUnaryKernel::Function(<Self as UnaryFunction<$t0>>::execute).into()
    }};
    ($self:ident, $t0:ty) => {{
        let s = $self.clone();
        TypedUnaryKernel::Closure(Rc::new(move |x0| {
            <Self as UnaryClosure<$t0>>::execute(&s, x0)
        }))
        .into()
    }};
}

macro_rules! binary_kernel {
    ($t0:ty, $t1:ty) => {{
        TypedBinaryKernel::Function(<Self as BinaryFunction<$t0, $t1>>::execute).into()
    }};
    ($self:ident, $t0:ty, $t1:ty) => {{
        let s = $self.clone();
        TypedBinaryKernel::Closure(Arc::new(move |x0, x1| {
            <Self as BinaryClosure<$t0, $t1>>::execute(&s, x0, x1)
        }))
        .into()
    }};
}

pub enum Kernel {
    Nullary(Arc<dyn Fn() -> Value + Send + Sync>),
    Unary(Arc<dyn Fn(Value) -> Value + Send + Sync>),
    Binary(Arc<dyn Fn(Value, Value) -> Value + Send + Sync>),
    Ternary(Arc<dyn Fn(Value, Value, Value) -> Value + Send + Sync>),
    Variadic(Arc<dyn Fn(&[Value]) -> Value + Send + Sync>),
}

pub enum AsyncKernel {
    Nullary(Box<dyn Fn() -> Sender<Value>>),
    Unary(Box<dyn Fn(Receiver<Value>) -> Sender<Value>>),
    Binary(Box<dyn Fn(Receiver<Value>, Receiver<Value>) -> Sender<Value>>),
    Ternary(Box<dyn Fn(Receiver<Value>, Receiver<Value>, Receiver<Value>) -> Sender<Value>>),
    Variadic(Box<dyn Fn(&[Receiver<Value>]) -> Sender<Value>>),
}

impl From<Kernel> for AsyncKernel {
    fn from(sync_kernel: Kernel) -> AsyncKernel {
        match sync_kernel {
            Kernel::Nullary(k) => AsyncKernel::Nullary(Box::new(move || {
                let (sender, _) = tokio::sync::broadcast::channel(1);
                let subscriber = sender.clone();
                let k = k.clone();
                tokio::spawn(async move {
                    let y = k();
                    sender.send(y)
                });
                subscriber
            })),
            Kernel::Unary(k) => AsyncKernel::Unary(Box::new(move |mut x0| {
                let (sender, _) = tokio::sync::broadcast::channel(1);
                let subscriber = sender.clone();
                let k = k.clone();
                tokio::spawn(async move {
                    let x0 = x0.recv().await.unwrap();
                    let y = k(x0);
                    sender.send(y)
                });
                subscriber
            })),
            Kernel::Binary(k) => AsyncKernel::Binary(Box::new(move |mut x0, mut x1| {
                let (sender, _) = tokio::sync::broadcast::channel(1);
                let subscriber = sender.clone();
                let k = k.clone();
                tokio::spawn(async move {
                    let x0 = x0.recv().await.unwrap();
                    let x1 = x1.recv().await.unwrap();
                    let y = k(x0, x1);
                    sender.send(y)
                });
                subscriber
            })),
            Kernel::Ternary(k) => AsyncKernel::Ternary(Box::new(move |mut x0, mut x1, mut x2| {
                let (sender, _) = tokio::sync::broadcast::channel(1);
                let subscriber = sender.clone();
                let k = k.clone();
                tokio::spawn(async move {
                    let x0 = x0.recv().await.unwrap();
                    let x1 = x1.recv().await.unwrap();
                    let x2 = x2.recv().await.unwrap();
                    let y = k(x0, x1, x2);
                    sender.send(y)
                });
                subscriber
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
#[derive(Serialize, Deserialize, Debug, Hash)]
pub enum Operator {
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
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingAddOp {
    lhs: Ty,
    rhs: Ty,
}

impl Compile for RingAddOp {
    fn compile(&self) -> Kernel {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => binary_kernel!(Ring64Tensor, Ring64Tensor),
            (Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                binary_kernel!(Ring128Tensor, Ring128Tensor)
            }
            _ => unimplemented!(),
        }
    }
}

impl<U> BinaryFunction<Ring64Tensor, U> for RingAddOp
where
    Ring64Tensor: Add<U>,
{
    type Output = <Ring64Tensor as Add<U>>::Output;
    fn execute(x: Ring64Tensor, y: U) -> Self::Output {
        x + y
    }
}

impl<U> BinaryFunction<Ring128Tensor, U> for RingAddOp
where
    Ring128Tensor: Add<U>,
{
    type Output = <Ring128Tensor as Add<U>>::Output;
    fn execute(x: Ring128Tensor, y: U) -> Self::Output {
        x + y
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingSubOp;

impl BinaryFunction<Ring64Tensor, Ring64Tensor> for RingSubOp {
    type Output = Ring64Tensor;
    fn execute(x: Ring64Tensor, y: Ring64Tensor) -> Ring64Tensor {
        x - y
    }
}

impl Compile for RingSubOp {
    fn compile(&self) -> Kernel {
        binary_kernel!(Ring64Tensor, Ring64Tensor)
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
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

#[derive(Serialize, Deserialize, Debug, Hash)]
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

#[derive(Serialize, Deserialize, Debug, Hash)]
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

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingShapeOp;

// TODO(Morten) rewrite
impl Compile for RingShapeOp {
    fn compile(&self) -> Kernel {
        Kernel::Unary(Arc::new(move |x| match x {
            Value::Ring64Tensor(x) => Value::Shape(x.0.shape().into()),
            _ => unimplemented!(),
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingFillOp {
    value: u64,
}

// TODO(Morten) rewrite
impl Compile for RingFillOp {
    fn compile(&self) -> Kernel {
        let value = self.value;
        Kernel::Unary(Arc::new(move |shape| match shape {
            Value::Shape(shape) => Value::Ring64Tensor(Ring64Tensor::fill(&shape, value)),
            _ => unimplemented!(),
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Default, Hash)]
pub struct RingSampleOp {
    pub max_value: Option<u64>,
}

// TODO(Morten) rewrite
impl Compile for RingSampleOp {
    fn compile(&self) -> Kernel {
        match self.max_value {
            None => Kernel::Binary(Arc::new(|shape, seed| match (shape, seed) {
                (Value::Shape(shape), Value::Seed(seed)) => {
                    Value::Ring64Tensor(Ring64Tensor::sample_uniform(&shape, &seed))
                }
                _ => unimplemented!(),
            })),
            Some(max_value) if max_value == 1 => {
                Kernel::Binary(Arc::new(|shape, seed| match (shape, seed) {
                    (Value::Shape(shape), Value::Seed(seed)) => {
                        Value::Ring64Tensor(Ring64Tensor::sample_bits(&shape, &seed))
                    }
                    _ => unimplemented!(),
                }))
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingShlOp {
    amount: usize,
}

// TODO(Morten) rewrite
impl Compile for RingShlOp {
    fn compile(&self) -> Kernel {
        let amount = self.amount;
        Kernel::Unary(Arc::new(move |x| {
            let x: Ring64Tensor = x.into();
            let y = x << amount;
            y.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
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

#[derive(Serialize, Deserialize, Debug, Hash)]
pub enum Placement {
    Host,
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct Operation {
    pub name: String,
    pub kind: Operator,
    pub inputs: Vec<String>,
    pub placement: Placement,
}

pub struct CompiledOperation<V> {
    operation_name: String,
    kernel: Box<dyn Fn(&Environment<V>) -> V>,
}

impl Operation {
    pub fn compile(&self) -> CompiledOperation<Value> {
        let operator_kernel: Kernel = self.kind.compile();
        match (operator_kernel, self.inputs.len()) {
            (Kernel::Nullary(k), 0) => CompiledOperation {
                operation_name: self.name.clone(),
                kernel: Box::new(move |_: &Environment<Value>| k()),
            },
            (Kernel::Unary(k), 1) => {
                let x0_name = self.inputs[0].clone();
                CompiledOperation {
                    operation_name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<Value>| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).unwrap().clone();
                        k(x0)
                    }),
                }
            }
            (Kernel::Binary(k), 2) => {
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                CompiledOperation {
                    operation_name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<Value>| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).unwrap().clone();
                        let x1 = env.get(&x1_name).unwrap().clone();
                        k(x0, x1)
                    }),
                }
            }
            (Kernel::Ternary(k), 3) => {
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                let x2_name = self.inputs[2].clone();
                CompiledOperation {
                    operation_name: self.name.clone(),
                    kernel: Box::new(move |env| {
                        // TODO(Morten) avoid cloning
                        let x0 = env.get(&x0_name).unwrap().clone();
                        let x1 = env.get(&x1_name).unwrap().clone();
                        let x2 = env.get(&x2_name).unwrap().clone();
                        k(x0, x1, x2)
                    }),
                }
            }
            (Kernel::Variadic(k), _) => {
                let inputs = self.inputs.clone();
                CompiledOperation {
                    operation_name: self.name.clone(),
                    kernel: Box::new(move |env| {
                        let xs: Vec<_> = inputs
                            .iter()
                            .map(|input| env.get(input).unwrap())
                            .map(|value| value.clone()) // TODO(Morten) avoid cloning
                            .collect();
                        k(&xs)
                    }),
                }
            }
            _ => unimplemented!(),
        }
    }

    pub fn async_compile(&self) -> CompiledOperation<Sender<Value>> {
        let operator_kernel: AsyncKernel = self.kind.async_compile();
        match (operator_kernel, self.inputs.len()) {
            (AsyncKernel::Nullary(k), 0) => CompiledOperation {
                operation_name: self.name.clone(),
                kernel: Box::new(move |_: &Environment<Sender<Value>>| k()),
            },
            (AsyncKernel::Unary(k), 1) => {
                let x0_name = self.inputs[0].clone();
                CompiledOperation {
                    operation_name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<Sender<Value>>| {
                        let x0 = env.get(&x0_name).unwrap().subscribe();
                        k(x0)
                    }),
                }
            }
            (AsyncKernel::Binary(k), 2) => {
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                CompiledOperation {
                    operation_name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<Sender<Value>>| {
                        let x0 = env.get(&x0_name).unwrap().subscribe();
                        let x1 = env.get(&x1_name).unwrap().subscribe();
                        k(x0, x1)
                    }),
                }
            }
            (AsyncKernel::Ternary(k), 3) => {
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                let x2_name = self.inputs[2].clone();
                CompiledOperation {
                    operation_name: self.name.clone(),
                    kernel: Box::new(move |env| {
                        let x0 = env.get(&x0_name).unwrap().subscribe();
                        let x1 = env.get(&x1_name).unwrap().subscribe();
                        let x2 = env.get(&x2_name).unwrap().subscribe();
                        k(x0, x1, x2)
                    }),
                }
            }
            (AsyncKernel::Variadic(k), _) => {
                let inputs = self.inputs.clone();
                CompiledOperation {
                    operation_name: self.name.clone(),
                    kernel: Box::new(move |env| {
                        let xs: Vec<_> = inputs
                            .iter()
                            .map(|input| env.get(input).unwrap())
                            .map(|value| value.subscribe())
                            .collect();
                        k(&xs)
                    }),
                }
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Hash)]
pub struct Computation {
    pub operations: Vec<Operation>,
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
    pub fn run_computation(&self, comp: &Computation, args: EagerEnvironment) {
        let mut env = args;
        for op in comp.operations.iter() {
            let compiled_op = op.compile();
            let value = (compiled_op.kernel)(&env);
            env.insert(op.name.clone(), value);
        }
        println!("{:?}", env);
    }
}

// pub struct AsyncExecutor;

// impl AsyncExecutor {
//     pub fn run_computation(&self, comp: &Computation, args: Environment<impl Future<Output=Value>>) {
//         use tokio::sync::broadcast;

//         let mut env = args;
//         for op in comp.operations.iter() {
//             let (tx, mut rx) = broadcast::channel(1);
//             // let mut rx = tx.subscribe();

//             let compiled_op = op.compile();
//             let kernel = compiled_op.kernel;

//             let task_handle = tokio::spawn(async move {
//                 let value = kernel(&env);
//             });

//             env.insert(op.name.clone(), value);
//         }
//         println!("{:?}", env);

//         // let mut env = inputs;
//         // for op in comp.operations.iter() {
//         //     let compiled_op = op.compile();
//         //     let kernel = compiled_op.kernel;
//         //     let value = kernel(&env);
//         //     env.insert(op.name.clone(), value);
//         // }
//         // println!("{:?}", env);

//         unimplemented!()
//     }
// }

#[test]
fn test_foo() {
    let x_op = Operation {
        name: "x".into(),
        kind: Operator::RingSample(RingSampleOp { max_value: Some(4) }),
        inputs: vec![],
        placement: Placement::Host,
    };

    let y_op = Operation {
        name: "y".into(),
        kind: Operator::RingSample(RingSampleOp { max_value: None }),
        inputs: vec![],
        placement: Placement::Host,
    };

    let z_op = Operation {
        name: "z".into(),
        kind: Operator::RingMul(RingMulOp),
        inputs: vec!["x".into(), "y".into()],
        placement: Placement::Host,
    };

    let v_op = Operation {
        name: "v".into(),
        kind: Operator::RingAdd(RingAddOp),
        inputs: vec!["x".into(), "y".into()],
        placement: Placement::Host,
    };

    let comp = Computation {
        operations: vec![x_op, y_op, z_op, v_op],
    };

    let exec = EagerExecutor;
    exec.run_computation(&comp, hashmap![]);

    assert!(false);
}
