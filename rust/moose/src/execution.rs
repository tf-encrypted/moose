use enum_dispatch::enum_dispatch;
use maplit::hashmap;
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::rc::Rc;
use std::{collections::HashMap, convert::TryFrom, marker::PhantomData};
use tokio;
use tokio::sync::broadcast::{Sender, Receiver};
use crate::ring::{Dot, Ring128Tensor, Ring64Tensor, Sample};

#[derive(Clone, Debug)]
pub enum Value {
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    Shape(Vec<usize>),
    Seed(Vec<u8>),
}

#[derive(Serialize, Deserialize, Clone, Debug, Hash)]
pub enum Ty {
    Ring64TensorTy,
    Ring128TensorTy,
    ShapeTy,
    SeedTy,
}

// TODO(Morten) this should be TryFrom instead, with proper error handling
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

// TODO(Morten) this should be TryFrom instead, with proper error handling
impl From<Value> for Vec<usize> {
    fn from(v: Value) -> Self {
        match v {
            Value::Shape(x) => x,
            _ => unimplemented!(),
        }
    }
}

// TODO(Morten) this should be TryFrom instead, with proper error handling
impl From<Value> for Vec<u8> {
    fn from(v: Value) -> Self {
        match v {
            Value::Seed(x) => x,
            _ => unimplemented!(),
        }
    }
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









enum NullaryKernel<Y> {
    Function(fn() -> Y),
    Closure(Rc<dyn Fn() -> Y>),
}

impl<Y> Lift for NullaryKernel<Y>
where
    Y: 'static,
    Value: From<Y>,
{
    fn lift(&self) -> SyncKernel {
        match self {
            NullaryKernel::Function(k) => {
                let k = k.clone();  // TODO(Morten) avoid clone if possible
                SyncKernel::Nullary(Box::new(move || {
                    let y = k();
                    Value::from(y)
                }))
            }
            NullaryKernel::Closure(k) => {
                let k = k.clone();
                SyncKernel::Nullary(Box::new(move || {
                    let y = k();
                    Value::from(y)
                }))
            }
        }
    }
}

enum UnaryKernel<X0, Y> {
    Function(fn(X0) -> Y),
    Closure(Rc<dyn Fn(X0) -> Y>),
}

impl<X0, Y> Lift for UnaryKernel<X0, Y>
where
    X0: 'static + From<Value>,
    Y: 'static,
    Value: From<Y>,
{
    fn lift(&self) -> SyncKernel {
        match self {
            UnaryKernel::Function(k) => {
                let k = k.clone();  // TODO(Morten) avoid clone if possible
                SyncKernel::Unary(Box::new(move |x0| {
                    let x0 = X0::from(x0);
                    let y = k(x0);
                    Value::from(y)
                }))
            }
            UnaryKernel::Closure(k) => {
                let k = k.clone();
                SyncKernel::Unary(Box::new(move |x0| {
                    let x0 = X0::from(x0);
                    let y = k(x0);
                    Value::from(y)
                }))
            }
        }
    }
}
enum BinaryKernel<X0, X1, Y> {
    Function(fn(X0, X1) -> Y),
    Closure(Rc<dyn Fn(X0, X1) -> Y>),
}

impl<X0, X1, Y> Lift for BinaryKernel<X0, X1, Y>
where
    X0: 'static + From<Value>,
    X1: 'static + From<Value>,
    Y: 'static,
    Value: From<Y>,
{
    fn lift(&self) -> SyncKernel {
        match self {
            BinaryKernel::Function(k) => {
                let k = k.clone();  // TODO(Morten) avoid clone if possible
                SyncKernel::Binary(Box::new(move |x0, x1| {
                    let x0 = X0::from(x0);
                    let x1 = X1::from(x1);
                    let y = k(x0, x1);
                    Value::from(y)
                }))
            }
            BinaryKernel::Closure(k) => {
                let k = k.clone();
                SyncKernel::Binary(Box::new(move |x0, x1| {
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
            let k = NullarayKernel::Function(<Self as NullarayFunction::execute);
            CompilableKernel(Box::new(k))
        }
    };
    ($self:ident) => {
        {
            let s = $self.clone();
            let k = NullarayKernel::Closure(Rc::new(move || <Self as NullarayClosure::execute(&s)));
            CompilableKernel(Box::new(k))
        }
    };
}

macro_rules! unary_kernel {
    ($t0:ty) => {
        {
            let k = UnaryKernel::Function(<Self as UnaryFunction::<$t0>>::execute);
            CompilableKernel(Box::new(k))
        }
    };
    ($self:ident, $t0:ty) => {
        {
            let s = $self.clone();
            let k = UnaryKernel::Closure(Rc::new(move |x0| <Self as UnaryClosure::<$t0>>::execute(&s, x0)));
            CompilableKernel(Box::new(k))
        }
    };
}

macro_rules! binary_kernel {
    ($t0:ty, $t1:ty) => {
        {
            let k = BinaryKernel::Function(<Self as BinaryFunction::<$t0, $t1>>::execute);
            CompilableKernel(Box::new(k))
        }
    };
    ($self:ident, $t0:ty, $t1:ty) => {
        {
            let s = $self.clone();
            let k = BinaryKernel::Closure(Rc::new(move |x0, x1| <Self as BinaryClosure::<$t0, $t1>>::execute(&s, x0, x1)));
            CompilableKernel(Box::new(k))
        }
    };
}





pub struct CompilableKernel(Box<dyn Lift>);





pub enum SyncKernel {
    Nullary(Box<dyn Fn() -> Value>),
    Unary(Box<dyn Fn(Value) -> Value>),
    Binary(Box<dyn Fn(Value, Value) -> Value>),
    Ternary(Box<dyn Fn(Value, Value, Value) -> Value>),
    Variadic(Box<dyn Fn(&[Value]) -> Value>),
}

pub enum AsyncKernel {
    Nullary(Box<dyn Fn() -> Sender<Value>>),
    Unary(Box<dyn Fn(Receiver<Value>) -> Sender<Value>>),
    Binary(Box<dyn Fn(Receiver<Value>, Receiver<Value>) -> Sender<Value>>),
    Ternary(Box<dyn Fn(Receiver<Value>, Receiver<Value>, Receiver<Value>) -> Sender<Value>>),
    Variadic(Box<dyn Fn(&[Receiver<Value>]) -> Sender<Value>>),
}




#[enum_dispatch]
trait Compile {
    fn compile(&self) -> SyncKernel;
    fn async_compile(&self) -> AsyncKernel {
        unimplemented!()
    }
}

pub trait Lift {
    fn lift(&self) -> SyncKernel;
    fn async_lift(&self) -> AsyncKernel {
        unimplemented!()
    }
}


trait Kernel {
    fn kernel(&self) -> CompilableKernel;
    fn async_kernel(&self) -> () {
        // TODO: default impl calling `kernel`
        unimplemented!()
    }
}









#[derive(Clone)]
pub struct FooOp(Ty);

impl Kernel for FooOp {
    fn kernel(&self) -> CompilableKernel {
        match self.0 {
            Ty::Ring64TensorTy => binary_kernel!(Ring64Tensor, Ring64Tensor),
            Ty::Ring128TensorTy => binary_kernel!(self, Ring128Tensor, Ring128Tensor),
            _ => unimplemented!()
        }
    }
}

impl BinaryFunction<Ring64Tensor, Ring64Tensor> for FooOp {
    type Output = Ring64Tensor;
    fn execute(x: Ring64Tensor, y: Ring64Tensor) -> Self::Output {
        x + y
    }
}

impl BinaryFunction<Ring128Tensor, Ring128Tensor> for FooOp {
    type Output = Ring128Tensor;
    fn execute(x: Ring128Tensor, y: Ring128Tensor) -> Ring128Tensor {
        x + y
    }
}

impl BinaryClosure<Ring128Tensor, Ring128Tensor> for FooOp {
    type Output = Ring128Tensor;
    fn execute(&self, x: Ring128Tensor, y: Ring128Tensor) -> Ring128Tensor {
        x + y
    }
}



fn fubar() {
    let op = FooOp(Ty::Ring64TensorTy);
    let k = op.kernel();
    let c = k.0.lift(); // TODO
}

// impl BinaryKernel<Ring64Tensor, Ring64Tensor> for RingAddOp {
//     type Output = Ring64Tensor;
//     fn execute(x: Ring64Tensor, y: Ring64Tensor) -> Self::Output {
//         x + y
//     }
// }

// impl BinaryKernel<Ring128Tensor, Ring128Tensor> for RingAddOp {
//     type Output = Ring128Tensor;
//     fn execute(x: Ring128Tensor, y: Ring128Tensor) -> Self::Output {
//         Ring128Tensor(x.0 + y.0)  // TODO(Morten)
//     }
// }

// impl<X0, X1, Y> From<RingAddKernel<X0, X1, Y>> for Kernel
// where
//     X0: From<Value>,
//     X1: From<Value>,
//     Y: Into<Value>,
//     RingAddKernel<X0, X1, Y>: BinaryKernel<X0, X1>,
// {
//     fn from(k: RingAddKernel<X0, X1, Y>) -> Self {
//         Kernel::Binary(Box::new(move |x0, x1| {
//             let x0 = X0::from(x0);
//             let x1 = X1::from(x1);
//             let y = k.execute(x0, x1);
//             Value::from(y)
//         }))
//     }
// }


#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingAddOp{ lhs: Ty, rhs: Ty }


impl Compile for RingAddOp {
    fn compile(&self) -> SyncKernel {
        SyncKernel::Binary(Box::new(move |x, y| {
            let x: Ring64Tensor = x.into();
            let y: Ring64Tensor = y.into();
            let z = x + y;
            z.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingSubOp;

impl RingSubOp {
    fn execute(x: Ring64Tensor, y: Ring64Tensor) -> Ring64Tensor {
        x - y
    }
}

impl Compile for RingSubOp {
    fn compile(&self) -> SyncKernel {
        SyncKernel::Binary(Box::new(move |x, y| {
            let x: Ring64Tensor = x.into();
            let y: Ring64Tensor = y.into();
            let z = RingSubOp::execute(x, y);
            z.into()
        }))
    }

    fn async_compile(&self) -> AsyncKernel {
        AsyncKernel::Binary(Box::new(move |mut x, mut y| {
            let (sender, _) = tokio::sync::broadcast::channel(1);
            let subscriber = sender.clone();
            tokio::spawn(async move {
                let x = x.recv().await.unwrap();
                let y = y.recv().await.unwrap();

                let x: Ring64Tensor = x.into();
                let y: Ring64Tensor = y.into();
                let z = x - y;
                let z = z.into();

                sender.send(z)
            });
            subscriber
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingMulOp;

impl Compile for RingMulOp {
    fn compile(&self) -> SyncKernel {
        SyncKernel::Binary(Box::new(move |x, y| {
            let x: Ring64Tensor = x.into();
            let y: Ring64Tensor = y.into();
            let z = x * y;
            z.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingDotOp;

impl Compile for RingDotOp {
    fn compile(&self) -> SyncKernel {
        SyncKernel::Binary(Box::new(move |x, y| {
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

impl Compile for RingSumOp {
    fn compile(&self) -> SyncKernel {
        let axis = self.axis;
        SyncKernel::Unary(Box::new(move |x| {
            let x: Ring64Tensor = x.into();
            let y = x.sum(axis);
            Value::from(y)
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingShapeOp;

impl Compile for RingShapeOp {
    fn compile(&self) -> SyncKernel {
        SyncKernel::Unary(Box::new(move |x| match x {
            Value::Ring64Tensor(x) => Value::Shape(x.0.shape().into()),
            _ => unimplemented!(),
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingFillOp {
    value: u64,
}

impl Compile for RingFillOp {
    fn compile(&self) -> SyncKernel {
        let value = self.value;
        SyncKernel::Unary(Box::new(move |shape| match shape {
            Value::Shape(shape) => Value::Ring64Tensor(Ring64Tensor::fill(&shape, value)),
            _ => unimplemented!(),
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Default, Hash)]
pub struct RingSampleOp {
    pub max_value: Option<u64>,
}

impl Compile for RingSampleOp {
    fn compile(&self) -> SyncKernel {
        match self.max_value {
            None => SyncKernel::Binary(Box::new(|shape, seed| match (shape, seed) {
                (Value::Shape(shape), Value::Seed(seed)) => {
                    Value::Ring64Tensor(Ring64Tensor::sample_uniform(&shape, &seed))
                }
                _ => unimplemented!(),
            })),
            Some(max_value) if max_value == 1 => {
                SyncKernel::Binary(Box::new(|shape, seed| match (shape, seed) {
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

impl Compile for RingShlOp {
    fn compile(&self) -> SyncKernel {
        let amount = self.amount;
        SyncKernel::Unary(Box::new(move |x| {
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

impl Compile for RingShrOp {
    fn compile(&self) -> SyncKernel {
        let amount = self.amount;
        SyncKernel::Unary(Box::new(move |x| {
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
        let operator_kernel: SyncKernel = self.kind.compile();
        match (operator_kernel, self.inputs.len()) {
            (SyncKernel::Nullary(k), 0) => CompiledOperation {
                operation_name: self.name.clone(),
                kernel: Box::new(move |_: &Environment<Value>| k()),
            },
            (SyncKernel::Unary(k), 1) => {
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
            (SyncKernel::Binary(k), 2) => {
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
            (SyncKernel::Ternary(k), 3) => {
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
            (SyncKernel::Variadic(k), _) => {
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
