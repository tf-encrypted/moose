use enum_dispatch::enum_dispatch;
use maplit::hashmap;
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::rc::Rc;
use std::{collections::HashMap, convert::TryFrom, marker::PhantomData};

use crate::ring::{Dot, Ring128Tensor, Ring64Tensor, Sample};

#[derive(Clone, Debug)]
pub enum Value {
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    Shape(Vec<usize>),
    Seed(Vec<u8>),
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

pub enum Kernel {
    Nullary(Box<dyn Fn() -> Value>),
    Unary(Box<dyn Fn(Value) -> Value>),
    Binary(Box<dyn Fn(Value, Value) -> Value>),
    Ternary(Box<dyn Fn(Value, Value, Value) -> Value>),
    Variadic(Box<dyn Fn(&[Value]) -> Value>),
}

#[enum_dispatch]
trait Compile {
    fn compile(&self) -> Kernel;
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingAddOp;

pub struct FooOp(u64);

trait AbstractKernel {
    fn wrap(&self) -> Kernel;
}

enum BinaryKernel<X0, X1, Y> {
    Static(fn(X0, X1) -> Y),
    Parameterized(Rc<dyn Fn(X0, X1) -> Y>),
}

impl<X0, X1, Y> AbstractKernel for BinaryKernel<X0, X1, Y>
where
    X0: 'static + From<Value>,
    X1: 'static + From<Value>,
    Y: 'static,
    Value: From<Y>,
{
    fn wrap(&self) -> Kernel {
        match self {
            BinaryKernel::Static(k) => {
                let k = k.clone();
                Kernel::Binary(Box::new(move |x0, x1| {
                    let x0 = X0::from(x0);
                    let x1 = X1::from(x1);
                    let y = k(x0, x1);
                    Value::from(y)
                }))
            }
            BinaryKernel::Parameterized(k) => {
                let k = k.clone();
                Kernel::Binary(Box::new(move |x0, x1| {
                    let x0 = X0::from(x0);
                    let x1 = X1::from(x1);
                    let y = k(x0, x1);
                    Value::from(y)
                }))
            }
        }
    }
}

impl FooOp {
    fn kernel(&self) -> Box<dyn AbstractKernel> {
        if self.0 == 0 {
            Box::new(BinaryKernel::Static::<
                Ring64Tensor,
                Ring64Tensor,
                Ring64Tensor,
            >(|x, y| {x + y}))
        } else {
            Box::new(BinaryKernel::Parameterized::<
                Ring128Tensor,
                Ring128Tensor,
                Ring128Tensor,
            >(Rc::new(|x, y| x + y)))
        }
    }
}

fn fubar() {
    let op = FooOp(6);
    let k = op.kernel();
    let c = k.wrap();
}

pub struct RingAddKernel<X0, X1, Y>(PhantomData<(X0, X1, Y)>);

impl<X0, X1, Y> RingAddKernel<X0, X1, Y> {
    pub fn new() -> Self {
        RingAddKernel(PhantomData)
    }
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

impl Compile for RingAddOp {
    fn compile(&self) -> Kernel {
        Kernel::Binary(Box::new(move |x, y| {
            let x: Ring64Tensor = x.into();
            let y: Ring64Tensor = y.into();
            let z = x + y;
            z.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingSubOp;

impl Compile for RingSubOp {
    fn compile(&self) -> Kernel {
        Kernel::Binary(Box::new(move |x, y| {
            let x: Ring64Tensor = x.into();
            let y: Ring64Tensor = y.into();
            let z = x - y;
            z.into()
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingMulOp;

impl Compile for RingMulOp {
    fn compile(&self) -> Kernel {
        Kernel::Binary(Box::new(move |x, y| {
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
    fn compile(&self) -> Kernel {
        Kernel::Binary(Box::new(move |x, y| {
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
    fn compile(&self) -> Kernel {
        let axis = self.axis;
        Kernel::Unary(Box::new(move |x| {
            let x: Ring64Tensor = x.into();
            let y = x.sum(axis);
            Value::from(y)
        }))
    }
}

#[derive(Serialize, Deserialize, Debug, Hash)]
pub struct RingShapeOp;

impl Compile for RingShapeOp {
    fn compile(&self) -> Kernel {
        Kernel::Unary(Box::new(move |x| match x {
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
    fn compile(&self) -> Kernel {
        let value = self.value;
        Kernel::Unary(Box::new(move |shape| match shape {
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
    fn compile(&self) -> Kernel {
        match self.max_value {
            None => Kernel::Binary(Box::new(|shape, seed| match (shape, seed) {
                (Value::Shape(shape), Value::Seed(seed)) => {
                    Value::Ring64Tensor(Ring64Tensor::sample_uniform(&shape, &seed))
                }
                _ => unimplemented!(),
            })),
            Some(max_value) if max_value == 1 => {
                Kernel::Binary(Box::new(|shape, seed| match (shape, seed) {
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
    fn compile(&self) -> Kernel {
        let amount = self.amount;
        Kernel::Unary(Box::new(move |x| {
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
    fn compile(&self) -> Kernel {
        let amount = self.amount;
        Kernel::Unary(Box::new(move |x| {
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

pub struct CompiledOperation {
    operation_name: String,
    kernel: Box<dyn Fn(&Environment<Value>) -> Value>,
}

impl Operation {
    pub fn compile(&self) -> CompiledOperation {
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
                        let x0 = env.get(&x0_name).unwrap();
                        k(x0.clone()) // TODO(Morten) avoid clone
                    }),
                }
            }
            (Kernel::Binary(k), 2) => {
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                CompiledOperation {
                    operation_name: self.name.clone(),
                    kernel: Box::new(move |env: &Environment<Value>| {
                        let x0 = env.get(&x0_name).unwrap();
                        let x1 = env.get(&x1_name).unwrap();
                        k(x0.clone(), x1.clone()) // TODO(Morten) avoid clone
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
                        let x0 = env.get(&x0_name).unwrap();
                        let x1 = env.get(&x1_name).unwrap();
                        let x2 = env.get(&x2_name).unwrap();
                        k(x0.clone(), x1.clone(), x2.clone()) // TODO(Morten) avoid clone
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
                            .map(|value| value.clone()) // TODO(Morten) avoid clone
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
