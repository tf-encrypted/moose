use serde::{Serialize, Deserialize};
use rmp_serde::{Serializer, Deserializer};
use enum_dispatch::enum_dispatch;
use std::collections::HashMap;

#[enum_dispatch]
trait Compile {
    fn compile(&self) -> CompiledKernel<Value>;
}

#[enum_dispatch(Compile)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Operator {
    RingSample(RingSampleOp),
    RingAdd(RingAddOp),
    RingSub(RingSubOp),
    // RingSum,
    // RingMul,
    // RingDot,
    // RingShl,
    // RingShr,
    // RingFill,
    RingShape(RingShapeOp),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSampleOp;

// impl From<RingSampleOp> for Operator {
//     fn from(op: RingSampleOp) -> Operator {
//         Operator::RingSample(op)
//     }
// }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingAddOp;

// impl From<RingAddOp> for Operator {
//     fn from(op: RingAddOp) -> Operator {
//         Operator::RingAdd(op)
//     }
// }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSubOp;

// impl From<RingSubOp> for Operator {
//     fn from(op: RingSubOp) -> Operator {
//         Operator::RingSub(op)
//     }
// }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShapeOp;

pub struct Operation {
    name: String,
    operator: Operator,
    inputs: Vec<String>,
    placement: String,
}

pub struct Computation {
    operations: Vec<Operation>,
}

fn foo() {
    let comp = Computation {
        operations: vec![
            Operation {
                name: "z".into(),
                operator: Operator::RingAdd(RingAddOp),
                inputs: vec!["x".into(), "y".into()],
                placement: "alice".into(),
            }
        ]
    };

}

#[derive(Clone)]
enum Value {
    Int(u64),
    Float(f64),
}

type Env = HashMap<String, Value>;

fn ring_add(x: u64, y: u64) -> u64 {
    x + y
}

fn ring_sub(x: u64, y: u64) -> u64 {
    x - y
}

pub enum CompiledKernel<V> {
    Nullary(Box<dyn Fn() -> V>),
    Unary(Box<dyn Fn(V) -> V>),
    Binary(Box<dyn Fn(V, V) -> V>),
    Ternary(Box<dyn Fn(V, V, V) -> V>),
    Veriadic(Box<dyn Fn(&[V]) -> V>),
}

impl Compile for RingSampleOp {
    fn compile(&self) -> CompiledKernel<Value> {
        CompiledKernel::Nullary(Box::new(move || {
            Value::Int(5)
        }))
    }
}

impl Compile for RingAddOp {
    fn compile(&self) -> CompiledKernel<Value> {
        CompiledKernel::Binary(Box::new(move |x, y| {
            match (x, y) {
                (Value::Int(x), Value::Int(y)) => Value::Int(ring_add(x, y)),
                _ => unimplemented!(),
            }
        }))
    }
}

impl Compile for RingSubOp {
    fn compile(&self) -> CompiledKernel<Value> {
        CompiledKernel::Binary(Box::new(move |x, y| {
            match (x, y) {
                (Value::Int(x), Value::Int(y)) => Value::Int(ring_sub(x, y)),
                _ => unimplemented!(),
            }
        }))
    }
}

impl Compile for RingShapeOp {
    fn compile(&self) -> CompiledKernel<Value> {
        CompiledKernel::Unary(Box::new(move |x| {
            match x {
                _ => unimplemented!(),
            }
        }))
    }
}

impl Operation {
    fn compile(&self) -> Box<dyn Fn(Env) -> Value> {
        use CompiledKernel::*;
        let kernel: CompiledKernel<Value> = self.operator.compile();
        match kernel {
            Nullary(k) => {
                Box::new(move |_| { k() })
            },
            Unary(k) => {
                assert_eq!(self.inputs.len(), 1);
                let x_name = self.inputs[0].clone();
                Box::new(move |env| {
                    let x = env.get(&x_name).unwrap();
                    k(x.clone())  // TODO avoid clone
                })
            },
            Binary(k) => {
                assert_eq!(self.inputs.len(), 2);
                let x_name = self.inputs[0].clone();
                let y_name = self.inputs[1].clone();
                Box::new(move |env| {
                    let x = env.get(&x_name).unwrap();
                    let y = env.get(&y_name).unwrap();
                    k(x.clone(), y.clone())  // TODO avoid clone
                })
            },
            Ternary(k) => {
                assert_eq!(self.inputs.len(), 3);
                let x_name = self.inputs[0].clone();
                let y_name = self.inputs[1].clone();
                let z_name = self.inputs[2].clone();
                Box::new(move |env| {
                    let x = env.get(&x_name).unwrap();
                    let y = env.get(&y_name).unwrap();
                    let z = env.get(&z_name).unwrap();
                    k(x.clone(), y.clone(), z.clone())  // TODO avoid clone
                })
            },
            Veriadic(k) => {
                let input_names = self.inputs.clone();
                Box::new(move |env| {
                    let inputs: Vec<_> = input_names.iter()
                        .map(|input_name| env.get(input_name).unwrap())
                        .map(|input| input.clone())  // TODO avoid clone
                        .collect();
                    k(&inputs)  
                })
            },
        }
    }
}

// type OperatorGraph = petgraph::Graph<&String, ()>;

fn compile(comp: &Computation) {
    let mut graph: petgraph::Graph<&String, ()> = petgraph::Graph::new();

    let nodes_map: HashMap<_, _> = comp.operations.iter()
        .map(|op| (&op.name, graph.add_node(&op.name)))
        .collect();

    for op in comp.operations.iter() {
        for input_op_name in op.inputs.iter() {
            graph.add_edge(
                *nodes_map.get(input_op_name).unwrap(),
                *nodes_map.get(&op.name).unwrap(), ()
            );
        }
    }

    
    // for op in comp.operations.iter() {
    //     graph.add_node(op.operator.clone());
    // }



}