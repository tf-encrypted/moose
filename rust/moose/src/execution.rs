#![allow(unused_macros)]

use crate::computation::*;
use crate::error::{Error, Result};
use crate::networking::{
    AsyncNetworking, LocalAsyncNetworking, LocalSyncNetworking, SyncNetworking,
};
use crate::storage::{AsyncStorage, LocalAsyncStorage, LocalSyncStorage, SyncStorage};

use futures::future::{Map, Shared};
use futures::prelude::*;
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;
use tokio::sync::oneshot;

#[macro_export]
macro_rules! function_kernel {
    ($f:expr) => {
        Ok(Kernel::NullaryFunction(|| {
            let y = $f();
            Ok(Value::from(y))
        }))
    };
    ($t0:ty, $f:expr) => {
        Ok(Kernel::UnaryFunction(|x0| {
            use std::convert::TryFrom;
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let g: fn($t0) -> _ = $f;
            let y = g(x0);
            Ok(Value::from(y))
        }))
    };
    ($t0:ty, $t1:ty, $f:expr) => {
        Ok(Kernel::BinaryFunction(|x0, x1| {
            use std::convert::TryFrom;
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let x1 = <$t1 as TryFrom<Value>>::try_from(x1)?;
            let g: fn($t0, $t1) -> _ = $f;
            let y = g(x0, x1);
            Ok(Value::from(y))
        }))
    };
    ($t0:ty, $t1:ty, $t2:ty, $f:expr) => {
        Ok(Kernel::TernaryFunction(|x0, x1, x2| {
            use std::convert::TryFrom;
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let x1 = <$t1 as TryFrom<Value>>::try_from(x1)?;
            let x2 = <$t2 as TryFrom<Value>>::try_from(x2)?;
            let g: fn($t0, $t1, $t2) -> _ = $f;
            let y = g(x0, x1, x2);
            Ok(Value::from(y))
        }))
    };
    (vec[$t:ty], $f:expr) => {
        Ok(Kernel::VariadicFunction(|xs| {
            use std::convert::TryFrom;
            let xs = xs
                .into_iter()
                .map(|xi| <$t as TryFrom<Value>>::try_from(xi))
                .collect::<Result<Vec<_>>>()?;
            let g: fn(Vec<$t>) -> _ = $f;
            let y = g(xs);
            Ok(Value::from(y))
        }))
    };
}

#[macro_export]
macro_rules! closure_kernel {
    ($f:expr) => {
        Ok(Kernel::NullaryClosure(Arc::new(move || {
            let y = $f();
            Ok(Value::from(y))
        })))
    };
    ($t0:ty, $f:expr) => {{
        use std::convert::TryFrom;
        use std::sync::Arc;

        #[inline(always)]
        fn g<F: Fn($t0) -> Y, Y>(f: F) -> F {
            f
        }

        Ok(Kernel::UnaryClosure(Arc::new(move |x0| {
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let y = g($f)(x0);
            Ok(Value::from(y))
        })))
    }};
    ($t0:ty, $t1:ty, $f:expr) => {{
        use std::convert::TryFrom;
        use std::sync::Arc;

        #[inline(always)]
        fn g<F: Fn($t0, $t1) -> Y, Y>(f: F) -> F {
            f
        }

        Ok(Kernel::BinaryClosure(Arc::new(move |x0, x1| {
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let x1 = <$t1 as TryFrom<Value>>::try_from(x1)?;
            let y = g($f)(x0, x1);
            Ok(Value::from(y))
        })))
    }};
    ($t0:ty, $t1:ty, $t2:ty, $f:expr) => {{
        use std::convert::TryFrom;
        use std::sync::Arc;

        #[inline(always)]
        fn g<F: Fn($t0, $t1, $t2) -> Y, Y>(f: F) -> F {
            f
        }

        Ok(Kernel::TernaryClosure(Arc::new(move |x0, x1, x2| {
            let x0 = <$t0 as TryFrom<Value>>::try_from(x0)?;
            let x1 = <$t1 as TryFrom<Value>>::try_from(x1)?;
            let x2 = <$t2 as TryFrom<Value>>::try_from(x2)?;
            let y = g($f)(x0, x1, x2);
            Ok(Value::from(y))
        })))
    }};
    (vec[$t:ty], $f:expr) => {{
        use std::convert::TryFrom;
        use std::sync::Arc;

        Ok(Kernel::VariadicClosure(Arc::new(move |xs| {
            let xs = xs
                .into_iter()
                .map(|xi| <$t as TryFrom<Value>>::try_from(xi))
                .collect::<Result<Vec<_>>>()?;
            let y = $f(xs);
            Ok(Value::from(y))
        })))
    }};
}

pub enum Kernel {
    NullaryClosure(Arc<dyn Fn() -> Result<Value> + Send + Sync>),
    UnaryClosure(Arc<dyn Fn(Value) -> Result<Value> + Send + Sync>),
    BinaryClosure(Arc<dyn Fn(Value, Value) -> Result<Value> + Send + Sync>),
    TernaryClosure(Arc<dyn Fn(Value, Value, Value) -> Result<Value> + Send + Sync>),
    VariadicClosure(Arc<dyn Fn(Vec<Value>) -> Result<Value> + Send + Sync>),

    NullaryFunction(fn() -> Result<Value>),
    UnaryFunction(fn(Value) -> Result<Value>),
    BinaryFunction(fn(Value, Value) -> Result<Value>),
    TernaryFunction(fn(Value, Value, Value) -> Result<Value>),
    VariadicFunction(fn(Vec<Value>) -> Result<Value>),
}

type NullarySyncKernel = Box<dyn Fn(&SyncSession) -> Result<Value> + Send + Sync>;

type UnarySyncKernel = Box<dyn Fn(&SyncSession, Value) -> Result<Value> + Send + Sync>;

type BinarySyncKernel = Box<dyn Fn(&SyncSession, Value, Value) -> Result<Value> + Send + Sync>;

type TernarySyncKernel =
    Box<dyn Fn(&SyncSession, Value, Value, Value) -> Result<Value> + Send + Sync>;

type VariadicSyncKernel = Box<dyn Fn(&SyncSession, Vec<Value>) -> Result<Value> + Send + Sync>;

pub enum SyncKernel {
    Nullary(NullarySyncKernel),
    Unary(UnarySyncKernel),
    Binary(BinarySyncKernel),
    Ternary(TernarySyncKernel),
    Variadic(VariadicSyncKernel),
}

type NullaryAsyncKernel = Box<dyn Fn(&Arc<AsyncSession>, AsyncSender) -> AsyncTask + Send + Sync>;

type UnaryAsyncKernel =
    Box<dyn Fn(&Arc<AsyncSession>, AsyncReceiver, AsyncSender) -> AsyncTask + Send + Sync>;

type BinaryAsyncKernel = Box<
    dyn Fn(&Arc<AsyncSession>, AsyncReceiver, AsyncReceiver, AsyncSender) -> AsyncTask
        + Send
        + Sync,
>;

type TernaryAsyncKernel = Box<
    dyn Fn(
            &Arc<AsyncSession>,
            AsyncReceiver,
            AsyncReceiver,
            AsyncReceiver,
            AsyncSender,
        ) -> AsyncTask
        + Send
        + Sync,
>;

type VariadicAsyncKernel =
    Box<dyn Fn(&Arc<AsyncSession>, Vec<AsyncReceiver>, AsyncSender) -> AsyncTask + Send + Sync>;

pub enum AsyncKernel {
    Nullary(NullaryAsyncKernel),
    Unary(UnaryAsyncKernel),
    Binary(BinaryAsyncKernel),
    Ternary(TernaryAsyncKernel),
    Variadic(VariadicAsyncKernel),
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

impl<O: Compile<Kernel>> Compile<SyncKernel> for O {
    fn compile(&self) -> Result<SyncKernel> {
        let kernel: Kernel = self.compile()?;
        match kernel {
            Kernel::NullaryClosure(k) => Ok(SyncKernel::Nullary(Box::new(move |_sess| k()))),
            Kernel::UnaryClosure(k) => Ok(SyncKernel::Unary(Box::new(move |_sess, x0| k(x0)))),
            Kernel::BinaryClosure(k) => {
                Ok(SyncKernel::Binary(Box::new(move |_sess, x0, x1| k(x0, x1))))
            }
            Kernel::TernaryClosure(k) => {
                Ok(SyncKernel::Ternary(Box::new(move |_sess, x0, x1, x2| {
                    k(x0, x1, x2)
                })))
            }
            Kernel::VariadicClosure(k) => {
                Ok(SyncKernel::Variadic(Box::new(move |_sess, xs| k(xs))))
            }
            Kernel::NullaryFunction(k) => Ok(SyncKernel::Nullary(Box::new(move |_sess| k()))),
            Kernel::UnaryFunction(k) => Ok(SyncKernel::Unary(Box::new(move |_sess, x0| k(x0)))),
            Kernel::BinaryFunction(k) => {
                Ok(SyncKernel::Binary(Box::new(move |_sess, x0, x1| k(x0, x1))))
            }
            Kernel::TernaryFunction(k) => {
                Ok(SyncKernel::Ternary(Box::new(move |_sess, x0, x1, x2| {
                    k(x0, x1, x2)
                })))
            }
            Kernel::VariadicFunction(k) => {
                Ok(SyncKernel::Variadic(Box::new(move |_sess, xs| k(xs))))
            }
        }
    }
}

pub fn map_send_error<T>(_: T) -> Error {
    tracing::debug!("Failed to send result on channel, receiver was dropped");
    Error::Unexpected
}

pub fn map_receive_error<T>(_: T) -> Error {
    Error::InputUnavailable
}

impl<O: Compile<Kernel>> Compile<AsyncKernel> for O {
    fn compile(&self) -> Result<AsyncKernel> {
        let kernel: Kernel = self.compile()?;
        match kernel {
            Kernel::NullaryClosure(k) => {
                Ok(AsyncKernel::Nullary(Box::new(move |_sess, sender| {
                    let k = Arc::clone(&k);
                    tokio::spawn(async move {
                        let y: Value = k()?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }
            Kernel::UnaryClosure(k) => {
                Ok(AsyncKernel::Unary(Box::new(move |_sess, x0, sender| {
                    let k = Arc::clone(&k);
                    tokio::spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let y: Value = k(x0)?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }
            Kernel::BinaryClosure(k) => Ok(AsyncKernel::Binary(Box::new(
                move |_sess, x0, x1, sender| {
                    let k = Arc::clone(&k);
                    tokio::spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let x1: Value = x1.await.map_err(map_receive_error)?;
                        let y: Value = k(x0, x1)?;
                        sender.send(y).map_err(map_send_error)
                    })
                },
            ))),
            Kernel::TernaryClosure(k) => Ok(AsyncKernel::Ternary(Box::new(
                move |_sess, x0, x1, x2, sender| {
                    let k = Arc::clone(&k);
                    tokio::spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let x1: Value = x1.await.map_err(map_receive_error)?;
                        let x2: Value = x2.await.map_err(map_receive_error)?;
                        let y: Value = k(x0, x1, x2)?;
                        sender.send(y).map_err(map_send_error)
                    })
                },
            ))),
            Kernel::VariadicClosure(k) => {
                Ok(AsyncKernel::Variadic(Box::new(move |_sess, xs, sender| {
                    let k = Arc::clone(&k);
                    tokio::spawn(async move {
                        use futures::future::try_join_all;
                        let xs: Vec<Value> = try_join_all(xs).await.map_err(map_receive_error)?;
                        let y: Value = k(xs)?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }

            Kernel::NullaryFunction(k) => {
                Ok(AsyncKernel::Nullary(Box::new(move |_sess, sender| {
                    tokio::spawn(async move {
                        let y = k()?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }
            Kernel::UnaryFunction(k) => {
                Ok(AsyncKernel::Unary(Box::new(move |_sess, x0, sender| {
                    tokio::spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let y: Value = k(x0)?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }
            Kernel::BinaryFunction(k) => Ok(AsyncKernel::Binary(Box::new(
                move |_sess, x0, x1, sender| {
                    tokio::spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let x1: Value = x1.await.map_err(map_receive_error)?;
                        let y: Value = k(x0, x1)?;
                        sender.send(y).map_err(map_send_error)
                    })
                },
            ))),
            Kernel::TernaryFunction(k) => Ok(AsyncKernel::Ternary(Box::new(
                move |_sess, x0, x1, x2, sender| {
                    tokio::spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let x1: Value = x1.await.map_err(map_receive_error)?;
                        let x2: Value = x2.await.map_err(map_receive_error)?;
                        let y: Value = k(x0, x1, x2)?;
                        sender.send(y).map_err(map_send_error)
                    })
                },
            ))),
            Kernel::VariadicFunction(k) => {
                Ok(AsyncKernel::Variadic(Box::new(move |_sess, xs, sender| {
                    tokio::spawn(async move {
                        use futures::future::try_join_all;
                        let xs: Vec<Value> = try_join_all(xs).await.map_err(map_receive_error)?;
                        let y: Value = k(xs)?;
                        sender.send(y).map_err(map_send_error)
                    })
                })))
            }
        }
    }
}

pub struct SyncSession {
    pub sid: SessionId,
    pub args: Environment<Value>,
    pub networking: SyncNetworkingImpl,
    pub storage: SyncStorageImpl,
}

pub type SyncNetworkingImpl = Rc<dyn SyncNetworking>;
pub type SyncStorageImpl = Rc<dyn SyncStorage>;

type SyncOperationKernel =
    Box<dyn Fn(&SyncSession, &Environment<Value>) -> Result<Value> + Send + Sync>;

pub struct CompiledSyncOperation {
    name: String,
    kernel: SyncOperationKernel,
}

impl CompiledSyncOperation {
    pub fn apply(&self, sess: &SyncSession, env: &Environment<Value>) -> Result<Value> {
        (self.kernel)(sess, env)
    }
}

type AsyncOperationKernel = Box<
    dyn Fn(&Arc<AsyncSession>, &Environment<AsyncReceiver>, AsyncSender) -> Result<AsyncTask>
        + Send
        + Sync,
>;

pub struct CompiledAsyncOperation {
    name: String,
    kernel: AsyncOperationKernel,
}

impl CompiledAsyncOperation {
    pub fn apply(
        &self,
        sess: &Arc<AsyncSession>,
        env: &Environment<AsyncReceiver>,
        sender: AsyncSender,
    ) -> Result<AsyncTask> {
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
        let operator_kernel = Compile::<SyncKernel>::compile(&self.kind)?;
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
                        let x0 = env
                            .get(&x0_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
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
                        let x0 = env
                            .get(&x0_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        let x1 = env
                            .get(&x1_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
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
                            .map(|input| env.get(input).cloned().ok_or(Error::MalformedEnvironment)) // TODO(Morten avoid cloning
                            .collect::<Result<Vec<_>>>()?;
                        k(sess, xs)
                    }),
                })
            }
        }
    }
}

impl Compile<CompiledAsyncOperation> for Operation {
    fn compile(&self) -> Result<CompiledAsyncOperation> {
        let operator_kernel = Compile::<AsyncKernel>::compile(&self.kind)?;
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
                        let x0 = env
                            .get(&x0_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
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
                        let x0 = env
                            .get(&x0_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
                        let x1 = env
                            .get(&x1_name)
                            .ok_or(Error::MalformedEnvironment)?
                            .clone();
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
                        Ok(k(sess, xs, sender))
                    }),
                })
            }
        }
    }
}

impl Operation {
    pub fn apply(&self, sess: &SyncSession, env: &Environment<Value>) -> Result<Value> {
        let compiled: CompiledSyncOperation = self.compile()?;
        compiled.apply(sess, env)
    }
}

impl Computation {
    pub fn toposort(&self) -> Result<Computation> {
        let mut graph = Graph::<String, ()>::new();

        let mut vertex_map: HashMap<&str, NodeIndex> = HashMap::new();
        let mut inv_map: HashMap<NodeIndex, usize> = HashMap::new();

        let mut send_nodes: HashMap<&str, NodeIndex> = HashMap::new();
        let mut recv_nodes: HashMap<&str, NodeIndex> = HashMap::new();

        let mut rdv_keys: HashSet<&str> = HashSet::new();

        for (i, op) in self.operations.iter().enumerate() {
            let vertex = graph.add_node(op.name.clone());
            match op.kind {
                Operator::Send(ref op) => {
                    let key = op.rendezvous_key.as_ref();

                    if send_nodes.contains_key(key) {
                        Error::MalformedComputation(format!(
                            "Already had a send node with same rdv key at key {}",
                            key
                        ));
                    }

                    send_nodes.insert(key, vertex);
                    rdv_keys.insert(key);
                }
                Operator::Receive(ref op) => {
                    let key = op.rendezvous_key.as_ref();

                    if recv_nodes.contains_key(key) {
                        Error::MalformedComputation(format!(
                            "Already had a recv node with same rdv key at key {}",
                            key
                        ));
                    }

                    recv_nodes.insert(key, vertex);
                    rdv_keys.insert(key);
                }
                _ => {}
            }
            vertex_map.insert(&op.name, vertex);
            inv_map.insert(vertex, i);
        }

        for op in self.operations.iter() {
            for ins in op.inputs.iter() {
                graph.add_edge(vertex_map[&ins.as_ref()], vertex_map[&op.name.as_ref()], ());
            }
        }

        for key in rdv_keys.into_iter() {
            if !send_nodes.contains_key(key) {
                Error::MalformedComputation(format!("No send node with rdv key {}", key));
            }
            if !recv_nodes.contains_key(key) {
                Error::MalformedComputation(format!("No recv node with rdv key {}", key));
            }
            // add edge send->recv (send must be evaluated before recv)
            graph.add_edge(send_nodes[key], recv_nodes[key], ());
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

    pub fn apply(&self, sess: &SyncSession) -> Result<Environment<Value>> {
        let mut env = Environment::<Value>::with_capacity(self.operations.len());
        for op in self.operations.iter() {
            let value = op.apply(sess, &env)?;
            env.insert(op.name.clone(), value);
        }
        Ok(env)
    }
}

type SyncComputationKernel = Box<dyn Fn(&SyncSession) -> Result<Environment<Value>>>;

pub struct CompiledSyncComputation(SyncComputationKernel);

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

        let output_names: Vec<String> = self
            .operations
            .iter() // guessing that par_iter won't help here
            .filter_map(|op| match op.kind {
                Operator::Output(_) => Some(op.name.clone()),
                _ => None,
            })
            .collect();

        Ok(CompiledSyncComputation(Box::new(
            move |sess: &SyncSession| {
                let mut env = Environment::with_capacity(compiled_ops.len());
                for compiled_op in compiled_ops.iter() {
                    let value = compiled_op.apply(sess, &env)?;
                    env.insert(compiled_op.name.clone(), value);
                }

                let outputs: HashMap<String, Value> = output_names
                    .iter()
                    .map(|op_name| (op_name.clone(), env.get(op_name).cloned().unwrap()))
                    .collect();
                Ok(outputs)
            },
        )))
    }
}

pub type SyncArgs = HashMap<String, Value>;

impl CompiledSyncComputation {
    pub fn apply(&self, sess: &SyncSession) -> Result<Environment<Value>> {
        (self.0)(sess)
    }
}

pub type AsyncNetworkingImpl = Arc<dyn AsyncNetworking + Send + Sync>;

type AsyncComputationKernel =
    Box<dyn Fn(&Arc<AsyncSession>) -> Result<(AsyncSessionJoinHandle, Environment<AsyncReceiver>)>>;

pub struct CompiledAsyncComputation(AsyncComputationKernel);

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

        let output_names: Vec<String> = self
            .operations
            .iter() // guessing that par_iter won't help here
            .filter_map(|op| match op.kind {
                Operator::Output(_) => Some(op.name.clone()),
                _ => None,
            })
            .collect();

        fn remove_err<T, E>(r: std::result::Result<T, E>) -> std::result::Result<T, ()> {
            r.map_err(|_| ())
        }

        Ok(CompiledAsyncComputation(Box::new(
            move |sess: &Arc<AsyncSession>| {
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
                    .map(|(sender, op)| op.apply(sess, &receivers, sender))
                    .collect::<Result<Vec<_>>>()?;

                let join_handle = AsyncSessionJoinHandle { tasks };
                let outputs: HashMap<String, AsyncReceiver> = output_names
                    .iter()
                    .map(|op_name| {
                        let val = receivers.get(op_name).cloned().unwrap(); // safe to unwrap per construction
                        (op_name.clone(), val)
                    })
                    .collect();

                Ok((join_handle, outputs))
            },
        )))
    }
}

impl CompiledAsyncComputation {
    pub fn apply(
        &self,
        sess: &Arc<AsyncSession>,
    ) -> Result<(AsyncSessionJoinHandle, Environment<AsyncReceiver>)> {
        (self.0)(sess)
    }
}

pub type Environment<V> = HashMap<String, V>;

/// In-order single-threaded executor.
///
/// This executor evaluates the operations of computations in-order, raising an error
/// in case data dependencies are not respected. This executor is intended for debug
/// and development only due to its unforgiving but highly predictable behaviour.
pub struct EagerExecutor {
    networking: Rc<dyn SyncNetworking>,
    pub storage: Rc<dyn SyncStorage>,
}

impl EagerExecutor {
    pub fn new() -> EagerExecutor {
        let networking = Rc::new(LocalSyncNetworking::default());
        let storage = Rc::new(LocalSyncStorage::default());
        EagerExecutor {
            networking,
            storage,
        }
    }

    pub fn run_computation(
        &self,
        comp: &Computation,
        sid: SessionId,
        args: Environment<Value>,
    ) -> Result<Environment<Value>> {
        let compiled_comp: CompiledSyncComputation = comp.compile()?;

        let sess = SyncSession {
            sid,
            args,
            networking: Rc::clone(&self.networking),
            storage: Rc::clone(&self.storage),
        };

        compiled_comp.apply(&sess)
    }
}

impl Default for EagerExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// A session is essentially the activation frame of the graph function call.
pub struct AsyncSession {
    pub sid: SessionId,
    pub args: AsyncArgs,
    pub networking: Arc<dyn Send + Sync + AsyncNetworking>,
    pub storage: Arc<dyn Send + Sync + AsyncStorage>,
}

pub type AsyncArgs = Environment<AsyncReceiver>;

pub struct AsyncSessionJoinHandle {
    tasks: Vec<AsyncTask>,
}

impl AsyncSessionJoinHandle {
    pub fn join(self) -> Result<()> {
        let runtime_handle = tokio::runtime::Handle::current();
        for task in self.tasks {
            let res = runtime_handle.block_on(task);
            if res.is_err() {
                unimplemented!() // TODO
            }
        }
        Ok(())
    }
}

pub struct AsyncExecutor {
    pub networking: Arc<dyn Send + Sync + AsyncNetworking>,
    pub storage: Arc<dyn Send + Sync + AsyncStorage>,
}

impl AsyncExecutor {
    pub fn new() -> AsyncExecutor {
        let networking = Arc::new(LocalAsyncNetworking::default());
        let storage = Arc::new(LocalAsyncStorage::default());
        AsyncExecutor {
            networking,
            storage,
        }
    }

    pub fn run_computation(
        &self,
        comp: &Computation,
        sid: SessionId,
        args: AsyncArgs,
    ) -> Result<HashMap<String, AsyncReceiver>> {
        let compiled_comp: CompiledAsyncComputation = comp.compile()?;

        let sess = Arc::new(AsyncSession {
            sid,
            args,
            networking: Arc::clone(&self.networking),
            storage: Arc::clone(&self.storage),
        });

        // TODO don't return unexpected error
        let (join_handle, outputs): (AsyncSessionJoinHandle, HashMap<_, AsyncReceiver>) =
            compiled_comp.apply(&sess).map_err(|_| Error::Unexpected)?;

        join_handle.join()?;
        Ok(outputs)
    }
}

impl Default for AsyncExecutor {
    fn default() -> Self {
        AsyncExecutor::new()
    }
}

#[test]
fn test_standard_prod_ops() {
    use crate::standard::Float32Tensor;
    use maplit::hashmap;
    use ndarray::prelude::*;

    let x = Float32Tensor::from(
        array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap(),
    );
    let x_op = Operation {
        name: "x".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Float32Tensor(x),
        }),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };
    let y = Float32Tensor::from(
        array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap(),
    );
    let y_op = Operation {
        name: "y".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Float32Tensor(y),
        }),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };
    let mul_op = Operation {
        name: "mul".into(),
        kind: Operator::StdMul(StdMulOp {
            lhs: Ty::Float32TensorTy,
            rhs: Ty::Float32TensorTy,
        }),
        inputs: vec!["x".into(), "y".into()],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };
    let dot_op = Operation {
        name: "dot".into(),
        kind: Operator::StdDot(StdDotOp {
            lhs: Ty::Float32TensorTy,
            rhs: Ty::Float32TensorTy,
        }),
        inputs: vec!["x".into(), "y".into()],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };
    let mean_op = Operation {
        name: "mean".into(),
        kind: Operator::StdMean(StdMeanOp {
            ty: Ty::Float32TensorTy,
            axis: Some(0),
        }),
        inputs: vec!["dot".into()],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };
    let operations = vec![x_op, y_op, mul_op, dot_op, mean_op];
    let comp = Computation { operations }.toposort().unwrap();

    let exec = EagerExecutor::new();
    let sid = 12345;
    let args = hashmap![];
    exec.run_computation(&comp, sid, args).ok();
}

#[test]
fn test_eager_executor() {
    use crate::prim::Nonce;
    use crate::standard::Shape;
    use maplit::hashmap;

    let key_op = Operation {
        name: "key".into(),
        kind: Operator::PrimGenPrfKey(PrimGenPrfKeyOp),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };

    let seed_op = Operation {
        name: "seed".into(),
        kind: Operator::PrimDeriveSeed(PrimDeriveSeedOp {
            nonce: Nonce(vec![1, 2, 3]),
        }),
        inputs: vec!["key".into()],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };

    let shape_op = Operation {
        name: "shape".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Shape(Shape(vec![2, 3])),
        }),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };

    let output_op = Operation {
        name: "z".into(),
        kind: Operator::Output(OutputOp {
            ty: Ty::Ring64TensorTy,
        }),
        inputs: vec!["x10".into()],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };

    let sample_ops: Vec<_> = (0..100)
        .map(|i| Operation {
            name: format!("x{}", i),
            kind: Operator::RingSample(RingSampleOp {
                output: Ty::Ring64TensorTy,
                max_value: None,
            }),
            inputs: vec!["shape".into(), "seed".into()],
            placement: Placement::Host(HostPlacement {
                name: "alice".into(),
            }),
        })
        .collect();

    let mut operations = sample_ops;
    operations.extend(vec![key_op, seed_op, shape_op, output_op]);

    let comp = Computation { operations }.toposort().unwrap();

    let exec = EagerExecutor::new();
    let sid = 12345;
    let args = hashmap![];
    let outputs = exec.run_computation(&comp, sid, args).unwrap();
    assert_eq!(outputs.keys().collect::<Vec<_>>(), vec!["z"]);
}
