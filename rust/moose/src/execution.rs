#![allow(unused_macros)]

use crate::computation::*;
use crate::error::{Error, Result};
use crate::networking::{AsyncNetworking, LocalSyncNetworking, SyncNetworking};
use crate::storage::{AsyncStorage, LocalSyncStorage, SyncStorage};

use derive_more::Display;
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

        #[inline(always)]
        fn g<F: Fn(Vec<$t>) -> Y, Y>(f: F) -> F {
            f
        }

        Ok(Kernel::VariadicClosure(Arc::new(move |xs| {
            let xs = xs
                .into_iter()
                .map(<$t as TryFrom<Value>>::try_from)
                .collect::<Result<Vec<_>>>()?;
            let y = g($f)(xs);
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
    fn compile(&self, ctx: &CompilationContext) -> Result<C>;
}

impl<O: Compile<Kernel>> Compile<SyncKernel> for O {
    fn compile(&self, ctx: &CompilationContext) -> Result<SyncKernel> {
        let kernel: Kernel = self.compile(ctx)?;
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

pub fn map_send_result(res: std::result::Result<(), Value>) -> std::result::Result<(), Error> {
    match res {
        Ok(_) => Ok(()),
        Err(val) => {
            if val.ty() == Ty::UnitTy {
                // ignoring unit value is okay
                Ok(())
            } else {
                Err(Error::ResultUnused)
            }
        }
    }
}

pub fn map_receive_error<T>(_: T) -> Error {
    tracing::debug!("Failed to receive on channel, sender was dropped");
    Error::OperandUnavailable
}

impl<O: Compile<Kernel>> Compile<AsyncKernel> for O {
    fn compile(&self, ctx: &CompilationContext) -> Result<AsyncKernel> {
        let kernel: Kernel = self.compile(ctx)?;
        match kernel {
            Kernel::NullaryClosure(k) => {
                Ok(AsyncKernel::Nullary(Box::new(move |_sess, sender| {
                    let k = Arc::clone(&k);
                    tokio::spawn(async move {
                        let y: Value = k()?;
                        map_send_result(sender.send(y))
                    })
                })))
            }
            Kernel::UnaryClosure(k) => {
                Ok(AsyncKernel::Unary(Box::new(move |_sess, x0, sender| {
                    let k = Arc::clone(&k);
                    tokio::spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let y: Value = k(x0)?;
                        map_send_result(sender.send(y))
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
                        map_send_result(sender.send(y))
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
                        map_send_result(sender.send(y))
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
                        map_send_result(sender.send(y))
                    })
                })))
            }

            Kernel::NullaryFunction(k) => {
                Ok(AsyncKernel::Nullary(Box::new(move |_sess, sender| {
                    tokio::spawn(async move {
                        let y = k()?;
                        map_send_result(sender.send(y))
                    })
                })))
            }
            Kernel::UnaryFunction(k) => {
                Ok(AsyncKernel::Unary(Box::new(move |_sess, x0, sender| {
                    tokio::spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let y: Value = k(x0)?;
                        map_send_result(sender.send(y))
                    })
                })))
            }
            Kernel::BinaryFunction(k) => Ok(AsyncKernel::Binary(Box::new(
                move |_sess, x0, x1, sender| {
                    tokio::spawn(async move {
                        let x0: Value = x0.await.map_err(map_receive_error)?;
                        let x1: Value = x1.await.map_err(map_receive_error)?;
                        let y: Value = k(x0, x1)?;
                        map_send_result(sender.send(y))
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
                        map_send_result(sender.send(y))
                    })
                },
            ))),
            Kernel::VariadicFunction(k) => {
                Ok(AsyncKernel::Variadic(Box::new(move |_sess, xs, sender| {
                    tokio::spawn(async move {
                        use futures::future::try_join_all;
                        let xs: Vec<Value> = try_join_all(xs).await.map_err(map_receive_error)?;
                        let y: Value = k(xs)?;
                        map_send_result(sender.send(y))
                    })
                })))
            }
        }
    }
}

pub struct CompilationContext<'s> {
    pub role_assignment: &'s HashMap<Role, Identity>,
    pub own_identity: &'s Identity,
}

pub type SyncArgs = HashMap<String, Value>;

pub struct SyncSession {
    pub sid: SessionId,
    pub arguments: SyncArgs,
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

fn find_env<T: Clone>(env: &HashMap<String, T>, name: &str) -> Result<T> {
    // TODO(Morten) avoid cloning
    env.get(name)
        .cloned()
        .ok_or_else(|| Error::MalformedEnvironment(name.to_string()))
}

impl Compile<CompiledSyncOperation> for Operation {
    fn compile(&self, ctx: &CompilationContext) -> Result<CompiledSyncOperation> {
        let operator_kernel = Compile::<SyncKernel>::compile(&self.kind, ctx)?;
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
                        let x0 = find_env(env, &x0_name)?;
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
                        let x0 = find_env(env, &x0_name)?;
                        let x1 = find_env(env, &x1_name)?;
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
                        let x0 = find_env(env, &x0_name)?;
                        let x1 = find_env(env, &x1_name)?;
                        let x2 = find_env(env, &x2_name)?;
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
                            .map(|input| find_env(env, input)) // TODO(Morten avoid cloning
                            .collect::<Result<Vec<_>>>()?;
                        k(sess, xs)
                    }),
                })
            }
        }
    }
}

impl Compile<CompiledAsyncOperation> for Operation {
    fn compile(&self, ctx: &CompilationContext) -> Result<CompiledAsyncOperation> {
        let operator_kernel = Compile::<AsyncKernel>::compile(&self.kind, ctx)?;
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
                        let x0 = find_env(env, &x0_name)?;
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
                        let x0 = find_env(env, &x0_name)?;
                        let x1 = find_env(env, &x1_name)?;
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
                        let x0 = find_env(env, &x0_name)?;
                        let x1 = find_env(env, &x1_name)?;
                        let x2 = find_env(env, &x2_name)?;
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
                            .map(|input| find_env(env, input))
                            .collect::<Result<Vec<_>>>()?;
                        Ok(k(sess, xs, sender))
                    }),
                })
            }
        }
    }
}

impl Operation {
    pub fn apply(
        &self,
        ctx: &CompilationContext,
        sess: &SyncSession,
        env: &Environment<Value>,
    ) -> Result<Value> {
        let compiled: CompiledSyncOperation = self.compile(ctx)?;
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

    pub fn apply(
        &self,
        ctx: &CompilationContext,
        sess: &SyncSession,
    ) -> Result<Environment<Value>> {
        let mut env = Environment::<Value>::with_capacity(self.operations.len());
        for op in self.operations.iter() {
            let value = op.apply(ctx, sess, &env)?;
            env.insert(op.name.clone(), value);
        }
        Ok(env)
    }
}

type SyncComputationKernel = Box<dyn Fn(&SyncSession) -> Result<Environment<Value>>>;

pub struct CompiledSyncComputation(SyncComputationKernel);

impl CompiledSyncComputation {
    pub fn apply(&self, sess: &SyncSession) -> Result<Environment<Value>> {
        (self.0)(sess)
    }
}

pub type AsyncNetworkingImpl = Arc<dyn AsyncNetworking + Send + Sync>;

type AsyncComputationKernel =
    Box<dyn Fn(AsyncSession) -> Result<(AsyncSessionHandle, Environment<AsyncReceiver>)>>;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Display)]
pub struct Identity(pub String);

impl From<&str> for Identity {
    fn from(s: &str) -> Self {
        Identity(s.to_string())
    }
}

impl From<&String> for Identity {
    fn from(s: &String) -> Self {
        Identity(s.clone())
    }
}

impl From<String> for Identity {
    fn from(s: String) -> Self {
        Identity(s)
    }
}

pub struct CompiledAsyncComputation(AsyncComputationKernel);

impl Computation {
    pub fn compile_sync(&self, ctx: &CompilationContext) -> Result<CompiledSyncComputation> {
        // TODO(Morten) type check computation
        let compiled_ops: Vec<CompiledSyncOperation> = self
            .operations
            .par_iter() // par_iter seems to make sense here, see benches
            .map(|op| op.compile(&ctx))
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

    pub fn compile_async(&self, ctx: &CompilationContext) -> Result<CompiledAsyncComputation> {
        // TODO(Morten) check that the role assignment is safe

        // using a Vec instead of eg HashSet here since we can expect it to be very small
        let own_roles: Vec<&Role> = ctx
            .role_assignment
            .iter()
            .filter_map(|(role, identity)| {
                if identity == ctx.own_identity {
                    Some(role)
                } else {
                    None
                }
            })
            .collect();

        // TODO(Morten) is this deterministic?
        let comp = self.toposort()?;

        let own_operations = comp
            .operations
            .iter() // guessing that par_iter won't help here
            .filter(|op| match &op.placement {
                Placement::Host(plc) => own_roles.iter().any(|owner| *owner == &plc.owner),
                Placement::Replicated(plc) => own_roles
                    .iter()
                    .any(|owner| plc.owners.iter().any(|plc_owner| *owner == plc_owner)),
            })
            .collect::<Vec<_>>();

        let own_kernels: Vec<CompiledAsyncOperation> = own_operations
            .par_iter() // par_iter seems to make sense here, see benches
            .map(|op| op.compile(&ctx))
            .collect::<Result<Vec<_>>>()?;

        let own_output_names: Vec<String> = own_operations
            .iter() // guessing that par_iter won't help here
            .filter_map(|op| match op.kind {
                Operator::Output(_) => Some(op.name.clone()),
                _ => None,
            })
            .collect();

        // TODO(Morten) make second attempt at inlining
        fn remove_err<T, E>(r: std::result::Result<T, E>) -> std::result::Result<T, ()> {
            r.map_err(|_| ())
        }

        Ok(CompiledAsyncComputation(Box::new(move |sess| {
            let sess = Arc::new(sess);

            // create channels to be used between tasks
            let (senders, receivers): (Vec<AsyncSender>, HashMap<String, AsyncReceiver>) =
                own_kernels
                    .iter() // par_iter doesn't seem to improve performance here
                    .map(|op| {
                        let (sender, receiver) = oneshot::channel();
                        let shared_receiver: AsyncReceiver =
                            receiver.map(remove_err as fn(_) -> _).shared();
                        (sender, (op.name.clone(), shared_receiver))
                    })
                    .unzip();

            // spawn tasks
            let tasks: Vec<AsyncTask> = senders
                .into_iter() // into_par_iter seems to hurt performance here
                .zip(&own_kernels)
                .map(|(sender, op)| op.apply(&sess, &receivers, sender))
                .collect::<Result<Vec<_>>>()?;
            let session_handle = AsyncSessionHandle { tasks };

            // collect output futures
            let outputs: HashMap<String, AsyncReceiver> = own_output_names
                .iter()
                .map(|op_name| {
                    let val = receivers.get(op_name).cloned().unwrap(); // safe to unwrap per construction
                    (op_name.clone(), val)
                })
                .collect();

            Ok((session_handle, outputs))
        })))
    }
}

impl CompiledAsyncComputation {
    pub fn apply(
        &self,
        session: AsyncSession,
    ) -> Result<(AsyncSessionHandle, Environment<AsyncReceiver>)> {
        (self.0)(session)
    }
}

pub type Environment<V> = HashMap<String, V>;

/// In-order single-threaded executor.
///
/// This executor evaluates the operations of computations in-order, raising an error
/// in case data dependencies are not respected. This executor is intended for debug
/// and development only due to its unforgiving but highly predictable behaviour.
pub struct EagerExecutor {}

impl EagerExecutor {
    pub fn run_computation(
        &self,
        computation: &Computation,
        role_assignment: &RoleAssignment,
        own_identity: &Identity,
        session: SyncSession,
    ) -> Result<Environment<Value>> {
        let ctx = CompilationContext {
            role_assignment,
            own_identity,
        };
        let compiled_comp: CompiledSyncComputation = computation.compile_sync(&ctx)?;
        compiled_comp.apply(&session)
    }
}

pub struct TestExecutor {
    networking: Rc<dyn SyncNetworking>,
    storage: Rc<dyn SyncStorage>,
}

impl Default for TestExecutor {
    fn default() -> Self {
        TestExecutor {
            networking: Rc::new(LocalSyncNetworking::default()),
            storage: Rc::new(LocalSyncStorage::default()),
        }
    }
}

impl TestExecutor {
    pub fn from_storage(storage: &Rc<dyn SyncStorage>) -> TestExecutor {
        TestExecutor {
            networking: Rc::new(LocalSyncNetworking::default()),
            storage: Rc::clone(storage),
        }
    }

    pub fn from_networking(networking: &Rc<dyn SyncNetworking>) -> TestExecutor {
        TestExecutor {
            networking: Rc::clone(networking),
            storage: Rc::new(LocalSyncStorage::default()),
        }
    }

    pub fn from_networking_storage(
        networking: &Rc<dyn SyncNetworking>,
        storage: &Rc<dyn SyncStorage>,
    ) -> TestExecutor {
        TestExecutor {
            networking: Rc::clone(networking),
            storage: Rc::clone(storage),
        }
    }

    pub fn run_computation(
        &self,
        computation: &Computation,
        arguments: SyncArgs,
    ) -> Result<HashMap<String, Value>> {
        let own_identity = Identity::from("tester");

        let all_roles = computation
            .operations
            .iter()
            .flat_map(|op| -> Box<dyn Iterator<Item = &Role>> {
                match &op.placement {
                    // TODO(Morten) box seems too complicated..?
                    Placement::Host(plc) => Box::new(std::iter::once(&plc.owner)),
                    Placement::Replicated(plc) => Box::new(plc.owners.iter()),
                }
            })
            .collect::<HashSet<_>>();

        let role_assignment = all_roles
            .into_iter()
            .map(|role| (role.clone(), own_identity.clone()))
            .collect();

        let session = SyncSession {
            arguments,
            sid: SessionId::from("abcdef"), // TODO sample random string
            networking: Rc::clone(&self.networking),
            storage: Rc::clone(&self.storage),
        };

        let eager_exec = EagerExecutor {};
        eager_exec.run_computation(computation, &role_assignment, &own_identity, session)
    }
}

/// A session is essentially the activation frame of the graph function call.
#[derive(Clone)]
pub struct AsyncSession {
    pub sid: SessionId,
    pub arguments: HashMap<String, Value>,
    pub networking: Arc<dyn Send + Sync + AsyncNetworking>,
    pub storage: Arc<dyn Send + Sync + AsyncStorage>,
}

pub type RoleAssignment = HashMap<Role, Identity>;

pub struct AsyncSessionHandle {
    tasks: Vec<AsyncTask>,
}

impl AsyncSessionHandle {
    fn process_task_result(
        res: std::result::Result<std::result::Result<(), Error>, tokio::task::JoinError>,
    ) -> Option<anyhow::Error> {
        match res {
            Ok(Ok(_)) => None,
            Ok(Err(e)) => Some(anyhow::Error::from(e)),
            Err(e) if e.is_cancelled() => None,
            Err(e) => Some(anyhow::Error::from(e)),
        }
    }

    pub fn block_on(self) -> Vec<anyhow::Error> {
        let runtime_handle = tokio::runtime::Handle::current();

        let mut errors = Vec::new();
        for task in self.tasks {
            let res = runtime_handle.block_on(task);
            if let Some(e) = AsyncSessionHandle::process_task_result(res) {
                errors.push(e);
            }
        }
        errors
    }

    pub async fn join(&mut self) -> Vec<anyhow::Error> {
        let mut errors = Vec::new();
        for task in &mut self.tasks {
            let res = task.await;
            if let Some(e) = AsyncSessionHandle::process_task_result(res) {
                errors.push(e);
            }
        }
        errors
    }

    // TODO(Morten)
    // async fn join(self) -> Result<()> {
    //     use futures::StreamExt;
    //     let tasks = self.tasks.into_iter().collect::<futures::stream::FuturesUnordered<_>>();
    //     let res = tasks.collect::<Vec<_>>().await;
    // }

    pub fn abort(&self) {
        for task in &self.tasks {
            task.abort()
        }
    }
}

pub struct AsyncExecutor {
    // TODO(Morten) keep cache of compiled computations
}

impl Default for AsyncExecutor {
    fn default() -> Self {
        AsyncExecutor {}
    }
}

impl AsyncExecutor {
    pub fn run_computation(
        &self,
        computation: &Computation,
        role_assignment: &RoleAssignment,
        own_identity: &Identity,
        session: AsyncSession,
    ) -> Result<(AsyncSessionHandle, HashMap<String, AsyncReceiver>)> {
        let ctx = CompilationContext {
            role_assignment,
            own_identity,
        };

        let compiled_comp = computation.compile_async(&ctx)?;

        compiled_comp.apply(session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_prod_ops() {
        use crate::standard::Float32Tensor;
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
                owner: Role::from("alice"),
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
                owner: Role::from("alice"),
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
                owner: Role::from("alice"),
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
                owner: Role::from("alice"),
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
                owner: Role::from("alice"),
            }),
        };
        let operations = vec![x_op, y_op, mul_op, dot_op, mean_op];
        let comp = Computation { operations }.toposort().unwrap();

        let exec = TestExecutor::default();
        let _outputs = exec.run_computation(&comp, SyncArgs::new()).unwrap();
    }

    use std::convert::TryInto;

    #[test]
    fn test_textual_represenation() -> std::result::Result<(), anyhow::Error> {
        let comp: Computation = "let x = Constant([1.0]) @alice".try_into()?;
        assert_eq!(format!("{:?}", comp), "Computation { operations: [Operation { name: \"x\", kind: Constant(ConstantOp { value: String(\"[1.0]\") }), inputs: [], placement: Host(HostPlacement { owner: Role(\"alice\") }) }] }");
        Ok(())
    }

    #[test]
    fn test_eager_executor() {
        use crate::prim::Nonce;
        use crate::standard::Shape;

        let key_op = Operation {
            name: "key".into(),
            kind: Operator::PrimGenPrfKey(PrimGenPrfKeyOp),
            inputs: vec![],
            placement: Placement::Host(HostPlacement {
                owner: Role::from("alice"),
            }),
        };

        let seed_op = Operation {
            name: "seed".into(),
            kind: Operator::PrimDeriveSeed(PrimDeriveSeedOp {
                nonce: Nonce(vec![1, 2, 3]),
            }),
            inputs: vec!["key".into()],
            placement: Placement::Host(HostPlacement {
                owner: Role::from("alice"),
            }),
        };

        let shape_op = Operation {
            name: "shape".into(),
            kind: Operator::Constant(ConstantOp {
                value: Value::Shape(Shape(vec![2, 3])),
            }),
            inputs: vec![],
            placement: Placement::Host(HostPlacement {
                owner: Role::from("alice"),
            }),
        };

        let output_op = Operation {
            name: "z".into(),
            kind: Operator::Output(OutputOp {
                ty: Ty::Ring64TensorTy,
            }),
            inputs: vec!["x10".into()],
            placement: Placement::Host(HostPlacement {
                owner: Role::from("alice"),
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
                    owner: Role::from("alice"),
                }),
            })
            .collect();

        let mut operations = sample_ops;
        operations.extend(vec![key_op, seed_op, shape_op, output_op]);

        let comp = Computation { operations }.toposort().unwrap();

        let exec = TestExecutor::default();
        let outputs = exec.run_computation(&comp, SyncArgs::new()).unwrap();
        assert_eq!(outputs.keys().collect::<Vec<_>>(), vec!["z"]);
    }
}
