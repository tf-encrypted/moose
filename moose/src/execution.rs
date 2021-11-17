#![allow(unused_macros)]

use crate::compilation::compile_passes;
use crate::compilation::Pass;
use crate::computation::*;
use crate::error::{Error, Result};
use crate::networking::{
    AsyncNetworking, LocalAsyncNetworking, LocalSyncNetworking, SyncNetworking,
};
use crate::storage::{AsyncStorage, LocalAsyncStorage, LocalSyncStorage, SyncStorage};

use derive_more::Display;
use futures::future::{Map, Shared};
use futures::prelude::*;
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::rc::Rc;
use std::sync::Arc;
use tokio::runtime::Runtime;
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

/// A bridge beetween the old and new framework.
///
/// We could implement those on the Operators themselves, but having it a separate struct will make it easier to delete once we are done.
struct Bridge;

impl Bridge {
    pub fn nullary(op: Operator) -> NullaryAsyncKernel {
        Box::new(move |sess, sender| {
            let new_sess = sess.new_sess.clone();
            let op = op.clone();
            let plc = sess.host.clone();
            tokio::spawn(async move {
                use crate::kernels::Session;
                let y = new_sess.execute(op, &plc, vec![])?;
                map_send_result(sender.send(y))
            })
        })
    }

    pub fn unary(op: Operator) -> UnaryAsyncKernel {
        Box::new(move |sess, v, sender| {
            let new_sess = sess.new_sess.clone();
            let op = op.clone();
            let plc = sess.host.clone();
            tokio::spawn(async move {
                use crate::kernels::Session;
                let v: Value = v.await.map_err(map_receive_error)?;
                let y = new_sess.execute(op, &plc, vec![v])?;
                map_send_result(sender.send(y))
            })
        })
    }

    pub fn binary(op: Operator) -> BinaryAsyncKernel {
        Box::new(move |sess, v0, v1, sender| {
            let new_sess = sess.new_sess.clone();
            let op = op.clone();
            let plc = sess.host.clone();
            tokio::spawn(async move {
                use crate::kernels::Session;
                let v0: Value = v0.await.map_err(map_receive_error)?;
                let v1: Value = v1.await.map_err(map_receive_error)?;
                let y = new_sess.execute(op, &plc, vec![v0, v1])?;
                map_send_result(sender.send(y))
            })
        })
    }

    pub fn ternary(op: Operator) -> TernaryAsyncKernel {
        Box::new(move |sess, v0, v1, v2, sender| {
            let new_sess = sess.new_sess.clone();
            let op = op.clone();
            let plc = sess.host.clone();
            tokio::spawn(async move {
                use crate::kernels::Session;
                let v0: Value = v0.await.map_err(map_receive_error)?;
                let v1: Value = v1.await.map_err(map_receive_error)?;
                let v2: Value = v2.await.map_err(map_receive_error)?;
                let y = new_sess.execute(op, &plc, vec![v0, v1, v2])?;
                map_send_result(sender.send(y))
            })
        })
    }

    pub fn variadic(op: Operator) -> VariadicAsyncKernel {
        Box::new(move |sess, xs, sender| {
            let new_sess = sess.new_sess.clone();
            let op = op.clone();
            let plc = sess.host.clone();
            tokio::spawn(async move {
                use crate::kernels::Session;
                let xs: Vec<Value> = futures::future::try_join_all(xs)
                    .await
                    .map_err(map_receive_error)?;
                let y = new_sess.execute(op, &plc, xs)?;
                map_send_result(sender.send(y))
            })
        })
    }
}

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
            if val.ty() == Ty::Unit {
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
    pub use_sync_bridge: bool,
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
        if ctx.use_sync_bridge {
            self.bridge_compile(ctx)
        } else {
            self.older_compile(ctx)
        }
    }
}

impl Operation {
    fn older_compile(&self, ctx: &CompilationContext) -> Result<CompiledAsyncOperation> {
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

    fn bridge_compile(&self, ctx: &CompilationContext) -> Result<CompiledAsyncOperation> {
        match self.kind {
            // This kinds of operators should not be using the brdiged Sync kernels. Instead, they should be using the old kernels,
            // because there is where the Sync/Async storage and networking is done right at the moment.
            Operator::Receive(_) | Operator::Send(_) | Operator::Load(_) | Operator::Save(_) => {
                return self.older_compile(ctx)
            }
            _ => {}
        };

        let op = self.clone();

        match self.kind.sig().arity() {
            Some(0) => {
                check_arity(&self.name, &self.inputs, 0)?;
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, _, sender| {
                        let k = Bridge::nullary(op.kind.clone());
                        Ok(k(sess, sender))
                    }),
                })
            }
            Some(1) => {
                check_arity(&self.name, &self.inputs, 1)?;
                let x0_name = self.inputs[0].clone();
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, env, sender| {
                        let x0 = find_env(env, &x0_name)?;
                        let k = Bridge::unary(op.kind.clone());
                        Ok(k(sess, x0, sender))
                    }),
                })
            }
            Some(2) => {
                check_arity(&self.name, &self.inputs, 2)?;
                let x0_name = self.inputs[0].clone();
                let x1_name = self.inputs[1].clone();
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, env, sender| {
                        let x0 = find_env(env, &x0_name)?;
                        let x1 = find_env(env, &x1_name)?;
                        let k = Bridge::binary(op.kind.clone());
                        Ok(k(sess, x0, x1, sender))
                    }),
                })
            }
            Some(3) => {
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
                        let k = Bridge::ternary(op.kind.clone());
                        Ok(k(sess, x0, x1, x2, sender))
                    }),
                })
            }
            None => {
                let inputs = self.inputs.clone();
                Ok(CompiledAsyncOperation {
                    name: self.name.clone(),
                    kernel: Box::new(move |sess, env, sender| {
                        let xs = inputs
                            .iter()
                            .map(|input| find_env(env, input))
                            .collect::<Result<Vec<_>>>()?;
                        let k = Bridge::variadic(op.kind.clone());
                        Ok(k(sess, xs, sender))
                    }),
                })
            }
            _ => unimplemented!("No support for the bridge of an unexpected arity"),
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
    pub fn as_graph(&self) -> Graph<(String, usize), ()> {
        let mut graph = Graph::new();

        let mut vertex_map: HashMap<&str, NodeIndex> = HashMap::new();

        let mut send_nodes: HashMap<&RendezvousKey, NodeIndex> = HashMap::new();
        let mut recv_nodes: HashMap<&RendezvousKey, NodeIndex> = HashMap::new();

        let mut rdv_keys: HashSet<&RendezvousKey> = HashSet::new();

        for (i, op) in self.operations.iter().enumerate() {
            let vertex = graph.add_node((op.name.clone(), i));
            match op.kind {
                Operator::Send(ref op) => {
                    let key = &op.rendezvous_key;

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
                    let key = &op.rendezvous_key;

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

        graph
    }

    pub fn toposort(&self) -> Result<Computation> {
        let graph = self.as_graph();
        let toposort = toposort(&graph, None).map_err(|_| {
            Error::MalformedComputation("There is a cycle detected in the runtime graph".into())
        })?;

        let operations = toposort
            .iter()
            .map(|node| self.operations[graph[*node].1].clone())
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
            .map(|op| op.compile(ctx))
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
                Placement::Additive(plc) => own_roles
                    .iter()
                    .any(|owner| plc.owners.iter().any(|plc_owner| *owner == plc_owner)),
            })
            .collect::<Vec<_>>();

        let own_kernels: Vec<CompiledAsyncOperation> = own_operations
            .par_iter() // par_iter seems to make sense here, see benches
            .map(|op| op.compile(ctx))
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
            use_sync_bridge: false, // Sync Session should never use the bridge
        };
        let computation = compile_passes(computation, &[Pass::Typing, Pass::DeprecatedLogical])
            .map_err(|e| {
                Error::MalformedComputation(format!(
                    "Failed to perform rust compiler passes: {}",
                    e
                ))
            })?;
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
                    Placement::Additive(plc) => Box::new(plc.owners.iter()),
                }
            })
            .collect::<HashSet<_>>();

        let role_assignment = all_roles
            .into_iter()
            .map(|role| (role.clone(), own_identity.clone()))
            .collect();

        let session = SyncSession {
            arguments,
            sid: SessionId::try_from("abcdef").unwrap(), // TODO sample random string
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
    // Two fields to support the new framework
    pub new_sess: Arc<crate::kernels::SyncSession>,
    pub host: Arc<Placement>,
}

pub type RoleAssignment = HashMap<Role, Identity>;

//struct CancelJoinHandle {
//    task: AsyncTask,
//}
//
//impl Drop for CancelJoinHandle {
//    fn drop(&mut self) {
//        self.task.abort();
//    }
//}
//
//impl Future for CancelJoinHandle {
//    type Output = Poll<std::result::Result<std::result::Result<(), Error>, JoinError>>;
//    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
//        std::task::Poll::Ready(Pin::new(&mut self.task).poll(cx))
//    }
//}

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

    pub async fn join_on_first_error(self) -> anyhow::Result<()> {
        use crate::error::Error::{OperandUnavailable, ResultUnused};
        use futures::StreamExt;

        // TODO:
        // iterate and wrap all tasks with MyJoinHandle
        // impl drop and future for MyJoinHandle
        //     in Drop: call join_handle.abort()

        //let mut task_vec = Vec::new();
        //for task in self.tasks.into_iter() {
        //    task_vec.push(CancelJoinHandle{task: task});
        //}
        //let mut tasks = task_vec.into_iter().collect::<futures::stream::FuturesUnordered<_>>();

        let mut tasks = self
            .tasks
            .into_iter()
            .collect::<futures::stream::FuturesUnordered<_>>();

        while let Some(x) = tasks.next().await {
            match x {
                Ok(Ok(_)) => {
                    continue;
                }
                Ok(Err(e)) => {
                    match e {
                        // OperandUnavailable and ResultUnused are typically not root causes.
                        // Wait to get an error that would indicate the root cause of the problem,
                        // and return it instead.
                        OperandUnavailable => continue,
                        ResultUnused => continue,
                        _ => {
                            for task in tasks.iter() {
                                task.abort();
                            }
                            return Err(anyhow::Error::from(e));
                        }
                    }
                }
                Err(e) => {
                    if e.is_cancelled() {
                        continue;
                    } else if e.is_panic() {
                        for task in tasks.iter() {
                            task.abort();
                        }
                        return Err(anyhow::Error::from(e));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn abort(&self) {
        for task in &self.tasks {
            task.abort()
        }
    }
}

pub struct AsyncExecutor {
    session_ids: HashSet<SessionId>,
}

impl Default for AsyncExecutor {
    fn default() -> Self {
        AsyncExecutor {
            session_ids: HashSet::new(),
        }
    }
}

impl AsyncExecutor {
    pub fn run_computation(
        &mut self,
        computation: &Computation,
        role_assignment: &RoleAssignment,
        own_identity: &Identity,
        session: AsyncSession,
        use_sync_bridge: bool,
    ) -> Result<(AsyncSessionHandle, HashMap<String, AsyncReceiver>)> {
        if !self.session_ids.insert(session.sid.clone()) {
            return Err(Error::SessionAlreadyExists(format!("{}", session.sid)));
        }

        let ctx = CompilationContext {
            role_assignment,
            own_identity,
            use_sync_bridge,
        };

        let computation = compile_passes(computation, &[Pass::Typing, Pass::DeprecatedLogical])
            .map_err(|e| {
                Error::MalformedComputation(format!(
                    "Failed to perform rust compiler passes: {}",
                    e
                ))
            })?;
        let compiled_comp = computation.compile_async(&ctx)?;
        compiled_comp.apply(session)
    }
}

pub struct AsyncTestRuntime {
    pub identities: Vec<Identity>,
    pub executors: HashMap<Identity, AsyncExecutor>,
    pub runtime_storage: HashMap<Identity, Arc<dyn Send + Sync + AsyncStorage>>,
    pub networking: AsyncNetworkingImpl,
}

impl AsyncTestRuntime {
    pub fn new(storage_mapping: HashMap<String, HashMap<String, Value>>) -> Self {
        let mut executors: HashMap<Identity, AsyncExecutor> = HashMap::new();
        let networking: Arc<dyn Send + Sync + AsyncNetworking> =
            Arc::new(LocalAsyncNetworking::default());
        let mut runtime_storage: HashMap<Identity, Arc<dyn Send + Sync + AsyncStorage>> =
            HashMap::new();
        let mut identities = Vec::new();
        for (identity_str, storage) in storage_mapping {
            let identity = Identity::from(identity_str.clone()).clone();
            identities.push(identity.clone());
            // TODO handle Result in map predicate instead of `unwrap`
            let storage = storage
                .iter()
                .map(|arg| (arg.0.to_owned(), arg.1.to_owned()))
                .collect::<HashMap<String, Value>>();

            let exec_storage: Arc<dyn Send + Sync + AsyncStorage> =
                Arc::new(LocalAsyncStorage::from_hashmap(storage));
            runtime_storage.insert(identity.clone(), exec_storage);

            let executor = AsyncExecutor::default();
            executors.insert(identity.clone(), executor);
        }

        AsyncTestRuntime {
            identities,
            executors,
            runtime_storage,
            networking,
        }
    }
    pub fn evaluate_computation(
        &mut self,
        computation: &Computation,
        role_assignments: HashMap<Role, Identity>,
        arguments: HashMap<String, Value>,
    ) -> Result<Option<HashMap<String, Value>>> {
        let mut session_handles: Vec<AsyncSessionHandle> = Vec::new();
        let mut output_futures: HashMap<String, AsyncReceiver> = HashMap::new();
        let rt = Runtime::new().unwrap();
        let _guard = rt.enter();

        let (valid_role_assignments, missing_role_assignments): (
            HashMap<Role, Identity>,
            HashMap<Role, Identity>,
        ) = role_assignments
            .into_iter()
            .partition(|kv| self.identities.contains(&kv.1));
        if !missing_role_assignments.is_empty() {
            let missing_roles: Vec<&Role> = missing_role_assignments.keys().collect();
            let missing_identities: Vec<&Identity> = missing_role_assignments.values().collect();
            return Err(Error::TestRuntime(format!("Role assignment included identities unknown to Moose runtime: missing identities {:?} for roles {:?}.",
                missing_identities, missing_roles)));
        }

        for (own_identity, executor) in self.executors.iter_mut() {
            let moose_session = AsyncSession {
                sid: SessionId::try_from("foobar").unwrap(),
                arguments: arguments.clone(),
                networking: Arc::clone(&self.networking),
                storage: Arc::clone(&self.runtime_storage[own_identity]),
                // Creating independent new framework's sync sessions for the bridge
                new_sess: Arc::new(crate::kernels::SyncSession::new(
                    SessionId::try_from("foobar").unwrap(),
                    arguments.clone(),
                    valid_role_assignments.clone(),
                )),
                host: Arc::new(Placement::Host(HostPlacement {
                    owner: own_identity.0.clone().into(),
                })),
            };
            let (moose_session_handle, outputs) = executor
                .run_computation(
                    computation,
                    &valid_role_assignments,
                    own_identity,
                    moose_session,
                    true, // Let's test with the new framework
                )
                .unwrap();

            for (output_name, output_future) in outputs {
                output_futures.insert(output_name, output_future);
            }

            session_handles.push(moose_session_handle)
        }

        for handle in session_handles {
            let result = rt.block_on(handle.join_on_first_error());
            if let Err(e) = result {
                return Err(Error::TestRuntime(e.to_string()));
            }
        }

        let outputs = rt.block_on(async {
            let mut outputs: HashMap<String, Value> = HashMap::new();
            for (output_name, output_future) in output_futures {
                let value = output_future.await.unwrap();
                outputs.insert(output_name, value);
            }

            outputs
        });

        Ok(Some(outputs))
    }

    pub fn read_value_from_storage(&self, identity: Identity, key: String) -> Result<Value> {
        let rt = Runtime::new().unwrap();
        let _guard = rt.enter();
        let val = rt.block_on(async {
            let val = self.runtime_storage[&identity]
                .load(&key, &SessionId::try_from("foobar").unwrap(), None, "")
                .await
                .unwrap();
            val
        });

        Ok(val)
    }

    pub fn write_value_to_storage(
        &self,
        identity: Identity,
        key: String,
        value: Value,
    ) -> Result<()> {
        let rt = Runtime::new().unwrap();
        let _guard = rt.enter();
        let identity_storage = match self.runtime_storage.get(&identity) {
            Some(store) => store,
            None => {
                return Err(Error::TestRuntime(format!(
                    "Runtime does not contain storage for identity {:?}.",
                    identity.to_string()
                )));
            }
        };

        let result = rt.block_on(async {
            identity_storage
                .save(&key, &SessionId::try_from("yo").unwrap(), &value)
                .await
        });
        if let Err(e) = result {
            return Err(Error::TestRuntime(e.to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::networking::NetworkingPass;
    use crate::host::{
        HostFloat32Tensor, HostFloat64Tensor, HostInt64Tensor, HostShape, HostString, HostTensor,
        RawShape,
    };
    use crate::host::{HostRing128Tensor, HostRing64Tensor};
    use crate::prim::{RawSeed, Seed};
    use itertools::Itertools;
    use maplit::hashmap;
    use ndarray::prelude::*;
    use std::convert::TryInto;

    fn _run_computation_test(
        computation: Computation,
        storage_mapping: HashMap<String, HashMap<String, Value>>,
        role_assignments: HashMap<String, String>,
        arguments: HashMap<String, Value>,
        run_async: bool,
    ) -> std::result::Result<HashMap<String, Value>, anyhow::Error> {
        match run_async {
            false => {
                let executor = crate::kernels::TestSyncExecutor::default();
                let session = crate::kernels::SyncSession::new(
                    SessionId::try_from("foobar").unwrap(),
                    arguments,
                    hashmap!(),
                );
                let outputs = executor.run_computation(&computation, &session)?;
                Ok(outputs)
            }
            true => {
                let valid_role_assignments = role_assignments
                    .into_iter()
                    .map(|arg| (Role::from(arg.1), Identity::from(arg.0)))
                    .collect::<HashMap<Role, Identity>>();
                let mut executor = AsyncTestRuntime::new(storage_mapping);
                let outputs = executor.evaluate_computation(
                    &computation,
                    valid_role_assignments,
                    arguments,
                )?;
                match outputs {
                    Some(outputs) => Ok(outputs),
                    None => Ok(hashmap!()),
                }
            }
        }
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_eager_executor(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let mut definition = String::from(
            r#"key = PrimPrfKeyGen() @Host(alice)
        seed = PrimDeriveSeed {sync_key = [1, 2, 3]}: (PrfKey) -> Seed (key) @Host(alice)
        shape = Constant{value = Shape([2, 3])} @Host(alice)
        "#,
        );
        let body = (0..100)
            .map(|i| {
                format!(
                    "x{} = RingSampleSeeded: (Shape, Seed) -> Ring64Tensor (shape, seed) @Host(alice)",
                    i
                )
            })
            .join("\n");
        definition.push_str(&body);
        definition.push_str("\nz = Output: (Ring64Tensor) -> Ring64Tensor (x0) @Host(alice)");

        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            definition.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        assert_eq!(outputs.keys().collect::<Vec<_>>(), vec!["z"]);
        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_constants_derive_seed(
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = r#"key = Constant{value=PrfKey(00000000000000000000000000000000)} @Host(alice)
        seed = PrimDeriveSeed {sync_key = [1, 2, 3]}: (PrfKey) -> Seed (key) @Host(alice)
        output = Output: (Seed) -> Seed (seed) @Host(alice)"#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let seed: Seed = (outputs.get("output").unwrap().clone()).try_into()?;
        assert_eq!(
            seed.0,
            RawSeed([224, 87, 133, 2, 90, 170, 32, 253, 25, 80, 93, 74, 122, 196, 50, 1])
        );
        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_constants_sample_ring(
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = r#"seed = Constant{value=Seed(00000000000000000000000000000000)} @Host(alice)
        xshape = Constant{value=Shape([2, 2])} @Host(alice)
        sampled = RingSampleSeeded: (Shape, Seed) -> Ring64Tensor (xshape, seed) @Host(alice)
        output = Output: (Ring64Tensor) -> Ring64Tensor (sampled) @Host(alice)
        "#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let x_sampled: HostRing64Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
        assert_eq!(x_sampled.shape().0, RawShape(vec![2, 2]));

        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_standard_input(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Input {arg_name = "x"}: () -> Int64Tensor @Host(alice)
        y = Input {arg_name = "y"}: () -> Int64Tensor @Host(alice)
        z = HostAdd: (Int64Tensor, Int64Tensor) -> Int64Tensor (x, y) @Host(alice)
        output = Output: (Int64Tensor) -> Int64Tensor (z) @Host(alice)
        "#;
        let x: Value = "Int64Tensor([5]) @Host(alice)".try_into()?;
        let y: Value = "Int64Tensor([10]) @Host(alice)".try_into()?;
        let arguments: HashMap<String, Value> = hashmap!("x".to_string() => x, "y".to_string()=> y);
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let z: HostInt64Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
        let expected: Value = "Int64Tensor([15]) @Host(alice)".try_into()?;
        assert_eq!(expected, z.into());
        Ok(())
    }

    #[rstest]
    #[case("Int64Tensor([8]) @Host(alice)", true)]
    #[case("Int32Tensor([8]) @Host(alice)", true)]
    #[case("Float32Tensor([8]) @Host(alice)", true)]
    #[case("Float64Tensor([8]) @Host(alice)", true)]
    #[case("Int64Tensor([8]) @Host(alice)", false)]
    #[case("Int32Tensor([8]) @Host(alice)", false)]
    #[case("Float32Tensor([8]) @Host(alice)", false)]
    #[case("Float64Tensor([8]) @Host(alice)", false)]
    fn test_load_save(
        #[case] input_data: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        use crate::textual::ToTextual;

        let data_type_str = input_data.ty().to_textual();
        let source_template = r#"x_uri = Input {arg_name="x_uri"}: () -> String () @Host(alice)
        x_query = Input {arg_name="x_query"}: () -> String () @Host(alice)
        saved_uri = Constant{value = String("saved_data")} () @Host(alice)
        x = Load: (String, String) -> TensorType (x_uri, x_query) @Host(alice)
        save = Save: (String, TensorType) -> Unit (saved_uri, x) @Host(alice)
        output = Output: (Unit) -> Unit (save) @Host(alice)
        "#;
        let source = source_template.replace("TensorType", &data_type_str);
        let plc = HostPlacement {
            owner: "alice".into(),
        };
        let arguments: HashMap<String, Value> = hashmap!("x_uri".to_string()=> HostString("input_data".to_string(), plc.clone()).into(),
            "x_query".to_string() => HostString("".to_string(), plc.clone()).into(),
            "saved_uri".to_string() => HostString("saved_data".to_string(), plc).into());

        let saved_data = match run_async {
            true => {
                let storage_mapping: HashMap<String, HashMap<String, Value>> = hashmap!("alice".to_string() => hashmap!("input_data".to_string() => input_data.clone()));
                let role_assignments: HashMap<String, String> =
                    hashmap!("alice".to_string() => "alice".to_string());
                let valid_role_assignments = role_assignments
                    .into_iter()
                    .map(|arg| (Role::from(arg.1), Identity::from(arg.0)))
                    .collect::<HashMap<Role, Identity>>();
                let mut executor = AsyncTestRuntime::new(storage_mapping);
                let _outputs = executor.evaluate_computation(
                    &source.try_into()?,
                    valid_role_assignments,
                    arguments,
                )?;

                executor.read_value_from_storage(
                    Identity::from("alice".to_string()),
                    "saved_data".to_string(),
                )?
            }
            false => {
                let store: HashMap<String, Value> =
                    hashmap!("input_data".to_string() => input_data.clone());
                let storage: Rc<dyn SyncStorage> = Rc::new(LocalSyncStorage::from_hashmap(store));
                let executor = TestExecutor::from_storage(&storage);
                let _outputs = executor.run_computation(&source.try_into()?, arguments)?;
                storage.load(
                    "saved_data",
                    &SessionId::try_from("foobar").unwrap(),
                    None,
                    "",
                )?
            }
        };

        assert_eq!(input_data, saved_data);
        Ok(())
    }

    use rstest::rstest;
    #[rstest]
    #[case(
        "0",
        "Int64Tensor([[1, 2], [3, 4], [5, 6], [7, 8]]) @Host(alice)",
        true
    )]
    #[case("1", "Int64Tensor([[1, 2, 5, 6], [3, 4, 7, 8]]) @Host(alice)", true)]
    #[case(
        "0",
        "Int64Tensor([[1, 2], [3, 4], [5, 6], [7, 8]]) @Host(alice)",
        false
    )]
    #[case("1", "Int64Tensor([[1, 2, 5, 6], [3, 4, 7, 8]]) @Host(alice)", false)]
    fn test_standard_concatenate(
        #[case] axis: usize,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source_template = r#"x_0 = Constant{value=Int64Tensor([[1,2], [3,4]])} @Host(alice)
        x_1 = Constant{value=Int64Tensor([[5, 6], [7,8]])} @Host(alice)
        concatenated = HostConcat {axis=test_axis}: ([Int64Tensor]) -> Int64Tensor (x_0, x_1) @Host(alice)
        output = Output: (Int64Tensor) -> Int64Tensor (concatenated) @Host(alice)
        "#;
        let source = source_template.replace("test_axis", &axis.to_string());
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let concatenated: HostInt64Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
        assert_eq!(expected_result, concatenated.into());
        Ok(())
    }

    #[rstest]
    #[case("HostAdd", "Int64Tensor([8]) @Host(alice)", true)]
    #[case("HostSub", "Int64Tensor([2]) @Host(alice)", true)]
    #[case("HostMul", "Int64Tensor([15]) @Host(alice)", true)]
    #[case("HostDiv", "Int64Tensor([1]) @Host(alice)", true)]
    #[case("HostAdd", "Int64Tensor([8]) @Host(alice)", false)]
    #[case("HostSub", "Int64Tensor([2]) @Host(alice)", false)]
    #[case("HostMul", "Int64Tensor([15]) @Host(alice)", false)]
    #[case("HostDiv", "Int64Tensor([1]) @Host(alice)", false)]
    fn test_standard_op(
        #[case] test_op: String,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source_template = r#"x0 = Constant{value=Int64Tensor([5])} @Host(alice)
        x1 = Constant{value=Int64Tensor([3])} @Host(bob)
        res = StdOp: (Int64Tensor, Int64Tensor) -> Int64Tensor (x0, x1) @Host(alice)
        output = Output: (Int64Tensor) -> Int64Tensor (res) @Host(alice)
        "#;
        let source = source_template.replace("StdOp", &test_op);
        let computation: Computation = source.try_into()?;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!(), "bob".to_string()=>hashmap!());
        let role_assignments: HashMap<String, String> = hashmap!("alice".to_string() => "alice".to_string(), "bob".to_string() => "bob".to_string());

        let outputs = match run_async {
            true => {
                let computation = NetworkingPass::pass(&computation).unwrap().unwrap();
                _run_computation_test(
                    computation,
                    storage_mapping,
                    role_assignments,
                    arguments,
                    run_async,
                )?
            }
            false => _run_computation_test(
                computation,
                storage_mapping,
                role_assignments,
                arguments,
                run_async,
            )?,
        };

        let res: HostInt64Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
        assert_eq!(expected_result, res.into());
        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_standard_dot(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let source = r#"x0 = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])} @Host(alice)
        x1 = Constant{value=Float32Tensor([[1.0, 0.0], [0.0, 1.0]])} @Host(bob)
        res = HostDot: (Float32Tensor, Float32Tensor) -> Float32Tensor (x0, x1) @Host(alice)
        output = Output: (Float32Tensor) -> Float32Tensor (res) @Host(alice)
        "#;
        let computation: Computation = source.try_into()?;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!(), "bob".to_string()=>hashmap!());
        let role_assignments: HashMap<String, String> = hashmap!("alice".to_string() => "alice".to_string(), "bob".to_string() => "bob".to_string());

        let outputs = match run_async {
            true => {
                let computation = compile_passes(&computation, &[Pass::Networking])?;
                _run_computation_test(
                    computation,
                    storage_mapping,
                    role_assignments,
                    arguments,
                    run_async,
                )?
            }
            false => _run_computation_test(
                computation,
                storage_mapping,
                role_assignments,
                arguments,
                run_async,
            )?,
        };

        let expected_output: Value = HostTensor::<f32>(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            HostPlacement {
                owner: "alice".into(),
            },
        )
        .into();
        assert_eq!(outputs["output"], expected_output);
        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_standard_inverse(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value=Float32Tensor([[3.0, 2.0], [2.0, 3.0]])} : () -> Float32Tensor @Host(alice)
        x_inv = HostInverse : (Float32Tensor) -> Float32Tensor (x) @Host(alice)
        output = Output: (Float32Tensor) -> Float32Tensor (x_inv) @Host(alice)
        "#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let expected_output = HostTensor::<f32>(
            array![[0.6, -0.40000004], [-0.40000004, 0.6]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            HostPlacement {
                owner: "alice".into(),
            },
        );
        let x_inv: HostFloat32Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
        assert_eq!(expected_output, x_inv);
        Ok(())
    }

    #[rstest]
    #[case("Float32Tensor", true)]
    #[case("Float64Tensor", true)]
    #[case("Int64Tensor", true)]
    #[case("Float32Tensor", false)]
    #[case("Float64Tensor", false)]
    #[case("Int64Tensor", false)]
    fn test_standard_ones(
        #[case] dtype: String,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let template = r#"s = Constant{value=Shape([2, 2])} @Host(alice)
        r = HostOnes : (Shape) -> dtype (s) @Host(alice)
        output = Output : (dtype) -> dtype (r) @Host(alice)
        "#;
        let source = template.replace("dtype", &dtype);
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        match dtype.as_str() {
            "Float32Tensor" => {
                let r: HostFloat32Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
                assert_eq!(
                    r,
                    HostTensor::<f32>(
                        array![[1.0, 1.0], [1.0, 1.0]]
                            .into_dimensionality::<IxDyn>()
                            .unwrap(),
                        HostPlacement {
                            owner: "alice".into()
                        },
                    )
                );
                Ok(())
            }
            "Float64Tensor" => {
                let r: HostFloat64Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
                assert_eq!(
                    r,
                    HostTensor::<f64>(
                        array![[1.0, 1.0], [1.0, 1.0]]
                            .into_dimensionality::<IxDyn>()
                            .unwrap(),
                        HostPlacement {
                            owner: "alice".into()
                        },
                    )
                );
                Ok(())
            }
            "Int64Tensor" => {
                let r: HostInt64Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
                assert_eq!(
                    r,
                    HostTensor::<i64>(
                        array![[1, 1], [1, 1]]
                            .into_dimensionality::<IxDyn>()
                            .unwrap(),
                        HostPlacement {
                            owner: "alice".into()
                        },
                    )
                );
                Ok(())
            }
            _ => Err(anyhow::anyhow!("Failed to parse test case")),
        }
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_standard_shape(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value = Float32Tensor([[1.0, 2.0], [3.0, 4.0]])} @Host(alice)
        shape = Shape: (Float32Tensor) -> Shape (x) @Host(alice)
        output = Output: (Shape) -> Shape (shape) @Host(alice)"#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let actual_shape: HostShape = (outputs.get("output").unwrap().clone()).try_into()?;
        let actual_raw_shape = actual_shape.0;
        let expected_raw_shape = RawShape(vec![2, 2]);
        assert_eq!(actual_raw_shape, expected_raw_shape);

        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_shape_slice(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value = Shape([2, 3, 4, 5])} @Host(alice)
        slice = HostSlice {start = 1, end = 3}: (Shape) -> Shape (x) @Host(alice)
        output = Output: (Shape) -> Shape (slice) @Host(alice)"#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;
        let res: HostShape = (outputs.get("output").unwrap().clone()).try_into()?;
        let actual_shape = res.0;
        let expected_shape = RawShape(vec![3, 4]);
        assert_eq!(expected_shape, actual_shape);
        Ok(())
    }

    // TODO test for axis as vector when textual representation can support it
    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_standard_expand_dims(
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value = Int64Tensor([1, 2])} @Host(alice)
        expand_dims = HostExpandDims {axis = [1]}: (Int64Tensor) -> Int64Tensor (x) @Host(alice)
        output = Output: (Int64Tensor) -> Int64Tensor (expand_dims) @Host(alice)"#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let res: HostInt64Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
        let actual_shape = res.shape().0;
        let expected_shape = RawShape(vec![2, 1]);
        assert_eq!(expected_shape, actual_shape);
        Ok(())
    }

    #[rstest]
    #[case("HostSum", None, "Float32(10.0) @Host(alice)", true, true)]
    #[case(
        "HostSum",
        Some(0),
        "Float32Tensor([4.0, 6.0]) @Host(alice)",
        false,
        true
    )]
    #[case(
        "HostSum",
        Some(1),
        "Float32Tensor([3.0, 7.0]) @Host(alice)",
        false,
        true
    )]
    #[case("HostMean", None, "Float32(2.5) @Host(alice)", true, true)]
    #[case(
        "HostMean",
        Some(0),
        "Float32Tensor([2.0, 3.0]) @Host(alice)",
        false,
        true
    )]
    #[case(
        "HostMean",
        Some(1),
        "Float32Tensor([1.5, 3.5]) @Host(alice)",
        false,
        true
    )]
    #[case("HostSum", None, "Float32(10.0) @Host(alice)", true, false)]
    #[case(
        "HostSum",
        Some(0),
        "Float32Tensor([4.0, 6.0]) @Host(alice)",
        false,
        false
    )]
    #[case(
        "HostSum",
        Some(1),
        "Float32Tensor([3.0, 7.0]) @Host(alice)",
        false,
        false
    )]
    #[case("HostMean", None, "Float32(2.5) @Host(alice)", true, false)]
    #[case(
        "HostMean",
        Some(0),
        "Float32Tensor([2.0, 3.0]) @Host(alice)",
        false,
        false
    )]
    #[case(
        "HostMean",
        Some(1),
        "Float32Tensor([1.5, 3.5]) @Host(alice)",
        false,
        false
    )]
    fn test_standard_reduce_op(
        #[case] reduce_op_test: String,
        #[case] axis_test: Option<usize>,
        #[case] expected_result: Value,
        #[case] unwrap_flag: bool,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let axis_str: String =
            axis_test.map_or_else(|| "".to_string(), |v| format!("{{axis={}}}", v));

        let source = format!(
            r#"s = Constant{{value=Float32Tensor([[1, 2], [3, 4]])}} @Host(alice)
            r = {} {}: (Float32Tensor) -> Float32Tensor (s) @Host(alice)
            output = Output : (Float32Tensor) -> Float32Tensor (r) @Host(alice)
        "#,
            reduce_op_test, axis_str
        );
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let comp_result = outputs
            .get("output")
            .ok_or_else(|| anyhow::anyhow!("Expected result missing"))?;

        if unwrap_flag {
            if let Value::HostFloat32Tensor(x) = comp_result {
                let shaped_result = x.clone().reshape(HostShape(
                    RawShape(vec![1]),
                    HostPlacement {
                        owner: "alice".into(),
                    },
                ));
                assert_eq!(
                    expected_result,
                    Value::Float32(Box::new(shaped_result.0[0]))
                );
            } else {
                panic!("Value of incorrect type {:?}", comp_result);
            }
        } else {
            assert_eq!(&expected_result, comp_result);
        }
        Ok(())
    }

    #[rstest]
    #[case("Int64Tensor([[1, 3], [2, 4]]) @Host(alice)", true)]
    #[case("Int64Tensor([[1, 3], [2, 4]]) @Host(alice)", false)]
    fn test_standard_transpose(
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = r#"s = Constant{value=Int64Tensor([[1,2], [3, 4]])} @Host(alice)
        r = HostTranspose : (Int64Tensor) -> Int64Tensor (s) @Host(alice)
        output = Output : (Int64Tensor) -> Int64Tensor (r) @Host(alice)
        "#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let comp_result: HostInt64Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
        assert_eq!(expected_result, comp_result.into());
        Ok(())
    }

    #[rstest]
    #[case(true, "Float64Tensor([[1.0], [1.0], [1.0]]) @Host(alice)", true)]
    #[case(false, "Float64Tensor([[1.0, 1.0, 1.0]]) @Host(alice)", true)]
    #[case(true, "Float64Tensor([[1.0], [1.0], [1.0]]) @Host(alice)", false)]
    #[case(false, "Float64Tensor([[1.0, 1.0, 1.0]]) @Host(alice)", false)]
    fn test_standard_atleast_2d(
        #[case] to_column_vector: bool,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = format!(
            r#"x =  Constant{{value=Float64Tensor([1.0, 1.0, 1.0])}} @Host(alice)
        res = HostAtLeast2D {{ to_column_vector = {} }} : (Float64Tensor) -> Float64Tensor (x) @Host(alice)
        output = Output : (Float64Tensor) -> Float64Tensor (res) @Host(alice)
        "#,
            to_column_vector
        );
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let comp_result: HostFloat64Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
        assert_eq!(expected_result, comp_result.into());
        Ok(())
    }

    #[rstest]
    #[case("RingAdd", "Ring64Tensor([5]) @Host(alice)", true)]
    #[case("RingMul", "Ring64Tensor([6]) @Host(alice)", true)]
    #[case("RingSub", "Ring64Tensor([1]) @Host(alice)", true)]
    #[case("RingAdd", "Ring64Tensor([5]) @Host(alice)", false)]
    #[case("RingMul", "Ring64Tensor([6]) @Host(alice)", false)]
    #[case("RingSub", "Ring64Tensor([1]) @Host(alice)", false)]
    fn test_ring_binop_invocation(
        #[case] test_op: String,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = format!(
            r#"x =  Constant{{value=Ring64Tensor([3])}} @Host(alice)
        y = Constant{{value=Ring64Tensor([2])}} @Host(alice)
        res = {} : (Ring64Tensor, Ring64Tensor) -> Ring64Tensor (x, y) @Host(alice)
        output = Output : (Ring64Tensor) -> Ring64Tensor (res) @Host(alice)
        "#,
            test_op
        );
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let comp_result: HostRing64Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
        assert_eq!(expected_result, comp_result.into());
        Ok(())
    }

    #[rstest]
    #[case(
        "Ring64Tensor",
        "Ring64Tensor([[1, 2], [3, 4]])",
        "Ring64Tensor([[1, 0], [0, 1]])",
        "Ring64Tensor([[1, 2], [3, 4]]) @Host(alice)",
        true
    )]
    #[case(
        "Ring64Tensor",
        "Ring64Tensor([[1, 2], [3, 4]])",
        "Ring64Tensor([1, 1])",
        "Ring64Tensor([3, 7]) @Host(alice)",
        true
    )]
    #[case(
        "Ring64Tensor",
        "Ring64Tensor([1, 1])",
        "Ring64Tensor([[1, 2], [3, 4]])",
        "Ring64Tensor([4, 6]) @Host(alice)",
        true
    )]
    #[case(
        "Ring64Tensor",
        "Ring64Tensor([[1, 2], [3, 4]])",
        "Ring64Tensor([[1, 0], [0, 1]])",
        "Ring64Tensor([[1, 2], [3, 4]]) @Host(alice)",
        false
    )]
    #[case(
        "Ring64Tensor",
        "Ring64Tensor([[1, 2], [3, 4]])",
        "Ring64Tensor([1, 1])",
        "Ring64Tensor([3, 7]) @Host(alice)",
        false
    )]
    #[case(
        "Ring64Tensor",
        "Ring64Tensor([1, 1])",
        "Ring64Tensor([[1, 2], [3, 4]])",
        "Ring64Tensor([4, 6]) @Host(alice)",
        false
    )]
    fn test_ring_dot_invocation(
        #[case] type_str: String,
        #[case] x_str: String,
        #[case] y_str: String,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = format!(
            r#"x = Constant{{value={}}} @Host(alice)
        y = Constant{{value={}}} @Host(alice)
        res = RingDot : (Ring64Tensor, Ring64Tensor) -> Ring64Tensor (x, y) @Host(alice)
        output = Output : (Ring64Tensor) -> Ring64Tensor (res) @Host(alice)
        "#,
            x_str, y_str
        );
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        match type_str.as_str() {
            "Ring64Tensor" => {
                let comp_result: HostRing64Tensor =
                    (outputs.get("output").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            "Ring128Tensor" => {
                let comp_result: HostRing128Tensor =
                    (outputs.get("output").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            _ => Err(anyhow::anyhow!("Failed to parse test case type")),
        }
    }

    #[rstest]
    #[case("Ring64", "2", "Ring64Tensor([1, 1]) @Host(alice)", true)]
    #[case("Ring128", "2", "Ring128Tensor([1, 1]) @Host(alice)", true)]
    #[case("Ring64", "2, 1", "Ring64Tensor([[1], [1]]) @Host(alice)", true)]
    #[case("Ring64", "2, 2", "Ring64Tensor([[1, 1], [1, 1]]) @Host(alice)", true)]
    #[case("Ring64", "1, 2", "Ring64Tensor([[1, 1]]) @Host(alice)", true)]
    #[case(
        "Ring128",
        "2, 3",
        "Ring128Tensor([[1, 1, 1], [1, 1, 1]]) @Host(alice)",
        true
    )]
    #[case("Ring64", "2", "Ring64Tensor([1, 1]) @Host(alice)", false)]
    #[case("Ring128", "2", "Ring128Tensor([1, 1]) @Host(alice)", false)]
    #[case("Ring64", "2, 1", "Ring64Tensor([[1], [1]]) @Host(alice)", false)]
    #[case("Ring64", "2, 2", "Ring64Tensor([[1, 1], [1, 1]]) @Host(alice)", false)]
    #[case("Ring64", "1, 2", "Ring64Tensor([[1, 1]]) @Host(alice)", false)]
    #[case(
        "Ring128",
        "2, 3",
        "Ring128Tensor([[1, 1, 1], [1, 1, 1]]) @Host(alice)",
        false
    )]
    fn test_fill(
        #[case] type_str: String,
        #[case] shape_str: String,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = format!(
            r#"shape = Constant{{value=Shape([{shape}])}} @Host(alice)
        res = RingFill {{value = {t}(1)}} : (Shape) -> {t}Tensor (shape) @Host(alice)
        output = Output : ({t}Tensor) -> {t}Tensor (res) @Host(alice)
        "#,
            t = type_str,
            shape = shape_str,
        );
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        match type_str.as_str() {
            "Ring64" => {
                let comp_result: HostRing64Tensor =
                    (outputs.get("output").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            "Ring128" => {
                let comp_result: HostRing128Tensor =
                    (outputs.get("output").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            _ => Err(anyhow::anyhow!("Failed to parse test case type")),
        }
    }

    #[rstest]
    #[case("Ring64Tensor([4, 6]) @Host(alice)", true)]
    #[case("Ring64Tensor([4, 6]) @Host(alice)", false)]
    fn test_ring_sum(
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value=Ring64Tensor([[1, 2], [3, 4]])} @Host(alice)
        r = RingSum {axis = 0}: (Ring64Tensor) -> Ring64Tensor (x) @Host(alice)
        output = Output: (Ring64Tensor) -> Ring64Tensor (r) @Host(alice)
        "#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        let comp_result: HostRing64Tensor = (outputs.get("output").unwrap().clone()).try_into()?;
        assert_eq!(expected_result, comp_result.into());
        Ok(())
    }

    #[rstest]
    #[case("Ring64Tensor", "Ring64Tensor([2, 2]) @Host(alice)", true)]
    #[case("Ring128Tensor", "Ring128Tensor([2, 2]) @Host(alice)", true)]
    #[case("Ring64Tensor", "Ring64Tensor([2, 2]) @Host(alice)", false)]
    #[case("Ring128Tensor", "Ring128Tensor([2, 2]) @Host(alice)", false)]
    fn test_ring_bitwise_ops(
        #[case] type_str: String,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let template_source = r#"x = Constant{value=Ring64Tensor([4, 4])} @Host(alice)
        res = RingShr {amount = 1}: (Ring64Tensor) -> Ring64Tensor (x) @Host(alice)
        output = Output: (Ring64Tensor) -> Ring64Tensor (res) @Host(alice)
        "#;
        let source = template_source.replace("Ring64Tensor", type_str.as_str());
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
            run_async,
        )?;

        match type_str.as_str() {
            "Ring64Tensor" => {
                let comp_result: HostRing64Tensor =
                    (outputs.get("output").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            "Ring128Tensor" => {
                let comp_result: HostRing128Tensor =
                    (outputs.get("output").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            _ => Err(anyhow::anyhow!("Failed to parse test case")),
        }
    }

    #[test]
    fn test_duplicate_session_ids() {
        let source = r#"key = Constant{value=PrfKey(00000000000000000000000000000000)} @Host(alice)
        seed = PrimDeriveSeed {sync_key = [1, 2, 3]}: (PrfKey) -> Seed (key) @Host(alice)
        output = Output: (Seed) -> Seed (seed) @Host(alice)"#;

        let networking: Arc<dyn Send + Sync + AsyncNetworking> =
            Arc::new(LocalAsyncNetworking::default());

        let identity = Identity::from("alice");

        let exec_storage: Arc<dyn Send + Sync + AsyncStorage> =
            Arc::new(LocalAsyncStorage::default());

        let valid_role_assignments: HashMap<Role, Identity> =
            hashmap!(Role::from("alice") => identity.clone());

        let mut executor = AsyncExecutor::default();

        let rt = Runtime::new().unwrap();
        let _guard = rt.enter();

        let moose_session = AsyncSession {
            sid: SessionId::try_from("foobar").unwrap(),
            arguments: hashmap!(),
            networking: Arc::clone(&networking),
            storage: Arc::clone(&exec_storage),
            new_sess: Arc::new(crate::kernels::SyncSession::new(
                SessionId::try_from("foobar").unwrap(),
                hashmap!(),
                valid_role_assignments.clone(),
            )),
            host: Arc::new(Placement::Host(HostPlacement {
                owner: "localhost".into(),
            })),
        };

        let computation: Computation = source.try_into().unwrap();
        let own_identity = identity;

        executor
            .run_computation(
                &computation,
                &valid_role_assignments,
                &own_identity,
                moose_session,
                true,
            )
            .unwrap();

        let moose_session = AsyncSession {
            sid: SessionId::try_from("foobar").unwrap(),
            arguments: hashmap!(),
            networking: Arc::clone(&networking),
            storage: Arc::clone(&exec_storage),
            new_sess: Arc::new(crate::kernels::SyncSession::new(
                SessionId::try_from("foobar").unwrap(),
                hashmap!(),
                valid_role_assignments.clone(),
            )),
            host: Arc::new(Placement::Host(HostPlacement {
                owner: "localhost".into(),
            })),
        };

        let expected =
            Error::SessionAlreadyExists(format!("{}", SessionId::try_from("foobar").unwrap()));

        let res = executor.run_computation(
            &computation,
            &valid_role_assignments,
            &own_identity,
            moose_session,
            true,
        );

        if let Err(e) = res {
            assert_eq!(e.to_string(), expected.to_string());
        } else {
            panic!("expected session already exists error")
        }
    }
}
