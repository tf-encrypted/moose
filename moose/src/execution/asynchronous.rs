use super::{RoleAssignment, RuntimeSession, Session, SetupGeneration};
use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::{Identity, Operands};
use crate::host::{HostPlacement, HostPrfKey, HostString};
use crate::kernels::{DispatchKernel, Kernel};
use crate::kernels::{NgDispatchKernel, NgKernel};
use crate::networking::{AsyncNetworking, LocalAsyncNetworking};
use crate::replicated::{RepSetup, ReplicatedPlacement};
use crate::storage::{AsyncStorage, LocalAsyncStorage};
use futures::future::{Map, Shared};
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::sync::{Arc, RwLock};
use tokio::runtime::Runtime;
use tokio::sync::oneshot;

pub type AsyncValue = crate::execution::AsyncReceiver;

pub(crate) type AsyncSender = oneshot::Sender<Value>;

pub(crate) type AsyncReceiver = Shared<
    Map<
        oneshot::Receiver<Value>,
        fn(anyhow::Result<Value, oneshot::error::RecvError>) -> anyhow::Result<Value, ()>,
    >,
>;

pub(crate) type AsyncTask = tokio::task::JoinHandle<Result<()>>;

pub type AsyncNetworkingImpl = Arc<dyn AsyncNetworking + Send + Sync>;

pub type AsyncStorageImpl = Arc<dyn AsyncStorage + Send + Sync>;

pub(crate) fn map_send_result(
    res: std::result::Result<(), Value>,
) -> std::result::Result<(), Error> {
    match res {
        Ok(_) => Ok(()),
        Err(val) => {
            if val.ty() == Ty::HostUnit {
                // ignoring unit value is okay
                Ok(())
            } else {
                Err(Error::ResultUnused)
            }
        }
    }
}

pub(crate) fn map_receive_error<T>(_: T) -> Error {
    tracing::debug!("Failed to receive on channel, sender was dropped");
    Error::OperandUnavailable
}

pub struct AsyncSessionHandle {
    pub(crate) tasks: Arc<RwLock<Vec<crate::execution::AsyncTask>>>,
}

impl AsyncSessionHandle {
    pub fn for_session(session: &AsyncSession) -> Self {
        AsyncSessionHandle {
            tasks: Arc::clone(&session.tasks),
        }
    }

    pub async fn join_on_first_error(self) -> anyhow::Result<()> {
        use crate::error::Error::{OperandUnavailable, ResultUnused};
        // use futures::StreamExt;

        let mut tasks_guard = self.tasks.write().unwrap();
        // TODO (lvorona): should really find a way to use FuturesUnordered here
        // let mut tasks = (*tasks_guard)
        //     .into_iter()
        //     .collect::<futures::stream::FuturesUnordered<_>>();

        let mut tasks = tasks_guard.iter_mut();

        while let Some(x) = tasks.next() {
            let x = x.await;
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
                            for task in tasks {
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
                        for task in tasks {
                            task.abort();
                        }
                        return Err(anyhow::Error::from(e));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Session object for asynchronous execution.
#[derive(Clone)]
pub struct AsyncSession {
    pub session_id: SessionId,
    pub arguments: Arc<HashMap<String, Value>>,
    pub role_assignments: Arc<HashMap<Role, Identity>>,
    pub networking: AsyncNetworkingImpl,
    pub storage: AsyncStorageImpl,
    pub(crate) tasks: Arc<RwLock<Vec<crate::execution::AsyncTask>>>,
}

impl AsyncSession {
    pub fn new(
        session_id: SessionId,
        arguments: HashMap<String, Value>,
        role_assignments: HashMap<Role, Identity>,
        networking: AsyncNetworkingImpl,
        storage: AsyncStorageImpl,
    ) -> Self {
        AsyncSession {
            session_id,
            arguments: Arc::new(arguments),
            role_assignments: Arc::new(role_assignments),
            networking,
            storage,
            tasks: Default::default(),
        }
    }
}

impl AsyncSession {
    fn storage_load(
        &self,
        op: &LoadOp,
        _plc: &HostPlacement,
        operands: Operands<AsyncValue>,
    ) -> Result<AsyncValue> {
        use std::convert::TryInto;

        assert_eq!(operands.len(), 2);
        let sess = self.clone();
        let type_hint = Some(op.sig.ret());

        let (sender, receiver) = new_async_value();
        let task = tokio::spawn(async move {
            let operands = futures::future::join_all(operands).await;
            let key: HostString = operands
                .get(0)
                .ok_or_else(|| Error::MalformedEnvironment(format!("Argument {} is missing", 0)))?
                .clone()
                .map_err(map_receive_error)?
                .try_into()?;
            let query: HostString = operands
                .get(1)
                .ok_or_else(|| Error::MalformedEnvironment(format!("Argument {} is missing", 1)))?
                .clone()
                .map_err(map_receive_error)?
                .try_into()?;

            let value: Value = sess
                .storage
                .load(&key.0, &sess.session_id, type_hint, &query.0)
                .await?;
            // TODO: Hmm, placement of a Value does not work like this... But perhaps it should?
            // let value = plc.place(&sess, value);
            map_send_result(sender.send(value))?;
            Ok(())
        });
        let mut tasks = self.tasks.write().unwrap();
        tasks.push(task);

        Ok(receiver)
    }

    fn storage_save(
        &self,
        plc: &HostPlacement,
        operands: Operands<AsyncValue>,
    ) -> Result<AsyncValue> {
        use std::convert::TryInto;

        assert_eq!(operands.len(), 2);
        let sess = self.clone(); // TODO(Morten) avoid clone
        let plc = plc.clone();

        let (sender, receiver) = new_async_value();
        let task = tokio::spawn(async move {
            let operands = futures::future::join_all(operands).await;
            let key: HostString = operands
                .get(0)
                .ok_or_else(|| Error::MalformedEnvironment(format!("Argument {} is missing", 0)))?
                .clone()
                .map_err(map_receive_error)?
                .try_into()?;
            let x: Value = operands
                .get(1)
                .ok_or_else(|| Error::MalformedEnvironment(format!("Argument {} is missing", 1)))?
                .clone()
                .map_err(map_receive_error)?;

            sess.storage.save(&key.0, &sess.session_id, &x).await?;

            let result = HostUnit(plc);
            map_send_result(sender.send(result.into()))?;
            Ok(())
        });
        let mut tasks = self.tasks.write().unwrap();
        tasks.push(task);

        Ok(receiver)
    }

    fn networking_receive(
        &self,
        op: ReceiveOp,
        _plc: &HostPlacement,
        operands: Operands<AsyncValue>,
    ) -> Result<AsyncValue> {
        assert_eq!(operands.len(), 0);
        let sess = self.clone();

        let (sender, receiver) = new_async_value();
        let task = tokio::spawn(async move {
            let net_sender = sess.find_role_assignment(&op.sender)?;

            let value = sess
                .networking
                .receive(net_sender, &op.rendezvous_key, &sess.session_id)
                .await?;
            // TODO: Hmm, placement of a Value does not work like this... But perhaps it should?
            // let value = plc.place(&sess, value);
            map_send_result(sender.send(value))?;
            Ok(())
        });
        let mut tasks = self.tasks.write().unwrap();
        tasks.push(task);

        Ok(receiver)
    }

    fn networking_send(
        &self,
        op: SendOp,
        plc: &HostPlacement,
        operands: Operands<AsyncValue>,
    ) -> Result<AsyncValue> {
        assert_eq!(operands.len(), 1);

        let sess = self.clone(); // TODO(Morten) avoid
        let plc = plc.clone();

        let (sender, receiver) = new_async_value();
        let task = tokio::spawn(async move {
            let receiver = sess.find_role_assignment(&op.receiver)?;
            let operands = futures::future::join_all(operands).await;
            let x = operands
                .get(0)
                .ok_or_else(|| Error::MalformedEnvironment(format!("Argument {} is missing", 0)))?
                .clone()
                .map_err(map_receive_error)?;

            sess.networking
                .send(&x, receiver, &op.rendezvous_key, &sess.session_id)
                .await?;

            let result = HostUnit(plc);
            map_send_result(sender.send(result.into()))?;
            Ok(())
        });
        let mut tasks = self.tasks.write().unwrap();
        tasks.push(task);

        Ok(receiver)
    }
}

pub(crate) fn new_async_value() -> (AsyncSender, AsyncReceiver) {
    // TODO(Morten) make second attempt at inlining
    use futures::FutureExt;
    fn remove_err<T, E>(r: std::result::Result<T, E>) -> std::result::Result<T, ()> {
        r.map_err(|_| ())
    }

    let (sender, receiver) = tokio::sync::oneshot::channel();
    let shared_receiver = receiver.map(remove_err as fn(_) -> _).shared();
    (sender, shared_receiver)
}

impl DispatchKernel<AsyncSession> for SendOp {
    fn compile(&self, plc: &Placement) -> Result<Kernel<AsyncSession>> {
        if let Placement::Host(plc) = plc {
            let plc = plc.clone();
            let op = self.clone();
            Ok(Box::new(move |sess, operands| {
                sess.networking_send(op.clone(), &plc, operands)
            }))
        } else {
            unimplemented!()
        }
    }
}

impl DispatchKernel<AsyncSession> for ReceiveOp {
    fn compile(&self, plc: &Placement) -> Result<Kernel<AsyncSession>> {
        if let Placement::Host(plc) = plc {
            let plc = plc.clone();
            let op = self.clone();
            Ok(Box::new(move |sess, operands| {
                sess.networking_receive(op.clone(), &plc, operands)
            }))
        } else {
            unimplemented!()
        }
    }
}

impl DispatchKernel<AsyncSession> for Operator {
    fn compile(&self, plc: &Placement) -> Result<Kernel<AsyncSession>> {
        use Operator::*;
        match self {
            // these must be handled elsewhere by AsyncSession
            Load(_) | Save(_) => unimplemented!(),
            Send(op) => DispatchKernel::compile(op, plc),
            Receive(op) => DispatchKernel::compile(op, plc),
            PrfKeyGen(op) => DispatchKernel::compile(op, plc),
            Fill(op) => DispatchKernel::compile(op, plc),
            Constant(op) => DispatchKernel::compile(op, plc),
            Input(op) => DispatchKernel::compile(op, plc),
            Output(op) => DispatchKernel::compile(op, plc),
            _ => unimplemented!(),
        }
    }
}

impl Session for AsyncSession {
    type Value = AsyncValue;
    fn execute(
        &self,
        op: &Operator,
        plc: &Placement,
        operands: Operands<Self::Value>,
    ) -> Result<Self::Value> {
        use Operator::*;
        use Placement::*;
        let kernel: NgKernel<AsyncSession, _> = match op {
            // The kernels that are doing funny things to the async context, such as awaiting for more than their inputs.
            Load(op) => {
                if let Host(plc) = plc {
                    return self.storage_load(op, plc, operands);
                } else {
                    unimplemented!()
                }
            }
            Save(_) => {
                if let Host(plc) = plc {
                    return self.storage_save(plc, operands);
                } else {
                    unimplemented!()
                }
            }
            Abs(op) => NgDispatchKernel::compile(op, plc),
            Add(op) => NgDispatchKernel::compile(op, plc),
            AdtToRep(op) => NgDispatchKernel::compile(op, plc),
            AddN(op) => NgDispatchKernel::compile(op, plc),
            And(op) => NgDispatchKernel::compile(op, plc),
            Argmax(op) => NgDispatchKernel::compile(op, plc),
            AtLeast2D(op) => NgDispatchKernel::compile(op, plc),
            BitCompose(op) => NgDispatchKernel::compile(op, plc),
            BitDecompose(op) => NgDispatchKernel::compile(op, plc),
            BitExtract(op) => NgDispatchKernel::compile(op, plc),
            Broadcast(op) => NgDispatchKernel::compile(op, plc),
            Cast(op) => NgDispatchKernel::compile(op, plc),
            Concat(op) => NgDispatchKernel::compile(op, plc),
            Decrypt(op) => NgDispatchKernel::compile(op, plc),
            Demirror(op) => NgDispatchKernel::compile(op, plc),
            DeriveSeed(op) => NgDispatchKernel::compile(op, plc),
            Dot(op) => NgDispatchKernel::compile(op, plc),
            Diag(op) => NgDispatchKernel::compile(op, plc),
            Div(op) => NgDispatchKernel::compile(op, plc),
            Equal(op) => NgDispatchKernel::compile(op, plc),
            EqualZero(op) => NgDispatchKernel::compile(op, plc),
            Exp(op) => NgDispatchKernel::compile(op, plc),
            ExpandDims(op) => NgDispatchKernel::compile(op, plc),
            FixedpointDecode(op) => NgDispatchKernel::compile(op, plc),
            FixedpointEncode(op) => NgDispatchKernel::compile(op, plc),
            GreaterThan(op) => NgDispatchKernel::compile(op, plc),
            Identity(op) => NgDispatchKernel::compile(op, plc),
            Index(op) => NgDispatchKernel::compile(op, plc),
            IndexAxis(op) => NgDispatchKernel::compile(op, plc),
            Inverse(op) => NgDispatchKernel::compile(op, plc),
            LessThan(op) => NgDispatchKernel::compile(op, plc),
            Log(op) => NgDispatchKernel::compile(op, plc),
            Log2(op) => NgDispatchKernel::compile(op, plc),
            Maximum(op) => NgDispatchKernel::compile(op, plc),
            Mean(op) => NgDispatchKernel::compile(op, plc),
            Mirror(op) => NgDispatchKernel::compile(op, plc),
            Msb(op) => NgDispatchKernel::compile(op, plc),
            Mul(op) => NgDispatchKernel::compile(op, plc),
            Mux(op) => NgDispatchKernel::compile(op, plc),
            Neg(op) => NgDispatchKernel::compile(op, plc),
            Ones(op) => NgDispatchKernel::compile(op, plc),
            Or(op) => NgDispatchKernel::compile(op, plc),
            Pow2(op) => NgDispatchKernel::compile(op, plc),
            Reshape(op) => NgDispatchKernel::compile(op, plc),
            Reveal(op) => NgDispatchKernel::compile(op, plc),
            RepToAdt(op) => NgDispatchKernel::compile(op, plc),
            RingFixedpointAbs(op) => NgDispatchKernel::compile(op, plc),
            RingFixedpointArgmax(op) => NgDispatchKernel::compile(op, plc),
            RingFixedpointDecode(op) => NgDispatchKernel::compile(op, plc),
            RingFixedpointEncode(op) => NgDispatchKernel::compile(op, plc),
            RingFixedpointMean(op) => NgDispatchKernel::compile(op, plc),
            RingInject(op) => NgDispatchKernel::compile(op, plc),
            Sample(op) => NgDispatchKernel::compile(op, plc),
            SampleSeeded(op) => NgDispatchKernel::compile(op, plc),
            Shape(op) => NgDispatchKernel::compile(op, plc),
            Share(op) => NgDispatchKernel::compile(op, plc),
            Shl(op) => NgDispatchKernel::compile(op, plc),
            ShlDim(op) => NgDispatchKernel::compile(op, plc),
            Shr(op) => NgDispatchKernel::compile(op, plc),
            Sigmoid(op) => NgDispatchKernel::compile(op, plc),
            Sign(op) => NgDispatchKernel::compile(op, plc),
            Slice(op) => NgDispatchKernel::compile(op, plc),
            Softmax(op) => NgDispatchKernel::compile(op, plc),
            Sqrt(op) => NgDispatchKernel::compile(op, plc),
            Squeeze(op) => NgDispatchKernel::compile(op, plc),
            Sub(op) => NgDispatchKernel::compile(op, plc),
            Sum(op) => NgDispatchKernel::compile(op, plc),
            Transpose(op) => NgDispatchKernel::compile(op, plc),
            TruncPr(op) => NgDispatchKernel::compile(op, plc),
            Xor(op) => NgDispatchKernel::compile(op, plc),
            // The regular kernels, which use the dispatch kernel to await for the inputs and are not touching async in their kernels.
            op => {
                let kernel = DispatchKernel::compile(op, plc)?;
                return kernel(self, operands);
            }
        }?;
        match kernel {
            NgKernel::Nullary { closure } => {
                let (sender, receiver) = crate::execution::asynchronous::new_async_value();

                let tasks = std::sync::Arc::clone(&self.tasks);
                let sess = self.clone();
                let plc = plc.clone();

                let task: tokio::task::JoinHandle<crate::error::Result<()>> =
                    tokio::spawn(async move {
                        assert_eq!(operands.len(), 0);
                        let y: Value = closure(&sess, &plc)?;
                        crate::execution::map_send_result(sender.send(y))?;
                        Ok(())
                    });
                let mut tasks = tasks.write().unwrap();
                tasks.push(task);
                Ok(receiver)
            }
            NgKernel::Unary { closure } => {
                let (sender, receiver) = crate::execution::asynchronous::new_async_value();

                let tasks = std::sync::Arc::clone(&self.tasks);
                let sess = self.clone();
                let plc = plc.clone();

                let task: tokio::task::JoinHandle<crate::error::Result<()>> =
                    tokio::spawn(async move {
                        assert_eq!(operands.len(), 1);
                        let mut operands = futures::future::join_all(operands).await;
                        let x0: Value = operands
                            .pop()
                            .unwrap()
                            .map_err(crate::execution::map_receive_error)?;

                        let y: Value = closure(&sess, &plc, x0)?;
                        crate::execution::map_send_result(sender.send(y))?;
                        Ok(())
                    });
                let mut tasks = tasks.write().unwrap();
                tasks.push(task);
                Ok(receiver)
            }
            NgKernel::Binary { closure } => {
                let (sender, receiver) = crate::execution::asynchronous::new_async_value();

                let tasks = std::sync::Arc::clone(&self.tasks);
                let sess = self.clone();
                let plc = plc.clone();

                let task: tokio::task::JoinHandle<crate::error::Result<()>> =
                    tokio::spawn(async move {
                        assert_eq!(operands.len(), 2);
                        let mut operands = futures::future::join_all(operands).await;
                        let x1: Value = operands
                            .pop()
                            .unwrap()
                            .map_err(crate::execution::map_receive_error)?;
                        let x0: Value = operands
                            .pop()
                            .unwrap()
                            .map_err(crate::execution::map_receive_error)?;

                        let y: Value = closure(&sess, &plc, x0, x1)?;
                        crate::execution::map_send_result(sender.send(y))?;
                        Ok(())
                    });
                let mut tasks = tasks.write().unwrap();
                tasks.push(task);
                Ok(receiver)
            }
            NgKernel::Ternary { closure } => {
                let (sender, receiver) = crate::execution::asynchronous::new_async_value();

                let tasks = std::sync::Arc::clone(&self.tasks);
                let sess = self.clone();
                let plc = plc.clone();

                let task: tokio::task::JoinHandle<crate::error::Result<()>> =
                    tokio::spawn(async move {
                        assert_eq!(operands.len(), 3);
                        let mut operands = futures::future::join_all(operands).await;
                        let x2: Value = operands
                            .pop()
                            .unwrap()
                            .map_err(crate::execution::map_receive_error)?;
                        let x1: Value = operands
                            .pop()
                            .unwrap()
                            .map_err(crate::execution::map_receive_error)?;
                        let x0: Value = operands
                            .pop()
                            .unwrap()
                            .map_err(crate::execution::map_receive_error)?;

                        let y: Value = closure(&sess, &plc, x0, x1, x2)?;
                        crate::execution::map_send_result(sender.send(y))?;
                        Ok(())
                    });
                let mut tasks = tasks.write().unwrap();
                tasks.push(task);

                Ok(receiver)
            }
            NgKernel::Variadic { closure } => {
                let (sender, receiver) = crate::execution::asynchronous::new_async_value();

                let tasks = std::sync::Arc::clone(&self.tasks);
                let sess = self.clone();
                let plc = plc.clone();

                let task: tokio::task::JoinHandle<crate::error::Result<()>> =
                    tokio::spawn(async move {
                        let operands = futures::future::join_all(operands).await;
                        let xs: std::result::Result<Operands<Value>, _> =
                            operands.into_iter().collect();
                        let xs = xs.map_err(crate::execution::map_receive_error)?;

                        let y: Value = closure(&sess, &plc, xs)?;
                        crate::execution::map_send_result(sender.send(y))?;
                        Ok(())
                    });
                let mut tasks = tasks.write().unwrap();
                tasks.push(task);

                Ok(receiver)
            }
        }
    }
}

impl SetupGeneration<ReplicatedPlacement> for AsyncSession {
    type Setup = RepSetup<HostPrfKey>;

    fn setup(&self, _plc: &ReplicatedPlacement) -> Result<Arc<Self::Setup>> {
        unimplemented!()
    }
}

impl RuntimeSession for AsyncSession {
    fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    fn find_argument(&self, key: &str) -> Option<Value> {
        self.arguments.get(key).cloned()
    }

    fn find_role_assignment(&self, role: &Role) -> Result<&Identity> {
        self.role_assignments
            .get(role)
            .ok_or_else(|| Error::Networking(format!("Missing role assignment for {}", role)))
    }
}

#[derive(Default)]
pub struct AsyncExecutor {
    session_ids: HashSet<SessionId>,
}

impl AsyncExecutor {
    // After execution the AsyncTasks to block on will be in session.tasks vector.
    pub fn run_computation(
        &mut self,
        computation: &Computation,
        role_assignment: &RoleAssignment,
        own_identity: &Identity,
        session: &AsyncSession,
    ) -> Result<HashMap<String, AsyncValue>> {
        if !self.session_ids.insert(session.session_id.clone()) {
            return Err(Error::SessionAlreadyExists(format!(
                "{}",
                session.session_id
            )));
        }

        // using a Vec instead of eg HashSet here since we can expect it to be very small
        let own_roles: Vec<&Role> = role_assignment
            .iter()
            .filter_map(|(role, identity)| {
                if identity == own_identity {
                    Some(role)
                } else {
                    None
                }
            })
            .collect();

        let own_operations = computation
            .operations
            .iter() // guessing that par_iter won't help here
            .filter(|op| match &op.placement {
                Placement::Additive(plc) => own_roles
                    .iter()
                    .any(|owner| plc.owners.iter().any(|plc_owner| *owner == plc_owner)),
                Placement::Host(plc) => own_roles.iter().any(|owner| *owner == &plc.owner),
                Placement::Mirrored3(plc) => own_roles
                    .iter()
                    .any(|owner| plc.owners.iter().any(|plc_owner| *owner == plc_owner)),
                Placement::Replicated(plc) => own_roles
                    .iter()
                    .any(|owner| plc.owners.iter().any(|plc_owner| *owner == plc_owner)),
            })
            .collect::<Vec<_>>();

        let mut env: HashMap<String, AsyncValue> = HashMap::with_capacity(own_operations.len());
        let mut outputs: HashMap<String, AsyncValue> = HashMap::default();

        for op in own_operations {
            let operands = op
                .inputs
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();
            let value = session
                .execute(&op.kind, &op.placement, operands)
                .map_err(|e| {
                    Error::KernelError(format!("AsyncSession failed due to an error: {:?}", e,))
                })?;
            if matches!(op.kind, Operator::Output(_)) {
                // If it is an output, we need to make sure we capture it for returning.
                outputs.insert(op.name.clone(), value.clone());
            } else {
                // Everything else should be available in the env for other ops to use.
                env.insert(op.name.clone(), value);
            }
        }

        Ok(outputs)
    }
}

pub struct AsyncTestRuntime {
    pub identities: Vec<Identity>,
    pub executors: HashMap<Identity, AsyncExecutor>,
    pub runtime_storage: HashMap<Identity, AsyncStorageImpl>,
    pub networking: AsyncNetworkingImpl,
}

impl AsyncTestRuntime {
    pub fn new(storage_mapping: HashMap<String, HashMap<String, Value>>) -> Self {
        let mut executors: HashMap<Identity, AsyncExecutor> = HashMap::new();
        let networking: AsyncNetworkingImpl = Arc::new(LocalAsyncNetworking::default());
        let mut runtime_storage: HashMap<Identity, AsyncStorageImpl> = HashMap::new();
        let mut identities = Vec::new();
        for (identity_str, storage) in storage_mapping {
            let identity = Identity::from(identity_str.clone()).clone();
            identities.push(identity.clone());
            // TODO handle Result in map predicate instead of `unwrap`
            let storage = storage
                .iter()
                .map(|arg| (arg.0.to_owned(), arg.1.to_owned()))
                .collect::<HashMap<String, Value>>();

            let exec_storage: AsyncStorageImpl = Arc::new(LocalAsyncStorage::from_hashmap(storage));
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
    ) -> Result<HashMap<String, Value>> {
        let mut session_handles: Vec<AsyncSessionHandle> = Vec::new();
        let mut output_futures: HashMap<String, AsyncValue> = HashMap::new();
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

        let session_id = SessionId::random();
        for (own_identity, executor) in self.executors.iter_mut() {
            let moose_session = AsyncSession::new(
                session_id.clone(),
                arguments.clone(),
                valid_role_assignments.clone(),
                Arc::clone(&self.networking),
                Arc::clone(&self.runtime_storage[own_identity]),
            );
            let outputs = executor
                .run_computation(
                    computation,
                    &valid_role_assignments,
                    own_identity,
                    &moose_session,
                )
                .unwrap();

            for (output_name, output_future) in outputs {
                output_futures.insert(output_name, output_future);
            }

            session_handles.push(AsyncSessionHandle::for_session(&moose_session))
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

        Ok(outputs)
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
