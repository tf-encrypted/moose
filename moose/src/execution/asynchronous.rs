use super::{RoleAssignment, RuntimeSession, Session, SetupGeneration};
use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::{Identity, Operands};
use crate::host::{HostPrfKey, HostString};
use crate::kernels::{DispatchKernel, Kernel};
use crate::networking::{AsyncNetworking, LocalAsyncNetworking};
use crate::replicated::{RepSetup, ReplicatedPlacement};
use crate::storage::{AsyncStorage, LocalAsyncStorage};
use futures::future::{Map, Shared};
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use tokio::sync::oneshot;

pub(crate) type AsyncTask = tokio::task::JoinHandle<Result<()>>;

pub type AsyncNetworkingImpl = Arc<dyn AsyncNetworking + Send + Sync>;

pub type AsyncStorageImpl = Arc<dyn AsyncStorage + Send + Sync>;

pub struct AsyncSessionHandle {
    tasks: FuturesUnordered<AsyncTask>,
}

impl AsyncSessionHandle {
    pub async fn join_on_first_error(mut self) -> anyhow::Result<()> {
        use crate::error::Error::{OperandUnavailable, ResultUnused};

        let mut maybe_error = None;
        while let Some(x) = self.tasks.next().await {
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
                            maybe_error = Some(Err(anyhow::Error::from(e)));
                            break;
                        }
                    }
                }
                Err(e) => {
                    if e.is_cancelled() {
                        continue;
                    } else if e.is_panic() {
                        maybe_error = Some(Err(anyhow::Error::from(e)));
                        break;
                    }
                }
            }
        }

        if let Some(e) = maybe_error {
            for task in self.tasks.iter_mut() {
                task.abort();
            }
            e
        } else {
            Ok(())
        }
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
    pub tasks: Arc<Mutex<Option<FuturesUnordered<AsyncTask>>>>,
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
            tasks: Arc::new(Mutex::new(Some(Default::default()))),
        }
    }

    /// Adds a task into the specified collection of tasks.
    ///
    /// The collection is usually a `&sess.tasks`. This is an associated function instead of a method due to
    /// the fact that the current kernels move the cloned session into the created task. To avoid cloning the
    /// entire session, this function expects just a reference to an Arc-cloned session's tasks.
    pub(crate) fn add_task(
        session_tasks: &Arc<Mutex<Option<FuturesUnordered<AsyncTask>>>>,
        task: AsyncTask,
    ) -> Result<()> {
        let tasks_guard = session_tasks.lock().map_err(|e| {
            Error::MalformedEnvironment(format!(
                "Failed to obtain a lock to the tasks in the session. {}",
                e
            ))
        })?;
        tasks_guard
            .as_ref()
            .map(|tasks| tasks.push(task))
            .ok_or_else(|| {
                Error::MalformedEnvironment(
                    "Impossible to add a task. Session has been already converted into a handle"
                        .to_string(),
                )
            })
    }

    /// Consumes self and return a handle with the tasks crated in the session.
    ///
    /// Errors if the session has been already converted into a handle.
    pub fn into_handle(self) -> Result<AsyncSessionHandle> {
        let mut tasks_guard = self.tasks.lock().map_err(|e| {
            Error::MalformedEnvironment(format!(
                "Failed to obtain a lock to the tasks in the session. {}",
                e
            ))
        })?;
        let tasks = tasks_guard.take().ok_or_else(|| {
            Error::MalformedEnvironment(
                "Session has been already converted into a handle".to_string(),
            )
        })?;
        Ok(AsyncSessionHandle { tasks })
    }
}

impl AsyncSession {
    fn storage_load(
        &self,
        op: &LoadOp,
        plc: &Placement,
        operands: Operands<AsyncValue>,
    ) -> Result<AsyncValue> {
        assert_eq!(operands.len(), 2);

        if let Placement::Host(_) = plc {
            let storage = Arc::clone(&self.storage);
            let session_id = self.session_id.clone();
            let expected_ty = op.sig.ret();

            let (sender, receiver) = new_channel();
            let task = tokio::spawn(async move {
                let mut operands = operands;

                let query: HostString = operands
                    .pop()
                    .unwrap()
                    .await
                    .map_err(map_receive_error)?
                    .try_into()?;

                let key: HostString = operands
                    .pop()
                    .unwrap()
                    .await
                    .map_err(map_receive_error)?
                    .try_into()?;

                let value: Value = storage
                    .load(&key.0, &session_id, Some(expected_ty), &query.0)
                    .await?;

                if value.ty() != expected_ty {
                    return Err(Error::TypeMismatch {
                        expected: expected_ty.to_string(),
                        found: value.ty(),
                    });
                }

                // TODO(Morten) check or fill-in placement?

                map_send_result(sender.send(value))?;
                Ok(())
            });
            Self::add_task(&self.tasks, task)?;
            Ok(receiver)
        } else {
            Err(Error::UnimplementedOperator(
                "Load is only supported on host placements".to_string(),
            ))
        }
    }

    fn storage_save(&self, plc: &Placement, operands: Operands<AsyncValue>) -> Result<AsyncValue> {
        assert_eq!(operands.len(), 2);

        if let Placement::Host(plc) = plc {
            let session_id = self.session_id.clone();
            let storage = Arc::clone(&self.storage);
            let unit = Value::from(HostUnit(plc.clone()));

            let (sender, receiver) = new_channel();
            let task = tokio::spawn(async move {
                let mut operands = operands;

                let x: Value = operands.pop().unwrap().await.map_err(map_receive_error)?;

                let key: HostString = operands
                    .pop()
                    .unwrap()
                    .await
                    .map_err(map_receive_error)?
                    .try_into()?;

                storage.save(&key.0, &session_id, &x).await?;

                map_send_result(sender.send(unit))?;
                Ok(())
            });
            Self::add_task(&self.tasks, task)?;
            Ok(receiver)
        } else {
            Err(Error::UnimplementedOperator(
                "Save is only supported on host placements".to_string(),
            ))
        }
    }

    fn networking_receive(
        &self,
        op: &ReceiveOp,
        plc: &Placement,
        operands: Operands<AsyncValue>,
    ) -> Result<AsyncValue> {
        assert_eq!(operands.len(), 0);

        if let Placement::Host(_) = plc {
            let networking_sender = self.find_role_assignment(&op.sender)?.clone();
            let session_id = self.session_id.clone();
            let rendezvous_key = op.rendezvous_key.clone();
            let networking = Arc::clone(&self.networking);
            let expected_ty = op.sig.ret();

            let (sender, receiver) = new_channel();
            let task = tokio::spawn(async move {
                let value = networking
                    .receive(&networking_sender, &rendezvous_key, &session_id)
                    .await?;

                if value.ty() != expected_ty {
                    return Err(Error::TypeMismatch {
                        expected: expected_ty.to_string(),
                        found: value.ty(),
                    });
                }

                // TODO(Morten) check or fill-in placement?

                map_send_result(sender.send(value))?;
                Ok(())
            });
            Self::add_task(&self.tasks, task)?;
            Ok(receiver)
        } else {
            Err(Error::UnimplementedOperator(
                "Receive is only supported on host placements".to_string(),
            ))
        }
    }

    fn networking_send(
        &self,
        op: &SendOp,
        plc: &Placement,
        operands: Operands<AsyncValue>,
    ) -> Result<AsyncValue> {
        assert_eq!(operands.len(), 1);

        if let Placement::Host(plc) = plc {
            let networking_receiver = self.find_role_assignment(&op.receiver)?.clone();
            let session_id = self.session_id.clone();
            let rendezvous_key = op.rendezvous_key.clone();
            let networking = Arc::clone(&self.networking);
            let unit = Value::from(HostUnit(plc.clone()));

            let (sender, receiver) = new_channel();
            let task = tokio::spawn(async move {
                let mut operands = operands;

                let value = operands.pop().unwrap().await.map_err(map_receive_error)?;

                networking
                    .send(&value, &networking_receiver, &rendezvous_key, &session_id)
                    .await?;

                map_send_result(sender.send(unit))?;
                Ok(())
            });
            Self::add_task(&self.tasks, task)?;
            Ok(receiver)
        } else {
            Err(Error::UnimplementedOperator(
                "Send is only supported on host placements".to_string(),
            ))
        }
    }
}

pub type AsyncValue = Receiver;

type Sender = oneshot::Sender<Value>;

type Receiver = Shared<
    Map<
        oneshot::Receiver<Value>,
        fn(anyhow::Result<Value, oneshot::error::RecvError>) -> anyhow::Result<Value, ()>,
    >,
>;

fn map_send_result(res: std::result::Result<(), Value>) -> std::result::Result<(), Error> {
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

fn map_receive_error<T>(_: T) -> Error {
    tracing::debug!("Failed to receive on channel, sender was dropped");
    Error::OperandUnavailable
}

fn new_channel() -> (Sender, Receiver) {
    use futures::FutureExt;

    fn remove_err<T, E>(r: std::result::Result<T, E>) -> std::result::Result<T, ()> {
        r.map_err(|_| ())
    }

    let (sender, receiver) = tokio::sync::oneshot::channel();
    let shared_receiver = receiver.map(remove_err as fn(_) -> _).shared();
    (sender, shared_receiver)
}

impl DispatchKernel<AsyncSession, Value> for Operator {
    fn compile(&self, plc: &Placement) -> Result<Kernel<AsyncSession, Value>> {
        use Operator::*;
        match self {
            // These must be handled elsewhere
            Load(_) => unimplemented!(),
            Save(_) => unimplemented!(),
            Send(_) => unimplemented!(),
            Receive(_) => unimplemented!(),
            Abs(op) => DispatchKernel::compile(op, plc),
            Add(op) => DispatchKernel::compile(op, plc),
            AdtToRep(op) => DispatchKernel::compile(op, plc),
            AddN(op) => DispatchKernel::compile(op, plc),
            And(op) => DispatchKernel::compile(op, plc),
            Argmax(op) => DispatchKernel::compile(op, plc),
            AtLeast2D(op) => DispatchKernel::compile(op, plc),
            BitCompose(op) => DispatchKernel::compile(op, plc),
            BitDecompose(op) => DispatchKernel::compile(op, plc),
            BitExtract(op) => DispatchKernel::compile(op, plc),
            Broadcast(op) => DispatchKernel::compile(op, plc),
            Cast(op) => DispatchKernel::compile(op, plc),
            Concat(op) => DispatchKernel::compile(op, plc),
            Constant(op) => DispatchKernel::compile(op, plc),
            Decrypt(op) => DispatchKernel::compile(op, plc),
            Demirror(op) => DispatchKernel::compile(op, plc),
            DeriveSeed(op) => DispatchKernel::compile(op, plc),
            Dot(op) => DispatchKernel::compile(op, plc),
            Diag(op) => DispatchKernel::compile(op, plc),
            Div(op) => DispatchKernel::compile(op, plc),
            Equal(op) => DispatchKernel::compile(op, plc),
            EqualZero(op) => DispatchKernel::compile(op, plc),
            Exp(op) => DispatchKernel::compile(op, plc),
            ExpandDims(op) => DispatchKernel::compile(op, plc),
            Fill(op) => DispatchKernel::compile(op, plc),
            FixedpointDecode(op) => DispatchKernel::compile(op, plc),
            FixedpointEncode(op) => DispatchKernel::compile(op, plc),
            Greater(op) => DispatchKernel::compile(op, plc),
            Identity(op) => DispatchKernel::compile(op, plc),
            Index(op) => DispatchKernel::compile(op, plc),
            IndexAxis(op) => DispatchKernel::compile(op, plc),
            Input(op) => DispatchKernel::compile(op, plc),
            Inverse(op) => DispatchKernel::compile(op, plc),
            Less(op) => DispatchKernel::compile(op, plc),
            Log(op) => DispatchKernel::compile(op, plc),
            Log2(op) => DispatchKernel::compile(op, plc),
            Maximum(op) => DispatchKernel::compile(op, plc),
            Mean(op) => DispatchKernel::compile(op, plc),
            Mirror(op) => DispatchKernel::compile(op, plc),
            Msb(op) => DispatchKernel::compile(op, plc),
            Mul(op) => DispatchKernel::compile(op, plc),
            Mux(op) => DispatchKernel::compile(op, plc),
            Neg(op) => DispatchKernel::compile(op, plc),
            Ones(op) => DispatchKernel::compile(op, plc),
            Or(op) => DispatchKernel::compile(op, plc),
            Pow2(op) => DispatchKernel::compile(op, plc),
            PrfKeyGen(op) => DispatchKernel::compile(op, plc),
            Relu(op) => DispatchKernel::compile(op, plc),
            Reshape(op) => DispatchKernel::compile(op, plc),
            Reveal(op) => DispatchKernel::compile(op, plc),
            RepToAdt(op) => DispatchKernel::compile(op, plc),
            RingFixedpointArgmax(op) => DispatchKernel::compile(op, plc),
            RingFixedpointDecode(op) => DispatchKernel::compile(op, plc),
            RingFixedpointEncode(op) => DispatchKernel::compile(op, plc),
            RingFixedpointExp(op) => DispatchKernel::compile(op, plc),
            RingFixedpointMean(op) => DispatchKernel::compile(op, plc),
            RingInject(op) => DispatchKernel::compile(op, plc),
            Sample(op) => DispatchKernel::compile(op, plc),
            SampleSeeded(op) => DispatchKernel::compile(op, plc),
            Shape(op) => DispatchKernel::compile(op, plc),
            Share(op) => DispatchKernel::compile(op, plc),
            Shl(op) => DispatchKernel::compile(op, plc),
            ShlDim(op) => DispatchKernel::compile(op, plc),
            Shr(op) => DispatchKernel::compile(op, plc),
            Sigmoid(op) => DispatchKernel::compile(op, plc),
            Sign(op) => DispatchKernel::compile(op, plc),
            Slice(op) => DispatchKernel::compile(op, plc),
            Softmax(op) => DispatchKernel::compile(op, plc),
            Sqrt(op) => DispatchKernel::compile(op, plc),
            Squeeze(op) => DispatchKernel::compile(op, plc),
            Sub(op) => DispatchKernel::compile(op, plc),
            Sum(op) => DispatchKernel::compile(op, plc),
            Transpose(op) => DispatchKernel::compile(op, plc),
            TruncPr(op) => DispatchKernel::compile(op, plc),
            Output(op) => DispatchKernel::compile(op, plc),
            Xor(op) => DispatchKernel::compile(op, plc),
            Zeros(op) => DispatchKernel::compile(op, plc),
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
        let kernel: Kernel<AsyncSession, _> = match op {
            // Kernels that take advantage of async
            Load(op) => return self.storage_load(op, plc, operands),
            Save(_) => return self.storage_save(plc, operands),
            Send(op) => return self.networking_send(op, plc, operands),
            Receive(op) => return self.networking_receive(op, plc, operands),
            // Compute kernels
            op => DispatchKernel::compile(op, plc),
        }?;

        let (sender, receiver) = new_channel();
        let sess = self.clone();
        let plc = plc.clone();

        let task = match kernel {
            Kernel::Nullary { closure } => {
                assert_eq!(operands.len(), 0);
                tokio::spawn(async move {
                    let y: Value = closure(&sess, &plc)?;
                    map_send_result(sender.send(y))?;
                    Ok(())
                })
            }
            Kernel::Unary { closure } => {
                assert_eq!(operands.len(), 1);
                tokio::spawn(async move {
                    let mut operands = operands;
                    let x0: Value = operands.pop().unwrap().await.map_err(map_receive_error)?;
                    let y: Value = closure(&sess, &plc, x0)?;
                    map_send_result(sender.send(y))?;
                    Ok(())
                })
            }
            Kernel::Binary { closure } => {
                assert_eq!(operands.len(), 2);
                tokio::spawn(async move {
                    let mut operands = operands;
                    let x1: Value = operands.pop().unwrap().await.map_err(map_receive_error)?;
                    let x0: Value = operands.pop().unwrap().await.map_err(map_receive_error)?;
                    let y: Value = closure(&sess, &plc, x0, x1)?;
                    map_send_result(sender.send(y))?;
                    Ok(())
                })
            }
            Kernel::Ternary { closure } => {
                assert_eq!(operands.len(), 3);
                tokio::spawn(async move {
                    let mut operands = operands;
                    let x2: Value = operands.pop().unwrap().await.map_err(map_receive_error)?;
                    let x1: Value = operands.pop().unwrap().await.map_err(map_receive_error)?;
                    let x0: Value = operands.pop().unwrap().await.map_err(map_receive_error)?;
                    let y: Value = closure(&sess, &plc, x0, x1, x2)?;
                    map_send_result(sender.send(y))?;
                    Ok(())
                })
            }
            Kernel::Variadic { closure } => tokio::spawn(async move {
                let operands = futures::future::join_all(operands).await;
                let xs: std::result::Result<Operands<Value>, _> = operands.into_iter().collect();
                let xs = xs.map_err(map_receive_error)?;
                let y: Value = closure(&sess, &plc, xs)?;
                map_send_result(sender.send(y))?;
                Ok(())
            }),
        };
        Self::add_task(&self.tasks, task)?;
        Ok(receiver)
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

            session_handles.push(moose_session.into_handle()?)
        }

        let mut futures: FuturesUnordered<_> = session_handles
            .into_iter()
            .map(|h| h.join_on_first_error())
            .collect();
        while let Some(result) = rt.block_on(futures.next()) {
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
