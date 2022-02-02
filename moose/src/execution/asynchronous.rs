use super::*;
use crate::error::{Error, Result};
use crate::execution::Identity;
use crate::host::*;
use crate::kernels::DispatchKernel;
use crate::replicated::*;
use crate::types::*;
use std::collections::HashMap;
use std::sync::Arc;

pub type AsyncSender = oneshot::Sender<Value>;

pub type AsyncReceiver = Shared<
    Map<
        oneshot::Receiver<Value>,
        fn(anyhow::Result<Value, oneshot::error::RecvError>) -> anyhow::Result<Value, ()>,
    >,
>;

pub type AsyncTask = tokio::task::JoinHandle<Result<()>>;

pub type AsyncNetworkingImpl = Arc<dyn AsyncNetworking + Send + Sync>;

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

pub struct AsyncSessionHandle {
    pub tasks: Arc<std::sync::RwLock<Vec<crate::execution::AsyncTask>>>,
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

/// Session object for asynchronous execution (in new framework).
#[derive(Clone)]
pub struct AsyncSession {
    pub session_id: SessionId,
    pub arguments: Arc<HashMap<String, Value>>,
    pub role_assignments: Arc<HashMap<Role, Identity>>,
    pub networking: Arc<dyn Send + Sync + crate::networking::AsyncNetworking>,
    pub storage: Arc<dyn Send + Sync + crate::storage::AsyncStorage>,
    pub host: Arc<Placement>,
    // replicated_keys: HashMap<ReplicatedPlacement, ReplicatedSetup>,
    pub tasks: Arc<std::sync::RwLock<Vec<crate::execution::AsyncTask>>>,
}

impl AsyncSession {
    pub fn new(
        session_id: SessionId,
        arguments: HashMap<String, Value>,
        role_assignments: HashMap<Role, Identity>,
        networking: Arc<dyn Send + Sync + crate::networking::AsyncNetworking>,
        storage: Arc<dyn Send + Sync + crate::storage::AsyncStorage>,
        host: Arc<Placement>,
    ) -> Self {
        AsyncSession {
            session_id,
            arguments: Arc::new(arguments),
            role_assignments: Arc::new(role_assignments),
            networking,
            storage,
            host,
            tasks: Default::default(),
        }
    }

    fn storage_load(
        &self,
        op: &LoadOp,
        _plc: &HostPlacement,
        operands: Vec<AsyncValue>,
    ) -> Result<AsyncValue> {
        use std::convert::TryInto;
        assert_eq!(operands.len(), 2);
        let sess = self.clone();
        let op = op.clone();
        // let plc = plc.clone();
        let (sender, result) = crate::computation::new_async_value();
        let tasks = std::sync::Arc::clone(&self.tasks);
        let task: tokio::task::JoinHandle<crate::error::Result<()>> = tokio::spawn(async move {
            let operands = futures::future::join_all(operands).await;
            let key: HostString = operands
                .get(0)
                .ok_or_else(|| {
                    crate::error::Error::MalformedEnvironment(format!("Argument {} is missing", 0))
                })?
                .clone()
                .map_err(crate::execution::map_receive_error)?
                .try_into()?;
            let query: HostString = operands
                .get(1)
                .ok_or_else(|| {
                    crate::error::Error::MalformedEnvironment(format!("Argument {} is missing", 1))
                })?
                .clone()
                .map_err(crate::execution::map_receive_error)?
                .try_into()?;

            let value: Value = sess
                .storage
                .load(&key.0, &sess.session_id, Some(op.sig.ret()), &query.0)
                .await?;
            // TODO: Hmm, placement of a Value does not work like this... But perhaps it should?
            // let value = plc.place(&sess, value);
            crate::execution::map_send_result(sender.send(value))?;
            Ok(())
        });
        let mut tasks = tasks.write().unwrap();
        tasks.push(task);

        Ok(result)
    }

    fn storage_save(
        &self,
        _op: &SaveOp,
        plc: &HostPlacement,
        operands: Vec<AsyncValue>,
    ) -> Result<AsyncValue> {
        use std::convert::TryInto;
        assert_eq!(operands.len(), 2);
        let sess = self.clone();
        let plc = plc.clone();
        let (sender, result) = crate::computation::new_async_value();
        let tasks = std::sync::Arc::clone(&self.tasks);
        let task: tokio::task::JoinHandle<crate::error::Result<()>> = tokio::spawn(async move {
            let operands = futures::future::join_all(operands).await;
            let key: HostString = operands
                .get(0)
                .ok_or_else(|| {
                    crate::error::Error::MalformedEnvironment(format!("Argument {} is missing", 0))
                })?
                .clone()
                .map_err(crate::execution::map_receive_error)?
                .try_into()?;
            let x: Value = operands
                .get(1)
                .ok_or_else(|| {
                    crate::error::Error::MalformedEnvironment(format!("Argument {} is missing", 1))
                })?
                .clone()
                .map_err(crate::execution::map_receive_error)?;

            sess.storage.save(&key.0, &sess.session_id, &x).await?;
            let result = Unit(plc);
            crate::execution::map_send_result(sender.send(result.into()))?;
            Ok(())
        });
        let mut tasks = tasks.write().unwrap();
        tasks.push(task);

        Ok(result)
    }

    fn networking_receive(
        &self,
        op: &ReceiveOp,
        _plc: &HostPlacement,
        operands: Vec<AsyncValue>,
    ) -> Result<AsyncValue> {
        assert_eq!(operands.len(), 0);
        let sess = self.clone();
        let op = op.clone();
        // let plc = plc.clone();
        let (sender, result) = crate::computation::new_async_value();
        let tasks = std::sync::Arc::clone(&self.tasks);
        let task: tokio::task::JoinHandle<crate::error::Result<()>> = tokio::spawn(async move {
            let net_sender = sess.find_role_assignment(&op.sender)?;

            let value: Value = sess
                .networking
                .receive(net_sender, &op.rendezvous_key, &sess.session_id)
                .await?;
            // TODO: Hmm, placement of a Value does not work like this... But perhaps it should?
            // let value = plc.place(&sess, value);
            crate::execution::map_send_result(sender.send(value))?;
            Ok(())
        });
        let mut tasks = tasks.write().unwrap();
        tasks.push(task);

        Ok(result)
    }

    fn networking_send(
        &self,
        op: &SendOp,
        plc: &HostPlacement,
        operands: Vec<AsyncValue>,
    ) -> Result<AsyncValue> {
        assert_eq!(operands.len(), 1);
        let sess = self.clone();
        let plc = plc.clone();
        let op = op.clone();
        let (sender, result) = crate::computation::new_async_value();
        let tasks = std::sync::Arc::clone(&self.tasks);
        let task: tokio::task::JoinHandle<crate::error::Result<()>> = tokio::spawn(async move {
            let receiver = sess.find_role_assignment(&op.receiver)?;
            let operands = futures::future::join_all(operands).await;
            let x: Value = operands
                .get(0)
                .ok_or_else(|| {
                    crate::error::Error::MalformedEnvironment(format!("Argument {} is missing", 0))
                })?
                .clone()
                .map_err(crate::execution::map_receive_error)?;

            sess.networking
                .send(&x, receiver, &op.rendezvous_key, &sess.session_id)
                .await?;
            let result = Unit(plc);
            crate::execution::map_send_result(sender.send(result.into()))?;
            Ok(())
        });
        let mut tasks = tasks.write().unwrap();
        tasks.push(task);

        Ok(result)
    }
}

impl Session for AsyncSession {
    type Value = AsyncValue;
    fn execute(
        &self,
        op: Operator,
        plc: &Placement,
        operands: Vec<Self::Value>,
    ) -> Result<Self::Value> {
        // The kernels that are doing funny things to the async context, such as awaiting for more than their inputs.
        match (&op, plc) {
            (Operator::Load(op), Placement::Host(plc)) => {
                return self.storage_load(op, plc, operands)
            }
            (Operator::Save(op), Placement::Host(plc)) => {
                return self.storage_save(op, plc, operands)
            }
            (Operator::Send(op), Placement::Host(plc)) => {
                return self.networking_send(op, plc, operands)
            }
            (Operator::Receive(op), Placement::Host(plc)) => {
                return self.networking_receive(op, plc, operands)
            }
            _ => (),
        };
        // The regular kernels, which use the dispatch kernel to await for the inputs and are not touching async in their kernels.
        use Operator::*;
        let kernel = match op {
            Shape(op) => DispatchKernel::compile(&op, plc)?,
            Broadcast(op) => DispatchKernel::compile(&op, plc)?,
            RingFill(op) => DispatchKernel::compile(&op, plc)?,
            PrimPrfKeyGen(op) => DispatchKernel::compile(&op, plc)?,
            BitSample(op) => DispatchKernel::compile(&op, plc)?,
            BitSampleSeeded(op) => DispatchKernel::compile(&op, plc)?,
            BitXor(op) => DispatchKernel::compile(&op, plc)?,
            BitAnd(op) => DispatchKernel::compile(&op, plc)?,
            Neg(op) => DispatchKernel::compile(&op, plc)?,
            BitOr(op) => DispatchKernel::compile(&op, plc)?,
            BitExtract(op) => DispatchKernel::compile(&op, plc)?,
            RingSample(op) => DispatchKernel::compile(&op, plc)?,
            RingSampleSeeded(op) => DispatchKernel::compile(&op, plc)?,
            Shl(op) => DispatchKernel::compile(&op, plc)?,
            Shr(op) => DispatchKernel::compile(&op, plc)?,
            RingFixedpointMean(op) => DispatchKernel::compile(&op, plc)?,
            RingFixedpointEncode(op) => DispatchKernel::compile(&op, plc)?,
            RingFixedpointDecode(op) => DispatchKernel::compile(&op, plc)?,
            RingInject(op) => DispatchKernel::compile(&op, plc)?,
            Fill(op) => DispatchKernel::compile(&op, plc)?,
            RepSetup(op) => DispatchKernel::compile(&op, plc)?,
            RepShare(op) => DispatchKernel::compile(&op, plc)?,
            RepReveal(op) => DispatchKernel::compile(&op, plc)?,
            RepTruncPr(op) => DispatchKernel::compile(&op, plc)?,
            RepMsb(op) => DispatchKernel::compile(&op, plc)?,
            Abs(op) => DispatchKernel::compile(&op, plc)?,
            RepToAdt(op) => DispatchKernel::compile(&op, plc)?,
            RepFixedpointMean(op) => DispatchKernel::compile(&op, plc)?,
            Diag(op) => DispatchKernel::compile(&op, plc)?,
            RepSlice(op) => DispatchKernel::compile(&op, plc)?,
            RepBitDec(op) => DispatchKernel::compile(&op, plc)?,
            RepShlDim(op) => DispatchKernel::compile(&op, plc)?,
            AdtFill(op) => DispatchKernel::compile(&op, plc)?,
            AdtReveal(op) => DispatchKernel::compile(&op, plc)?,
            AdtToRep(op) => DispatchKernel::compile(&op, plc)?,
            PrimDeriveSeed(op) => DispatchKernel::compile(&op, plc)?,
            Constant(op) => DispatchKernel::compile(&op, plc)?,
            HostOnes(op) => DispatchKernel::compile(&op, plc)?,
            Input(op) => DispatchKernel::compile(&op, plc)?,
            Output(op) => DispatchKernel::compile(&op, plc)?,
            Load(op) => DispatchKernel::compile(&op, plc)?,
            Save(op) => DispatchKernel::compile(&op, plc)?,
            HostMean(op) => DispatchKernel::compile(&op, plc)?,
            Sqrt(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointEncode(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointDecode(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointTruncPr(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointMean(op) => DispatchKernel::compile(&op, plc)?,
            HostSlice(op) => DispatchKernel::compile(&op, plc)?,
            HostShlDim(op) => DispatchKernel::compile(&op, plc)?,
            Sign(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointOnes(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointConcat(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointMean(op) => DispatchKernel::compile(&op, plc)?,
            HostBitDec(op) => DispatchKernel::compile(&op, plc)?,
            Identity(op) => DispatchKernel::compile(&op, plc)?,
            Cast(op) => DispatchKernel::compile(&op, plc)?,
            Send(op) => DispatchKernel::compile(&op, plc)?,
            Receive(op) => DispatchKernel::compile(&op, plc)?,
            AtLeast2D(op) => DispatchKernel::compile(&op, plc)?,
            Slice(op) => DispatchKernel::compile(&op, plc)?,
            Ones(op) => DispatchKernel::compile(&op, plc)?,
            ExpandDims(op) => DispatchKernel::compile(&op, plc)?,
            Concat(op) => DispatchKernel::compile(&op, plc)?,
            Reshape(op) => DispatchKernel::compile(&op, plc)?,
            Squeeze(op) => DispatchKernel::compile(&op, plc)?,
            Transpose(op) => DispatchKernel::compile(&op, plc)?,
            Dot(op) => DispatchKernel::compile(&op, plc)?,
            Inverse(op) => DispatchKernel::compile(&op, plc)?,
            Add(op) => DispatchKernel::compile(&op, plc)?,
            Sub(op) => DispatchKernel::compile(&op, plc)?,
            Mul(op) => DispatchKernel::compile(&op, plc)?,
            Mean(op) => DispatchKernel::compile(&op, plc)?,
            Sum(op) => DispatchKernel::compile(&op, plc)?,
            Div(op) => DispatchKernel::compile(&op, plc)?,
            AddN(op) => DispatchKernel::compile(&op, plc)?,
            Exp(op) => DispatchKernel::compile(&op, plc)?,
            Mux(op) => DispatchKernel::compile(&op, plc)?,
            Equal(op) => DispatchKernel::compile(&op, plc)?,
            Less(op) => DispatchKernel::compile(&op, plc)?,
            GreaterThan(op) => DispatchKernel::compile(&op, plc)?,
            IndexAxis(op) => DispatchKernel::compile(&op, plc)?,
            Maximum(op) => DispatchKernel::compile(&op, plc)?,
            Softmax(op) => DispatchKernel::compile(&op, plc)?,
            Demirror(op) => DispatchKernel::compile(&op, plc)?,
            Mirror(op) => DispatchKernel::compile(&op, plc)?,
            _ => todo!(),
        };
        kernel(self, operands)
    }

    type ReplicatedSetup = ReplicatedSetup;
    fn replicated_setup(&self, _plc: &ReplicatedPlacement) -> Arc<Self::ReplicatedSetup> {
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
            .ok_or_else(|| Error::Networking(format!("Missing role assignemnt for {}", role)))
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
    ) -> Result<HashMap<String, AsyncReceiver>> {
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

        let mut env: HashMap<String, AsyncValue> = HashMap::default();
        let mut outputs: HashMap<String, AsyncReceiver> = HashMap::default();

        for op in own_operations {
            let operator = op.kind.clone();
            let operands = op
                .inputs
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();
            let value = session
                .execute(operator, &op.placement, operands)
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
    ) -> Result<HashMap<String, Value>> {
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
            let moose_session = AsyncSession::new(
                SessionId::try_from("foobar").unwrap(),
                arguments.clone(),
                valid_role_assignments.clone(),
                Arc::clone(&self.networking),
                Arc::clone(&self.runtime_storage[own_identity]),
                Arc::new(Placement::Host(HostPlacement {
                    owner: own_identity.0.clone().into(),
                })),
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
