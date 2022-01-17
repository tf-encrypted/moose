use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::{Identity, SyncNetworkingImpl, SyncStorageImpl};
use crate::for_all_values;
use crate::host::*;
use crate::mirrored::*;
use crate::networking::LocalSyncNetworking;
use crate::replicated::*;
use crate::storage::LocalSyncStorage;
use crate::types::*;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::rc::Rc;
use std::sync::Arc;

/// General session trait determining basic properties for session objects.
pub trait Session {
    type Value;
    fn execute(
        &self,
        op: Operator,
        plc: &Placement,
        operands: Vec<Self::Value>,
    ) -> Result<Self::Value>;

    type ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> Arc<Self::ReplicatedSetup>;
}

/// Trait for sessions that are intended for run-time use only.
///
/// This trait is used to make a distinct between functionality that may
/// only be executed during run-time as opposed to at compile-time, such
/// as for instance key generation. Moreover, it also offers access to
/// information that is only known at run-time, such as the concrete
/// session id under which execution is happening.
pub trait RuntimeSession: Session {
    fn session_id(&self) -> &SessionId;
    fn find_argument(&self, key: &str) -> Option<Value>;
    fn find_role_assignment(&self, role: &Role) -> Result<&Identity>;
}

/// Session object for synchronous/eager execution (in new framework).
pub struct SyncSession {
    session_id: SessionId,
    replicated_keys: std::sync::RwLock<HashMap<ReplicatedPlacement, Arc<ReplicatedSetup>>>,
    arguments: HashMap<String, Value>,
    role_assignments: HashMap<Role, Identity>,
    storage: SyncStorageImpl,
    networking: SyncNetworkingImpl,
}

impl Default for SyncSession {
    /// Default session should only be used in tests.
    ///
    /// Use new() for the real sessions instead.
    fn default() -> Self {
        SyncSession {
            session_id: SessionId::random(),
            replicated_keys: Default::default(),
            arguments: Default::default(),
            role_assignments: Default::default(),
            storage: Rc::new(LocalSyncStorage::default()),
            networking: Rc::new(LocalSyncNetworking::default()),
        }
    }
}

impl SyncSession {
    pub fn from_session_id(sid: SessionId) -> Self {
        SyncSession {
            session_id: sid,
            replicated_keys: Default::default(),
            arguments: Default::default(),
            role_assignments: Default::default(),
            storage: Rc::new(LocalSyncStorage::default()),
            networking: Rc::new(LocalSyncNetworking::default()),
        }
    }

    pub fn from_storage(
        sid: SessionId,
        arguments: HashMap<String, Value>,
        role_assignments: HashMap<Role, Identity>,
        storage: SyncStorageImpl,
    ) -> Self {
        SyncSession {
            session_id: sid,
            replicated_keys: Default::default(),
            arguments,
            role_assignments,
            storage,
            networking: Rc::new(LocalSyncNetworking::default()),
        }
    }

    pub fn from_networking(
        sid: SessionId,
        arguments: HashMap<String, Value>,
        role_assignments: HashMap<Role, Identity>,
        networking: SyncNetworkingImpl,
    ) -> Self {
        SyncSession {
            session_id: sid,
            replicated_keys: Default::default(),
            arguments,
            role_assignments,
            storage: Rc::new(LocalSyncStorage::default()),
            networking,
        }
    }

    pub fn from_roles<'a>(roles: impl Iterator<Item = &'a Role>) -> Self {
        let own_identity = Identity::from("tester");
        let role_assignment = roles
            .map(|role| (role.clone(), own_identity.clone()))
            .collect();
        SyncSession {
            session_id: SessionId::random(),
            replicated_keys: Default::default(),
            arguments: Default::default(),
            role_assignments: role_assignment,
            storage: Rc::new(LocalSyncStorage::default()),
            networking: Rc::new(LocalSyncNetworking::default()),
        }
    }
}

impl Session for SyncSession {
    type Value = Value;

    fn execute(&self, op: Operator, plc: &Placement, operands: Vec<Value>) -> Result<Value> {
        use Operator::*;
        let kernel_output = match op {
            Shape(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingFill(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            PrimPrfKeyGen(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitSample(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitSampleSeeded(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitXor(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitAnd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitNeg(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitOr(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            BitExtract(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingSample(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingSampleSeeded(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingNeg(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingShl(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingShr(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingSum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingFixedpointMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingFixedpointEncode(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingFixedpointDecode(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RingInject(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Fill(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepSetup(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepShare(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepReveal(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepAnd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepXor(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepNeg(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepTruncPr(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepMsb(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepAbs(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepToAdt(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepFixedpointMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepSum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AddN(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepShl(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Index(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepDiag(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepSlice(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepBitDec(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepBitCompose(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepShlDim(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtShl(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtFill(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtReveal(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AdtToRep(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            PrimDeriveSeed(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AesDecrypt(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Constant(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostOnes(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Input(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Output(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Load(op) => {
                use std::convert::TryInto;
                assert_eq!(operands.len(), 2);
                let key: HostString = operands.get(0).unwrap().clone().try_into()?;
                let query: HostString = operands.get(1).unwrap().clone().try_into()?;
                self.storage
                    .load(&key.0, &self.session_id, Some(op.sig.ret()), &query.0)?
            }
            Save(_) => {
                use std::convert::TryInto;
                assert_eq!(operands.len(), 2);
                let key: HostString = operands.get(0).unwrap().clone().try_into()?;
                let x = operands.get(1).unwrap().clone();
                self.storage.save(&key.0, &self.session_id, &x)?;
                let host = match plc {
                    Placement::Host(host) => host,
                    _ => unimplemented!(
                        "SyncSession does not support running Save on non-host placements yet"
                    ),
                };
                Unit(host.clone()).into()
            }
            HostAtLeast2D(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSqrt(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointEncode(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointDecode(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointDiv(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointTruncPr(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointSum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSlice(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostDiag(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostShlDim(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostDiv(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostExpandDims(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSqueeze(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Sign(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointDiv(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointAtLeast2D(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointOnes(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointConcat(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointExpandDims(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointTranspose(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointInverse(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointSum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostTranspose(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostInverse(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostBitDec(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Identity(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Cast(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Send(op) => {
                assert_eq!(operands.len(), 1);
                let x = operands.get(0).unwrap();
                self.networking.send(
                    x,
                    self.find_role_assignment(&op.receiver)?,
                    &op.rendezvous_key,
                    &self.session_id,
                )?;
                let host = match plc {
                    Placement::Host(host) => host,
                    _ => unimplemented!(
                        "SyncSession does not support running Send on non-host placements yet"
                    ),
                };
                Unit(host.clone()).into()
            }
            Receive(op) => self.networking.receive(
                self.find_role_assignment(&op.sender)?,
                &op.rendezvous_key,
                &self.session_id,
            )?,
            HostReshape(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            AtLeast2D(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            IndexAxis(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Slice(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Ones(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            ExpandDims(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Concat(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Transpose(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Dot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Inverse(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Add(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Sub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Mul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Mean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Neg(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Sum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Div(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            RepEqual(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Mux(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Pow2(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Exp(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Sigmoid(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Less(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            GreaterThan(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Maximum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Demirror(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Mirror(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
        };
        Ok(kernel_output)
    }

    type ReplicatedSetup = ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> Arc<Self::ReplicatedSetup> {
        let mut replicated_keys = self.replicated_keys.write().unwrap();
        let setup = replicated_keys
            .entry(plc.clone())
            .or_insert_with(|| Arc::new(plc.gen_setup(self)));
        Arc::clone(setup)
    }
}

impl RuntimeSession for SyncSession {
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
            RingFill(op) => DispatchKernel::compile(&op, plc)?,
            PrimPrfKeyGen(op) => DispatchKernel::compile(&op, plc)?,
            BitSample(op) => DispatchKernel::compile(&op, plc)?,
            BitSampleSeeded(op) => DispatchKernel::compile(&op, plc)?,
            BitXor(op) => DispatchKernel::compile(&op, plc)?,
            BitAnd(op) => DispatchKernel::compile(&op, plc)?,
            BitNeg(op) => DispatchKernel::compile(&op, plc)?,
            BitOr(op) => DispatchKernel::compile(&op, plc)?,
            BitExtract(op) => DispatchKernel::compile(&op, plc)?,
            RingSample(op) => DispatchKernel::compile(&op, plc)?,
            RingSampleSeeded(op) => DispatchKernel::compile(&op, plc)?,
            RingAdd(op) => DispatchKernel::compile(&op, plc)?,
            RingSub(op) => DispatchKernel::compile(&op, plc)?,
            RingMul(op) => DispatchKernel::compile(&op, plc)?,
            RingDot(op) => DispatchKernel::compile(&op, plc)?,
            RingNeg(op) => DispatchKernel::compile(&op, plc)?,
            RingShl(op) => DispatchKernel::compile(&op, plc)?,
            RingShr(op) => DispatchKernel::compile(&op, plc)?,
            RingSum(op) => DispatchKernel::compile(&op, plc)?,
            RingFixedpointMean(op) => DispatchKernel::compile(&op, plc)?,
            RingFixedpointEncode(op) => DispatchKernel::compile(&op, plc)?,
            RingFixedpointDecode(op) => DispatchKernel::compile(&op, plc)?,
            RingInject(op) => DispatchKernel::compile(&op, plc)?,
            Fill(op) => DispatchKernel::compile(&op, plc)?,
            RepSetup(op) => DispatchKernel::compile(&op, plc)?,
            RepShare(op) => DispatchKernel::compile(&op, plc)?,
            RepReveal(op) => DispatchKernel::compile(&op, plc)?,
            RepAdd(op) => DispatchKernel::compile(&op, plc)?,
            RepSub(op) => DispatchKernel::compile(&op, plc)?,
            RepMul(op) => DispatchKernel::compile(&op, plc)?,
            RepDot(op) => DispatchKernel::compile(&op, plc)?,
            RepTruncPr(op) => DispatchKernel::compile(&op, plc)?,
            RepMsb(op) => DispatchKernel::compile(&op, plc)?,
            RepNeg(op) => DispatchKernel::compile(&op, plc)?,
            RepAbs(op) => DispatchKernel::compile(&op, plc)?,
            RepToAdt(op) => DispatchKernel::compile(&op, plc)?,
            RepFixedpointMean(op) => DispatchKernel::compile(&op, plc)?,
            RepSum(op) => DispatchKernel::compile(&op, plc)?,
            RepShl(op) => DispatchKernel::compile(&op, plc)?,
            RepDiag(op) => DispatchKernel::compile(&op, plc)?,
            RepSlice(op) => DispatchKernel::compile(&op, plc)?,
            RepBitDec(op) => DispatchKernel::compile(&op, plc)?,
            RepShlDim(op) => DispatchKernel::compile(&op, plc)?,
            AdtAdd(op) => DispatchKernel::compile(&op, plc)?,
            AdtSub(op) => DispatchKernel::compile(&op, plc)?,
            AdtShl(op) => DispatchKernel::compile(&op, plc)?,
            AdtMul(op) => DispatchKernel::compile(&op, plc)?,
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
            HostAtLeast2D(op) => DispatchKernel::compile(&op, plc)?,
            HostMean(op) => DispatchKernel::compile(&op, plc)?,
            HostSqrt(op) => DispatchKernel::compile(&op, plc)?,
            HostSum(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointEncode(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointDecode(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointAdd(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointSub(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointMul(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointDiv(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointDot(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointTruncPr(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointSum(op) => DispatchKernel::compile(&op, plc)?,
            FixedpointMean(op) => DispatchKernel::compile(&op, plc)?,
            HostSlice(op) => DispatchKernel::compile(&op, plc)?,
            HostDiag(op) => DispatchKernel::compile(&op, plc)?,
            HostShlDim(op) => DispatchKernel::compile(&op, plc)?,
            HostAdd(op) => DispatchKernel::compile(&op, plc)?,
            HostSub(op) => DispatchKernel::compile(&op, plc)?,
            HostMul(op) => DispatchKernel::compile(&op, plc)?,
            HostDiv(op) => DispatchKernel::compile(&op, plc)?,
            HostDot(op) => DispatchKernel::compile(&op, plc)?,
            HostExpandDims(op) => DispatchKernel::compile(&op, plc)?,
            HostSqueeze(op) => DispatchKernel::compile(&op, plc)?,
            Sign(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointAdd(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointSub(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointMul(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointDiv(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointDot(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointAtLeast2D(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointOnes(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointConcat(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointExpandDims(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointTranspose(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointInverse(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointMean(op) => DispatchKernel::compile(&op, plc)?,
            FloatingpointSum(op) => DispatchKernel::compile(&op, plc)?,
            HostTranspose(op) => DispatchKernel::compile(&op, plc)?,
            HostInverse(op) => DispatchKernel::compile(&op, plc)?,
            HostBitDec(op) => DispatchKernel::compile(&op, plc)?,
            Identity(op) => DispatchKernel::compile(&op, plc)?,
            Cast(op) => DispatchKernel::compile(&op, plc)?,
            Send(op) => DispatchKernel::compile(&op, plc)?,
            Receive(op) => DispatchKernel::compile(&op, plc)?,
            HostReshape(op) => DispatchKernel::compile(&op, plc)?,
            AtLeast2D(op) => DispatchKernel::compile(&op, plc)?,
            Slice(op) => DispatchKernel::compile(&op, plc)?,
            Ones(op) => DispatchKernel::compile(&op, plc)?,
            ExpandDims(op) => DispatchKernel::compile(&op, plc)?,
            Concat(op) => DispatchKernel::compile(&op, plc)?,
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
            RepEqual(op) => DispatchKernel::compile(&op, plc)?,
            Mux(op) => DispatchKernel::compile(&op, plc)?,
            Less(op) => DispatchKernel::compile(&op, plc)?,
            GreaterThan(op) => DispatchKernel::compile(&op, plc)?,
            IndexAxis(op) => DispatchKernel::compile(&op, plc)?,
            Maximum(op) => DispatchKernel::compile(&op, plc)?,
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

pub trait DispatchKernel<S: Session> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(
        &self,
        plc: &Placement,
    ) -> Result<Box<dyn Fn(&S, Vec<S::Value>) -> Result<S::Value> + Send>>;
}

// TODO if rustc can't figure out how to optimize Box<dyn Fn...> for
// function kernels then we could consider returning an enum over
// fn.. and Box<dyn Fn...> in the traits below instead

pub(crate) trait NullaryKernel<S: Session, P, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P) -> Result<Y> + Send>>;
}

pub(crate) trait UnaryKernel<S: Session, P, X0, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, X0) -> Result<Y> + Send>>;
}

pub(crate) trait BinaryKernel<S: Session, P, X0, X1, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, X0, X1) -> Result<Y> + Send>>;
}

pub(crate) trait TernaryKernel<S: Session, P, X0, X1, X2, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, X0, X1, X2) -> Result<Y> + Send>>;
}

pub(crate) trait VariadicKernel<S: Session, P, XS, Y> {
    #[allow(clippy::type_complexity)] // TODO
    fn compile(&self, plc: &P) -> Result<Box<dyn Fn(&S, &P, Vec<XS>) -> Result<Y> + Send>>;
}

pub(crate) trait NullaryKernelCheck<S: Session, P, Y>
where
    Self: NullaryKernel<S, P, Y>,
{
}

pub(crate) trait UnaryKernelCheck<S: Session, P, X0, Y>
where
    Self: UnaryKernel<S, P, X0, Y>,
{
}

pub(crate) trait BinaryKernelCheck<S: Session, P, X0, X1, Y>
where
    Self: BinaryKernel<S, P, X0, X1, Y>,
{
}

pub(crate) trait TernaryKernelCheck<S: Session, P, X0, X1, X2, Y>
where
    Self: TernaryKernel<S, P, X0, X1, X2, Y>,
{
}

pub(crate) trait VariadicKernelCheck<S: Session, P, XS, Y>
where
    Self: VariadicKernel<S, P, XS, Y>,
{
}

pub trait TensorLike<S: Session> {
    type Scalar;
}

pub trait PlacementShape<S: Session, T, ShapeT> {
    fn shape(&self, sess: &S, x: &T) -> ShapeT;
}

pub trait PlacementReshape<S: Session, T, ShapeT, O> {
    fn reshape(&self, sess: &S, x: &T, shape: &ShapeT) -> O;
}

pub trait PlacementDecrypt<S: Session, KeyT, C, O> {
    fn decrypt(&self, sess: &S, key: &KeyT, ciphertext: &C) -> O;
}

pub trait PlacementKeyGen<S: Session, KeyT> {
    fn gen_key(&self, sess: &S) -> KeyT;
}

pub trait PlacementSetupGen<S: Session, SetupT> {
    fn gen_setup(&self, sess: &S) -> SetupT;
}

pub trait PlacementDeriveSeed<S: Session, KeyT, SeedT> {
    fn derive_seed(&self, sess: &S, sync_key: SyncKey, key: &KeyT) -> SeedT;
}

pub trait PlacementAdd<S: Session, T, U, O> {
    fn add(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementSub<S: Session, T, U, O> {
    fn sub(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementNeg<S: Session, T, O> {
    fn neg(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementMul<S: Session, T, U, O> {
    fn mul(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementDiv<S: Session, T, U, O> {
    fn div(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementDot<S: Session, T, U, O> {
    fn dot(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementShl<S: Session, T, O> {
    fn shl(&self, sess: &S, amount: usize, x: &T) -> O;
}

pub trait PlacementShr<S: Session, T, O> {
    fn shr(&self, sess: &S, amount: usize, x: &T) -> O;
}

pub trait PlacementXor<S: Session, T, U, O> {
    fn xor(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementAnd<S: Session, T, U, O> {
    fn and(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementOr<S: Session, T, U, O> {
    fn or(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementBitExtract<S: Session, T, O> {
    fn bit_extract(&self, sess: &S, bit_idx: usize, x: &T) -> O;
}

pub trait PlacementBitDec<S: Session, T, O> {
    fn bit_decompose(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementBitCompose<S: Session, T, O> {
    fn bit_compose(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementRingInject<S: Session, T, O> {
    fn ring_inject(&self, sess: &S, bit_idx: usize, x: &T) -> O;
}

pub trait PlacementShare<S: Session, T, O> {
    fn share(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementReveal<S: Session, T, O> {
    fn reveal(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementFill<S: Session, ShapeT, O> {
    fn fill(&self, sess: &S, value: Constant, shape: &ShapeT) -> O;
}

pub trait PlacementZeros<S: Session, ShapeT, O> {
    fn zeros(&self, sess: &S, shape: &ShapeT) -> O;
}

pub trait PlacementMean<S: Session, T, O> {
    fn mean(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

pub trait PlacementMeanAsFixedpoint<S: Session, T, O> {
    fn mean_as_fixedpoint(
        &self,
        sess: &S,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: &T,
    ) -> O;
}

pub trait PlacementSqrt<S: Session, T, O> {
    fn sqrt(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementAddN<S: Session, T, O> {
    fn add_n(&self, sess: &S, x: &[T]) -> O;
}

pub trait PlacementSum<S: Session, T, O> {
    fn sum(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

pub trait PlacementEqual<S: Session, T, U, O> {
    fn equal(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementMux<S: Session, T, U, V, O> {
    fn mux(&self, sess: &S, s: &T, x: &U, y: &V) -> O;
}

pub trait PlacementPow2<S: Session, T, O> {
    fn pow2(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementExp<S: Session, T, O> {
    fn exp(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementSigmoid<S: Session, T, O> {
    fn sigmoid(&self, sess: &S, x: &T) -> O;
}
pub trait PlacementLessThan<S: Session, T, U, O> {
    fn less(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementGreaterThan<S: Session, T, U, O> {
    fn greater_than(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementDemirror<S: Session, T, O> {
    fn demirror(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementMirror<S: Session, T, O> {
    fn mirror(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementMaximum<S: Session, TS, O> {
    fn maximum(&self, sess: &S, x: &[TS]) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementZeros<S, ShapeT, O> for P
where
    P: PlacementFill<S, ShapeT, O>,
    O: TensorLike<S>,
    O::Scalar: Into<Constant>,
    O::Scalar: From<u8>,
{
    fn zeros(&self, sess: &S, shape: &ShapeT) -> O {
        let value = O::Scalar::from(0).into();
        self.fill(sess, value, shape)
    }
}

modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostFloat32Tensor, HostOnesOp);
modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostFloat64Tensor, HostOnesOp);
modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostInt8Tensor, HostOnesOp);
modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostInt16Tensor, HostOnesOp);
modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostInt32Tensor, HostOnesOp);
modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> HostInt64Tensor, HostOnesOp);

kernel! {
    HostOnesOp, [
        (HostPlacement, (HostShape) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

pub trait PlacementOnes<S: Session, ShapeT, O> {
    fn ones(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementOnes<S, ShapeT, O> for P
where
    P: PlacementFill<S, ShapeT, O>,
    O: TensorLike<S>,
    O::Scalar: Into<Constant>,
    O::Scalar: From<u8>,
{
    fn ones(&self, sess: &S, shape: &ShapeT) -> O {
        let value = O::Scalar::from(1).into();
        self.fill(sess, value, shape)
    }
}

pub trait PlacementSample<S: Session, ShapeT, O> {
    fn sample(&self, sess: &S, max_value: Option<u64>, shape: &ShapeT) -> O;
}

pub trait PlacementSampleUniform<S: Session, ShapeT, O> {
    fn sample_uniform(&self, sess: &S, shape: &ShapeT) -> O;
}

pub trait PlacementSampleBits<S: Session, ShapeT, O> {
    fn sample_bits(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementSampleUniform<S, ShapeT, O> for P
where
    P: PlacementSample<S, ShapeT, O>,
{
    fn sample_uniform(&self, sess: &S, shape: &ShapeT) -> O {
        self.sample(sess, None, shape)
    }
}

impl<S: Session, ShapeT, O, P> PlacementSampleBits<S, ShapeT, O> for P
where
    P: PlacementSample<S, ShapeT, O>,
{
    fn sample_bits(&self, sess: &S, shape: &ShapeT) -> O {
        self.sample(sess, Some(1), shape)
    }
}

pub trait PlacementSampleSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_seeded(&self, sess: &S, max_value: Option<u64>, shape: &ShapeT, seed: &SeedT) -> O;
}

pub trait PlacementSampleUniformSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_uniform_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

pub trait PlacementSampleBitsSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_bits_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

impl<S: Session, ShapeT, SeedT, O, P> PlacementSampleUniformSeeded<S, ShapeT, SeedT, O> for P
where
    P: PlacementSampleSeeded<S, ShapeT, SeedT, O>,
{
    fn sample_uniform_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O {
        self.sample_seeded(sess, None, shape, seed)
    }
}

impl<S: Session, ShapeT, SeedT, O, P> PlacementSampleBitsSeeded<S, ShapeT, SeedT, O> for P
where
    P: PlacementSampleSeeded<S, ShapeT, SeedT, O>,
{
    fn sample_bits_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O {
        self.sample_seeded(sess, Some(1), shape, seed)
    }
}

pub trait PlacementRepToAdt<S: Session, T, O> {
    fn rep_to_adt(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementAdtToRep<S: Session, T, O> {
    fn adt_to_rep(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementTruncPr<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: u32, x: &T) -> O;
}

pub trait TruncPrProvider<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: usize, provider: &HostPlacement, x: &T) -> O;
}

pub trait PlacementAbs<S: Session, T, O> {
    fn abs(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementMsb<S: Session, T, O> {
    fn msb(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementSign<S: Session, T, O> {
    fn sign(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementPlace<S: Session, T> {
    fn place(&self, sess: &S, x: T) -> T;
}

pub trait PlacementConstant<S: Session, O> {
    fn constant(&self, sess: &S, value: Constant) -> O;
}

pub trait PlacementIdentity<S: Session, T, O> {
    fn identity(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementInput<S: Session, O> {
    fn input(&self, sess: &S, arg_name: String) -> O;
}

pub trait PlacementOutput<S: Session, T, O> {
    fn output(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementLoad<S: Session, KeyT, QueryT, O> {
    fn load(&self, sess: &S, key: &KeyT, query: &QueryT) -> O;
}

pub trait PlacementSave<S: Session, KeyT, T, O> {
    fn save(&self, sess: &S, key: &KeyT, x: &T) -> O;
}

pub trait PlacementSend<S: Session, T, O> {
    fn send(&self, sess: &S, rendezvous_key: RendezvousKey, receiver: Role, x: &T) -> O;
}

pub trait PlacementReceive<S: Session, O> {
    fn receive(&self, sess: &S, rendezvous_key: RendezvousKey, sender: Role) -> O;
}

pub trait PlacementAtLeast2D<S: Session, T, O> {
    fn at_least_2d(&self, sess: &S, to_column_vector: bool, x: &T) -> O;
}

pub trait PlacementRingFixedpointEncode<S: Session, T, O> {
    fn fixedpoint_ring_encode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
}

pub trait PlacementRingFixedpointDecode<S: Session, T, O> {
    fn fixedpoint_ring_decode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
}

pub trait PlacementFixedpointEncode<S: Session, T, O> {
    fn fixedpoint_encode(
        &self,
        sess: &S,
        fractional_precision: u32,
        integral_precision: u32,
        x: &T,
    ) -> O;
}

pub trait PlacementFixedpointDecode<S: Session, T, O> {
    fn fixedpoint_decode(&self, sess: &S, precision: u32, x: &T) -> O;
}

pub trait PlacementExpandDims<S: Session, T, O> {
    fn expand_dims(&self, sess: &S, axis: Vec<u32>, x: &T) -> O;
}

pub trait PlacementSqueeze<S: Session, T, O> {
    fn squeeze(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

pub trait PlacementConcatenate<S: Session, TS, O> {
    fn concatenate(&self, sess: &S, axis: u32, xs: &[TS]) -> O;
}

pub trait PlacementTranspose<S: Session, T, O> {
    fn transpose(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementInverse<S: Session, T, O> {
    fn inverse(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementCast<S: Session, T, O> {
    fn cast(&self, sess: &S, x: &T) -> O;
}

pub trait EmptyTypeHolder<T> {}

pub trait PlacementSlice<S: Session, T, O> {
    fn slice(&self, sess: &S, slice_info: SliceInfo, x: &T) -> O;
}

pub trait PlacementDiag<S: Session, T, O> {
    fn diag(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementIndexAxis<S: Session, T, O> {
    fn index_axis(&self, sess: &S, axis: usize, index: usize, x: &T) -> O;
}

pub trait PlacementIndex<S: Session, T, O> {
    fn index(&self, sess: &S, index: usize, x: &T) -> O;
}

pub trait PlacementShlDim<S: Session, T, O> {
    fn shl_dim(&self, sess: &S, amount: usize, ring_size: usize, x: &T) -> O;
}

macro_rules! constant_kernels {
    ($($val:ident),+) => {
        $(
            modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> $val, ConstantOp);
        )+
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> HostString, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> HostShape, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> PrfKey, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Seed, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Float32Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Float64Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, Mirrored3Placement, attributes[value: Constant] () -> Float32Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, Mirrored3Placement, attributes[value: Constant] () -> Float64Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, Mirrored3Placement, attributes[value: Constant] () -> Tensor, ConstantOp);


        kernel! {
            ConstantOp, [
                $(
                    (HostPlacement, () -> $val => [runtime] attributes[value: $val] Self::kernel),
                )+
                (HostPlacement, () -> HostString => [runtime] attributes[value: String] Self::string_kernel),
                (HostPlacement, () -> HostShape => [runtime] attributes[value: RawShape] Self::shape_kernel),
                (HostPlacement, () -> PrfKey => [runtime] attributes[value: RawPrfKey] Self::prf_key_kernel),
                (HostPlacement, () -> Seed => [runtime] attributes[value: RawSeed] Self::seed_kernel),
                (HostPlacement, () -> Tensor => [concrete] attributes[sig, value] Self::logical_kernel),
                (HostPlacement, () -> Float32Tensor => [concrete] attributes[value] Self::float_kernel),
                (HostPlacement, () -> Float64Tensor => [concrete] attributes[value] Self::float_kernel),
                (Mirrored3Placement, () -> Tensor => [concrete] attributes[sig, value] Self::mir3_logical_kernel),
                (Mirrored3Placement, () -> Float32Tensor => [concrete] attributes[value] Self::mir3_float_kernel),
                (Mirrored3Placement, () -> Float64Tensor => [concrete] attributes[value] Self::mir3_float_kernel),

            ]
        }
    };
}

constant_kernels![
    HostRing64Tensor,
    HostRing128Tensor,
    HostFloat32Tensor,
    HostFloat64Tensor,
    HostInt8Tensor,
    HostInt16Tensor,
    HostInt32Tensor,
    HostInt64Tensor,
    HostUint8Tensor,
    HostUint16Tensor,
    HostUint32Tensor,
    HostUint64Tensor
];

macro_rules! wrapping_constant_kernel {
    ($name:ident for $wrapping:tt($inner:ty)) => {
        impl ConstantOp {
            fn $name<S: RuntimeSession>(
                _sess: &S,
                plc: &HostPlacement,
                value: $inner,
            ) -> Result<$wrapping> {
                Ok($wrapping(value.clone(), plc.clone()))
            }
        }
    };
}

wrapping_constant_kernel!(string_kernel for HostString(String));
wrapping_constant_kernel!(shape_kernel for HostShape(RawShape));
wrapping_constant_kernel!(prf_key_kernel for PrfKey(RawPrfKey));
wrapping_constant_kernel!(seed_kernel for Seed(RawSeed));

impl ConstantOp {
    fn kernel<S: RuntimeSession, T: Placed>(sess: &S, plc: &HostPlacement, value: T) -> Result<T>
    where
        HostPlacement: PlacementPlace<S, T>,
    {
        Ok(plc.place(sess, value))
    }
}

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementSend::send, HostPlacement, attributes[rendezvous_key: RendezvousKey, receiver: Role] ($value) -> Unit, SendOp);
    )*
)}

kernel! {
    SendOp, [
        (HostPlacement, (HostString) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (Unit) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostShape) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (Seed) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (PrfKey) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostBitTensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostRing64Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostFloat32Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostUint8Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostUint16Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostFixed64Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
        (HostPlacement, (HostFixed128Tensor) -> Unit => [runtime] attributes[rendezvous_key, receiver] Self::kernel),
    ]
}

impl SendOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _receiver: Role,
        _x: T,
    ) -> Result<Unit>
    where
        Value: From<T>,
    {
        // let x: Value = x.into();
        // sess.networking.send(&x, &receiver, &rendezvous_key)?;
        // Ok(Unit(plc.clone()))
        todo!()
    }
}

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementReceive::receive, HostPlacement, attributes[rendezvous_key: RendezvousKey, sender: Role] () -> $value, ReceiveOp);
    )*
)}

kernel! {
    ReceiveOp, [
        (HostPlacement, () -> HostString => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> Unit => [runtime] attributes[rendezvous_key, sender] Self::missing_kernel),
        (HostPlacement, () -> HostShape => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> Seed => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> PrfKey => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostBitTensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostRing64Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostRing128Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostFloat32Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostFloat64Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostInt8Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostInt16Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostInt32Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostInt64Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostUint8Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostUint16Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostUint32Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostUint64Tensor => [runtime] attributes[rendezvous_key, sender] Self::kernel),
        (HostPlacement, () -> HostFixed64Tensor => [runtime] attributes[rendezvous_key, sender] Self::missing_kernel),
        (HostPlacement, () -> HostFixed128Tensor => [runtime] attributes[rendezvous_key, sender] Self::missing_kernel),

    ]
}

impl ReceiveOp {
    fn kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _sender: Role,
    ) -> Result<T>
    where
        T: TryFrom<Value, Error = Error>,
        T: std::fmt::Debug,
        HostPlacement: PlacementPlace<S, T>,
    {
        // use std::convert::TryInto;
        // let value = sess.networking.receive(&sender, &rendezvous_key)?;
        // Ok(plc.place(sess, value.try_into()?))
        todo!()
    }

    fn missing_kernel<S: RuntimeSession, T>(
        _sess: &S,
        _plc: &HostPlacement,
        _rendezvous_key: RendezvousKey,
        _sender: Role,
    ) -> Result<T>
    where
        T: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<T as KnownType<S>>::TY
        )))
    }
}

modelled_kernel! {
    PlacementIdentity::identity, IdentityOp,
    [
        (HostPlacement, (BooleanTensor) -> BooleanTensor => [concrete] Self::boolean_host_kernel),
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::host_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::host_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_inner_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_inner_kernel),
        (ReplicatedPlacement, (BooleanTensor) -> BooleanTensor => [concrete] Self::boolean_rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_inner_kernel),
        // TODO higher-level kernels for these
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint8Tensor) -> HostUint8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint16Tensor) -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString) -> HostString => [runtime] Self::kernel),
        (HostPlacement, (Unit) -> Unit => [runtime] Self::missing_kernel),
        (HostPlacement, (HostShape) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (Seed) -> Seed => [runtime] Self::kernel),
        (HostPlacement, (PrfKey) -> PrfKey => [runtime] Self::kernel),
    ]
}

impl IdentityOp {
    fn kernel<S: RuntimeSession, T>(sess: &S, plc: &HostPlacement, x: T) -> Result<T>
    where
        HostPlacement: PlacementPlace<S, T>,
    {
        let value = plc.place(sess, x);
        Ok(value)
    }

    fn missing_kernel<S: RuntimeSession, T>(_sess: &S, _plc: &HostPlacement, _x: T) -> Result<T>
    where
        T: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<T as KnownType<S>>::TY
        )))
    }
}

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> $value, InputOp);
    )*
)}

modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Float32Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Float64Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostBitArray64, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostBitArray128, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostBitArray224, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostAesKey, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitTensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedRing64Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedRing128Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedFixed64Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedFixed128Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitArray64, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitArray128, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitArray224, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedAesKey, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> AesKey, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> AesKey, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> AesTensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Fixed128AesTensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostFixed128AesTensor, InputOp);

kernel! {
    InputOp, [
        (HostPlacement, () -> HostString => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> Unit => [runtime] attributes[arg_name] Self::missing_kernel),
        (HostPlacement, () -> HostShape => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> Seed => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> PrfKey => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostBitArray64 => [concrete] attributes[arg_name] Self::host_bitarray64),
        (HostPlacement, () -> HostBitArray128 => [concrete] attributes[arg_name] Self::host_bitarray128),
        (HostPlacement, () -> HostBitArray224 => [concrete] attributes[arg_name] Self::host_bitarray224),
        (HostPlacement, () -> HostBitTensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostRing64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostRing128Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostFloat32Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostFloat64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt8Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt16Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt32Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint8Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint16Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint32Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostFixed64Tensor => [runtime] attributes[arg_name] Self::missing_kernel),
        (HostPlacement, () -> HostFixed128Tensor => [runtime] attributes[arg_name] Self::missing_kernel),
        (HostPlacement, () -> Tensor => [concrete] attributes[sig, arg_name] Self::logical_kernel),
        (HostPlacement, () -> Float32Tensor => [concrete] attributes[arg_name] Self::float_kernel),
        (HostPlacement, () -> Float64Tensor => [concrete] attributes[arg_name] Self::float_kernel),
        (HostPlacement, () -> AesKey => [concrete] attributes[arg_name] Self::aes_kernel_on_host),
        (HostPlacement, () -> HostAesKey => [concrete] attributes[arg_name] Self::host_aes_kernel),
        (HostPlacement, () -> AesTensor => [concrete] attributes[arg_name] Self::aestensor),
        (HostPlacement, () -> Fixed128AesTensor => [concrete] attributes[arg_name] Self::fixed_aestensor),
        (HostPlacement, () -> HostFixed128AesTensor => [concrete] attributes[sig, arg_name] Self::host_fixed_aestensor),
        (ReplicatedPlacement, () -> ReplicatedBitTensor => [concrete] attributes[arg_name] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedRing64Tensor => [concrete] attributes[arg_name] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedRing128Tensor => [concrete] attributes[arg_name] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedFixed64Tensor => [concrete] attributes[sig, arg_name] Self::replicated_fixed_kernel),
        (ReplicatedPlacement, () -> ReplicatedFixed128Tensor => [concrete] attributes[sig, arg_name] Self::replicated_fixed_kernel),
        (ReplicatedPlacement, () -> ReplicatedBitArray64 => [concrete] attributes[arg_name] Self::replicated_bitarray64),
        (ReplicatedPlacement, () -> ReplicatedBitArray128 => [concrete] attributes[arg_name] Self::replicated_bitarray128),
        (ReplicatedPlacement, () -> ReplicatedBitArray224 => [concrete] attributes[arg_name] Self::replicated_bitarray224),
        (ReplicatedPlacement, () -> AesKey => [concrete] attributes[arg_name] Self::aes_kernel_on_replicated),
        (ReplicatedPlacement, () -> ReplicatedAesKey => [concrete] attributes[arg_name] Self::replicated_aes_kernel),
    ]
}

impl InputOp {
    fn kernel<S: RuntimeSession, O>(sess: &S, plc: &HostPlacement, arg_name: String) -> Result<O>
    where
        O: TryFrom<Value, Error = Error>,
        HostPlacement: PlacementPlace<S, O>,
    {
        use std::convert::TryInto;
        let value = sess
            .find_argument(&arg_name)
            .ok_or_else(|| Error::MissingArgument(arg_name.clone()))?;
        let value = plc.place(sess, value.try_into()?);
        Ok(value)
    }

    fn missing_kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _arg_name: String,
    ) -> Result<O>
    where
        O: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<O as KnownType<S>>::TY
        )))
    }
}

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementOutput::output, HostPlacement, ($value) -> $value, OutputOp);
    )*
)}
modelled!(PlacementOutput::output, HostPlacement, (Tensor) -> Tensor, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (Float32Tensor) -> Float32Tensor, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (Float64Tensor) -> Float64Tensor, OutputOp);

kernel! {
    OutputOp, [
        (HostPlacement, (Unit) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (Seed) -> Seed => [runtime] Self::kernel),
        (HostPlacement, (PrfKey) -> PrfKey => [runtime] Self::kernel),
        (HostPlacement, (HostString) -> HostString => [runtime] Self::kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint8Tensor) -> HostUint8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint16Tensor) -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [runtime] Self::non_placing_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [runtime] Self::non_placing_kernel),
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::logical_kernel),
        (HostPlacement, (BooleanTensor) -> BooleanTensor => [hybrid] Self::bool_kernel),
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_kernel),
    ]
}

impl OutputOp {
    fn kernel<S: RuntimeSession, O>(sess: &S, plc: &HostPlacement, x: O) -> Result<O>
    where
        HostPlacement: PlacementPlace<S, O>,
    {
        // Output is not doing anything now, it is just a marker on the graph.
        // But it has to return a value because that's how we collect outputs in the old framework
        let x = plc.place(sess, x);
        Ok(x)
    }

    fn non_placing_kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        x: O,
    ) -> Result<O> {
        // Output is not doing anything now, it is just a marker on the graph.
        // But it has to return a value because that's how we collect outputs in the old framework
        Ok(x)
    }
}

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementSave::save, HostPlacement, (HostString, $value) -> Unit, SaveOp);
    )*
)}

modelled!(PlacementSave::save, HostPlacement, (HostString, Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, Float32Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, Float64Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, BooleanTensor) -> Unit, SaveOp);

kernel! {
    SaveOp, [
        (HostPlacement, (HostString, Unit) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostShape) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, Seed) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, PrfKey) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostBitTensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostRing64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostRing128Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFloat32Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFloat64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt8Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt16Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt32Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint8Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint16Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint32Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFixed64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFixed128Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, Tensor) -> Unit => [hybrid] Self::logical_kernel),
        (HostPlacement, (HostString, Float32Tensor) -> Unit => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, Float64Tensor) -> Unit => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, BooleanTensor) -> Unit => [hybrid] Self::bool_kernel),
    ]
}

impl SaveOp {
    fn kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: HostString,
        _x: O,
    ) -> Result<Unit>
    where
        Value: From<O>,
    {
        // let x: Value = x.into();
        // sess.storage.save(&key.0, &x)?;
        // Ok(Unit(plc.clone()))
        todo!()
    }
}

modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> HostFloat64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> Float64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> Tensor, LoadOp);

kernel! {
    LoadOp, [
        (HostPlacement, (HostString, HostString) -> Unit => [runtime] Self::missing_kernel),
        (HostPlacement, (HostString, HostString) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> Seed => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> PrfKey => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostString => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostBitTensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostRing128Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostFixed64Tensor => [runtime] Self::missing_kernel),
        (HostPlacement, (HostString, HostString) -> HostFixed128Tensor => [runtime] Self::missing_kernel),
        (HostPlacement, (HostString, HostString) -> Float64Tensor => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, HostString) -> Tensor => [hybrid] Self::logical_kernel),
    ]
}

impl LoadOp {
    fn kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: HostString,
        _query: HostString,
    ) -> Result<O>
    where
        O: KnownType<S>,
        O: TryFrom<Value, Error = Error>,
        HostPlacement: PlacementPlace<S, O>,
    {
        // use std::convert::TryInto;
        // let value = sess.storage.load(&key.0, &query.0, Some(<O as KnownType<S>>::TY))?;
        // let value = plc.place(sess, value.try_into()?);
        // Ok(value)
        todo!()
    }

    fn missing_kernel<S: RuntimeSession, O>(
        _sess: &S,
        _plc: &HostPlacement,
        _key: HostString,
        _query: HostString,
    ) -> Result<O>
    where
        O: KnownType<S>,
    {
        Err(Error::KernelError(format!(
            "missing HostPlacement: PlacementPlace trait implementation for '{}'",
            &<O as KnownType<S>>::TY
        )))
    }
}

kernel! {
    SigmoidOp,
    [
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [transparent] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [transparent] Self::rep_rep_kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] Self::logical_kernel),
    ]
}

kernel! {
    LessOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> BooleanTensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> BooleanTensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (Float32Tensor, Float32Tensor) -> BooleanTensor => [concrete] Self::float_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> BooleanTensor => [concrete] Self::float_kernel),
        (HostPlacement, (HostFloat32Tensor, HostFloat32Tensor) -> HostBitTensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostFloat64Tensor, HostFloat64Tensor) -> HostBitTensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostBitTensor => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostFixed128Tensor, HostFixed128Tensor) -> HostBitTensor => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostBitTensor => [runtime] Self::host_ring64_kernel),
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostBitTensor => [runtime] Self::host_ring128_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> BooleanTensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> BooleanTensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_mir_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_mir_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, Mirrored3Ring64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, Mirrored3Ring128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
    ]
}

kernel! {
    GreaterThanOp,
    [
        (HostPlacement, (HostRing128Tensor, HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_mir_fixed_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::mir_rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, Mirrored3Fixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_mir_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, Mirrored3Fixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_mir_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedBitTensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, Mirrored3Ring128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, Mirrored3Ring64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_mir_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [transparent] Self::rep_kernel),
    ]
}

kernel! {
    FillOp,
    [
        (HostPlacement, (HostShape) -> HostBitTensor => [runtime] custom |op| {
            use std::convert::TryInto;
            let value: u8 = match op.value {
                Constant::Bit(v) => v,
                Constant::Ring64(v) => v.try_into().map_err(|_| {
                    Error::KernelError("Cannot fill HostBitTensor with non-binary value.".to_string())
                })?,
                Constant::Ring128(v) => v.try_into().map_err(|_| {
                    Error::KernelError("Cannot fill HostBitTensor with non-binary value.".to_string())
                })?,
                _ => {
                    return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a HostBitTensor", op.value.ty())))
                }
            };
            if !(value == 0 || value == 1) {
                return Err(Error::KernelError(
                    "Cannot fill HostBitTensor with non-binary value.".to_string(),
                ));
            }
            assert!(value == 0 || value == 1);
            Ok(Box::new(move |sess, host, host_shape| {
                Self::bit_kernel(sess, host, value, host_shape)
            }))
        }),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedRing64Tensor => [concrete] custom |op| {
                let value: u64 = match op.value {
                    Constant::Bit(v) => v as u64,
                    Constant::Ring64(v) => v,
                    Constant::Float64(v) => v as u64,
                    Constant::Fixed(FixedpointConstant {
                        value, precision
                    }) => {
                        (value * ((1u64 << precision) as f64)) as u64
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a ReplicatedRing64Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::ring64_kernel(sess, rep, value, rep_shape)
                }))
            }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Ring64Tensor => [concrete] custom |op| {
                let value: u64 = match op.value {
                    Constant::Bit(v) => v as u64,
                    Constant::Ring64(v) => v,
                    Constant::Float64(v) => v as u64,
                    Constant::Fixed(FixedpointConstant {
                        value, precision
                    }) => {
                        (value * ((1u64 << precision) as f64)) as u64
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Ring64Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_ring64_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedRing128Tensor => [concrete] custom |op| {
                let value: u128 = match op.value {
                    Constant::Bit(v) => v as u128,
                    Constant::Ring64(v) => v as u128,
                    Constant::Ring128(v) => v,
                    Constant::Float64(v) => v as u128,
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                            (value * ((1u128 << precision) as f64)) as u128
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a ReplicatedRing128Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::ring128_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Ring128Tensor => [concrete] custom |op| {
                let value: u128 = match op.value {
                    Constant::Bit(v) => v as u128,
                    Constant::Ring64(v) => v as u128,
                    Constant::Ring128(v) => v,
                    Constant::Float64(v) => v as u128,
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                            (value * ((1u128 << precision) as f64)) as u128
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Ring128Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_ring128_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedBitTensor => [concrete] custom |op| {
                let value: u8 = match op.value {
                    Constant::Bit(v) => v,
                    Constant::Ring64(v) => v as u8,
                    Constant::Ring128(v) => v as u8,
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a ReplicatedBitTensor", op.value.ty()))),
                };
                if value != 0 && value != 1 {
                    return Err(Error::InvalidArgument(format!("Could only support 0 and 1 for the bit tensor fill, got {}", value)));
                }
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::rep_bit_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3BitTensor => [concrete] custom |op| {
                let value: u8 = match op.value {
                    Constant::Bit(v) => v,
                    Constant::Ring64(v) => v as u8,
                    Constant::Ring128(v) => v as u8,
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3BitTensor", op.value.ty()))),
                };
                if value != 0 && value != 1 {
                    return Err(Error::InvalidArgument(format!("Could only support 0 and 1 for the bit tensor fill, got {}", value)));
                }
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_bit_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Fixed64Tensor => [hybrid] custom |op| {
                let (ring_value, fractional_precision, integral_precision) = match op.value {
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                        let ring_value: u64 = (value * ((1u64 << precision) as f64)) as u64;
                        let fractional_precision = precision as u32;
                        let integral_precision = value.log2().ceil() as u32;
                        (ring_value, fractional_precision, integral_precision)
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Fixed64Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_fixed_kernel(sess, rep, Constant::Ring64(ring_value), rep_shape, fractional_precision, integral_precision)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Fixed128Tensor => [hybrid] custom |op| {
                let (ring_value, fractional_precision, integral_precision) = match op.value {
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                        let ring_value: u128 = (value * ((1u128 << precision) as f64)) as u128;
                        let fractional_precision = precision as u32;
                        let integral_precision = value.log2().ceil() as u32;
                        (ring_value, fractional_precision, integral_precision)
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Fixed128Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_fixed_kernel(sess, rep, Constant::Ring128(ring_value), rep_shape, fractional_precision, integral_precision)
                }))
        }),
    ]
}

kernel! {
    MuxOp,
    [
        (ReplicatedPlacement, (Tensor, Tensor, Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (BooleanTensor, Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (BooleanTensor, Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor  => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor  => [transparent] Self::rep_bit_selector_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_bit_selector_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::rep_bit_selector_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_bit_selector_fixed_kernel),
    ]
}

kernel! {
    BitOrOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
        (HostPlacement, (BooleanTensor, BooleanTensor) -> BooleanTensor => [concrete] Self::bool_kernel),
        (HostPlacement, (HostBitTensor, HostBitTensor) -> HostBitTensor => [runtime] Self::host_kernel),
    ]
}

modelled_kernel! {
    PlacementIndexAxis::index_axis, IndexAxisOp{axis: usize, index: usize},
    [

        (HostPlacement, (BooleanTensor) -> BooleanTensor => [concrete] Self::bool_host_kernel),
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::logical_host_kernel),
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_host_kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::host_bit_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::hostfixed_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::host_float_kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::host_ring_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::host_ring_kernel),
        (ReplicatedPlacement, (BooleanTensor) -> BooleanTensor => [concrete]  Self::bool_rep_kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::repfixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [concrete]  Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [concrete]  Self::rep_kernel),
    ]
}

modelled_kernel! {
    PlacementFixedpointEncode::fixedpoint_encode, FixedpointEncodeOp{fractional_precision: u32, integral_precision: u32},
    [
        (HostPlacement, (Float32Tensor) -> Fixed64Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (Float64Tensor) -> Fixed128Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFixed64Tensor => [hybrid] Self::hostfixed_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFixed128Tensor => [hybrid] Self::hostfixed_kernel),
        (Mirrored3Placement, (Float32Tensor) -> Fixed64Tensor => [concrete] Self::mir_fixed_kernel),
        (Mirrored3Placement, (Float64Tensor) -> Fixed128Tensor => [concrete] Self::mir_fixed_kernel),
        (Mirrored3Placement, (Mirrored3Float32) -> Mirrored3Fixed64Tensor => [hybrid] Self::mir_fixed_lower_kernel),
        (Mirrored3Placement, (Mirrored3Float64) -> Mirrored3Fixed128Tensor => [hybrid] Self::mir_fixed_lower_kernel),
    ]
}

modelled_kernel! {
    PlacementRingFixedpointEncode::fixedpoint_ring_encode, RingFixedpointEncodeOp{scaling_base: u64, scaling_exp: u32},
    [
        (HostPlacement, (HostFloat32Tensor) -> HostRing64Tensor => [runtime] Self::float32_kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostRing128Tensor => [runtime] Self::float64_kernel),
        (Mirrored3Placement, (Mirrored3Float32) -> Mirrored3Ring64Tensor => [concrete] Self::mir_kernel),
        (Mirrored3Placement, (Mirrored3Float64) -> Mirrored3Ring128Tensor => [concrete] Self::mir_kernel),
    ]
}

modelled_kernel! {
    PlacementFixedpointDecode::fixedpoint_decode, FixedpointDecodeOp{fractional_precision: u32},
    [
        (HostPlacement, (Fixed64Tensor) -> Float32Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (Fixed128Tensor) -> Float64Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFloat32Tensor => [hybrid] Self::hostfixed_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFloat64Tensor => [hybrid] Self::hostfixed_kernel),
        (Mirrored3Placement, (Fixed64Tensor) -> Float32Tensor => [concrete] Self::mir_fixed_kernel),
        (Mirrored3Placement, (Fixed128Tensor) -> Float64Tensor => [concrete] Self::mir_fixed_kernel),
        (Mirrored3Placement, (Mirrored3Fixed64Tensor) -> Mirrored3Float32 => [hybrid] Self::mir_fixed_lower_kernel),
        (Mirrored3Placement, (Mirrored3Fixed128Tensor) -> Mirrored3Float64 => [hybrid] Self::mir_fixed_lower_kernel),
    ]
}

modelled_kernel! {
    PlacementRingFixedpointDecode::fixedpoint_ring_decode, RingFixedpointDecodeOp{scaling_base: u64, scaling_exp: u32},
    [
        (HostPlacement, (HostRing64Tensor) -> HostFloat32Tensor => [runtime] Self::float32_kernel),
        (HostPlacement, (HostRing128Tensor) -> HostFloat64Tensor => [runtime] Self::float64_kernel),
        (Mirrored3Placement, (Mirrored3Ring64Tensor) -> Mirrored3Float32 => [concrete] Self::mir_kernel),
        (Mirrored3Placement, (Mirrored3Ring128Tensor) -> Mirrored3Float64 => [concrete] Self::mir_kernel),
    ]
}

modelled_kernel! {
    PlacementDemirror::demirror, DemirrorOp,
    [
        (HostPlacement, (Mirrored3BitTensor) -> HostBitTensor => [hybrid] Self::kernel),
        (HostPlacement, (Mirrored3Fixed64Tensor) -> HostFixed64Tensor => [hybrid] Self::fixed_kernel),
        (HostPlacement, (Mirrored3Fixed128Tensor) -> HostFixed128Tensor => [hybrid] Self::fixed_kernel),
        (HostPlacement, (Mirrored3Float32) -> HostFloat32Tensor => [hybrid] Self::kernel),
        (HostPlacement, (Mirrored3Float64) -> HostFloat64Tensor => [hybrid] Self::kernel),
        (HostPlacement, (Mirrored3Ring64Tensor) -> HostRing64Tensor => [hybrid] Self::kernel),
        (HostPlacement, (Mirrored3Ring128Tensor) -> HostRing128Tensor => [hybrid] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementShare::share, RepShareOp,
    [
        (ReplicatedPlacement, (HostFixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::fixed_kernel),
        (ReplicatedPlacement, (HostFixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::fixed_kernel),
        (ReplicatedPlacement, (HostRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::ring_kernel),
        (ReplicatedPlacement, (HostRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::ring_kernel),
        (ReplicatedPlacement, (HostBitTensor) -> ReplicatedBitTensor => [hybrid] Self::ring_kernel),
        (ReplicatedPlacement, (HostBitArray64) -> ReplicatedBitArray64 => [concrete] Self::array_kernel),
        (ReplicatedPlacement, (HostBitArray128) -> ReplicatedBitArray128 => [concrete] Self::array_kernel),
        (ReplicatedPlacement, (HostBitArray224) -> ReplicatedBitArray224 => [concrete] Self::array_kernel),
        (ReplicatedPlacement, (HostAesKey) -> ReplicatedAesKey => [concrete] Self::aeskey_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed64Tensor) -> ReplicatedFixed64Tensor => [concrete] Self::fixed_mir_kernel),
        (ReplicatedPlacement, (Mirrored3Fixed128Tensor) -> ReplicatedFixed128Tensor => [concrete] Self::fixed_mir_kernel),
        (ReplicatedPlacement, (Mirrored3Ring64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::ring_mir_kernel),
        (ReplicatedPlacement, (Mirrored3Ring128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::ring_mir_kernel),
    ]
}

modelled_kernel! {
    PlacementReveal::reveal, RepRevealOp,
    [
        (HostPlacement, (ReplicatedFixed64Tensor) -> HostFixed64Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (ReplicatedFixed128Tensor) -> HostFixed128Tensor => [concrete] Self::fixed_kernel),
        (HostPlacement, (ReplicatedRing64Tensor) -> HostRing64Tensor => [hybrid] Self::ring_kernel),
        (HostPlacement, (ReplicatedRing128Tensor) -> HostRing128Tensor => [hybrid] Self::ring_kernel),
        (HostPlacement, (ReplicatedBitTensor) -> HostBitTensor => [hybrid] Self::ring_kernel),
        (HostPlacement, (ReplicatedBitArray64) -> HostBitArray64 => [concrete] Self::bit_array_kernel),
        (HostPlacement, (ReplicatedBitArray128) -> HostBitArray128 => [concrete] Self::bit_array_kernel),
        (HostPlacement, (ReplicatedBitArray224) -> HostBitArray224 => [concrete] Self::bit_array_kernel),
        (HostPlacement, (ReplicatedAesKey) -> HostAesKey => [concrete] Self::aeskey_kernel),
        (Mirrored3Placement, (ReplicatedBitTensor) -> Mirrored3BitTensor => [concrete] Self::mir_ring_kernel),
        (Mirrored3Placement, (ReplicatedRing64Tensor) -> Mirrored3Ring64Tensor => [concrete] Self::mir_ring_kernel),
        (Mirrored3Placement, (ReplicatedRing128Tensor) -> Mirrored3Ring128Tensor => [concrete] Self::mir_ring_kernel),
        (Mirrored3Placement, (ReplicatedFixed64Tensor) -> Mirrored3Fixed64Tensor => [concrete] Self::mir_fixed_kernel),
        (Mirrored3Placement, (ReplicatedFixed128Tensor) -> Mirrored3Fixed128Tensor => [concrete] Self::mir_fixed_kernel),
    ]
}

modelled_kernel! {
    PlacementMirror::mirror, MirrorOp,
    [
        (Mirrored3Placement, (HostFixed64Tensor) -> Mirrored3Fixed64Tensor => [concrete] Self::fixed_kernel),
        (Mirrored3Placement, (HostFixed128Tensor) -> Mirrored3Fixed128Tensor => [concrete] Self::fixed_kernel),
        (Mirrored3Placement, (HostFloat32Tensor) -> Mirrored3Float32 => [hybrid] Self::kernel),
        (Mirrored3Placement, (HostFloat64Tensor) -> Mirrored3Float64 => [hybrid] Self::kernel),
        (Mirrored3Placement, (HostRing64Tensor) -> Mirrored3Ring64Tensor => [hybrid] Self::kernel),
        (Mirrored3Placement, (HostRing128Tensor) -> Mirrored3Ring128Tensor => [hybrid] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementMaximum::maximum, MaximumOp,
    [
        (ReplicatedPlacement, vec[Tensor] -> Tensor => [concrete] Self::rep_logical_kernel),
        (ReplicatedPlacement, vec[Fixed64Tensor] -> Fixed64Tensor => [concrete] Self::fixed_kernel),
        (ReplicatedPlacement, vec[Fixed128Tensor] -> Fixed128Tensor => [concrete] Self::fixed_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed64Tensor] -> ReplicatedFixed64Tensor => [concrete] Self::rep_fixed_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed128Tensor] -> ReplicatedFixed128Tensor => [concrete] Self::rep_fixed_kernel),
        (ReplicatedPlacement, vec[ReplicatedRing64Tensor] -> ReplicatedRing64Tensor => [transparent] Self::kernel),
        (ReplicatedPlacement, vec[ReplicatedRing128Tensor] -> ReplicatedRing128Tensor => [transparent] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementRingInject::ring_inject, RingInjectOp{bit_idx: usize},
    [
        (HostPlacement, (HostBitTensor) -> HostRing64Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostBitTensor) -> HostRing128Tensor => [runtime] Self::host_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedRing64Tensor => [concrete] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedRing128Tensor => [concrete] Self::rep_kernel),
    ]
}

#[derive(Default)]
pub struct TestSyncExecutor {
    // Placeholder for the future state we want to keep
}

impl TestSyncExecutor {
    pub fn run_computation(
        &self,
        computation: &Computation,
        session: &SyncSession,
    ) -> anyhow::Result<HashMap<String, Value>> {
        let mut env: HashMap<String, Value> = HashMap::default();

        let output_names: Vec<String> = computation
            .operations
            .iter() // guessing that par_iter won't help here
            .filter_map(|op| match op.kind {
                Operator::Output(_) => Some(op.name.clone()),
                _ => None,
            })
            .collect();

        for op in computation.operations.iter() {
            let operator = op.kind.clone();
            let operands = op
                .inputs
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();
            let value = session
                .execute(operator, &op.placement, operands)
                .map_err(|e| {
                    Error::Compilation(format!(
                        "SyncSession failed to execute computation due to an error: {:?}",
                        e,
                    ))
                })?;
            env.insert(op.name.clone(), value);
        }

        let outputs: HashMap<String, Value> = output_names
            .iter()
            .map(|op_name| (op_name.clone(), env.get(op_name).cloned().unwrap()))
            .collect();
        Ok(outputs)
    }
}
