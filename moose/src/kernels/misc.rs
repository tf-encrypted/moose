use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::{Identity, SyncNetworkingImpl, SyncStorageImpl};
use crate::for_all_values;
use crate::host::*;
use crate::mirrored::*;
use crate::networking::LocalSyncNetworking;
use crate::prim::{PrfKey, RawPrfKey, RawSeed, Seed, SyncKey};
use crate::replicated::*;
use crate::storage::LocalSyncStorage;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use super::*;

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
            HostConcat(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
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
            HostConcat(op) => DispatchKernel::compile(&op, plc)?,
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

pub trait EmptyTypeHolder<T> {}
