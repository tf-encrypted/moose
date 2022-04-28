//! Synchronous/eager execution of computations

use super::{Identity, Operands, RuntimeSession, Session, SetupGeneration};
use crate::computation::*;
use crate::error::{Error, Result};
use crate::host::*;
use crate::kernels::{DispatchKernel, Kernel};
use crate::networking::{LocalSyncNetworking, SyncNetworking};
use crate::replicated::*;
use crate::storage::LocalSyncStorage;
use crate::storage::SyncStorage;
use std::collections::HashMap;
use std::convert::TryInto;
use std::rc::Rc;
use std::sync::Arc;

pub type SyncNetworkingImpl = Rc<dyn SyncNetworking>;
pub type SyncStorageImpl = Rc<dyn SyncStorage>;

/// Session object for synchronous/eager execution.
pub struct SyncSession {
    session_id: SessionId,
    replicated_keys: std::sync::RwLock<HashMap<ReplicatedPlacement, Arc<RepSetup<HostPrfKey>>>>,
    arguments: HashMap<String, Value>,
    role_assignments: HashMap<Role, Identity>,
    storage: SyncStorageImpl,
    networking: SyncNetworkingImpl,
}

/// Default session should only be used in tests.
///
/// Use `new()` for the real sessions instead.
impl Default for SyncSession {
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

impl DispatchKernel<SyncSession, Value> for SendOp {
    fn compile(&self, plc: &Placement) -> Result<Kernel<SyncSession, Value>> {
        if let Placement::Host(plc) = plc {
            let plc = plc.clone();
            let op = self.clone();
            Ok(Kernel::Unary {
                closure: Box::new(move |sess, _plc, x| {
                    sess.networking.send(
                        &x,
                        sess.find_role_assignment(&op.receiver)?,
                        &op.rendezvous_key,
                        &sess.session_id,
                    )?;
                    Ok(HostUnit(plc.clone()).into())
                }),
            })
        } else {
            unimplemented!()
        }
    }
}

impl DispatchKernel<SyncSession, Value> for ReceiveOp {
    fn compile(&self, plc: &Placement) -> Result<Kernel<SyncSession, Value>> {
        if let Placement::Host(_plc) = plc {
            let op = self.clone();
            Ok(Kernel::Nullary {
                closure: Box::new(move |sess, _plc| {
                    // TODO(Morten) we should verify type of received value
                    sess.networking.receive(
                        sess.find_role_assignment(&op.sender)?,
                        &op.rendezvous_key,
                        &sess.session_id,
                    )
                }),
            })
        } else {
            Err(Error::UnimplementedOperator(format!(
                "ReceiveOp is not implemented for placement {:?}",
                plc
            )))
        }
    }
}

impl DispatchKernel<SyncSession, Value> for Operator {
    fn compile(&self, plc: &Placement) -> Result<Kernel<SyncSession, Value>> {
        use Operator::*;
        match self {
            Load(_) => unimplemented!(),
            Save(_) => unimplemented!(),
            Send(op) => DispatchKernel::compile(op, plc),
            Receive(op) => DispatchKernel::compile(op, plc),

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
            GreaterThan(op) => DispatchKernel::compile(op, plc),
            Identity(op) => DispatchKernel::compile(op, plc),
            Index(op) => DispatchKernel::compile(op, plc),
            IndexAxis(op) => DispatchKernel::compile(op, plc),
            Input(op) => DispatchKernel::compile(op, plc),
            Inverse(op) => DispatchKernel::compile(op, plc),
            LessThan(op) => DispatchKernel::compile(op, plc),
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

impl Session for SyncSession {
    type Value = Value;

    fn execute(&self, op: &Operator, plc: &Placement, operands: Operands<Value>) -> Result<Value> {
        let mut operands = operands;
        let kernel: Kernel<SyncSession, _> = match op {
            Operator::Load(op) => {
                assert_eq!(operands.len(), 2);
                let query: HostString = operands.pop().unwrap().try_into()?;
                let key: HostString = operands.pop().unwrap().try_into()?;
                // TODO(Morten) we should verify type of loaded value
                return self
                    .storage
                    .load(&key.0, &self.session_id, Some(op.sig.ret()), &query.0);
            }
            Operator::Save(_) => {
                assert_eq!(operands.len(), 2);
                let value: Value = operands.pop().unwrap();
                let key: HostString = operands.pop().unwrap().try_into()?;
                self.storage.save(&key.0, &self.session_id, &value)?;
                let host = match plc {
                    Placement::Host(host) => host,
                    _ => unimplemented!(
                        "SyncSession does not support running Save on non-host placements yet"
                    ),
                };
                return Ok(HostUnit(host.clone()).into());
            }
            op => DispatchKernel::compile(op, plc),
        }?;
        match kernel {
            Kernel::Nullary { closure } => {
                assert_eq!(operands.len(), 0);
                closure(self, plc)
            }
            Kernel::Unary { closure } => {
                assert_eq!(operands.len(), 1);
                let x0 = operands.pop().unwrap();
                closure(self, plc, x0)
            }
            Kernel::Binary { closure } => {
                assert_eq!(operands.len(), 2);
                let x1 = operands.pop().unwrap();
                let x0 = operands.pop().unwrap();
                closure(self, plc, x0, x1)
            }
            Kernel::Ternary { closure } => {
                assert_eq!(operands.len(), 3);
                let x2 = operands.pop().unwrap();
                let x1 = operands.pop().unwrap();
                let x0 = operands.pop().unwrap();
                closure(self, plc, x0, x1, x2)
            }
            Kernel::Variadic { closure } => closure(self, plc, operands),
        }
    }
}

impl SetupGeneration<ReplicatedPlacement> for SyncSession {
    type Setup = RepSetup<HostPrfKey>;

    fn setup(&self, plc: &ReplicatedPlacement) -> Result<Arc<Self::Setup>> {
        let mut replicated_keys = self.replicated_keys.write().unwrap();
        let setup = replicated_keys
            .entry(plc.clone())
            .or_insert_with(|| Arc::new(plc.gen_setup(self).unwrap())); // TODO don't unwrap
        Ok(Arc::clone(setup))
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
            .ok_or_else(|| Error::Networking(format!("Missing role assignment for {}", role)))
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
        let mut env: HashMap<String, Value> = HashMap::with_capacity(computation.operations.len());

        let output_names: Vec<String> = computation
            .operations
            .iter() // guessing that par_iter won't help here
            .filter_map(|op| match op.kind {
                Operator::Output(_) => Some(op.name.clone()),
                _ => None,
            })
            .collect();

        for op in computation.operations.iter() {
            let operands = op
                .inputs
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();
            let value = session
                .execute(&op.kind, &op.placement, operands)
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
