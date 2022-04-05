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

impl DispatchKernel<SyncSession> for SendOp {
    fn compile(&self, plc: &Placement) -> Result<Kernel<SyncSession>> {
        if let Placement::Host(plc) = plc {
            let plc = plc.clone();
            let op = self.clone();
            Ok(Box::new(move |sess, operands| {
                assert_eq!(operands.len(), 1);
                let x = operands.get(0).unwrap();
                sess.networking.send(
                    x,
                    sess.find_role_assignment(&op.receiver)?,
                    &op.rendezvous_key,
                    &sess.session_id,
                )?;
                Ok(HostUnit(plc.clone()).into())
            }))
        } else {
            unimplemented!()
        }
    }
}

impl DispatchKernel<SyncSession> for ReceiveOp {
    fn compile(&self, plc: &Placement) -> Result<Kernel<SyncSession>> {
        if let Placement::Host(_plc) = plc {
            let op = self.clone();
            Ok(Box::new(move |sess, operands| {
                assert_eq!(operands.len(), 0);
                // TODO(Morten) we should verify type of received value
                sess.networking.receive(
                    sess.find_role_assignment(&op.sender)?,
                    &op.rendezvous_key,
                    &sess.session_id,
                )
            }))
        } else {
            Err(Error::UnimplementedOperator(format!(
                "ReceiveOp is not implemented for placement {:?}",
                plc
            )))
        }
    }
}

impl Session for SyncSession {
    type Value = Value;

    fn execute(
        &self,
        op: &Operator,
        plc: &Placement,
        mut operands: Operands<Value>,
    ) -> Result<Value> {
        use Operator::*;
        let kernel_output = match op {
            Send(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Receive(op) => DispatchKernel::compile(op, plc)?(self, operands)?,

            // TODO(Morten) we should verify type of loaded value
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
                HostUnit(host.clone()).into()
            }

            // The regular kernels
            Shape(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Broadcast(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            PrfKeyGen(op) => {
                use crate::kernels::{NgDispatchKernel, NgKernel};
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Nullary { closure } => {
                        closure(self, plc)?
                    }
                    _ => {
                        return Err(Error::Compilation(
                            "PrfKeyGen should be an unary kernel".to_string(),
                        ))
                    }
                }
            }
            Xor(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            And(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Or(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            BitExtract(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Sample(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            SampleSeeded(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Shl(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Shr(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            RingFixedpointAbs(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            RingFixedpointArgmax(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            RingFixedpointMean(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            RingFixedpointEncode(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            RingFixedpointDecode(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            RingInject(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Fill(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Share(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Reveal(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            TruncPr(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Msb(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Abs(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            RepToAdt(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            AddN(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Index(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            BitDecompose(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            BitCompose(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            ShlDim(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            AdtToRep(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            DeriveSeed(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Decrypt(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Constant(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Input(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Output(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Sqrt(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            FixedpointEncode(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            FixedpointDecode(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Diag(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            ExpandDims(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Squeeze(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Sign(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Identity(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Cast(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            AtLeast2D(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            IndexAxis(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Slice(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Ones(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Concat(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Reshape(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Transpose(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Dot(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Inverse(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Add(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Sub(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Mul(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Mean(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Neg(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Sum(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Div(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Mux(op) => {
                use crate::kernels::{NgDispatchKernel, NgKernel};
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Ternary { closure } => {
                        assert_eq!(operands.len(), 3);
                        let x2 = operands.pop().unwrap();
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0, x1, x2)?
                    }
                    _ => {
                        return Err(Error::Compilation(
                            "MuxOp should be a ternary kernel".to_string(),
                        ))
                    }
                }
            }
            Pow2(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Exp(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Sigmoid(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Log2(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Log(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Equal(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            EqualZero(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            LessThan(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            GreaterThan(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Maximum(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Softmax(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Argmax(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Demirror(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Mirror(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
        };
        Ok(kernel_output)
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
