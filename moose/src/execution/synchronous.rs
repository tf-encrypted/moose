use super::*;
use crate::error::{Error, Result};
use crate::execution::Identity;
use crate::host::*;
use crate::kernels::{DispatchKernel, PlacementSetupGen};
use crate::networking::LocalSyncNetworking;
use crate::replicated::*;
use crate::storage::LocalSyncStorage;
use crate::types::*;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

pub type SyncNetworkingImpl = Rc<dyn SyncNetworking>;
pub type SyncStorageImpl = Rc<dyn SyncStorage>;

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
            Broadcast(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
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
            FixedpointEncode(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointDecode(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointDiv(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointTruncPr(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FixedpointMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSlice(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostDiag(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostShlDim(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostAdd(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostSub(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostMul(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostDiv(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            HostDot(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            ExpandDims(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
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
            FloatingpointTranspose(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointInverse(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            FloatingpointMean(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
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
            Log2(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Less(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            GreaterThan(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Maximum(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
            Softmax(op) => DispatchKernel::compile(&op, plc)?(self, operands)?,
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
