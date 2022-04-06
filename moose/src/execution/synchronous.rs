//! Synchronous/eager execution of computations

use super::{Identity, Operands, RuntimeSession, Session, SetupGeneration};
use crate::computation::*;
use crate::error::{Error, Result};
use crate::host::*;
use crate::kernels::{DispatchKernel, Kernel};
use crate::kernels::{NgDispatchKernel, NgKernel};
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
            Shape(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Broadcast(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            PrfKeyGen(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Nullary { closure } => closure(self, plc)?,
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Xor(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            And(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Or(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            BitExtract(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Sample(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            SampleSeeded(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Shl(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Shr(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingFixedpointAbs(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingFixedpointArgmax(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingFixedpointMean(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingFixedpointEncode(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingFixedpointDecode(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingInject(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            // this op skips the modelled_kernel macro
            Fill(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Share(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Reveal(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            TruncPr(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Msb(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Abs(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RepToAdt(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            AddN(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Index(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            BitDecompose(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            BitCompose(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            ShlDim(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            AdtToRep(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            DeriveSeed(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Decrypt(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Constant(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Input(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Output(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Sqrt(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            FixedpointEncode(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            FixedpointDecode(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Diag(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            ExpandDims(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Squeeze(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Sign(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Identity(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Cast(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            AtLeast2D(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            IndexAxis(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Slice(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Ones(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Concat(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Reshape(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Transpose(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Dot(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Inverse(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Add(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Sub(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Mul(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Mean(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Neg(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Sum(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Div(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Mux(op) => {
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
            Pow2(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Exp(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Sigmoid(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Log2(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Log(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Equal(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            EqualZero(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            LessThan(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            GreaterThan(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Maximum(op) => DispatchKernel::compile(op, plc)?(self, operands)?,
            Softmax(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Argmax(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Demirror(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Mirror(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(self, plc, x0)?
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
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
