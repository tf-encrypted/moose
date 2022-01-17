#![allow(unused_macros)]

use crate::computation::*;
use crate::error::{Error, Result};
use crate::networking::{AsyncNetworking, LocalAsyncNetworking, SyncNetworking};
use crate::storage::{AsyncStorage, LocalAsyncStorage, SyncStorage};

use derive_more::Display;
use futures::future::{Map, Shared};
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::rc::Rc;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::oneshot;

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

pub type AsyncSender = oneshot::Sender<Value>;

pub type AsyncReceiver = Shared<
    Map<
        oneshot::Receiver<Value>,
        fn(anyhow::Result<Value, oneshot::error::RecvError>) -> anyhow::Result<Value, ()>,
    >,
>;

pub type AsyncTask = tokio::task::JoinHandle<Result<()>>;

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

pub type SyncNetworkingImpl = Rc<dyn SyncNetworking>;
pub type SyncStorageImpl = Rc<dyn SyncStorage>;

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
}

pub type AsyncNetworkingImpl = Arc<dyn AsyncNetworking + Send + Sync>;

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

pub type Environment<V> = HashMap<String, V>;

pub type RoleAssignment = HashMap<Role, Identity>;

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
        session: &crate::kernels::AsyncSession,
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
            use crate::kernels::Session;
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
        let mut session_handles: Vec<crate::kernels::AsyncSessionHandle> = Vec::new();
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
            let moose_session = crate::kernels::AsyncSession::new(
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

            session_handles.push(crate::kernels::AsyncSessionHandle::for_session(
                &moose_session,
            ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::compile_passes;
    use crate::compilation::Pass;
    use crate::error::Error;
    use crate::host::{HostTensor, RawSeed, RawShape, Seed};
    use crate::networking::{AsyncNetworking, LocalAsyncNetworking};
    use crate::storage::{AsyncStorage, LocalAsyncStorage, LocalSyncStorage, SyncStorage};
    use crate::types::{
        HostFloat32Tensor, HostFloat64Tensor, HostInt64Tensor, HostRing128Tensor, HostRing64Tensor,
        HostShape, HostString,
    };
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
                let session = crate::kernels::SyncSession::from_storage(
                    SessionId::try_from("foobar").unwrap(),
                    arguments,
                    hashmap!(),
                    Rc::new(LocalSyncStorage::default()),
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
                Ok(outputs)
            }
        }
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_eager_executor(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let mut definition = String::from(
            r#"key = PrimPrfKeyGen: () -> PrfKey () @Host(alice)
        seed = PrimDeriveSeed {sync_key = [1, 2, 3]}: (PrfKey) -> Seed (key) @Host(alice)
        shape = Constant{value = Shape([2, 3])}: () -> Shape @Host(alice)
        "#,
        );
        let body = (0..100)
            .map(|i| {
                format!(
                    "x{} = RingSampleSeeded{{}}: (Shape, Seed) -> Ring64Tensor (shape, seed) @Host(alice)",
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
        let source = r#"key = Constant{value=PrfKey(00000000000000000000000000000000)}: () -> PrfKey @Host(alice)
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
        let source = r#"seed = Constant{value=Seed(00000000000000000000000000000000)}: () -> Seed @Host(alice)
        xshape = Constant{value=Shape([2, 2])}: () -> Shape @Host(alice)
        sampled = RingSampleSeeded{}: (Shape, Seed) -> Ring64Tensor (xshape, seed) @Host(alice)
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
        saved_uri = Constant{value = String("saved_data")}: () -> String () @Host(alice)
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
                let executor = crate::kernels::TestSyncExecutor::default();
                let session = crate::kernels::SyncSession::from_storage(
                    SessionId::try_from("foobar").unwrap(),
                    arguments,
                    hashmap!(),
                    storage.clone(),
                );
                let _outputs = executor.run_computation(&source.try_into()?, &session)?;
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
        let source_template = r#"x_0 = Constant{value=Int64Tensor([[1,2], [3,4]])}: () -> Int64Tensor @Host(alice)
        x_1 = Constant{value=Int64Tensor([[5, 6], [7,8]])}: () -> Int64Tensor @Host(alice)
        concatenated = Concat {axis=test_axis}: [Int64Tensor] -> Int64Tensor (x_0, x_1) @Host(alice)
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
        let source_template = r#"
        x0 = Constant{value=Int64Tensor([5])}: () -> Int64Tensor @Host(alice)
        x1 = Constant{value=Int64Tensor([3])}: () -> Int64Tensor @Host(bob)
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
                let computation =
                    compile_passes(&computation, &[Pass::Networking, Pass::Toposort])?;
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
        let source = r#"
        x0 = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(alice)
        x1 = Constant{value=Float32Tensor([[1.0, 0.0], [0.0, 1.0]])}: () -> Float32Tensor @Host(bob)
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
                let computation =
                    compile_passes(&computation, &[Pass::Networking, Pass::Toposort])?;
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

    #[cfg(feature = "blas")]
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
        let template = r#"
        s = Constant{value=Shape([2, 2])}: () -> Shape @Host(alice)
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
        let source = r#"
        x = Constant{value = Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(alice)
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
        let source = r#"x = Constant{value = Shape([2, 3, 4, 5])}: () -> Shape @Host(alice)
        slice = HostSlice {slice = {start = 1, end = 3}}: (Shape) -> Shape (x) @Host(alice)
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
        let source = r#"
        x = Constant{value = Int64Tensor([1, 2])}: () -> Int64Tensor @Host(alice)
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
            axis_test.map_or_else(|| "{}".to_string(), |v| format!("{{axis={}}}", v));

        let source = format!(
            r#"
            s = Constant{{value=Float32Tensor([[1, 2], [3, 4]])}}: () -> Float32Tensor @Host(alice)
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
        let source = r#"
        s = Constant{value=Int64Tensor([[1,2], [3, 4]])}: () -> Int64Tensor @Host(alice)
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
            r#"
            x = Constant{{value=Float64Tensor([1.0, 1.0, 1.0])}}: () -> Float64Tensor @Host(alice)
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
            r#"
            x =  Constant{{value=Ring64Tensor([3])}}: () -> Ring64Tensor @Host(alice)
            y = Constant{{value=Ring64Tensor([2])}}: () -> Ring64Tensor @Host(alice)
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
            r#"x = Constant{{value={}}}: () -> Ring64Tensor @Host(alice)
        y = Constant{{value={}}}: () -> Ring64Tensor @Host(alice)
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
            r#"shape = Constant{{value=Shape([{shape}])}}: () -> Shape @Host(alice)
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
        let source = r#"x = Constant{value=Ring64Tensor([[1, 2], [3, 4]])}: () -> Ring64Tensor @Host(alice)
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
        let template_source = r#"x = Constant{value=Ring64Tensor([4, 4])}: () -> Ring64Tensor @Host(alice)
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

    fn _create_async_session(
        networking: &Arc<dyn Send + Sync + AsyncNetworking>,
        exec_storage: &Arc<dyn Send + Sync + AsyncStorage>,
        role_assignments: HashMap<Role, Identity>,
    ) -> crate::kernels::AsyncSession {
        crate::kernels::AsyncSession::new(
            SessionId::try_from("foobar").unwrap(),
            hashmap!(),
            role_assignments,
            Arc::clone(networking),
            Arc::clone(exec_storage),
            Arc::new(Placement::Host(HostPlacement {
                owner: "localhost".into(),
            })),
        )
    }

    #[test]
    fn test_duplicate_session_ids() {
        let source = r#"key = Constant{value=PrfKey(00000000000000000000000000000000)}: () -> PrfKey @Host(alice)
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

        let moose_session =
            _create_async_session(&networking, &exec_storage, valid_role_assignments.clone());

        let computation: Computation = source.try_into().unwrap();
        let own_identity = identity;

        executor
            .run_computation(
                &computation,
                &valid_role_assignments,
                &own_identity,
                &moose_session,
            )
            .unwrap();

        let moose_session =
            _create_async_session(&networking, &exec_storage, valid_role_assignments.clone());

        let expected =
            Error::SessionAlreadyExists(format!("{}", SessionId::try_from("foobar").unwrap()));

        let res = executor.run_computation(
            &computation,
            &valid_role_assignments,
            &own_identity,
            &moose_session,
        );

        if let Err(e) = res {
            assert_eq!(e.to_string(), expected.to_string());
        } else {
            panic!("expected session already exists error")
        }
    }

    fn _run_new_async_computation_test(
        computation: Computation,
        storage_mapping: HashMap<String, HashMap<String, Value>>,
        role_assignments: HashMap<String, String>,
        arguments: HashMap<String, Value>,
    ) -> std::result::Result<HashMap<String, Value>, anyhow::Error> {
        let valid_role_assignments = role_assignments
            .into_iter()
            .map(|arg| (Role::from(arg.1), Identity::from(arg.0)))
            .collect::<HashMap<Role, Identity>>();
        let mut executor = AsyncTestRuntime::new(storage_mapping);
        let outputs =
            executor.evaluate_computation(&computation, valid_role_assignments, arguments)?;
        Ok(outputs)
    }

    #[test]
    fn test_new_async_session() -> std::result::Result<(), anyhow::Error> {
        let source = r#"key = Constant{value=PrfKey(00000000000000000000000000000000)}: () -> PrfKey @Host(alice)
        seed = PrimDeriveSeed {sync_key = [1, 2, 3]}: (PrfKey) -> Seed (key) @Host(alice)
        output = Output: (Seed) -> Seed (seed) @Host(alice)"#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let role_assignments: HashMap<String, String> =
            hashmap!("alice".to_string() => "alice".to_string());
        let outputs = _run_new_async_computation_test(
            source.try_into()?,
            storage_mapping,
            role_assignments,
            arguments,
        )?;

        let seed: Seed = (outputs.get("output").unwrap().clone()).try_into()?;
        assert_eq!(
            seed.0,
            RawSeed([224, 87, 133, 2, 90, 170, 32, 253, 25, 80, 93, 74, 122, 196, 50, 1])
        );
        Ok(())
    }
}
