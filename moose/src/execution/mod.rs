//! Support for executing computations.

use crate::computation::{Operator, Placement, Role, SessionId, Value};
use crate::error::Result;
use derive_more::Display;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "async_execute")]
pub mod asynchronous;
pub mod context;
pub mod grpc;
pub(crate) mod kernel_helpers;
#[cfg(feature = "compile")]
pub mod symbolic;
#[cfg(feature = "sync_execute")]
pub mod synchronous;

#[cfg(feature = "async_execute")]
pub use asynchronous::*;
pub use context::ExecutionContext;
#[cfg(feature = "compile")]
pub use symbolic::*;
#[cfg(feature = "sync_execute")]
pub use synchronous::*;

pub type Operands<V> = Vec<V>;

/// General session trait determining basic properties for session objects.
pub trait Session {
    type Value;
    fn execute(
        &self,
        op: &Operator,
        plc: &Placement,
        operands: Operands<Self::Value>,
    ) -> Result<Self::Value>;
}

pub(crate) trait SetupGeneration<P> {
    type Setup;
    fn setup(&self, plc: &P) -> Result<Arc<Self::Setup>>;
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

/// Runtime identity of player.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Display, Serialize, Deserialize)]
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

#[cfg(all(feature = "async_execute", feature = "sync_execute"))]
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "compile")]
    use crate::compilation::{compile, Pass};
    use crate::error::Error;
    use crate::execution::{SyncSession, TestSyncExecutor};
    use crate::host::{HostPlacement, HostSeed, HostTensor, RawSeed, RawShape};
    use crate::networking::{local::LocalAsyncNetworking, AsyncNetworking};
    use crate::prelude::*;
    use crate::storage::{
        local::LocalAsyncStorage, local::LocalSyncStorage, AsyncStorage, SyncStorage,
    };
    use itertools::Itertools;
    use maplit::hashmap;
    use ndarray::prelude::*;
    use rstest::rstest;
    use std::convert::{TryFrom, TryInto};
    use std::rc::Rc;
    use tokio::runtime::Runtime;

    fn _run_computation_test(
        computation: Computation,
        storage_mapping: HashMap<String, HashMap<String, Value>>,
        arguments: HashMap<String, Value>,
        run_async: bool,
    ) -> std::result::Result<HashMap<String, Value>, anyhow::Error> {
        match run_async {
            false => {
                let executor = TestSyncExecutor::default();
                let session = SyncSession::from_storage(
                    SessionId::try_from("foobar").unwrap(),
                    arguments,
                    hashmap!(),
                    Rc::new(LocalSyncStorage::default()),
                );
                let outputs = executor.run_computation(&computation, &session)?;
                Ok(outputs)
            }
            true => {
                let mut executor = AsyncTestRuntime::new(storage_mapping);
                let outputs = executor.evaluate_computation(&computation, arguments)?;
                Ok(outputs)
            }
        }
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_eager_executor(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let mut definition = String::from(
            r#"key = PrfKeyGen: () -> HostPrfKey () @Host(alice)
        seed = DeriveSeed {sync_key = [1, 2, 3]}: (HostPrfKey) -> HostSeed (key) @Host(alice)
        shape = Constant{value = HostShape([2, 3])}: () -> HostShape @Host(alice)
        "#,
        );
        let body = (0..100)
            .map(|i| {
                format!(
                    "x{} = SampleSeeded{{}}: (HostShape, HostSeed) -> HostRing64Tensor (shape, seed) @Host(alice)",
                    i
                )
            })
            .join("\n");
        definition.push_str(&body);
        definition
            .push_str(r#"
            z = Output{tag = "output_0"}: (HostRing64Tensor) -> HostRing64Tensor (x0) @Host(alice)"#);

        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs = _run_computation_test(
            definition.try_into()?,
            storage_mapping,
            arguments,
            run_async,
        )?;

        assert_eq!(outputs.keys().collect::<Vec<_>>(), vec!["output_0"]);
        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_constants_derive_seed(
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = r#"key = Constant{value=HostPrfKey(00000000000000000000000000000000)}: () -> HostPrfKey @Host(alice)
        seed = DeriveSeed {sync_key = [1, 2, 3]}: (HostPrfKey) -> HostSeed (key) @Host(alice)
        output = Output{tag = "output_0"}: (HostSeed) -> HostSeed (seed) @Host(alice)"#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        let seed: HostSeed = (outputs.get("output_0").unwrap().clone()).try_into()?;
        if run_async {
            // Async session uses a random session id, so we can not predict the output.
            // But at least it should not be empty
            assert_ne!(
                seed.0,
                RawSeed([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            );
        } else {
            assert_eq!(
                seed.0,
                RawSeed([79, 203, 243, 208, 77, 199, 116, 216, 2, 206, 173, 36, 20, 204, 200, 146])
            );
        }
        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_constants_sample_ring(
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = r#"seed = Constant{value=HostSeed(00000000000000000000000000000000)}: () -> HostSeed @Host(alice)
        xshape = Constant{value=HostShape([2, 2])}: () -> HostShape @Host(alice)
        sampled = SampleSeeded{}: (HostShape, HostSeed) -> HostRing64Tensor (xshape, seed) @Host(alice)
        shape = Shape: (HostRing64Tensor) -> HostShape (sampled) @Host(alice)
        output = Output{tag = "output_0"}: (HostShape) -> HostShape (shape) @Host(alice)
        "#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        let output: HostShape = (outputs.get("output_0").unwrap().clone()).try_into()?;
        assert_eq!(output.0, RawShape(vec![2, 2]));
        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_standard_input(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Input {arg_name = "x"}: () -> HostInt64Tensor @Host(alice)
        y = Input {arg_name = "y"}: () -> HostInt64Tensor @Host(alice)
        z = Add: (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor (x, y) @Host(alice)
        output = Output{tag = "output_0"}: (HostInt64Tensor) -> HostInt64Tensor (z) @Host(alice)
        "#;
        let x: Value = "HostInt64Tensor([5]) @Host(alice)".try_into()?;
        let y: Value = "HostInt64Tensor([10]) @Host(alice)".try_into()?;
        let arguments: HashMap<String, Value> = hashmap!("x".to_string() => x, "y".to_string()=> y);
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        let z: HostInt64Tensor = (outputs.get("output_0").unwrap().clone()).try_into()?;
        let expected: Value = "HostInt64Tensor([15]) @Host(alice)".try_into()?;
        assert_eq!(expected, z.into());
        Ok(())
    }

    #[rstest]
    #[case("HostInt64Tensor([8]) @Host(alice)", true)]
    #[case("HostInt32Tensor([8]) @Host(alice)", true)]
    #[case("HostFloat32Tensor([8]) @Host(alice)", true)]
    #[case("HostFloat64Tensor([8]) @Host(alice)", true)]
    #[case("HostInt64Tensor([8]) @Host(alice)", false)]
    #[case("HostInt32Tensor([8]) @Host(alice)", false)]
    #[case("HostFloat32Tensor([8]) @Host(alice)", false)]
    #[case("HostFloat64Tensor([8]) @Host(alice)", false)]
    fn test_load_save(
        #[case] input_data: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        use crate::textual::ToTextual;

        let data_type_str = input_data.ty().to_textual();
        let source_template = r#"x_uri = Input {arg_name="x_uri"}: () -> HostString () @Host(alice)
        x_query = Input {arg_name="x_query"}: () -> HostString () @Host(alice)
        saved_uri = Constant{value = HostString("saved_data")}: () -> HostString () @Host(alice)
        x = Load: (HostString, HostString) -> TensorType (x_uri, x_query) @Host(alice)
        save = Save: (HostString, TensorType) -> HostUnit (saved_uri, x) @Host(alice)
        output = Output{tag = "output_0"}: (HostUnit) -> HostUnit (save) @Host(alice)
        "#;
        let source = source_template.replace("TensorType", &data_type_str);
        let plc = HostPlacement::from("alice");

        let arguments: HashMap<String, Value> = hashmap!("x_uri".to_string()=> HostString("input_data".to_string(), plc.clone()).into(),
            "x_query".to_string() => HostString("".to_string(), plc.clone()).into(),
            "saved_uri".to_string() => HostString("saved_data".to_string(), plc).into());

        let saved_data = match run_async {
            true => {
                let storage_mapping: HashMap<String, HashMap<String, Value>> = hashmap!("alice".to_string() => hashmap!("input_data".to_string() => input_data.clone()));
                let mut executor = AsyncTestRuntime::new(storage_mapping);
                let _outputs = executor.evaluate_computation(&source.try_into()?, arguments)?;

                executor.read_value_from_storage(
                    Identity::from("alice".to_string()),
                    "saved_data".to_string(),
                )?
            }
            false => {
                let store: HashMap<String, Value> =
                    hashmap!("input_data".to_string() => input_data.clone());
                let storage: Rc<dyn SyncStorage> = Rc::new(LocalSyncStorage::from_hashmap(store));
                let executor = TestSyncExecutor::default();
                let session = SyncSession::from_storage(
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

    #[rstest]
    #[case(
        "0",
        "HostInt64Tensor([[1, 2], [3, 4], [5, 6], [7, 8]]) @Host(alice)",
        true
    )]
    #[case(
        "1",
        "HostInt64Tensor([[1, 2, 5, 6], [3, 4, 7, 8]]) @Host(alice)",
        true
    )]
    #[case(
        "0",
        "HostInt64Tensor([[1, 2], [3, 4], [5, 6], [7, 8]]) @Host(alice)",
        false
    )]
    #[case(
        "1",
        "HostInt64Tensor([[1, 2, 5, 6], [3, 4, 7, 8]]) @Host(alice)",
        false
    )]
    fn test_standard_concatenate(
        #[case] axis: usize,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source_template = r#"x_0 = Constant{value=HostInt64Tensor([[1,2], [3,4]])}: () -> HostInt64Tensor @Host(alice)
        x_1 = Constant{value=HostInt64Tensor([[5, 6], [7,8]])}: () -> HostInt64Tensor @Host(alice)
        concatenated = Concat {axis=test_axis}: [HostInt64Tensor] -> HostInt64Tensor (x_0, x_1) @Host(alice)
        output = Output{tag = "output_0"}: (HostInt64Tensor) -> HostInt64Tensor (concatenated) @Host(alice)
        "#;
        let source = source_template.replace("test_axis", &axis.to_string());
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        let concatenated: HostInt64Tensor =
            (outputs.get("output_0").unwrap().clone()).try_into()?;
        assert_eq!(expected_result, concatenated.into());
        Ok(())
    }

    #[cfg(feature = "compile")]
    #[rstest]
    #[case("Add", "HostInt64Tensor([8]) @Host(alice)", true)]
    #[case("Sub", "HostInt64Tensor([2]) @Host(alice)", true)]
    #[case("Mul", "HostInt64Tensor([15]) @Host(alice)", true)]
    #[case("Div", "HostInt64Tensor([1]) @Host(alice)", true)]
    #[case("Add", "HostInt64Tensor([8]) @Host(alice)", false)]
    #[case("Sub", "HostInt64Tensor([2]) @Host(alice)", false)]
    #[case("Mul", "HostInt64Tensor([15]) @Host(alice)", false)]
    #[case("Div", "HostInt64Tensor([1]) @Host(alice)", false)]
    fn test_standard_op(
        #[case] test_op: String,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source_template = r#"
        x0 = Constant{value=HostInt64Tensor([5])}: () -> HostInt64Tensor @Host(alice)
        x1 = Constant{value=HostInt64Tensor([3])}: () -> HostInt64Tensor @Host(bob)
        res = StdOp: (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor (x0, x1) @Host(alice)
        output = Output{tag = "output_0"}: (HostInt64Tensor) -> HostInt64Tensor (res) @Host(alice)
        "#;
        let source = source_template.replace("StdOp", &test_op);
        let computation: Computation = source.try_into()?;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!(), "bob".to_string()=>hashmap!());

        let outputs = match run_async {
            true => {
                let computation =
                    compile(computation, Some(vec![Pass::Networking, Pass::Toposort]))?;
                _run_computation_test(computation, storage_mapping, arguments, run_async)?
            }
            false => _run_computation_test(computation, storage_mapping, arguments, run_async)?,
        };

        let res: HostInt64Tensor = (outputs.get("output_0").unwrap().clone()).try_into()?;
        assert_eq!(expected_result, res.into());
        Ok(())
    }

    #[cfg(feature = "compile")]
    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_standard_dot(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        x0 = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        x1 = Constant{value=HostFloat32Tensor([[1.0, 0.0], [0.0, 1.0]])}: () -> HostFloat32Tensor @Host(bob)
        res = Dot: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x0, x1) @Host(alice)
        output = Output{tag = "output_0"}: (HostFloat32Tensor) -> HostFloat32Tensor (res) @Host(alice)
        "#;
        let computation: Computation = source.try_into()?;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!(), "bob".to_string()=>hashmap!());

        let outputs = match run_async {
            true => {
                let computation =
                    compile(computation, Some(vec![Pass::Networking, Pass::Toposort]))?;
                _run_computation_test(computation, storage_mapping, arguments, run_async)?
            }
            false => _run_computation_test(computation, storage_mapping, arguments, run_async)?,
        };

        let expected_output: Value = HostTensor::<f32>(
            array![[1.0, 2.0], [3.0, 4.0]].into_shared().into_dyn(),
            HostPlacement::from("alice"),
        )
        .into();
        assert_eq!(outputs["output_0"], expected_output);
        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_standard_inverse(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value=HostFloat32Tensor([[3.0, 2.0], [2.0, 3.0]])} : () -> HostFloat32Tensor @Host(alice)
        x_inv = Inverse : (HostFloat32Tensor) -> HostFloat32Tensor (x) @Host(alice)
        output = Output{tag = "output_0"}: (HostFloat32Tensor) -> HostFloat32Tensor (x_inv) @Host(alice)
        "#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        let expected_output = HostTensor::<f32>(
            array![[0.6, -0.40000004], [-0.40000004, 0.6]]
                .into_shared()
                .into_dyn(),
            HostPlacement::from("alice"),
        );
        let x_inv: HostFloat32Tensor = (outputs.get("output_0").unwrap().clone()).try_into()?;
        assert_eq!(expected_output, x_inv);
        Ok(())
    }

    #[rstest]
    #[case("HostFloat32Tensor", true)]
    #[case("HostFloat64Tensor", true)]
    #[case("HostInt64Tensor", true)]
    #[case("HostFloat32Tensor", false)]
    #[case("HostFloat64Tensor", false)]
    #[case("HostInt64Tensor", false)]
    fn test_standard_ones(
        #[case] dtype: String,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let template = r#"
        s = Constant{value=HostShape([2, 2])}: () -> HostShape @Host(alice)
        r = Ones : (HostShape) -> dtype (s) @Host(alice)
        output = Output{tag = "output_0"} : (dtype) -> dtype (r) @Host(alice)
        "#;
        let source = template.replace("dtype", &dtype);
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        match dtype.as_str() {
            "HostFloat32Tensor" => {
                let r: HostFloat32Tensor = (outputs.get("output_0").unwrap().clone()).try_into()?;
                assert_eq!(
                    r,
                    HostTensor::<f32>(
                        array![[1.0, 1.0], [1.0, 1.0]].into_shared().into_dyn(),
                        HostPlacement::from("alice"),
                    )
                );
                Ok(())
            }
            "HostFloat64Tensor" => {
                let r: HostFloat64Tensor = (outputs.get("output_0").unwrap().clone()).try_into()?;
                assert_eq!(
                    r,
                    HostTensor::<f64>(
                        array![[1.0, 1.0], [1.0, 1.0]].into_shared().into_dyn(),
                        HostPlacement::from("alice"),
                    )
                );
                Ok(())
            }
            "HostInt64Tensor" => {
                let r: HostInt64Tensor = (outputs.get("output_0").unwrap().clone()).try_into()?;
                assert_eq!(
                    r,
                    HostTensor::<i64>(
                        array![[1, 1], [1, 1]].into_shared().into_dyn(),
                        HostPlacement::from("alice"),
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
        x = Constant{value = HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        shape = Shape: (HostFloat32Tensor) -> HostShape (x) @Host(alice)
        output = Output{tag = "output_0"}: (HostShape) -> HostShape (shape) @Host(alice)"#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        let actual_shape: HostShape = (outputs.get("output_0").unwrap().clone()).try_into()?;
        let actual_raw_shape = actual_shape.0;
        let expected_raw_shape = RawShape(vec![2, 2]);
        assert_eq!(actual_raw_shape, expected_raw_shape);

        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_shape_slice(#[case] run_async: bool) -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value = HostShape([2, 3, 4, 5])}: () -> HostShape @Host(alice)
        slice = Slice {slice = {start = 1, end = 3}}: (HostShape) -> HostShape (x) @Host(alice)
        output = Output{tag = "output_0"}: (HostShape) -> HostShape (slice) @Host(alice)"#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;
        let res: HostShape = (outputs.get("output_0").unwrap().clone()).try_into()?;
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
        x = Constant{value = HostInt64Tensor([1, 2])}: () -> HostInt64Tensor @Host(alice)
        expand_dims = ExpandDims {axis = [1]}: (HostInt64Tensor) -> HostInt64Tensor (x) @Host(alice)
        output = Output{tag = "output_0"}: (HostInt64Tensor) -> HostInt64Tensor (expand_dims) @Host(alice)"#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        let res: HostInt64Tensor = (outputs.get("output_0").unwrap().clone()).try_into()?;
        let actual_shape = res.shape().0;
        let expected_shape = RawShape(vec![2, 1]);
        assert_eq!(expected_shape, actual_shape);
        Ok(())
    }

    #[rstest]
    #[case("Sum", None, "Float32(10.0) @Host(alice)", true, true)]
    #[case(
        "Sum",
        Some(0),
        "HostFloat32Tensor([4.0, 6.0]) @Host(alice)",
        false,
        true
    )]
    #[case(
        "Sum",
        Some(1),
        "HostFloat32Tensor([3.0, 7.0]) @Host(alice)",
        false,
        true
    )]
    #[case("Mean", None, "Float32(2.5) @Host(alice)", true, true)]
    #[case(
        "Mean",
        Some(0),
        "HostFloat32Tensor([2.0, 3.0]) @Host(alice)",
        false,
        true
    )]
    #[case(
        "Mean",
        Some(1),
        "HostFloat32Tensor([1.5, 3.5]) @Host(alice)",
        false,
        true
    )]
    #[case("Sum", None, "Float32(10.0) @Host(alice)", true, false)]
    #[case(
        "Sum",
        Some(0),
        "HostFloat32Tensor([4.0, 6.0]) @Host(alice)",
        false,
        false
    )]
    #[case(
        "Sum",
        Some(1),
        "HostFloat32Tensor([3.0, 7.0]) @Host(alice)",
        false,
        false
    )]
    #[case("Mean", None, "Float32(2.5) @Host(alice)", true, false)]
    #[case(
        "Mean",
        Some(0),
        "HostFloat32Tensor([2.0, 3.0]) @Host(alice)",
        false,
        false
    )]
    #[case(
        "Mean",
        Some(1),
        "HostFloat32Tensor([1.5, 3.5]) @Host(alice)",
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
            s = Constant{{value=HostFloat32Tensor([[1, 2], [3, 4]])}}: () -> HostFloat32Tensor @Host(alice)
            r = {} {}: (HostFloat32Tensor) -> HostFloat32Tensor (s) @Host(alice)
            output = Output{{tag = "output_0"}} : (HostFloat32Tensor) -> HostFloat32Tensor (r) @Host(alice)
        "#,
            reduce_op_test, axis_str
        );
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        let comp_result = outputs
            .get("output_0")
            .ok_or_else(|| anyhow::anyhow!("Expected result missing"))?;

        if unwrap_flag {
            if let Value::HostFloat32Tensor(x) = comp_result {
                let shaped_result = x
                    .clone()
                    .reshape(HostShape(RawShape(vec![1]), HostPlacement::from("alice")));
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
    #[case("HostInt64Tensor([[1, 3], [2, 4]]) @Host(alice)", true)]
    #[case("HostInt64Tensor([[1, 3], [2, 4]]) @Host(alice)", false)]
    fn test_standard_transpose(
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        s = Constant{value=HostInt64Tensor([[1,2], [3, 4]])}: () -> HostInt64Tensor @Host(alice)
        r = Transpose : (HostInt64Tensor) -> HostInt64Tensor (s) @Host(alice)
        output = Output{tag = "output_0"} : (HostInt64Tensor) -> HostInt64Tensor (r) @Host(alice)
        "#;
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        let comp_result: HostInt64Tensor = (outputs.get("output_0").unwrap().clone()).try_into()?;
        assert_eq!(expected_result, comp_result.into());
        Ok(())
    }

    #[rstest]
    #[case(true, "HostFloat64Tensor([[1.0], [1.0], [1.0]]) @Host(alice)", true)]
    #[case(false, "HostFloat64Tensor([[1.0, 1.0, 1.0]]) @Host(alice)", true)]
    #[case(true, "HostFloat64Tensor([[1.0], [1.0], [1.0]]) @Host(alice)", false)]
    #[case(false, "HostFloat64Tensor([[1.0, 1.0, 1.0]]) @Host(alice)", false)]
    fn test_standard_atleast_2d(
        #[case] to_column_vector: bool,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = format!(
            r#"
            x = Constant{{value=HostFloat64Tensor([1.0, 1.0, 1.0])}}: () -> HostFloat64Tensor @Host(alice)
        res = AtLeast2D {{ to_column_vector = {} }} : (HostFloat64Tensor) -> HostFloat64Tensor (x) @Host(alice)
        output = Output{{tag = "output_0"}} : (HostFloat64Tensor) -> HostFloat64Tensor (res) @Host(alice)
        "#,
            to_column_vector
        );
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        let comp_result: HostFloat64Tensor =
            (outputs.get("output_0").unwrap().clone()).try_into()?;
        assert_eq!(expected_result, comp_result.into());
        Ok(())
    }

    #[rstest]
    #[case("Add", "HostRing64Tensor([5]) @Host(alice)", true)]
    #[case("Mul", "HostRing64Tensor([6]) @Host(alice)", true)]
    #[case("Sub", "HostRing64Tensor([1]) @Host(alice)", true)]
    #[case("Add", "HostRing64Tensor([5]) @Host(alice)", false)]
    #[case("Mul", "HostRing64Tensor([6]) @Host(alice)", false)]
    #[case("Sub", "HostRing64Tensor([1]) @Host(alice)", false)]
    fn test_ring_binop_invocation(
        #[case] test_op: String,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = format!(
            r#"
            x =  Constant{{value=HostRing64Tensor([3])}}: () -> HostRing64Tensor @Host(alice)
            y = Constant{{value=HostRing64Tensor([2])}}: () -> HostRing64Tensor @Host(alice)
            res = {} : (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor (x, y) @Host(alice)
            output = Output{{tag = "output_0"}} : (HostRing64Tensor) -> HostRing64Tensor (res) @Host(alice)
            "#,
            test_op
        );
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        let comp_result: HostRing64Tensor =
            (outputs.get("output_0").unwrap().clone()).try_into()?;
        assert_eq!(expected_result, comp_result.into());
        Ok(())
    }

    #[rstest]
    #[case(
        "HostRing64Tensor",
        "HostRing64Tensor([[1, 2], [3, 4]])",
        "HostRing64Tensor([[1, 0], [0, 1]])",
        "HostRing64Tensor([[1, 2], [3, 4]]) @Host(alice)",
        true
    )]
    #[case(
        "HostRing64Tensor",
        "HostRing64Tensor([[1, 2], [3, 4]])",
        "HostRing64Tensor([1, 1])",
        "HostRing64Tensor([3, 7]) @Host(alice)",
        true
    )]
    #[case(
        "HostRing64Tensor",
        "HostRing64Tensor([1, 1])",
        "HostRing64Tensor([[1, 2], [3, 4]])",
        "HostRing64Tensor([4, 6]) @Host(alice)",
        true
    )]
    #[case(
        "HostRing64Tensor",
        "HostRing64Tensor([[1, 2], [3, 4]])",
        "HostRing64Tensor([[1, 0], [0, 1]])",
        "HostRing64Tensor([[1, 2], [3, 4]]) @Host(alice)",
        false
    )]
    #[case(
        "HostRing64Tensor",
        "HostRing64Tensor([[1, 2], [3, 4]])",
        "HostRing64Tensor([1, 1])",
        "HostRing64Tensor([3, 7]) @Host(alice)",
        false
    )]
    #[case(
        "HostRing64Tensor",
        "HostRing64Tensor([1, 1])",
        "HostRing64Tensor([[1, 2], [3, 4]])",
        "HostRing64Tensor([4, 6]) @Host(alice)",
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
            r#"x = Constant{{value={}}}: () -> HostRing64Tensor @Host(alice)
        y = Constant{{value={}}}: () -> HostRing64Tensor @Host(alice)
        res = Dot : (HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor (x, y) @Host(alice)
        output = Output{{tag = "output_0"}} : (HostRing64Tensor) -> HostRing64Tensor (res) @Host(alice)
        "#,
            x_str, y_str
        );
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        match type_str.as_str() {
            "HostRing64Tensor" => {
                let comp_result: HostRing64Tensor =
                    (outputs.get("output_0").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            "HostRing128Tensor" => {
                let comp_result: HostRing128Tensor =
                    (outputs.get("output_0").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            _ => Err(anyhow::anyhow!("Failed to parse test case type")),
        }
    }

    #[rstest]
    #[case("Ring64", "2", "HostRing64Tensor([1, 1]) @Host(alice)", true)]
    #[case("Ring128", "2", "HostRing128Tensor([1, 1]) @Host(alice)", true)]
    #[case("Ring64", "2, 1", "HostRing64Tensor([[1], [1]]) @Host(alice)", true)]
    #[case(
        "Ring64",
        "2, 2",
        "HostRing64Tensor([[1, 1], [1, 1]]) @Host(alice)",
        true
    )]
    #[case("Ring64", "1, 2", "HostRing64Tensor([[1, 1]]) @Host(alice)", true)]
    #[case(
        "Ring128",
        "2, 3",
        "HostRing128Tensor([[1, 1, 1], [1, 1, 1]]) @Host(alice)",
        true
    )]
    #[case("Ring64", "2", "HostRing64Tensor([1, 1]) @Host(alice)", false)]
    #[case("Ring128", "2", "HostRing128Tensor([1, 1]) @Host(alice)", false)]
    #[case("Ring64", "2, 1", "HostRing64Tensor([[1], [1]]) @Host(alice)", false)]
    #[case(
        "Ring64",
        "2, 2",
        "HostRing64Tensor([[1, 1], [1, 1]]) @Host(alice)",
        false
    )]
    #[case("Ring64", "1, 2", "HostRing64Tensor([[1, 1]]) @Host(alice)", false)]
    #[case(
        "Ring128",
        "2, 3",
        "HostRing128Tensor([[1, 1, 1], [1, 1, 1]]) @Host(alice)",
        false
    )]
    fn test_fill(
        #[case] type_str: String,
        #[case] shape_str: String,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let source = format!(
            r#"shape = Constant{{value=HostShape([{shape}])}}: () -> HostShape @Host(alice)
        res = Fill {{value = {t}(1)}} : (HostShape) -> Host{t}Tensor (shape) @Host(alice)
        output = Output{{tag = "output_0"}} : (Host{t}Tensor) -> Host{t}Tensor (res) @Host(alice)
        "#,
            t = type_str,
            shape = shape_str,
        );
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        match type_str.as_str() {
            "Ring64" => {
                let comp_result: HostRing64Tensor =
                    (outputs.get("output_0").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            "Ring128" => {
                let comp_result: HostRing128Tensor =
                    (outputs.get("output_0").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            _ => Err(anyhow::anyhow!("Failed to parse test case type")),
        }
    }

    #[rstest]
    #[case("HostRing64Tensor", "HostRing64Tensor([2, 2]) @Host(alice)", true)]
    #[case("HostRing128Tensor", "HostRing128Tensor([2, 2]) @Host(alice)", true)]
    #[case("HostRing64Tensor", "HostRing64Tensor([2, 2]) @Host(alice)", false)]
    #[case("HostRing128Tensor", "HostRing128Tensor([2, 2]) @Host(alice)", false)]
    fn test_ring_bitwise_ops(
        #[case] type_str: String,
        #[case] expected_result: Value,
        #[case] run_async: bool,
    ) -> std::result::Result<(), anyhow::Error> {
        let template_source = r#"x = Constant{value=HostRing64Tensor([4, 4])}: () -> HostRing64Tensor @Host(alice)
        res = Shr {amount = 1}: (HostRing64Tensor) -> HostRing64Tensor (x) @Host(alice)
        output = Output{tag = "output_0"}: (HostRing64Tensor) -> HostRing64Tensor (res) @Host(alice)
        "#;
        let source = template_source.replace("HostRing64Tensor", type_str.as_str());
        let arguments: HashMap<String, Value> = hashmap!();
        let storage_mapping: HashMap<String, HashMap<String, Value>> =
            hashmap!("alice".to_string() => hashmap!());
        let outputs =
            _run_computation_test(source.try_into()?, storage_mapping, arguments, run_async)?;

        match type_str.as_str() {
            "HostRing64Tensor" => {
                let comp_result: HostRing64Tensor =
                    (outputs.get("output_0").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            "HostRing128Tensor" => {
                let comp_result: HostRing128Tensor =
                    (outputs.get("output_0").unwrap().clone()).try_into()?;
                assert_eq!(expected_result, comp_result.into());
                Ok(())
            }
            _ => Err(anyhow::anyhow!("Failed to parse test case")),
        }
    }

    #[cfg(feature = "async_execute")]
    fn _create_async_session(
        networking: &Arc<dyn Send + Sync + AsyncNetworking>,
        exec_storage: &Arc<dyn Send + Sync + AsyncStorage>,
        role_assignments: HashMap<Role, Identity>,
    ) -> AsyncSession {
        AsyncSession::new(
            SessionId::try_from("foobar").unwrap(),
            hashmap!(),
            role_assignments,
            Arc::clone(networking),
            Arc::clone(exec_storage),
        )
    }

    #[cfg(feature = "async_execute")]
    #[test]
    fn test_duplicate_session_ids() {
        let source = r#"key = Constant{value=HostPrfKey(00000000000000000000000000000000)}: () -> HostPrfKey @Host(alice)
        seed = DeriveSeed {sync_key = [1, 2, 3]}: (HostPrfKey) -> HostSeed (key) @Host(alice)
        output = Output{tag = "output_0"}: (HostSeed) -> HostSeed (seed) @Host(alice)"#;

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
}
