#[cfg(test)]
mod tests {
    use approx::AbsDiffEq;
    use moose::compilation::typing::update_types_one_hop;
    use moose::execution::*;
    use moose::storage::{LocalSyncStorage, SyncStorage};
    use moose::{computation::*, host::HostFloat64Tensor, python_computation::PyComputation};
    use ndarray::prelude::*;
    use numpy::ToPyArray;
    use pyo3::prelude::*;
    use rand::Rng;
    use std::collections::HashMap;
    use std::convert::TryFrom;
    use std::convert::TryInto;
    use std::rc::Rc;

    fn create_computation_graph_from_python(py_any: &PyAny) -> Computation {
        let buf: Vec<u8> = py_any.extract().unwrap();
        let comp: PyComputation = rmp_serde::from_read_ref(&buf).unwrap();

        let rust_comp: Computation = comp.try_into().unwrap();
        let rust_comp = update_types_one_hop(&rust_comp).unwrap().unwrap();
        rust_comp.toposort().unwrap()
    }

    fn generate_python_names() -> (String, String) {
        const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
        const STRING_LEN: usize = 30;
        let mut rng = rand::thread_rng();

        let file_name: String = (0..STRING_LEN)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect();
        let module_name = file_name.clone();

        (file_name + ".py", module_name)
    }
    fn run_binary_func(x: &ArrayD<f64>, y: &ArrayD<f64>, py_code: &str) -> Value {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let xc = x.to_pyarray(py);
        let yc = y.to_pyarray(py);

        let (file_name, module_name) = generate_python_names();
        let comp_graph_py = PyModule::from_code(py, py_code, &file_name, &module_name)
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap();

        let py_any = comp_graph_py
            .getattr("f")
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap()
            .call1((xc, yc))
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap();

        let exec = TestExecutor::default();
        let outputs = exec
            .run_computation(
                &create_computation_graph_from_python(py_any),
                SyncArgs::new(),
            )
            .unwrap();
        outputs["result"].clone()
    }
    fn run_unary_func(x: &ArrayD<f64>, py_code: &str) -> Value {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let xc = x.to_pyarray(py);

        let (file_name, module_name) = generate_python_names();
        let comp_graph_py = PyModule::from_code(py, py_code, &file_name, &module_name)
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap();

        let py_any = comp_graph_py
            .getattr("f")
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap()
            .call1((xc,))
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap();

        let exec = TestExecutor::default();
        let outputs = exec
            .run_computation(
                &create_computation_graph_from_python(py_any),
                SyncArgs::new(),
            )
            .unwrap();
        outputs["result"].clone()
    }

    fn graph_from_run_call0_func(py_code: &str) -> Computation {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let (file_name, module_name) = generate_python_names();
        let comp_graph_py = PyModule::from_code(py, py_code, &file_name, &module_name)
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap();

        let py_any = comp_graph_py
            .getattr("f")
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap()
            .call0()
            .map_err(|e| {
                e.print(py);
                e
            })
            .unwrap();

        create_computation_graph_from_python(py_any)
    }

    #[test]
    fn test_deserialize_host_op() {
        let py_code = r#"
import numpy as np
from pymoose.computation import ring as ring_dialect
from pymoose.computation import standard as standard_dialect
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.utils import serialize_computation
from pymoose.computation.standard import TensorType
from pymoose.computation.standard import TensorConstant
from pymoose.computation.standard import UnitType
from pymoose.computation import dtypes
def f(arg1, arg2):
    comp = Computation(operations={}, placements={})
    alice = comp.add_placement(HostPlacement(name="alice"))

    x = np.array(arg1, dtype=np.float64)
    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="alice_input_x",
            value=TensorConstant(value = x),
            placement_name=alice.name,
            inputs={},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    y = np.array(arg2, dtype=np.float64)
    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="alice_input_y",
            value=TensorConstant(value = y),
            placement_name=alice.name,
            inputs={},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )
    comp.add_operation(
        standard_dialect.SPECIAL_OP(
                name="add",
                inputs={"lhs": "alice_input_x", "rhs": "alice_input_y"},
                placement_name=alice.name,
                output_type=TensorType(dtype=dtypes.float64),
        )
    )
    comp.add_operation(
        standard_dialect.OutputOperation(
                name="result",
                inputs={"value": "add"},
                placement_name=alice.name,
                output_type=UnitType(),
        )
    )

    return serialize_computation(comp)
        "#;

        let mul_code = py_code.replace("SPECIAL_OP", "MulOperation");
        let x1 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y1 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let result = run_binary_func(&x1, &y1, &mul_code);

        assert_eq!(
            result,
            Value::HostFloat64Tensor(HostFloat64Tensor::from(x1) * HostFloat64Tensor::from(y1))
        );

        let x2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let add_code = py_code.replace("SPECIAL_OP", "AddOperation");
        let result = run_binary_func(&x2, &y2, &add_code);

        assert_eq!(
            result,
            Value::HostFloat64Tensor(HostFloat64Tensor::from(x2) + HostFloat64Tensor::from(y2))
        );

        let x3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let sub_code = py_code.replace("SPECIAL_OP", "SubOperation");
        let result = run_binary_func(&x3, &y3, &sub_code);

        assert_eq!(
            result,
            Value::HostFloat64Tensor(HostFloat64Tensor::from(x3) - HostFloat64Tensor::from(y3))
        );

        let x4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let dot_code = py_code.replace("SPECIAL_OP", "DotOperation");
        let result = run_binary_func(&x4, &y4, &dot_code);

        assert_eq!(
            result,
            Value::HostFloat64Tensor(HostFloat64Tensor::from(x4).dot(HostFloat64Tensor::from(y4)))
        );
    }

    #[test]
    fn test_deserialize_replicated_op() {
        let py_code = r#"
import numpy as np

from pymoose.deprecated.compiler.compiler import Compiler
from pymoose.computation import dtypes
from pymoose.computation import ring as ring_dialect
from pymoose.computation import standard as standard_dialect
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.replicated import ReplicatedPlacement
from pymoose.computation.standard import TensorType
from pymoose.computation.standard import UnitType
from pymoose.computation.utils import serialize_computation
from pymoose.computation.ring import RingTensorType
from pymoose.computation import dtypes
from pymoose.deprecated.computation import fixedpoint as fixedpoint_ops

alice = HostPlacement(name="alice")
bob = HostPlacement(name="bob")
carole = HostPlacement(name="carole")
rep = ReplicatedPlacement(name="rep", player_names=["alice", "bob", "carole"])


def f(arg1, arg2):
    comp = Computation(operations={}, placements={})
    comp.add_placement(alice)
    comp.add_placement(bob)
    comp.add_placement(carole)
    comp.add_placement(rep)

    x = np.array(arg1, dtype=np.float64)
    y = np.array(arg2, dtype=np.float64)

    fp_dtype = dtypes.fixed(8, 27)


    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="alice_input",
            value=standard_dialect.TensorConstant(value=x),
            placement_name=alice.name,
            inputs={},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    comp.add_operation(
        fixedpoint_ops.EncodeOperation(
            name="encode_alice",
            inputs={"value": "alice_input"},
            placement_name="alice",
            output_type=fixedpoint_ops.EncodedTensorType(
                dtype=fp_dtype, precision=fp_dtype.fractional_precision
            ),
            precision=fp_dtype.fractional_precision,
        )
    )

    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="bob_input",
            value=standard_dialect.TensorConstant(value=y),
            placement_name=bob.name,
            inputs={},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    comp.add_operation(
        fixedpoint_ops.EncodeOperation(
            name="encode_bob",
            inputs={"value": "bob_input"},
            placement_name="bob",
            output_type=fixedpoint_ops.EncodedTensorType(
                dtype=fp_dtype, precision=fp_dtype.fractional_precision
            ),
            precision=fp_dtype.fractional_precision,
        )
    )

    comp.add_operation(
        standard_dialect.SPECIAL_OP(
            name="rep_add",
            placement_name=rep.name,
            inputs={"lhs": "encode_alice", "rhs": "encode_bob"},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    comp.add_operation(
        fixedpoint_ops.DecodeOperation(
            name="decode_carole",
            inputs={"value": "rep_add"},
            placement_name=carole.name,
            output_type=TensorType(dtype=dtypes.float64),
            precision=fp_dtype.fractional_precision,
        )
    )

    comp.add_operation(
        standard_dialect.OutputOperation(
            name="result", placement_name=carole.name, inputs={"value": "decode_carole"},
            output_type=RingTensorType(),
        )
    )

    compiler = Compiler(ring=128)
    comp = compiler.run_passes(comp)

    return serialize_computation(comp)

"#;
        let mul_code = py_code.replace("SPECIAL_OP", "MulOperation");
        let x1 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y1 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let result = run_binary_func(&x1, &y1, &mul_code);

        assert_eq!(
            result,
            Value::HostFloat64Tensor(HostFloat64Tensor::from(x1) * HostFloat64Tensor::from(y1))
        );

        let x2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y2 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let add_code = py_code.replace("SPECIAL_OP", "AddOperation");
        let result = run_binary_func(&x2, &y2, &add_code);

        assert_eq!(
            result,
            Value::HostFloat64Tensor(HostFloat64Tensor::from(x2) + HostFloat64Tensor::from(y2))
        );

        let x3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y3 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let sub_code = py_code.replace("SPECIAL_OP", "SubOperation");
        let result = run_binary_func(&x3, &y3, &sub_code);

        assert_eq!(
            result,
            Value::HostFloat64Tensor(HostFloat64Tensor::from(x3) - HostFloat64Tensor::from(y3))
        );

        let x4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y4 = array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let dot_code = py_code.replace("SPECIAL_OP", "DotOperation");
        let result = run_binary_func(&x4, &y4, &dot_code);

        assert_eq!(
            result,
            Value::HostFloat64Tensor(HostFloat64Tensor::from(x4).dot(HostFloat64Tensor::from(y4)))
        );
    }
    #[test]
    fn test_constant() {
        let py_code = r#"
import numpy as np
from pymoose.computation import ring as ring_dialect
from pymoose.computation import standard as standard_dialect
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.utils import serialize_computation
from pymoose.computation import dtypes

def f():
    comp = Computation(operations={}, placements={})
    alice = comp.add_placement(HostPlacement(name="alice"))

    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="constant_0",
            inputs={},
            placement_name="alice",
            value=standard_dialect.StringConstant(value="w_uri"),
            output_type=standard_dialect.StringType(),
        )
    )

    return serialize_computation(comp)

    "#;

        let exec = TestExecutor::default();
        let _ = exec.run_computation(&graph_from_run_call0_func(py_code), SyncArgs::new());
    }
    #[test]
    fn test_deserialize_linear_regression() {
        let py_code = r#"
import numpy as np
from pymoose import edsl
from pymoose.computation import standard as standard_dialect
from pymoose.computation.utils import serialize_computation


FIXED = edsl.fixed(8, 27)

def mse(y_pred, y_true):
    return edsl.mean(edsl.square(edsl.sub(y_pred, y_true)), axis=0)


def ss_res(y_pred, y_true):
    squared_residuals = edsl.square(edsl.sub(y_true, y_pred))
    return edsl.sum(squared_residuals, axis=0)


def ss_tot(y_true):
    y_mean = edsl.mean(y_true)
    squared_deviations = edsl.square(edsl.sub(y_true, y_mean))
    return edsl.sum(squared_deviations, axis=0)


def r_squared(ss_res, ss_tot):
    residuals_ratio = edsl.div(ss_res, ss_tot)
    return edsl.sub(edsl.constant(np.array([1], dtype=np.float64), dtype=edsl.float64), residuals_ratio)


def f():
    x_owner = edsl.host_placement(name="x-owner")
    model_owner = edsl.host_placement(name="model-owner")
    y_owner = edsl.host_placement(name="y-owner")
    replicated_plc = edsl.replicated_placement(
        players=[x_owner, y_owner, model_owner], name="replicated-plc"
    )


    @edsl.computation
    def my_comp():

        with x_owner:
            X = edsl.atleast_2d(
                edsl.load("x_uri", dtype=edsl.float64),to_column_vector=True
            )
            bias_shape = edsl.slice(edsl.shape(X), begin=0, end=1)
            bias = edsl.ones(bias_shape, dtype=edsl.float64)
            reshaped_bias = edsl.expand_dims(bias, 1)
            X_b = edsl.concatenate([reshaped_bias, X], axis=1)
            A = edsl.inverse(edsl.dot(edsl.transpose(X_b), X_b))
            B = edsl.dot(A, edsl.transpose(X_b))
            X_b = edsl.cast(X_b, dtype=FIXED)
            B = edsl.cast(B, dtype=FIXED)


        with y_owner:
            y_true = edsl.atleast_2d(
                edsl.load("y_uri", dtype=edsl.float64), to_column_vector=True
            )
            totals_ss = ss_tot(y_true)
            y_true = edsl.cast(y_true, dtype=FIXED)


        with replicated_plc:
            w = edsl.dot(B, y_true)
            y_pred = edsl.dot(X_b, w)
            mse_result = mse(y_pred, y_true)
            residuals_ss = ss_res(y_pred, y_true)

        with model_owner:
            residuals_ss = edsl.cast(residuals_ss, dtype=edsl.float64)
            rsquared_result = r_squared(residuals_ss, totals_ss)

        with model_owner:
            w = edsl.cast(w, dtype=edsl.float64)
            mse_result = edsl.cast(mse_result, dtype=edsl.float64)
            res = (
                edsl.save("regression_weights", w),
                edsl.save("mse_result", mse_result),
                edsl.save("rsquared_result", rsquared_result),
            )

        return res

    concrete_comp = edsl.trace_and_compile(my_comp, ring=128)
    return serialize_computation(concrete_comp)

"#;

        let comp = graph_from_run_call0_func(py_code);
        let x = Value::from(HostFloat64Tensor::from(
            array![
                [-0.76943992],
                [0.32067753],
                [-0.61509169],
                [0.11511809],
                [1.49598442],
                [0.37012138],
                [-0.49693762],
                [0.96914636],
                [0.19892362],
                [-0.98655745]
            ]
            .into_dimensionality::<IxDyn>()
            .unwrap(),
        ));

        let y = Value::from(HostFloat64Tensor::from(
            array![
                7.69168025,
                10.9620326,
                8.15472493,
                10.34535427,
                14.48795325,
                11.11036415,
                8.50918715,
                12.90743909,
                10.59677087,
                7.04032766
            ]
            .into_dimensionality::<IxDyn>()
            .unwrap(),
        ));

        let mut storage_inputs: HashMap<String, Value> = HashMap::new();
        storage_inputs.insert("x_uri".to_string(), x);
        storage_inputs.insert("y_uri".to_string(), y);

        let storage: Rc<dyn SyncStorage> = Rc::new(LocalSyncStorage::from_hashmap(storage_inputs));
        let exec = TestExecutor::from_storage(&storage);
        exec.run_computation(&comp, SyncArgs::new()).unwrap();

        let res = array![[9.9999996], [2.999999]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let diff = HostFloat64Tensor::try_from(
            storage
                .load("regression_weights", &SessionId::from("foobar"), None, "")
                .unwrap(),
        )
        .unwrap();

        assert!(diff.0.abs_diff_eq(&res, 0.000001));
    }

    #[test]
    fn test_deserialize_replicated_abs() {
        let py_code = r#"
import numpy as np

from pymoose.deprecated.compiler.compiler import Compiler
from pymoose.computation import dtypes
from pymoose.computation import ring as ring_dialect
from pymoose.computation import standard as standard_dialect
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.replicated import ReplicatedPlacement
from pymoose.computation.standard import TensorType
from pymoose.computation.standard import UnitType
from pymoose.computation.utils import serialize_computation
from pymoose.computation.ring import RingTensorType
from pymoose.computation import dtypes
from pymoose.deprecated.computation import fixedpoint as fixedpoint_ops

alice = HostPlacement(name="alice")
bob = HostPlacement(name="bob")
carole = HostPlacement(name="carole")
rep = ReplicatedPlacement(name="rep", player_names=["alice", "bob", "carole"])


def f(arg1):
    comp = Computation(operations={}, placements={})
    comp.add_placement(alice)
    comp.add_placement(bob)
    comp.add_placement(carole)
    comp.add_placement(rep)

    x = np.array(arg1, dtype=np.float64)

    fp_dtype = dtypes.fixed(8, 27)

    comp.add_operation(
        standard_dialect.ConstantOperation(
            name="alice_input",
            value=standard_dialect.TensorConstant(value=x),
            placement_name=alice.name,
            inputs={},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    comp.add_operation(
        fixedpoint_ops.EncodeOperation(
            name="encode_alice",
            inputs={"value": "alice_input"},
            placement_name="alice",
            output_type=fixedpoint_ops.EncodedTensorType(
                dtype=fp_dtype, precision=fp_dtype.fractional_precision
            ),
            precision=fp_dtype.fractional_precision,
        )
    )

    comp.add_operation(
        standard_dialect.AbsOperation(
            name="rep_abs",
            placement_name=rep.name,
            inputs={"x": "encode_alice"},
            output_type=TensorType(dtype=dtypes.float64),
        )
    )

    comp.add_operation(
        fixedpoint_ops.DecodeOperation(
            name="decode_carole",
            inputs={"value": "rep_abs"},
            placement_name=carole.name,
            output_type=TensorType(dtype=dtypes.float64),
            precision=fp_dtype.fractional_precision,
        )
    )

    comp.add_operation(
        standard_dialect.OutputOperation(
            name="result", placement_name=carole.name, inputs={"value": "decode_carole"},
            output_type=RingTensorType(),
        )
    )

    compiler = Compiler(ring=128)
    comp = compiler.run_passes(comp)

    return serialize_computation(comp)

"#;
        let x1 = array![[-1.0, -2.0], [-3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let result = run_unary_func(&x1, py_code);
        let y1 = x1.mapv(f64::abs);
        assert_eq!(
            result,
            Value::HostFloat64Tensor(HostFloat64Tensor::from(y1))
        );
    }
}
