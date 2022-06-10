use crate::computation::PyComputation;
use moose::compilation::compile;
use moose::compilation::toposort;
use moose::execution::grpc::GrpcMooseRuntime;
use moose::execution::AsyncTestRuntime;
use moose::host::HostTensor;
use moose::prelude::*;
use moose::textual::{parallel_parse_computation, ToTextual};
use moose::tokio;
use ndarray::LinalgScalar;
use numpy::{Element, PyArrayDescr, PyArrayDyn, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyBytes, PyFloat, PyString, PyType};
use pyo3::wrap_pymodule;
use pyo3::{exceptions::PyTypeError, prelude::*, AsPyPointer};
use std::collections::HashMap;
use std::convert::TryInto;

type PyGrpcOutputs = (HashMap<String, PyObject>, Option<HashMap<String, PyObject>>);

fn create_computation_graph_from_py_bytes(computation: Vec<u8>) -> Computation {
    let comp: PyComputation = rmp_serde::from_read_ref(&computation).unwrap();
    let rust_comp: Computation = comp.try_into().unwrap();
    // TODO(Morten) we should not call toposort here
    toposort::toposort(rust_comp).unwrap()
}

fn pyobj_to_value(py: Python, obj: &PyObject) -> PyResult<Value> {
    let obj_ref = obj.as_ref(py);
    if obj_ref.is_instance_of::<PyString>()? {
        let string_value: String = obj.extract(py)?;
        Ok(Value::HostString(Box::new(HostString(
            string_value,
            HostPlacement::from("fake"),
        ))))
    } else if obj_ref.is_instance_of::<PyFloat>()? {
        let float_value: f64 = obj.extract(py)?;
        Ok(Value::Float64(Box::new(float_value)))
    } else if obj_ref.is_instance_of::<PyArrayDyn<f32>>()? {
        // NOTE: this passes for any inner dtype, since python's isinstance will
        // only do a shallow typecheck. inside the pyobj_tensor_to_value we do further
        // introspection on the array & its dtype to map to the correct kind of Value
        let value = pyobj_tensor_to_value(py, obj).unwrap();
        Ok(value)
    } else {
        Err(PyTypeError::new_err(
            r#"Unsupported type found in `evaluate_computation` arguments."#,
        ))
    }
}

fn pyobj_tensor_to_host_tensor<T>(py: Python, obj: &PyObject) -> HostTensor<T>
where
    T: Element + LinalgScalar,
{
    let plc = HostPlacement::from("TODO");
    let pyarray = obj.cast_as::<PyArrayDyn<T>>(py).unwrap();
    plc.from_raw(pyarray.to_owned_array())
}

fn pyobj_tensor_to_host_bit_tensor(py: Python, obj: &PyObject) -> HostBitTensor {
    use moose::host::BitArrayRepr;
    let plc = HostPlacement::from("TODO");
    let pyarray = obj.cast_as::<PyArrayDyn<bool>>(py).unwrap();
    let data = pyarray.to_owned_array().iter().collect();
    HostBitTensor(BitArrayRepr::from_raw(data, pyarray.dims()), plc)
}

fn pyobj_tensor_to_value(py: Python, obj: &PyObject) -> Result<Value, anyhow::Error> {
    let dtype_obj = obj.getattr(py, "dtype")?;
    let dtype: &PyArrayDescr = dtype_obj.cast_as(py).unwrap();
    match dtype {
        dt if dt.is_equiv_to(numpy::dtype::<f32>(py)) => {
            Ok(Value::from(pyobj_tensor_to_host_tensor::<f32>(py, obj)))
        }
        dt if dt.is_equiv_to(numpy::dtype::<f64>(py)) => {
            Ok(Value::from(pyobj_tensor_to_host_tensor::<f64>(py, obj)))
        }
        dt if dt.is_equiv_to(numpy::dtype::<i8>(py)) => {
            Ok(Value::from(pyobj_tensor_to_host_tensor::<i8>(py, obj)))
        }
        dt if dt.is_equiv_to(numpy::dtype::<i16>(py)) => {
            Ok(Value::from(pyobj_tensor_to_host_tensor::<i16>(py, obj)))
        }
        dt if dt.is_equiv_to(numpy::dtype::<i32>(py)) => {
            Ok(Value::from(pyobj_tensor_to_host_tensor::<i32>(py, obj)))
        }
        dt if dt.is_equiv_to(numpy::dtype::<i64>(py)) => {
            Ok(Value::from(pyobj_tensor_to_host_tensor::<i64>(py, obj)))
        }
        dt if dt.is_equiv_to(numpy::dtype::<u8>(py)) => {
            Ok(Value::from(pyobj_tensor_to_host_tensor::<u8>(py, obj)))
        }
        dt if dt.is_equiv_to(numpy::dtype::<u16>(py)) => {
            Ok(Value::from(pyobj_tensor_to_host_tensor::<u16>(py, obj)))
        }
        dt if dt.is_equiv_to(numpy::dtype::<u32>(py)) => {
            Ok(Value::from(pyobj_tensor_to_host_tensor::<u32>(py, obj)))
        }
        dt if dt.is_equiv_to(numpy::dtype::<u64>(py)) => {
            Ok(Value::from(pyobj_tensor_to_host_tensor::<u64>(py, obj)))
        }
        dt if dt.is_equiv_to(numpy::dtype::<bool>(py)) => {
            Ok(Value::from(pyobj_tensor_to_host_bit_tensor(py, obj)))
        }
        otherwise => Err(anyhow::Error::msg(format!(
            "Unsupported numpy datatype {:?}",
            otherwise
        ))),
    }
}

fn tensorval_to_pyobj(py: Python, tensor: Value) -> PyResult<PyObject> {
    match tensor {
        Value::HostFloat32Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::HostFloat64Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::HostInt8Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::HostInt16Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::HostInt32Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::HostInt64Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::HostUint8Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::HostUint16Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::HostUint32Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::HostUint64Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::HostRing64Tensor(t) => Ok(t.0.map(|v| v.0).to_pyarray(py).to_object(py)),
        Value::HostBitTensor(t) => {
            t.0.into_array::<u8>()
                .map(|arr| arr.map(|x| *x != 0))
                .map(|arr| arr.to_pyarray(py).to_object(py))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }
        otherwise => Err(PyTypeError::new_err(format!(
            r#"Values of type {:?} cannot be handled by runtime storage: must be a tensor of supported dtype."#,
            otherwise
        ))),
    }
}

#[pyclass(subclass)]
pub struct LocalRuntime {
    runtime: AsyncTestRuntime,
}

#[pymethods]
impl LocalRuntime {
    #[new]
    fn new(py: Python, storage_mapping: HashMap<String, HashMap<String, PyObject>>) -> Self {
        let mut moose_storage_mapping: HashMap<String, HashMap<String, Value>> = HashMap::new();
        for (identity_str, storage) in storage_mapping {
            // TODO handle Result in map predicate instead of `unwrap`
            let storage = storage
                .iter()
                .map(|arg| (arg.0.to_owned(), pyobj_tensor_to_value(py, arg.1).unwrap()))
                .collect::<HashMap<String, Value>>();

            moose_storage_mapping.insert(identity_str, storage);
        }

        let runtime = AsyncTestRuntime::new(moose_storage_mapping);

        LocalRuntime { runtime }
    }

    fn evaluate_computation(
        &mut self,
        py: Python,
        computation: Vec<u8>,
        arguments: HashMap<String, PyObject>,
        compiler_passes: Option<Vec<String>>,
    ) -> PyResult<Option<HashMap<String, PyObject>>> {
        let computation = create_computation_graph_from_py_bytes(computation);
        let computation = compile(computation, compiler_passes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.evaluate_compiled_computation(py, &computation, arguments)
    }

    fn evaluate_compiled(
        &mut self,
        py: Python,
        computation: PyObject,
        arguments: HashMap<String, PyObject>,
    ) -> PyResult<Option<HashMap<String, PyObject>>> {
        let moose = MooseComputation::from_py(py, computation)?;
        let computation = moose.try_borrow(py)?;
        self.evaluate_compiled_computation(py, &computation.computation, arguments)
    }

    fn write_value_to_storage(
        &self,
        py: Python,
        identity: String,
        key: String,
        value: PyObject,
    ) -> PyResult<()> {
        let identity = Identity::from(identity);
        let value_to_store = pyobj_to_value(py, &value)?;
        let _result = self
            .runtime
            .write_value_to_storage(identity, key, value_to_store)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }

    fn read_value_from_storage(
        &self,
        py: Python,
        identity: String,
        key: String,
    ) -> PyResult<PyObject> {
        let val = self
            .runtime
            .read_value_from_storage(Identity::from(identity), key)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        tensorval_to_pyobj(py, val)
    }
}

impl LocalRuntime {
    fn evaluate_compiled_computation(
        &mut self,
        py: Python,
        computation: &Computation,
        arguments: HashMap<String, PyObject>,
    ) -> PyResult<Option<HashMap<String, PyObject>>> {
        let arguments = arguments
            .iter()
            .map(|(name, value)| (name.clone(), pyobj_to_value(py, value).unwrap()))
            .collect::<HashMap<String, Value>>();

        let outputs = self.runtime.evaluate_computation(computation, arguments);

        let mut outputs_py_val: HashMap<String, PyObject> = HashMap::new();
        match outputs {
            Ok(outputs) => {
                for (output_name, value) in outputs {
                    match value {
                        Value::HostUnit(_) => None,
                        // TODO: not sure what to support, should eventually standardize output types of computations
                        Value::HostString(s) => Some(PyString::new(py, &s.0).to_object(py)),
                        Value::Float64(f) => Some(PyFloat::new(py, *f).to_object(py)),
                        // assume it's a tensor
                        _ => outputs_py_val
                            .insert(output_name, tensorval_to_pyobj(py, value).unwrap()),
                    };
                }
            }
            Err(e) => return Err(PyRuntimeError::new_err(e.to_string())),
        }

        Ok(Some(outputs_py_val))
    }
}

#[pyclass(subclass)]
pub struct GrpcRuntime {
    tokio_runtime: tokio::runtime::Runtime,
    grpc_runtime: GrpcMooseRuntime,
}

#[pymethods]
impl GrpcRuntime {
    #[new]
    fn new(role_assignment: HashMap<String, String>) -> Self {
        let typed_role_assignment = role_assignment
            .into_iter()
            .map(|(role, identity)| (Role::from(role), Identity::from(identity)))
            .collect::<HashMap<Role, Identity>>();

        let tokio_runtime = tokio::runtime::Runtime::new().expect("failed to create Tokio runtime");

        let grpc_runtime = {
            let _guard = tokio_runtime.enter();
            GrpcMooseRuntime::new(typed_role_assignment, None).unwrap()
        };

        GrpcRuntime {
            grpc_runtime,
            tokio_runtime,
        }
    }

    fn evaluate_computation(
        &mut self,
        py: Python,
        computation: Vec<u8>,
        arguments: HashMap<String, PyObject>,
    ) -> PyResult<PyGrpcOutputs> {
        let logical_computation = create_computation_graph_from_py_bytes(computation);

        let physical_computation = compile::<moose::compilation::Pass>(logical_computation, None)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let typed_arguments = arguments
            .iter()
            .map(|(name, value)| (name.clone(), pyobj_to_value(py, value).unwrap()))
            .collect::<HashMap<String, Value>>();

        let session_id = SessionId::random();

        let output_metrics = self
            .tokio_runtime
            .block_on(self.grpc_runtime.run_computation(
                &session_id,
                &physical_computation,
                typed_arguments,
            ))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut outputs_py_val: HashMap<String, PyObject> = HashMap::new();
        for (output_name, value) in output_metrics.outputs {
            match value {
                Value::HostUnit(_) => None,
                // TODO: not sure what to support, should eventually standardize output types of computations
                Value::HostString(s) => Some(PyString::new(py, &s.0).to_object(py)),
                Value::Float64(f) => Some(PyFloat::new(py, *f).to_object(py)),
                // assume it's a tensor
                _ => outputs_py_val.insert(output_name, tensorval_to_pyobj(py, value).unwrap()),
            };
        }

        if let Some(timings) = output_metrics.elapsed_time {
            let mut timings_py_val: HashMap<String, PyObject> = HashMap::new();
            for (role, duration) in timings.iter() {
                timings_py_val.insert(role.0.clone(), duration.as_micros().into_py(py));
            }
            Ok((outputs_py_val, Some(timings_py_val)))
        } else {
            Ok((outputs_py_val, None))
        }
    }
}

#[pyclass]
pub struct MooseComputation {
    computation: Computation,
}

impl MooseComputation {
    /// Convert an object after checking its type.
    ///
    /// The function uses an unsafe block inside exactly the way it is used in the PyO3 library.
    /// The conversion traits already present inside library do not work due to the erroneous constraint
    /// of PyNativeType on them.
    pub fn from_py(py: Python, computation: PyObject) -> PyResult<Py<Self>> {
        assert!(format!("{}", computation.as_ref(py).str()?)
            .starts_with("<builtins.MooseComputation object at "));
        let moose = unsafe { Py::from_borrowed_ptr(py, computation.as_ptr()) };
        Ok(moose)
    }
}

const DEFAULT_PARSE_CHUNKS: usize = 12;

#[pymethods]
impl MooseComputation {
    #[classmethod]
    pub fn from_bytes(_cls: &PyType, py: Python, bytes: &PyBytes) -> PyResult<Py<Self>> {
        let mybytes: Vec<u8> = bytes.extract()?;
        let computation = Computation::from_msgpack(mybytes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let moose_comp = MooseComputation { computation };
        Py::new(py, moose_comp)
    }

    pub fn to_bytes<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        let comp_bytes = self
            .computation
            .to_msgpack()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &comp_bytes))
    }

    #[classmethod]
    pub fn from_disk(_cls: &PyType, py: Python, path: &PyString) -> PyResult<Py<Self>> {
        let mypath: &str = path.extract()?;
        let computation =
            Computation::from_disk(mypath).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let moose_comp = MooseComputation { computation };
        Py::new(py, moose_comp)
    }

    pub fn to_disk(&mut self, path: &PyString) -> PyResult<()> {
        let mypath: &str = path.extract()?;
        self.computation
            .to_disk(mypath)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[classmethod]
    pub fn from_textual(_cls: &PyType, py: Python, text: &PyString) -> PyResult<Py<Self>> {
        let text: &str = text.extract()?;
        let computation: Computation = parallel_parse_computation(text, DEFAULT_PARSE_CHUNKS)
            .map_err(|e: anyhow::Error| PyRuntimeError::new_err(e.to_string()))?;
        let moose_comp = MooseComputation { computation };
        Py::new(py, moose_comp)
    }

    pub fn to_textual(&mut self, py: Python) -> PyResult<PyObject> {
        let comp_text = self.computation.to_textual();
        Ok(comp_text.into_py(py))
    }
}

#[pymodule]
fn elk_compiler(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, computation, passes = "None")]
    #[pyo3(name = "compile_computation")]
    pub fn compile_computation(
        _py: Python,
        computation: Vec<u8>,
        passes: Option<Vec<String>>,
    ) -> PyResult<MooseComputation> {
        let computation = create_computation_graph_from_py_bytes(computation);
        let computation =
            compile(computation, passes).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(MooseComputation { computation })
    }

    Ok(())
}

#[pymodule]
fn moose_runtime(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LocalRuntime>()?;
    m.add_class::<GrpcRuntime>()?;
    m.add_class::<MooseComputation>()?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "_rust")]
fn pymoose_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(elk_compiler))?;
    m.add_wrapped(wrap_pymodule!(moose_runtime))?;
    Ok(())
}

#[cfg(test)]
mod compatibility_tests {
    use moose::compilation::{compile, Pass};
    use moose::textual::parallel_parse_computation;
    use rstest::rstest;

    #[rstest]
    #[case("compatibility/aes-lingreg-logical-0.1.2.moose")]
    #[case("compatibility/aes-lingreg-logical-0.1.3.moose")]
    #[case("compatibility/mean-logical-0.1.4.moose")]
    #[case("compatibility/mean-logical-0.1.5.moose")]
    fn test_old_versions_parsing(#[case] path: String) -> Result<(), anyhow::Error> {
        let source = std::fs::read_to_string(path)?;
        let computation =
            parallel_parse_computation(&source, crate::bindings::DEFAULT_PARSE_CHUNKS)?;
        let _ = compile::<Pass>(computation, None)?;
        Ok(())
    }

    #[rstest]
    #[case("compatibility/aes-lingreg-physical-0.1.2.moose.gz")]
    #[case("compatibility/aes-lingreg-physical-0.1.5.moose.gz")]
    fn test_old_versions_parsing_gzip(#[case] path: String) -> Result<(), anyhow::Error> {
        use flate2::read::GzDecoder;
        use std::io::Read;
        let mut decoder = GzDecoder::new(std::fs::File::open(path)?);
        let mut source = String::new();
        decoder.read_to_string(&mut source)?;
        let _ = parallel_parse_computation(&source, crate::bindings::DEFAULT_PARSE_CHUNKS)?;
        Ok(())
    }
}
