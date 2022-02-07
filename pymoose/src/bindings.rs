use crate::computation::PyComputation;
use moose::compilation::{compile_passes, into_pass, Pass};
use moose::computation::{Computation, Role, Value};
use moose::execution::AsyncTestRuntime;
use moose::execution::Identity;
use moose::host::{FromRaw, HostBitTensor, HostPlacement, HostString, HostTensor};
use moose::textual::{parallel_parse_computation, ToTextual};
use ndarray::IxDyn;
use ndarray::LinalgScalar;
use numpy::{Element, PyArrayDescr, PyArrayDyn, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyBytes, PyFloat, PyString, PyType};
use pyo3::wrap_pymodule;
use pyo3::{exceptions::PyTypeError, prelude::*, AsPyPointer};
use std::collections::HashMap;
use std::convert::TryInto;

fn create_computation_graph_from_py_bytes(computation: Vec<u8>) -> Computation {
    let comp: PyComputation = rmp_serde::from_read_ref(&computation).unwrap();
    let rust_comp: Computation = comp.try_into().unwrap();
    rust_comp.toposort().unwrap()
}

fn pyobj_to_value(py: Python, obj: PyObject) -> PyResult<Value> {
    let obj_ref = obj.as_ref(py);
    if obj_ref.is_instance::<PyString>()? {
        let string_value: String = obj.extract(py)?;
        Ok(Value::HostString(Box::new(HostString(
            string_value,
            HostPlacement::from("fake"),
        ))))
    } else if obj_ref.is_instance::<PyFloat>()? {
        let float_value: f64 = obj.extract(py)?;
        Ok(Value::Float64(Box::new(float_value)))
    } else if obj_ref.is_instance::<PyArrayDyn<f32>>()? {
        // NOTE: this passes for any inner dtype, since python's isinstance will
        // only do a shallow typecheck. inside the pyobj_tensor_to_value we do further
        // introspection on the array & its dtype to map to the correct kind of Value
        let value = pyobj_tensor_to_value(py, &obj).unwrap();
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
    let plc = HostPlacement::from("TODO");
    let pyarray = obj.cast_as::<PyArrayDyn<bool>>(py).unwrap();
    plc.from_raw(
        pyarray
            .to_owned_array()
            .map(|b| *b as u8)
            .into_dimensionality::<IxDyn>()
            .unwrap(),
    )
}

fn pyobj_tensor_to_value(py: Python, obj: &PyObject) -> Result<Value, anyhow::Error> {
    let dtype_obj = obj.getattr(py, "dtype")?;
    let dtype: &PyArrayDescr = dtype_obj.cast_as(py).unwrap();
    let np_dtype = dtype.get_datatype().unwrap();
    match np_dtype {
        numpy::DataType::Float32 => Ok(Value::from(pyobj_tensor_to_host_tensor::<f32>(py, obj))),
        numpy::DataType::Float64 => Ok(Value::from(pyobj_tensor_to_host_tensor::<f64>(py, obj))),
        numpy::DataType::Int8 => Ok(Value::from(pyobj_tensor_to_host_tensor::<i8>(py, obj))),
        numpy::DataType::Int16 => Ok(Value::from(pyobj_tensor_to_host_tensor::<i16>(py, obj))),
        numpy::DataType::Int32 => Ok(Value::from(pyobj_tensor_to_host_tensor::<i32>(py, obj))),
        numpy::DataType::Int64 => Ok(Value::from(pyobj_tensor_to_host_tensor::<i64>(py, obj))),
        numpy::DataType::Uint8 => Ok(Value::from(pyobj_tensor_to_host_tensor::<u8>(py, obj))),
        numpy::DataType::Uint16 => Ok(Value::from(pyobj_tensor_to_host_tensor::<u16>(py, obj))),
        numpy::DataType::Uint32 => Ok(Value::from(pyobj_tensor_to_host_tensor::<u32>(py, obj))),
        numpy::DataType::Uint64 => Ok(Value::from(pyobj_tensor_to_host_tensor::<u64>(py, obj))),
        numpy::DataType::Bool => Ok(Value::from(pyobj_tensor_to_host_bit_tensor(py, obj))),
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
        Value::HostBitTensor(t) => Ok(t.0.map(|v| *v != 0).to_pyarray(py).to_object(py)),
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
        role_assignments: HashMap<String, String>,
        arguments: HashMap<String, PyObject>,
        compiler_passes: Vec<String>,
    ) -> PyResult<Option<HashMap<String, PyObject>>> {
        let computation = create_computation_graph_from_py_bytes(computation);
        let passes: Vec<Pass> =
            into_pass(&compiler_passes[..]).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let computation = compile_passes(&computation, &passes[..])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.evaluate_compiled_computation(py, &computation, role_assignments, arguments)
    }

    fn evaluate_compiled(
        &mut self,
        py: Python,
        computation: PyObject,
        role_assignments: HashMap<String, String>,
        arguments: HashMap<String, PyObject>,
    ) -> PyResult<Option<HashMap<String, PyObject>>> {
        let moose = MooseComputation::from_py(py, computation)?;
        let computation = moose.try_borrow(py)?;
        self.evaluate_compiled_computation(
            py,
            &computation.computation,
            role_assignments,
            arguments,
        )
    }

    fn write_value_to_storage(
        &self,
        py: Python,
        identity: String,
        key: String,
        value: PyObject,
    ) -> PyResult<()> {
        let identity = Identity::from(identity);
        let value_to_store = pyobj_to_value(py, value)?;
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
        role_assignments: HashMap<String, String>,
        arguments: HashMap<String, PyObject>,
    ) -> PyResult<Option<HashMap<String, PyObject>>> {
        let arguments = arguments
            .iter()
            .map(|arg| (arg.0.clone(), pyobj_to_value(py, arg.1.clone()).unwrap()))
            .collect::<HashMap<String, Value>>();

        let valid_role_assignments = role_assignments
            .into_iter()
            .map(|arg| (Role::from(&arg.0), Identity::from(&arg.1)))
            .collect::<HashMap<Role, Identity>>();

        let outputs =
            self.runtime
                .evaluate_computation(computation, valid_role_assignments, arguments);

        let mut outputs_py_val: HashMap<String, PyObject> = HashMap::new();
        match outputs {
            Ok(outputs) => {
                for (output_name, value) in outputs {
                    match value {
                        Value::Unit(_) => None,
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
        let computation =
            Computation::from_bytes(mybytes).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let moose_comp = MooseComputation { computation };
        Py::new(py, moose_comp)
    }

    pub fn to_bytes<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        let comp_bytes = self
            .computation
            .to_bytes()
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
        let passes: Vec<String> = passes.unwrap_or_else(|| {
            vec![
                "typing".into(),
                "full".into(),
                "prune".into(),
                "networking".into(),
                "toposort".into(),
            ]
        });
        let passes = into_pass(&passes).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let computation = compile_passes(&computation, &passes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(MooseComputation { computation })
    }

    Ok(())
}

#[pymodule]
fn moose_runtime(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LocalRuntime>()?;
    m.add_class::<MooseComputation>()?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "rust")]
fn pymoose_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(elk_compiler))?;
    m.add_wrapped(wrap_pymodule!(moose_runtime))?;
    Ok(())
}
