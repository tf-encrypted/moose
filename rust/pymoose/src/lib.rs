use moose::bit::BitTensor;
use moose::compilation::typing::update_types_one_hop;
use moose::computation::{Computation, Role, SessionId, Value};
use moose::execution::{
    AsyncExecutor, AsyncNetworkingImpl, AsyncReceiver, AsyncSession, AsyncSessionHandle, Identity,
};
use moose::fixedpoint::Convert;
use moose::networking::{AsyncNetworking, LocalAsyncNetworking};
use moose::prim::RawSeed;
use moose::prng::AesRng;
use moose::python_computation::PyComputation;
use moose::ring::Ring64Tensor;
use moose::standard::{Float64Tensor, RawShape, StandardTensor};
use moose::utils;
use ndarray::IxDyn;
use ndarray::{ArrayD, LinalgScalar};
use numpy::{Element, PyArrayDescr, PyArrayDyn, PyReadonlyArrayDyn, ToPyArray};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::{PyFloat, PyString};
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyBytes, types::PyList};
use std::collections::HashMap;
use std::convert::TryInto;
use std::num::Wrapping;
use std::sync::Arc;
pub mod python_computation;
use moose::storage::{AsyncStorage, LocalAsyncStorage};
use tokio::runtime::Runtime;

fn dynarray_to_ring64(arr: &PyReadonlyArrayDyn<u64>) -> Ring64Tensor {
    let arr_wrap = arr.as_array().mapv(Wrapping);
    Ring64Tensor::new(arr_wrap)
}

fn ring64_to_array(r: Ring64Tensor) -> ArrayD<u64> {
    let inner_arr = r.0;
    let shape = inner_arr.shape();
    let unwrapped = inner_arr.mapv(|x| x.0);
    unwrapped.into_shape(shape).unwrap()
}

fn binary_pyfn<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<u64>,
    y: PyReadonlyArrayDyn<u64>,
    binary_op: impl Fn(Ring64Tensor, Ring64Tensor) -> Ring64Tensor,
) -> &'py PyArrayDyn<u64> {
    let x_ring = dynarray_to_ring64(&x);
    let y_ring = dynarray_to_ring64(&y);
    let res = binary_op(x_ring, y_ring);
    let res_array = ring64_to_array(res);
    res_array.to_pyarray(py)
}

fn create_computation_graph_from_py_bytes(computation: Vec<u8>) -> Computation {
    let comp: PyComputation = rmp_serde::from_read_ref(&computation).unwrap();
    let rust_comp: Computation = comp.try_into().unwrap();
    rust_comp.toposort().unwrap()
}

#[pymodule]
fn moose_kernels(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "ring_add")]
    fn ring_add<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArrayDyn<u64> {
        binary_pyfn(py, x, y, |a, b| a + b)
    }

    #[pyfn(m, "ring_mul")]
    fn ring_mul<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArrayDyn<u64> {
        binary_pyfn(py, x, y, |a, b| a * b)
    }

    #[pyfn(m, "ring_dot")]
    fn ring_dot<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArrayDyn<u64> {
        let x_ring = dynarray_to_ring64(&x);
        let y_ring = dynarray_to_ring64(&y);
        let res = x_ring.dot(y_ring);
        let res_array = ring64_to_array(res);
        res_array.to_pyarray(py)
    }

    #[pyfn(m, "ring_sub")]
    fn ring_sub<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArrayDyn<u64> {
        binary_pyfn(py, x, y, |a, b| a - b)
    }

    #[pyfn(m, "ring_sum")]
    fn ring_sum<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        axis: Option<usize>,
    ) -> &'py PyArrayDyn<u64> {
        let x_ring = dynarray_to_ring64(&x);
        let res = x_ring.sum(axis);
        let res_array = ring64_to_array(res);
        res_array.to_pyarray(py)
    }

    #[pyfn(m, "ring_shape")]
    fn ring_shape<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u64>) -> &'py PyList {
        let shape: &[usize] = x.shape();
        PyList::new(py, shape.iter())
    }

    #[pyfn(m, "sample_key")]
    fn sample_key(py: Python) -> &PyBytes {
        let key: [u8; 16] = AesRng::generate_random_key();
        PyBytes::new(py, &key)
    }

    #[pyfn(m, "derive_seed")]
    fn derive_seed<'py>(py: Python<'py>, seed: &'py PyBytes, nonce: &'py PyBytes) -> &'py PyBytes {
        let new_seed = utils::derive_seed(seed.as_bytes(), nonce.as_bytes());
        PyBytes::new(py, &new_seed)
    }

    #[pyfn(m, "ring_fill")]
    fn ring_fill(py: Python<'_>, shape: Vec<usize>, el: u64) -> &'_ PyArrayDyn<u64> {
        let shape = RawShape(shape);
        let res = Ring64Tensor::fill(&shape, el);
        let res_array = ring64_to_array(res);
        res_array.to_pyarray(py)
    }

    #[pyfn(m, "ring_sample")]
    fn ring_sample<'py>(
        py: Python<'py>,
        shape: Vec<usize>,
        seed: &'py PyBytes,
        max_value: Option<u64>,
    ) -> &'py PyArrayDyn<u64> {
        let res = match max_value {
            None => Ring64Tensor::sample_uniform(
                &RawShape(shape),
                &RawSeed(seed.as_bytes().try_into().unwrap()),
            ),
            Some(max_value) => {
                if max_value == 1 {
                    Ring64Tensor::sample_bits(
                        &RawShape(shape),
                        &RawSeed(seed.as_bytes().try_into().unwrap()),
                    )
                } else {
                    unimplemented!()
                }
            }
        };
        let res_array = ring64_to_array(res);
        res_array.to_pyarray(py)
    }

    #[pyfn(m, "ring_shl")]
    fn ring_shl<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        amount: u64,
    ) -> &'py PyArrayDyn<u64> {
        let x_ring = dynarray_to_ring64(&x);
        let res = x_ring << (amount as usize);
        let res_array = ring64_to_array(res);
        res_array.to_pyarray(py)
    }

    #[pyfn(m, "ring_shr")]
    fn ring_shr<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        amount: u64,
    ) -> &'py PyArrayDyn<u64> {
        let x_ring = dynarray_to_ring64(&x);
        let res = x_ring >> (amount as usize);
        let res_array = ring64_to_array(res);
        res_array.to_pyarray(py)
    }

    #[pyfn(m, "bit_xor")]
    fn bit_xor<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u8>,
        y: PyReadonlyArrayDyn<u8>,
    ) -> &'py PyArrayDyn<u8> {
        let b1 = BitTensor::from(x.to_owned_array());
        let b2 = BitTensor::from(y.to_owned_array());
        ArrayD::<u8>::from(b1 ^ b2).to_pyarray(py)
    }

    #[pyfn(m, "bit_and")]
    fn bit_and<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u8>,
        y: PyReadonlyArrayDyn<u8>,
    ) -> &'py PyArrayDyn<u8> {
        let b1 = BitTensor::from(x.to_owned_array());
        let b2 = BitTensor::from(y.to_owned_array());
        ArrayD::<u8>::from(b1 & b2).to_pyarray(py)
    }

    #[pyfn(m, "bit_sample")]
    fn bit_sample<'py>(
        py: Python<'py>,
        shape: Vec<usize>,
        seed: &'py PyBytes,
    ) -> &'py PyArrayDyn<u8> {
        let shape = RawShape(shape);
        let seed = RawSeed(seed.as_bytes().try_into().unwrap());
        let b = BitTensor::sample_uniform(&shape, &seed);
        ArrayD::<u8>::from(b).to_pyarray(py)
    }

    #[pyfn(m, "bit_fill")]
    fn bit_fill(py: Python<'_>, shape: Vec<usize>, el: u8) -> &'_ PyArrayDyn<u8> {
        let shape = RawShape(shape);
        let res = BitTensor::fill(&shape, el);
        ArrayD::<u8>::from(res).to_pyarray(py)
    }

    #[pyfn(m, "bit_extract")]
    fn bit_extract<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        bit_idx: usize,
    ) -> &'py PyArrayDyn<u8> {
        let x_ring = dynarray_to_ring64(&x);
        let res = x_ring.bit_extract(bit_idx);
        ArrayD::<u8>::from(res).to_pyarray(py)
    }

    #[pyfn(m, "ring_inject")]
    fn ring_inject<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u8>,
        bit_idx: usize,
    ) -> &'py PyArrayDyn<u64> {
        let b = BitTensor::from(x.to_owned_array());
        let res = Ring64Tensor::from(b) << bit_idx;
        ring64_to_array(res).to_pyarray(py)
    }

    #[pyfn(m, "bit_shape")]
    fn bit_shape<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>) -> &'py PyList {
        let shape: &[usize] = x.shape();
        PyList::new(py, shape.iter())
    }

    #[pyfn(m, "fixedpoint_encode")]
    fn fixedpoint_encode<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        scaling_factor: u64,
    ) -> &'py PyArrayDyn<u64> {
        let x = Float64Tensor::from(x.to_owned_array());
        let y = Ring64Tensor::encode(&x, scaling_factor);
        ring64_to_array(y).to_pyarray(py)
    }

    #[pyfn(m, "fixedpoint_decode")]
    fn fixedpoint_decode<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        scaling_factor: u64,
    ) -> &'py PyArrayDyn<f64> {
        let x_ring = dynarray_to_ring64(&x);
        let y = Ring64Tensor::decode(&x_ring, scaling_factor);
        y.0.to_pyarray(py)
    }

    #[pyfn(m, "fixedpoint_ring_mean")]
    fn fixedpoint_ring_mean<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        axis: Option<usize>,
        precision: u32,
    ) -> &'py PyArrayDyn<u64> {
        let x_ring = dynarray_to_ring64(&x);
        let y = Ring64Tensor::ring_mean(x_ring, axis, 2u64.pow(precision));
        ring64_to_array(y).to_pyarray(py)
    }

    Ok(())
}

fn pyobj_to_value(py: Python, obj: PyObject) -> PyResult<Value> {
    let obj_ref = obj.as_ref(py);
    if obj_ref.is_instance::<PyString>()? {
        let string_value: String = obj.extract(py)?;
        Ok(Value::String(string_value))
    } else if obj_ref.is_instance::<PyFloat>()? {
        let float_value: f64 = obj.extract(py)?;
        Ok(Value::Float64(float_value))
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

fn pyobj_tensor_to_std_tensor<T>(py: Python, obj: &PyObject) -> StandardTensor<T>
where
    T: Element + LinalgScalar,
{
    let pyarray = obj.cast_as::<PyArrayDyn<T>>(py).unwrap();
    StandardTensor::from(
        pyarray
            .to_owned_array()
            .into_dimensionality::<IxDyn>()
            .unwrap(),
    )
}

fn pyobj_tensor_to_value(py: Python, obj: &PyObject) -> Result<Value, anyhow::Error> {
    let dtype_obj = obj.getattr(py, "dtype")?;
    let dtype: &PyArrayDescr = dtype_obj.cast_as(py).unwrap();
    let np_dtype = dtype.get_datatype().unwrap();
    match np_dtype {
        numpy::DataType::Float32 => Ok(Value::from(pyobj_tensor_to_std_tensor::<f32>(py, obj))),
        numpy::DataType::Float64 => Ok(Value::from(pyobj_tensor_to_std_tensor::<f64>(py, obj))),
        numpy::DataType::Int8 => Ok(Value::from(pyobj_tensor_to_std_tensor::<i8>(py, obj))),
        numpy::DataType::Int16 => Ok(Value::from(pyobj_tensor_to_std_tensor::<i16>(py, obj))),
        numpy::DataType::Int32 => Ok(Value::from(pyobj_tensor_to_std_tensor::<i32>(py, obj))),
        numpy::DataType::Int64 => Ok(Value::from(pyobj_tensor_to_std_tensor::<i64>(py, obj))),
        numpy::DataType::Uint8 => Ok(Value::from(pyobj_tensor_to_std_tensor::<u8>(py, obj))),
        numpy::DataType::Uint16 => Ok(Value::from(pyobj_tensor_to_std_tensor::<u16>(py, obj))),
        numpy::DataType::Uint32 => Ok(Value::from(pyobj_tensor_to_std_tensor::<u32>(py, obj))),
        numpy::DataType::Uint64 => Ok(Value::from(pyobj_tensor_to_std_tensor::<u64>(py, obj))),
        otherwise => Err(anyhow::Error::msg(format!(
            "Unsupported numpy datatype {:?}",
            otherwise
        ))),
    }
}

fn tensorval_to_pyobj(py: Python, tensor: Value) -> PyResult<PyObject> {
    match tensor {
        Value::Float32Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::Float64Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::Int8Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::Int16Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::Int32Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::Int64Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::Uint8Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::Uint16Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::Uint32Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        Value::Uint64Tensor(t) => Ok(t.0.to_pyarray(py).to_object(py)),
        otherwise => Err(PyTypeError::new_err(format!(
            r#"Values of type {:?} cannot be handled by runtime storage: must be a tensor of supported dtype."#,
            otherwise
        ))),
    }
}

#[pyclass(subclass)]
pub struct LocalRuntime {
    #[pyo3(get, set)]
    identities: Vec<String>,
    executors: HashMap<Identity, AsyncExecutor>,
    runtime_storage: HashMap<Identity, Arc<dyn Send + Sync + AsyncStorage>>,
    networking: AsyncNetworkingImpl,
}

#[pymethods]
impl LocalRuntime {
    #[new]
    fn new(py: Python, storage_mapping: HashMap<String, HashMap<String, PyObject>>) -> Self {
        let mut executors: HashMap<Identity, AsyncExecutor> = HashMap::new();
        let networking: Arc<dyn Send + Sync + AsyncNetworking> =
            Arc::new(LocalAsyncNetworking::default());
        let mut runtime_storage: HashMap<Identity, Arc<dyn Send + Sync + AsyncStorage>> =
            HashMap::new();
        let mut identities = Vec::new();
        for (identity_str, storage) in storage_mapping {
            let identity = Identity::from(identity_str.clone());
            identities.push(identity_str);
            // TODO handle Result in map predicate instead of `unwrap`
            let storage = storage
                .iter()
                .map(|arg| (arg.0.to_owned(), pyobj_tensor_to_value(py, arg.1).unwrap()))
                .collect::<HashMap<String, Value>>();

            let exec_storage: Arc<dyn Send + Sync + AsyncStorage> =
                Arc::new(LocalAsyncStorage::from_hashmap(storage));
            runtime_storage.insert(identity.clone(), exec_storage);

            let executor = AsyncExecutor::default();
            executors.insert(identity, executor);
        }
        LocalRuntime {
            identities,
            executors,
            runtime_storage,
            networking,
        }
    }

    fn evaluate_computation(
        &self,
        py: Python,
        computation: Vec<u8>,
        role_assignments: HashMap<String, String>,
        arguments: HashMap<String, PyObject>,
    ) -> PyResult<Option<HashMap<String, PyObject>>> {
        let computation = create_computation_graph_from_py_bytes(computation);
        let compiled_computation = update_types_one_hop(&computation).unwrap().unwrap();
        compiled_computation.toposort().unwrap();

        self.evaluate_compiled_computation(py, &compiled_computation, role_assignments, arguments)
    }

    fn evaluate_compiled(
        &self,
        py: Python,
        computation: Py<MooseComputation>,
        role_assignments: HashMap<String, String>,
        arguments: HashMap<String, PyObject>,
    ) -> PyResult<Option<HashMap<String, PyObject>>> {
        let cell: &PyCell<MooseComputation> = computation.as_ref(py);
        let computation = cell.try_borrow()?;
        self.evaluate_compiled_computation(py, &computation.computation, role_assignments, arguments)
    }

    fn get_value_from_storage(
        &self,
        py: Python,
        identity: String,
        key: String,
    ) -> PyResult<PyObject> {
        let rt = Runtime::new().unwrap();
        let _guard = rt.enter();
        let val = rt.block_on(async {
            let val = self.runtime_storage[&Identity::from(identity)]
                .load(&key, &SessionId::from("foobar"), None, "")
                .await
                .unwrap();
            val
        });

        // Return value as PyObject
        tensorval_to_pyobj(py, val)
    }
}

impl LocalRuntime {
    fn evaluate_compiled_computation(
        &self,
        py: Python,
        computation: &Computation,
        role_assignments: HashMap<String, String>,
        arguments: HashMap<String, PyObject>,
    ) -> PyResult<Option<HashMap<String, PyObject>>> {

        let arguments = arguments
            .iter()
            .map(|arg| (arg.0.clone(), pyobj_to_value(py, arg.1.clone()).unwrap()))
            .collect::<HashMap<String, Value>>();

        let (valid_role_assignments, missing_role_assignments): (
            HashMap<String, String>,
            HashMap<String, String>,
        ) = role_assignments
            .into_iter()
            .partition(|kv| self.identities.contains(&kv.1));
        if !missing_role_assignments.is_empty() {
            let missing_roles: Vec<&String> = missing_role_assignments.keys().collect();
            let missing_identities: Vec<&String> = missing_role_assignments.values().collect();
            return Err(PyValueError::new_err(format!("Role assignment included identities unknown to Moose runtime: missing identities {:?} for roles {:?}.", missing_identities, missing_roles)));
        }

        let valid_role_assignments = valid_role_assignments
            .into_iter()
            .map(|arg| (Role::from(&arg.0), Identity::from(&arg.1)))
            .collect::<HashMap<Role, Identity>>();

        let mut session_handles: Vec<AsyncSessionHandle> = Vec::new();
        let mut output_futures: HashMap<String, AsyncReceiver> = HashMap::new();
        let rt = Runtime::new().unwrap();
        let _guard = rt.enter();

        for (own_identity, executor) in self.executors.iter() {
            let moose_session = AsyncSession {
                sid: SessionId::from("foobar"),
                arguments: arguments.clone(),
                networking: Arc::clone(&self.networking),
                storage: Arc::clone(&self.runtime_storage[own_identity]),
            };
            let (moose_session_handle, outputs) = executor
                .run_computation(
                    &computation,
                    &valid_role_assignments,
                    &own_identity,
                    moose_session,
                )
                .unwrap();

            for (output_name, output_future) in outputs {
                output_futures.insert(output_name, output_future);
            }

            session_handles.push(moose_session_handle)
        }

        for handle in session_handles {
            let result = rt.block_on(handle.join_on_first_error());
            if let Err(e) = result {
                return Err(PyRuntimeError::new_err(e.to_string()));
            }
        }

        let outputs = rt.block_on(async {
            let mut outputs: HashMap<String, PyObject> = HashMap::new();
            for (output_name, output_future) in output_futures {
                let value = output_future.await.unwrap();
                match value {
                    Value::Unit => None,
                    // TODO: not sure what to support, should eventually standardize output types of computations
                    Value::String(s) => Some(PyString::new(py, &s).to_object(py)),
                    Value::Float64(f) => Some(PyFloat::new(py, f).to_object(py)),
                    // assume it's a tensor
                    _ => outputs.insert(output_name, tensorval_to_pyobj(py, value).unwrap()),
                };
            }
            outputs
        });

        Ok(Some(outputs))
    }
}

#[pyclass]
pub struct MooseComputation {
    computation: Computation,
}

#[pymodule]
fn moose_compiler(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "compile_computation")]
    pub fn compile_computation(py: Python, computation: Vec<u8>) -> Py<MooseComputation> {
        let computation = create_computation_graph_from_py_bytes(computation);
        let computation = update_types_one_hop(&computation).unwrap().unwrap();
        computation.toposort().unwrap();
        Py::new(py, MooseComputation {computation}).unwrap()
    }

    Ok(())
}

#[pymodule]
fn moose_runtime(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LocalRuntime>()?;
    m.add_class::<MooseComputation>()?;
    Ok(())
}
