use moose::bit::BitTensor;
use moose::compilation::typing::update_types_one_hop;
use moose::computation::{Computation, Role, SessionId, Value};
use moose::execution::AsyncTestRuntime;
use moose::execution::Identity;
use moose::fixedpoint::Convert;
use moose::prim::RawSeed;
use moose::prng::AesRng;
use moose::python_computation::PyComputation;
use moose::ring::Ring64Tensor;
use moose::standard::{Float64Tensor, RawShape, StandardTensor};
use moose::utils;
use ndarray::IxDyn;
use ndarray::{ArrayD, LinalgScalar};
use numpy::{Element, PyArrayDescr, PyArrayDyn, PyReadonlyArrayDyn, ToPyArray};

use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyFloat, PyString};
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyBytes, types::PyList};
use std::collections::HashMap;
use std::convert::TryInto;
use std::num::Wrapping;
pub mod python_computation;
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
        &self,
        py: Python,
        computation: Vec<u8>,
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

        let computation = create_computation_graph_from_py_bytes(computation);
        let compiled_computation = update_types_one_hop(&computation).unwrap().unwrap();
        compiled_computation.toposort().unwrap();

        let outputs =
            self.runtime
                .evaluate_computation(computation, valid_role_assignments, arguments);

        let mut outputs_py_val: HashMap<String, PyObject> = HashMap::new();
        match outputs {
            Ok(Some(outputs)) => {
                for (output_name, value) in outputs {
                    match value {
                        Value::Unit => None,
                        // TODO: not sure what to support, should eventually standardize output types of computations
                        Value::String(s) => Some(PyString::new(py, &s).to_object(py)),
                        Value::Float64(f) => Some(PyFloat::new(py, f).to_object(py)),
                        // assume it's a tensor
                        _ => outputs_py_val
                            .insert(output_name, tensorval_to_pyobj(py, value).unwrap()),
                    };
                }
            }
            Ok(None) => (),
            Err(e) => return Err(PyRuntimeError::new_err(e.to_string())),
        }

        Ok(Some(outputs_py_val))
    }

    fn write_value_to_storage(
        &self,
        py: Python,
        identity: String,
        key: String,
        value: PyObject,
    ) -> PyResult<()> {
        let rt = Runtime::new().unwrap();
        let _guard = rt.enter();
        let identity = Identity::from(identity);
        let identity_storage = match self.runtime.runtime_storage.get(&identity) {
            Some(store) => store,
            None => {
                return Err(PyRuntimeError::new_err(format!(
                    "Runtime does not contain storage for identity {:?}.",
                    identity.to_string()
                )))
            }
        };
        let value_to_store = pyobj_to_value(py, value)?;
        let result = rt.block_on(async {
            identity_storage
                .save(&key, &SessionId::from("yo"), &value_to_store)
                .await
        });
        if let Err(e) = result {
            return Err(PyRuntimeError::new_err(e.to_string()));
        }
        Ok(())
    }

    fn read_value_from_storage(
        &self,
        py: Python,
        identity: String,
        key: String,
    ) -> PyResult<PyObject> {
        let rt = Runtime::new().unwrap();
        let _guard = rt.enter();
        let val = rt.block_on(async {
            let val = self.runtime.runtime_storage[&Identity::from(identity)]
                .load(&key, &SessionId::from("foobar"), None, "")
                .await
                .unwrap();
            val
        });

        tensorval_to_pyobj(py, val)
    }
}

#[pymodule]
fn moose_runtime(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LocalRuntime>()?;
    Ok(())
}
