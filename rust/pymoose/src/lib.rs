use moose::bit::BitTensor;
use moose::computation::SessionId;
use moose::computation::Value;
use moose::computation::{Computation, Role};
use moose::execution::{AsyncExecutor, Identity};
use moose::execution::{AsyncSession, TestExecutor};
use moose::fixedpoint::Convert;
use moose::networking::LocalAsyncNetworking;
use moose::prim::Seed;
use moose::prng::AesRng;
use moose::python_computation::PyComputation;
use moose::ring::Ring64Tensor;
use moose::standard::{Float64Tensor, Shape};
use moose::utils;
use ndarray::IxDyn;
use ndarray::{array, ArrayD};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, ToPyArray};
use pyo3::types::IntoPyDict;
use pyo3::{prelude::*, types::PyBytes, types::PyDict, types::PyList};
use std::ascii::AsciiExt;
use std::collections::HashMap;
use std::convert::TryInto;
use std::num::Wrapping;
use std::sync::Arc;
pub mod python_computation;
use moose::storage::{LocalAsyncStorage, LocalSyncStorage};
use std::convert::TryFrom;

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

fn float64_to_array(r: Float64Tensor) -> ArrayD<f64> {
    let inner_arr = r.0;
    let shape = inner_arr.shape();
    let unwrapped = inner_arr.mapv(|x| x);
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

fn dynarray_to_value(arr: &PyReadonlyArrayDyn<f64>) -> Value {
    let arr_wrap = arr
        .as_array()
        .to_owned()
        .into_dimensionality::<IxDyn>()
        .unwrap();
    let v = Value::from(Float64Tensor::from(arr_wrap));
    v
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
        let new_seed = utils::derive_seed(seed.as_bytes(), &nonce.as_bytes());
        PyBytes::new(py, &new_seed)
    }

    #[pyfn(m, "ring_fill")]
    fn ring_fill(py: Python<'_>, shape: Vec<usize>, el: u64) -> &'_ PyArrayDyn<u64> {
        let shape = Shape(shape);
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
                &Shape(shape),
                &Seed(seed.as_bytes().try_into().unwrap()),
            ),
            Some(max_value) => {
                if max_value == 1 {
                    Ring64Tensor::sample_bits(
                        &Shape(shape),
                        &Seed(seed.as_bytes().try_into().unwrap()),
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
        let shape = Shape(shape);
        let seed = Seed(seed.as_bytes().try_into().unwrap());
        let b = BitTensor::sample_uniform(&shape, &seed);
        ArrayD::<u8>::from(b).to_pyarray(py)
    }

    #[pyfn(m, "bit_fill")]
    fn bit_fill(py: Python<'_>, shape: Vec<usize>, el: u8) -> &'_ PyArrayDyn<u8> {
        let shape = Shape(shape);
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

    // Note: Can we avoid resintantiating the LocalSyncStorage and TestExecutor
    // everytime we evaludate a computation?
    // What argument type can we expect?
    // #[pyfn(m, "run_py_computation")]
    // fn run_py_computation<'py>(
    //     py: Python<'py>,
    //     storage: HashMap<String, PyReadonlyArrayDyn<f64>>,
    //     computation: Vec<u8>,
    //     arguments: HashMap<String, String>,
    //     // ) -> &'py HashMap<String, PyArrayDyn<f64>> {
    // ) -> &'py PyDict {
    //     let comp = create_computation_graph_from_py_bytes(computation);

    //     let storage_inputs = storage
    //         .iter()
    //         .map(|arg| (arg.0.to_owned(), dynarray_to_value(arg.1)))
    //         .collect::<HashMap<String, Value>>();

    //     let arguments = arguments
    //         .iter()
    //         .map(|arg| (arg.0.to_owned(), Value::from(arg.1.to_owned())))
    //         .collect::<HashMap<String, moose::computation::Value>>();

    //     // Extract ouput keys from saved ops directly from the graph or should be passed as an input the func
    //     let storage: Rc<dyn SyncStorage> = Rc::new(LocalSyncStorage::from_hashmap(storage_inputs));
    //     let exec = TestExecutor::from_storage(&storage);
    //     exec.run_computation(&comp, arguments).unwrap();

    //     let output_keys = vec!["output"];
    //     let mut outputs: HashMap<String, &PyArrayDyn<f64>> = HashMap::new();
    //     for key in output_keys {
    //         let mut value = Float64Tensor::try_from(
    //             storage
    //                 .load(key, &SessionId::from("foobar"), None, "")
    //                 .unwrap(),
    //         )
    //         .unwrap();
    //         outputs.insert(key.to_string(), float64_to_array(value).to_pyarray(py));
    //     }

    //     // outputs.into()
    //     outputs.into_py_dict(py)
    // }

    Ok(())
}

#[pyclass]
pub struct TestRuntime {
    executors: HashMap<String, AsyncExecutor>,
    networking: LocalAsyncNetworking,
    storages: HashMap<String, LocalAsyncStorage>,
}

#[pymethods]
impl TestRuntime {
    #[new]
    fn new(storages: HashMap<String, HashMap<String, PyReadonlyArrayDyn<f64>>>) -> Self {
        let mut executors: HashMap<String, AsyncExecutor> = HashMap::new();
        let networking = LocalAsyncNetworking::default();
        let mut runtime_storages: HashMap<String, LocalAsyncStorage> = HashMap::new();

        for (placement, storage) in storages {
            let storage = storage
                .iter()
                .map(|arg| (arg.0.to_owned(), dynarray_to_value(arg.1)))
                .collect::<HashMap<String, Value>>();

            let exec_storage = LocalAsyncStorage::from_hashmap(storage);
            runtime_storages.insert(placement.clone(), exec_storage);

            let executor = AsyncExecutor::default();
            executors.insert(placement, executor);
        }
        TestRuntime {
            executors,
            networking,
            storages: runtime_storages,
        }
    }

    fn evaluate_computation(&self, computation: Vec<u8>, arguments: HashMap<String, String>) {
        let moose_sessions: HashMap<String, AsyncSession> = HashMap::new();

        let arguments = arguments
            .iter()
            .map(|arg| (arg.0.clone(), Value::from(arg.1.clone())))
            .collect::<HashMap<String, Value>>();

        let role_assignment = &self
            .executors
            .keys()
            .into_iter()
            .map(|arg| (Role::from(arg), Identity::from(arg)))
            .collect::<HashMap<Role, Identity>>();

        for (placement, executor) in &self.executors {
            let mut moose_session = AsyncSession {
                sid: SessionId::from("foo"),
                arguments: arguments.clone(),
                networking: Arc::new(self.networking),
                storage: Arc::new(self.storages[placement]),
            };

            let own_identity = Identity::from(placement);
            let computation = create_computation_graph_from_py_bytes(computation.clone());

            let (mut moose_session_handle, _outputs) = executor
                .run_computation(&computation, &role_assignment, &own_identity, moose_session)
                .unwrap();

            moose_session_handle.join();
        }
    }

    fn get_value_from_storage(&self, key: String) {}
}

#[pymodule]
fn test_runtime(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<TestRuntime>();
    Ok(())
}
