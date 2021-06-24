use moose::bit::BitTensor;
use moose::computation::SessionId;
use moose::computation::Value;
use moose::computation::{Computation, Role};
use moose::execution::{AsyncSession, AsyncSessionHandle};
use moose::execution::{AsyncExecutor, AsyncNetworkingImpl, Identity};
use moose::fixedpoint::Convert;
use moose::networking::{AsyncNetworking, LocalAsyncNetworking};
use moose::prim::Seed;
use moose::prng::AesRng;
use moose::python_computation::PyComputation;
use moose::ring::Ring64Tensor;
use moose::standard::{Float64Tensor, Shape};
use moose::utils;
use ndarray::IxDyn;
use ndarray::{array, ArrayD};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, ToPyArray};

use pyo3::{prelude::*, types::PyBytes, types::PyDict, types::PyList};
use std::collections::HashMap;
use std::convert::TryInto;
use std::num::Wrapping;
use std::sync::Arc;
pub mod python_computation;
use moose::storage::{AsyncStorage, LocalAsyncStorage, LocalSyncStorage};
use std::convert::TryFrom;
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

    Ok(())
}

#[pyclass]
pub struct MooseRuntime {
    executors: HashMap<String, AsyncExecutor>,
    networking: AsyncNetworkingImpl,
    storages: HashMap<String, Arc<dyn Send + Sync + AsyncStorage>>,
}

#[pymethods]
impl MooseRuntime {
    #[new]
    fn new(storages: HashMap<String, HashMap<String, PyReadonlyArrayDyn<f64>>>) -> Self {
        let mut executors: HashMap<String, AsyncExecutor> = HashMap::new();
        let networking: Arc<dyn Send + Sync + AsyncNetworking> =
            Arc::new(LocalAsyncNetworking::default());
        let mut runtime_storages: HashMap<String, Arc<dyn Send + Sync + AsyncStorage>> =
            HashMap::new();

        for (placement, storage) in storages {
            let storage = storage
                .iter()
                .map(|arg| (arg.0.to_owned(), dynarray_to_value(arg.1)))
                .collect::<HashMap<String, Value>>();

            let exec_storage: Arc<dyn Send + Sync + AsyncStorage> =
                Arc::new(LocalAsyncStorage::from_hashmap(storage));
            runtime_storages.insert(placement.clone(), exec_storage);

            let executor = AsyncExecutor::default();
            executors.insert(placement, executor);
        }
        MooseRuntime {
            executors,
            networking,
            storages: runtime_storages,
        }
    }

    fn evaluate_computation(&self, computation: Vec<u8>, arguments: HashMap<String, String>) -> PyResult<()> {
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

        let session_handles: Vec<AsyncSessionHandle>;
        for (placement, executor) in self.executors.iter() {
            let mut moose_session = AsyncSession {
                sid: SessionId::from("foobar"),
                arguments: arguments.clone(),
                networking: Arc::clone(&self.networking),
                storage: Arc::clone(&self.storages[placement]),
            };

            let own_identity = Identity::from(placement);
            let computation = create_computation_graph_from_py_bytes(computation.clone());

            let (mut moose_session_handle, _outputs) = executor
                .run_computation(&computation, &role_assignment, &own_identity, moose_session)
                .unwrap();

            session_handles.push(moose_session_handle)
            // Then await and output and filter units.
        };
        let (_, errors): (Vec<_>, Vec<anyhow::Error>) = session_handles.iter()
            .map(|handle| handle.block_on())
            .partition(|errs| errs.is_empty());
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    // Can we use a block_on approach or do we wan to use pyo3-asyncio
    // to await an async rust in python? https://pyo3.rs/v0.13.2/ecosystem/async-await.html
    fn get_value_from_storage(&self, placement: String, key: String) {
        // If we use this Tokio runtime, it should be moved the class
        let mut rt = Runtime::new().unwrap();
        let val = rt.block_on(async {
            let val = self.storages[&placement]
                .load(&key, &SessionId::from("foobar"), None, "")
                .await
                .unwrap();
            val
        });

        // Return value
    }
}

#[pymodule]
fn moose_runtime(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<MooseRuntime>();
    Ok(())
}
