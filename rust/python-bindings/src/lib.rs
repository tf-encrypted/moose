use crypto::prng::AesRng;
use crypto::utils;
use crypto::ring::{Dot, Fill, Ring64Tensor};
use ndarray::ArrayD;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, ToPyArray};
use pyo3::{prelude::*, types::PyBytes, types::PyList};
use std::num::Wrapping;

fn dynarray_to_ring64(arr: &PyReadonlyArrayDyn<u64>) -> Ring64Tensor {
    let arr_wrap = arr.as_array().mapv(Wrapping);
    Ring64Tensor(arr_wrap)
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
    fn derive_seed(py: Python, seed: &PyBytes, nonce: u128) -> PyObject {
        // (Dragos) for some unknow reason pyo3 doesn't support returing a PyBytes type without a lifetime attached to it
        // https://users.rust-lang.org/t/pyo3-best-way-to-return-bytes-from-function-call/46577/3
        let new_seed = utils::derive_seed(seed.as_bytes(), nonce);
        PyBytes::new(py, &new_seed).into()
    }

    #[pyfn(m, "ring_fill")]
    fn ring_fill(py: Python<'_>, shape: Vec<usize>, el: u64) -> &'_ PyArrayDyn<u64> {
        let res = Ring64Tensor::fill(&shape, el);
        let res_array = ring64_to_array(res);
        res_array.to_pyarray(py)
    }

    Ok(())
}
