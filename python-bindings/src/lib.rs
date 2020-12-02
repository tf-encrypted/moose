use crypto::Ring64Tensor;
use ndarray::ArrayD;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use std::num::Wrapping;

#[pymodule]
fn moose_kernels(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn dynarray_to_ring64(arr: &PyReadonlyArrayDyn<u64>) -> Ring64Tensor {
        let arr_wrap = arr.as_array().mapv(Wrapping);
        Ring64Tensor(arr_wrap)
    }

    fn ring64_to_array(r: Ring64Tensor, shape: &[usize]) -> ArrayD<u64> {
        let inner_arr = r.0;
        let unwrapped = inner_arr.mapv(|x| x.0);
        unwrapped.into_shape(shape).unwrap()
    }

    #[pyfn(m, "ring_add")]
    fn ring_add<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArrayDyn<u64> {
        let x_shape = x.shape();
        let x_ring = dynarray_to_ring64(&x);
        let y_ring = dynarray_to_ring64(&y);
        let addn = x_ring + y_ring;
        let res = ring64_to_array(addn, x_shape);
        res.to_pyarray(py)
    }

    #[pyfn(m, "ring_mul")]
    fn ring_mul<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArrayDyn<u64> {
        let x_shape = x.shape();
        let x_ring = dynarray_to_ring64(&x);
        let y_ring = dynarray_to_ring64(&y);
        let prod = x_ring * y_ring;
        let res = ring64_to_array(prod, x_shape);
        res.to_pyarray(py)
    }

    #[pyfn(m, "ring_sub")]
    fn ring_sub<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArrayDyn<u64> {
        let x_shape = x.shape();
        let x_ring = dynarray_to_ring64(&x);
        let y_ring = dynarray_to_ring64(&y);
        let diff = x_ring - y_ring;
        let res = ring64_to_array(diff, x_shape);
        res.to_pyarray(py)
    }

    Ok(())
}
