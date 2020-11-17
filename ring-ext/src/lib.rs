use crypto::{Ring64Vector};
use::ndarray::Ix1;
use numpy::{PyReadonlyArrayDyn, PyArray, ToPyArray};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn ring_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // immutable example
    fn ring_add_impl(x: Vec<u64>, y: Vec<u64>) -> Vec<u64> {
        let rx = Ring64Vector::from(x);
        let ry = Ring64Vector::from(y);
        Vec::from(rx + ry)
    }

    // wrapper of `ring_add`
    #[pyfn(m, "ring_add")]
    fn ring_add<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArray<u64, Ix1> {
        let x = x.to_vec().expect("Input array `x` must be contiguous.");
        let y = y.to_vec().expect("Input array `y` must be contiguous.");
        let addn = ring_add_impl(x, y);
        addn.to_pyarray(py)
    }

    Ok(())
}
