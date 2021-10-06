use moose::fixedpoint::Convert;
use moose::host::{HostBitTensor, HostFloat64Tensor, HostRing64Tensor, RawShape};
use moose::prim::RawSeed;
use moose::prng::AesRng;
use moose::utils;
use ndarray::ArrayD;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, ToPyArray};
use pyo3::{prelude::*, types::PyBytes, types::PyList};
use std::convert::TryInto;
use std::num::Wrapping;

fn dynarray_to_ring64(arr: &PyReadonlyArrayDyn<u64>) -> HostRing64Tensor {
    let arr_wrap = arr.as_array().mapv(Wrapping);
    HostRing64Tensor::new(arr_wrap)
}

fn ring64_to_array(r: HostRing64Tensor) -> ArrayD<u64> {
    let inner_arr = r.0;
    let shape = inner_arr.shape();
    let unwrapped = inner_arr.mapv(|x| x.0);
    unwrapped.into_shape(shape).unwrap()
}

fn binary_pyfn<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<u64>,
    y: PyReadonlyArrayDyn<u64>,
    binary_op: impl Fn(HostRing64Tensor, HostRing64Tensor) -> HostRing64Tensor,
) -> &'py PyArrayDyn<u64> {
    let x_ring = dynarray_to_ring64(&x);
    let y_ring = dynarray_to_ring64(&y);
    let res = binary_op(x_ring, y_ring);
    let res_array = ring64_to_array(res);
    res_array.to_pyarray(py)
}

#[pymodule]
fn moose_kernels(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "ring_add")]
    fn ring_add<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArrayDyn<u64> {
        binary_pyfn(py, x, y, |a, b| a + b)
    }

    #[pyfn(m)]
    #[pyo3(name = "ring_mul")]
    fn ring_mul<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArrayDyn<u64> {
        binary_pyfn(py, x, y, |a, b| a * b)
    }

    #[pyfn(m)]
    #[pyo3(name = "ring_dot")]
    fn ring_dot<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArrayDyn<u64> {
        let x_ring = dynarray_to_ring64(&x);
        let y_ring = dynarray_to_ring64(&y);
        let res = x_ring.dot(y_ring).unwrap();
        let res_array = ring64_to_array(res);
        res_array.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "ring_sub")]
    fn ring_sub<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        y: PyReadonlyArrayDyn<u64>,
    ) -> &'py PyArrayDyn<u64> {
        binary_pyfn(py, x, y, |a, b| a - b)
    }

    #[pyfn(m)]
    #[pyo3(name = "ring_sum")]
    fn ring_sum<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        axis: Option<usize>,
    ) -> &'py PyArrayDyn<u64> {
        let x_ring = dynarray_to_ring64(&x);
        let res = x_ring.sum(axis).unwrap();
        let res_array = ring64_to_array(res);
        res_array.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "ring_shape")]
    fn ring_shape<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u64>) -> &'py PyList {
        let shape: &[usize] = x.shape();
        PyList::new(py, shape.iter())
    }

    #[pyfn(m)]
    #[pyo3(name = "sample_key")]
    fn sample_key(py: Python) -> &PyBytes {
        let key: [u8; 16] = AesRng::generate_random_key();
        PyBytes::new(py, &key)
    }

    #[pyfn(m)]
    #[pyo3(name = "derive_seed")]
    fn derive_seed<'py>(py: Python<'py>, seed: &'py PyBytes, nonce: &'py PyBytes) -> &'py PyBytes {
        let new_seed = utils::derive_seed(seed.as_bytes(), nonce.as_bytes());
        PyBytes::new(py, &new_seed)
    }

    #[pyfn(m)]
    #[pyo3(name = "ring_fill")]
    fn ring_fill(py: Python<'_>, shape: Vec<usize>, el: u64) -> &'_ PyArrayDyn<u64> {
        let shape = RawShape(shape);
        let res = HostRing64Tensor::fill(&shape, el);
        let res_array = ring64_to_array(res);
        res_array.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "ring_sample")]
    fn ring_sample<'py>(
        py: Python<'py>,
        shape: Vec<usize>,
        seed: &'py PyBytes,
        max_value: Option<u64>,
    ) -> &'py PyArrayDyn<u64> {
        let res = match max_value {
            None => HostRing64Tensor::sample_uniform_seeded(
                &RawShape(shape),
                &RawSeed(seed.as_bytes().try_into().unwrap()),
            ),
            Some(max_value) => {
                if max_value == 1 {
                    HostRing64Tensor::sample_bits_seeded(
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

    #[pyfn(m)]
    #[pyo3(name = "ring_shl")]
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

    #[pyfn(m)]
    #[pyo3(name = "ring_shr")]
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

    #[pyfn(m)]
    #[pyo3(name = "bit_xor")]
    fn bit_xor<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u8>,
        y: PyReadonlyArrayDyn<u8>,
    ) -> &'py PyArrayDyn<u8> {
        let b1 = HostBitTensor::from(x.to_owned_array());
        let b2 = HostBitTensor::from(y.to_owned_array());
        ArrayD::<u8>::from(b1 ^ b2).to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "bit_and")]
    fn bit_and<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u8>,
        y: PyReadonlyArrayDyn<u8>,
    ) -> &'py PyArrayDyn<u8> {
        let b1 = HostBitTensor::from(x.to_owned_array());
        let b2 = HostBitTensor::from(y.to_owned_array());
        ArrayD::<u8>::from(b1 & b2).to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "bit_sample")]
    fn bit_sample<'py>(
        py: Python<'py>,
        shape: Vec<usize>,
        seed: &'py PyBytes,
    ) -> &'py PyArrayDyn<u8> {
        let shape = RawShape(shape);
        let seed = RawSeed(seed.as_bytes().try_into().unwrap());
        let b = HostBitTensor::sample_uniform_seeded(&shape, &seed);
        ArrayD::<u8>::from(b).to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "bit_fill")]
    fn bit_fill(py: Python<'_>, shape: Vec<usize>, el: u8) -> &'_ PyArrayDyn<u8> {
        let shape = RawShape(shape);
        let res = HostBitTensor::fill(&shape, el);
        ArrayD::<u8>::from(res).to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "bit_extract")]
    fn bit_extract<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        bit_idx: usize,
    ) -> &'py PyArrayDyn<u8> {
        let x_ring = dynarray_to_ring64(&x);
        let res = x_ring.bit_extract(bit_idx);
        ArrayD::<u8>::from(res).to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "ring_inject")]
    fn ring_inject<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u8>,
        bit_idx: usize,
    ) -> &'py PyArrayDyn<u64> {
        let b = HostBitTensor::from(x.to_owned_array());
        let res = HostRing64Tensor::from(b) << bit_idx;
        ring64_to_array(res).to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "bit_shape")]
    fn bit_shape<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>) -> &'py PyList {
        let shape: &[usize] = x.shape();
        PyList::new(py, shape.iter())
    }

    #[pyfn(m)]
    #[pyo3(name = "fixedpoint_encode")]
    fn fixedpoint_encode<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        scaling_factor: u64,
    ) -> &'py PyArrayDyn<u64> {
        let x = HostFloat64Tensor::from(x.to_owned_array());
        let y = HostRing64Tensor::encode(&x, scaling_factor);
        ring64_to_array(y).to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "fixedpoint_decode")]
    fn fixedpoint_decode<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        scaling_factor: u64,
    ) -> &'py PyArrayDyn<f64> {
        let x_ring = dynarray_to_ring64(&x);
        let y = HostRing64Tensor::decode(&x_ring, scaling_factor);
        y.0.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "fixedpoint_ring_mean")]
    fn fixedpoint_ring_mean<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<u64>,
        axis: Option<usize>,
        precision: u32,
    ) -> &'py PyArrayDyn<u64> {
        let x_ring = dynarray_to_ring64(&x);
        let y = HostRing64Tensor::fixedpoint_mean(x_ring, axis, 2u64.pow(precision)).unwrap();
        ring64_to_array(y).to_pyarray(py)
    }

    Ok(())
}
