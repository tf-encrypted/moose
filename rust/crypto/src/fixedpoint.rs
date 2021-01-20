use crate::ring::Ring64Tensor;
use ndarray::prelude::*;
use std::ops::Mul;

pub fn ring_encode(x: &ArrayViewD<f64>, scaling_factor: u64) -> Ring64Tensor {
    let x_upshifted = x * scaling_factor as f64;
    let x_converted: ArrayD<i64> = x_upshifted.mapv(|el| el as i64);
    Ring64Tensor::from(x_converted)
}

pub fn ring_decode(x: &Ring64Tensor, scaling_factor: u64) -> ArrayD<f64> {
    let x_upshifted: ArrayD<i64> = x.into();
    let x_converted = x_upshifted.mapv(|el| el as f64);
    x_converted / scaling_factor as f64
}

pub fn ring_mean(x: Ring64Tensor, axis: Option<usize>, scaling_factor: u64) -> Ring64Tensor {
    let shape: &[usize] = x.0.shape();
    if let Some(ax) = axis {
        let dim_len = shape[ax] as f64;
        let mean_weight = Array::from_elem([], 1.0 / dim_len)
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let encoded_weight = ring_encode(&mean_weight.view(), scaling_factor);
        let operand_sum = x.sum(axis);
        operand_sum.mul(encoded_weight)
    } else {
        let dim_prod: usize = std::iter::Product::product(shape.iter());
        let prod_inv = 1.0 / dim_prod as f64;
        let mean_weight = Array::from_elem([], prod_inv)
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let encoded_weight = ring_encode(&mean_weight.view(), scaling_factor);
        let operand_sum = x.sum(None);
        operand_sum.mul(encoded_weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_fixedpoint() {
        let x = array![1.0, -2.0, 3.0, -4.0]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let scaling_factor = 2u64.pow(16);
        let x_encoded = ring_encode(&x.view(), scaling_factor);
        assert_eq!(
            x_encoded,
            Ring64Tensor::from(vec![
                65536,
                18446744073709420544,
                196608,
                18446744073709289472
            ])
        );

        let x_decoded = ring_decode(&x_encoded, scaling_factor);
        assert_eq!(x_decoded, x);
    }

    #[test]
    fn ring_mean_with_axis() {
        let x_backing: ArrayD<f64> = array![[1., 2.], [3., 4.]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let encoding_factor = 2u64.pow(16);
        let decoding_factor = 2u64.pow(32);
        let x = ring_encode(&x_backing.view(), encoding_factor);
        let out = ring_mean(x, Some(0), encoding_factor);
        let dec = ring_decode(&out, decoding_factor);
        assert_eq!(dec, array![2., 3.].into_dimensionality::<IxDyn>().unwrap());
    }

    #[test]
    fn ring_mean_no_axis() {
        let x_backing: ArrayD<f64> = array![[1., 2.], [3., 4.]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let encoding_factor = 2u64.pow(16);
        let decoding_factor = 2u64.pow(32);
        let x = ring_encode(&x_backing.view(), encoding_factor);
        let out = ring_mean(x, None, encoding_factor);
        let dec = ring_decode(&out, decoding_factor);
        assert_eq!(dec.into_shape((1,)).unwrap(), array![2.5].into_shape((1,)).unwrap());
    }
}
