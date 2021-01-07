use crate::ring::Ring64Tensor;
use ndarray::prelude::*;

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
}
