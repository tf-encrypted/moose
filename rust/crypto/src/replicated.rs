use crate::ring::Ring64Tensor;
use ndarray::prelude::*;

pub fn fixedpoint_encode(x: &ArrayViewD<f64>, scaling_factor: u64) -> Ring64Tensor {
    let x_upshifted = x * scaling_factor as f64;
    let x_converted = x_upshifted.mapv(|element| element as i64 as u64);
    Ring64Tensor::from(x_converted)
}

pub fn fixedpoint_decode(x: &Ring64Tensor, scaling_factor: u64) -> ArrayD<f64> {
    let x_converted = x.0.map(|element| element.0 as i64 as f64);
    x_converted / scaling_factor as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicated_fixedpoint() {
        let x = array![1.0, -2.0, 3.0, -4.0]
            .into_dimensionality::<IxDyn>()
            .unwrap();

        let scaling_factor = 2u64.pow(16);
        let x_encoded = fixedpoint_encode(&x.view(), scaling_factor);
        assert_eq!(
            x_encoded,
            Ring64Tensor::from(vec![
                65536,
                18446744073709420544,
                196608,
                18446744073709289472
            ])
        );

        let x_decoded = fixedpoint_decode(&x_encoded, scaling_factor);
        assert_eq!(x_decoded, x);
    }
}
