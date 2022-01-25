use super::*;

pub trait Convert<T> {
    type Scale: num_traits::One + Clone;
    fn encode(x: &T, scaling_factor: Self::Scale) -> Self;
    fn decode(x: &Self, scaling_factor: Self::Scale) -> T;
}

impl Convert<HostFloat64Tensor> for HostRing64Tensor {
    type Scale = u64;
    fn encode(x: &HostFloat64Tensor, scaling_factor: Self::Scale) -> HostRing64Tensor {
        let x_upshifted = &x.0 * (scaling_factor as f64);
        let x_converted: ArrayD<u64> = x_upshifted.mapv(|el| (el as i64) as u64);
        HostRing64Tensor::from_raw_plc(x_converted, x.1.clone())
    }
    fn decode(x: &Self, scaling_factor: Self::Scale) -> HostFloat64Tensor {
        let x_upshifted: ArrayD<i64> = ArrayD::from(x);
        let x_converted = x_upshifted.mapv(|el| el as f64);
        HostFloat64Tensor::from_raw_plc(x_converted / scaling_factor as f64, x.1.clone())
    }
}

impl Convert<HostFloat64Tensor> for HostRing128Tensor {
    type Scale = u128;
    fn encode(x: &HostFloat64Tensor, scaling_factor: Self::Scale) -> HostRing128Tensor {
        let x_upshifted = &x.0 * (scaling_factor as f64);
        let x_converted: ArrayD<u128> = x_upshifted.mapv(|el| (el as i128) as u128);
        HostRing128Tensor::from_raw_plc(x_converted, x.1.clone())
    }
    fn decode(x: &Self, scaling_factor: Self::Scale) -> HostFloat64Tensor {
        let x_upshifted: ArrayD<i128> = ArrayD::from(x);
        let x_converted = x_upshifted.mapv(|el| el as f64);
        HostFloat64Tensor::from_raw_plc(x_converted / scaling_factor as f64, x.1.clone())
    }
}

impl<T> HostRingTensor<T>
where
    Wrapping<T>: Clone + num_traits::Zero + std::ops::Mul<Wrapping<T>, Output = Wrapping<T>>,
    HostRingTensor<T>: Convert<HostFloat64Tensor>,
{
    fn mul(self, other: HostRingTensor<T>) -> HostRingTensor<T> {
        HostRingTensor(self.0 * other.0, self.1)
    }

    fn compute_mean_weight(&self, axis: &Option<usize>) -> Result<HostFloat64Tensor> {
        let shape: &[usize] = self.0.shape();
        if let Some(ax) = axis {
            let dim_len = shape[*ax] as f64;
            Ok(HostFloat64Tensor::from_raw_plc(
                Array::from_elem([], 1.0 / dim_len)
                    .into_dimensionality::<IxDyn>()
                    .map_err(|e| Error::KernelError(e.to_string()))?,
                self.1.clone(),
            ))
        } else {
            let dim_prod: usize = std::iter::Product::product(shape.iter());
            let prod_inv = 1.0 / dim_prod as f64;
            Ok(HostFloat64Tensor::from_raw_plc(
                Array::from_elem([], prod_inv)
                    .into_dimensionality::<IxDyn>()
                    .map_err(|e| Error::KernelError(e.to_string()))?,
                self.1.clone(),
            ))
        }
    }

    pub(super) fn fixedpoint_mean(
        x: Self,
        axis: Option<usize>,
        scaling_factor: <HostRingTensor<T> as Convert<HostFloat64Tensor>>::Scale,
    ) -> Result<HostRingTensor<T>> {
        let mean_weight = x.compute_mean_weight(&axis)?;
        let encoded_weight = HostRingTensor::<T>::encode(&mean_weight, scaling_factor);
        let operand_sum = x.sum(axis)?;
        Ok(operand_sum.mul(encoded_weight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixedpoint_mean_with_axis() {
        let plc = HostPlacement::from("alice");

        let x_backing = HostFloat64Tensor::from_raw_plc(
            array![[1., 2.], [3., 4.]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            plc.clone(),
        );
        let encoding_factor = 2u64.pow(16);
        let decoding_factor = 2u64.pow(32);
        let x = HostRing64Tensor::encode(&x_backing, encoding_factor);
        let out = HostRing64Tensor::fixedpoint_mean(x, Some(0), encoding_factor).unwrap();
        let dec = HostRing64Tensor::decode(&out, decoding_factor);
        assert_eq!(
            dec,
            HostFloat64Tensor::from_raw_plc(
                array![2., 3.].into_dimensionality::<IxDyn>().unwrap(),
                plc
            )
        );
    }

    #[test]
    fn fixedpoint_mean_no_axis() {
        let plc = HostPlacement::from("alice");

        let x_backing = HostFloat64Tensor::from_raw_plc(
            array![[1., 2.], [3., 4.]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            plc.clone(),
        );
        let encoding_factor = 2u64.pow(16);
        let decoding_factor = 2u64.pow(32);
        let x = HostRing64Tensor::encode(&x_backing, encoding_factor);
        let out = HostRing64Tensor::fixedpoint_mean(x, None, encoding_factor).unwrap();
        let dec = HostRing64Tensor::decode(&out, decoding_factor);
        assert_eq!(
            dec.0.into_shape((1,)).unwrap(),
            array![2.5].into_shape((1,)).unwrap()
        );
    }
}
