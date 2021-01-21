use ndarray::prelude::*;
use rand::prelude::*;
use std::convert::TryInto;
use std::num::Wrapping;
use std::ops::{BitAnd, BitXor};

use crate::prng::{AesRng, PRNGSeed};

#[derive(Clone, Debug, PartialEq)]
pub struct BitTensor(pub ArrayD<u8>);

pub trait Sample {
    fn sample_uniform(shape: &[usize], key: &[u8]) -> Self;
}

impl Sample for BitTensor {
    fn sample_uniform(shape: &[usize], key: &[u8]) -> Self {
        let seed: PRNGSeed = key.try_into().unwrap();
        let mut rng = AesRng::from_seed(seed);
        let length = shape.iter().product();
        let values: Vec<_> = (0..length).map(|_| rng.get_bit()).collect();
        let ix = IxDyn(shape);
        BitTensor(Array::from_shape_vec(ix, values).unwrap())
    }
}

impl BitTensor {
    pub fn fill(shape: &[usize], el: u8) -> BitTensor {
        BitTensor(ArrayD::from_elem(shape, el & 1))
    }
}

impl From<ArrayD<u8>> for BitTensor {
    fn from(a: ArrayD<u8>) -> BitTensor {
        let wrapped = a.mapv(|ai| (ai & 1) as u8);
        BitTensor(wrapped)
    }
}

impl From<&BitTensor> for ArrayD<u8> {
    fn from(r: &BitTensor) -> ArrayD<u8> {
        r.0.mapv(|element| element as u8)
    }
}

impl From<Vec<u8>> for BitTensor {
    fn from(v: Vec<u8>) -> BitTensor {
        let ix = IxDyn(&[v.len()]);
        BitTensor(Array::from_shape_vec(ix, v).unwrap())
    }
}

impl From<&[u8]> for BitTensor {
    fn from(v: &[u8]) -> BitTensor {
        let ix = IxDyn(&[v.len()]);
        let v_wrapped: Vec<_> = v.iter().map(|vi| *vi & 1).collect();
        BitTensor(Array::from_shape_vec(ix, v_wrapped).unwrap())
    }
}

impl BitXor for BitTensor {
    type Output = BitTensor;
    fn bitxor(self, other: Self) -> Self::Output {
        BitTensor(self.0 ^ other.0)
    }
}

impl BitAnd for BitTensor {
    type Output = BitTensor;
    fn bitand(self, other: Self) -> Self::Output {
        BitTensor(self.0 & other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_sample() {
        let key = [0u8; 16];
        let r = BitTensor::sample_uniform(&[5], &key);
        assert_eq!(
            r,
            BitTensor::from(vec![
                0,
                1,
                1,
                0,
                0,
            ])
        );
    }

    #[test]
    fn bit_fill() {
        let r = BitTensor::fill(&[2], 1);
        assert_eq!(r, BitTensor::from(vec![1, 1]))
    }

    #[test]
    fn bit_ops() {
        let shape = 5;

        // test xor
        assert_eq!(BitTensor::fill(&[shape], 0) ^ BitTensor::fill(&[shape], 1), BitTensor::fill(&[shape], 1));
        assert_eq!(BitTensor::fill(&[shape], 1) ^ BitTensor::fill(&[shape], 0), BitTensor::fill(&[shape], 1));
        assert_eq!(BitTensor::fill(&[shape], 1) ^ BitTensor::fill(&[shape], 1), BitTensor::fill(&[shape], 0));
        assert_eq!(BitTensor::fill(&[shape], 0) ^ BitTensor::fill(&[shape], 0), BitTensor::fill(&[shape], 0));

        // test and
        assert_eq!(BitTensor::fill(&[shape], 0) & BitTensor::fill(&[shape], 1), BitTensor::fill(&[shape], 0));
        assert_eq!(BitTensor::fill(&[shape], 1) & BitTensor::fill(&[shape], 0), BitTensor::fill(&[shape], 0));
        assert_eq!(BitTensor::fill(&[shape], 1) & BitTensor::fill(&[shape], 1), BitTensor::fill(&[shape], 1));
        assert_eq!(BitTensor::fill(&[shape], 0) & BitTensor::fill(&[shape], 0), BitTensor::fill(&[shape], 0));

    }

}
