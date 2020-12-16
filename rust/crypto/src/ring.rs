use ndarray::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use std::num::Wrapping;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Debug, PartialEq)]
pub struct Ring64Tensor(pub ArrayD<Wrapping<u64>>);

pub trait Sample {
    fn sample_uniform(shape: &[usize]) -> Self;
}

impl Sample for Ring64Tensor {
    fn sample_uniform(shape: &[usize]) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let length = shape.iter().product();
        let values: Vec<_> = (0..length).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape);
        Ring64Tensor(Array::from_shape_vec(ix, values).unwrap())
    }
}

pub trait Fill {
    fn fill(shape: &[usize], el: u64) -> Self;
}

impl Fill for Ring64Tensor {
    fn fill(shape: &[usize], el: u64) -> Self {
        Ring64Tensor(ArrayD::from_elem(shape, Wrapping(el)))
    }
  }

impl From<ArrayD<u64>> for Ring64Tensor {
    fn from(a: ArrayD<u64>) -> Ring64Tensor {
        let wrapped = a.mapv(Wrapping);
        Ring64Tensor(wrapped)
    }
}

impl From<Vec<u64>> for Ring64Tensor {
    fn from(v: Vec<u64>) -> Ring64Tensor {
        let ix = IxDyn(&[v.len()]);
        use vec_utils::VecExt;
        let v_wrapped: Vec<_> = v.map(Wrapping);
        Ring64Tensor(Array::from_shape_vec(ix, v_wrapped).unwrap())
    }
}

impl From<&[u64]> for Ring64Tensor {
    fn from(v: &[u64]) -> Ring64Tensor {
        let ix = IxDyn(&[v.len()]);
        let v_wrapped: Vec<_> = v.iter().map(|vi| Wrapping(*vi)).collect();
        Ring64Tensor(Array::from_shape_vec(ix, v_wrapped).unwrap())
    }
}

impl Add for Ring64Tensor {
    type Output = Ring64Tensor;
    fn add(self, other: Self) -> Self::Output {
        Ring64Tensor(self.0.add(other.0))
    }
}

impl Mul for Ring64Tensor {
    type Output = Ring64Tensor;
    fn mul(self, other: Self) -> Self::Output {
        Ring64Tensor(self.0.mul(other.0))
    }
}

impl Sub for Ring64Tensor {
    type Output = Ring64Tensor;
    fn sub(self, other: Self) -> Self::Output {
        Ring64Tensor(self.0.sub(other.0))
    }
}

impl Dot<Ring64Tensor<Ix1>> for Ring64Tensor<Ix2> {
    type Output = Ring64Tensor<Ix1>;
    fn dot(&self, rhs: &Ring64Tensor<Ix1>) -> Self::Output {
        Ring64Tensor(self.0.dot(&rhs.0))
    }
}

pub fn share(x: &Ring64Tensor) -> Replicated<Ring64Tensor> {
    // TODO
    Replicated(x.clone(), x.clone(), x.clone())
}

pub fn reconstruct<T>(x: Replicated<T>) -> T {
    // TODO
    x.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = Ring64Tensor::from(vec![1, 2, 3]);
        let b = Ring64Tensor::from(vec![4, 5, 6]);

        let a_shared = share(&a);
        let b_shared = share(&b);

        let c_shared = a_shared * b_shared;
        let c: Ring64Tensor = reconstruct(c_shared);
        assert_eq!(c, a * b);
    }

    //     let a_shared = share(&a);
    //     let b_shared = share(&b);

    //     let c_shared = a_shared * b_shared;
    //     let c: Ring64Tensor<Ix1> = reconstruct(c_shared);
    //     assert_eq!(c, a * b);
    // }

    #[test]
    fn ring_dot() {
        let x = Ring64Tensor::<Ix2>::from(array![[1, 2], [3, 4]]);
        let y = Ring64Tensor::<Ix1>::from(vec![1, 1]);
        let z = x.dot(&y);

        let result = Ring64Tensor::<Ix1>::from(vec![3, 7]);
        assert_eq!(result, z)
    }

    #[test]
    fn ring_sample() {
        let r = Ring64Tensor::sample_uniform(&[5]);
        assert_eq!(
            r,
            Ring64Tensor::from(vec![
                9482535800248027256,
                7566832397956113305,
                1804347359131428821,
                3088291667719571736,
                3009633425676235349
            ])
        );
    }

    #[test]
    fn ring_fill() {
        let r = Ring64Tensor::fill(&[2], 1);
        assert_eq!(r, Ring64Tensor::from(vec![1, 1]))
    }
}
