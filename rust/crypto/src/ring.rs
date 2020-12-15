use ndarray::prelude::*;
use ndarray::linalg::Dot;
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use std::num::Wrapping;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Debug, PartialEq)]
pub struct Ring64Tensor<D: Dimension>(pub Array<Wrapping<u64>, D>);

pub trait Sample {
    fn sample_uniform(shape: &[usize]) -> Self;
}

impl Sample for Ring64Tensor<IxDyn> {
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

impl From<Vec<u64>> for Ring64Tensor {
    fn from(v: Vec<u64>) -> Ring64Tensor {
        let ix = IxDyn(&[v.len()]);
        use vec_utils::VecExt;
        let v_wrapped: Vec<_> = v.map(Wrapping);
        Ring64Tensor(Array::from_shape_vec(ix, v_wrapped).unwrap())
    }
}

impl From<Vec<u64>> for Ring64Tensor<Ix1> {
    fn from(v: Vec<u64>) -> Ring64Tensor<Ix1> {
        let ix = Ix1(v.len());
        use vec_utils::VecExt;
        let v_wrapped: Vec<_> = v.map(Wrapping);
        Ring64Tensor(Array::from_shape_vec(ix, v_wrapped).unwrap())
    }
}

impl From<&[u64]> for Ring64Tensor<IxDyn> {
    fn from(v: &[u64]) -> Ring64Tensor<IxDyn> {
        let ix = IxDyn(&[v.len()]);
        let v_wrapped: Vec<_> = v.iter().map(|vi| Wrapping(*vi)).collect();
        Ring64Tensor(Array::from_shape_vec(ix, v_wrapped).unwrap())
    }
}

impl<D: Dimension> Add for Ring64Tensor<D> {
    type Output = Ring64Tensor<D>;
    fn add(self, other: Self) -> Self::Output {
        Ring64Tensor(self.0.add(other.0))
    }
}

impl<D: Dimension> Mul for Ring64Tensor<D> {
    type Output = Ring64Tensor<D>;
    fn mul(self, other: Self) -> Self::Output {
        Ring64Tensor(self.0.mul(other.0))
    }
}

impl<D: Dimension> Sub for Ring64Tensor<D> {
    type Output = Ring64Tensor<D>;
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

// impl Dot<Ring64Tensor1> for Ring64Tensor1 {
//     fn dot(self, other: Ring64Tensor1) -> Ring64Tensor1 {
//         self.0.dot(other.0)
//     }
// }

// impl Dot<Ring64Tensor2> for Ring64Tensor2 {
//     fn dot(self, other: Ring64Tensor2) -> Ring64Tensor2 {
//         self.0.dot(other.0)
//     }
// }

// pub struct Replicated<T>(T, T, T);

// impl<T> Mul<Replicated<T>> for Replicated<T>
// where
//     T: Mul<T, Output = T>,
// {
//     type Output = Replicated<T>;
//     fn mul(self, other: Replicated<T>) -> Self::Output {
//         // TODO
//         Replicated(self.0 * other.0, self.1 * other.1, self.2 * other.2)
//     }
// }

// pub fn share(x: &Ring64Tensor) -> Replicated<Ring64Tensor> {
//     // TODO
//     Replicated(x.clone(), x.clone(), x.clone())
// }

// pub fn reconstruct<T>(x: Replicated<T>) -> T {
//     // TODO
//     x.0
// }

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn it_works() {
    //     let a = Ring64Tensor::from(vec![1, 2, 3]);
    //     let b = Ring64Tensor::from(vec![4, 5, 6]);

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
