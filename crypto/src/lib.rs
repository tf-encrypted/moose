use ndarray::prelude::*;

use std::num::Wrapping;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Debug, PartialEq)]
pub struct Ring64Tensor(pub ArrayD<Wrapping<u64>>);

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

impl Add<Ring64Tensor> for Ring64Tensor {
    type Output = Ring64Tensor;
    fn add(self, other: Ring64Tensor) -> Self::Output {
        Ring64Tensor(self.0.add(other.0))
    }
}

impl Mul<Ring64Tensor> for Ring64Tensor {
    type Output = Ring64Tensor;
    fn mul(self, other: Ring64Tensor) -> Self::Output {
        Ring64Tensor(self.0.mul(other.0))
    }
}

impl Sub<Ring64Vector> for Ring64Vector {
    type Output = Ring64Vector;
    fn sub(self, other: Ring64Vector) -> Self::Output {
        Ring64Vector(self.0.sub(other.0))
    }
}

pub struct Replicated<T>(T, T, T);

impl<T> Mul<Replicated<T>> for Replicated<T>
where
    T: Mul<T, Output = T>,
{
    type Output = Replicated<T>;
    fn mul(self, other: Replicated<T>) -> Self::Output {
        // TODO
        Replicated(self.0 * other.0, self.1 * other.1, self.2 * other.2)
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
}
