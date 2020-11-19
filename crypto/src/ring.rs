use ndarray::prelude::*;
use rand::prelude::*;
use std::num::Wrapping;
use std::ops::{Add, Mul};
use rand_chacha::ChaCha20Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Ring64Vector(Array1<Wrapping<u64>>);

pub trait Sample {
    fn sample_uniform(size: usize) -> Self;
}

impl Sample for Ring64Vector {
    fn sample_uniform(size: usize) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        Ring64Vector(Array1::from(values))
    }
}

impl From<Vec<u64>> for Ring64Vector {
    fn from(v: Vec<u64>) -> Ring64Vector {
        use vec_utils::VecExt;
        let v_wrapped: Vec<_> = v.map(Wrapping);
        Ring64Vector(Array1::from(v_wrapped))
    }
}

impl From<&[u64]> for Ring64Vector {
    fn from(v: &[u64]) -> Ring64Vector {
        let v_wrapped: Vec<_> = v.iter().map(|vi| Wrapping(*vi)).collect();
        Ring64Vector(Array1::from(v_wrapped))
    }
}

impl Add<Ring64Vector> for Ring64Vector {
    type Output = Ring64Vector;
    fn add(self, other: Ring64Vector) -> Self::Output {
        Ring64Vector(self.0.add(other.0))
    }
}

impl Mul<Ring64Vector> for Ring64Vector {
    type Output = Ring64Vector;
    fn mul(self, other: Ring64Vector) -> Self::Output {
        Ring64Vector(self.0.mul(other.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_mul() {
        let a = Ring64Vector::from(vec![1, 2, 3]);
        let b = Ring64Vector::from(vec![4, 5, 6]);
        let c = a * b;
        assert_eq!(c, Ring64Vector::from(vec![4, 10, 18]));
    }

    #[test]
    fn ring_sample() {
        let r = Ring64Vector::sample_uniform(5);
        assert_eq!(r, Ring64Vector::from(vec![4, 10, 18]));
    }
}
