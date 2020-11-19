use ndarray::prelude::*;

use std::num::Wrapping;
use std::ops::{Add, Mul};

#[derive(Clone, Debug, PartialEq)]
pub struct Ring64Vector(pub Array1<Wrapping<u64>>);

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

pub fn share(x: &Ring64Vector) -> Replicated<Ring64Vector> {
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
        let a = Ring64Vector::from(vec![1, 2, 3]);
        let b = Ring64Vector::from(vec![4, 5, 6]);

        let a_shared = share(&a);
        let b_shared = share(&b);

        let c_shared = a_shared * b_shared;
        let c: Ring64Vector = reconstruct(c_shared);
        assert_eq!(c, a * b);
    }
}
