pub mod prng;
pub mod ring;

use ring::Ring64Vector;
use std::ops::{Mul};

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
