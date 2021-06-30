use crate::computation::Placed;
use crate::computation::{BitAndOp, BitSampleOp, BitXorOp, HostPlacement};
use crate::kernels::ConcreteContext;
use crate::prim::{RawSeed, Seed};
use crate::prng::AesRng;
use crate::standard::{RawShape, Shape};
use ndarray::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{BitAnd, BitXor};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BitTensor(pub ArrayD<u8>, HostPlacement);

impl Placed for BitTensor {
    type Placement = HostPlacement;

    fn placement(&self) -> Self::Placement {
        self.1.clone()
    }
}

kernel! {
    BitSampleOp,
    [
        (HostPlacement, (Seed, Shape) -> BitTensor => Self::kernel),
    ]
}

impl BitSampleOp {
    fn kernel(_ctx: &ConcreteContext, plc: &HostPlacement, seed: Seed, shape: Shape) -> BitTensor {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        BitTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }
}

kernel! {
    BitXorOp,
    [
        (HostPlacement, (BitTensor, BitTensor) -> BitTensor => Self::kernel),
    ]
}

impl BitXorOp {
    fn kernel(
        _ctx: &ConcreteContext,
        plc: &HostPlacement,
        x: BitTensor,
        y: BitTensor,
    ) -> BitTensor {
        BitTensor(x.0 ^ y.0, plc.clone())
    }
}

kernel! {
    BitAndOp,
    [
        (HostPlacement, (BitTensor, BitTensor) -> BitTensor => Self::kernel),
    ]
}

impl BitAndOp {
    fn kernel(
        _ctx: &ConcreteContext,
        plc: &HostPlacement,
        x: BitTensor,
        y: BitTensor,
    ) -> BitTensor {
        BitTensor(x.0 & y.0, plc.clone())
    }
}

impl BitTensor {
    pub fn sample_uniform(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let ix = IxDyn(shape.0.as_ref());
        BitTensor(
            Array::from_shape_vec(ix, values).unwrap(),
            HostPlacement {
                owner: "TODO".into(),
            },
        )
    }
}

impl BitTensor {
    pub fn fill(shape: &RawShape, el: u8) -> BitTensor {
        assert!(
            el == 0 || el == 1,
            "cannot fill a BitTensor with a value {:?}",
            el
        );
        BitTensor(
            ArrayD::from_elem(shape.0.as_ref(), el & 1),
            HostPlacement {
                owner: "TODO".into(),
            },
        )
    }
}

impl From<ArrayD<u8>> for BitTensor {
    fn from(a: ArrayD<u8>) -> BitTensor {
        let wrapped = a.mapv(|ai| (ai & 1) as u8);
        BitTensor(
            wrapped,
            HostPlacement {
                owner: "TODO".into(),
            },
        )
    }
}

impl From<Vec<u8>> for BitTensor {
    fn from(v: Vec<u8>) -> BitTensor {
        let ix = IxDyn(&[v.len()]);
        BitTensor(
            Array::from_shape_vec(ix, v).unwrap(),
            HostPlacement {
                owner: "TODO".into(),
            },
        )
    }
}

impl From<&[u8]> for BitTensor {
    fn from(v: &[u8]) -> BitTensor {
        let ix = IxDyn(&[v.len()]);
        let v_wrapped: Vec<_> = v.iter().map(|vi| *vi & 1).collect();
        BitTensor(
            Array::from_shape_vec(ix, v_wrapped).unwrap(),
            HostPlacement {
                owner: "TODO".into(),
            },
        )
    }
}

impl From<BitTensor> for ArrayD<u8> {
    fn from(b: BitTensor) -> ArrayD<u8> {
        b.0
    }
}

impl BitXor for BitTensor {
    type Output = BitTensor;
    fn bitxor(self, other: Self) -> Self::Output {
        assert_eq!(self.1, other.1);
        BitTensor(self.0 ^ other.0, self.1)
    }
}

impl BitAnd for BitTensor {
    type Output = BitTensor;
    fn bitand(self, other: Self) -> Self::Output {
        assert_eq!(self.1, other.1);
        BitTensor(self.0 & other.0, self.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_sample() {
        let shape = RawShape(vec![5]);
        let seed = RawSeed([0u8; 16]);
        let r = BitTensor::sample_uniform(&shape, &seed);
        assert_eq!(r, BitTensor::from(vec![0, 1, 1, 0, 0,]));
    }

    #[test]
    fn bit_fill() {
        let shape = RawShape(vec![2]);
        let r = BitTensor::fill(&shape, 1);
        assert_eq!(r, BitTensor::from(vec![1, 1]))
    }

    #[test]
    fn bit_ops() {
        let shape = RawShape(vec![5]);

        // test xor
        assert_eq!(
            BitTensor::fill(&shape, 0) ^ BitTensor::fill(&shape, 1),
            BitTensor::fill(&shape, 1)
        );
        assert_eq!(
            BitTensor::fill(&shape, 1) ^ BitTensor::fill(&shape, 0),
            BitTensor::fill(&shape, 1)
        );
        assert_eq!(
            BitTensor::fill(&shape, 1) ^ BitTensor::fill(&shape, 1),
            BitTensor::fill(&shape, 0)
        );
        assert_eq!(
            BitTensor::fill(&shape, 0) ^ BitTensor::fill(&shape, 0),
            BitTensor::fill(&shape, 0)
        );

        // test and
        assert_eq!(
            BitTensor::fill(&shape, 0) & BitTensor::fill(&shape, 1),
            BitTensor::fill(&shape, 0)
        );
        assert_eq!(
            BitTensor::fill(&shape, 1) & BitTensor::fill(&shape, 0),
            BitTensor::fill(&shape, 0)
        );
        assert_eq!(
            BitTensor::fill(&shape, 1) & BitTensor::fill(&shape, 1),
            BitTensor::fill(&shape, 1)
        );
        assert_eq!(
            BitTensor::fill(&shape, 0) & BitTensor::fill(&shape, 0),
            BitTensor::fill(&shape, 0)
        );
    }
}
