use crate::computation::{HostPlacement, Placed};
use crate::prim::Seed;
use crate::prng::AesRng;
use crate::standard::Shape;
use ndarray::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{BitAnd, BitXor};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BitTensor(pub ArrayD<u8>);

impl BitTensor {
    pub fn sample_uniform(shape: &Shape, seed: &Seed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let ix = IxDyn(shape.0.as_ref());
        BitTensor(Array::from_shape_vec(ix, values).unwrap())
    }
}

impl BitTensor {
    pub fn fill(shape: &Shape, el: u8) -> BitTensor {
        assert!(
            el == 0 || el == 1,
            "cannot fill a BitTensor with a value {:?}",
            el
        );
        BitTensor(ArrayD::from_elem(shape.0.as_ref(), el & 1))
    }
}

impl From<ArrayD<u8>> for BitTensor {
    fn from(a: ArrayD<u8>) -> BitTensor {
        let wrapped = a.mapv(|ai| (ai & 1) as u8);
        BitTensor(wrapped)
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

impl From<BitTensor> for ArrayD<u8> {
    fn from(b: BitTensor) -> ArrayD<u8> {
        b.0
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacedBitTensor(BitTensor, HostPlacement);

impl Placed for PlacedBitTensor {
    type Placement = HostPlacement;

    fn placement(&self) -> Self::Placement {
        self.1.clone()
    }
}

impl PlacedBitTensor {
    pub fn fill(shape: &Shape, el: u8, plc: &HostPlacement) -> Self {
        PlacedBitTensor(BitTensor::fill(shape, el), plc.clone())
    }
}

impl BitXor for PlacedBitTensor {
    type Output = PlacedBitTensor;
    fn bitxor(self, other: Self) -> Self::Output {
        assert_eq!(self.placement(), other.placement());
        PlacedBitTensor(BitTensor(self.0 .0 ^ other.0 .0), self.1)
    }
}

impl BitAnd for PlacedBitTensor {
    type Output = PlacedBitTensor;
    fn bitand(self, other: Self) -> Self::Output {
        assert_eq!(self.placement(), other.placement());
        PlacedBitTensor(BitTensor(self.0 .0 & other.0 .0), self.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_sample() {
        let shape = Shape(vec![5]);
        let seed = Seed([0u8; 16]);
        let r = BitTensor::sample_uniform(&shape, &seed);
        assert_eq!(r, BitTensor::from(vec![0, 1, 1, 0, 0,]));
    }

    #[test]
    fn bit_fill() {
        let shape = Shape(vec![2]);
        let r = BitTensor::fill(&shape, 1);
        assert_eq!(r, BitTensor::from(vec![1, 1]))
    }

    #[test]
    fn bit_ops() {
        let shape = Shape(vec![5]);
        let alice = HostPlacement {
            owner: "alice".into(),
        };

        // test xor
        assert_eq!(
            PlacedBitTensor::fill(&shape, 0, &alice) ^ PlacedBitTensor::fill(&shape, 1, &alice),
            PlacedBitTensor::fill(&shape, 1, &alice)
        );
        assert_eq!(
            PlacedBitTensor::fill(&shape, 1, &alice) ^ PlacedBitTensor::fill(&shape, 0, &alice),
            PlacedBitTensor::fill(&shape, 1, &alice)
        );
        assert_eq!(
            PlacedBitTensor::fill(&shape, 1, &alice) ^ PlacedBitTensor::fill(&shape, 1, &alice),
            PlacedBitTensor::fill(&shape, 0, &alice)
        );
        assert_eq!(
            PlacedBitTensor::fill(&shape, 0, &alice) ^ PlacedBitTensor::fill(&shape, 0, &alice),
            PlacedBitTensor::fill(&shape, 0, &alice)
        );

        // test and
        assert_eq!(
            PlacedBitTensor::fill(&shape, 0, &alice) & PlacedBitTensor::fill(&shape, 1, &alice),
            PlacedBitTensor::fill(&shape, 0, &alice)
        );
        assert_eq!(
            PlacedBitTensor::fill(&shape, 1, &alice) & PlacedBitTensor::fill(&shape, 0, &alice),
            PlacedBitTensor::fill(&shape, 0, &alice)
        );
        assert_eq!(
            PlacedBitTensor::fill(&shape, 1, &alice) & PlacedBitTensor::fill(&shape, 1, &alice),
            PlacedBitTensor::fill(&shape, 1, &alice)
        );
        assert_eq!(
            PlacedBitTensor::fill(&shape, 0, &alice) & PlacedBitTensor::fill(&shape, 0, &alice),
            PlacedBitTensor::fill(&shape, 0, &alice)
        );
    }
}
