use crate::computation::{
    BitAndOp, BitFillOp, BitSampleOp, BitXorOp, Constant, HostPlacement, Placed, ShapeOp,
};
use crate::error::Result;
use crate::kernels::{
    PlacementAdd, PlacementAnd, PlacementFill, PlacementMul, PlacementPlace,
    PlacementSampleUniform, PlacementSub, PlacementXor, RuntimeSession, Session, SyncSession,
    Tensor,
};
use crate::prim::{RawSeed, Seed};
use crate::prng::AesRng;
use crate::standard::{RawShape, Shape};
use ndarray::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{BitAnd, BitXor};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BitTensor(pub ArrayD<u8>, HostPlacement);

impl<S: Session> Tensor<S> for BitTensor {
    type Scalar = u8;
}

impl Placed for BitTensor {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl PlacementPlace<SyncSession, BitTensor> for HostPlacement {
    fn place(&self, _sess: &SyncSession, x: BitTensor) -> BitTensor {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                BitTensor(x.0, self.clone())
            }
        }
    }
}

impl ShapeOp {
    pub(crate) fn bit_kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: BitTensor,
    ) -> Shape {
        let raw_shape = RawShape(x.0.shape().into());
        Shape(raw_shape, plc.clone().into())
    }
}

modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (Shape) -> BitTensor, BitFillOp);

kernel! {
    BitFillOp,
    [
        (HostPlacement, (Shape) -> BitTensor => attributes[value: Bit] Self::kernel),
    ]
}

impl BitFillOp {
    fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        value: u8,
        shape: Shape,
    ) -> BitTensor {
        assert!(value == 0 || value == 1);
        let raw_shape = shape.0 .0;
        let raw_tensor = ArrayD::from_elem(raw_shape.as_ref(), value as u8);
        BitTensor(raw_tensor, plc.clone())
    }
}

modelled!(PlacementSampleUniform::sample_uniform, HostPlacement, (Seed, Shape) -> BitTensor, BitSampleOp);

kernel! {
    BitSampleOp,
    [
        (HostPlacement, (Seed, Shape) -> BitTensor => Self::kernel),
    ]
}

impl BitSampleOp {
    fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        seed: Seed,
        shape: Shape,
    ) -> BitTensor {
        let mut rng = AesRng::from_seed(seed.0 .0);
        let size = shape.0 .0.iter().product();
        let values: Vec<_> = (0..size).map(|_| rng.get_bit()).collect();
        let ix = IxDyn(shape.0 .0.as_ref());
        BitTensor(Array::from_shape_vec(ix, values).unwrap(), plc.clone())
    }
}

modelled!(PlacementXor::xor, HostPlacement, (BitTensor, BitTensor) -> BitTensor, BitXorOp);
modelled_alias!(PlacementAdd::add, HostPlacement, (BitTensor, BitTensor) -> BitTensor => PlacementXor::xor); // add = xor in Z2
modelled_alias!(PlacementSub::sub, HostPlacement, (BitTensor, BitTensor) -> BitTensor => PlacementXor::xor); // sub = xor in Z2

kernel! {
    BitXorOp,
    [
        (HostPlacement, (BitTensor, BitTensor) -> BitTensor => Self::kernel),
    ]
}

impl BitXorOp {
    fn kernel<S: RuntimeSession>(
        _sess: &S,
        plc: &HostPlacement,
        x: BitTensor,
        y: BitTensor,
    ) -> BitTensor {
        BitTensor(x.0 ^ y.0, plc.clone())
    }
}

modelled!(PlacementAnd::and, HostPlacement, (BitTensor, BitTensor) -> BitTensor, BitAndOp);
modelled_alias!(PlacementMul::mul, HostPlacement, (BitTensor, BitTensor) -> BitTensor => PlacementAnd::and); // mul = and in Z2

kernel! {
    BitAndOp,
    [
        (HostPlacement, (BitTensor, BitTensor) -> BitTensor => Self::kernel),
    ]
}

impl BitAndOp {
    fn kernel<S: RuntimeSession>(
        _sess: &S,
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
