//! Placements backed by additive secret sharing
use crate::computation::{AdditivePlacement, HostPlacement, Placed};
use crate::error::Result;
use crate::host::{HostBitTensor, HostRing128Tensor, HostRing64Tensor, HostShape};
use crate::kernels::{PlacementAdd, PlacementPlace, PlacementShl, Session};
use serde::{Deserialize, Serialize};

mod convert;
mod dabit;
mod ops;
mod trunc;
pub use convert::*;
pub use dabit::*;
pub use ops::*;
pub use trunc::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdtTensor<HostTensorT> {
    pub shares: [HostTensorT; 2],
}

moose_type!(AdditiveRing64Tensor = AdtTensor<HostRing64Tensor>);
moose_type!(AdditiveRing128Tensor = AdtTensor<HostRing128Tensor>);
moose_type!(AdditiveBitTensor = AdtTensor<HostBitTensor>);

impl<HostT> Placed for AdtTensor<HostT>
where
    HostT: Placed<Placement = HostPlacement>,
{
    type Placement = AdditivePlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let AdtTensor { shares: [x0, x1] } = self;

        let owner0 = x0.placement()?.owner;
        let owner1 = x1.placement()?.owner;

        let owners = [owner0, owner1];
        Ok(AdditivePlacement { owners })
    }
}

impl<S: Session, HostT> PlacementPlace<S, AdtTensor<HostT>> for AdditivePlacement
where
    AdtTensor<HostT>: Placed<Placement = AdditivePlacement>,
    HostPlacement: PlacementPlace<S, HostT>,
{
    fn place(&self, sess: &S, x: AdtTensor<HostT>) -> AdtTensor<HostT> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                let AdtTensor { shares: [x0, x1] } = x;
                let (player0, player1) = self.host_placements();
                AdtTensor {
                    shares: [player0.place(sess, x0), player1.place(sess, x1)],
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdtShape<HostShapeT> {
    pub shapes: [HostShapeT; 2],
}

moose_type!(AdditiveShape = AdtShape<HostShape>);

impl<HostT> Placed for AdtShape<HostT>
where
    HostT: Placed<Placement = HostPlacement>,
{
    type Placement = AdditivePlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let AdtShape { shapes: [s0, s1] } = self;

        let owner0 = s0.placement()?.owner;
        let owner1 = s1.placement()?.owner;

        let owners = [owner0, owner1];
        Ok(AdditivePlacement { owners })
    }
}

pub trait BitCompose<S: Session, R> {
    fn bit_compose(&self, sess: &S, bits: &[R]) -> R;
}

impl<S: Session, R> BitCompose<S, R> for HostPlacement
where
    R: Clone,
    HostPlacement: PlacementShl<S, R, R>,
    HostPlacement: TreeReduce<S, R>,
{
    fn bit_compose(&self, sess: &S, bits: &[R]) -> R {
        let shifted_bits: Vec<_> = (0..bits.len())
            .map(|i| self.shl(sess, i, &bits[i]))
            .collect();
        self.tree_reduce(sess, &shifted_bits)
    }
}

pub trait TreeReduce<S: Session, R> {
    fn tree_reduce(&self, sess: &S, sequence: &[R]) -> R;
}

impl<S: Session, R> TreeReduce<S, R> for HostPlacement
where
    R: Clone,
    HostPlacement: PlacementAdd<S, R, R, R>,
{
    fn tree_reduce(&self, sess: &S, sequence: &[R]) -> R {
        let n = sequence.len();
        if n == 1 {
            sequence[0].clone()
        } else {
            let mut reduced: Vec<_> = (0..n / 2)
                .map(|i| {
                    let x0: &R = &sequence[2 * i];
                    let x1: &R = &sequence[2 * i + 1];
                    self.add(sess, x0, x1)
                })
                .collect();
            if n % 2 == 1 {
                reduced.push(sequence[n - 1].clone());
            }
            self.tree_reduce(sess, &reduced)
        }
    }
}
