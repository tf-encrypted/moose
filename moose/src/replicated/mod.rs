//! Placements backed by three-party replicated secret sharing
use crate::additive::{AdditivePlacement, AdtTensor, DaBitProvider, TruncPrProvider};
use crate::computation::*;
use crate::error::{Error, Result};
use crate::fixedpoint::FixedpointTensor;
use crate::host::{
    AbstractHostAesKey, HostBitArray, HostFixedTensor, HostPlacement, PrfKey, Seed, SliceInfo,
    SyncKey,
};
use crate::kernels::*;
use crate::mirrored::{Mir3Tensor, MirFixedTensor, Mirrored3Placement};
use crate::symbolic::Symbolic;
use crate::types::*;
use crate::{BitArray, Const, Ring};
use macros::with_context;
use serde::{Deserialize, Serialize};
use std::convert::{TryFrom, TryInto};
use std::marker::PhantomData;

mod aes;
mod control_flow;
mod division;
mod exp;
mod input;
mod log;
mod softmax;
pub use self::aes::AbstractReplicatedAesKey;

/// Placement type for three-party replicated secret sharing
#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct ReplicatedPlacement {
    pub owners: [Role; 3],
}

impl ReplicatedPlacement {
    pub fn host_placements(&self) -> (HostPlacement, HostPlacement, HostPlacement) {
        let player0 = HostPlacement {
            owner: self.owners[0].clone(),
        };
        let player1 = HostPlacement {
            owner: self.owners[1].clone(),
        };
        let player2 = HostPlacement {
            owner: self.owners[2].clone(),
        };
        (player0, player1, player2)
    }
}

pub(crate) trait ShapeFill<S, TenT> {
    type Result;

    fn shape_fill<C: Into<Constant>>(
        &self,
        sess: &S,
        fill_value: C,
        shape_from: &TenT,
    ) -> Self::Result;
}

impl<S: Session, TenT> ShapeFill<S, TenT> for ReplicatedPlacement
where
    TenT: MirroredCounterpart,
    Self: PlacementShape<S, TenT, m!(ReplicatedShape)>,
    Mirrored3Placement: PlacementFill<S, m!(ReplicatedShape), TenT::MirroredType>,
    ReplicatedShape: KnownType<S>,
{
    type Result = TenT::MirroredType;

    fn shape_fill<C: Into<Constant>>(
        &self,
        sess: &S,
        fill_value: C,
        shape_from: &TenT,
    ) -> Self::Result {
        let shape = self.shape(sess, shape_from);

        let (player0, player1, player2) = self.host_placements();

        let mir = Mirrored3Placement {
            owners: [player0.owner, player1.owner, player2.owner],
        };

        mir.fill(sess, fill_value.into(), &shape)
    }
}

/// Secret tensor used by replicated placements
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepTensor<HostRingT> {
    pub shares: [[HostRingT; 2]; 3],
}

impl<HostRingT: Ring> Ring for RepTensor<HostRingT> {
    type BitLength = HostRingT::BitLength;
}

pub(crate) trait Underlying {
    type TensorType;
}

pub(crate) trait MirroredCounterpart {
    type MirroredType;
}

impl<HostRingT> Underlying for RepTensor<HostRingT> {
    type TensorType = HostRingT;
}

impl<HostRingT> MirroredCounterpart for RepTensor<HostRingT> {
    type MirroredType = Mir3Tensor<HostRingT>;
}

impl<HostRingT> Underlying for Mir3Tensor<HostRingT> {
    type TensorType = HostRingT;
}

impl<T: Placed + Underlying> Underlying for Symbolic<T> {
    type TensorType = <T as Underlying>::TensorType;
}

impl<T: Placed + MirroredCounterpart> MirroredCounterpart for Symbolic<T>
where
    <T as MirroredCounterpart>::MirroredType: Placed,
{
    type MirroredType = Symbolic<<T as MirroredCounterpart>::MirroredType>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepBitArray<RepBitTensorT, N>(RepBitTensorT, PhantomData<N>);

impl<RepBitTensorT, N: Const> BitArray for RepBitArray<RepBitTensorT, N> {
    type Len = N;
}

impl<RepBitTensorT: Placed, N: Const> BitArray for Symbolic<RepBitArray<RepBitTensorT, N>> {
    type Len = N;
}

// TODO implement using moose_type macro
impl<RepBitTensorT: Placed, N> Placed for RepBitArray<RepBitTensorT, N> {
    type Placement = RepBitTensorT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.0.placement()
    }
}

impl<N> PartiallySymbolicType for RepBitArray<ReplicatedBitTensor, N> {
    type Type = RepBitArray<<ReplicatedBitTensor as SymbolicType>::Type, N>;
}

impl<N> CanonicalType for RepBitArray<ReplicatedBitTensor, N> {
    type Type = Self;
}

impl<N> CanonicalType for RepBitArray<<ReplicatedBitTensor as SymbolicType>::Type, N> {
    type Type = RepBitArray<ReplicatedBitTensor, N>;
}

impl<N> CanonicalType for Symbolic<RepBitArray<<ReplicatedBitTensor as SymbolicType>::Type, N>> {
    type Type = RepBitArray<ReplicatedBitTensor, N>;
}

impl<RepBitT: Placed, N> From<RepBitArray<RepBitT, N>> for Symbolic<RepBitArray<RepBitT, N>>
where
    RepBitT: Placed<Placement = ReplicatedPlacement>,
{
    fn from(x: RepBitArray<RepBitT, N>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<RepBitT, N> TryFrom<Symbolic<RepBitArray<RepBitT, N>>> for RepBitArray<RepBitT, N>
where
    RepBitT: Placed<Placement = ReplicatedPlacement>,
{
    type Error = Error;
    fn try_from(v: Symbolic<RepBitArray<RepBitT, N>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected(None)), // TODO err message
        }
    }
}

impl<HostTenT> Placed for RepTensor<HostTenT>
where
    HostTenT: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = self;

        let owner0 = x00.placement()?.owner;
        let owner1 = x11.placement()?.owner;
        let owner2 = x22.placement()?.owner;

        if x10.placement()?.owner == owner0
            && x21.placement()?.owner == owner1
            && x02.placement()?.owner == owner2
        {
            let owners = [owner0, owner1, owner2];
            Ok(ReplicatedPlacement { owners })
        } else {
            Err(Error::MalformedPlacement)
        }
    }
}

impl<S: Session, R> PlacementPlace<S, RepTensor<R>> for ReplicatedPlacement
where
    RepTen<R>: Placed<Placement = ReplicatedPlacement>,
    HostPlacement: PlacementPlace<S, R>,
{
    fn place(&self, sess: &S, x: RepTen<R>) -> RepTen<R> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                let RepTen {
                    shares: [[x00, x10], [x11, x21], [x22, x02]],
                } = x;

                let (player0, player1, player2) = self.host_placements();
                RepTen {
                    shares: [
                        [player0.place(sess, x00), player0.place(sess, x10)],
                        [player1.place(sess, x11), player1.place(sess, x21)],
                        [player2.place(sess, x22), player2.place(sess, x02)],
                    ],
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RepFixedTensor<RepRingT> {
    pub tensor: RepRingT,
    pub fractional_precision: u32,
    pub integral_precision: u32,
}

impl<RepRingT: Underlying> Underlying for RepFixedTensor<RepRingT> {
    type TensorType = RepRingT::TensorType;
}

impl<RepRingT: MirroredCounterpart> MirroredCounterpart for RepFixedTensor<RepRingT> {
    type MirroredType = MirFixedTensor<RepRingT::MirroredType>;
}

impl<RepRingT: Placed> Placed for RepFixedTensor<RepRingT> {
    type Placement = RepRingT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.tensor.placement()
    }
}

impl<S: Session, RepRingT> PlacementPlace<S, RepFixedTensor<RepRingT>> for ReplicatedPlacement
where
    RepFixedTensor<RepRingT>: Placed<Placement = ReplicatedPlacement>,
    ReplicatedPlacement: PlacementPlace<S, RepRingT>,
{
    fn place(&self, sess: &S, x: RepFixedTensor<RepRingT>) -> RepFixedTensor<RepRingT> {
        match x.placement() {
            Ok(place) if self == &place => x,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                RepFixedTensor {
                    tensor: self.place(sess, x.tensor),
                    integral_precision: x.integral_precision,
                    fractional_precision: x.fractional_precision,
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepSetup<K> {
    pub keys: [[K; 2]; 3],
}

impl<K> Placed for RepSetup<K>
where
    K: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let RepSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = self;

        let owner0 = k00.placement()?.owner;
        let owner1 = k11.placement()?.owner;
        let owner2 = k22.placement()?.owner;

        if k10.placement()?.owner == owner0
            && k21.placement()?.owner == owner1
            && k02.placement()?.owner == owner2
        {
            let owners = [owner0, owner1, owner2];
            Ok(ReplicatedPlacement { owners })
        } else {
            Err(Error::MalformedPlacement)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepShape<S> {
    pub shapes: [S; 3],
}

impl<KeyT> Placed for RepShape<KeyT>
where
    KeyT: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let RepShape {
            shapes: [s0, s1, s2],
        } = self;

        let owner0 = s0.placement()?.owner;
        let owner1 = s1.placement()?.owner;
        let owner2 = s2.placement()?.owner;

        let owners = [owner0, owner1, owner2];
        Ok(ReplicatedPlacement { owners })
    }
}

// Type aliases to shorten out impl in replicated protocols
pub(crate) type RepTen<T> = RepTensor<T>;
pub(crate) type AdtTen<T> = AdtTensor<T>;
pub(crate) type MirTen<T> = Mir3Tensor<T>;

impl RepSetupOp {
    pub(crate) fn kernel<S: Session, K: Clone>(
        sess: &S,
        rep: &ReplicatedPlacement,
    ) -> Result<RepSetup<K>>
    where
        HostPlacement: PlacementKeyGen<S, K>,
        HostPlacement: PlacementPlace<S, K>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let k0 = player0.gen_key(sess);
        let k1 = player1.gen_key(sess);
        let k2 = player2.gen_key(sess);

        Ok(RepSetup {
            keys: [
                [
                    player0.place(sess, k0.clone()),
                    player0.place(sess, k1.clone()),
                ],
                [player1.place(sess, k1), player1.place(sess, k2.clone())],
                [player2.place(sess, k2), player2.place(sess, k0)],
            ],
        })
    }
}

impl RepShareOp {
    pub(crate) fn aeskey_kernel<S: Session, HostBitArrayT, RepBitArrayT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        key: AbstractHostAesKey<HostBitArrayT>,
    ) -> Result<AbstractReplicatedAesKey<RepBitArrayT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostBitArrayT, RepBitArrayT>,
    {
        let bit_array = plc.share(sess, &key.0);
        Ok(AbstractReplicatedAesKey(bit_array))
    }

    pub(crate) fn fixed_kernel<S: Session, HostRingT, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: HostFixedTensor<HostRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostRingT, RepRingT>,
    {
        Ok(RepFixedTensor {
            tensor: plc.share(sess, &x.tensor),
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    pub(crate) fn array_kernel<S: Session, HostBitTensorT, RepBitTensorT, N: Const>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: HostBitArray<HostBitTensorT, N>,
    ) -> Result<RepBitArray<RepBitTensorT, N>>
    where
        ReplicatedPlacement: PlacementShare<S, HostBitTensorT, RepBitTensorT>,
    {
        let shared_tensor = plc.share(sess, &x.0);
        Ok(RepBitArray(shared_tensor, x.1))
    }

    pub(crate) fn ring_kernel<S: Session, ShapeT, SeedT, KeyT, RingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RingT,
    ) -> Result<RepTen<RingT>>
    where
        <S as Session>::ReplicatedSetup: Clone,
        <S as Session>::ReplicatedSetup: TryInto<RepSetup<KeyT>>,
        RingT: Clone + Placed<Placement = HostPlacement>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        HostPlacement: PlacementSampleUniformSeeded<S, ShapeT, SeedT, RingT>,
        HostPlacement: PlacementZeros<S, ShapeT, RingT>,
        HostPlacement: PlacementDeriveSeed<S, KeyT, SeedT>,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementSub<S, RingT, RingT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<RingT>>,
    {
        let x_player = x.placement()?;

        let setup = match (*sess.replicated_setup(plc)).clone().try_into() {
            Ok(setup) => setup,
            _ => todo!("not sure what to do with the symbolic setup yet"), // TODO: perhaps a custom kernel could instead output a symbolic op
        };
        let RepSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = &setup;

        let (player0, player1, player2) = plc.host_placements();

        let shares = match () {
            _ if x_player == player0 => {
                let sync_key = SyncKey::random();
                let shape = x_player.shape(sess, &x);

                let seed0 = player0.derive_seed(sess, sync_key.clone(), k00);
                let x00 = x_player.sample_uniform_seeded(sess, &shape, &seed0);
                let x10 = with_context!(x_player, sess, x - x00);

                let seed2 = player2.derive_seed(sess, sync_key, k02);
                let x22 = player2.zeros(sess, &shape);
                let x02 = player2.sample_uniform_seeded(sess, &shape, &seed2);

                let x11 = x10.clone();
                let x21 = player1.zeros(sess, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ if x_player == player1 => {
                let sync_key = SyncKey::random();
                let shape = x_player.shape(sess, &x);

                let seed1 = player1.derive_seed(sess, sync_key.clone(), k11);
                let x11 = x_player.sample_uniform_seeded(sess, &shape, &seed1);
                let x21 = with_context!(x_player, sess, x - x11);

                let seed0 = player0.derive_seed(sess, sync_key, k10);
                let x00 = player0.zeros(sess, &shape);
                let x10 = player0.sample_uniform_seeded(sess, &shape, &seed0);

                let x22 = x21.clone();
                let x02 = player2.zeros(sess, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ if x_player == player2 => {
                let sync_key = SyncKey::random();
                let shape = x_player.shape(sess, &x);

                let seed2 = player2.derive_seed(sess, sync_key.clone(), k22);
                let x22 = player2.sample_uniform_seeded(sess, &shape, &seed2);
                let x02 = with_context!(x_player, sess, x - x22);

                let seed1 = player1.derive_seed(sess, sync_key, k21);
                let x11 = player1.zeros(sess, &shape);
                let x21 = player1.sample_uniform_seeded(sess, &shape, &seed1);

                let x00 = x02.clone();
                let x10 = player0.zeros(sess, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ => {
                // in this case, where x_owner is _not_ among the replicated players,
                // we cannot use the zeros optimization trick but we can still make sure
                // that seeds are used as much as possible instead of dense random tensors;
                // however, we must make sure keys are not revealed to x_owner and only seeds

                let sync_key0 = SyncKey::random();
                let sync_key1 = SyncKey::random();
                let shape = x_player.shape(sess, &x);

                let seed00 = player0.derive_seed(sess, sync_key0.clone(), k00);
                let seed02 = player2.derive_seed(sess, sync_key0, k02);

                let seed11 = player1.derive_seed(sess, sync_key1.clone(), k11);
                let seed10 = player0.derive_seed(sess, sync_key1, k10);

                let x0 = x_player.sample_uniform_seeded(sess, &shape, &seed00);
                let x1 = x_player.sample_uniform_seeded(sess, &shape, &seed11);
                let x2 = with_context!(x_player, sess, x - x0 - x1);

                let x00 = player0.sample_uniform_seeded(sess, &shape, &seed00);
                let x10 = player0.sample_uniform_seeded(sess, &shape, &seed10);

                let x11 = player1.sample_uniform_seeded(sess, &shape, &seed11);
                let x21 = x2.clone();

                let x22 = x2;
                let x02 = player2.sample_uniform_seeded(sess, &shape, &seed02);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
        };

        Ok(plc.place(sess, RepTen { shares }))
    }
}

impl RepRevealOp {
    pub(crate) fn aeskey_kernel<S: Session, RepBitArrayT, HostBitArrayT>(
        sess: &S,
        receiver: &HostPlacement,
        key: AbstractReplicatedAesKey<RepBitArrayT>,
    ) -> Result<AbstractHostAesKey<HostBitArrayT>>
    where
        HostPlacement: PlacementReveal<S, RepBitArrayT, HostBitArrayT>,
    {
        let bit_array = receiver.reveal(sess, &key.0);
        Ok(AbstractHostAesKey(bit_array))
    }

    pub(crate) fn fixed_kernel<S: Session, RepRingT, HostRingT>(
        sess: &S,
        receiver: &HostPlacement,
        xe: RepFixedTensor<RepRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementReveal<S, RepRingT, HostRingT>,
    {
        let x = receiver.reveal(sess, &xe.tensor);
        Ok(HostFixedTensor {
            tensor: x,
            fractional_precision: xe.fractional_precision,
            integral_precision: xe.integral_precision,
        })
    }

    pub(crate) fn bit_array_kernel<S: Session, RepBitT, HostBitT, N>(
        sess: &S,
        receiver: &HostPlacement,
        xe: RepBitArray<RepBitT, N>,
    ) -> Result<HostBitArray<HostBitT, N>>
    where
        HostPlacement: PlacementReveal<S, RepBitT, HostBitT>,
    {
        let x = receiver.reveal(sess, &xe.0);
        Ok(HostBitArray(x, PhantomData))
    }

    pub(crate) fn ring_kernel<S: Session, R: Clone>(
        sess: &S,
        receiver: &HostPlacement,
        xe: RepTen<R>,
    ) -> Result<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, R, R, R>,
    {
        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &xe;

        let (player0, player1, player2) = &xe.placement()?.host_placements();

        let res = match () {
            _ if receiver == player0 => {
                // make sure to use both shares on player0
                with_context!(receiver, sess, x00 + x10 + x21)
            }
            _ if receiver == player1 => {
                // make sure to use both shares on player1
                with_context!(receiver, sess, x02 + x11 + x21)
            }
            _ if receiver == player2 => {
                // make sure to use both shares on player2
                with_context!(receiver, sess, x02 + x10 + x22)
            }
            _ => {
                with_context!(receiver, sess, x00 + x10 + x21)
            }
        };
        Ok(res)
    }
}

impl IdentityOp {
    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementIdentity<S, RepRingT, RepRingT>,
    {
        let tensor = rep.identity(sess, &x.tensor);
        Ok(RepFixedTensor {
            tensor,
            integral_precision: x.integral_precision,
            fractional_precision: x.fractional_precision,
        })
    }

    pub(crate) fn rep_inner_kernel<S: Session, HostT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<HostT>,
    ) -> Result<RepTen<HostT>>
    where
        HostPlacement: PlacementIdentity<S, HostT, HostT>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;
        let y00 = player0.identity(sess, x00);
        let y10 = player0.identity(sess, x10);
        let y11 = player1.identity(sess, x11);
        let y21 = player1.identity(sess, x21);
        let y22 = player2.identity(sess, x22);
        let y02 = player2.identity(sess, x02);
        Ok(RepTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        })
    }
}

impl RepAndOp {
    pub(crate) fn bit_kernel<S: Session, RepT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepT,
        y: RepT,
    ) -> Result<RepT>
    where
        ReplicatedPlacement: PlacementMul<S, RepT, RepT, RepT>,
    {
        // and = mul in Z2
        Ok(rep.mul(sess, &x, &y))
    }
}

impl RepXorOp {
    pub(crate) fn bit_kernel<S: Session, X1, X2, Y>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: X1,
        y: X2,
    ) -> Result<Y>
    where
        ReplicatedPlacement: PlacementAdd<S, X1, X2, Y>,
    {
        // add = xor in Z2
        Ok(rep.add(sess, &x, &y))
    }
}

impl RepNegOp {
    pub(crate) fn rep_bit_kernel<S: Session, HostBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<HostBitT>,
    ) -> Result<RepTen<HostBitT>>
    where
        HostPlacement: PlacementNeg<S, HostBitT, HostBitT>,
    {
        let (player0, _player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        // TODO(Morten)
        // we could choose share to change at random
        // to more fairly distribute compute load
        let y00 = player0.neg(sess, &x00);
        let y10 = x10;
        let y11 = x11;
        let y21 = x21;
        let y22 = x22;
        let y02 = player2.neg(sess, &x02);

        Ok(RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        })
    }

    pub(crate) fn rep_rep_kernel<S: Session, HostRepT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<HostRepT>,
    ) -> Result<RepTen<HostRepT>>
    where
        HostPlacement: PlacementNeg<S, HostRepT, HostRepT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let y00 = player0.neg(sess, &x00);
        let y10 = player0.neg(sess, &x10);
        let y11 = player1.neg(sess, &x11);
        let y21 = player1.neg(sess, &x21);
        let y22 = player2.neg(sess, &x22);
        let y02 = player2.neg(sess, &x02);

        Ok(RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        })
    }
}

impl RepAddOp {
    pub(crate) fn rep_rep_kernel<S: Session, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<HostRingT>,
        y: RepTen<HostRingT>,
    ) -> Result<RepTen<HostRingT>>
    where
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, sess, x00 + y00);
        let z10 = with_context!(player0, sess, x10 + y10);

        let z11 = with_context!(player1, sess, x11 + y11);
        let z21 = with_context!(player1, sess, x21 + y21);

        let z22 = with_context!(player2, sess, x22 + y22);
        let z02 = with_context!(player2, sess, x02 + y02);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn mir_rep_kernel<S: Session, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: MirTen<HostRingT>,
        y: RepTen<HostRingT>,
    ) -> Result<RepTen<HostRingT>>
    where
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, _player1, player2) = rep.host_placements();

        let MirTen {
            values: [x0, _x1, x2],
        } = x;

        let RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = y;

        let z00 = with_context!(player0, sess, x0 + y00);
        let z02 = with_context!(player2, sess, x2 + y02);

        Ok(RepTen {
            shares: [[z00, y10], [y11, y21], [y22, z02]],
        })
    }

    pub(crate) fn rep_mir_kernel<S: Session, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<HostRingT>,
        y: MirTen<HostRingT>,
    ) -> Result<RepTen<HostRingT>>
    where
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, _player1, player2) = rep.host_placements();

        let MirTen {
            values: [y0, _y1, y2],
        } = y;

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let z00 = with_context!(player0, sess, x00 + y0);
        let z02 = with_context!(player2, sess, x02 + y2);

        Ok(RepTen {
            shares: [[z00, x10], [x11, x21], [x22, z02]],
        })
    }
}

impl RepSubOp {
    pub(crate) fn rep_rep_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<R>,
        y: RepTen<R>,
    ) -> Result<RepTen<R>>
    where
        HostPlacement: PlacementSub<S, R, R, R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, sess, x00 - y00);
        let z10 = with_context!(player0, sess, x10 - y10);

        let z11 = with_context!(player1, sess, x11 - y11);
        let z21 = with_context!(player1, sess, x21 - y21);

        let z22 = with_context!(player2, sess, x22 - y22);
        let z02 = with_context!(player2, sess, x02 - y02);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn mir_rep_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: MirTen<R>,
        y: RepTen<R>,
    ) -> Result<RepTen<R>>
    where
        HostPlacement: PlacementSub<S, R, R, R>,
        HostPlacement: PlacementNeg<S, R, R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let MirTen {
            values: [x0, _x1, x2],
        } = &x;

        let RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, sess, x0 - y00);
        let z10 = player0.neg(sess, y10);
        let z11 = player1.neg(sess, y11);
        let z21 = player1.neg(sess, y21);
        let z22 = player2.neg(sess, y22);
        let z02 = with_context!(player2, sess, x2 - y02);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn rep_mir_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<R>,
        y: MirTen<R>,
    ) -> Result<RepTen<R>>
    where
        HostPlacement: PlacementSub<S, R, R, R>,
    {
        let (player0, _player1, player2) = rep.host_placements();

        let MirTen {
            values: [y0, _y1, y2],
        } = y;

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let z00 = with_context!(player0, sess, x00 - y0);
        let z02 = with_context!(player2, sess, x02 - y2);

        Ok(RepTen {
            shares: [[z00, x10], [x11, x21], [x22, z02]],
        })
    }
}

impl RepMulOp {
    pub(crate) fn rep_rep_kernel<S: Session, RingT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<RingT>,
        y: RepTen<RingT>,
    ) -> Result<RepTen<RingT>>
    where
        RingT: Clone,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        ReplicatedPlacement: ZeroShareGen<S, ShapeT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<RingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let v0 = with_context!(player0, sess, { x00 * y00 + x00 * y10 + x10 * y00 });
        let v1 = with_context!(player1, sess, { x11 * y11 + x11 * y21 + x21 * y11 });
        let v2 = with_context!(player2, sess, { x22 * y22 + x22 * y02 + x02 * y22 });

        let s0 = player0.shape(sess, &v0);
        let s1 = player1.shape(sess, &v1);
        let s2 = player2.shape(sess, &v2);
        let zero_shape = RepShape {
            shapes: [s0, s1, s2],
        };

        let AbstractReplicatedZeroShare {
            alphas: [a0, a1, a2],
        } = rep.gen_zero_share(sess, &zero_shape);

        let z0 = with_context!(player0, sess, { v0 + a0 });
        let z1 = with_context!(player1, sess, { v1 + a1 });
        let z2 = with_context!(player2, sess, { v2 + a2 });

        Ok(rep.place(
            sess,
            RepTen {
                shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
            },
        ))
    }

    pub(crate) fn mir_rep_kernel<S: Session, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: MirTen<RingT>,
        y: RepTen<RingT>,
    ) -> Result<RepTen<RingT>>
    where
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let MirTen {
            values: [x0, x1, x2],
        } = &x;

        let z00 = with_context!(player0, sess, x0 * y00);
        let z10 = with_context!(player0, sess, x0 * y10);

        let z11 = with_context!(player1, sess, x1 * y11);
        let z21 = with_context!(player1, sess, x1 * y21);

        let z22 = with_context!(player2, sess, x2 * y22);
        let z02 = with_context!(player2, sess, x2 * y02);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn rep_mir_kernel<S: Session, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<RingT>,
        y: MirTen<RingT>,
    ) -> Result<RepTen<RingT>>
    where
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let MirTen {
            values: [y0, y1, y2],
        } = &y;

        let z00 = with_context!(player0, sess, x00 * y0);
        let z10 = with_context!(player0, sess, x10 * y0);

        let z11 = with_context!(player1, sess, x11 * y1);
        let z21 = with_context!(player1, sess, x21 * y1);

        let z22 = with_context!(player2, sess, x22 * y2);
        let z02 = with_context!(player2, sess, x02 * y2);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl RepDotOp {
    pub(crate) fn rep_rep_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<RingT>,
        y: RepTen<RingT>,
    ) -> Result<RepTen<RingT>>
    where
        RingT: Clone,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementDot<S, RingT, RingT, RingT>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        ReplicatedPlacement: ZeroShareGen<S, ShapeT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<RingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let v0 = with_context!(player0, sess, {
            dot(x00, y00) + dot(x00, y10) + dot(x10, y00)
        });
        let v1 = with_context!(player1, sess, {
            dot(x11, y11) + dot(x11, y21) + dot(x21, y11)
        });
        let v2 = with_context!(player2, sess, {
            dot(x22, y22) + dot(x22, y02) + dot(x02, y22)
        });

        let s0 = player0.shape(sess, &v0);
        let s1 = player1.shape(sess, &v1);
        let s2 = player2.shape(sess, &v2);
        let zero_shape = RepShape {
            shapes: [s0, s1, s2],
        };

        let AbstractReplicatedZeroShare {
            alphas: [a0, a1, a2],
        } = rep.gen_zero_share(sess, &zero_shape);

        let z0 = with_context!(player0, sess, { v0 + a0 });
        let z1 = with_context!(player1, sess, { v1 + a1 });
        let z2 = with_context!(player2, sess, { v2 + a2 });

        Ok(rep.place(
            sess,
            RepTen {
                shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
            },
        ))
    }
}

impl RepFixedpointMeanOp {
    pub(crate) fn kernel<S: Session, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: RepTen<HostRingT>,
    ) -> Result<RepTen<HostRingT>>
    where
        HostPlacement: PlacementMeanAsFixedpoint<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x00);
        let z10 = player0.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x10);
        let z11 = player1.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x11);
        let z21 = player1.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x21);
        let z22 = player2.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x22);
        let z02 = player2.mean_as_fixedpoint(sess, axis, scaling_base, scaling_exp, x02);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl AddNOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        xs: &[RepTen<HostRingT>],
    ) -> Result<RepTen<HostRingT>>
    where
        HostPlacement: PlacementAddN<S, HostRingT, HostRingT>,
        HostRingT: Clone,
    {
        let (player0, player1, player2) = rep.host_placements();

        let mut z00s: Vec<HostRingT> = Vec::new();
        let mut z10s: Vec<HostRingT> = Vec::new();
        let mut z11s: Vec<HostRingT> = Vec::new();
        let mut z21s: Vec<HostRingT> = Vec::new();
        let mut z22s: Vec<HostRingT> = Vec::new();
        let mut z02s: Vec<HostRingT> = Vec::new();
        for x in xs.iter() {
            let RepTen {
                shares: [[x00, x10], [x11, x21], [x22, x02]],
            } = &x;

            z00s.push(x00.clone());
            z10s.push(x10.clone());
            z11s.push(x11.clone());
            z21s.push(x21.clone());
            z22s.push(x22.clone());
            z02s.push(x02.clone());
        }

        let z00 = player0.add_n(sess, &z00s);
        let z10 = player0.add_n(sess, &z10s);
        let z11 = player1.add_n(sess, &z11s);
        let z21 = player1.add_n(sess, &z21s);
        let z22 = player2.add_n(sess, &z22s);
        let z02 = player2.add_n(sess, &z02s);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl ConcatOp {
    pub(crate) fn rep_rep_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: u32,
        xs: &[RepTen<HostRingT>],
    ) -> Result<RepTen<HostRingT>>
    where
        HostPlacement: PlacementConcatenate<S, HostRingT, HostRingT>,
        HostRingT: Clone,
    {
        let mut z00s: Vec<HostRingT> = Vec::new();
        let mut z10s: Vec<HostRingT> = Vec::new();
        let mut z11s: Vec<HostRingT> = Vec::new();
        let mut z21s: Vec<HostRingT> = Vec::new();
        let mut z22s: Vec<HostRingT> = Vec::new();
        let mut z02s: Vec<HostRingT> = Vec::new();

        let (player0, player1, player2) = plc.host_placements();
        for x in xs.iter() {
            let RepTen {
                shares: [[x00, x10], [x11, x21], [x22, x02]],
            } = &x;

            z00s.push(x00.clone());
            z10s.push(x10.clone());
            z11s.push(x11.clone());
            z21s.push(x21.clone());
            z22s.push(x22.clone());
            z02s.push(x02.clone());
        }
        let z00 = player0.concatenate(sess, axis, &z00s);
        let z10 = player0.concatenate(sess, axis, &z10s);
        let z11 = player1.concatenate(sess, axis, &z11s);
        let z21 = player1.concatenate(sess, axis, &z21s);
        let z22 = player2.concatenate(sess, axis, &z22s);
        let z02 = player2.concatenate(sess, axis, &z02s);
        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl RepSumOp {
    pub(crate) fn kernel<S: Session, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: Option<u32>,
        x: RepTen<RingT>,
    ) -> Result<RepTen<RingT>>
    where
        HostPlacement: PlacementSum<S, RingT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<RingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.sum(sess, axis, x00);
        let z10 = player0.sum(sess, axis, x10);
        let z11 = player1.sum(sess, axis, x11);
        let z21 = player1.sum(sess, axis, x21);
        let z22 = player2.sum(sess, axis, x22);
        let z02 = player2.sum(sess, axis, x02);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl RepTruncPrOp {
    pub(crate) fn kernel<S: Session, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        amount: u32,
        xe: RepTen<HostRingT>,
    ) -> Result<RepTen<HostRingT>>
    where
        AdditivePlacement: PlacementRepToAdt<S, RepTen<HostRingT>, AdtTen<HostRingT>>,
        AdditivePlacement: TruncPrProvider<S, AdtTen<HostRingT>, AdtTen<HostRingT>>,
        ReplicatedPlacement: PlacementAdtToRep<S, AdtTen<HostRingT>, RepTen<HostRingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let adt = AdditivePlacement {
            owners: [player0.owner, player1.owner],
        };
        let provider = player2;

        let x_adt = adt.rep_to_adt(sess, &xe);
        let y_adt = adt.trunc_pr(sess, amount as usize, &provider, &x_adt);
        Ok(rep.adt_to_rep(sess, &y_adt))
    }
}

impl AdtToRepOp {
    pub(crate) fn kernel<S: Session, ShapeT, SeedT, KeyT, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AdtTen<HostRingT>,
    ) -> Result<RepTen<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement> + Clone,
        AdtTen<HostRingT>: CanonicalType,
        <AdtTen<HostRingT> as CanonicalType>::Type: KnownType<S>,
        HostPlacement: PlacementShape<S, HostRingT, ShapeT>,
        HostPlacement: PlacementKeyGen<S, KeyT>,
        HostPlacement: PlacementSampleUniformSeeded<S, ShapeT, SeedT, HostRingT>,
        HostPlacement: PlacementDeriveSeed<S, KeyT, SeedT>,
        AdditivePlacement: PlacementSub<S, AdtTen<HostRingT>, AdtTen<HostRingT>, AdtTen<HostRingT>>,
        AdtTen<HostRingT>: Into<st!(AdtTen<HostRingT>, S)>,
        HostPlacement: PlacementReveal<S, st!(AdtTen<HostRingT>, S), HostRingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<HostRingT>>,
    {
        let AdtTen { shares: [x0, x1] } = &x;

        let adt = x.placement()?;
        let (adt_player0, adt_player1) = adt.host_placements();
        let (rep_player0, rep_player1, rep_player2) = rep.host_placements();
        let (provider, provider_index, rep_others) = match () {
            _ if rep_player0 != adt_player0 && rep_player0 != adt_player1 => {
                (rep_player0, 0, [rep_player1, rep_player2])
            }
            _ if rep_player1 != adt_player0 && rep_player1 != adt_player1 => {
                (rep_player1, 1, [rep_player2, rep_player0])
            }
            _ if rep_player2 != adt_player0 && rep_player2 != adt_player1 => {
                (rep_player2, 2, [rep_player0, rep_player1])
            }
            _ => unimplemented!("protocol error in AdtToRep kernel"), // something is wrong in the protocol otherwise
        };

        let sync_key0 = SyncKey::random();
        let sync_key1 = SyncKey::random();

        let k = provider.gen_key(sess);
        let seed1 = provider.derive_seed(sess, sync_key0, &k);
        let seed2 = provider.derive_seed(sess, sync_key1, &k);

        let shape0 = adt_player0.shape(sess, x0);
        let shape1 = adt_player1.shape(sess, x1);

        let y0 = adt_player0.sample_uniform_seeded(sess, &shape0, &seed1);
        let y1 = adt_player1.sample_uniform_seeded(sess, &shape1, &seed2);

        let y0_provider = provider.sample_uniform_seeded(sess, &shape0, &seed1);
        let y1_provider = provider.sample_uniform_seeded(sess, &shape0, &seed2);

        let y = AdtTen {
            shares: [y0.clone(), y1.clone()],
        };
        let c = adt_player0.reveal(sess, &adt.sub(sess, &x, &y).into());

        let shares = match () {
            _ if provider_index == 0 => {
                match () {
                    // (D, adt_0, adt_1) case
                    _ if adt_player0 == rep_others[0] => {
                        [[y1_provider, y0_provider], [y0, c.clone()], [c, y1]]
                    }
                    // (D, adt_1, adt_0) case
                    _ if adt_player0 == rep_others[1] => {
                        [[y0_provider, y1_provider], [y1, c.clone()], [c, y0]]
                    }
                    // same as previously, we don't care since parties sends their shares
                    _ => [[y0_provider, y1_provider], [y1, c.clone()], [c, y0]],
                }
            }
            _ if provider_index == 1 => {
                match () {
                    // (adt_1, D, adt_0)
                    _ if adt_player0 == rep_others[0] => {
                        [[c.clone(), y1], [y1_provider, y0_provider], [y0, c]]
                    }
                    // (adt_0, D, adt_1)
                    _ if adt_player0 == rep_others[1] => {
                        [[c.clone(), y0], [y0_provider, y1_provider], [y1, c]]
                    }
                    // same as previously, we don't care since parties sends their shares
                    _ => [[c.clone(), y0], [y0_provider, y1_provider], [y1, c]],
                }
            }
            _ => {
                match () {
                    // (adt0, adt1, D)
                    _ if adt_player0 == rep_others[0] => {
                        [[y0, c.clone()], [c, y1], [y1_provider, y0_provider]]
                    }
                    // (adt1, adt0, D)
                    _ if adt_player0 == rep_others[1] => {
                        [[y1, c.clone()], [c, y0], [y0_provider, y1_provider]]
                    }
                    // same as previously, we don't care since parties sends their shares
                    _ => [[y1, c.clone()], [c, y0], [y0_provider, y1_provider]],
                }
            }
        };
        Ok(rep.place(sess, RepTen { shares }))
    }
}

impl FillOp {
    pub(crate) fn ring64_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        value: u64,
        rep_shape: RepShape<ShapeT>,
    ) -> Result<RepTen<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return PublicReplicatedTensor, but we don't have that type yet
        let (player0, player1, player2) = rep.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &rep_shape;

        let shares = [
            [
                player0.fill(sess, Constant::Ring64(value), s0),
                player0.fill(sess, Constant::Ring64(0_u64), s0),
            ],
            [
                player1.fill(sess, Constant::Ring64(0_u64), s1),
                player1.fill(sess, Constant::Ring64(0_u64), s1),
            ],
            [
                player2.fill(sess, Constant::Ring64(0_u64), s2),
                player2.fill(sess, Constant::Ring64(value), s2),
            ],
        ];

        Ok(RepTen { shares })
    }

    pub(crate) fn ring128_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        value: u128,
        rep_shape: RepShape<ShapeT>,
    ) -> Result<RepTen<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return PublicReplicatedTensor, but we don't have that type yet
        let (player0, player1, player2) = rep.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &rep_shape;

        let shares = [
            [
                player0.fill(sess, Constant::Ring128(value), s0),
                player0.fill(sess, Constant::Ring128(0_u128), s0),
            ],
            [
                player1.fill(sess, Constant::Ring128(0_u128), s1),
                player1.fill(sess, Constant::Ring128(0_u128), s1),
            ],
            [
                player2.fill(sess, Constant::Ring128(0_u128), s2),
                player2.fill(sess, Constant::Ring128(value), s2),
            ],
        ];

        Ok(RepTen { shares })
    }

    pub(crate) fn mir_ring64_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        mir: &Mirrored3Placement,
        value: u64,
        rep_shape: RepShape<ShapeT>,
    ) -> Result<MirTen<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        let (player0, player1, player2) = mir.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &rep_shape;

        let values = [
            player0.fill(sess, Constant::Ring64(value), s0),
            player1.fill(sess, Constant::Ring64(value), s1),
            player2.fill(sess, Constant::Ring64(value), s2),
        ];

        Ok(MirTen { values })
    }

    pub(crate) fn mir_ring128_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        mir: &Mirrored3Placement,
        value: u128,
        rep_shape: RepShape<ShapeT>,
    ) -> Result<MirTen<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        let (player0, player1, player2) = mir.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &rep_shape;

        let values = [
            player0.fill(sess, Constant::Ring128(value), s0),
            player1.fill(sess, Constant::Ring128(value), s1),
            player2.fill(sess, Constant::Ring128(value), s2),
        ];

        Ok(MirTen { values })
    }

    pub(crate) fn rep_bit_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        value: u8,
        rep_shape: RepShape<ShapeT>,
    ) -> Result<RepTen<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return PublicReplicatedTensor, but we don't have that type yet
        let (player0, player1, player2) = rep.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &rep_shape;

        let shares = [
            [
                player0.fill(sess, Constant::Bit(value), s0),
                player0.fill(sess, Constant::Bit(0_u8), s0),
            ],
            [
                player1.fill(sess, Constant::Bit(0_u8), s1),
                player1.fill(sess, Constant::Bit(0_u8), s1),
            ],
            [
                player2.fill(sess, Constant::Bit(0_u8), s2),
                player2.fill(sess, Constant::Bit(value), s2),
            ],
        ];

        Ok(RepTen { shares })
    }

    pub(crate) fn mir_bit_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        mir: &Mirrored3Placement,
        value: u8,
        shape: RepShape<ShapeT>,
    ) -> Result<MirTen<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        let (player0, player1, player2) = mir.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &shape;

        let values = [
            player0.fill(sess, Constant::Bit(value), s0),
            player1.fill(sess, Constant::Bit(value), s1),
            player2.fill(sess, Constant::Bit(value), s2),
        ];

        Ok(MirTen { values })
    }
}

impl RepShlOp {
    pub(crate) fn kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        amount: usize,
        x: RepTen<HostRingT>,
    ) -> Result<RepTen<HostRingT>>
    where
        HostPlacement: PlacementShl<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;
        let z00 = player0.shl(sess, amount, x00);
        let z10 = player0.shl(sess, amount, x10);

        let z11 = player1.shl(sess, amount, x11);
        let z21 = player1.shl(sess, amount, x21);

        let z22 = player2.shl(sess, amount, x22);
        let z02 = player2.shl(sess, amount, x02);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl ExpandDimsOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Vec<u32>,
        x: RepTen<HostRingT>,
    ) -> Result<RepTen<HostRingT>>
    where
        HostPlacement: PlacementExpandDims<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.expand_dims(sess, axis.clone(), x00);
        let z10 = player0.expand_dims(sess, axis.clone(), x10);

        let z11 = player1.expand_dims(sess, axis.clone(), x11);
        let z21 = player1.expand_dims(sess, axis.clone(), x21);

        let z22 = player2.expand_dims(sess, axis.clone(), x22);
        let z02 = player2.expand_dims(sess, axis, x02);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl IndexAxisOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        index: usize,
        x: RepTen<HostRingT>,
    ) -> Result<RepTen<HostRingT>>
    where
        HostPlacement: PlacementIndexAxis<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.index_axis(sess, axis, index, x00);
        let z10 = player0.index_axis(sess, axis, index, x10);

        let z11 = player1.index_axis(sess, axis, index, x11);
        let z21 = player1.index_axis(sess, axis, index, x21);

        let z22 = player2.index_axis(sess, axis, index, x22);
        let z02 = player2.index_axis(sess, axis, index, x02);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl IndexOp {
    pub(crate) fn rep_kernel<S: Session, RepBitT, N>(
        sess: &S,
        plc: &ReplicatedPlacement,
        index: usize,
        x: RepBitArray<RepBitT, N>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementIndexAxis<S, RepBitT, RepBitT>,
    {
        // TODO until we have HostBitArrays we simply delegate to IndexAxis operations
        let stacked_tensor = x.0;
        Ok(plc.index_axis(sess, 0, index, &stacked_tensor))
    }
}

impl RepDiagOp {
    pub(crate) fn kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepTen<HostRingT>,
    ) -> Result<RepTen<HostRingT>>
    where
        HostPlacement: PlacementDiag<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.diag(sess, x00);
        let z10 = player0.diag(sess, x10);

        let z11 = player1.diag(sess, x11);
        let z21 = player1.diag(sess, x21);

        let z22 = player2.diag(sess, x22);
        let z02 = player2.diag(sess, x02);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl RepSliceOp {
    pub(crate) fn shape_kernel<S: Session, ShapeT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        slice: SliceInfo,
        shape: RepShape<ShapeT>,
    ) -> Result<RepShape<ShapeT>>
    where
        HostPlacement: PlacementSlice<S, ShapeT, ShapeT>,
    {
        let (player0, player1, player2) = plc.host_placements();

        let RepShape {
            shapes: [shape0, shape1, shape2],
        } = shape;

        let new_shape0 = player0.slice(sess, slice.clone(), &shape0);
        let new_shape1 = player1.slice(sess, slice.clone(), &shape1);
        let new_shape2 = player2.slice(sess, slice, &shape2);

        Ok(RepShape {
            shapes: [new_shape0, new_shape1, new_shape2],
        })
    }
}

impl RepShlDimOp {
    pub(crate) fn kernel<S: Session, HostBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        amount: usize,
        bit_length: usize,
        x: RepTen<HostBitT>,
    ) -> Result<RepTen<HostBitT>>
    where
        HostPlacement: PlacementShlDim<S, HostBitT, HostBitT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let z00 = player0.shl_dim(sess, amount, bit_length, &x00);
        let z10 = player0.shl_dim(sess, amount, bit_length, &x10);

        let z11 = player1.shl_dim(sess, amount, bit_length, &x11);
        let z21 = player1.shl_dim(sess, amount, bit_length, &x21);

        let z22 = player2.shl_dim(sess, amount, bit_length, &x22);
        let z02 = player2.shl_dim(sess, amount, bit_length, &x02);

        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl RepMsbOp {
    pub(crate) fn bit_kernel<S: Session, RepRingT, RepBitT, RepBitArrayT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
    ) -> Result<RepBitT>
    where
        RepRingT: Ring<BitLength = N>,
        RepBitArrayT: BitArray<Len = N>,
        ReplicatedPlacement: PlacementBitDec<S, RepRingT, RepBitArrayT>,
        ReplicatedPlacement: PlacementIndex<S, RepBitArrayT, RepBitT>,
    {
        let bits = rep.bit_decompose(sess, &x);
        Ok(rep.index(sess, N::VALUE - 1, &bits))
    }

    pub(crate) fn ring_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
    ) -> Result<RepRingT>
    where
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
    {
        let x_bin = rep.msb(sess, &x);
        Ok(rep.ring_inject(sess, 0, &x_bin))
    }
}

impl RepAbsOp {
    pub(crate) fn kernel<S: Session, RepRingT, MirRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
    ) -> Result<RepRingT>
    where
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementSub<S, MirRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: ShapeFill<S, RepRingT, Result = MirRingT>,
    {
        let msb_ring = rep.msb(sess, &x);
        let double = rep.shl(sess, 1, &msb_ring);
        let ones = rep.shape_fill(sess, Constant::Ring64(1), &msb_ring);
        let sign = rep.sub(sess, &ones, &double);
        Ok(rep.mul(sess, &sign, &x))
    }
}

impl ShapeOp {
    pub(crate) fn rep_kernel<S: Session, RingT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<RingT>,
    ) -> Result<RepShape<ShapeT>>
    where
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let RepTen {
            shares: [[x00, _x10], [x11, _x21], [x22, _x02]],
        } = &x;
        Ok(RepShape {
            shapes: [
                player0.shape(sess, x00),
                player1.shape(sess, x11),
                player2.shape(sess, x22),
            ],
        })
    }

    pub(crate) fn rep_repfixed_kernel<S: Session, RepRingT, RepShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepShapeT>
    where
        ReplicatedPlacement: PlacementShape<S, RepRingT, RepShapeT>,
    {
        Ok(rep.shape(sess, &x.tensor))
    }
}

pub(crate) trait BinaryAdder<S: Session, RepBitT> {
    fn binary_adder(&self, sess: &S, x: &RepBitT, y: &RepBitT, ring_size: usize) -> RepBitT;
}

/// Binary addition protocol for tensors
impl<S: Session, RepBitT> BinaryAdder<S, RepBitT> for ReplicatedPlacement
where
    RepBitT: Clone,
    ReplicatedPlacement: PlacementAnd<S, RepBitT, RepBitT, RepBitT>,
    ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
    ReplicatedPlacement: PlacementShlDim<S, RepBitT, RepBitT>,
{
    fn binary_adder(&self, sess: &S, x: &RepBitT, y: &RepBitT, ring_size: usize) -> RepBitT {
        #![allow(clippy::many_single_char_names)]

        let rep = self;
        let log_r = (ring_size as f64).log2() as usize; // we know that R = 64/128

        // g is part of the generator set, p propagator set
        // A few helpful diagrams to understand what is happening here:
        // https://www.chessprogramming.org/Kogge-Stone_Algorithm or here: https://inst.eecs.berkeley.edu/~eecs151/sp19/files/lec20-adders.pdf

        // consider we have inputs a, b to the P,G computing gate
        // P = P_a and P_b
        // G = G_b xor (G_a and P_b)

        // P, G can be computed in a tree fashion, performing ops on chunks of len 2^i
        // Note the first level is computed as P0 = x ^ y, G0 = x & y;

        // Perform `g = x * y` for every tensor
        let mut g = rep.and(sess, x, y);

        // Perform `p_store = x + y` (just a helper to avoid compute xor() twice)
        let p_store = rep.xor(sess, x, y);
        let mut p = p_store.clone();

        // (Dragos) Note that in the future we might want to delete shl_dim op and replace it with
        // slice + stack op - however atm we can't do this. It can be unblocked after the following are implemented:
        // 1) slice tensors with unknown shape at compile time
        // 2) stack variable length of replicated tensors (variadic kernels + stack op)

        for i in 0..log_r {
            // computes p << (1<<i)
            // [ a[0], ... a[amount] ... a[ring_size - 1]
            // [ a[amount]...a[ring_size-1] 0 ... 0 ]
            let p1 = rep.shl_dim(sess, 1 << i, ring_size, &p);
            // computes g >> (1<<i)
            let g1 = rep.shl_dim(sess, 1 << i, ring_size, &g);

            // Note that the original algorithm had G_a and P_b, but we can have
            // G_a and P_a instead because the 1s in P_a do not matter in the final result
            // since they are cancelled out by the zeros in G_a
            let p_and_g = rep.and(sess, &p, &g1);

            // update `g = g xor p1 and g1`
            g = rep.xor(sess, &g, &p_and_g);

            // update `p = p * p1`
            p = rep.and(sess, &p, &p1);
        }

        // c is a copy of g with the first tensor (corresponding to the first bit) zeroed out
        let c = rep.shl_dim(sess, 1, ring_size, &g);

        // final result is `z = c xor p_store`
        rep.xor(sess, &c, &p_store)
    }
}

impl RingInjectOp {
    pub(crate) fn rep_kernel<S: Session, HostBitT, HostRingT, HostShapeT, AdtRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        bit_idx: usize,
        x: RepTen<HostBitT>,
    ) -> Result<RepTen<HostRingT>>
    where
        AdtTen<HostRingT>: CanonicalType,
        <AdtTen<HostRingT> as CanonicalType>::Type: KnownType<S>,
        AdtTen<HostRingT>: Into<st!(AdtTen<HostRingT>)>,

        AdtTen<HostBitT>: CanonicalType,
        <AdtTen<HostBitT> as CanonicalType>::Type: KnownType<S>,
        AdtTen<HostBitT>: Into<st!(AdtTen<HostBitT>)>,

        AdtTen<HostRingT>: Into<AdtRingT>,
        st!(AdtTen<HostRingT>): TryInto<AdtTen<HostRingT>>,
        AdtRingT: TryInto<AdtTen<HostRingT>>,

        HostPlacement: PlacementShape<S, HostBitT, HostShapeT>,
        ReplicatedPlacement: PlacementAdtToRep<S, AdtTen<HostRingT>, RepTen<HostRingT>>,
        AdditivePlacement: PlacementFill<S, HostShapeT, AdtRingT>,
        HostPlacement: PlacementFill<S, HostShapeT, HostRingT>,
        AdditivePlacement: DaBitProvider<S, HostShapeT, AdtTen<HostRingT>, AdtTen<HostBitT>>,
        AdditivePlacement: PlacementRepToAdt<S, RepTen<HostBitT>, AdtTen<HostBitT>>,
        AdditivePlacement: PlacementAdd<S, AdtTen<HostBitT>, AdtTen<HostBitT>, AdtTen<HostBitT>>,
        AdditivePlacement: PlacementAdd<S, AdtRingT, HostRingT, AdtRingT>,
        AdditivePlacement: PlacementMul<S, AdtRingT, HostRingT, AdtRingT>,
        AdditivePlacement: PlacementSub<S, AdtRingT, AdtRingT, AdtRingT>,
        AdditivePlacement: PlacementShl<S, AdtRingT, AdtRingT>,
        HostPlacement: PlacementReveal<S, st!(AdtTen<HostBitT>), HostBitT>,
        HostPlacement: PlacementRingInject<S, HostBitT, HostRingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let adt = AdditivePlacement {
            owners: [player0.clone().owner, player1.owner],
        };
        let provider = player2;

        let RepTensor {
            shares: [[x00, _x10], [_x11, _x21], [x22, _x02]],
        } = &x;

        let shape_provider = provider.shape(sess, x22);
        let shape_player0 = player0.shape(sess, x00);

        // One could think to wrap this up into an additive shape for Hosts@(P0, P2)
        // but the additive placement that generates a dabit is Hosts@(P0, P1)
        // to avoid confusion the API corresponding gen_dabit takes two input shapes
        // 1) shape_provider - provider (dealer) shape
        // 2) shape_player0 - shape that corresponds to the party expanding the seeds received from provider.

        let (b_ring, b_bin) = adt.gen_dabit(sess, shape_provider, shape_player0, &provider);
        let x_adt = adt.rep_to_adt(sess, &x);

        // TODO(Morten) the following block would likely clean up nicely if we instead
        // revealed to a mirrored-2 placement, which would use only concrete kernels
        let c = with_context!(adt, sess, x_adt + b_bin);
        let c_open = player0.reveal(sess, &c.into());
        let c_ring = player0.ring_inject(sess, 0, &c_open);
        let b_ring = b_ring.into();
        let x_adt_ring = with_context!(
            adt,
            sess,
            b_ring + c_ring - b_ring * c_ring - b_ring * c_ring
        );
        let shifted_x_adt = adt.shl(sess, bit_idx, &x_adt_ring);
        let shifted_x_adt = shifted_x_adt.try_into().ok().unwrap();

        Ok(rep.adt_to_rep(sess, &shifted_x_adt))
    }
}

/// Split a replicated secret `x` into two replicated bits `(x1, x2)`
///
/// This is done such that when interpreted as ring tensors, they reconstruct to `x`
/// i.e. `(x1 + x2) = x` over the ring. This is useful for some protocols that don't
/// necessarily need the full bit-decomposition such as exponentiation.
pub(crate) trait PlacementSplit<S: Session, T, O1, O2> {
    fn split(&self, sess: &S, x: &T) -> (O1, O2);
}

impl<
        S: Session,
        HostRingT: Placed<Placement = HostPlacement>,
        HostBitT: Placed<Placement = HostPlacement>,
    >
    PlacementSplit<
        S,
        Symbolic<RepTen<HostRingT>>,
        Symbolic<RepTen<HostBitT>>,
        Symbolic<RepTen<HostBitT>>,
    > for ReplicatedPlacement
where
    ReplicatedPlacement: PlacementSplit<S, RepTen<HostRingT>, RepTen<HostBitT>, RepTen<HostBitT>>,
    RepTen<HostRingT>: Into<Symbolic<RepTen<HostRingT>>>,
    RepTen<HostBitT>: Into<Symbolic<RepTen<HostBitT>>>,
{
    fn split(
        &self,
        sess: &S,
        x: &Symbolic<RepTen<HostRingT>>,
    ) -> (Symbolic<RepTen<HostBitT>>, Symbolic<RepTen<HostBitT>>) {
        let concrete_x = match x {
            Symbolic::Concrete(x) => x,
            Symbolic::Symbolic(_) => {
                unimplemented!()
            }
        };
        let (a, b) = Self::split(self, sess, concrete_x);
        (a.into(), b.into())
    }
}

impl<S: Session, HostRingT, HostBitT>
    PlacementSplit<S, RepTen<HostRingT>, RepTen<HostBitT>, RepTen<HostBitT>> for ReplicatedPlacement
where
    HostShape: KnownType<S>,
    HostBitT: Clone,

    HostPlacement: PlacementFill<S, cs!(HostShape), HostBitT>,
    HostPlacement: PlacementShape<S, HostBitT, cs!(HostShape)>,
    HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,

    RepTen<HostBitT>: CanonicalType,
    <RepTen<HostBitT> as CanonicalType>::Type: KnownType<S>,
    st!(RepTen<HostBitT>): TryInto<RepTen<HostBitT>>,

    ReplicatedPlacement: PlacementShare<S, HostBitT, st!(RepTen<HostBitT>)>,
    HostPlacement: PlacementBitDec<S, HostRingT, HostBitT>,
    ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
{
    fn split(&self, sess: &S, x: &RepTen<HostRingT>) -> (RepTen<HostBitT>, RepTen<HostBitT>) {
        let (player0, player1, player2) = self.host_placements();

        let RepTen {
            shares: [[x00, x10], [_x11, x21], [x22, _x02]],
        } = &x;

        let left = with_context!(player0, sess, x00 + x10);
        let bsl = player0.bit_decompose(sess, &left);

        // transform x2 into boolean sharing
        let x2_on_1 = player1.bit_decompose(sess, x21);
        let x2_on_2 = player2.bit_decompose(sess, x22);

        let p0_zero = player0.fill(sess, 0_u8.into(), &player0.shape(sess, &bsl));
        let p1_zero = player1.fill(sess, 0_u8.into(), &player1.shape(sess, &x2_on_1));
        let p2_zero = player2.fill(sess, 0_u8.into(), &player2.shape(sess, &x2_on_2));

        let rep_bsl = self.share(sess, &bsl);
        let rep_bsr = RepTen {
            shares: [
                [p0_zero.clone(), p0_zero],
                [p1_zero, x2_on_1],
                [x2_on_2, p2_zero],
            ],
        };

        (rep_bsl.try_into().ok().unwrap(), rep_bsr)
    }
}

impl RepBitDecOp {
    pub(crate) fn ring_kernel<S: Session, RepRingT, RepBitT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
    ) -> Result<RepBitArray<RepBitT, N>>
    where
        RepRingT: Ring<BitLength = N>,
        ReplicatedPlacement: PlacementSplit<S, RepRingT, RepBitT, RepBitT>,
        ReplicatedPlacement: BinaryAdder<S, RepBitT>,
    {
        let (x0, x1) = rep.split(sess, &x);
        let res = rep.binary_adder(sess, &x0, &x1, RepRingT::BitLength::VALUE);
        Ok(RepBitArray(res, PhantomData))
    }
}

impl RepBitComposeOp {
    pub(crate) fn rep_kernel<S: Session, ShapeT, RepRingT, RepBitArrayT, RepBitT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepBitArrayT,
    ) -> Result<RepRingT>
    where
        RepRingT: Ring<BitLength = N>,

        ReplicatedPlacement: PlacementIndex<S, RepBitArrayT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShape<S, RepBitT, ShapeT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepRingT>,
    {
        let v: Vec<_> = (0..RepRingT::BitLength::VALUE)
            .map(|i| rep.index(sess, i, &x))
            .collect();

        let zeros = rep.fill(sess, 0u64.into(), &rep.shape(sess, &v[0]));
        Ok(v.iter().enumerate().fold(zeros, |x, (i, y)| {
            rep.add(sess, &x, &rep.ring_inject(sess, i, y))
        }))
    }
}

/// Shift all shares of replicated secret to the right
///
/// It should be used carefully since `[x] >> amount` is *not* equal to
/// `[ [x0 >> amount, x1 >> amount], [x1 >> amount, x2 >> amount], [x2 >> amount, x0>>amount]`.
/// Used in conjunction with split operation so that we don't use the full
/// bit-decomposition in order to perform exact truncation.
pub(crate) trait PlacementShrRaw<S: Session, T, O> {
    fn shr_raw(&self, sess: &S, amount: usize, x: &T) -> O;
}

impl<S: Session, HostRingT: Placed<Placement = HostPlacement>>
    PlacementShrRaw<S, Symbolic<RepTen<HostRingT>>, Symbolic<RepTen<HostRingT>>>
    for ReplicatedPlacement
where
    ReplicatedPlacement: PlacementShrRaw<S, RepTen<HostRingT>, RepTen<HostRingT>>,
    RepTen<HostRingT>: Into<Symbolic<RepTen<HostRingT>>>,
{
    fn shr_raw(
        &self,
        sess: &S,
        amount: usize,
        x: &Symbolic<RepTen<HostRingT>>,
    ) -> Symbolic<RepTen<HostRingT>> {
        let concrete_x = match x {
            Symbolic::Concrete(x) => x,
            Symbolic::Symbolic(_) => {
                unimplemented!()
            }
        };
        let concrete_y = Self::shr_raw(self, sess, amount, concrete_x);
        concrete_y.into()
    }
}

impl<S: Session, HostRingT> PlacementShrRaw<S, RepTen<HostRingT>, RepTen<HostRingT>>
    for ReplicatedPlacement
where
    HostPlacement: PlacementShr<S, HostRingT, HostRingT>,
{
    fn shr_raw(&self, sess: &S, amount: usize, x: &RepTen<HostRingT>) -> RepTen<HostRingT> {
        let (player0, player1, player2) = self.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.shr(sess, amount, x00);
        let z10 = player0.shr(sess, amount, x10);

        let z11 = player1.shr(sess, amount, x11);
        let z21 = player1.shr(sess, amount, x21);

        let z22 = player2.shr(sess, amount, x22);
        let z02 = player2.shr(sess, amount, x02);

        RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
}

pub(crate) struct AbstractReplicatedSeeds<T> {
    seeds: [[T; 2]; 3],
}

pub(crate) trait ReplicatedSeedsGen<S: Session, KeyT, SeedT> {
    fn gen_seeds(&self, ctx: &S, setup: &RepSetup<KeyT>) -> AbstractReplicatedSeeds<SeedT>;
}

impl<S: Session> ReplicatedSeedsGen<S, cs!(PrfKey), cs!(Seed)> for ReplicatedPlacement
where
    PrfKey: KnownType<S>,
    Seed: KnownType<S>,
    HostPlacement: PlacementDeriveSeed<S, cs!(PrfKey), cs!(Seed)>,
{
    fn gen_seeds(
        &self,
        ctx: &S,
        setup: &RepSetup<cs!(PrfKey)>,
    ) -> AbstractReplicatedSeeds<cs!(Seed)> {
        let (player0, player1, player2) = self.host_placements();

        let RepSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = setup;

        // NOTE for now we pick random sync_keys _at compile time_, which is okay from
        // a security perspective since the seeds depend on both the keys and the sid.
        // however, with sub-computations we could fix these as eg `0`, `1`, and `2`
        // and make compilation a bit more deterministic
        let sync_key0 = SyncKey::random();
        let sync_key1 = SyncKey::random();
        let sync_key2 = SyncKey::random();

        let s00 = player0.derive_seed(ctx, sync_key0.clone(), k00);
        let s10 = player0.derive_seed(ctx, sync_key1.clone(), k10);

        let s11 = player1.derive_seed(ctx, sync_key1, k11);
        let s21 = player1.derive_seed(ctx, sync_key2.clone(), k21);

        let s22 = player2.derive_seed(ctx, sync_key2, k22);
        let s02 = player2.derive_seed(ctx, sync_key0, k02);

        let seeds = [[s00, s10], [s11, s21], [s22, s02]];
        AbstractReplicatedSeeds { seeds }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct AbstractReplicatedZeroShare<R> {
    alphas: [R; 3],
}

pub(crate) trait ZeroShareGen<S: Session, ShapeT, RingT> {
    fn gen_zero_share(
        &self,
        sess: &S,
        shape: &RepShape<ShapeT>,
    ) -> AbstractReplicatedZeroShare<RingT>;
}

impl<S: Session, RingT, HostShapeT> ZeroShareGen<S, HostShapeT, RingT> for ReplicatedPlacement
where
    PrfKey: KnownType<S>,
    Seed: KnownType<S>,
    <S as Session>::ReplicatedSetup: Clone,
    <S as Session>::ReplicatedSetup: TryInto<RepSetup<m!(PrfKey)>>,
    HostPlacement: PlacementSampleUniformSeeded<S, HostShapeT, m!(Seed), RingT>,
    HostPlacement: PlacementSub<S, RingT, RingT, RingT>,
    ReplicatedPlacement: ReplicatedSeedsGen<S, m!(PrfKey), m!(Seed)>,
{
    fn gen_zero_share(
        &self,
        sess: &S,
        shape: &RepShape<HostShapeT>,
    ) -> AbstractReplicatedZeroShare<RingT> {
        let setup = match (*sess.replicated_setup(self)).clone().try_into() {
            Ok(setup) => setup,
            _ => todo!("not sure what to do with the symbolic setup yet"), // TODO: perhaps a custom kernel could instead output a symbolic op
        };

        let (player0, player1, player2) = self.host_placements();

        let RepShape {
            shapes: [shape0, shape1, shape2],
        } = shape;

        let AbstractReplicatedSeeds {
            seeds: [[s00, s10], [s11, s21], [s22, s02]],
        } = &self.gen_seeds(sess, &setup);

        let r00 = player0.sample_uniform_seeded(sess, shape0, s00);
        let r10 = player0.sample_uniform_seeded(sess, shape0, s10);
        let alpha0 = with_context!(player0, sess, r00 - r10);

        let r11 = player1.sample_uniform_seeded(sess, shape1, s11);
        let r21 = player1.sample_uniform_seeded(sess, shape1, s21);
        let alpha1 = with_context!(player1, sess, r11 - r21);

        let r22 = player2.sample_uniform_seeded(sess, shape2, s22);
        let r02 = player2.sample_uniform_seeded(sess, shape2, s02);
        let alpha2 = with_context!(player2, sess, r22 - r02);

        AbstractReplicatedZeroShare {
            alphas: [alpha0, alpha1, alpha2],
        }
    }
}

impl ReplicatedPlacement {
    pub fn prefix_op<S, RepT>(
        &self,
        sess: &S,
        x: Vec<RepT>,
        op: fn(&Self, &S, &RepT, &RepT) -> RepT,
    ) -> Vec<RepT> {
        let v_len = x.len();

        let log_r = ((v_len as f64).log2().ceil()) as u32;

        let mut res = x;
        for i in 0..log_r {
            for j in 0..(2_i32.pow(log_r) / 2_i32.pow(i + 1)) {
                let y = (2_i32.pow(i) + j * 2_i32.pow(i + 1) - 1) as usize;
                let k_bound = (2_i32.pow(i) + 1) as usize;
                for k in 1..k_bound {
                    if y + k < v_len {
                        res[y + k] = op(self, sess, &res[y], &res[y + k]);
                    }
                }
            }
        }
        res
    }

    pub fn prefix_or<S: Session, RepT>(&self, sess: &S, x: Vec<RepT>) -> Vec<RepT>
    where
        ReplicatedPlacement: PlacementAnd<S, RepT, RepT, RepT>,
        ReplicatedPlacement: PlacementXor<S, RepT, RepT, RepT>,
    {
        let elementwise_or = |rep: &ReplicatedPlacement, sess: &S, x: &RepT, y: &RepT| -> RepT {
            rep.xor(sess, &rep.xor(sess, x, y), &rep.and(sess, x, y))
        };

        self.prefix_op(sess, x, elementwise_or)
    }

    pub fn prefix_and<S: Session, RepT>(&self, sess: &S, x: Vec<RepT>) -> Vec<RepT>
    where
        ReplicatedPlacement: PlacementAnd<S, RepT, RepT, RepT>,
    {
        let elementwise_and = |rep: &ReplicatedPlacement, sess: &S, x: &RepT, y: &RepT| -> RepT {
            rep.and(sess, x, y)
        };

        self.prefix_op(sess, x, elementwise_and)
    }
}

impl SigmoidOp {
    pub(crate) fn rep_rep_kernel<S: Session, RepFixedT, ShapeT, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepFixedT,
    ) -> Result<RepFixedT>
    where
        RepRingT: Clone,
        RepFixedT: FixedpointTensor,
        RepFixedTensor<RepRingT>: Into<RepFixedT>,
        ReplicatedPlacement: PlacementShape<S, RepFixedT, ShapeT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementDiv<S, RepFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementExp<S, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementNeg<S, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementGreaterThan<S, RepFixedT, RepFixedT, RepBitT>,
        ReplicatedPlacement: PlacementMux<S, RepRingT, RepFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
    {
        // TODO [Yann]: revisit once we support mixed arithmetic for division
        let ones = 1.0_f64.as_fixedpoint(x.fractional_precision() as usize);

        let ones_fill = rep.fill(sess, ones.into(), &rep.shape(sess, &x));
        let zeros_fill = rep.fill(sess, 0_u8.into(), &rep.shape(sess, &x));

        let ones_rep = RepFixedTensor {
            tensor: ones_fill,
            integral_precision: x.integral_precision(),
            fractional_precision: x.fractional_precision(),
        }
        .into();
        let zeros_rep = RepFixedTensor {
            tensor: zeros_fill,
            integral_precision: x.integral_precision(),
            fractional_precision: x.fractional_precision(),
        }
        .into();

        let denominator = rep.add(sess, &ones_rep, &rep.exp(sess, &rep.neg(sess, &x)));
        let output = rep.div(sess, &ones_rep, &denominator);

        // input sanitization
        let max_val = 2_f64
            .powf((x.integral_precision() - 1).into())
            .ln()
            .as_fixedpoint(x.fractional_precision() as usize);
        let max_val_fill = rep.fill(sess, max_val.into(), &rep.shape(sess, &x));
        let max_val_rep = RepFixedTensor {
            tensor: max_val_fill,
            integral_precision: x.integral_precision(),
            fractional_precision: x.fractional_precision(),
        }
        .into();

        // compute upper bound
        let upper = rep.greater_than(sess, &x, &max_val_rep); // x > max_val?
        let upper_ring = rep.ring_inject(sess, 0, &upper);
        let upper_wall = rep.mux(sess, &upper_ring, &ones_rep, &output);

        // compute lower bound
        let lower = rep.greater_than(sess, &rep.neg(sess, &max_val_rep), &x); // -max_val > x?
        let lower_ring = rep.ring_inject(sess, 0, &lower);
        let res = rep.mux(sess, &lower_ring, &zeros_rep, &upper_wall);

        Ok(res)
    }
}

impl LessOp {
    pub(crate) fn rep_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &x, &y);
        Ok(rep.msb(sess, &z))
    }

    pub(crate) fn rep_mir_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: MirRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, MirRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &x, &y);
        Ok(rep.msb(sess, &z))
    }

    pub(crate) fn mir_rep_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: MirRingT,
        y: RepRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, MirRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &x, &y);
        Ok(rep.msb(sess, &z))
    }
}

impl GreaterThanOp {
    pub(crate) fn rep_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &y, &x);
        Ok(rep.msb(sess, &z))
    }

    pub(crate) fn rep_mir_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: MirRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, MirRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &y, &x);
        Ok(rep.msb(sess, &z))
    }

    pub(crate) fn mir_rep_kernel<S: Session, RepRingT, MirRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: MirRingT,
        y: RepRingT,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, MirRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
    {
        let z = rep.sub(sess, &y, &x);
        Ok(rep.msb(sess, &z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{FromRawPlc, HostRingTensor};
    use crate::kernels::{
        PlacementRingFixedpointDecode, PlacementRingFixedpointEncode, SyncSession,
    };
    use crate::{N128, N64};
    use ndarray::array;
    use proptest::prelude::*;

    #[test]
    fn test_ring_identity() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x = HostRingTensor::from_raw_plc(array![1u64, 2, 3], alice.clone());
        let expected = x.clone();

        let sess = SyncSession::default();

        let x_shared = rep.share(&sess, &x);

        let iden = rep.identity(&sess, &x_shared);
        let opened_result = alice.reveal(&sess, &iden);
        assert_eq!(opened_result, expected);
    }

    #[test]
    fn test_identity_diff_plc() {
        let alice0 = HostPlacement {
            owner: "alice-0".into(),
        };
        let rep0 = ReplicatedPlacement {
            owners: ["alice-0".into(), "bob-0".into(), "carole-0".into()],
        };
        let rep1 = ReplicatedPlacement {
            owners: ["alice-1".into(), "bob-1".into(), "carole-1".into()],
        };

        let x = HostRingTensor::from_raw_plc(array![1u64, 2, 3], alice0.clone());
        let expected = x.clone();

        let sess = SyncSession::default();

        let x_shared = rep0.share(&sess, &x);

        let iden = rep1.identity(&sess, &x_shared);
        let opened_result = alice0.reveal(&sess, &iden);
        assert_eq!(opened_result, expected);
    }

    #[test]
    fn test_adt_to_rep() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let bob = HostPlacement {
            owner: "bob".into(),
        };
        let carole = HostPlacement {
            owner: "carole".into(),
        };

        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x1 = AdditiveRing64Tensor {
            shares: [
                HostRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "alice".into(),
                    },
                ),
                HostRingTensor::from_raw_plc(
                    array![4, 5, 6],
                    HostPlacement {
                        owner: "bob".into(),
                    },
                ),
            ],
        };

        let sess = SyncSession::default();

        let x1_rep = rep.adt_to_rep(&sess, &x1);
        assert_eq!(alice.reveal(&sess, &x1_rep), alice.reveal(&sess, &x1));
        assert_eq!(bob.reveal(&sess, &x1_rep), bob.reveal(&sess, &x1));
        assert_eq!(carole.reveal(&sess, &x1_rep), carole.reveal(&sess, &x1));

        let x2 = AdditiveRing64Tensor {
            shares: [
                HostRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "bob".into(),
                    },
                ),
                HostRingTensor::from_raw_plc(
                    array![4, 5, 6],
                    HostPlacement {
                        owner: "alice".into(),
                    },
                ),
            ],
        };

        let x2_rep = rep.adt_to_rep(&sess, &x2);
        assert_eq!(alice.reveal(&sess, &x2_rep), alice.reveal(&sess, &x2));
        assert_eq!(bob.reveal(&sess, &x2_rep), bob.reveal(&sess, &x2));
        assert_eq!(carole.reveal(&sess, &x2_rep), carole.reveal(&sess, &x2));

        let x3 = AdditiveRing64Tensor {
            shares: [
                HostRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "david".into(),
                    },
                ),
                HostRingTensor::from_raw_plc(
                    array![4, 5, 6],
                    HostPlacement {
                        owner: "eric".into(),
                    },
                ),
            ],
        };

        let x3_rep = rep.adt_to_rep(&sess, &x3);
        assert_eq!(alice.reveal(&sess, &x3_rep), alice.reveal(&sess, &x3));
        assert_eq!(bob.reveal(&sess, &x3_rep), bob.reveal(&sess, &x3));
        assert_eq!(carole.reveal(&sess, &x3_rep), carole.reveal(&sess, &x3));

        let x4 = AdditiveRing64Tensor {
            shares: [
                HostRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "alice".into(),
                    },
                ),
                HostRingTensor::from_raw_plc(
                    array![4, 5, 6],
                    HostPlacement {
                        owner: "eric".into(),
                    },
                ),
            ],
        };

        let x4_rep = rep.adt_to_rep(&sess, &x4);
        assert_eq!(alice.reveal(&sess, &x4_rep), alice.reveal(&sess, &x4));
        assert_eq!(bob.reveal(&sess, &x4_rep), bob.reveal(&sess, &x4));
        assert_eq!(carole.reveal(&sess, &x4_rep), carole.reveal(&sess, &x4));
    }

    #[test]
    fn test_rep_mean() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };

        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let sess = SyncSession::default();

        let scaling_base = 2;
        let scaling_exp = 24;

        let x = HostFloat64Tensor::from_raw_plc(
            array![1.0, 2.0, 3.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            alice.clone(),
        );
        let x = alice.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &x);
        let x_shared = rep.share(&sess, &x);

        let mean = rep.mean_as_fixedpoint(&sess, None, scaling_base, scaling_exp, &x_shared);
        let mean = rep.trunc_pr(&sess, scaling_exp, &mean);
        let opened_result = alice.reveal(&sess, &mean);
        let decoded_result =
            alice.fixedpoint_ring_decode(&sess, scaling_base, scaling_exp, &opened_result);

        assert!(num_traits::abs(2.0 - decoded_result.0[[]]) < 0.01);
    }

    use ndarray::prelude::*;
    use rstest::rstest;

    #[test]
    fn test_rep_add_n() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let bob = HostPlacement {
            owner: "bob".into(),
        };
        let carole = HostPlacement {
            owner: "carole".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        // 64 bit
        let a = HostRingTensor::from_raw_plc(array![1u64, 2, 3], alice.clone());
        let b = HostRingTensor::from_raw_plc(array![2u64, 3, 4], bob.clone());
        let c = HostRingTensor::from_raw_plc(array![5u64, 12, 13], carole.clone());

        let expected = HostRingTensor::from_raw_plc(array![8u64, 17, 20], alice.clone());

        let inputs = vec![a, b, c];

        let sess = SyncSession::default();

        let shares: Vec<RepTensor<HostRingTensor<u64>>> =
            inputs.into_iter().map(|x| rep.share(&sess, &x)).collect();

        let sum = rep.add_n(&sess, &shares);
        let opened_result = alice.reveal(&sess, &sum);

        assert_eq!(expected, opened_result);

        // 128 bit
        let a = HostRingTensor::from_raw_plc(array![[1u128, 2, 3], [2u128, 3, 4]], alice.clone());
        let b = HostRingTensor::from_raw_plc(array![[2u128, 3, 4], [2u128, 3, 4]], bob);
        let c = HostRingTensor::from_raw_plc(array![[5u128, 12, 13], [1u128, 2, 3]], carole);

        let expected =
            HostRingTensor::from_raw_plc(array![[8u128, 17, 20], [5, 8, 11]], alice.clone());

        let inputs = vec![a, b, c];

        let sess = SyncSession::default();

        let shares: Vec<RepTensor<HostRingTensor<u128>>> =
            inputs.into_iter().map(|x| rep.share(&sess, &x)).collect();

        let sum = rep.add_n(&sess, &shares);
        let opened_result = alice.reveal(&sess, &sum);

        assert_eq!(expected, opened_result);
    }

    #[test]
    fn test_rep_sum() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x = HostRingTensor::from_raw_plc(array![1u64, 2, 3], alice.clone());

        let sess = SyncSession::default();

        let x_shared = rep.share(&sess, &x);

        let sum = rep.sum(&sess, None, &x_shared);
        let opened_result = alice.reveal(&sess, &sum);

        assert_eq!(6, opened_result.0[[]].0);
    }

    macro_rules! diag_op_test {
        ($func_name:ident, $rt:ty, $tt:ident) => {
            fn $func_name() {
                let x = array![[1 as $rt, 2], [3, 4]].into_dyn();
                let exp = array![1 as $rt, 4].into_dyn();

                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let xr = $tt::from_raw_plc(x, alice.clone());

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &xr);

                let diag = rep.diag(&sess, &x_shared);
                let opened_diag = alice.reveal(&sess, &diag);
                assert_eq!(opened_diag, $tt::from_raw_plc(exp, alice.clone()))
            }
        };
    }

    diag_op_test!(rep_diag_bit, u8, HostBitTensor);
    diag_op_test!(rep_diag_ring64, u64, HostRingTensor);
    diag_op_test!(rep_diag_ring128, u128, HostRingTensor);

    #[test]
    fn test_rep_diag_bit() {
        rep_diag_bit()
    }

    #[test]
    fn test_rep_diag_ring64() {
        rep_diag_ring64()
    }

    #[test]
    fn test_rep_diag_ring128() {
        rep_diag_ring128()
    }

    macro_rules! index_axis_op_test {
        ($func_name:ident, $rt:ty, $tt:ident) => {
            fn $func_name() {
                let x = array![[[1 as $rt, 2], [3, 4]], [[4, 5], [6, 7]]].into_dyn();
                let exp = array![[4 as $rt, 5], [6, 7]].into_dyn();

                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let xr = $tt::from_raw_plc(x, alice.clone());

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &xr);

                let index_axis = rep.index_axis(&sess, 0, 1, &x_shared);
                let opened_index_axis = alice.reveal(&sess, &index_axis);
                assert_eq!(opened_index_axis, $tt::from_raw_plc(exp, alice.clone()))
            }
        };
    }

    index_axis_op_test!(rep_index_axis_bit, u8, HostBitTensor);
    index_axis_op_test!(rep_index_axis_ring64, u64, HostRingTensor);
    index_axis_op_test!(rep_index_axis_ring128, u128, HostRingTensor);

    #[test]
    fn test_rep_index_axis_bit() {
        rep_index_axis_bit()
    }

    #[test]
    fn test_rep_index_axis_ring64() {
        rep_index_axis_ring64()
    }

    #[test]
    fn test_rep_index_axis_ring128() {
        rep_index_axis_ring128()
    }

    macro_rules! index_op_test {
        ($func_name:ident, $rt:ty, $tt:ident, $n:ty) => {
            fn $func_name() {
                let x = array![[1 as $rt, 2], [3, 4]].into_dyn();
                let exp = array![1 as $rt, 2].into_dyn();

                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let xr = HostBitTensor::from_raw_plc(x, alice.clone());

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &xr);
                let x_shared_bit_array =
                    RepBitArray::<ReplicatedBitTensor, $n>(x_shared, PhantomData);

                let index = rep.index(&sess, 0, &x_shared_bit_array);
                let opened_index = alice.reveal(&sess, &index);
                assert_eq!(opened_index, $tt::from_raw_plc(exp, alice.clone()))
            }
        };
    }

    index_op_test!(rep_index_bit64, u8, HostBitTensor, N64);
    index_op_test!(rep_index_bit128, u8, HostBitTensor, N128);

    #[test]
    fn test_rep_index_bit64() {
        rep_index_bit64()
    }

    #[test]
    fn test_rep_index_bit128() {
        rep_index_bit128()
    }

    macro_rules! rep_add_test {
        ($func_name:ident, $tt: ident) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = HostRingTensor::from_raw_plc(xs, alice.clone());
                let y = HostRingTensor::from_raw_plc(ys, alice.clone());

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &x);
                let y_shared = rep.share(&sess, &y);

                let sum = rep.add(&sess, &x_shared, &y_shared);
                let opened_sum = alice.reveal(&sess, &sum);
                assert_eq!(opened_sum, HostRingTensor::from_raw_plc(zs, alice.clone()));
            }
        };
    }

    rep_add_test!(test_rep_add64, u64);
    rep_add_test!(test_rep_add128, u128);

    #[rstest]
    #[case(array![1_u64, 2, 3].into_dyn(),
        array![1_u64, 2, 3].into_dyn(),
        array![2_u64, 4, 6].into_dyn())
    ]
    #[case(array![-1_i64 as u64, -2_i64 as u64, -3_i64 as u64].into_dyn(),
        array![1_u64, 2, 3].into_dyn(),
        array![0_u64, 0, 0].into_dyn())
    ]
    fn test_rep_add_64(#[case] x: ArrayD<u64>, #[case] y: ArrayD<u64>, #[case] z: ArrayD<u64>) {
        test_rep_add64(x, y, z);
    }

    #[rstest]
    #[case(array![1_u128, 2, 3].into_dyn(),
        array![1_u128, 2, 3].into_dyn(),
        array![2_u128, 4, 6].into_dyn())
    ]
    #[case(array![-1_i128 as u128, -2_i128 as u128, -3_i128 as u128].into_dyn(),
        array![1_u128, 2, 3].into_dyn(),
        array![0_u128, 0, 0].into_dyn())
    ]
    fn test_rep_add_128(#[case] x: ArrayD<u128>, #[case] y: ArrayD<u128>, #[case] z: ArrayD<u128>) {
        test_rep_add128(x, y, z);
    }

    macro_rules! rep_binary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = HostRingTensor::from_raw_plc(xs, alice.clone());
                let y = HostRingTensor::from_raw_plc(ys, alice.clone());

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &x);
                let y_shared = rep.share(&sess, &y);

                let sum: RepTensor<HostRingTensor<$tt>> =
                    rep.$test_func(&sess, &x_shared, &y_shared);
                let opened_product = alice.reveal(&sess, &sum);
                assert_eq!(
                    opened_product,
                    HostRingTensor::from_raw_plc(zs, alice.clone())
                );
            }
        };
    }

    rep_binary_func_test!(test_rep_mul64, mul<u64>);
    rep_binary_func_test!(test_rep_mul128, mul<u128>);
    rep_binary_func_test!(test_rep_dot64, dot<u64>);
    rep_binary_func_test!(test_rep_dot128, dot<u128>);

    macro_rules! pairwise_same_length {
        ($func_name:ident, $tt: ident) => {
            fn $func_name() -> impl Strategy<Value = (ArrayD<$tt>, ArrayD<$tt>)> {
                (1usize..25)
                    .prop_flat_map(|length| {
                        (
                            proptest::collection::vec(any::<$tt>(), length),
                            proptest::collection::vec(any::<$tt>(), length),
                        )
                    })
                    .prop_map(|(x, y)| {
                        let a = Array::from_shape_vec(IxDyn(&[x.len()]), x).unwrap();
                        let b = Array::from_shape_vec(IxDyn(&[y.len()]), y).unwrap();
                        (a, b)
                    })
                    .boxed()
            }
        };
    }

    pairwise_same_length!(pairwise_same_length64, u64);
    pairwise_same_length!(pairwise_same_length128, u128);

    proptest! {
        #[test]
        fn test_fuzzy_rep_mul64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_rep_mul64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_mul128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_rep_mul128(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_dot64((a,b) in pairwise_same_length64())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_rep_dot64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_dot128((a,b) in pairwise_same_length128())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_rep_dot128(a, b, target);
        }

    }

    macro_rules! rep_mir_binary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, ys: $tt, zs_mir: ArrayD<$tt>, zmir_s: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let mir = Mirrored3Placement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = HostRingTensor::from_raw_plc(xs, alice.clone());
                let target_rep_mir = HostRingTensor::from_raw_plc(zs_mir, alice.clone());
                let target_mir_rep = HostRingTensor::from_raw_plc(zmir_s, alice.clone());

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &x);
                let y_mir: MirTen<HostRingTensor<$tt>> =
                    mir.fill(&sess, ys.into(), &rep.shape(&sess, &x_shared));

                let result_rep_mir = rep.$test_func(&sess, &x_shared, &y_mir);
                let opened_result = alice.reveal(&sess, &result_rep_mir);

                assert_eq!(opened_result, target_rep_mir);

                let result_mir_rep = rep.$test_func(&sess, &y_mir, &x_shared);
                let opened_result = alice.reveal(&sess, &result_mir_rep);

                assert_eq!(opened_result, target_mir_rep);
            }
        };
    }

    rep_mir_binary_func_test!(test_rep_mir_add64, add<u64>);
    rep_mir_binary_func_test!(test_rep_mir_add128, add<u128>);
    rep_mir_binary_func_test!(test_rep_mir_sub64, sub<u64>);
    rep_mir_binary_func_test!(test_rep_mir_sub128, sub<u128>);

    #[test]
    fn test_rep_mir_add_64() {
        let x = array![0u64, 1, 2].into_dyn();
        let y = 2u64;
        let target_rep_mir = array![2u64, 3, 4].into_dyn();
        let target_mir_rep = array![2u64, 3, 4].into_dyn();
        test_rep_mir_add64(x, y, target_rep_mir, target_mir_rep);
    }

    #[test]
    fn test_rep_mir_add_128() {
        let x = array![0u128, 1, 2].into_dyn();
        let y = 2u128;
        let target_rep_mir = array![2u128, 3, 4].into_dyn();
        let target_mir_rep = array![2u128, 3, 4].into_dyn();
        test_rep_mir_add128(x, y, target_rep_mir, target_mir_rep);
    }

    #[test]
    fn test_rep_mir_sub_64() {
        let x = array![2u64, 3, 4].into_dyn();
        let y = 2u64;
        let target_rep_mir = array![0u64, 1, 2].into_dyn();
        let target_mir_rep = array![0u64, 18446744073709551615, 18446744073709551614].into_dyn();
        test_rep_mir_sub64(x, y, target_rep_mir, target_mir_rep);
    }

    #[test]
    fn test_rep_mir_sub_128() {
        let x = array![2u128, 3, 4].into_dyn();
        let y = 2u128;
        let target_rep_mir = array![0u128, 1, 2].into_dyn();
        let target_mir_rep = array![
            0u128,
            340282366920938463463374607431768211455,
            340282366920938463463374607431768211454
        ]
        .into_dyn();
        test_rep_mir_sub128(x, y, target_rep_mir, target_mir_rep);
    }

    macro_rules! rep_mir_mul_setup_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, ys: $tt, zs: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };
                let mir3 = Mirrored3Placement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = HostRingTensor::from_raw_plc(xs, alice.clone());
                let target = HostRingTensor::from_raw_plc(zs, alice.clone());

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &x);
                let y_mir: MirTen<HostRingTensor<$tt>> =
                    mir3.fill(&sess, ys.into(), &rep.shape(&sess, &x_shared));

                let result_rep_mir = rep.$test_func(&sess, &x_shared, &y_mir);
                let opened_result = alice.reveal(&sess, &result_rep_mir);

                assert_eq!(opened_result, target);

                let result_mir_rep = rep.$test_func(&sess, &y_mir, &x_shared);
                let opened_result = alice.reveal(&sess, &result_mir_rep);

                assert_eq!(opened_result, target);
            }
        };
    }

    rep_mir_mul_setup_func_test!(test_rep_mir_mul64, mul<u64>);
    rep_mir_mul_setup_func_test!(test_rep_mir_mul128, mul<u128>);

    #[test]
    fn test_rep_mir_mul_64() {
        let x = array![0u64, 1, 2].into_dyn();
        let y = 2u64;
        let target = array![0u64, 2, 4].into_dyn();
        test_rep_mir_mul64(x, y, target);
    }

    #[test]
    fn test_rep_mir_mul_128() {
        let x = array![0u128, 1, 2].into_dyn();
        let y = 2u128;
        let target = array![0u128, 2, 4].into_dyn();
        test_rep_mir_mul128(x, y, target);
    }

    macro_rules! rep_truncation_test {
        ($func_name:ident, $tt: ident) => {
            fn $func_name(xs: ArrayD<$tt>, amount: u32, ys: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let bob = HostPlacement {
                    owner: "bob".into(),
                };
                let carole = HostPlacement {
                    owner: "carole".into(),
                };

                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let sess = SyncSession::default();

                let alice_x1 = HostRingTensor::from_raw_plc(xs.clone(), alice.clone());
                let alice_rep = rep.share(&sess, &alice_x1);
                let alice_tr = rep.trunc_pr(&sess, amount, &alice_rep);
                let alice_open = alice.reveal(&sess, &alice_tr);

                let alice_y = HostRingTensor::from_raw_plc(ys.clone(), alice.clone());
                assert_eq!(alice_open.1, alice_y.1); // make sure placements are equal

                // truncation can be off by 1
                for (i, value) in alice_y.0.iter().enumerate() {
                    let diff = value - &alice_open.0[i];
                    assert!(
                        diff == std::num::Wrapping(1 as $tt)
                            || diff == std::num::Wrapping($tt::MAX)
                            || diff == std::num::Wrapping(0),
                        "difference = {}, lhs = {}, rhs = {}",
                        diff,
                        value,
                        &alice_open.0[i]
                    );
                }

                let bob_x1 = HostRingTensor::from_raw_plc(xs.clone(), bob.clone());
                let bob_rep = rep.share(&sess, &bob_x1);
                let bob_tr = rep.trunc_pr(&sess, amount, &bob_rep);
                let bob_open = bob.reveal(&sess, &bob_tr);

                let bob_y = HostRingTensor::from_raw_plc(ys.clone(), bob.clone());
                assert_eq!(bob_open.1, bob);

                for (i, value) in bob_y.0.iter().enumerate() {
                    let diff = value - &bob_open.0[i];
                    assert!(
                        diff == std::num::Wrapping(1 as $tt)
                            || diff == std::num::Wrapping($tt::MAX)
                            || diff == std::num::Wrapping(0),
                        "difference = {}, lhs = {}, rhs = {}",
                        diff,
                        value,
                        &bob_open.0[i]
                    );
                }

                let carole_x1 = HostRingTensor::from_raw_plc(xs.clone(), carole.clone());
                let carole_rep = rep.share(&sess, &carole_x1);
                let carole_tr = rep.trunc_pr(&sess, amount, &carole_rep);
                let carole_open = carole.reveal(&sess, &carole_tr);

                let carole_y = HostRingTensor::from_raw_plc(ys.clone(), bob.clone());
                assert_eq!(carole_open.1, carole);

                for (i, value) in carole_y.0.iter().enumerate() {
                    let diff = value - &carole_open.0[i];
                    assert!(
                        diff == std::num::Wrapping(1 as $tt)
                            || diff == std::num::Wrapping($tt::MAX)
                            || diff == std::num::Wrapping(0),
                        "difference = {}, lhs = {}, rhs = {}",
                        diff,
                        value,
                        &carole_open.0[i]
                    );
                }
            }
        };
    }

    rep_truncation_test!(test_rep_truncation64, u64);
    rep_truncation_test!(test_rep_truncation128, u128);

    #[rstest]
    #[case(array![1_u64, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        1,
        array![0_u64, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952].into_dyn())
    ]
    #[case(array![1_u64, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        62,
        array![0_u64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1].into_dyn())
    ]
    #[case(array![1_u64, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        61,
        array![0_u64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2].into_dyn())
    ]
    #[case(array![-10_i64 as u64].into_dyn(), 1, array![-5_i64 as u64].into_dyn())]
    #[case(array![-10_i64 as u64].into_dyn(), 0, array![-10_i64 as u64].into_dyn())]
    #[case(array![-1152921504606846976_i64 as u64].into_dyn(), 60, array![-1_i64 as u64].into_dyn())]
    fn test_rep_truncation_64(
        #[case] x: ArrayD<u64>,
        #[case] amount: u32,
        #[case] target: ArrayD<u64>,
    ) {
        test_rep_truncation64(x, amount, target);
    }

    #[rstest]
    #[case(array![1_u128, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        1,
        array![0_u128, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952].into_dyn())
    ]
    #[case(array![1_u128, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        62,
        array![0_u128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1].into_dyn())
    ]
    #[case(array![1_u128, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        61,
        array![0_u128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2].into_dyn())
    ]
    #[case(array![-10_i128 as u128].into_dyn(), 1, array![-5_i128 as u128].into_dyn())]
    #[case(array![-10_i128 as u128].into_dyn(), 0, array![-10_i128 as u128].into_dyn())]
    #[case(array![-1152921504606846976_i128 as u128].into_dyn(), 60, array![-1_i128 as u128].into_dyn())]
    fn test_rep_truncation_128(
        #[case] x: ArrayD<u128>,
        #[case] amount: u32,
        #[case] target: ArrayD<u128>,
    ) {
        test_rep_truncation128(x, amount, target);
    }

    fn any_bounded_u64() -> impl Strategy<Value = u64> {
        any::<u64>().prop_map(|x| (x >> 2) - 1)
    }

    fn any_bounded_u128() -> impl Strategy<Value = u128> {
        any::<u128>().prop_map(|x| (x >> 2) - 1)
    }

    proptest! {

        #[test]
        fn test_fuzzy_rep_trunc64(raw_vector in proptest::collection::vec(any_bounded_u64(), 1..5), amount in 0u32..62
        ) {
            let target = raw_vector.iter().map(|x| x >> amount).collect::<Vec<_>>();
            test_rep_truncation64(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), amount, Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }

        #[test]
        fn test_fuzzy_rep_trunc128(raw_vector in proptest::collection::vec(any_bounded_u128(), 1..5), amount in 0u32..126
        ) {
            let target = raw_vector.iter().map(|x| x >> amount).collect::<Vec<_>>();
            test_rep_truncation128(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), amount, Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }
    }

    macro_rules! rep_unary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = HostRingTensor::from_raw_plc(xs, alice.clone());

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &x);

                let result: RepTensor<HostRingTensor<$tt>> = rep.$test_func(&sess, &x_shared);
                let opened_result = alice.reveal(&sess, &result);
                assert_eq!(
                    opened_result,
                    HostRingTensor::from_raw_plc(zs, alice.clone())
                );
            }
        };
    }

    rep_unary_func_test!(test_rep_msb64, msb<u64>);
    rep_unary_func_test!(test_rep_msb128, msb<u128>);

    #[rstest]
    #[case(array![-10_i64 as u64, -100_i64 as u64, -200000_i64 as u64, 0, 1].into_dyn(), array![1_u64, 1, 1, 0, 0].into_dyn())]
    fn test_rep_msb_64(#[case] x: ArrayD<u64>, #[case] target: ArrayD<u64>) {
        test_rep_msb64(x, target);
    }

    #[rstest]
    #[case(array![-10_i128 as u128, -100_i128 as u128, -200000_i128 as u128, 0, 1].into_dyn(), array![1_u128, 1, 1, 0, 0].into_dyn())]
    fn test_rep_msb_128(#[case] x: ArrayD<u128>, #[case] target: ArrayD<u128>) {
        test_rep_msb128(x, target);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn test_fuzzy_rep_msb64(raw_vector in proptest::collection::vec(any::<i64>().prop_map(|x| x as u64), 1..5)) {
            let target = raw_vector.iter().map(|x|
                (*x as i64).is_negative() as u64
            ).collect::<Vec<_>>();
            test_rep_msb64(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }

        #[test]
        fn test_fuzzy_rep_msb128(raw_vector in proptest::collection::vec(any::<i128>().prop_map(|x| x as u128), 1..5)) {
            let target = raw_vector.iter().map(|x|
                (*x as i128).is_negative() as u128
            ).collect::<Vec<_>>();
            test_rep_msb128(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }
    }

    #[rstest]
    #[case(array![0_u8, 1, 0].into_dyn())]
    fn test_ring_inject(#[case] xs: ArrayD<u8>) {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x = HostBitTensor::from_raw_plc(xs.clone(), alice.clone());

        let sess = SyncSession::default();

        let x_shared = rep.share(&sess, &x);

        let x_ring64: ReplicatedRing64Tensor = rep.ring_inject(&sess, 0, &x_shared);
        let x_ring128: ReplicatedRing128Tensor = rep.ring_inject(&sess, 0, &x_shared);

        let target64 = HostRing64Tensor::from_raw_plc(xs.map(|x| *x as u64), alice.clone());
        let target128 = HostRing128Tensor::from_raw_plc(xs.map(|x| *x as u128), alice.clone());

        assert_eq!(alice.reveal(&sess, &x_ring64), target64);
        assert_eq!(alice.reveal(&sess, &x_ring128), target128);

        let shifted_x_ring64: ReplicatedRing64Tensor = rep.ring_inject(&sess, 20, &x_shared);
        assert_eq!(alice.reveal(&sess, &shifted_x_ring64), target64 << 20);
    }

    rep_unary_func_test!(test_rep_abs64, abs<u64>);
    rep_unary_func_test!(test_rep_abs128, abs<u128>);

    #[rstest]
    #[case(array![-10_i64 as u64, -100_i64 as u64, -200000_i64 as u64, 0, 1000].into_dyn(), array![10_u64, 100, 200000, 0, 1000].into_dyn())]
    fn test_rep_abs_64(#[case] x: ArrayD<u64>, #[case] target: ArrayD<u64>) {
        test_rep_abs64(x, target);
    }

    #[rstest]
    #[case(array![-10_i128 as u128, -100_i128 as u128, -200000_i128 as u128, 0, 1000].into_dyn(), array![10_u128, 100, 200000, 0, 1000].into_dyn())]
    fn test_rep_abs_128(#[case] x: ArrayD<u128>, #[case] target: ArrayD<u128>) {
        test_rep_abs128(x, target);
    }

    fn test_rep_bit_dec64(xs: ArrayD<u64>, zs: ArrayD<u8>) {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x = HostRingTensor::from_raw_plc(xs, alice.clone());

        let sess = SyncSession::default();

        let x_shared = rep.share(&sess, &x);

        let result: ReplicatedBitArray64 = rep.bit_decompose(&sess, &x_shared);
        let opened_result = alice.reveal(&sess, &result);
        assert_eq!(opened_result, HostBitArray::from_raw_plc(zs, alice));
    }

    #[rstest]
    #[case(array![1073741823].into_dyn(),
        array![
            [1_u8],[1],[1],[1],[1],[1],[1],[1],
            [1],[1],[1],[1],[1],[1],[1],[1],
            [1],[1],[1],[1],[1],[1],[1],[1],
            [1],[1],[1],[1],[1],[1],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],
        ].into_dyn()
    )]
    fn test_rep_bit_dec_64(#[case] x: ArrayD<u64>, #[case] y: ArrayD<u8>) {
        test_rep_bit_dec64(x, y);
    }

    macro_rules! rep_bit_compose_test {
        ($func_name:ident, $tt:ty) => {
            fn $func_name(xs: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let expected = xs.clone();

                let x = HostRingTensor::from_raw_plc(xs, alice.clone());

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &x);

                let decomposed = rep.bit_decompose(&sess, &x_shared);

                let composed = rep.bit_compose(&sess, &decomposed);

                let opened_result = alice.reveal(&sess, &composed);

                assert_eq!(opened_result, HostRingTensor::from_raw_plc(expected, alice));
            }
        };
    }

    rep_bit_compose_test!(test_rep_bit_compose64, u64);
    rep_bit_compose_test!(test_rep_bit_compose128, u128);

    #[rstest]
    #[case(array![1073741823i128 as u128, 0, 6].into_dyn())]
    fn test_rep_bit_compose_128(#[case] xs: ArrayD<u128>) {
        test_rep_bit_compose128(xs);
    }

    #[rstest]
    #[case(array![1073741823, 0, 6].into_dyn())]
    fn test_rep_bit_compose_64(#[case] x: ArrayD<u64>) {
        test_rep_bit_compose64(x);
    }

    #[test]
    fn test_bit_dec_different_plc() {
        let xs = array![1073741823].into_dyn();
        let zs = array![
            [1_u8],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        ]
        .into_dyn();

        let alice = HostPlacement {
            owner: "alice".into(),
        };

        let bob = HostPlacement {
            owner: "bob".into(),
        };

        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x = HostRingTensor::from_raw_plc(xs, bob);

        let sess = SyncSession::default();

        let x_shared = rep.share(&sess, &x);

        let result: ReplicatedBitArray64 = rep.bit_decompose(&sess, &x_shared);
        let opened_result = alice.reveal(&sess, &result);
        assert_eq!(opened_result, HostBitArray::from_raw_plc(zs, alice));
    }

    macro_rules! rep_prefix_op_bit_test {
        ($func_name:ident, $test_func: ident) => {
            fn $func_name(x: ArrayD<u64>, y_target: Vec<u8>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = HostRingTensor::from_raw_plc(x, alice.clone());
                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &x);
                let x_bits: ReplicatedBitArray64 = rep.bit_decompose(&sess, &x_shared);
                let x_bits_vec: Vec<ReplicatedBitTensor> =
                    (0..64).map(|i| rep.index(&sess, i, &x_bits)).collect();

                let out = rep.$test_func(&sess, x_bits_vec);

                for (i, el) in out.iter().enumerate() {
                    let b = alice.reveal(&sess, el);
                    assert_eq!(b.0[0], y_target[i]);
                }
            }
        };
    }

    rep_prefix_op_bit_test!(test_rep_prefix_or, prefix_or);
    rep_prefix_op_bit_test!(test_rep_prefix_and, prefix_and);

    #[test]
    fn test_prefix_or() {
        let x = array![1024u64].into_dyn();
        let y_target = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
        ];
        test_rep_prefix_or(x, y_target);
    }

    #[test]
    fn test_prefix_and() {
        let x = array![7u64].into_dyn();
        let y_target = vec![
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];
        test_rep_prefix_and(x, y_target);
    }

    macro_rules! rep_binary_func_test_bit {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<u8>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = HostRingTensor::from_raw_plc(xs, alice.clone());
                let y = HostRingTensor::from_raw_plc(ys, alice.clone());

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &x);
                let y_shared = rep.share(&sess, &y);

                let sum: RepTensor<HostBitTensor> = rep.$test_func(&sess, &x_shared, &y_shared);
                let opened_product = alice.reveal(&sess, &sum);
                assert_eq!(
                    opened_product,
                    HostBitTensor::from_raw_plc(zs, alice.clone())
                );
            }
        };
    }

    rep_binary_func_test_bit!(test_rep_lt64, less<u64>);
    rep_binary_func_test_bit!(test_rep_lt128, less<u128>);

    #[test]
    fn test_rep_lt_64() {
        let x = array![0u64, 1, 2, -1_i64 as u64, -2_i64 as u64].into_dyn();
        let y = array![
            -1_i64 as u64,
            -2_i64 as u64,
            3_u64,
            -1_i64 as u64,
            -1_i64 as u64
        ]
        .into_dyn();
        let target = array![0, 0, 1, 0, 1].into_dyn();
        test_rep_lt64(x, y, target);
    }

    #[test]
    fn test_rep_lt_128() {
        let x = array![0u128, 1, 2, -1_i128 as u128, -2_i128 as u128].into_dyn();
        let y = array![
            -1_i128 as u128,
            -2_i128 as u128,
            3_u128,
            -1_i128 as u128,
            -1_i128 as u128
        ]
        .into_dyn();
        let target = array![0, 0, 1, 0, 1].into_dyn();
        test_rep_lt128(x, y, target);
    }

    rep_binary_func_test_bit!(test_rep_gt64, greater_than<u64>);
    rep_binary_func_test_bit!(test_rep_gt128, greater_than<u128>);

    #[test]
    fn test_rep_gt_64() {
        let x = array![0u64, 1, 2, -1_i64 as u64, -2_i64 as u64, 2u64.pow(62)].into_dyn();
        let y = array![
            -1_i64 as u64,
            -2_i64 as u64,
            3_u64,
            -1_i64 as u64,
            -1_i64 as u64,
            (-4611686018427387904_i64 + 1) as u64 // -2^62+1
        ]
        .into_dyn();
        let target = array![1, 1, 0, 0, 0, 1].into_dyn();
        test_rep_gt64(x, y, target);
    }

    #[test]
    fn test_rep_gt_128() {
        let x = array![0u128, 1, 2, -1_i128 as u128, -2_i128 as u128].into_dyn();
        let y = array![
            -1_i128 as u128,
            -2_i128 as u128,
            3_u128,
            -1_i128 as u128,
            -1_i128 as u128
        ]
        .into_dyn();
        let target = array![1, 1, 0, 0, 0].into_dyn();
        test_rep_gt128(x, y, target);
    }
}
