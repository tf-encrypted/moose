//! Placements backed by replicated secret sharing
use crate::additive::{AdditiveRing128Tensor, AdditiveRing64Tensor, AdtTen};
use crate::computation::{
    AdditivePlacement, AdtToRepOp, CanonicalType, Constant, HostPlacement, KnownType, Placed,
    RepAbsOp, RepAddOp, RepDotOp, RepFillOp, RepFixedpointMeanOp, RepIndexAxisOp, RepMsbOp,
    RepMulOp, RepRevealOp, RepSetupOp, RepShareOp, RepShlOp, RepSubOp, RepSumOp, RepTruncPrOp,
    ReplicatedPlacement, RingInjectOp, ShapeOp, SymbolicType,
};
use crate::error::{Error, Result};
use crate::host::{
    AbstractHostFixedTensor, HostBitTensor, HostFixed128Tensor, HostFixed64Tensor,
    HostRing128Tensor, HostRing64Tensor, HostShape, RingSize,
};
use crate::kernels::{
    PlacementAbs, PlacementAdd, PlacementAdtToRep, PlacementAnd, PlacementBitExtract,
    PlacementDaBitProvider, PlacementDeriveSeed, PlacementDot, PlacementDotSetup, PlacementFill,
    PlacementIndex, PlacementKeyGen, PlacementMeanAsFixedpoint, PlacementMsb, PlacementMul,
    PlacementMulSetup, PlacementOnes, PlacementPlace, PlacementRepToAdt, PlacementReveal,
    PlacementRingInject, PlacementSampleUniformSeeded, PlacementSetupGen, PlacementShape,
    PlacementShareSetup, PlacementShl, PlacementShr, PlacementSub, PlacementSum, PlacementTruncPr,
    PlacementTruncPrProvider, PlacementZeros, Session, Tensor,
};
use crate::prim::{PrfKey, Seed, SyncKey};
use macros::with_context;
use serde::{Deserialize, Serialize};
use std::convert::{TryFrom, TryInto};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractReplicatedRingTensor<R> {
    pub shares: [[R; 2]; 3],
}

/// Replicated tensor over Z_{2^64}.
pub type ReplicatedRing64Tensor = AbstractReplicatedRingTensor<HostRing64Tensor>;

impl SymbolicType for ReplicatedRing64Tensor {
    type Type = Symbolic<AbstractReplicatedRingTensor<<HostRing64Tensor as SymbolicType>::Type>>;
}

/// Replicated tensor over Z_{2^128}.
pub type ReplicatedRing128Tensor = AbstractReplicatedRingTensor<HostRing128Tensor>;

impl SymbolicType for ReplicatedRing128Tensor {
    type Type = Symbolic<AbstractReplicatedRingTensor<<HostRing128Tensor as SymbolicType>::Type>>;
}

/// Replicated tensor over Z_2.
pub type ReplicatedBitTensor = AbstractReplicatedRingTensor<HostBitTensor>;

impl SymbolicType for ReplicatedBitTensor {
    type Type = Symbolic<AbstractReplicatedRingTensor<<HostBitTensor as SymbolicType>::Type>>;
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AbstractReplicatedFixedTensor<RepRingT>(pub RepRingT);

pub type ReplicatedFixed64Tensor = AbstractReplicatedFixedTensor<ReplicatedRing64Tensor>;

impl SymbolicType for ReplicatedFixed64Tensor {
    type Type =
        Symbolic<AbstractReplicatedFixedTensor<<ReplicatedRing64Tensor as SymbolicType>::Type>>;
}

pub type ReplicatedFixed128Tensor = AbstractReplicatedFixedTensor<ReplicatedRing128Tensor>;

impl SymbolicType for ReplicatedFixed128Tensor {
    type Type =
        Symbolic<AbstractReplicatedFixedTensor<<ReplicatedRing128Tensor as SymbolicType>::Type>>;
}

impl<HostRingT> From<AbstractReplicatedRingTensor<HostRingT>>
    for Symbolic<AbstractReplicatedRingTensor<HostRingT>>
where
    HostRingT: Placed<Placement = HostPlacement>,
{
    fn from(x: AbstractReplicatedRingTensor<HostRingT>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<RepRingT> From<AbstractReplicatedFixedTensor<RepRingT>>
    for Symbolic<AbstractReplicatedFixedTensor<RepRingT>>
where
    RepRingT: Placed<Placement = ReplicatedPlacement>,
{
    fn from(x: AbstractReplicatedFixedTensor<RepRingT>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<HostKeyT> TryFrom<Symbolic<AbstractReplicatedSetup<HostKeyT>>>
    for AbstractReplicatedSetup<HostKeyT>
where
    HostKeyT: Placed<Placement = HostPlacement>,
{
    type Error = Error;
    fn try_from(v: Symbolic<AbstractReplicatedSetup<HostKeyT>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl<HostShapeT> TryFrom<Symbolic<AbstractReplicatedShape<HostShapeT>>>
    for AbstractReplicatedShape<HostShapeT>
where
    HostShapeT: Placed<Placement = HostPlacement>,
{
    type Error = Error;
    fn try_from(v: Symbolic<AbstractReplicatedShape<HostShapeT>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl<HostRingT> TryFrom<Symbolic<AbstractReplicatedRingTensor<HostRingT>>>
    for AbstractReplicatedRingTensor<HostRingT>
where
    HostRingT: Placed<Placement = HostPlacement>,
{
    type Error = Error;
    fn try_from(
        v: Symbolic<AbstractReplicatedRingTensor<HostRingT>>,
    ) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl<RepRingT> TryFrom<Symbolic<AbstractReplicatedFixedTensor<RepRingT>>>
    for AbstractReplicatedFixedTensor<RepRingT>
where
    RepRingT: Placed<Placement = ReplicatedPlacement>,
{
    type Error = Error;
    fn try_from(
        v: Symbolic<AbstractReplicatedFixedTensor<RepRingT>>,
    ) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl<K> From<AbstractReplicatedSetup<K>> for Symbolic<AbstractReplicatedSetup<K>>
where
    K: Placed<Placement = HostPlacement>,
{
    fn from(x: AbstractReplicatedSetup<K>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<S> From<AbstractReplicatedShape<S>> for Symbolic<AbstractReplicatedShape<S>>
where
    S: Placed<Placement = HostPlacement>,
{
    fn from(x: AbstractReplicatedShape<S>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<RepRingT: Placed> Placed for AbstractReplicatedFixedTensor<RepRingT> {
    type Placement = RepRingT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.0.placement()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractReplicatedSetup<K> {
    pub keys: [[K; 2]; 3],
}

pub type ReplicatedSetup = AbstractReplicatedSetup<PrfKey>;

impl SymbolicType for ReplicatedSetup {
    type Type = Symbolic<AbstractReplicatedSetup<<PrfKey as SymbolicType>::Type>>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractReplicatedShape<S> {
    pub shapes: [S; 3],
}

pub type ReplicatedShape = AbstractReplicatedShape<HostShape>;

impl SymbolicType for ReplicatedShape {
    type Type = Symbolic<AbstractReplicatedShape<<HostShape as SymbolicType>::Type>>;
}

/// Type aliases to shorten out impl in replicated protocols
type RepTen<T> = AbstractReplicatedRingTensor<T>;

impl<R> Placed for RepTen<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let RepTen {
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

impl<K> Placed for AbstractReplicatedSetup<K>
where
    K: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let AbstractReplicatedSetup {
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

impl<S> Placed for AbstractReplicatedShape<S>
where
    S: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let AbstractReplicatedShape {
            shapes: [s0, s1, s2],
        } = self;

        let owner0 = s0.placement()?.owner;
        let owner1 = s1.placement()?.owner;
        let owner2 = s2.placement()?.owner;

        let owners = [owner0, owner1, owner2];
        Ok(ReplicatedPlacement { owners })
    }
}

impl<S: Session, R> PlacementPlace<S, RepTen<R>> for ReplicatedPlacement
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

modelled!(PlacementSetupGen::gen_setup, ReplicatedPlacement, () -> ReplicatedSetup, RepSetupOp);

kernel! {
    RepSetupOp,
    [
        (ReplicatedPlacement, () -> ReplicatedSetup => [hybrid] Self::kernel),
    ]
}

impl RepSetupOp {
    fn kernel<S: Session, K: Clone>(
        sess: &S,
        rep: &ReplicatedPlacement,
    ) -> AbstractReplicatedSetup<K>
    where
        HostPlacement: PlacementKeyGen<S, K>,
        HostPlacement: PlacementPlace<S, K>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let k0 = player0.gen_key(sess);
        let k1 = player1.gen_key(sess);
        let k2 = player2.gen_key(sess);

        AbstractReplicatedSetup {
            keys: [
                [
                    player0.place(sess, k0.clone()),
                    player0.place(sess, k1.clone()),
                ],
                [player1.place(sess, k1), player1.place(sess, k2.clone())],
                [player2.place(sess, k2), player2.place(sess, k0)],
            ],
        }
    }
}

modelled!(PlacementShareSetup::share, ReplicatedPlacement, (ReplicatedSetup, HostFixed64Tensor) -> ReplicatedFixed64Tensor, RepShareOp);
modelled!(PlacementShareSetup::share, ReplicatedPlacement, (ReplicatedSetup, HostFixed128Tensor) -> ReplicatedFixed128Tensor, RepShareOp);
modelled!(PlacementShareSetup::share, ReplicatedPlacement, (ReplicatedSetup, HostRing64Tensor) -> ReplicatedRing64Tensor, RepShareOp);
modelled!(PlacementShareSetup::share, ReplicatedPlacement, (ReplicatedSetup, HostRing128Tensor) -> ReplicatedRing128Tensor, RepShareOp);
modelled!(PlacementShareSetup::share, ReplicatedPlacement, (ReplicatedSetup, HostBitTensor) -> ReplicatedBitTensor, RepShareOp);

kernel! {
    RepShareOp,
    [
        (ReplicatedPlacement, (ReplicatedSetup, HostFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::fixed_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, HostFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::fixed_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, HostRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::ring_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, HostRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::ring_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, HostBitTensor) -> ReplicatedBitTensor => [hybrid] Self::ring_kernel),
    ]
}

impl RepShareOp {
    fn fixed_kernel<S: Session, SetupT, HostRingT, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        setup: SetupT,
        x: AbstractHostFixedTensor<HostRingT>,
    ) -> AbstractReplicatedFixedTensor<RepRingT>
    where
        ReplicatedPlacement: PlacementShareSetup<S, SetupT, HostRingT, RepRingT>,
    {
        AbstractReplicatedFixedTensor(plc.share(sess, &setup, &x.0))
    }

    fn ring_kernel<S: Session, ShapeT, SeedT, KeyT, RingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        setup: AbstractReplicatedSetup<KeyT>,
        x: RingT,
    ) -> RepTen<RingT>
    where
        RingT: Clone + Placed<Placement = HostPlacement>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        HostPlacement: PlacementSampleUniformSeeded<S, ShapeT, SeedT, RingT>,
        HostPlacement: PlacementZeros<S, ShapeT, RingT>,
        HostPlacement: PlacementDeriveSeed<S, KeyT, SeedT>,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementSub<S, RingT, RingT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<RingT>>,
    {
        let x_player = x.placement().unwrap();

        let AbstractReplicatedSetup {
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

        plc.place(sess, RepTen { shares })
    }
}

modelled!(PlacementReveal::reveal, HostPlacement, (ReplicatedFixed64Tensor) -> HostFixed64Tensor, RepRevealOp);
modelled!(PlacementReveal::reveal, HostPlacement, (ReplicatedFixed128Tensor) -> HostFixed128Tensor, RepRevealOp);
modelled!(PlacementReveal::reveal, HostPlacement, (ReplicatedRing64Tensor) -> HostRing64Tensor, RepRevealOp);
modelled!(PlacementReveal::reveal, HostPlacement, (ReplicatedRing128Tensor) -> HostRing128Tensor, RepRevealOp);
modelled!(PlacementReveal::reveal, HostPlacement, (ReplicatedBitTensor) -> HostBitTensor, RepRevealOp);

kernel! {
    RepRevealOp,
    [
        (HostPlacement, (ReplicatedFixed64Tensor) -> HostFixed64Tensor => [hybrid] Self::fixed_kernel),
        (HostPlacement, (ReplicatedFixed128Tensor) -> HostFixed128Tensor => [hybrid] Self::fixed_kernel),
        (HostPlacement, (ReplicatedRing64Tensor) -> HostRing64Tensor => [hybrid] Self::ring_kernel),
        (HostPlacement, (ReplicatedRing128Tensor) -> HostRing128Tensor => [hybrid] Self::ring_kernel),
        (HostPlacement, (ReplicatedBitTensor) -> HostBitTensor => [hybrid] Self::ring_kernel),
    ]
}

impl RepRevealOp {
    fn fixed_kernel<S: Session, RepRingT, HostRingT>(
        sess: &S,
        receiver: &HostPlacement,
        xe: AbstractReplicatedFixedTensor<RepRingT>,
    ) -> AbstractHostFixedTensor<HostRingT>
    where
        HostPlacement: PlacementReveal<S, RepRingT, HostRingT>,
    {
        let x = receiver.reveal(sess, &xe.0);
        AbstractHostFixedTensor(x)
    }

    fn ring_kernel<S: Session, R: Clone>(sess: &S, receiver: &HostPlacement, xe: RepTen<R>) -> R
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, R, R, R>,
    {
        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &xe;

        let (player0, player1, player2) = &xe.placement().unwrap().host_placements();

        match () {
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
        }
    }
}

modelled!(PlacementAdd::add, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (HostRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (HostRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (ReplicatedRing64Tensor, HostRing64Tensor) -> ReplicatedRing64Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (ReplicatedRing128Tensor, HostRing128Tensor) -> ReplicatedRing128Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor, RepAddOp);

kernel! {
    RepAddOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_rep_kernel),
        (ReplicatedPlacement, (HostRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::ring_rep_kernel),
        (ReplicatedPlacement, (HostRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, HostRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, HostRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [hybrid] Self::rep_rep_kernel),
    ]
}

impl RepAddOp {
    fn rep_rep_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<R>,
        y: RepTen<R>,
    ) -> RepTen<R>
    where
        HostPlacement: PlacementAdd<S, R, R, R>,
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

        RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn ring_rep_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: R,
        y: RepTen<R>,
    ) -> RepTen<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, R, R, R>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<R>>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let x_plc = x.placement().unwrap();

        let RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = y;

        let shares = match () {
            _ if x_plc == player0 => {
                // add x to y0
                [
                    [with_context!(player0, sess, x + y00), y10],
                    [y11, y21],
                    [y22, with_context!(player2, sess, x + y02)],
                ]
            }
            _ if x_plc == player1 => {
                // add x to y1
                [
                    [y00, with_context!(player0, sess, x + y10)],
                    [with_context!(player1, sess, x + y11), y21],
                    [y22, y02],
                ]
            }
            _ if x_plc == player2 => {
                // add x to y2
                [
                    [y00, y10],
                    [y11, with_context!(player1, sess, x + y21)],
                    [with_context!(player2, sess, x + y22), y02],
                ]
            }
            _ => {
                // add x to y0; we could randomize this
                [
                    [with_context!(player0, sess, x + y00), y10],
                    [y11, y21],
                    [y22, with_context!(player2, sess, x + y02)],
                ]
            }
        };

        rep.place(sess, RepTen { shares })
    }

    fn rep_ring_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<R>,
        y: R,
    ) -> RepTen<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, R, R, R>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<R>>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let y_plc = y.placement().unwrap();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let shares = match () {
            _ if y_plc == player0 => {
                // add y to x0
                [
                    [with_context!(player0, sess, x00 + y), x10],
                    [x11, x21],
                    [x22, with_context!(player2, sess, x02 + y)],
                ]
            }
            _ if y_plc == player1 => {
                // add y to x1
                [
                    [x00, with_context!(player0, sess, x10 + y)],
                    [with_context!(player1, sess, x11 + y), x21],
                    [x22, x02],
                ]
            }
            _ if y_plc == player2 => {
                // add y to x2
                [
                    [x00, x10],
                    [x11, with_context!(player1, sess, x21 + y)],
                    [with_context!(player2, sess, x22 + y), x02],
                ]
            }
            _ => {
                // add y to x0; we could randomize this
                [
                    [with_context!(player0, sess, x00 + y), x10],
                    [x11, x21],
                    [x22, with_context!(player2, sess, x02 + y)],
                ]
            }
        };

        rep.place(sess, RepTen { shares })
    }
}

modelled!(PlacementSub::sub, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (HostRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (HostRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (ReplicatedRing64Tensor, HostRing64Tensor) -> ReplicatedRing64Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (ReplicatedRing128Tensor, HostRing128Tensor) -> ReplicatedRing128Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor, RepSubOp);

kernel! {
    RepSubOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_rep_kernel),
        (ReplicatedPlacement, (HostRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::ring_rep_kernel),
        (ReplicatedPlacement, (HostRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, HostRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, HostRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [hybrid] Self::rep_rep_kernel),
    ]
}

impl RepSubOp {
    fn rep_rep_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<R>,
        y: RepTen<R>,
    ) -> RepTen<R>
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

        RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn ring_rep_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: R,
        y: RepTen<R>,
    ) -> RepTen<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<S, R, R, R>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<R>>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let x_plc = x.placement().unwrap();

        let RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = y;

        let shares = match () {
            _ if x_plc == player0 => {
                // sub y0 from x
                [
                    [with_context!(player0, sess, x - y00), y10],
                    [y11, y21],
                    [y22, with_context!(player2, sess, x - y02)],
                ]
            }
            _ if x_plc == player1 => {
                // sub y1 from x
                [
                    [y00, with_context!(player0, sess, x - y10)],
                    [with_context!(player1, sess, x - y11), y21],
                    [y22, y02],
                ]
            }
            _ if x_plc == player2 => {
                // sub y2 from x
                [
                    [y00, y10],
                    [y11, with_context!(player1, sess, x - y21)],
                    [with_context!(player2, sess, x - y22), y02],
                ]
            }
            _ => {
                // sub y0 from x; we could randomize this
                [
                    [with_context!(player0, sess, x - y00), y10],
                    [y11, y21],
                    [y22, with_context!(player2, sess, x - y02)],
                ]
            }
        };

        rep.place(sess, RepTen { shares })
    }

    fn rep_ring_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<R>,
        y: R,
    ) -> RepTen<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<S, R, R, R>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<R>>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let y_plc = y.placement().unwrap();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let shares = match () {
            _ if y_plc == player0 => {
                // sub y0 from x
                [
                    [with_context!(player0, sess, x00 - y), x10],
                    [x11, x21],
                    [x22, with_context!(player2, sess, x02 - y)],
                ]
            }
            _ if y_plc == player1 => {
                // sub y1 from x
                [
                    [x00, with_context!(player0, sess, x10 - y)],
                    [with_context!(player1, sess, x11 - y), x21],
                    [x22, x02],
                ]
            }
            _ if y_plc == player2 => {
                // sub y2 from x
                [
                    [x00, x10],
                    [x11, with_context!(player1, sess, x21 - y)],
                    [with_context!(player2, sess, x22 - y), x02],
                ]
            }
            _ => {
                // sub y0 from x; we could randomize this
                [
                    [with_context!(player0, sess, x00 - y), x10],
                    [x11, x21],
                    [x22, with_context!(player2, sess, x02 - y)],
                ]
            }
        };

        rep.place(sess, RepTen { shares })
    }
}

modelled!(PlacementMulSetup::mul_setup, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul_setup, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul_setup, ReplicatedPlacement, (ReplicatedSetup, HostRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul_setup, ReplicatedPlacement, (ReplicatedSetup, HostRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul_setup, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing64Tensor, HostRing64Tensor) -> ReplicatedRing64Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul_setup, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing128Tensor, HostRing128Tensor) -> ReplicatedRing128Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul_setup, ReplicatedPlacement, (ReplicatedSetup, ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor, RepMulOp);

kernel! {
    RepMulOp,
    [
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => [hybrid] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, HostRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, HostRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing64Tensor, HostRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing128Tensor, HostRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_ring_kernel),
    ]
}

impl RepMulOp {
    fn rep_rep_kernel<S: Session, RingT, KeyT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        setup: AbstractReplicatedSetup<KeyT>,
        x: RepTen<RingT>,
        y: RepTen<RingT>,
    ) -> RepTen<RingT>
    where
        RingT: Clone,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        ReplicatedPlacement: ZeroShareGen<S, KeyT, RingT, ShapeT>,
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
        let zero_shape = AbstractReplicatedShape {
            shapes: [s0, s1, s2],
        };

        let AbstractReplicatedZeroShare {
            alphas: [a0, a1, a2],
        } = rep.gen_zero_share(sess, &setup, &zero_shape);

        let z0 = with_context!(player0, sess, { v0 + a0 });
        let z1 = with_context!(player1, sess, { v1 + a1 });
        let z2 = with_context!(player2, sess, { v2 + a2 });

        rep.place(
            sess,
            RepTen {
                shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
            },
        )
    }

    fn ring_rep_kernel<S: Session, RingT, KeyT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<KeyT>,
        x: RingT,
        y: RepTen<RingT>,
    ) -> RepTen<RingT>
    where
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, sess, x * y00);
        let z10 = with_context!(player0, sess, x * y10);

        let z11 = with_context!(player1, sess, x * y11);
        let z21 = with_context!(player1, sess, x * y21);

        let z22 = with_context!(player2, sess, x * y22);
        let z02 = with_context!(player2, sess, x * y02);

        RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn rep_ring_kernel<S: Session, RingT, KeyT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<KeyT>,
        x: RepTen<RingT>,
        y: RingT,
    ) -> RepTen<RingT>
    where
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = with_context!(player0, sess, x00 * y);
        let z10 = with_context!(player0, sess, x10 * y);

        let z11 = with_context!(player1, sess, x11 * y);
        let z21 = with_context!(player1, sess, x21 * y);

        let z22 = with_context!(player2, sess, x22 * y);
        let z02 = with_context!(player2, sess, x02 * y);

        RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
}

modelled!(PlacementDotSetup::dot_setup, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepDotOp);
modelled!(PlacementDotSetup::dot_setup, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepDotOp);
modelled!(PlacementDotSetup::dot_setup, ReplicatedPlacement, (ReplicatedSetup, HostRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepDotOp);
modelled!(PlacementDotSetup::dot_setup, ReplicatedPlacement, (ReplicatedSetup, HostRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepDotOp);
modelled!(PlacementDotSetup::dot_setup, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing64Tensor, HostRing64Tensor) -> ReplicatedRing64Tensor, RepDotOp);
modelled!(PlacementDotSetup::dot_setup, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing128Tensor, HostRing128Tensor) -> ReplicatedRing128Tensor, RepDotOp);

kernel! {
    RepDotOp,
    [
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, HostRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, HostRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing64Tensor, HostRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing128Tensor, HostRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::rep_ring_kernel),
    ]
}

impl RepDotOp {
    fn rep_rep_kernel<S: Session, RingT, KeyT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        setup: AbstractReplicatedSetup<KeyT>,
        x: RepTen<RingT>,
        y: RepTen<RingT>,
    ) -> RepTen<RingT>
    where
        RingT: Clone,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementDot<S, RingT, RingT, RingT>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        ReplicatedPlacement: ZeroShareGen<S, KeyT, RingT, ShapeT>,
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
        let zero_shape = AbstractReplicatedShape {
            shapes: [s0, s1, s2],
        };

        let AbstractReplicatedZeroShare {
            alphas: [a0, a1, a2],
        } = rep.gen_zero_share(sess, &setup, &zero_shape);

        let z0 = with_context!(player0, sess, { v0 + a0 });
        let z1 = with_context!(player1, sess, { v1 + a1 });
        let z2 = with_context!(player2, sess, { v2 + a2 });

        rep.place(
            sess,
            RepTen {
                shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
            },
        )
    }

    fn ring_rep_kernel<S: Session, RingT, KeyT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<KeyT>,
        x: RingT,
        y: RepTen<RingT>,
    ) -> RepTen<RingT>
    where
        HostPlacement: PlacementDot<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, sess, dot(&x, y00));
        let z10 = with_context!(player0, sess, dot(&x, y10));

        let z11 = with_context!(player1, sess, dot(&x, y11));
        let z21 = with_context!(player1, sess, dot(&x, y21));

        let z22 = with_context!(player2, sess, dot(&x, y22));
        let z02 = with_context!(player2, sess, dot(&x, y02));

        RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn rep_ring_kernel<S: Session, RingT, KeyT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<KeyT>,
        x: RepTen<RingT>,
        y: RingT,
    ) -> RepTen<RingT>
    where
        HostPlacement: PlacementDot<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = with_context!(player0, sess, dot(x00, &y));
        let z10 = with_context!(player0, sess, dot(x10, &y));

        let z11 = with_context!(player1, sess, dot(x11, &y));
        let z21 = with_context!(player1, sess, dot(x21, &y));

        let z22 = with_context!(player2, sess, dot(x22, &y));
        let z02 = with_context!(player2, sess, dot(x02, &y));

        RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
}

modelled!(PlacementMeanAsFixedpoint::mean_as_fixedpoint, ReplicatedPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepFixedpointMeanOp);
modelled!(PlacementMeanAsFixedpoint::mean_as_fixedpoint, ReplicatedPlacement, attributes[axis: Option<u32>, scaling_base: u64, scaling_exp: u32] (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepFixedpointMeanOp);

kernel! {
    RepFixedpointMeanOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] attributes[axis, scaling_base, scaling_exp] Self::kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] attributes[axis, scaling_base, scaling_exp] Self::kernel),
    ]
}

impl RepFixedpointMeanOp {
    fn kernel<S: Session, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: RepTen<HostRingT>,
    ) -> RepTen<HostRingT>
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

        RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
}

modelled!(PlacementSum::sum, ReplicatedPlacement, attributes[axis: Option<u32>] (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepSumOp);
modelled!(PlacementSum::sum, ReplicatedPlacement, attributes[axis: Option<u32>] (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepSumOp);

kernel! {
    RepSumOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] attributes[axis] Self::kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] attributes[axis] Self::kernel),
    ]
}

impl RepSumOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: Option<u32>,
        x: RepTen<RingT>,
    ) -> RepTen<RingT>
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

        rep.place(
            sess,
            RepTen {
                shares: [[z00, z10], [z11, z21], [z22, z02]],
            },
        )
    }
}

// TODO(Morten) should we rename this as a shift?
modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[amount: u32] (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[amount: u32] (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepTruncPrOp);

kernel! {
    RepTruncPrOp,
    [
        (ReplicatedPlacement,  (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] attributes[amount] Self::kernel),
        (ReplicatedPlacement,  (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] attributes[amount] Self::kernel),
    ]
}

impl RepTruncPrOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        amount: u32,
        xe: RepTen<RingT>,
    ) -> st!(RepTen<RingT>, S)
    where
        RingT: Clone,
        RepTen<RingT>: Into<st!(RepTen<RingT>)>,
        st!(AdtTen<RingT>): TryInto<AdtTen<RingT>>,
        AdtTen<RingT>: Into<st!(AdtTen<RingT>)>,
        st!(AdtTen<RingT>): TryInto<AdtTen<RingT>>,

        AdtTen<RingT>: CanonicalType,
        <AdtTen<RingT> as CanonicalType>::Type: KnownType<S>,
        RepTen<RingT>: CanonicalType,
        <RepTen<RingT> as CanonicalType>::Type: KnownType<S>,

        AdditivePlacement: PlacementRepToAdt<S, st!(RepTen<RingT>), st!(AdtTen<RingT>)>,

        AdditivePlacement: PlacementTruncPrProvider<S, AdtTen<RingT>, AdtTen<RingT>>,

        ReplicatedPlacement: PlacementAdtToRep<S, st!(AdtTen<RingT>), st!(RepTen<RingT>)>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let adt = AdditivePlacement {
            owners: [player0.owner, player1.owner],
        };
        let provider = player2;

        let x_adt = adt.rep_to_adt(sess, &xe.into()).try_into().ok().unwrap();
        let y_adt = adt.trunc_pr(sess, amount as usize, &provider, &x_adt);
        rep.adt_to_rep(sess, &y_adt.into())
    }
}

use crate::host::AbstractHostRingTensor;
use crate::symbolic::Symbolic;

impl<T> CanonicalType for AbstractHostRingTensor<T> {
    type Type = AbstractHostRingTensor<T>;
}

impl<T> CanonicalType for Symbolic<AbstractHostRingTensor<T>> {
    type Type = AbstractHostRingTensor<T>;
}

impl<KeyT: CanonicalType + Placed<Placement = HostPlacement>> CanonicalType
    for Symbolic<AbstractReplicatedSetup<KeyT>>
{
    type Type = AbstractReplicatedSetup<<KeyT as CanonicalType>::Type>;
}

impl<ShapeT: CanonicalType> CanonicalType for AbstractReplicatedShape<ShapeT> {
    type Type = AbstractReplicatedShape<<ShapeT as CanonicalType>::Type>;
}

impl<ShapeT: CanonicalType + Placed<Placement = HostPlacement>> CanonicalType
    for Symbolic<AbstractReplicatedShape<ShapeT>>
{
    type Type = AbstractReplicatedShape<<ShapeT as CanonicalType>::Type>;
}

impl<RingT: CanonicalType> CanonicalType for AdtTen<RingT> {
    type Type = AdtTen<<RingT as CanonicalType>::Type>;
}

impl<RingT: CanonicalType + Placed<Placement = HostPlacement>> CanonicalType
    for Symbolic<AdtTen<RingT>>
{
    type Type = AdtTen<<RingT as CanonicalType>::Type>;
}

impl<RingT: CanonicalType> CanonicalType for RepTen<RingT> {
    type Type = RepTen<<RingT as CanonicalType>::Type>;
}

impl<RingT: CanonicalType + Placed<Placement = HostPlacement>> CanonicalType
    for Symbolic<RepTen<RingT>>
{
    type Type = RepTen<<RingT as CanonicalType>::Type>;
}

modelled!(PlacementAdtToRep::adt_to_rep, ReplicatedPlacement, (AdditiveRing64Tensor) -> ReplicatedRing64Tensor, AdtToRepOp);
modelled!(PlacementAdtToRep::adt_to_rep, ReplicatedPlacement, (AdditiveRing128Tensor) -> ReplicatedRing128Tensor, AdtToRepOp);

kernel! {
    AdtToRepOp,
    [
        (ReplicatedPlacement, (AdditiveRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::kernel),
        (ReplicatedPlacement, (AdditiveRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::kernel),
    ]
}

impl AdtToRepOp {
    fn kernel<S: Session, ShapeT, SeedT, KeyT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AdtTen<RingT>,
    ) -> RepTen<RingT>
    where
        RingT: Placed<Placement = HostPlacement> + Clone,
        AdtTen<RingT>: CanonicalType,
        <AdtTen<RingT> as CanonicalType>::Type: KnownType<S>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        HostPlacement: PlacementKeyGen<S, KeyT>,
        HostPlacement: PlacementSampleUniformSeeded<S, ShapeT, SeedT, RingT>,
        HostPlacement: PlacementDeriveSeed<S, KeyT, SeedT>,
        AdditivePlacement:
            PlacementSub<S, st!(AdtTen<RingT>, S), st!(AdtTen<RingT>, S), st!(AdtTen<RingT>, S)>,
        AdtTen<RingT>: Into<st!(AdtTen<RingT>, S)>,
        HostPlacement: PlacementReveal<S, st!(AdtTen<RingT>, S), RingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<RingT>>,
    {
        let AdtTen { shares: [x0, x1] } = &x;

        let adt = x.placement().unwrap();
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
            _ => unimplemented!(), // something is wrong in the protocol otherwise
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
        let c = adt_player0.reveal(sess, &adt.sub(sess, &x.into(), &y.into()));

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
        rep.place(sess, RepTen { shares })
    }
}

modelled!(PlacementFill::fill, ReplicatedPlacement, attributes[value: Constant] (ReplicatedShape) -> ReplicatedRing64Tensor, RepFillOp);
modelled!(PlacementFill::fill, ReplicatedPlacement, attributes[value: Constant] (ReplicatedShape) -> ReplicatedRing128Tensor, RepFillOp);
modelled!(PlacementFill::fill, ReplicatedPlacement, attributes[value: Constant] (ReplicatedShape) -> ReplicatedBitTensor, RepFillOp);

kernel! {
    RepFillOp,
    [
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedRing64Tensor => [hybrid] custom |op| {
                let value: u64 = match op.value {
                    Constant::Ring64(v) => v,
                    _ => unimplemented!()
                };
                assert!(value == 0 || value == 1);
                Box::new(move |sess, rep, rep_shape| {
                    Self::ring64_kernel(sess, rep, value, rep_shape)
                })
            }),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedRing128Tensor => [hybrid] custom |op| {
                let value: u128 = match op.value {
                    Constant::Ring128(v) => v,
                    _ => unimplemented!()
                };
                assert!(value == 0 || value == 1);
                Box::new(move |sess, rep, rep_shape| {
                    Self::ring128_kernel(sess, rep, value, rep_shape)
                })
        }),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedBitTensor => [hybrid] custom |op| {
                let value: u8 = match op.value {
                    Constant::Bit(v) => v,
                    _ => unimplemented!()
                };
                assert!(value == 0 || value == 1);
                Box::new(move |sess, rep, rep_shape| {
                    Self::bit_kernel(sess, rep, value, rep_shape)
                })
        }),
    ]
}

impl RepFillOp {
    fn bit_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        value: u8,
        rep_shape: AbstractReplicatedShape<ShapeT>,
    ) -> RepTen<RingT>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return PublicReplicatedTensor, but we don't have that type yet
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedShape {
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

        RepTen { shares }
    }

    fn ring64_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        value: u64,
        rep_shape: AbstractReplicatedShape<ShapeT>,
    ) -> RepTen<RingT>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return PublicReplicatedTensor, but we don't have that type yet
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedShape {
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

        RepTen { shares }
    }

    fn ring128_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        value: u128,
        rep_shape: AbstractReplicatedShape<ShapeT>,
    ) -> RepTen<RingT>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return PublicReplicatedTensor, but we don't have that type yet
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedShape {
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

        RepTen { shares }
    }
}

modelled!(PlacementShl::shl, ReplicatedPlacement, attributes[amount: usize] (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepShlOp);
modelled!(PlacementShl::shl, ReplicatedPlacement, attributes[amount: usize] (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepShlOp);

kernel! {
    RepShlOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] attributes[amount] Self::kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] attributes[amount] Self::kernel),
    ]
}

impl RepShlOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        amount: usize,
        x: RepTen<RingT>,
    ) -> RepTen<RingT>
    where
        HostPlacement: PlacementShl<S, RingT, RingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let AbstractReplicatedRingTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;
        let z00 = player0.shl(sess, amount, x00);
        let z10 = player0.shl(sess, amount, x10);

        let z11 = player1.shl(sess, amount, x11);
        let z21 = player1.shl(sess, amount, x21);

        let z22 = player2.shl(sess, amount, x22);
        let z02 = player2.shl(sess, amount, x02);

        RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
}

modelled!(PlacementIndex::index_axis, ReplicatedPlacement, attributes[axis: usize, index: usize] (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepIndexAxisOp);
modelled!(PlacementIndex::index_axis, ReplicatedPlacement, attributes[axis: usize, index: usize] (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepIndexAxisOp);
modelled!(PlacementIndex::index_axis, ReplicatedPlacement, attributes[axis: usize, index: usize] (ReplicatedBitTensor) -> ReplicatedBitTensor, RepIndexAxisOp);

kernel! {
    RepIndexAxisOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] attributes[axis, index] Self::kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] attributes[axis, index] Self::kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor) -> ReplicatedBitTensor => [hybrid] attributes[axis, index] Self::kernel),
    ]
}

impl RepIndexAxisOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        index: usize,
        x: RepTen<RingT>,
    ) -> RepTen<RingT>
    where
        HostPlacement: PlacementIndex<S, RingT, RingT>,
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

        RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
}

kernel! {
    RepMsbOp,
    [
        (ReplicatedPlacement,  (ReplicatedSetup, ReplicatedRing64Tensor) -> ReplicatedBitTensor => [hybrid] Self::bit_kernel),
        (ReplicatedPlacement,  (ReplicatedSetup, ReplicatedRing128Tensor) -> ReplicatedBitTensor => [hybrid] Self::bit_kernel),
        (ReplicatedPlacement,  (ReplicatedSetup, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::ring_kernel),
        (ReplicatedPlacement,  (ReplicatedSetup, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::ring_kernel),
    ]
}

modelled!(PlacementMsb::msb, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing64Tensor) -> ReplicatedBitTensor, RepMsbOp);
modelled!(PlacementMsb::msb, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing128Tensor) -> ReplicatedBitTensor, RepMsbOp);
modelled!(PlacementMsb::msb, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepMsbOp);
modelled!(PlacementMsb::msb, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepMsbOp);

impl RepMsbOp {
    fn bit_kernel<S: Session, SetupT, RingT, BitTensorT, ReplicatedBitTensorT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        setup: SetupT,
        x: RepTen<RingT>,
    ) -> ReplicatedBitTensorT
    where
        RingT: RingSize,
        BitTensorT: Clone,
        ReplicatedBitTensorT: From<RepTen<BitTensorT>>,
        ReplicatedBitTensorT: Clone,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: RingBitDecompose<S, RingT>,
        HostPlacement: PlacementBitExtract<S, RingT, BitTensorT>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        HostPlacement: PlacementFill<S, ShapeT, BitTensorT>,
        ReplicatedPlacement: PlacementShareSetup<S, SetupT, BitTensorT, ReplicatedBitTensorT>,
        ReplicatedPlacement: PlacementPlace<S, RepTen<RingT>>,
        ReplicatedPlacement: BinaryAdder<S, SetupT, ReplicatedBitTensorT>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let RepTen {
            shares: [[x00, x10], [x11, x21], [x22, _x02]],
        } = &x;

        let left = with_context!(player0, sess, x00 + x10);
        let left_ring_bs = player0.bit_decompose(sess, &left);

        let p0_zero = player0.fill(sess, 0_u8.into(), &player0.shape(sess, x00));
        let p1_zero = player1.fill(sess, 0_u8.into(), &player1.shape(sess, x11));
        let p2_zero = player2.fill(sess, 0_u8.into(), &player2.shape(sess, x22));

        // transform x2 into boolean sharing
        let x2_on_1: Vec<_> = player1
            .bit_decompose(sess, x21)
            .iter()
            .map(|item| player1.bit_extract(sess, 0, item))
            .collect();
        let x2_on_2: Vec<_> = player2
            .bit_decompose(sess, x22)
            .iter()
            .map(|item| player2.bit_extract(sess, 0, item))
            .collect();

        // bit-decompose bsl
        let bsl: Vec<_> = left_ring_bs
            .iter()
            .map(|item| player0.bit_extract(sess, 0, item))
            .collect();

        let rep_bsl: Vec<_> = bsl
            .iter()
            .map(|item| rep.share(sess, &setup, item))
            .collect();

        let rep_bsr: Vec<_> = (0..RingT::SIZE)
            .map(|i| {
                RepTen {
                    shares: [
                        [p0_zero.clone(), p0_zero.clone()],
                        [p1_zero.clone(), x2_on_1[i].clone()],
                        [x2_on_2[i].clone(), p2_zero.clone()],
                    ],
                }
                .into()
            })
            .collect();

        let bits = rep.binary_adder(sess, setup, rep_bsl, rep_bsr);
        bits[bits.len() - 1].clone()
    }
}

impl RepMsbOp {
    fn ring_kernel<S: Session, SetupT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        setup: SetupT,
        x: RepTen<RingT>,
    ) -> st!(RepTen<RingT>)
    where
        RepTen<RingT>: Into<st!(RepTen<RingT>)>,

        RepTen<RingT>: CanonicalType,
        <RepTen<RingT> as CanonicalType>::Type: KnownType<S>,
        RepTen<HostBitTensor>: KnownType<S>,

        ReplicatedPlacement: PlacementMsb<S, SetupT, st!(RepTen<RingT>), st!(ReplicatedBitTensor)>,
        ReplicatedPlacement: PlacementRingInject<S, st!(ReplicatedBitTensor), st!(RepTen<RingT>)>,
    {
        let x_bin = rep.msb(sess, &setup, &x.into());
        rep.ring_inject(sess, 0, &x_bin)
    }
}

modelled!(PlacementAbs::abs, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepAbsOp);
modelled!(PlacementAbs::abs, ReplicatedPlacement, (ReplicatedSetup, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepAbsOp);

kernel! {
    RepAbsOp,
    [
        (ReplicatedPlacement,  (ReplicatedSetup, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [hybrid] Self::kernel),
        (ReplicatedPlacement,  (ReplicatedSetup, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] Self::kernel),
    ]
}

impl RepAbsOp {
    fn kernel<S: Session, SetupT, RingT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        setup: SetupT,
        x: RepTen<RingT>,
    ) -> st!(RepTen<RingT>)
    where
        RepTen<RingT>: Into<st!(RepTen<RingT>)>,
        RepTen<RingT>: CanonicalType,
        <RepTen<RingT> as CanonicalType>::Type: KnownType<S>,

        RepTen<RingT>: Clone,

        RingT: Tensor<S>,
        RingT::Scalar: Into<Constant>,
        RingT::Scalar: From<u8>,

        ReplicatedPlacement: PlacementMsb<S, SetupT, st!(RepTen<RingT>), st!(RepTen<RingT>)>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, st!(RepTen<RingT>)>,
        ReplicatedPlacement: PlacementShape<S, st!(RepTen<RingT>), ShapeT>,
        ReplicatedPlacement: PlacementMulSetup<
            S,
            SetupT,
            st!(RepTen<RingT>),
            st!(RepTen<RingT>),
            st!(RepTen<RingT>),
        >,
        ReplicatedPlacement: PlacementShl<S, st!(RepTen<RingT>), st!(RepTen<RingT>)>,
        ReplicatedPlacement:
            PlacementSub<S, st!(RepTen<RingT>), st!(RepTen<RingT>), st!(RepTen<RingT>)>,
    {
        // TODO(Dragos) Remove un-necessary cloning due to st! macro
        let msb_ring: st!(RepTen<RingT>) = rep.msb(sess, &setup, &x.clone().into());
        let double = rep.shl(sess, 1, &msb_ring);

        let one_r = RingT::Scalar::from(1).into();
        let ones = rep.fill(sess, one_r, &rep.shape(sess, &msb_ring));
        let sign = rep.sub(sess, &ones, &double);

        rep.mul_setup(sess, &setup, &sign, &x.into())
    }
}

// TODO(Morten): might be able to return [R; R::SIZE] in the future, see https://github.com/rust-lang/rust/issues/60551
trait RingBitDecompose<S: Session, R> {
    fn bit_decompose(&self, sess: &S, x: &R) -> Vec<R>;
}

impl<S: Session, R> RingBitDecompose<S, R> for HostPlacement
where
    R: RingSize,
    HostShape: KnownType<S>,
    HostPlacement: PlacementOnes<S, cs!(HostShape), R>,
    HostPlacement: PlacementShape<S, R, cs!(HostShape)>,
    HostPlacement: PlacementShr<S, R, R>,
    HostPlacement: PlacementAnd<S, R, R, R>,
{
    fn bit_decompose(&self, sess: &S, x: &R) -> Vec<R> {
        let k = R::SIZE;
        let shape = self.shape(sess, x);
        let ones = self.ones(sess, &shape);
        (0..k)
            .map(|i| self.and(sess, &self.shr(sess, i, x), &ones))
            .collect()
    }
}

impl ShapeOp {
    pub(crate) fn rep_kernel<S: Session, RingT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTen<RingT>,
    ) -> AbstractReplicatedShape<ShapeT>
    where
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let RepTen {
            shares: [[x00, _x10], [x11, _x21], [x22, _x02]],
        } = &x;
        AbstractReplicatedShape {
            shapes: [
                player0.shape(sess, x00),
                player1.shape(sess, x11),
                player2.shape(sess, x22),
            ],
        }
    }
}

trait BinaryAdder<S: Session, SetupT, R> {
    fn binary_adder(&self, sess: &S, setup: SetupT, x: Vec<R>, y: Vec<R>) -> Vec<R>;
}

/// Binary addition protocol
impl<S: Session, SetupT, RT> BinaryAdder<S, SetupT, RT> for ReplicatedPlacement
where
    RT: Clone,
    RT: Placed<Placement = ReplicatedPlacement>,
    AbstractReplicatedShape<HostShape>: KnownType<S>,
    ReplicatedPlacement: PlacementMulSetup<S, SetupT, RT, RT, RT>,
    ReplicatedPlacement: PlacementAdd<S, RT, RT, RT>,
    ReplicatedPlacement: PlacementFill<S, st!(AbstractReplicatedShape<HostShape>), RT>,
    ReplicatedPlacement: PlacementShape<S, RT, st!(AbstractReplicatedShape<HostShape>)>,
{
    /// Binary addition protocol
    ///
    /// `x` and `y` are collections of tensors. The number of tensors matches the ring size (64 or 128)
    fn binary_adder(&self, sess: &S, setup: SetupT, x: Vec<RT>, y: Vec<RT>) -> Vec<RT> {
        #![allow(clippy::many_single_char_names)]

        assert_eq!(x.len(), y.len());
        assert!(!x.is_empty());

        let ring_size = x.len();
        let log_r = (ring_size as f64).log2() as usize; // we know that R = 64/128

        let rep = self;

        let bitwise_and = |a: &Vec<RT>, b: &Vec<RT>| -> Vec<RT> {
            assert!(a.len() == ring_size);
            assert!(b.len() == ring_size);
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| rep.mul_setup(sess, &setup, x, y))
                .collect()
        };

        let bitwise_xor = |a: &Vec<RT>, b: &Vec<RT>| -> Vec<RT> {
            assert!(a.len() == ring_size);
            assert!(b.len() == ring_size);
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| rep.add(sess, x, y))
                .collect()
        };
        // g is part of the generator set, p propagator set
        // A few helpful diagrams to understand what is happening here:
        // https://www.chessprogramming.org/Kogge-Stone_Algorithm or here: https://inst.eecs.berkeley.edu/~eecs151/sp19/files/lec20-adders.pdf

        // P[i:j] = propagate bits for positions [i...i+j]
        // G[i:j] = generator bits for positions [i...i+j]

        // consider we have inputs a, b to the P,G computing gate
        // P = P_a & P_b
        // G = G_b + G_a & P_b
        // C_{i+1} = G_i + P_i & C_i

        // P, G can be computed in a tree fashion, performing ops on chunks of len 2^i
        // Note the first level is computed as P0 = x ^ y, G0 = x & y;

        // Perform `g = x * y` for every tensor
        let mut g = bitwise_and(&x, &y);

        // Perform `p_store = x + y` (just a helper to avoid compute bitwise_xor() twice)
        let p_store = bitwise_xor(&x, &y);

        let rep_shape = rep.shape(sess, &x[0]);

        // We will need tensors of zeros later
        let zero = rep.fill(sess, Constant::Bit(0), &rep_shape);

        let mut p = p_store.clone();

        // computes a >> amount
        let rotate_left = |a: &Vec<RT>, amount: usize| -> Vec<RT> {
            assert!(amount <= a.len());
            (0..a.len())
                .map(|i| {
                    if i < a.len() - amount {
                        a[amount + i].clone()
                    } else {
                        zero.clone()
                    }
                })
                .collect()
        };

        // For chunk of length 2^i...
        for i in 0..log_r {
            // Compute g1 and p1, zeroed out up until the current bit
            let g1 = rotate_left(&g, 1 << i);
            let p1 = rotate_left(&p, 1 << i);

            // `p_and_g = p1 * g1` for every tensor
            let p_and_g = bitwise_and(&p1, &g1);

            // Update `g = g + p_and_g`
            g = bitwise_xor(&g, &p_and_g);

            // update `p = p * p1`
            p = bitwise_and(&p, &p1);
        }

        // c is a copy of g with the first tensor (corresponding to the first bit) zeroed out
        let c = rotate_left(&g, 1);
        // final result is `z = c + p_store` (which is the original `x + y`)
        bitwise_xor(&c, &p_store)
    }
}

impl RingInjectOp {
    pub(crate) fn rep_kernel<S: Session, RingT, ReplicatedBitT, BitT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        bit_idx: usize,
        x: ReplicatedBitT,
    ) -> RepTen<RingT>
    where
        ReplicatedPlacement: PlacementShape<S, ReplicatedBitT, AbstractReplicatedShape<ShapeT>>,
        ReplicatedPlacement: PlacementAdtToRep<S, AdtTen<RingT>, RepTen<RingT>>,
        AdditivePlacement: PlacementDaBitProvider<S, ShapeT, AdtTen<RingT>, AdtTen<BitT>>,
        AdditivePlacement: PlacementRepToAdt<S, ReplicatedBitT, AdtTen<BitT>>,
        AdditivePlacement: PlacementAdd<S, AdtTen<BitT>, AdtTen<BitT>, AdtTen<BitT>>,
        AdditivePlacement: PlacementAdd<S, AdtTen<RingT>, RingT, AdtTen<RingT>>,
        AdditivePlacement: PlacementMul<S, AdtTen<RingT>, RingT, AdtTen<RingT>>,
        AdditivePlacement: PlacementSub<S, AdtTen<RingT>, AdtTen<RingT>, AdtTen<RingT>>,
        AdditivePlacement: PlacementShl<S, AdtTen<RingT>, AdtTen<RingT>>,
        HostPlacement: PlacementReveal<S, AdtTen<BitT>, BitT>,
        HostPlacement: PlacementRingInject<S, BitT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let adt = AdditivePlacement {
            owners: [player0.clone().owner, player1.owner],
        };
        let provider = player2;

        let x_shape = rep.shape(sess, &x);
        let AbstractReplicatedShape {
            shapes: [s0, _, s_provider],
        } = x_shape;
        // One could think to wrap this up into an additive shape for Hosts@(P0, P2)
        // but the additive placement that generates a dabit is Hosts@(P0, P1)
        // to avoid confusion the API corresponding gen_dabit takes two input shapes
        // 1) s_provider - provider (dealer) shape
        // 2) s_0 - shape that corresponds to the party expanding the seeds received from provider.

        let (b_ring, b_bin): (AdtTen<RingT>, AdtTen<BitT>) =
            adt.gen_dabit(sess, s_provider, s0, &provider);

        let x_adt = adt.rep_to_adt(sess, &x);

        let c = with_context!(adt, sess, x_adt + b_bin);
        let c_open = player0.reveal(sess, &c);
        let c_ring = player0.ring_inject(sess, 0, &c_open);
        let x_adt_ring = with_context!(
            adt,
            sess,
            b_ring + c_ring - b_ring * c_ring - b_ring * c_ring
        );
        let shifted_x_adt = adt.shl(sess, bit_idx, &x_adt_ring);
        rep.adt_to_rep(sess, &shifted_x_adt)
    }
}

struct AbstractReplicatedSeeds<T> {
    seeds: [[T; 2]; 3],
}

trait ReplicatedSeedsGen<S: Session, KeyT, SeedT> {
    fn gen_seeds(
        &self,
        ctx: &S,
        setup: &AbstractReplicatedSetup<KeyT>,
    ) -> AbstractReplicatedSeeds<SeedT>;
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
        setup: &AbstractReplicatedSetup<cs!(PrfKey)>,
    ) -> AbstractReplicatedSeeds<cs!(Seed)> {
        let (player0, player1, player2) = self.host_placements();

        let AbstractReplicatedSetup {
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
struct AbstractReplicatedZeroShare<R> {
    alphas: [R; 3],
}

trait ZeroShareGen<S: Session, KeyT, RingT, ShapeT> {
    fn gen_zero_share(
        &self,
        sess: &S,
        setup: &AbstractReplicatedSetup<KeyT>,
        shape: &AbstractReplicatedShape<ShapeT>,
    ) -> AbstractReplicatedZeroShare<RingT>;
}

impl<S: Session, RingT> ZeroShareGen<S, cs!(PrfKey), RingT, cs!(HostShape)> for ReplicatedPlacement
where
    PrfKey: KnownType<S>,
    Seed: KnownType<S>,
    HostShape: KnownType<S>,
    HostPlacement: PlacementSampleUniformSeeded<S, cs!(HostShape), cs!(Seed), RingT>,
    HostPlacement: PlacementSub<S, RingT, RingT, RingT>,
    ReplicatedPlacement: ReplicatedSeedsGen<S, cs!(PrfKey), cs!(Seed)>,
{
    fn gen_zero_share(
        &self,
        sess: &S,
        setup: &AbstractReplicatedSetup<cs!(PrfKey)>,
        shape: &AbstractReplicatedShape<cs!(HostShape)>,
    ) -> AbstractReplicatedZeroShare<RingT> {
        let (player0, player1, player2) = self.host_placements();

        let AbstractReplicatedShape {
            shapes: [shape0, shape1, shape2],
        } = shape;

        let AbstractReplicatedSeeds {
            seeds: [[s00, s10], [s11, s21], [s22, s02]],
        } = &self.gen_seeds(sess, setup);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixedpoint::Convert;
    use crate::host::AbstractHostRingTensor;
    use crate::host::FromRawPlc;
    use crate::kernels::{
        PlacementFixedpointEncode, PlacementRingFixedpointDecode, PlacementRingFixedpointEncode,
        SyncSession,
    };
    use ndarray::array;
    use proptest::prelude::*;

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
                AbstractHostRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "alice".into(),
                    },
                ),
                AbstractHostRingTensor::from_raw_plc(
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
                AbstractHostRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "bob".into(),
                    },
                ),
                AbstractHostRingTensor::from_raw_plc(
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
                AbstractHostRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "david".into(),
                    },
                ),
                AbstractHostRingTensor::from_raw_plc(
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
                AbstractHostRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "alice".into(),
                    },
                ),
                AbstractHostRingTensor::from_raw_plc(
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
        let setup = rep.gen_setup(&sess);

        let scaling_base = 2;
        let scaling_exp = 24;

        let x = crate::host::HostFloat64Tensor::from_raw_plc(
            array![1.0, 2.0, 3.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
            alice.clone(),
        );
        let x = alice.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &x);
        let x_shared = rep.share(&sess, &setup, &x);

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
    fn test_rep_sum() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x = AbstractHostRingTensor::from_raw_plc(array![1u64, 2, 3], alice.clone());

        let sess = SyncSession::default();
        let setup = rep.gen_setup(&sess);

        let x_shared = rep.share(&sess, &setup, &x);

        let sum = rep.sum(&sess, None, &x_shared);
        let opened_result = alice.reveal(&sess, &sum);

        assert_eq!(6, opened_result.0[[]].0);
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

                let x = AbstractHostRingTensor::from_raw_plc(xs, alice.clone());
                let y = AbstractHostRingTensor::from_raw_plc(ys, alice.clone());

                let sess = SyncSession::default();
                let setup = rep.gen_setup(&sess);

                let x_shared = rep.share(&sess, &setup, &x);
                let y_shared = rep.share(&sess, &setup, &y);

                let sum = rep.add(&sess, &x_shared, &y_shared);
                let opened_sum = alice.reveal(&sess, &sum);
                assert_eq!(
                    opened_sum,
                    AbstractHostRingTensor::from_raw_plc(zs, alice.clone())
                );
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

                let x = AbstractHostRingTensor::from_raw_plc(xs, alice.clone());
                let y = AbstractHostRingTensor::from_raw_plc(ys, alice.clone());

                let sess = SyncSession::default();
                let setup = rep.gen_setup(&sess);

                let x_shared = rep.share(&sess, &setup, &x);
                let y_shared = rep.share(&sess, &setup, &y);

                let sum = rep.$test_func(&sess, &setup, &x_shared, &y_shared);
                let opened_product = alice.reveal(&sess, &sum);
                assert_eq!(
                    opened_product,
                    AbstractHostRingTensor::from_raw_plc(zs, alice.clone())
                );
            }
        };
    }

    rep_binary_func_test!(test_rep_mul64, mul_setup<u64>);
    rep_binary_func_test!(test_rep_mul128, mul_setup<u128>);
    rep_binary_func_test!(test_rep_dot64, dot_setup<u64>);
    rep_binary_func_test!(test_rep_dot128, dot_setup<u128>);

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
                let setup = rep.gen_setup(&sess);

                let alice_x1 = AbstractHostRingTensor::from_raw_plc(xs.clone(), alice.clone());
                let alice_rep = rep.share(&sess, &setup, &alice_x1);
                let alice_tr = rep.trunc_pr(&sess, amount, &alice_rep);
                let alice_open = alice.reveal(&sess, &alice_tr);

                let alice_y = AbstractHostRingTensor::from_raw_plc(ys.clone(), alice.clone());
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

                let bob_x1 = AbstractHostRingTensor::from_raw_plc(xs.clone(), bob.clone());
                let bob_rep = rep.share(&sess, &setup, &bob_x1);
                let bob_tr = rep.trunc_pr(&sess, amount, &bob_rep);
                let bob_open = bob.reveal(&sess, &bob_tr);

                let bob_y = AbstractHostRingTensor::from_raw_plc(ys.clone(), bob.clone());
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

                let carole_x1 = AbstractHostRingTensor::from_raw_plc(xs.clone(), carole.clone());
                let carole_rep = rep.share(&sess, &setup, &carole_x1);
                let carole_tr = rep.trunc_pr(&sess, amount, &carole_rep);
                let carole_open = carole.reveal(&sess, &carole_tr);

                let carole_y = AbstractHostRingTensor::from_raw_plc(ys.clone(), bob.clone());
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

                let x = AbstractHostRingTensor::from_raw_plc(xs, alice.clone());

                let sess = SyncSession::default();
                let setup = rep.gen_setup(&sess);

                let x_shared = rep.share(&sess, &setup, &x);

                let result: AbstractReplicatedRingTensor<AbstractHostRingTensor<$tt>> =
                    rep.$test_func(&sess, &setup, &x_shared);
                let opened_result = alice.reveal(&sess, &result);
                assert_eq!(
                    opened_result,
                    AbstractHostRingTensor::from_raw_plc(zs, alice.clone())
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
        let setup = rep.gen_setup(&sess);

        let x_shared = rep.share(&sess, &setup, &x);

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
}
