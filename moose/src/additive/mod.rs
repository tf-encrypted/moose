//! Placements backed by additive secret sharing
use crate::computation::{
    AdditivePlacement, AdtAddOp, AdtFillOp, AdtMulOp, AdtRevealOp, AdtShlOp, AdtSubOp,
    CanonicalType, Constant, HostPlacement, KnownType, Placed, RepToAdtOp, ShapeOp,
};
use crate::error::Result;
use crate::host::{HostBitTensor, HostRing128Tensor, HostRing64Tensor, HostShape};
use crate::kernels::*;
use crate::prim::{PrfKey, Seed, SyncKey};
use crate::replicated::{
    AbstractReplicatedRingTensor, ReplicatedBitTensor, ReplicatedRing128Tensor,
    ReplicatedRing64Tensor,
};
use crate::{Const, Ring};
use macros::with_context;
use serde::{Deserialize, Serialize};
use std::convert::TryInto;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdtTensor<HostT> {
    pub shares: [HostT; 2],
}

moose_type!(AdditiveRing64Tensor = AdtTensor<HostRing64Tensor>);
moose_type!(AdditiveRing128Tensor = AdtTensor<HostRing128Tensor>);
moose_type!(AdditiveBitTensor = AdtTensor<HostBitTensor>);

impl<R> Placed for AdtTensor<R>
where
    R: Placed<Placement = HostPlacement>,
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

impl<S: Session, R> PlacementPlace<S, AdtTensor<R>> for AdditivePlacement
where
    AdtTensor<R>: Placed<Placement = AdditivePlacement>,
    HostPlacement: PlacementPlace<S, R>,
{
    fn place(&self, sess: &S, x: AdtTensor<R>) -> AdtTensor<R> {
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
pub struct AbstractAdditiveShape<S> {
    pub shapes: [S; 2],
}

moose_type!(AdditiveShape = AbstractAdditiveShape<HostShape>);

impl<S> Placed for AbstractAdditiveShape<S>
where
    S: Placed<Placement = HostPlacement>,
{
    type Placement = AdditivePlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let AbstractAdditiveShape { shapes: [s0, s1] } = self;

        let owner0 = s0.placement()?.owner;
        let owner1 = s1.placement()?.owner;

        let owners = [owner0, owner1];
        Ok(AdditivePlacement { owners })
    }
}

impl ShapeOp {
    pub(crate) fn adt_kernel<S: Session, HostT, ShapeT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<HostT>,
    ) -> Result<AbstractAdditiveShape<ShapeT>>
    where
        HostPlacement: PlacementShape<S, HostT, ShapeT>,
    {
        let (player0, player1) = adt.host_placements();
        let AdtTensor { shares: [x0, x1] } = &x;
        Ok(AbstractAdditiveShape {
            shapes: [player0.shape(sess, x0), player1.shape(sess, x1)],
        })
    }
}

modelled_kernel! {
    PlacementFill::fill, AdtFillOp{value: Constant},
    [
        (AdditivePlacement, (HostShape) -> AdditiveRing64Tensor => [hybrid] Self::host_kernel),
        (AdditivePlacement, (HostShape) -> AdditiveRing128Tensor => [hybrid] Self::host_kernel),
        (AdditivePlacement, (AdditiveShape) -> AdditiveRing64Tensor => [concrete] Self::adt_kernel),
        (AdditivePlacement, (AdditiveShape) -> AdditiveRing128Tensor => [concrete] Self::adt_kernel),
    ]
}

impl AdtFillOp {
    fn host_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        plc: &AdditivePlacement,
        value: Constant,
        shape: ShapeT,
    ) -> Result<AdtTensor<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return Mirrored2Tensor, but we don't have that type yet

        let (player0, player1) = plc.host_placements();

        let shares = [
            player0.fill(sess, value, &shape),
            player1.fill(sess, Constant::Ring64(0), &shape),
        ];
        Ok(AdtTensor { shares })
    }

    fn adt_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        plc: &AdditivePlacement,
        value: Constant,
        shape: AbstractAdditiveShape<ShapeT>,
    ) -> Result<AdtTensor<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return Mirrored2Tensor, but we don't have that type yet

        let AbstractAdditiveShape {
            shapes: [shape0, shape1],
        } = &shape;

        let (player0, player1) = plc.host_placements();

        let shares = [
            player0.fill(sess, value, shape0),
            player1.fill(sess, Constant::Ring64(0), shape1),
        ];
        Ok(AdtTensor { shares })
    }
}

modelled_kernel! {
    PlacementReveal::reveal, AdtRevealOp,
    [
        (HostPlacement, (AdditiveRing64Tensor) -> HostRing64Tensor => [hybrid] Self::kernel),
        (HostPlacement, (AdditiveRing128Tensor) -> HostRing128Tensor => [hybrid] Self::kernel),
        (HostPlacement, (AdditiveBitTensor) -> HostBitTensor => [hybrid] Self::kernel),
    ]
}

impl AdtRevealOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        plc: &HostPlacement,
        xe: AdtTensor<RingT>,
    ) -> Result<RingT>
    where
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
    {
        let AdtTensor { shares: [x0, x1] } = &xe;
        Ok(with_context!(plc, sess, x0 + x1))
    }
}

modelled_kernel! {
    PlacementAdd::add, AdtAddOp,
    [
        (AdditivePlacement, (AdditiveRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [concrete] Self::adt_adt_kernel),
        // TODO(Morten) replace host tensors with mirrored tensors in the below
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::adt_ring_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::adt_ring_kernel),
        (AdditivePlacement, (AdditiveBitTensor, HostBitTensor) -> AdditiveBitTensor => [hybrid] Self::adt_ring_kernel),
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::ring_adt_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::ring_adt_kernel),
        (AdditivePlacement, (HostBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [hybrid] Self::ring_adt_kernel),
    ]
}

impl AdtAddOp {
    fn adt_adt_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<HostRingT>,
        y: AdtTensor<HostRingT>,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, player1) = adt.host_placements();

        let AdtTensor { shares: [x0, x1] } = &x;
        let AdtTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x0 + y0);
        let z1 = with_context!(player1, sess, x1 + y1);

        Ok(AdtTensor { shares: [z0, z1] })
    }

    fn adt_ring_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<HostRingT>,
        y: HostRingT,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
        AdditivePlacement: PlacementPlace<S, AdtTensor<HostRingT>>,
    {
        let (player0, player1) = adt.host_placements();
        let y_plc = y.placement()?;

        let AdtTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, sess, x0 + y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, sess, x1 + y)],
            _ => [with_context!(player0, sess, x0 + y), x1],
        };
        Ok(adt.place(sess, AdtTensor { shares }))
    }

    fn ring_adt_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: HostRingT,
        y: AdtTensor<HostRingT>,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
        AdditivePlacement: PlacementPlace<S, AdtTensor<HostRingT>>,
    {
        let (player0, player1) = adt.host_placements();
        let x_plc = x.placement()?;

        let AdtTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, sess, y0 + x), y1],
            _ if x_plc == player1 => [y0, with_context!(player1, sess, x + y1)],
            _ => [with_context!(player0, sess, x + y0), y1],
        };
        Ok(adt.place(sess, AdtTensor { shares }))
    }
}

modelled_kernel! {
    PlacementSub::sub, AdtSubOp,
    [
        (AdditivePlacement, (AdditiveRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [concrete] Self::adt_adt_kernel),
        // TODO(Morten) replace host tensors with mirrored tensors in the below
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::adt_ring_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::adt_ring_kernel),
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::ring_adt_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::ring_adt_kernel),
    ]
}

impl AdtSubOp {
    fn adt_adt_kernel<S: Session, R>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<R>,
        y: AdtTensor<R>,
    ) -> Result<AdtTensor<R>>
    where
        HostPlacement: PlacementSub<S, R, R, R>,
    {
        let (player0, player1) = adt.host_placements();

        let AdtTensor { shares: [x0, x1] } = &x;
        let AdtTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x0 - y0);
        let z1 = with_context!(player1, sess, x1 - y1);

        Ok(AdtTensor { shares: [z0, z1] })
    }

    fn adt_ring_kernel<S: Session, R>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<R>,
        y: R,
    ) -> Result<AdtTensor<R>>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<S, R, R, R>,
        AdditivePlacement: PlacementPlace<S, AdtTensor<R>>,
    {
        let (player0, player1) = adt.host_placements();
        let y_plc = y.placement()?;

        let AdtTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, sess, x0 - y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, sess, x1 - y)],
            _ => [with_context!(player0, sess, x0 - y), x1],
        };
        Ok(adt.place(sess, AdtTensor { shares }))
    }

    fn ring_adt_kernel<S: Session, R>(
        sess: &S,
        adt: &AdditivePlacement,
        x: R,
        y: AdtTensor<R>,
    ) -> Result<AdtTensor<R>>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<S, R, R, R>,
        HostPlacement: PlacementNeg<S, R, R>,
        AdditivePlacement: PlacementPlace<S, AdtTensor<R>>,
    {
        let (player0, player1) = adt.host_placements();
        let x_plc = x.placement()?;

        let AdtTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, sess, x - y0), player1.neg(sess, &y1)],
            _ if x_plc == player1 => [player0.neg(sess, &y0), with_context!(player1, sess, x - y1)],
            _ => [with_context!(player0, sess, x - y0), player1.neg(sess, &y1)],
        };
        Ok(adt.place(sess, AdtTensor { shares }))
    }
}

modelled_kernel! {
    PlacementMul::mul, AdtMulOp,
    [
        // TODO(Morten) replace host tensors with mirrored tensors in the below
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::ring_adt_kernel),
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::adt_ring_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::adt_ring_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::ring_adt_kernel),
        (AdditivePlacement, (AdditiveBitTensor, HostBitTensor) -> AdditiveBitTensor => [hybrid] Self::adt_ring_kernel),
        (AdditivePlacement, (HostBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [hybrid] Self::ring_adt_kernel),

    ]
}

impl AdtMulOp {
    fn ring_adt_kernel<S: Session, R>(
        sess: &S,
        adt: &AdditivePlacement,
        x: R,
        y: AdtTensor<R>,
    ) -> Result<AdtTensor<R>>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<S, R, R, R>,
    {
        let (player0, player1) = adt.host_placements();

        let AdtTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x * y0);
        let z1 = with_context!(player1, sess, x * y1);

        Ok(AdtTensor { shares: [z0, z1] })
    }

    fn adt_ring_kernel<S: Session, R>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<R>,
        y: R,
    ) -> Result<AdtTensor<R>>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<S, R, R, R>,
    {
        let (player0, player1) = adt.host_placements();

        let AdtTensor { shares: [x0, x1] } = &x;

        let z0 = with_context!(player0, sess, x0 * y);
        let z1 = with_context!(player1, sess, x1 * y);

        Ok(AdtTensor { shares: [z0, z1] })
    }
}

modelled_kernel! {
    PlacementShl::shl, AdtShlOp{amount: usize},
    [
        (AdditivePlacement, (AdditiveRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::kernel),
        (AdditivePlacement, (AdditiveRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::kernel),
    ]
}

impl AdtShlOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        plc: &AdditivePlacement,
        amount: usize,
        x: AdtTensor<RingT>,
    ) -> Result<AdtTensor<RingT>>
    where
        HostPlacement: PlacementShl<S, RingT, RingT>,
    {
        let (player0, player1) = plc.host_placements();
        let AdtTensor { shares: [x0, x1] } = &x;
        let z0 = player0.shl(sess, amount, x0);
        let z1 = player1.shl(sess, amount, x1);
        Ok(AdtTensor { shares: [z0, z1] })
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

pub trait TruncMaskGen<S: Session, ShapeT, RingT> {
    fn gen_trunc_mask(
        &self,
        sess: &S,
        amount: usize,
        shape: &ShapeT,
    ) -> (
        AdtTensor<RingT>,
        AdtTensor<RingT>,
        AdtTensor<RingT>,
    );
}

impl<S: Session, HostShapeT, HostRingT> TruncMaskGen<S, HostShapeT, HostRingT> for HostPlacement
where
    PrfKey: KnownType<S>,
    Seed: KnownType<S>,
    HostRingT: Ring + Clone,
    HostPlacement: PlacementDeriveSeed<S, m!(PrfKey), m!(Seed)>,
    HostPlacement: PlacementSampleUniform<S, HostShapeT, HostRingT>,
    HostPlacement: PlacementSampleUniformSeeded<S, HostShapeT, m!(Seed), HostRingT>,
    HostPlacement: PlacementKeyGen<S, m!(PrfKey)>,
    HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
    HostPlacement: PlacementShr<S, HostRingT, HostRingT>,
    HostPlacement: PlacementShl<S, HostRingT, HostRingT>,
{
    fn gen_trunc_mask(
        &self,
        sess: &S,
        amount: usize,
        shape: &HostShapeT, // TODO(Morten) take AdditiveShape instead?
    ) -> (
        AdtTensor<HostRingT>,
        AdtTensor<HostRingT>,
        AdtTensor<HostRingT>,
    ) {
        let r = self.sample_uniform(sess, shape);
        let r_msb = self.shr(sess, HostRingT::BitLength::VALUE - 1, &r);
        let r_top = self.shr(sess, amount + 1, &self.shl(sess, 1, &r));

        let key = self.gen_key(sess);
        let share = |x| {
            // TODO(Dragos) this could be optimized by instead sending the key (or seeds) to p0
            let sync_key = SyncKey::random();
            let seed = self.derive_seed(sess, sync_key, &key);
            let x0 = self.sample_uniform_seeded(sess, shape, &seed);
            let x1 = self.sub(sess, x, &x0);
            AdtTensor { shares: [x0, x1] }
        };

        let r_shared = share(&r);
        let r_top_shared = share(&r_top);
        let r_msb_shared = share(&r_msb);

        (r_shared, r_top_shared, r_msb_shared)
    }
}

impl<S: Session, HostRingT>
    PlacementTruncPrProvider<
        S,
        AdtTensor<HostRingT>,
        AdtTensor<HostRingT>,
    > for AdditivePlacement
where
    AdtTensor<HostRingT>: CanonicalType,
    <AdtTensor<HostRingT> as CanonicalType>::Type: KnownType<S>,
    AbstractReplicatedRingTensor<HostRingT>: CanonicalType,
    <AbstractReplicatedRingTensor<HostRingT> as CanonicalType>::Type: KnownType<S>,
    HostRingT: Ring,
    HostShape: KnownType<S>,
    HostPlacement: TruncMaskGen<S, m!(HostShape), HostRingT>,
    HostPlacement: PlacementReveal<S, st!(AdtTensor<HostRingT>), HostRingT>,
    HostPlacement: PlacementOnes<S, m!(HostShape), HostRingT>,
    HostPlacement: PlacementShape<S, HostRingT, m!(HostShape)>,
    HostPlacement: PlacementShl<S, HostRingT, HostRingT>,
    HostPlacement: PlacementShr<S, HostRingT, HostRingT>,
    AdtTensor<HostRingT>: Clone + Into<st!(AdtTensor<HostRingT>)>,
    st!(AdtTensor<HostRingT>): TryInto<AdtTensor<HostRingT>>,
    AdditivePlacement: PlacementAdd<
        S,
        st!(AdtTensor<HostRingT>),
        HostRingT,
        st!(AdtTensor<HostRingT>),
    >,
    AdditivePlacement: PlacementAdd<
        S,
        st!(AdtTensor<HostRingT>),
        st!(AdtTensor<HostRingT>),
        st!(AdtTensor<HostRingT>),
    >,
    AdditivePlacement: PlacementAdd<
        S,
        AdtTensor<HostRingT>,
        AdtTensor<HostRingT>,
        AdtTensor<HostRingT>,
    >,
    AdditivePlacement: PlacementSub<
        S,
        HostRingT,
        st!(AdtTensor<HostRingT>),
        st!(AdtTensor<HostRingT>),
    >,
    AdditivePlacement: PlacementMul<
        S,
        st!(AdtTensor<HostRingT>),
        HostRingT,
        st!(AdtTensor<HostRingT>),
    >,
    AdditivePlacement: PlacementShl<
        S,
        st!(AdtTensor<HostRingT>),
        st!(AdtTensor<HostRingT>),
    >,
    AdditivePlacement: PlacementSub<
        S,
        st!(AdtTensor<HostRingT>),
        st!(AdtTensor<HostRingT>),
        st!(AdtTensor<HostRingT>),
    >,
    AdditivePlacement: PlacementSub<
        S,
        AdtTensor<HostRingT>,
        AdtTensor<HostRingT>,
        AdtTensor<HostRingT>,
    >,
    AdditivePlacement: PlacementSub<
        S,
        st!(AdtTensor<HostRingT>),
        HostRingT,
        st!(AdtTensor<HostRingT>),
    >,
{
    fn trunc_pr(
        &self,
        sess: &S,
        amount: usize,
        provider: &HostPlacement,
        x: &AdtTensor<HostRingT>,
    ) -> AdtTensor<HostRingT> {
        #![allow(clippy::many_single_char_names)]

        let (player0, player1) = self.host_placements();
        assert!(*provider != player0);
        assert!(*provider != player1);

        let AdtTensor { shares: [x0, _x1] } = x;

        let shape = player0.shape(sess, x0);

        let (r, r_top, r_msb) = provider.gen_trunc_mask(sess, amount, &shape);
        // NOTE we consider input is always signed, and the following positive
        // conversion would be optional for unsigned numbers
        // NOTE we assume that input numbers are in range -2^{k-2} <= x < 2^{k-2}
        // so that 0 <= x + 2^{k-2} < 2^{k-1}
        // TODO we could insert debug_assert! to check above conditions
        let k = HostRingT::BitLength::VALUE - 1;
        let ones = player0.ones(sess, &shape);
        let upshifter = player0.shl(sess, k - 1, &ones);
        let downshifter = player0.shl(sess, k - amount - 1, &ones);

        // TODO(Morten) think the rest of this would clean up nicely if we instead revealed to a mirrored placement
        let x_positive: AdtTensor<HostRingT> = self
            .add(sess, &x.clone().into(), &upshifter)
            .try_into()
            .ok()
            .unwrap();
        let masked = self.add(sess, &x_positive, &r);
        let c = player0.reveal(sess, &masked.into());
        let c_no_msb = player0.shl(sess, 1, &c);
        // also called shifted
        let c_top = player0.shr(sess, amount + 1, &c_no_msb);
        let c_msb = player0.shr(sess, HostRingT::BitLength::VALUE - 1, &c);

        // OK
        let overflow = with_context!(
            self,
            sess,
            r_msb.clone().into() + c_msb - r_msb.clone().into() * c_msb - r_msb.into() * c_msb
        ); // a xor b = a+b-2ab
        let shifted_overflow = self.shl(sess, k - amount, &overflow);
        // shifted - upper + overflow << (k - m)
        let y_positive = with_context!(self, sess, c_top - r_top.into() + shifted_overflow);

        with_context!(self, sess, y_positive - downshifter)
            .try_into()
            .ok()
            .unwrap()
    }
}

modelled_kernel! {
    PlacementRepToAdt::rep_to_adt, RepToAdtOp,
    [
        (AdditivePlacement, (ReplicatedRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::rep_to_adt_kernel),
        (AdditivePlacement, (ReplicatedRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::rep_to_adt_kernel),
        (AdditivePlacement, (ReplicatedBitTensor) -> AdditiveBitTensor => [concrete] Self::rep_to_adt_kernel),
    ]
}

impl RepToAdtOp {
    fn rep_to_adt_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AbstractReplicatedRingTensor<HostRingT>,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
        AdditivePlacement: PlacementPlace<S, AdtTensor<HostRingT>>,
    {
        let (adt_player0, adt_player1) = adt.host_placements();
        let (rep_player0, rep_player1, rep_player2) = x.placement()?.host_placements();

        let AbstractReplicatedRingTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let shares = match () {
            _ if adt_player0 == rep_player0 => {
                let y0 = with_context!(rep_player0, sess, x00 + x10);
                let y1 = match () {
                    _ if adt_player1 == rep_player1 => x21,
                    _ if adt_player1 == rep_player2 => x22,
                    _ => x21,
                };
                [y0, y1]
            }
            _ if adt_player0 == rep_player1 => {
                let y0 = with_context!(rep_player1, sess, x11 + x21);
                let y1 = match () {
                    _ if adt_player1 == rep_player2 => x02,
                    _ if adt_player1 == rep_player0 => x00,
                    _ => x02,
                };
                [y0, y1]
            }
            _ if adt_player0 == rep_player2 => {
                let y0 = with_context!(rep_player2, sess, x22 + x02);
                let y1 = match () {
                    _ if adt_player1 == rep_player0 => x10,
                    _ if adt_player1 == rep_player1 => x11,
                    _ => x10,
                };
                [y0, y1]
            }
            _ if adt_player1 == rep_player0 => {
                let y0 = x21;
                let y1 = with_context!(rep_player0, sess, x00 + x10);
                [y0, y1]
            }
            _ if adt_player1 == rep_player1 => {
                let y0 = x02;
                let y1 = with_context!(rep_player1, sess, x11 + x21);
                [y0, y1]
            }
            _ if adt_player1 == rep_player2 => {
                let y0 = x10;
                let y1 = with_context!(rep_player2, sess, x22 + x02);
                [y0, y1]
            }
            _ => {
                let y0 = with_context!(rep_player0, sess, x00 + x10);
                let y1 = x21;
                [y0, y1]
            }
        };
        Ok(adt.place(sess, AdtTensor { shares }))
    }
}

pub trait PlacementDaBitProvider<S: Session, HostShapeT, O1, O2> {
    fn gen_dabit(
        &self,
        sess: &S,
        shape_provider: HostShapeT,
        shape_player0: HostShapeT,
        provider: &HostPlacement,
    ) -> (O1, O2);
}

impl<S: Session, HostShapeT, HostRingT, HostBitT>
    PlacementDaBitProvider<
        S,
        HostShapeT,
        AdtTensor<HostRingT>,
        AdtTensor<HostBitT>,
    > for AdditivePlacement
where
    HostRingT: Clone,
    Seed: KnownType<S>,
    PrfKey: KnownType<S>,
    HostPlacement: PlacementKeyGen<S, m!(PrfKey)>,
    HostPlacement: PlacementDeriveSeed<S, m!(PrfKey), m!(Seed)>,
    HostPlacement: PlacementSampleUniform<S, HostShapeT, HostBitT>,
    HostPlacement: PlacementSampleUniformSeeded<S, HostShapeT, m!(Seed), HostBitT>,
    HostPlacement: PlacementSampleUniformSeeded<S, HostShapeT, m!(Seed), HostRingT>,
    HostPlacement: PlacementSub<S, HostBitT, HostBitT, HostBitT>,
    HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
    HostPlacement: PlacementRingInject<S, HostBitT, HostRingT>,
    HostPlacement: PlacementPlace<S, HostRingT>,
    HostPlacement: PlacementPlace<S, HostBitT>,
{
    fn gen_dabit(
        &self,
        sess: &S,
        shape_provider: HostShapeT,
        shape_player0: HostShapeT,
        provider: &HostPlacement,
    ) -> (
        AdtTensor<HostRingT>,
        AdtTensor<HostBitT>,
    ) {
        let (player0, player1) = self.host_placements();
        assert!(*provider != player0);
        assert!(*provider != player1);

        let b: HostBitT = provider.sample_uniform(sess, &shape_provider);
        let br: HostRingT = provider.ring_inject(sess, 0, &b);

        let key = provider.gen_key(sess);
        let seed_b = provider.derive_seed(sess, SyncKey::random(), &key);
        let seed_br = provider.derive_seed(sess, SyncKey::random(), &key);

        let b0_provider: HostBitT = provider.sample_uniform_seeded(sess, &shape_provider, &seed_b);
        let b0: HostBitT = player0.sample_uniform_seeded(sess, &shape_player0, &seed_b);
        let b1: HostBitT = player1.place(sess, with_context!(provider, sess, b - b0_provider));

        let br0_provider: HostRingT =
            provider.sample_uniform_seeded(sess, &shape_provider, &seed_br);
        let br0: HostRingT = player0.sample_uniform_seeded(sess, &shape_player0, &seed_br);
        let br1: HostRingT = player1.place(sess, with_context!(provider, sess, br - br0_provider));

        let b_shared = AdtTensor { shares: [b0, b1] };
        let br_shared = AdtTensor { shares: [br0, br1] };

        (br_shared, b_shared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        computation::{Operation, Operator, Placement, RingAddOp},
        host::AbstractHostRingTensor,
        symbolic::{Symbolic, SymbolicHandle, SymbolicSession},
    };
    use ndarray::array;
    use ndarray::prelude::*;
    use proptest::prelude::*;

    #[test]
    fn test_add() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let bob = HostPlacement {
            owner: "bob".into(),
        };
        let adt = AdditivePlacement {
            owners: ["alice".into(), "bob".into()],
        };

        let x = AdditiveRing64Tensor {
            shares: [
                AbstractHostRingTensor::from_raw_plc(array![1, 2, 3], alice.clone()),
                AbstractHostRingTensor::from_raw_plc(array![4, 5, 6], bob.clone()),
            ],
        };

        let y = AdditiveRing64Tensor {
            shares: [
                AbstractHostRingTensor::from_raw_plc(array![7, 8, 9], alice.clone()),
                AbstractHostRingTensor::from_raw_plc(array![1, 2, 3], bob.clone()),
            ],
        };

        let sess = SyncSession::default();
        let AdtTensor { shares: [z0, z1] } = adt.add(&sess, &x, &y);

        assert_eq!(
            z0,
            AbstractHostRingTensor::from_raw_plc(array![1 + 7, 2 + 8, 3 + 9], alice.clone())
        );
        assert_eq!(
            z1,
            AbstractHostRingTensor::from_raw_plc(array![4 + 1, 5 + 2, 6 + 3], bob.clone())
        );

        let r_alice = AbstractHostRingTensor::from_raw_plc(array![7, 8, 9], alice.clone());
        let AdtTensor { shares: [zr0, zr1] } = adt.add(&sess, &x, &r_alice);

        assert_eq!(
            zr0,
            AbstractHostRingTensor::from_raw_plc(array![1 + 7, 2 + 8, 3 + 9], alice.clone())
        );
        assert_eq!(
            zr1,
            AbstractHostRingTensor::from_raw_plc(array![4, 5, 6], bob.clone())
        );

        let r_bob = AbstractHostRingTensor::from_raw_plc(array![7, 8, 9], bob.clone());
        let AdtTensor {
            shares: [zrb0, zrb1],
        } = adt.add(&sess, &x, &r_bob);

        assert_eq!(
            zrb0,
            AbstractHostRingTensor::from_raw_plc(array![1, 2, 3], alice)
        );
        assert_eq!(
            zrb1,
            AbstractHostRingTensor::from_raw_plc(array![4 + 7, 5 + 8, 6 + 9], bob)
        );
    }

    #[test]
    fn test_trunc() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let bob = HostPlacement {
            owner: "bob".into(),
        };
        let carole = HostPlacement {
            owner: "carole".into(),
        };
        let adt = AdditivePlacement {
            owners: ["alice".into(), "bob".into()],
        };

        let x = AdditiveRing64Tensor {
            shares: [
                AbstractHostRingTensor::from_raw_plc(array![0_u64, 0, 0], alice),
                AbstractHostRingTensor::from_raw_plc(
                    array![
                        4611686018427387903,
                        -1152921504606846976_i64 as u64,
                        1152921504606846975
                    ],
                    bob,
                ),
            ],
        };

        let sess = SyncSession::default();
        let x_trunc = adt.trunc_pr(&sess, 60, &carole, &x);
        let _y = carole.reveal(&sess, &x_trunc);

        let target = AbstractHostRingTensor::from_raw_plc(array![3, -1_i64 as u64, 0], carole);

        // probabilistic truncation can be off by 1
        for (i, value) in _y.0.iter().enumerate() {
            let diff = value - target.0[i];
            assert!(
                diff == std::num::Wrapping(1)
                    || diff == std::num::Wrapping(u64::MAX)
                    || diff == std::num::Wrapping(0),
                "difference = {}, lhs = {}, rhs = {}",
                diff,
                value,
                target.0[i]
            );
        }
    }

    fn any_bounded_u64() -> impl Strategy<Value = u64> {
        any::<u64>().prop_map(|x| (x >> 2) - 1)
    }

    fn any_bounded_u128() -> impl Strategy<Value = u128> {
        any::<u128>().prop_map(|x| (x >> 2) - 1)
    }

    macro_rules! adt_truncation_test {
        ($func_name:ident, $tt: ident) => {
            fn $func_name(xs: ArrayD<$tt>, amount: usize, ys: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let bob = HostPlacement {
                    owner: "bob".into(),
                };

                let carole = HostPlacement {
                    owner: "carole".into(),
                };

                let adt = AdditivePlacement {
                    owners: ["alice".into(), "bob".into()],
                };

                // creates an additive sharing of xs
                let zero =
                    Array::from_shape_vec(IxDyn(&[xs.len()]), vec![0 as $tt; xs.len()]).unwrap();
                let x = AbstractAdditiveTensor {
                    shares: [
                        AbstractHostRingTensor::from_raw_plc(zero, alice),
                        AbstractHostRingTensor::from_raw_plc(xs.clone(), bob),
                    ],
                };

                let sess = SyncSession::default();
                let x_trunc = adt.trunc_pr(&sess, amount, &carole, &x);
                let _y = carole.reveal(&sess, &x_trunc);

                let target_y = AbstractHostRingTensor::from_raw_plc(ys.clone(), carole.clone());
                for (i, value) in _y.0.iter().enumerate() {
                    let diff = value - target_y.0[i];
                    assert!(
                        diff == std::num::Wrapping(1)
                            || diff == std::num::Wrapping($tt::MAX)
                            || diff == std::num::Wrapping(0),
                        "difference = {}, lhs = {}, rhs = {}",
                        diff,
                        value,
                        target_y.0[i]
                    );
                }
            }
        };
    }

    adt_truncation_test!(test_adt_trunc64, u64);
    adt_truncation_test!(test_adt_trunc128, u128);

    proptest! {
        #[test]
        fn test_fuzzy_adt_trunc64(raw_vector in proptest::collection::vec(any_bounded_u64(), 1..5), amount in 0usize..62
        ) {
            let target = raw_vector.iter().map(|x| x >> amount).collect::<Vec<_>>();
            test_adt_trunc64(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), amount, Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }

        #[test]
        fn test_fuzzy_adt_trunc128(raw_vector in proptest::collection::vec(any_bounded_u128(), 1..5), amount in 0usize..126
        ) {
            let target = raw_vector.iter().map(|x| x >> amount).collect::<Vec<_>>();
            test_adt_trunc128(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), amount, Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }
    }

    #[test]
    fn test_symbolic_add() {
        let adt = AdditivePlacement {
            owners: ["alice".into(), "bob".into()],
        };

        let x: <AdditiveRing64Tensor as KnownType<SymbolicSession>>::Type =
            Symbolic::Symbolic(SymbolicHandle {
                op: "x".into(),
                plc: adt.clone(),
            });

        let y: <AdditiveRing64Tensor as KnownType<SymbolicSession>>::Type =
            Symbolic::Symbolic(SymbolicHandle {
                op: "x".into(),
                plc: adt.clone(),
            });

        let sess = SymbolicSession::default();
        let z = adt.add(&sess, &x, &y);

        let op_name = match z {
            Symbolic::Symbolic(handle) => {
                assert_eq!("op_0", handle.op);
                handle.op
            }
            _ => panic!("Expected a symbolic result from the symbolic addition"),
        };

        sess.ops_iter(|mut iter| match iter.find(|o| o.name == op_name) {
            None => panic!("Newly created operation was not placed on graph"),
            Some(op) => assert!(matches!(
                op,
                Operation {
                    kind: Operator::AdtAdd(AdtAddOp { sig: _ }),
                    ..
                }
            )),
        });
    }

    #[test]
    fn test_concrete_symbolic_add() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let bob = HostPlacement {
            owner: "bob".into(),
        };

        let adt = AdditivePlacement {
            owners: ["alice".into(), "bob".into()],
        };

        let x: <AdditiveRing64Tensor as KnownType<SymbolicSession>>::Type =
            Symbolic::Concrete(AdtTensor {
                shares: [
                    Symbolic::Symbolic(SymbolicHandle {
                        op: "x0".into(),
                        plc: alice.clone(),
                    }),
                    Symbolic::Symbolic(SymbolicHandle {
                        op: "x1".into(),
                        plc: bob.clone(),
                    }),
                ],
            });

        let y: <AdditiveRing64Tensor as KnownType<SymbolicSession>>::Type =
            Symbolic::Concrete(AdtTensor {
                shares: [
                    Symbolic::Symbolic(SymbolicHandle {
                        op: "y0".into(),
                        plc: alice,
                    }),
                    Symbolic::Symbolic(SymbolicHandle {
                        op: "y1".into(),
                        plc: bob,
                    }),
                ],
            });

        let sess = SymbolicSession::default();
        let z = adt.add(&sess, &x, &y);

        match &z {
            Symbolic::Concrete(AdtTensor { shares: [z0, z1] }) => {
                match z0 {
                    Symbolic::Symbolic(handle) => {
                        assert_eq!("op_0", handle.op);
                    }
                    _ => panic!("Expected a symbolic result from the symbolic addition"),
                }
                match z1 {
                    Symbolic::Symbolic(handle) => {
                        assert_eq!("op_1", handle.op);
                    }
                    _ => panic!("Expected a symbolic result from the symbolic addition"),
                }
            }
            _ => {
                panic!("Expected a concrete result from the symbolic addition on a concrete value")
            }
        }

        sess.ops_iter(|mut iter| {
            assert!(iter.any(|o| matches!(o,
                Operation {
                    name,
                    kind: Operator::RingAdd(RingAddOp { sig: _ }),
                    inputs,
                    placement: Placement::Host(HostPlacement { owner }),
                    ..
                }
                if name == "op_0" && inputs == &vec!["x0", "y0"] && owner.0 == "alice"
            )));
        });

        sess.ops_iter(|mut iter| {
            assert!(iter.any(|o| matches!(o,
                Operation {
                    name,
                    kind: Operator::RingAdd(RingAddOp { sig: _ }),
                    inputs,
                    placement: Placement::Host(HostPlacement { owner }),
                    ..
                }
                if name == "op_1" && inputs == &vec!["x1", "y1"] && owner.0 == "bob"
            )));
        });
    }
}
