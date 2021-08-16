//! Placements backed by additive secret sharing

use crate::computation::{
    AdditivePlacement, AdtAddOp, AdtFillOp, AdtMulOp, AdtRevealOp, AdtShlOp, AdtSubOp, Constant,
    HostPlacement, KnownType, Placed, RepToAdtOp, ReplicatedPlacement,
};
use crate::error::Result;
use crate::host::{HostRing128Tensor, HostRing64Tensor, HostShape, RingSize};
use crate::kernels::{
    PlacementAdd, PlacementDeriveSeed, PlacementFill, PlacementKeyGen, PlacementMul, PlacementNeg,
    PlacementOnes, PlacementPlace, PlacementRepToAdt, PlacementReveal, PlacementSampleBits,
    PlacementSampleUniform, PlacementShape, PlacementShl, PlacementShr, PlacementSub,
    PlacementTruncPrProvider, Session,
};
use crate::prim::{PrfKey, RawNonce, Seed};
use crate::replicated::CanonicalType;
use crate::replicated::{
    AbstractReplicatedTensor, ReplicatedRing128Tensor, ReplicatedRing64Tensor,
};
use macros::with_context;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractAdditiveTensor<R> {
    pub shares: [R; 2],
}

pub type AdditiveRing64Tensor = AbstractAdditiveTensor<HostRing64Tensor>;

pub type AdditiveRing128Tensor = AbstractAdditiveTensor<HostRing128Tensor>;

impl<R> Placed for AbstractAdditiveTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Placement = AdditivePlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let AbstractAdditiveTensor { shares: [x0, x1] } = self;

        let owner0 = x0.placement()?.owner;
        let owner1 = x1.placement()?.owner;

        let owners = [owner0, owner1];
        Ok(AdditivePlacement { owners })
    }
}

impl<S: Session, R> PlacementPlace<S, AbstractAdditiveTensor<R>> for AdditivePlacement
where
    AbstractAdditiveTensor<R>: Placed<Placement = AdditivePlacement>,
    HostPlacement: PlacementPlace<S, R>,
{
    fn place(&self, sess: &S, x: AbstractAdditiveTensor<R>) -> AbstractAdditiveTensor<R> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                let AbstractAdditiveTensor { shares: [x0, x1] } = x;
                let (player0, player1) = self.host_placements();
                AbstractAdditiveTensor {
                    shares: [player0.place(sess, x0), player1.place(sess, x1)],
                }
            }
        }
    }
}

modelled!(PlacementFill::fill, AdditivePlacement, attributes[value: Constant] (HostShape) -> AdditiveRing64Tensor, AdtFillOp);
modelled!(PlacementFill::fill, AdditivePlacement, attributes[value: Constant] (HostShape) -> AdditiveRing128Tensor, AdtFillOp);

hybrid_kernel! {
    AdtFillOp,
    [
        (AdditivePlacement, (HostShape) -> AdditiveRing64Tensor => attributes[value] Self::kernel),
        (AdditivePlacement, (HostShape) -> AdditiveRing128Tensor => attributes[value] Self::kernel),
    ]
}

impl AdtFillOp {
    fn kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        plc: &AdditivePlacement,
        value: Constant,
        shape: ShapeT,
    ) -> AbstractAdditiveTensor<RingT>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return PublicAdditiveTensor, but we don't have that type yet

        let (player0, player1) = plc.host_placements();

        let shares = [
            player0.fill(sess, value, &shape),
            player1.fill(sess, Constant::Ring64(0), &shape),
        ];
        AbstractAdditiveTensor { shares }
    }
}

modelled!(PlacementReveal::reveal, HostPlacement, (AdditiveRing64Tensor) -> HostRing64Tensor, AdtRevealOp);
modelled!(PlacementReveal::reveal, HostPlacement, (AdditiveRing128Tensor) -> HostRing128Tensor, AdtRevealOp);

hybrid_kernel! {
    AdtRevealOp,
    [
        (HostPlacement, (AdditiveRing64Tensor) -> HostRing64Tensor => Self::kernel),
        (HostPlacement, (AdditiveRing128Tensor) -> HostRing128Tensor => Self::kernel),
    ]
}

impl AdtRevealOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        plc: &HostPlacement,
        xe: AbstractAdditiveTensor<RingT>,
    ) -> RingT
    where
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
    {
        let AbstractAdditiveTensor { shares: [x0, x1] } = &xe;
        with_context!(plc, sess, x1 + x0)
    }
}

modelled!(PlacementAdd::add, AdditivePlacement, (AdditiveRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor, AdtAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (AdditiveRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor, AdtAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor, AdtAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor, AdtAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor, AdtAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor, AdtAddOp);

hybrid_kernel! {
    AdtAddOp,
    [
        (AdditivePlacement, (AdditiveRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => Self::ring_adt_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => Self::ring_adt_kernel),
    ]
}

impl AdtAddOp {
    fn adt_adt_kernel<S: Session, RingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AbstractAdditiveTensor<RingT>,
        y: AbstractAdditiveTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
    {
        let (player0, player1) = adt.host_placements();

        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;
        let AbstractAdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x0 + y0);
        let z1 = with_context!(player1, sess, x1 + y1);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }

    fn adt_ring_kernel<S: Session, RingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AbstractAdditiveTensor<RingT>,
        y: RingT,
    ) -> AbstractAdditiveTensor<RingT>
    where
        RingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        AdditivePlacement: PlacementPlace<S, AbstractAdditiveTensor<RingT>>,
    {
        let (player0, player1) = adt.host_placements();
        let y_plc = y.placement().unwrap();

        let AbstractAdditiveTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, sess, x0 + y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, sess, x1 + y)],
            _ => [with_context!(player0, sess, x0 + y), x1],
        };
        adt.place(sess, AbstractAdditiveTensor { shares })
    }

    fn ring_adt_kernel<S: Session, RingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: RingT,
        y: AbstractAdditiveTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        RingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        AdditivePlacement: PlacementPlace<S, AbstractAdditiveTensor<RingT>>,
    {
        let (player0, player1) = adt.host_placements();
        let x_plc = x.placement().unwrap();

        let AbstractAdditiveTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, sess, y0 + x), y1],
            _ if x_plc == player1 => [y0, with_context!(player1, sess, x + y1)],
            _ => [with_context!(player0, sess, x + y0), y1],
        };
        adt.place(sess, AbstractAdditiveTensor { shares })
    }
}

modelled!(PlacementSub::sub, AdditivePlacement, (AdditiveRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor, AdtSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (AdditiveRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor, AdtSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor, AdtSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor, AdtSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor, AdtSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor, AdtSubOp);

hybrid_kernel! {
    AdtSubOp,
    [
        (AdditivePlacement, (AdditiveRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => Self::ring_adt_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => Self::ring_adt_kernel),
    ]
}

impl AdtSubOp {
    fn adt_adt_kernel<S: Session, R>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AbstractAdditiveTensor<R>,
        y: AbstractAdditiveTensor<R>,
    ) -> AbstractAdditiveTensor<R>
    where
        HostPlacement: PlacementSub<S, R, R, R>,
    {
        let (player0, player1) = adt.host_placements();

        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;
        let AbstractAdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x0 - y0);
        let z1 = with_context!(player1, sess, x1 - y1);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }

    fn adt_ring_kernel<S: Session, R>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AbstractAdditiveTensor<R>,
        y: R,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<S, R, R, R>,
        AdditivePlacement: PlacementPlace<S, AbstractAdditiveTensor<R>>,
    {
        let (player0, player1) = adt.host_placements();
        let y_plc = y.placement().unwrap();

        let AbstractAdditiveTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, sess, x0 - y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, sess, x1 - y)],
            _ => [with_context!(player0, sess, x0 - y), x1],
        };
        adt.place(sess, AbstractAdditiveTensor { shares })
    }

    fn ring_adt_kernel<S: Session, R>(
        sess: &S,
        adt: &AdditivePlacement,
        x: R,
        y: AbstractAdditiveTensor<R>,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<S, R, R, R>,
        HostPlacement: PlacementNeg<S, R, R>,
        AdditivePlacement: PlacementPlace<S, AbstractAdditiveTensor<R>>,
    {
        let (player0, player1) = adt.host_placements();
        let x_plc = x.placement().unwrap();

        let AbstractAdditiveTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, sess, x - y0), player1.neg(sess, &y1)],
            _ if x_plc == player1 => [player0.neg(sess, &y0), with_context!(player1, sess, x - y1)],
            _ => [with_context!(player0, sess, x - y0), player1.neg(sess, &y1)],
        };
        adt.place(sess, AbstractAdditiveTensor { shares })
    }
}

modelled!(PlacementMul::mul, AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor, AdtMulOp);
modelled!(PlacementMul::mul, AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor, AdtMulOp);
modelled!(PlacementMul::mul, AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor, AdtMulOp);
modelled!(PlacementMul::mul, AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor, AdtMulOp);

hybrid_kernel! {
    AdtMulOp,
    [
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => Self::ring_adt_kernel),
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => Self::ring_adt_kernel),
    ]
}

impl AdtMulOp {
    fn ring_adt_kernel<S: Session, R>(
        sess: &S,
        adt: &AdditivePlacement,
        x: R,
        y: AbstractAdditiveTensor<R>,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<S, R, R, R>,
    {
        let (player0, player1) = adt.host_placements();

        let AbstractAdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x * y0);
        let z1 = with_context!(player1, sess, x * y1);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }

    fn adt_ring_kernel<S: Session, R>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AbstractAdditiveTensor<R>,
        y: R,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<S, R, R, R>,
    {
        let (player0, player1) = adt.host_placements();

        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;

        let z0 = with_context!(player0, sess, x0 * y);
        let z1 = with_context!(player1, sess, x1 * y);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }
}

modelled!(PlacementShl::shl, AdditivePlacement, attributes[amount: usize] (AdditiveRing64Tensor) -> AdditiveRing64Tensor, AdtShlOp);
modelled!(PlacementShl::shl, AdditivePlacement, attributes[amount: usize] (AdditiveRing128Tensor) -> AdditiveRing128Tensor, AdtShlOp);

hybrid_kernel! {
    AdtShlOp,
    [
        (AdditivePlacement, (AdditiveRing64Tensor) -> AdditiveRing64Tensor => attributes[amount] Self::kernel),
        (AdditivePlacement, (AdditiveRing128Tensor) -> AdditiveRing128Tensor => attributes[amount] Self::kernel),
    ]
}

impl AdtShlOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        plc: &AdditivePlacement,
        amount: usize,
        x: AbstractAdditiveTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        HostPlacement: PlacementShl<S, RingT, RingT>,
    {
        let (player0, player1) = plc.host_placements();
        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;
        let z0 = player0.shl(sess, amount, x0);
        let z1 = player1.shl(sess, amount, x1);
        AbstractAdditiveTensor { shares: [z0, z1] }
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
        AbstractAdditiveTensor<RingT>,
        AbstractAdditiveTensor<RingT>,
        AbstractAdditiveTensor<RingT>,
    );
}

impl<S: Session, R> TruncMaskGen<S, cs!(HostShape), R> for HostPlacement
where
    PrfKey: KnownType<S>,
    HostShape: KnownType<S>,
    Seed: KnownType<S>,
    R: RingSize + Clone,
    HostPlacement: PlacementDeriveSeed<S, cs!(PrfKey), cs!(Seed)>,
    HostPlacement: PlacementSampleBits<S, cs!(HostShape), cs!(Seed), R>,
    HostPlacement: PlacementSampleUniform<S, cs!(HostShape), cs!(Seed), R>,
    HostPlacement: PlacementKeyGen<S, cs!(PrfKey)>,
    HostPlacement: PlacementSub<S, R, R, R>,
    HostPlacement: PlacementShr<S, R, R>,
    HostPlacement: PlacementShl<S, R, R>,
{
    fn gen_trunc_mask(
        &self,
        sess: &S,
        amount: usize,
        shape: &cs!(HostShape), // TODO(Morten) take AdditiveShape instead?
    ) -> (
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
    ) {
        let key = self.gen_key(sess);

        let sync_key = RawNonce::generate();
        let seed = self.derive_seed(sess, sync_key, &key);

        let r = self.sample_uniform(sess, shape, &seed);
        let r_msb = self.shr(sess, R::SIZE - 1, &r);
        let r_top = self.shr(sess, amount + 1, &self.shl(sess, 1, &r));

        let share = |x| {
            // TODO(Dragos) this could probably be optimized by sending the key to p0
            let share_sync_key = RawNonce::generate();
            let seed = self.derive_seed(sess, share_sync_key, &key);
            let x0 = self.sample_uniform(sess, shape, &seed);
            let x1 = self.sub(sess, x, &x0);
            AbstractAdditiveTensor { shares: [x0, x1] }
        };

        let r_shared = share(&r);
        let r_top_shared = share(&r_top);
        let r_msb_shared = share(&r_msb);

        (r_shared, r_top_shared, r_msb_shared)
    }
}

use std::convert::TryInto;

impl<S: Session, R>
    PlacementTruncPrProvider<S, AbstractAdditiveTensor<R>, AbstractAdditiveTensor<R>>
    for AdditivePlacement
where
    AbstractAdditiveTensor<R>: CanonicalType,
    <AbstractAdditiveTensor<R> as CanonicalType>::Type: KnownType<S>,
    AbstractReplicatedTensor<R>: CanonicalType,
    <AbstractReplicatedTensor<R> as CanonicalType>::Type: KnownType<S>,
    R: RingSize,
    HostShape: KnownType<S>,
    HostPlacement: TruncMaskGen<S, cs!(HostShape), R>,
    HostPlacement: PlacementReveal<S, st!(AbstractAdditiveTensor<R>), R>,
    HostPlacement: PlacementOnes<S, cs!(HostShape), R>,
    HostPlacement: PlacementShape<S, R, cs!(HostShape)>,
    HostPlacement: PlacementShl<S, R, R>,
    HostPlacement: PlacementShr<S, R, R>,
    AbstractAdditiveTensor<R>: Clone + Into<st!(AbstractAdditiveTensor<R>)>,
    st!(AbstractAdditiveTensor<R>): TryInto<AbstractAdditiveTensor<R>>,
    AdditivePlacement:
        PlacementAdd<S, st!(AbstractAdditiveTensor<R>), R, st!(AbstractAdditiveTensor<R>)>,
    AdditivePlacement: PlacementAdd<
        S,
        st!(AbstractAdditiveTensor<R>),
        st!(AbstractAdditiveTensor<R>),
        st!(AbstractAdditiveTensor<R>),
    >,
    AdditivePlacement:
        PlacementSub<S, R, st!(AbstractAdditiveTensor<R>), st!(AbstractAdditiveTensor<R>)>,
    AdditivePlacement:
        PlacementMul<S, st!(AbstractAdditiveTensor<R>), R, st!(AbstractAdditiveTensor<R>)>,
    AdditivePlacement:
        PlacementShl<S, st!(AbstractAdditiveTensor<R>), st!(AbstractAdditiveTensor<R>)>,
    AdditivePlacement: PlacementSub<
        S,
        st!(AbstractAdditiveTensor<R>),
        st!(AbstractAdditiveTensor<R>),
        st!(AbstractAdditiveTensor<R>),
    >,
    AdditivePlacement:
        PlacementSub<S, st!(AbstractAdditiveTensor<R>), R, st!(AbstractAdditiveTensor<R>)>,
{
    fn trunc_pr(
        &self,
        sess: &S,
        amount: usize,
        provider: &HostPlacement,
        x: &AbstractAdditiveTensor<R>,
    ) -> AbstractAdditiveTensor<R> {
        #![allow(clippy::many_single_char_names)]

        // Hack to get around https://github.com/tf-encrypted/runtime/issues/372
        let adt = self;

        let (player_a, player_b) = self.host_placements();
        assert!(provider != &player_a);
        assert!(provider != &player_b);

        let AbstractAdditiveTensor { shares: [x0, _x1] } = x;

        let shape = player_a.shape(sess, x0);

        let (r, r_top, r_msb) = provider.gen_trunc_mask(sess, amount, &shape);
        // NOTE we consider input is always signed, and the following positive
        // conversion would be optional for unsigned numbers
        // NOTE we assume that input numbers are in range -2^{k-2} <= x < 2^{k-2}
        // so that 0 <= x + 2^{k-2} < 2^{k-1}
        // TODO we could insert debug_assert! to check above conditions
        let k = R::SIZE - 1;
        let ones = player_a.ones(sess, &shape);
        let upshifter = player_a.shl(sess, k - 1, &ones);
        let downshifter = player_a.shl(sess, k - amount - 1, &ones);

        let x_positive: AbstractAdditiveTensor<R> = self
            .add(sess, &x.clone().into(), &upshifter)
            .try_into()
            .ok()
            .unwrap();
        let masked: AbstractAdditiveTensor<R> = adt
            .add(sess, &x_positive.into(), &r.into())
            .try_into()
            .ok()
            .unwrap();
        let c = player_a.reveal(sess, &masked.into());
        let c_no_msb = player_a.shl(sess, 1, &c);
        // also called shifted
        let c_top = player_a.shr(sess, amount + 1, &c_no_msb);
        let c_msb = player_a.shr(sess, R::SIZE - 1, &c);

        // OK
        let overflow = with_context!(
            adt,
            sess,
            r_msb.clone().into() + c_msb - r_msb.clone().into() * c_msb - r_msb.into() * c_msb
        )
        .try_into()
        .ok()
        .unwrap(); // a xor b = a+b-2ab
        let shifted_overflow = self
            .shl(sess, k - amount, &overflow.into())
            .try_into()
            .ok()
            .unwrap();
        // shifted - upper + overflow << (k - m)
        let y_positive: AbstractAdditiveTensor<R> =
            with_context!(adt, sess, c_top - r_top.into() + shifted_overflow.into())
                .try_into()
                .ok()
                .unwrap();

        with_context!(adt, sess, y_positive.into() - downshifter)
            .try_into()
            .ok()
            .unwrap()
    }
}

modelled!(PlacementRepToAdt::rep_to_adt, AdditivePlacement, (ReplicatedRing64Tensor) -> AdditiveRing64Tensor, RepToAdtOp);
modelled!(PlacementRepToAdt::rep_to_adt, AdditivePlacement, (ReplicatedRing128Tensor) -> AdditiveRing128Tensor, RepToAdtOp);

hybrid_kernel! {
    RepToAdtOp,
    [
        (AdditivePlacement, (ReplicatedRing64Tensor) -> AdditiveRing64Tensor => Self::rep_to_adt_kernel),
        (AdditivePlacement, (ReplicatedRing128Tensor) -> AdditiveRing128Tensor => Self::rep_to_adt_kernel),
    ]
}

impl RepToAdtOp {
    fn rep_to_adt_kernel<S: Session, RingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AbstractReplicatedTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        AbstractReplicatedTensor<RingT>: Placed<Placement = ReplicatedPlacement>,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        AdditivePlacement: PlacementPlace<S, AbstractAdditiveTensor<RingT>>,
    {
        let (adt_player0, adt_player1) = adt.host_placements();
        let (rep_player0, rep_player1, rep_player2) = x.placement().unwrap().host_placements();

        let AbstractReplicatedTensor {
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
        adt.place(sess, AbstractAdditiveTensor { shares })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        computation::{Operation, Operator, Placement, RingAddOp},
        host::AbstractHostRingTensor,
        kernels::SyncSession,
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
        let AbstractAdditiveTensor { shares: [z0, z1] } = adt.add(&sess, &x, &y);

        assert_eq!(
            z0,
            AbstractHostRingTensor::from_raw_plc(array![1 + 7, 2 + 8, 3 + 9], alice.clone())
        );
        assert_eq!(
            z1,
            AbstractHostRingTensor::from_raw_plc(array![4 + 1, 5 + 2, 6 + 3], bob.clone())
        );

        let r_alice = AbstractHostRingTensor::from_raw_plc(array![7, 8, 9], alice.clone());
        let AbstractAdditiveTensor { shares: [zr0, zr1] } = adt.add(&sess, &x, &r_alice);

        assert_eq!(
            zr0,
            AbstractHostRingTensor::from_raw_plc(array![1 + 7, 2 + 8, 3 + 9], alice.clone())
        );
        assert_eq!(
            zr1,
            AbstractHostRingTensor::from_raw_plc(array![4, 5, 6], bob.clone())
        );

        let r_bob = AbstractHostRingTensor::from_raw_plc(array![7, 8, 9], bob.clone());
        let AbstractAdditiveTensor {
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

        let ops = sess.ops.read().unwrap();
        match ops.iter().find(|o| o.name == op_name) {
            None => panic!("Newly created operation was not placed on graph"),
            Some(op) => assert!(matches!(
                op,
                Operation {
                    kind: Operator::AdtAdd(AdtAddOp { sig: _ }),
                    ..
                }
            )),
        }
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
            Symbolic::Concrete(AbstractAdditiveTensor {
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
            Symbolic::Concrete(AbstractAdditiveTensor {
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
            Symbolic::Concrete(AbstractAdditiveTensor { shares: [z0, z1] }) => {
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

        let ops = sess.ops.read().unwrap();

        assert!(ops.iter().any(|o| matches!(o,
            Operation {
                name,
                kind: Operator::RingAdd(RingAddOp { sig: _ }),
                inputs,
                placement: Placement::Host(HostPlacement { owner }),
                ..
            }
            if name == "op_0" && inputs == &vec!["x0", "y0"] && owner.0 == "alice"
        )));

        assert!(ops.iter().any(|o| matches!(o,
            Operation {
                name,
                kind: Operator::RingAdd(RingAddOp { sig: _ }),
                inputs,
                placement: Placement::Host(HostPlacement { owner }),
                ..
            }
            if name == "op_1" && inputs == &vec!["x1", "y1"] && owner.0 == "bob"
        )));
    }
}
