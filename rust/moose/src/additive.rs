use crate::computation::{
    AdditivePlacement, AdtAddOp, AdtFillOp, AdtMulOp, AdtRevealOp, AdtShlOp, AdtSubOp, Constant,
    HostPlacement, KnownType, Placed, RepToAdtOp, ReplicatedPlacement,
};
use crate::kernels::{
    PlacementAdd, PlacementDeriveSeed, PlacementFill, PlacementKeyGen, PlacementMul, PlacementNeg,
    PlacementOnes, PlacementRepToAdt, PlacementReveal, PlacementSampleBits, PlacementSampleUniform,
    PlacementShape, PlacementShl, PlacementShr, PlacementSub, PlacementTruncPrProvider, Session,
};
use crate::prim::{PrfKey, RawNonce, Seed};
use crate::replicated::{AbstractReplicatedTensor, Replicated128Tensor, Replicated64Tensor};
use crate::ring::{Ring128Tensor, Ring64Tensor, RingSize};
use crate::standard::Shape;
use macros::with_context;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractAdditiveTensor<R> {
    pub shares: [R; 2],
}

pub type Additive64Tensor = AbstractAdditiveTensor<Ring64Tensor>;

pub type Additive128Tensor = AbstractAdditiveTensor<Ring128Tensor>;

impl<R> Placed for AbstractAdditiveTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Placement = AdditivePlacement;

    fn placement(&self) -> Self::Placement {
        let AbstractAdditiveTensor { shares: [x0, x1] } = self;

        let owner0 = x0.placement().owner;
        let owner1 = x1.placement().owner;

        let owners = [owner0, owner1];
        AdditivePlacement { owners }
    }
}

modelled!(PlacementFill::fill, AdditivePlacement, attributes[value: Constant] (Shape) -> Additive64Tensor, AdtFillOp);
modelled!(PlacementFill::fill, AdditivePlacement, attributes[value: Constant] (Shape) -> Additive128Tensor, AdtFillOp);

hybrid_kernel! {
    AdtFillOp,
    [
        (AdditivePlacement, (Shape) -> Additive64Tensor => attributes[value] Self::kernel),
        (AdditivePlacement, (Shape) -> Additive128Tensor => attributes[value] Self::kernel),
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

modelled!(PlacementReveal::reveal, HostPlacement, (Additive64Tensor) -> Ring64Tensor, AdtRevealOp);
modelled!(PlacementReveal::reveal, HostPlacement, (Additive128Tensor) -> Ring128Tensor, AdtRevealOp);

hybrid_kernel! {
    AdtRevealOp,
    [
        (HostPlacement, (Additive64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Additive128Tensor) -> Ring128Tensor => Self::kernel),
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

modelled!(PlacementAdd::add, AdditivePlacement, (Additive64Tensor, Additive64Tensor) -> Additive64Tensor, AdtAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (Additive128Tensor, Additive128Tensor) -> Additive128Tensor, AdtAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor, AdtAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor, AdtAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor, AdtAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor, AdtAddOp);

hybrid_kernel! {
    AdtAddOp,
    [
        (AdditivePlacement, (Additive64Tensor, Additive64Tensor) -> Additive64Tensor => Self::adt_adt_kernel),
        (AdditivePlacement, (Additive128Tensor, Additive128Tensor) -> Additive128Tensor => Self::adt_adt_kernel),
        (AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor => Self::ring_adt_kernel),
        (AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor => Self::ring_adt_kernel),
    ]
}

impl AdtAddOp {
    fn adt_adt_kernel<S: Session, RingT>(
        sess: &S,
        add: &AdditivePlacement,
        x: AbstractAdditiveTensor<RingT>,
        y: AbstractAdditiveTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
    {
        let (player0, player1) = add.host_placements();

        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;
        let AbstractAdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x0 + y0);
        let z1 = with_context!(player1, sess, x1 + y1);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }

    fn adt_ring_kernel<S: Session, RingT>(
        sess: &S,
        add: &AdditivePlacement,
        x: AbstractAdditiveTensor<RingT>,
        y: RingT,
    ) -> AbstractAdditiveTensor<RingT>
    where
        RingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
    {
        let (player0, player1) = add.host_placements();
        let y_plc = y.placement();

        let AbstractAdditiveTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, sess, x0 + y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, sess, x1 + y)],
            _ => [with_context!(player0, sess, x0 + y), x1],
        };
        AbstractAdditiveTensor { shares }
    }

    fn ring_adt_kernel<S: Session, RingT>(
        sess: &S,
        add: &AdditivePlacement,
        x: RingT,
        y: AbstractAdditiveTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        RingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
    {
        let (player0, player1) = add.host_placements();
        let x_plc = x.placement();

        let AbstractAdditiveTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, sess, y0 + x), y1],
            _ if x_plc == player1 => [y0, with_context!(player1, sess, x + y1)],
            _ => [with_context!(player0, sess, x + y0), y1],
        };
        AbstractAdditiveTensor { shares }
    }
}

modelled!(PlacementSub::sub, AdditivePlacement, (Additive64Tensor, Additive64Tensor) -> Additive64Tensor, AdtSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (Additive128Tensor, Additive128Tensor) -> Additive128Tensor, AdtSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor, AdtSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor, AdtSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor, AdtSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor, AdtSubOp);

hybrid_kernel! {
    AdtSubOp,
    [
        (AdditivePlacement, (Additive64Tensor, Additive64Tensor) -> Additive64Tensor => Self::adt_adt_kernel),
        (AdditivePlacement, (Additive128Tensor, Additive128Tensor) -> Additive128Tensor => Self::adt_adt_kernel),
        (AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor => Self::ring_adt_kernel),
        (AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor => Self::ring_adt_kernel),
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
    {
        let (player0, player1) = adt.host_placements();
        let y_plc = y.placement();

        let AbstractAdditiveTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, sess, x0 - y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, sess, x1 - y)],
            _ => [with_context!(player0, sess, x0 - y), x1],
        };
        AbstractAdditiveTensor { shares }
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
    {
        let (player0, player1) = adt.host_placements();
        let x_plc = x.placement();

        let AbstractAdditiveTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, sess, x - y0), player1.neg(sess, &y1)],
            _ if x_plc == player1 => [player0.neg(sess, &y0), with_context!(player1, sess, x - y1)],
            _ => [with_context!(player0, sess, x - y0), player1.neg(sess, &y1)],
        };
        AbstractAdditiveTensor { shares }
    }
}

modelled!(PlacementMul::mul, AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor, AdtMulOp);
modelled!(PlacementMul::mul, AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor, AdtMulOp);
modelled!(PlacementMul::mul, AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor, AdtMulOp);
modelled!(PlacementMul::mul, AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor, AdtMulOp);

hybrid_kernel! {
    AdtMulOp,
    [
        (AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor => Self::ring_adt_kernel),
        (AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor => Self::adt_ring_kernel),
        (AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor => Self::ring_adt_kernel),
    ]
}

impl AdtMulOp {
    fn ring_adt_kernel<S: Session, R>(
        sess: &S,
        add: &AdditivePlacement,
        x: R,
        y: AbstractAdditiveTensor<R>,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<S, R, R, R>,
    {
        let (player0, player1) = add.host_placements();

        let AbstractAdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x * y0);
        let z1 = with_context!(player1, sess, x * y1);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }

    fn adt_ring_kernel<S: Session, R>(
        sess: &S,
        add: &AdditivePlacement,
        x: AbstractAdditiveTensor<R>,
        y: R,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<S, R, R, R>,
    {
        let (player0, player1) = add.host_placements();

        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;

        let z0 = with_context!(player0, sess, x0 * y);
        let z1 = with_context!(player1, sess, x1 * y);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }
}

modelled!(PlacementShl::shl, AdditivePlacement, attributes[amount: usize] (Additive64Tensor) -> Additive64Tensor, AdtShlOp);
modelled!(PlacementShl::shl, AdditivePlacement, attributes[amount: usize] (Additive128Tensor) -> Additive128Tensor, AdtShlOp);

hybrid_kernel! {
    AdtShlOp,
    [
        (AdditivePlacement, (Additive64Tensor) -> Additive64Tensor => attributes[amount] Self::kernel),
        (AdditivePlacement, (Additive128Tensor) -> Additive128Tensor => attributes[amount] Self::kernel),
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

impl<S: Session, R> TruncMaskGen<S, cs!(Shape), R> for HostPlacement
where
    PrfKey: KnownType<S>,
    Seed: KnownType<S>,
    Shape: KnownType<S>,
    R: RingSize + Clone,
    HostPlacement: PlacementDeriveSeed<S, cs!(PrfKey), cs!(Seed)>,
    HostPlacement: PlacementSampleBits<S, cs!(Seed), cs!(Shape), R>,
    HostPlacement: PlacementSampleUniform<S, cs!(Seed), cs!(Shape), R>,
    HostPlacement: PlacementKeyGen<S, cs!(PrfKey)>,
    HostPlacement: PlacementSub<S, R, R, R>,
    HostPlacement: PlacementShr<S, R, R>,
    HostPlacement: PlacementShl<S, R, R>,
{
    fn gen_trunc_mask(
        &self,
        sess: &S,
        amount: usize,
        shape: &cs!(Shape),
    ) -> (
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
    ) {
        let key = self.gen_key(sess);

        let nonce = RawNonce::generate();
        let seed = self.derive_seed(sess, nonce, &key);

        let r = self.sample_uniform(sess, &seed, shape);
        let r_msb = self.shr(sess, R::SIZE - 1, &r);
        let r_top = self.shr(sess, amount + 1, &self.shl(sess, 1, &r));

        let share = |x| {
            // TODO(Dragos) this could probably be optimized by sending the key to p0
            let nonce = RawNonce::generate();
            let seed = self.derive_seed(sess, nonce, &key);
            let x0 = self.sample_uniform(sess, &seed, shape);
            let x1 = self.sub(sess, x, &x0);
            AbstractAdditiveTensor { shares: [x0, x1] }
        };

        let r_shared = share(&r);
        let r_top_shared = share(&r_top);
        let r_msb_shared = share(&r_msb);

        (r_shared, r_top_shared, r_msb_shared)
    }
}

impl<S: Session, R>
    PlacementTruncPrProvider<S, AbstractAdditiveTensor<R>, AbstractAdditiveTensor<R>>
    for AdditivePlacement
where
    R: RingSize,
    Shape: KnownType<S>,
    HostPlacement: TruncMaskGen<S, cs!(Shape), R>,
    AdditivePlacement: PlacementAdd<
        S,
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
    >,
    AdditivePlacement: PlacementSub<S, R, AbstractAdditiveTensor<R>, AbstractAdditiveTensor<R>>,
    AdditivePlacement: PlacementAdd<S, AbstractAdditiveTensor<R>, R, AbstractAdditiveTensor<R>>,
    AdditivePlacement: PlacementMul<S, AbstractAdditiveTensor<R>, R, AbstractAdditiveTensor<R>>,
    AdditivePlacement: PlacementShl<S, AbstractAdditiveTensor<R>, AbstractAdditiveTensor<R>>,
    AdditivePlacement: PlacementSub<
        S,
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
    >,
    AdditivePlacement: PlacementSub<S, AbstractAdditiveTensor<R>, R, AbstractAdditiveTensor<R>>,
    HostPlacement: PlacementOnes<S, cs!(Shape), R>,
    HostPlacement: PlacementReveal<S, AbstractAdditiveTensor<R>, R>,
    HostPlacement: PlacementShape<S, R, cs!(Shape)>,
    HostPlacement: PlacementShl<S, R, R>,
    HostPlacement: PlacementShr<S, R, R>,
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
        let ones = player_a.ones(sess, &shape);
        let upshifter = player_a.shl(sess, R::SIZE - 2, &ones);
        let downshifter = player_a.shl(sess, R::SIZE - 2 - amount, &ones);

        let x_positive = self.add(sess, x, &upshifter);
        let masked = adt.add(sess, &x_positive, &r);
        let c = player_a.reveal(sess, &masked);
        let c_no_msb = player_a.shl(sess, 1, &c);
        let c_top = player_a.shr(sess, amount + 1, &c_no_msb);
        let c_msb = player_a.shr(sess, R::SIZE - 1, &c);
        let b = with_context!(adt, sess, r_msb + c_msb - r_msb * c_msb - r_msb * c_msb); // a xor b = a+b-2ab
        let shifted_b = self.shl(sess, R::SIZE - 1 - amount, &b);
        let y_positive = with_context!(adt, sess, c_top - r_top + shifted_b);
        let y = with_context!(adt, sess, y_positive - downshifter);
        y
    }
}

modelled!(PlacementRepToAdt::rep_to_adt, AdditivePlacement, (Replicated64Tensor) -> Additive64Tensor, RepToAdtOp);
modelled!(PlacementRepToAdt::rep_to_adt, AdditivePlacement, (Replicated128Tensor) -> Additive128Tensor, RepToAdtOp);

hybrid_kernel! {
    RepToAdtOp,
    [
        (AdditivePlacement, (Replicated64Tensor) -> Additive64Tensor => Self::rep_to_adt_kernel),
        (AdditivePlacement, (Replicated128Tensor) -> Additive128Tensor => Self::rep_to_adt_kernel),
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
    {
        let (adt_player0, adt_player1) = adt.host_placements();
        let (rep_player0, rep_player1, rep_player2) = x.placement().host_placements();

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
        AbstractAdditiveTensor { shares }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kernels::NewSyncSession, ring::AbstractRingTensor};
    use ndarray::array;

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

        let x = Additive64Tensor {
            shares: [
                AbstractRingTensor::from_raw_plc(array![1, 2, 3], alice.clone()),
                AbstractRingTensor::from_raw_plc(array![4, 5, 6], bob.clone()),
            ],
        };

        let y = Additive64Tensor {
            shares: [
                AbstractRingTensor::from_raw_plc(array![7, 8, 9], alice.clone()),
                AbstractRingTensor::from_raw_plc(array![1, 2, 3], bob.clone()),
            ],
        };

        let sess = NewSyncSession::default();
        let AbstractAdditiveTensor { shares: [z0, z1] } = adt.add(&sess, &x, &y);

        assert_eq!(
            z0,
            AbstractRingTensor::from_raw_plc(array![1 + 7, 2 + 8, 3 + 9], alice)
        );
        assert_eq!(
            z1,
            AbstractRingTensor::from_raw_plc(array![4 + 1, 5 + 2, 6 + 3], bob)
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

        let x = Additive64Tensor {
            shares: [
                AbstractRingTensor::from_raw_plc(array![80908, 0, 40454], alice),
                AbstractRingTensor::from_raw_plc(array![0, -80908_i64 as u64, 40454], bob),
            ],
        };

        let sess = NewSyncSession::default();
        let x_trunc = adt.trunc_pr(&sess, 8, &carole, &x);
        let _y = carole.reveal(&sess, &x_trunc);

        // TODO allowed as long as \in {316, 317}
        assert_eq!(
            _y.0,
            array![
                std::num::Wrapping(316),
                std::num::Wrapping(-316_i64 as u64),
                std::num::Wrapping(316)
            ]
            .into_dyn()
        );
    }
}
