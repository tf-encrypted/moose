use crate::computation::{
    AdditivePlacement, AdtAddOp, AdtFillOp, AdtMulOp, AdtRevealOp, AdtShlOp, AdtSubOp, Constant,
    HostPlacement, KnownType, Placed, RepToAdtOp, ReplicatedPlacement,
};
use crate::kernels::{
    Context, PlacementAdd, PlacementDeriveSeed, PlacementFill, PlacementKeyGen, PlacementMul,
    PlacementNeg, PlacementOnes, PlacementPlace, PlacementRepToAdt, PlacementReveal,
    PlacementSampleBits, PlacementSampleUniform, PlacementShape, PlacementShl, PlacementShr,
    PlacementSub, PlacementTruncPrProvider,
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

impl<C: Context, R> PlacementPlace<C, AbstractAdditiveTensor<R>> for AdditivePlacement
where
    AbstractAdditiveTensor<R>: Placed<Placement = AdditivePlacement>,
    HostPlacement: PlacementPlace<C, R>,
{
    fn place(&self, ctx: &C, x: AbstractAdditiveTensor<R>) -> AbstractAdditiveTensor<R> {
        if self == &x.placement() {
            x
        } else {
            let AbstractAdditiveTensor { shares: [x0, x1] } = x;

            let (player0, player1) = self.host_placements();

            AbstractAdditiveTensor {
                shares: [player0.place(ctx, x0), player1.place(ctx, x1)],
            }
        }
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
    fn kernel<C: Context, ShapeT, RingT>(
        ctx: &C,
        plc: &AdditivePlacement,
        value: Constant,
        shape: ShapeT,
    ) -> AbstractAdditiveTensor<RingT>
    where
        HostPlacement: PlacementFill<C, ShapeT, RingT>,
    {
        // TODO should really return PublicAdditiveTensor, but we don't have that type yet

        let (player0, player1) = plc.host_placements();

        let shares = [
            player0.fill(ctx, value, &shape),
            player1.fill(ctx, Constant::Ring64(0), &shape),
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
    fn kernel<C: Context, RingT>(
        ctx: &C,
        plc: &HostPlacement,
        xe: AbstractAdditiveTensor<RingT>,
    ) -> RingT
    where
        HostPlacement: PlacementAdd<C, RingT, RingT, RingT>,
    {
        let AbstractAdditiveTensor { shares: [x0, x1] } = &xe;
        with_context!(plc, ctx, x1 + x0)
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
    fn adt_adt_kernel<C: Context, RingT>(
        ctx: &C,
        adt: &AdditivePlacement,
        x: AbstractAdditiveTensor<RingT>,
        y: AbstractAdditiveTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        HostPlacement: PlacementAdd<C, RingT, RingT, RingT>,
    {
        let (player0, player1) = adt.host_placements();

        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;
        let AbstractAdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, ctx, x0 + y0);
        let z1 = with_context!(player1, ctx, x1 + y1);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }

    fn adt_ring_kernel<C: Context, RingT>(
        ctx: &C,
        adt: &AdditivePlacement,
        x: AbstractAdditiveTensor<RingT>,
        y: RingT,
    ) -> AbstractAdditiveTensor<RingT>
    where
        RingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, RingT, RingT, RingT>,
        AdditivePlacement: PlacementPlace<C, AbstractAdditiveTensor<RingT>>,
    {
        let (player0, player1) = adt.host_placements();
        let y_plc = y.placement();

        let AbstractAdditiveTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, ctx, x0 + y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, ctx, x1 + y)],
            _ => [with_context!(player0, ctx, x0 + y), x1],
        };
        adt.place(ctx, AbstractAdditiveTensor { shares })
    }

    fn ring_adt_kernel<C: Context, RingT>(
        ctx: &C,
        adt: &AdditivePlacement,
        x: RingT,
        y: AbstractAdditiveTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        RingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, RingT, RingT, RingT>,
        AdditivePlacement: PlacementPlace<C, AbstractAdditiveTensor<RingT>>,
    {
        let (player0, player1) = adt.host_placements();
        let x_plc = x.placement();

        let AbstractAdditiveTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, ctx, y0 + x), y1],
            _ if x_plc == player1 => [y0, with_context!(player1, ctx, x + y1)],
            _ => [with_context!(player0, ctx, x + y0), y1],
        };
        adt.place(ctx, AbstractAdditiveTensor { shares })
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
    fn adt_adt_kernel<C: Context, R>(
        ctx: &C,
        adt: &AdditivePlacement,
        x: AbstractAdditiveTensor<R>,
        y: AbstractAdditiveTensor<R>,
    ) -> AbstractAdditiveTensor<R>
    where
        HostPlacement: PlacementSub<C, R, R, R>,
    {
        let (player0, player1) = adt.host_placements();

        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;
        let AbstractAdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, ctx, x0 - y0);
        let z1 = with_context!(player1, ctx, x1 - y1);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }

    fn adt_ring_kernel<C: Context, R>(
        ctx: &C,
        adt: &AdditivePlacement,
        x: AbstractAdditiveTensor<R>,
        y: R,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<C, R, R, R>,
        AdditivePlacement: PlacementPlace<C, AbstractAdditiveTensor<R>>,
    {
        let (player0, player1) = adt.host_placements();
        let y_plc = y.placement();

        let AbstractAdditiveTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, ctx, x0 - y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, ctx, x1 - y)],
            _ => [with_context!(player0, ctx, x0 - y), x1],
        };
        adt.place(ctx, AbstractAdditiveTensor { shares })
    }

    fn ring_adt_kernel<C: Context, R>(
        ctx: &C,
        adt: &AdditivePlacement,
        x: R,
        y: AbstractAdditiveTensor<R>,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<C, R, R, R>,
        HostPlacement: PlacementNeg<C, R, R>,
        AdditivePlacement: PlacementPlace<C, AbstractAdditiveTensor<R>>,
    {
        let (player0, player1) = adt.host_placements();
        let x_plc = x.placement();

        let AbstractAdditiveTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, ctx, x - y0), player1.neg(ctx, &y1)],
            _ if x_plc == player1 => [player0.neg(ctx, &y0), with_context!(player1, ctx, x - y1)],
            _ => [with_context!(player0, ctx, x - y0), player1.neg(ctx, &y1)],
        };
        adt.place(ctx, AbstractAdditiveTensor { shares })
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
    fn ring_adt_kernel<C: Context, R>(
        ctx: &C,
        adt: &AdditivePlacement,
        x: R,
        y: AbstractAdditiveTensor<R>,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<C, R, R, R>,
    {
        let (player0, player1) = adt.host_placements();

        let AbstractAdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, ctx, x * y0);
        let z1 = with_context!(player1, ctx, x * y1);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }

    fn adt_ring_kernel<C: Context, R>(
        ctx: &C,
        adt: &AdditivePlacement,
        x: AbstractAdditiveTensor<R>,
        y: R,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<C, R, R, R>,
    {
        let (player0, player1) = adt.host_placements();

        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;

        let z0 = with_context!(player0, ctx, x0 * y);
        let z1 = with_context!(player1, ctx, x1 * y);

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
    fn kernel<C: Context, RingT>(
        ctx: &C,
        plc: &AdditivePlacement,
        amount: usize,
        x: AbstractAdditiveTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        HostPlacement: PlacementShl<C, RingT, RingT>,
    {
        let (player0, player1) = plc.host_placements();
        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;
        let z0 = player0.shl(ctx, amount, x0);
        let z1 = player1.shl(ctx, amount, x1);
        AbstractAdditiveTensor { shares: [z0, z1] }
    }
}

pub trait BitCompose<C: Context, R> {
    fn bit_compose(&self, ctx: &C, bits: &[R]) -> R;
}

impl<C: Context, R> BitCompose<C, R> for HostPlacement
where
    R: Clone,
    HostPlacement: PlacementShl<C, R, R>,
    HostPlacement: TreeReduce<C, R>,
{
    fn bit_compose(&self, ctx: &C, bits: &[R]) -> R {
        let shifted_bits: Vec<_> = (0..bits.len())
            .map(|i| self.shl(ctx, i, &bits[i]))
            .collect();
        self.tree_reduce(ctx, &shifted_bits)
    }
}

pub trait TreeReduce<C: Context, R> {
    fn tree_reduce(&self, ctx: &C, sequence: &[R]) -> R;
}

impl<C: Context, R> TreeReduce<C, R> for HostPlacement
where
    R: Clone,
    HostPlacement: PlacementAdd<C, R, R, R>,
{
    fn tree_reduce(&self, ctx: &C, sequence: &[R]) -> R {
        let n = sequence.len();
        if n == 1 {
            sequence[0].clone()
        } else {
            let mut reduced: Vec<_> = (0..n / 2)
                .map(|i| {
                    let x0: &R = &sequence[2 * i];
                    let x1: &R = &sequence[2 * i + 1];
                    self.add(ctx, x0, x1)
                })
                .collect();
            if n % 2 == 1 {
                reduced.push(sequence[n - 1].clone());
            }
            self.tree_reduce(ctx, &reduced)
        }
    }
}

pub trait TruncMaskGen<C: Context, ShapeT, RingT> {
    fn gen_trunc_mask(
        &self,
        ctx: &C,
        amount: usize,
        shape: &ShapeT,
    ) -> (
        AbstractAdditiveTensor<RingT>,
        AbstractAdditiveTensor<RingT>,
        AbstractAdditiveTensor<RingT>,
    );
}

impl<C: Context, R> TruncMaskGen<C, cs!(Shape), R> for HostPlacement
where
    PrfKey: KnownType<C>,
    Seed: KnownType<C>,
    Shape: KnownType<C>,
    R: RingSize + Clone,
    HostPlacement: PlacementDeriveSeed<C, cs!(PrfKey), cs!(Seed)>,
    HostPlacement: PlacementSampleBits<C, cs!(Seed), cs!(Shape), R>,
    HostPlacement: PlacementSampleUniform<C, cs!(Seed), cs!(Shape), R>,
    HostPlacement: PlacementKeyGen<C, cs!(PrfKey)>,
    HostPlacement: PlacementSub<C, R, R, R>,
    HostPlacement: PlacementShr<C, R, R>,
    HostPlacement: PlacementShl<C, R, R>,
{
    fn gen_trunc_mask(
        &self,
        ctx: &C,
        amount: usize,
        shape: &cs!(Shape),
    ) -> (
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
    ) {
        let key = self.gen_key(ctx);

        let nonce = RawNonce::generate();
        let seed = self.derive_seed(ctx, nonce, &key);

        let r = self.sample_uniform(ctx, &seed, shape);
        let r_msb = self.shr(ctx, R::SIZE - 1, &r);
        let r_top = self.shr(ctx, amount + 1, &self.shl(ctx, 1, &r));

        let share = |x| {
            // TODO(Dragos) this could probably be optimized by sending the key to p0
            let nonce = RawNonce::generate();
            let seed = self.derive_seed(ctx, nonce, &key);
            let x0 = self.sample_uniform(ctx, &seed, shape);
            let x1 = self.sub(ctx, x, &x0);
            AbstractAdditiveTensor { shares: [x0, x1] }
        };

        let r_shared = share(&r);
        let r_top_shared = share(&r_top);
        let r_msb_shared = share(&r_msb);

        (r_shared, r_top_shared, r_msb_shared)
    }
}

impl<C: Context, R>
    PlacementTruncPrProvider<C, AbstractAdditiveTensor<R>, AbstractAdditiveTensor<R>>
    for AdditivePlacement
where
    R: RingSize,
    Shape: KnownType<C>,
    HostPlacement: TruncMaskGen<C, cs!(Shape), R>,
    AdditivePlacement: PlacementAdd<
        C,
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
    >,
    AdditivePlacement: PlacementSub<C, R, AbstractAdditiveTensor<R>, AbstractAdditiveTensor<R>>,
    AdditivePlacement: PlacementAdd<C, AbstractAdditiveTensor<R>, R, AbstractAdditiveTensor<R>>,
    AdditivePlacement: PlacementMul<C, AbstractAdditiveTensor<R>, R, AbstractAdditiveTensor<R>>,
    AdditivePlacement: PlacementShl<C, AbstractAdditiveTensor<R>, AbstractAdditiveTensor<R>>,
    AdditivePlacement: PlacementSub<
        C,
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
        AbstractAdditiveTensor<R>,
    >,
    AdditivePlacement: PlacementSub<C, AbstractAdditiveTensor<R>, R, AbstractAdditiveTensor<R>>,
    HostPlacement: PlacementOnes<C, cs!(Shape), R>,
    HostPlacement: PlacementReveal<C, AbstractAdditiveTensor<R>, R>,
    HostPlacement: PlacementShape<C, R, cs!(Shape)>,
    HostPlacement: PlacementShl<C, R, R>,
    HostPlacement: PlacementShr<C, R, R>,
{
    fn trunc_pr(
        &self,
        ctx: &C,
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

        let shape = player_a.shape(ctx, x0);
        let (r, r_top, r_msb) = provider.gen_trunc_mask(ctx, amount, &shape);

        // NOTE we consider input is always signed, and the following positive
        // conversion would be optional for unsigned numbers
        // NOTE we assume that input numbers are in range -2^{k-2} <= x < 2^{k-2}
        // so that 0 <= x + 2^{k-2} < 2^{k-1}
        // TODO we could insert debug_assert! to check above conditions
        let ones = player_a.ones(ctx, &shape);
        let upshifter = player_a.shl(ctx, R::SIZE - 2, &ones);
        let downshifter = player_a.shl(ctx, R::SIZE - 2 - amount, &ones);

        let x_positive = self.add(ctx, x, &upshifter);
        let masked = adt.add(ctx, &x_positive, &r);
        let c = player_a.reveal(ctx, &masked);
        let c_no_msb = player_a.shl(ctx, 1, &c);
        let c_top = player_a.shr(ctx, amount + 1, &c_no_msb);
        let c_msb = player_a.shr(ctx, R::SIZE - 1, &c);
        let b = with_context!(adt, ctx, r_msb + c_msb - r_msb * c_msb - r_msb * c_msb); // a xor b = a+b-2ab
        let shifted_b = self.shl(ctx, R::SIZE - 1 - amount, &b);
        let y_positive = with_context!(adt, ctx, c_top - r_top + shifted_b);
        let y = with_context!(adt, ctx, y_positive - downshifter);
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
    fn rep_to_adt_kernel<C: Context, RingT>(
        ctx: &C,
        adt: &AdditivePlacement,
        x: AbstractReplicatedTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        AbstractReplicatedTensor<RingT>: Placed<Placement = ReplicatedPlacement>,
        HostPlacement: PlacementAdd<C, RingT, RingT, RingT>,
        AdditivePlacement: PlacementPlace<C, AbstractAdditiveTensor<RingT>>,
    {
        let (adt_player0, adt_player1) = adt.host_placements();
        let (rep_player0, rep_player1, rep_player2) = x.placement().host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let shares = match () {
            _ if adt_player0 == rep_player0 => {
                let y0 = with_context!(rep_player0, ctx, x00 + x10);
                let y1 = match () {
                    _ if adt_player1 == rep_player1 => x21,
                    _ if adt_player1 == rep_player2 => x22,
                    _ => x21,
                };
                [y0, y1]
            }
            _ if adt_player0 == rep_player1 => {
                let y0 = with_context!(rep_player1, ctx, x11 + x21);
                let y1 = match () {
                    _ if adt_player1 == rep_player2 => x02,
                    _ if adt_player1 == rep_player0 => x00,
                    _ => x02,
                };
                [y0, y1]
            }
            _ if adt_player0 == rep_player2 => {
                let y0 = with_context!(rep_player2, ctx, x22 + x02);
                let y1 = match () {
                    _ if adt_player1 == rep_player0 => x10,
                    _ if adt_player1 == rep_player1 => x11,
                    _ => x10,
                };
                [y0, y1]
            }
            _ if adt_player1 == rep_player0 => {
                let y0 = x21;
                let y1 = with_context!(rep_player0, ctx, x00 + x10);
                [y0, y1]
            }
            _ if adt_player1 == rep_player1 => {
                let y0 = x02;
                let y1 = with_context!(rep_player1, ctx, x11 + x21);
                [y0, y1]
            }
            _ if adt_player1 == rep_player2 => {
                let y0 = x10;
                let y1 = with_context!(rep_player2, ctx, x22 + x02);
                [y0, y1]
            }
            _ => {
                let y0 = with_context!(rep_player0, ctx, x00 + x10);
                let y1 = x21;
                [y0, y1]
            }
        };
        adt.place(ctx, AbstractAdditiveTensor { shares })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kernels::ConcreteContext, ring::AbstractRingTensor};
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

        let ctx = ConcreteContext::default();
        let AbstractAdditiveTensor { shares: [z0, z1] } = adt.add(&ctx, &x, &y);

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

        let ctx = ConcreteContext::default();
        let x_trunc = adt.trunc_pr(&ctx, 8, &carole, &x);
        let _y = carole.reveal(&ctx, &x_trunc);

        // TODO allowed as long as \in {316, 317}
        // assert_eq!(
        //     _y.0,
        //     array![
        //         std::num::Wrapping(316),
        //         std::num::Wrapping(-316_i64 as u64),
        //         std::num::Wrapping(316)
        //     ]
        //     .into_dyn()
        // );
    }
}
