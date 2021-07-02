use crate::additive::{AbstractAdditiveTensor, Additive128Tensor, Additive64Tensor};
use crate::bit::BitTensor;
use crate::computation::KnownType;
use crate::computation::{
    AdditivePlacement, HostPlacement, Placed, RepAddOp, RepMulOp, RepRevealOp, RepSetupOp,
    RepShareOp, RepToAdtOp, ReplicatedPlacement,
};
use crate::kernels::{
    Context, PlacementAdd, PlacementDeriveSeed, PlacementFill, PlacementKeyGen, PlacementMul,
    PlacementMulSetup, PlacementRepToAdt, PlacementReveal, PlacementSampleUniform, PlacementShape,
    PlacementShareSetup, PlacementSub,
};
use crate::prim::{PrfKey, RawNonce, Seed};
use crate::ring::{Ring128Tensor, Ring64Tensor};
use crate::standard::Shape;
use macros::with_context;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractReplicatedTensor<R> {
    pub shares: [[R; 2]; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractReplicatedSetup<K> {
    pub keys: [[K; 2]; 3],
}

pub type Replicated64Tensor = AbstractReplicatedTensor<Ring64Tensor>;

pub type Replicated128Tensor = AbstractReplicatedTensor<Ring128Tensor>;

pub type ReplicatedBitTensor = AbstractReplicatedTensor<BitTensor>;

pub type ReplicatedSetup = AbstractReplicatedSetup<PrfKey>;

impl<R> Placed for AbstractReplicatedTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Self::Placement {
        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = self;

        let owner0 = x00.placement().owner;
        assert_eq!(x10.placement().owner, owner0);

        let owner1 = x11.placement().owner;
        assert_eq!(x21.placement().owner, owner1);

        let owner2 = x22.placement().owner;
        assert_eq!(x02.placement().owner, owner2);

        let owners = [owner0, owner1, owner2];
        ReplicatedPlacement { owners }
    }
}

impl<K> Placed for AbstractReplicatedSetup<K>
where
    K: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Self::Placement {
        let AbstractReplicatedSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = self;

        let owner0 = k00.placement().owner;
        assert_eq!(k10.placement().owner, owner0);

        let owner1 = k11.placement().owner;
        assert_eq!(k21.placement().owner, owner1);

        let owner2 = k22.placement().owner;
        assert_eq!(k02.placement().owner, owner2);

        let owners = [owner0, owner1, owner2];
        ReplicatedPlacement { owners }
    }
}

// TODO modelled! for RepSetupOp

hybrid_kernel! {
    RepSetupOp,
    [
        (ReplicatedPlacement, () -> ReplicatedSetup => Self::kernel),
    ]
}

impl RepSetupOp {
    fn kernel<C: Context, K: Clone>(
        ctx: &C,
        rep: &ReplicatedPlacement,
    ) -> AbstractReplicatedSetup<K>
    where
        HostPlacement: PlacementKeyGen<C, K>,
    {
        let (owner0, owner1, owner2) = rep.host_placements();

        let k0 = owner0.keygen(ctx);
        let k1 = owner1.keygen(ctx);
        let k2 = owner2.keygen(ctx);

        AbstractReplicatedSetup {
            keys: [[k0.clone(), k1.clone()], [k1, k2.clone()], [k2, k0]],
        }
    }
}

modelled!(PlacementShareSetup::share, ReplicatedPlacement, (ReplicatedSetup, Ring64Tensor) -> Replicated64Tensor, RepShareOp);
modelled!(PlacementShareSetup::share, ReplicatedPlacement, (ReplicatedSetup, Ring128Tensor) -> Replicated128Tensor, RepShareOp);
// modelled!(PlacementShareSetup::share, ReplicatedPlacement, (ReplicatedSetup, BitTensor) -> ReplicatedBitTensor, RepShareOp); // TODO

hybrid_kernel! {
    RepShareOp,
    [
        (ReplicatedPlacement, (ReplicatedSetup, Ring64Tensor) -> Replicated64Tensor => Self::kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Ring128Tensor) -> Replicated128Tensor => Self::kernel),
        // (ReplicatedPlacement, (ReplicatedSetup, BitTensor) -> ReplicatedBitTensor => Self::kernel),
    ]
}

impl RepShareOp {
    fn kernel<C: Context, SeedT, ShapeT, KeyT, RingT>(
        ctx: &C,
        plc: &ReplicatedPlacement,
        setup: AbstractReplicatedSetup<KeyT>,
        x: RingT,
    ) -> AbstractReplicatedTensor<RingT>
    where
        RingT: Clone + Placed<Placement = HostPlacement>,
        HostPlacement: PlacementShape<C, RingT, ShapeT>,
        HostPlacement: PlacementSampleUniform<C, SeedT, ShapeT, RingT>,
        HostPlacement: PlacementFill<C, ShapeT, RingT>,
        HostPlacement: PlacementDeriveSeed<C, KeyT, SeedT>,
        HostPlacement: PlacementAdd<C, RingT, RingT, RingT>,
        HostPlacement: PlacementSub<C, RingT, RingT, RingT>,
    {
        let x_owner = x.placement();

        let AbstractReplicatedSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = &setup;

        let (owner0, owner1, owner2) = plc.host_placements();

        let shares = match () {
            _ if x_owner == owner0 => {
                let sync_key = RawNonce::generate();
                let shape = x_owner.shape(ctx, &x);

                let seed0 = owner0.derive_seed(ctx, sync_key.clone(), k00);
                let x00 = x_owner.sample_uniform(ctx, &seed0, &shape);
                let x10 = with_context!(x_owner, ctx, x - x00);

                let seed2 = owner2.derive_seed(ctx, sync_key, k02);
                let x22 = owner2.fill(ctx, 0, &shape);
                let x02 = owner2.sample_uniform(ctx, &seed2, &shape);

                let x11 = x10.clone();
                let x21 = owner1.fill(ctx, 0, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ if x_owner == owner1 => {
                let sync_key = RawNonce::generate();
                let shape = x_owner.shape(ctx, &x);

                let seed1 = owner1.derive_seed(ctx, sync_key.clone(), k11);
                let x11 = x_owner.sample_uniform(ctx, &seed1, &shape);
                let x21 = with_context!(x_owner, ctx, x - x11);

                let seed0 = owner0.derive_seed(ctx, sync_key, k10);
                let x00 = owner0.fill(ctx, 0, &shape);
                let x10 = owner0.sample_uniform(ctx, &seed0, &shape);

                let x22 = x21.clone();
                let x02 = owner2.fill(ctx, 0, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ if x_owner == owner2 => {
                let sync_key = RawNonce::generate();
                let shape = x_owner.shape(ctx, &x);

                let seed2 = owner2.derive_seed(ctx, sync_key.clone(), k22);
                let x22 = owner2.sample_uniform(ctx, &seed2, &shape);
                let x02 = with_context!(x_owner, ctx, x - x22);

                let seed1 = owner1.derive_seed(ctx, sync_key, k21);
                let x11 = owner1.fill(ctx, 0, &shape);
                let x21 = owner1.sample_uniform(ctx, &seed1, &shape);

                let x00 = x02.clone();
                let x10 = owner0.fill(ctx, 0, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ => {
                // in this case, where x_owner is _not_ among the replicated players,
                // we cannot use the zeros optimization trick but we can still make sure
                // that seeds are used as much as possible instead of dense random tensors;
                // however, we must make sure keys are not revealed to x_owner and only seeds

                let sync_key0 = RawNonce::generate();
                let sync_key1 = RawNonce::generate();
                let shape = x_owner.shape(ctx, &x);

                let seed00 = owner0.derive_seed(ctx, sync_key0.clone(), k00);
                let seed02 = owner2.derive_seed(ctx, sync_key0, k02);

                let seed11 = owner1.derive_seed(ctx, sync_key1.clone(), k11);
                let seed10 = owner0.derive_seed(ctx, sync_key1, k10);

                let x0 = x_owner.sample_uniform(ctx, &seed00, &shape);
                let x1 = x_owner.sample_uniform(ctx, &seed11, &shape);
                let x2 = with_context!(x_owner, ctx, x - x0 - x1);

                let x00 = owner0.sample_uniform(ctx, &seed00, &shape);
                let x10 = owner0.sample_uniform(ctx, &seed10, &shape);

                let x11 = owner1.sample_uniform(ctx, &seed11, &shape);
                let x21 = x2.clone();

                let x22 = x2;
                let x02 = owner2.sample_uniform(ctx, &seed02, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
        };

        AbstractReplicatedTensor { shares }
    }
}

modelled!(PlacementReveal::reveal, HostPlacement, (Replicated64Tensor) -> Ring64Tensor, RepRevealOp);
modelled!(PlacementReveal::reveal, HostPlacement, (Replicated128Tensor) -> Ring128Tensor, RepRevealOp);
modelled!(PlacementReveal::reveal, HostPlacement, (ReplicatedBitTensor) -> BitTensor, RepRevealOp);

hybrid_kernel! {
    RepRevealOp,
    [
        (HostPlacement, (Replicated64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Replicated128Tensor) -> Ring128Tensor => Self::kernel),
        (HostPlacement, (ReplicatedBitTensor) -> BitTensor => Self::kernel),
    ]
}

impl RepRevealOp {
    fn kernel<C: Context, R: Clone>(
        ctx: &C,
        receiver: &HostPlacement,
        xe: AbstractReplicatedTensor<R>,
    ) -> R
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &xe;

        let (owner0, owner1, owner2) = xe.placement().host_placements();

        match () {
            _ if receiver == &owner0 => {
                // make sure to use both shares on owner0
                with_context!(receiver, ctx, x00 + x10 + x21)
            }
            _ if receiver == &owner1 => {
                // make sure to use both shares on owner1
                with_context!(receiver, ctx, x02 + x11 + x21)
            }
            _ if receiver == &owner2 => {
                // make sure to use both shares on owner2
                with_context!(receiver, ctx, x02 + x10 + x22)
            }
            _ => {
                with_context!(receiver, ctx, x00 + x10 + x21)
            }
        }
    }
}

modelled!(PlacementAdd::add, ReplicatedPlacement, (Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Ring64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Ring128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Replicated64Tensor, Ring64Tensor) -> Replicated64Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Replicated128Tensor, Ring128Tensor) -> Replicated128Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor, RepAddOp);

hybrid_kernel! {
    RepAddOp,
    [
        (ReplicatedPlacement, (Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (Ring64Tensor, Replicated64Tensor) -> Replicated64Tensor => Self::ring_rep_kernel),
        (ReplicatedPlacement, (Ring128Tensor, Replicated128Tensor) -> Replicated128Tensor => Self::ring_rep_kernel),
        (ReplicatedPlacement, (Replicated64Tensor, Ring64Tensor) -> Replicated64Tensor => Self::rep_ring_kernel),
        (ReplicatedPlacement, (Replicated128Tensor, Ring128Tensor) -> Replicated128Tensor => Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => Self::rep_rep_kernel),
    ]
}

impl RepAddOp {
    fn rep_rep_kernel<C: Context, R>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedTensor<R>,
        y: AbstractReplicatedTensor<R>,
    ) -> AbstractReplicatedTensor<R>
    where
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (owner0, owner1, owner2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let AbstractReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(owner0, ctx, x00 + y00);
        let z10 = with_context!(owner0, ctx, x10 + y10);

        let z11 = with_context!(owner1, ctx, x11 + y11);
        let z21 = with_context!(owner1, ctx, x21 + y21);

        let z22 = with_context!(owner2, ctx, x22 + y22);
        let z02 = with_context!(owner2, ctx, x02 + y02);

        AbstractReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn ring_rep_kernel<C: Context, R>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: R,
        y: AbstractReplicatedTensor<R>,
    ) -> AbstractReplicatedTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (owner0, owner1, owner2) = rep.host_placements();
        let x_plc = x.placement();

        let AbstractReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = y;

        let shares = match () {
            _ if x_plc == owner0 => {
                // add x to y0
                [
                    [with_context!(owner0, ctx, x + y00), y10],
                    [y11, y21],
                    [y22, with_context!(owner2, ctx, x + y02)],
                ]
            }
            _ if x_plc == owner1 => {
                // add x to y1
                [
                    [y00, with_context!(owner0, ctx, x + y10)],
                    [with_context!(owner1, ctx, x + y11), y21],
                    [y22, y02],
                ]
            }
            _ if x_plc == owner2 => {
                // add x to y2
                [
                    [y00, y10],
                    [y11, with_context!(owner1, ctx, x + y21)],
                    [with_context!(owner2, ctx, x + y22), y02],
                ]
            }
            _ => {
                // add x to y0; we could randomize this
                [
                    [with_context!(owner0, ctx, x + y00), y10],
                    [y11, y21],
                    [y22, with_context!(owner2, ctx, x + y02)],
                ]
            }
        };

        AbstractReplicatedTensor { shares }
    }

    fn rep_ring_kernel<C: Context, R>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedTensor<R>,
        y: R,
    ) -> AbstractReplicatedTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let y_plc = y.placement();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let shares = match () {
            _ if y_plc == player0 => {
                // add y to x0
                [
                    [with_context!(player0, ctx, x00 + y), x10],
                    [x11, x21],
                    [x22, with_context!(player2, ctx, x02 + y)],
                ]
            }
            _ if y_plc == player1 => {
                // add y to x1
                [
                    [x00, with_context!(player0, ctx, x10 + y)],
                    [with_context!(player1, ctx, x11 + y), x21],
                    [x22, x02],
                ]
            }
            _ if y_plc == player2 => {
                // add y to x2
                [
                    [x00, x10],
                    [x11, with_context!(player1, ctx, x21 + y)],
                    [with_context!(player2, ctx, x22 + y), x02],
                ]
            }
            _ => {
                // add y to x0; we could randomize this
                [
                    [with_context!(player0, ctx, x00 + y), x10],
                    [x11, x21],
                    [x22, with_context!(player2, ctx, x02 + y)],
                ]
            }
        };

        AbstractReplicatedTensor { shares }
    }
}

modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Ring64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Ring128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Ring64Tensor) -> Replicated64Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Ring128Tensor) -> Replicated128Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor, RepMulOp);

hybrid_kernel! {
    RepMulOp,
    [
        (ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Ring64Tensor, Replicated64Tensor) -> Replicated64Tensor => Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Ring128Tensor, Replicated128Tensor) -> Replicated128Tensor => Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Ring64Tensor) -> Replicated64Tensor => Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Ring128Tensor) -> Replicated128Tensor => Self::rep_ring_kernel),
    ]
}

struct AbstractReplicatedShape<S> {
    shapes: [S; 3],
}

impl RepMulOp {
    fn rep_rep_kernel<C: Context, RingT, KeyT, ShapeT>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        setup: AbstractReplicatedSetup<KeyT>,
        x: AbstractReplicatedTensor<RingT>,
        y: AbstractReplicatedTensor<RingT>,
    ) -> AbstractReplicatedTensor<RingT>
    where
        RingT: Clone,
        HostPlacement: PlacementAdd<C, RingT, RingT, RingT>,
        HostPlacement: PlacementMul<C, RingT, RingT, RingT>,
        HostPlacement: PlacementShape<C, RingT, ShapeT>,
        ReplicatedPlacement: ZeroShareGen<C, KeyT, RingT, ShapeT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let AbstractReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let v0 = with_context!(player0, ctx, { x00 * y00 + x00 * y10 + x10 * y00 });
        let v1 = with_context!(player1, ctx, { x11 * y11 + x11 * y21 + x21 * y11 });
        let v2 = with_context!(player2, ctx, { x22 * y22 + x22 * y02 + x02 * y22 });

        let s0 = player0.shape(ctx, &v0);
        let s1 = player1.shape(ctx, &v1);
        let s2 = player2.shape(ctx, &v2);
        let zero_shape = AbstractReplicatedShape {
            shapes: [s0, s1, s2],
        };

        let AbstractReplicatedZeroShare {
            alphas: [a0, a1, a2],
        } = rep.gen_zero_share(ctx, &setup, &zero_shape);

        let z0 = with_context!(player0, ctx, { v0 + a0 });
        let z1 = with_context!(player1, ctx, { v1 + a1 });
        let z2 = with_context!(player2, ctx, { v2 + a2 });

        AbstractReplicatedTensor {
            shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
        }
    }

    fn ring_rep_kernel<C: Context, RingT, KeyT>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<KeyT>,
        x: RingT,
        y: AbstractReplicatedTensor<RingT>,
    ) -> AbstractReplicatedTensor<RingT>
    where
        HostPlacement: PlacementMul<C, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, ctx, x * y00);
        let z10 = with_context!(player0, ctx, x * y10);

        let z11 = with_context!(player1, ctx, x * y11);
        let z21 = with_context!(player1, ctx, x * y21);

        let z22 = with_context!(player2, ctx, x * y22);
        let z02 = with_context!(player2, ctx, x * y02);

        AbstractReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn rep_ring_kernel<C: Context, RingT, KeyT>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<KeyT>,
        x: AbstractReplicatedTensor<RingT>,
        y: RingT,
    ) -> AbstractReplicatedTensor<RingT>
    where
        HostPlacement: PlacementMul<C, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = with_context!(player0, ctx, x00 * y);
        let z10 = with_context!(player0, ctx, x10 * y);

        let z11 = with_context!(player1, ctx, x11 * y);
        let z21 = with_context!(player1, ctx, x21 * y);

        let z22 = with_context!(player2, ctx, x22 * y);
        let z02 = with_context!(player2, ctx, x02 * y);

        AbstractReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
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
    {
        let (adt_owner0, adt_owner1) = adt.host_placements();
        let (rep_owner0, rep_owner1, rep_owner2) = x.placement().host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let shares = match () {
            _ if adt_owner0 == rep_owner0 => {
                let y0 = with_context!(rep_owner0, ctx, x00 + x10);
                let y1 = match () {
                    _ if adt_owner1 == rep_owner1 => x21,
                    _ if adt_owner1 == rep_owner2 => x22,
                    _ => x21,
                };
                [y0, y1]
            }
            _ if adt_owner0 == rep_owner1 => {
                let y0 = with_context!(rep_owner1, ctx, x11 + x21);
                let y1 = match () {
                    _ if adt_owner1 == rep_owner2 => x02,
                    _ if adt_owner1 == rep_owner0 => x00,
                    _ => x02,
                };
                [y0, y1]
            }
            _ if adt_owner0 == rep_owner2 => {
                let y0 = with_context!(rep_owner2, ctx, x22 + x02);
                let y1 = match () {
                    _ if adt_owner1 == rep_owner0 => x10,
                    _ if adt_owner1 == rep_owner1 => x11,
                    _ => x10,
                };
                [y0, y1]
            }
            _ if adt_owner1 == rep_owner0 => {
                let y0 = x21;
                let y1 = with_context!(rep_owner0, ctx, x00 + x10);
                [y0, y1]
            }
            _ if adt_owner1 == rep_owner1 => {
                let y0 = x02;
                let y1 = with_context!(rep_owner1, ctx, x11 + x21);
                [y0, y1]
            }
            _ if adt_owner1 == rep_owner2 => {
                let y0 = x10;
                let y1 = with_context!(rep_owner2, ctx, x22 + x02);
                [y0, y1]
            }
            _ => {
                let y0 = with_context!(rep_owner0, ctx, x00 + x10);
                let y1 = x21;
                [y0, y1]
            }
        };
        AbstractAdditiveTensor { shares }
    }
}

struct AbstractReplicatedSeeds<T> {
    seeds: [[T; 2]; 3],
}

trait ReplicatedSeedsGen<C: Context, KeyT, SeedT> {
    fn gen_seeds(
        &self,
        ctx: &C,
        setup: &AbstractReplicatedSetup<KeyT>,
    ) -> AbstractReplicatedSeeds<SeedT>;
}

impl<C: Context> ReplicatedSeedsGen<C, cs!(PrfKey), cs!(Seed)> for ReplicatedPlacement
where
    PrfKey: KnownType<C>,
    Seed: KnownType<C>,
    HostPlacement: PlacementDeriveSeed<C, cs!(PrfKey), cs!(Seed)>,
{
    fn gen_seeds(
        &self,
        ctx: &C,
        setup: &AbstractReplicatedSetup<cs!(PrfKey)>,
    ) -> AbstractReplicatedSeeds<cs!(Seed)> {
        let (owner0, owner1, owner2) = self.host_placements();

        let AbstractReplicatedSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = setup;

        // NOTE for now we pick random sync_keys _at compile time_, which is okay from
        // a security perspective since the seeds depend on both the keys and the sid.
        // however, with sub-computations we could fix these as eg `0`, `1`, and `2`
        // and also make compilation more deterministic
        let sync_key_0 = RawNonce::generate();
        let sync_key_1 = RawNonce::generate();
        let sync_key_2 = RawNonce::generate();

        let s00 = owner0.derive_seed(ctx, sync_key_0.clone(), k00);
        let s10 = owner0.derive_seed(ctx, sync_key_1.clone(), k10);

        let s11 = owner1.derive_seed(ctx, sync_key_1, k11);
        let s21 = owner1.derive_seed(ctx, sync_key_2.clone(), k21);

        let s22 = owner2.derive_seed(ctx, sync_key_2, k22);
        let s02 = owner2.derive_seed(ctx, sync_key_0, k02);

        let seeds = [[s00, s10], [s11, s21], [s22, s02]];
        AbstractReplicatedSeeds { seeds }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct AbstractReplicatedZeroShare<R> {
    alphas: [R; 3],
}

trait ZeroShareGen<C: Context, KeyT, RingT, ShapeT> {
    fn gen_zero_share(
        &self,
        ctx: &C,
        setup: &AbstractReplicatedSetup<KeyT>,
        shape: &AbstractReplicatedShape<ShapeT>,
    ) -> AbstractReplicatedZeroShare<RingT>;
}

impl<C: Context, RingT> ZeroShareGen<C, cs!(PrfKey), RingT, cs!(Shape)> for ReplicatedPlacement
where
    PrfKey: KnownType<C>,
    Seed: KnownType<C>,
    Shape: KnownType<C>,
    HostPlacement: PlacementSampleUniform<C, cs!(Seed), cs!(Shape), RingT>,
    HostPlacement: PlacementSub<C, RingT, RingT, RingT>,
    ReplicatedPlacement: ReplicatedSeedsGen<C, cs!(PrfKey), cs!(Seed)>,
{
    fn gen_zero_share(
        &self,
        ctx: &C,
        setup: &AbstractReplicatedSetup<cs!(PrfKey)>,
        shape: &AbstractReplicatedShape<cs!(Shape)>,
    ) -> AbstractReplicatedZeroShare<RingT> {
        let (owner0, owner1, owner2) = self.host_placements();

        let AbstractReplicatedShape {
            shapes: [shape0, shape1, shape2],
        } = shape;

        let AbstractReplicatedSeeds {
            seeds: [[s00, s10], [s11, s21], [s22, s02]],
        } = &self.gen_seeds(ctx, setup);

        let r00 = owner0.sample_uniform(ctx, s00, shape0);
        let r10 = owner0.sample_uniform(ctx, s10, shape0);
        let alpha0 = with_context!(owner0, ctx, r00 - r10);

        let r11 = owner1.sample_uniform(ctx, s11, shape1);
        let r21 = owner1.sample_uniform(ctx, s21, shape1);
        let alpha1 = with_context!(owner1, ctx, r11 - r21);

        let r22 = owner2.sample_uniform(ctx, s22, shape2);
        let r02 = owner2.sample_uniform(ctx, s02, shape2);
        let alpha2 = with_context!(owner2, ctx, r22 - r02);

        AbstractReplicatedZeroShare {
            alphas: [alpha0, alpha1, alpha2],
        }
    }
}
