use crate::computation::{
    AdditivePlacement, AdtAddOp, AdtFillOp, AdtShlOp, AdtMulOp, AdtRevealOp, AdtSubOp, AdtToRepOp, HostPlacement, Placed, ReplicatedPlacement,
};
use crate::kernels::{
    Context, PlacementAdd, PlacementFill, PlacementMul, PlacementShl, PlacementNeg, PlacementReveal, PlacementSub, PlacementAdtToRep,
};
use crate::prim::RawNonce;
use crate::ring::{Ring128Tensor, Ring64Tensor};
use crate::replicated::{AbstractReplicatedTensor, Replicated64Tensor, Replicated128Tensor};
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

modelled!(PlacementFill::fill, AdditivePlacement, attributes[value: u64] (Shape) -> Additive64Tensor, AdtFillOp);
modelled!(PlacementFill::fill, AdditivePlacement, attributes[value: u64] (Shape) -> Additive128Tensor, AdtFillOp);

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
        value: u64,
        shape: ShapeT,
    ) -> AbstractAdditiveTensor<RingT>
    where
        HostPlacement: PlacementFill<C, ShapeT, RingT>,
    {
        // TODO should really return PublicAdditiveTensor, but we don't have that type yet

        let (player0, player1) = plc.host_placements();

        let shares = [
            player0.fill(ctx, value, &shape),
            player1.fill(ctx, 0, &shape),
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
        add: &AdditivePlacement,
        x: AbstractAdditiveTensor<RingT>,
        y: AbstractAdditiveTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        HostPlacement: PlacementAdd<C, RingT, RingT, RingT>,
    {
        let (player0, player1) = add.host_placements();

        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;
        let AbstractAdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, ctx, x0 + y0);
        let z1 = with_context!(player1, ctx, x1 + y1);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }

    fn adt_ring_kernel<C: Context, RingT>(
        ctx: &C,
        add: &AdditivePlacement,
        x: AbstractAdditiveTensor<RingT>,
        y: RingT,
    ) -> AbstractAdditiveTensor<RingT>
    where
        RingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, RingT, RingT, RingT>,
    {
        let (player0, player1) = add.host_placements();
        let y_plc = y.placement();

        let AbstractAdditiveTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, ctx, x0 + y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, ctx, x1 + y)],
            _ => [with_context!(player0, ctx, x0 + y), x1],
        };
        AbstractAdditiveTensor { shares }
    }

    fn ring_adt_kernel<C: Context, RingT>(
        ctx: &C,
        add: &AdditivePlacement,
        x: RingT,
        y: AbstractAdditiveTensor<RingT>,
    ) -> AbstractAdditiveTensor<RingT>
    where
        RingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, RingT, RingT, RingT>,
    {
        let (player0, player1) = add.host_placements();
        let x_plc = x.placement();

        let AbstractAdditiveTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, ctx, y0 + x), y1],
            _ if x_plc == player1 => [y0, with_context!(player1, ctx, x + y1)],
            _ => [with_context!(player0, ctx, x + y0), y1],
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
    {
        let (player0, player1) = adt.host_placements();
        let y_plc = y.placement();

        let AbstractAdditiveTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, ctx, x0 - y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, ctx, x1 - y)],
            _ => [with_context!(player0, ctx, x0 - y), x1],
        };
        AbstractAdditiveTensor { shares }
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
    {
        let (player0, player1) = adt.host_placements();
        let x_plc = x.placement();

        let AbstractAdditiveTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, ctx, x - y0), player1.neg(ctx, &y1)],
            _ if x_plc == player1 => [player0.neg(ctx, &y0), with_context!(player1, ctx, x - y1)],
            _ => [with_context!(player0, ctx, x - y0), player1.neg(ctx, &y1)],
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
    fn ring_adt_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: R,
        y: AbstractAdditiveTensor<R>,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();

        let AbstractAdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, ctx, x * y0);
        let z1 = with_context!(player1, ctx, x * y1);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }

    fn adt_ring_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: AbstractAdditiveTensor<R>,
        y: R,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();

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

modelled!(PlacementAdtToRep::add_to_rep, ReplicatedPlacement, (Additive64Tensor) -> Replicated64Tensor, AdtToRepOp);
modelled!(PlacementAdtToRep::add_to_rep, ReplicatedPlacement, (Additive128Tensor) -> Replicated128Tensor, AdtToRepOp);

hybrid_kernel! {
    AdtToRepOp,
    [
        (ReplicatedPlacement, (Additive64Tensor) -> Replicated64Tensor => Self::kernel),
        (ReplicatedPlacement, (Additive128Tensor) -> Replicated128Tensor => Self::kernel),
    ]
}

use crate::kernels::{PlacementShape, PlacementKeyGen, PlacementSampleUniform};

impl AdtToRepOp {
    fn kernel<C: Context, SeedT, ShapeT, KeyT, RingT>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: AbstractAdditiveTensor<RingT>,
    ) -> AbstractReplicatedTensor<RingT>
    where
        RingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementShape<C, RingT, ShapeT>,
        HostPlacement: PlacementKeyGen<C, KeyT>,
        HostPlacement: PlacementSampleUniform<C, SeedT, ShapeT, RingT>,
        HostPlacement: PlacementDeriveSeed<C, KeyT, SeedT>,
        AdditivePlacement: PlacementSub<C, AbstractAdditiveTensor<RingT>, AbstractAdditiveTensor<RingT>, AbstractAdditiveTensor<RingT>>,
        HostPlacement: PlacementReveal<C, AbstractAdditiveTensor<RingT>, RingT>,
    {
        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;

        let adt = x.placement();
        let (adt_player0, adt_player1) = adt.host_placements();
        let (rep_player0, rep_player1, rep_player2) = rep.host_placements();

        match () {
            _ if rep_player0 != adt_player0 && rep_player0 != adt_player1 => {
                let provider = rep_player0;
                
                


            }
        }


        // assume that Additive Host Placements are included Replicated Host Placements
        // the player that is not on the additive is the provider
        let provider = match () {
            _ if rep_player0 != adt_player0 && rep_player0 != adt_player1 => rep_player0,
            _ if rep_player1 != adt_player0 && rep_player1 != adt_player1 => rep_player1,
            _ if rep_player2 != adt_player0 && rep_player2 != adt_player1 => rep_player2,
            _ => unimplemented!()
        };

        let shape = adt_player0.shape(ctx, &x0);

        let sync_key0 = RawNonce::generate();
        let sync_key1 = RawNonce::generate();



        let seed0 = provider.derive_seed()
        
        let y13 = provider.sample_uniform(ctx);
        let y33 = provider.sample_uniform(ctx);

        let y1 = adt_player0.sample_uniform(ctx);
        let y2 = adt_player1.sample_uniform(ctx);
        let y = AbstractAdditiveTensor {
            shares: [y1.clone(), y2.clone()],
        };
        let c = adt_player0.reveal(ctx, &adt.sub(ctx, &x, &y));

        AbstractReplicatedTensor {
            shares: [[y1, c.clone()], [c, y2], [y33, y13]],
        }
    }
}
