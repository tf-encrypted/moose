use crate::computation::{
    AdditivePlacement, AdtAddOp, AdtMulOp, AdtRevealOp, AdtSubOp, HostPlacement, Placed,
};
use crate::kernels::{
    Context, PlacementAdd, PlacementMul, PlacementNeg, PlacementReveal, PlacementSub,
};
use crate::ring::{Ring128Tensor, Ring64Tensor};
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
