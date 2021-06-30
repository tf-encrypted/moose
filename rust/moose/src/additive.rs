use crate::computation::{
    AdditivePlacement, AdtAddOp, AdtMulOp, AdtRevealOp, HostPlacement, Placed,
};
use crate::kernels::{Context, PlacementAdd, PlacementMul};
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

impl AdtRevealOp {
    fn kernel<C: Context, R>(ctx: &C, plc: &HostPlacement, xe: AbstractAdditiveTensor<R>) -> R
    where
        R: Clone,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let AbstractAdditiveTensor { shares: [x0, x1] } = &xe;
        with_context!(plc, ctx, x1 + x0)
    }
}

impl AdtAddOp {
    fn add_add_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: AbstractAdditiveTensor<R>,
        y: AbstractAdditiveTensor<R>,
    ) -> AbstractAdditiveTensor<R>
    where
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();

        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;

        let AbstractAdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, ctx, x0 + y0);
        let z1 = with_context!(player1, ctx, x1 + y1);

        AbstractAdditiveTensor { shares: [z0, z1] }
    }

    fn add_ring_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: AbstractAdditiveTensor<R>,
        y: R,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();
        let AbstractAdditiveTensor { shares: [x0, x1] } = x;

        let y_plc = y.placement();

        let shares = match y_plc {
            _ if y_plc == player0 => [with_context!(player0, ctx, x0 + y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, ctx, x1 + y)],
            _ => [with_context!(player0, ctx, x0 + y), x1],
        };
        AbstractAdditiveTensor { shares }
    }

    fn ring_add_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: R,
        y: AbstractAdditiveTensor<R>,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();
        let AbstractAdditiveTensor { shares: [y0, y1] } = y;

        let x_plc = x.placement();

        let shares = match x_plc {
            _ if x_plc == player0 => [with_context!(player0, ctx, y0 + x), y1],
            _ if x_plc == player1 => [y0, with_context!(player1, ctx, x + y1)],
            _ => [with_context!(player0, ctx, x + y0), y1],
        };
        AbstractAdditiveTensor { shares }
    }
}

impl AdtMulOp {
    fn ring_add_kernel<C: Context, R>(
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

    fn add_ring_kernel<C: Context, R>(
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
