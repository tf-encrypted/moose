use crate::additive::AbstractAdditiveTensor;
use crate::bit::BitTensor;
use crate::computation::{
    AdditivePlacement, HostPlacement, Placed, RepAddOp, RepMulOp, RepRevealOp, RepSetupOp,
    RepShareOp, RepToAdtOp, ReplicatedPlacement,
};
use crate::kernels::{
    Context, PlacementAdd, PlacementKeyGen, PlacementMul, PlacementSample, PlacementSub,
};
use crate::prim::PrfKey;
use crate::ring::{Ring128Tensor, Ring64Tensor};
use macros::with_context;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractReplicatedTensor<R> {
    pub shares: [[R; 2]; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractReplicatedSetup<K> {
    pub keys: [[K; 2]; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct AbstractReplicatedZeroShare<R> {
    alphas: [R; 3],
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

impl RepShareOp {
    fn kernel<C: Context, R: Clone>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: R,
    ) -> AbstractReplicatedTensor<R>
    where
        R: Into<C::Value> + TryFrom<C::Value> + 'static,
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSample<C, R>,
        HostPlacement: PlacementAdd<C, R, R, R>,
        HostPlacement: PlacementSub<C, R, R, R>,
    {
        let owner = x.placement();

        let x0 = owner.sample(ctx);
        let x1 = owner.sample(ctx);
        let x2 = with_context!(owner, ctx, x - (x0 + x1));

        AbstractReplicatedTensor {
            shares: [[x0.clone(), x1.clone()], [x1, x2.clone()], [x2, x0]],
        }
    }
}

impl RepRevealOp {
    fn kernel<C: Context, R: Clone>(
        ctx: &C,
        plc: &HostPlacement,
        xe: AbstractReplicatedTensor<R>,
    ) -> R
    where
        R: Clone + 'static,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &xe;

        with_context!(plc, ctx, x00 + x10 + x21)
    }
}

impl RepSetupOp {
    fn kernel<C: Context, K: Clone>(
        ctx: &C,
        rep: &ReplicatedPlacement,
    ) -> AbstractReplicatedSetup<K>
    where
        HostPlacement: PlacementKeyGen<C, K>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let k0 = player0.keygen(ctx);
        let k1 = player1.keygen(ctx);
        let k2 = player2.keygen(ctx);

        AbstractReplicatedSetup {
            keys: [[k0.clone(), k1.clone()], [k1, k2.clone()], [k2, k0]],
        }
    }
}

impl RepAddOp {
    fn rep_rep_kernel<C: Context, R>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedTensor<R>,
        y: AbstractReplicatedTensor<R>,
    ) -> AbstractReplicatedTensor<R>
    where
        R: Clone,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let AbstractReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, ctx, x00 + y00);
        let z10 = with_context!(player0, ctx, x10 + y10);

        let z11 = with_context!(player1, ctx, x11 + y11);
        let z21 = with_context!(player1, ctx, x21 + y21);

        let z22 = with_context!(player2, ctx, x22 + y22);
        let z02 = with_context!(player2, ctx, x02 + y02);

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
        R: Clone,
        // R: KnownType<C>, // TODO needed?
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let x_plc = x.placement();

        let AbstractReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = y;

        let shares = match x_plc {
            _ if x_plc == player0 => {
                // add x to y0
                [
                    [with_context!(player0, ctx, x + y00), y10],
                    [y11, y21],
                    [y22, with_context!(player2, ctx, x + y02)],
                ]
            }
            _ if x_plc == player1 => {
                // add x to y1
                [
                    [y00, with_context!(player0, ctx, x + y10)],
                    [with_context!(player1, ctx, x + y11), y21],
                    [y22, y02],
                ]
            }
            _ if x_plc == player2 => {
                // add x to y2
                [
                    [y00, y10],
                    [y11, with_context!(player1, ctx, x + y21)],
                    [with_context!(player2, ctx, x + y22), y02],
                ]
            }
            _ => {
                // add x to y0; we could randomize this
                [
                    [with_context!(player0, ctx, x + y00), y10],
                    [y11, y21],
                    [y22, with_context!(player2, ctx, x + y02)],
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
        R: Clone,
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let y_plc = y.placement();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let shares = match y_plc {
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

impl RepMulOp {
    fn rep_rep_kernel<C: Context, R, K>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        setup: AbstractReplicatedSetup<K>,
        x: AbstractReplicatedTensor<R>,
        y: AbstractReplicatedTensor<R>,
    ) -> AbstractReplicatedTensor<R>
    where
        R: Clone + Into<C::Value> + TryFrom<C::Value> + 'static,
        HostPlacement: PlacementSample<C, R>,
        HostPlacement: PlacementAdd<C, R, R, R>,
        HostPlacement: PlacementMul<C, R, R, R>,
        ReplicatedPlacement: PlacementZeroShare<C, K, R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let AbstractReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let AbstractReplicatedZeroShare {
            alphas: [a0, a1, a2],
        } = rep.zero_share(ctx, &setup);

        let z0 = with_context!(player0, ctx, { x00 * y00 + x00 * y10 + x10 * y00 + a0 });
        let z1 = with_context!(player1, ctx, { x11 * y11 + x11 * y21 + x21 * y11 + a1 });
        let z2 = with_context!(player2, ctx, { x22 * y22 + x22 * y02 + x02 * y22 + a2 });

        AbstractReplicatedTensor {
            shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
        }
    }

    fn ring_rep_kernel<C: Context, R, K>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<K>,
        x: R,
        y: AbstractReplicatedTensor<R>,
    ) -> AbstractReplicatedTensor<R>
    where
        HostPlacement: PlacementMul<C, R, R, R>,
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

    fn rep_ring_kernel<C: Context, R, K>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<K>,
        x: AbstractReplicatedTensor<R>,
        y: R,
    ) -> AbstractReplicatedTensor<R>
    where
        HostPlacement: PlacementMul<C, R, R, R>,
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

trait PlacementZeroShare<C: Context, K, R> {
    fn zero_share(
        &self,
        ctx: &C,
        setup: &AbstractReplicatedSetup<K>,
    ) -> AbstractReplicatedZeroShare<R>;
}

// NOTE this is an un-modelled operation (as opposed to the modelled! operations that have
// a representation in computations); should we have a macro for this as well?
impl<C: Context, K, R> PlacementZeroShare<C, K, R> for ReplicatedPlacement
where
    R: Clone + 'static,
    HostPlacement: PlacementSample<C, R>,
    HostPlacement: PlacementSub<C, R, R, R>,
{
    fn zero_share(
        &self,
        ctx: &C,
        s: &AbstractReplicatedSetup<K>,
    ) -> AbstractReplicatedZeroShare<R> {
        let (player0, player1, player2) = self.host_placements();

        let AbstractReplicatedSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = s;

        // TODO use keys when sampling!

        let r00 = player0.sample(ctx);
        let r10 = player0.sample(ctx);
        let alpha0 = with_context!(player0, ctx, r00 - r10);

        let r11 = player1.sample(ctx);
        let r21 = player1.sample(ctx);
        let alpha1 = with_context!(player1, ctx, r11 - r21);

        let r22 = player2.sample(ctx);
        let r02 = player2.sample(ctx);
        let alpha2 = with_context!(player2, ctx, r22 - r02);

        AbstractReplicatedZeroShare {
            alphas: [alpha0, alpha1, alpha2],
        }
    }
}

impl RepToAdtOp {
    fn rep_to_add_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: AbstractReplicatedTensor<R>,
    ) -> AbstractAdditiveTensor<R>
    where
        R: Clone,
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (player_a, player_b) = add.host_placements();
        let (player0, player1, player2) = x.placement().host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let shares = match () {
            _ if player_a == player0 && player_b == player1 => {
                [with_context!(player0, ctx, x00 + x10), x21]
            }
            _ if player_a == player0 && player_b == player2 => {
                [with_context!(player0, ctx, x00 + x10), x22]
            }
            _ if player_a == player1 && player_b == player2 => {
                [with_context!(player1, ctx, x11 + x21), x02]
            }
            _ if player_a == player1 && player_b == player0 => {
                [x21, with_context!(player0, ctx, x00 + x10)]
            }
            _ if player_a == player2 && player_b == player0 => {
                [x22, with_context!(player0, ctx, x00 + x10)]
            }
            _ => [with_context!(player_a, ctx, x00 + x10), x21],
        };
        AbstractAdditiveTensor { shares }
    }
}
