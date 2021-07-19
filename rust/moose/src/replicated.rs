use crate::additive::{AbstractAdditiveTensor, Additive128Tensor, Additive64Tensor};
use crate::bit::BitTensor;
use crate::computation::{
    AdditivePlacement, AdtToRepOp, HostPlacement, KnownType, Placed, RepAddOp, RepDotOp, RepMeanOp,
    RepMulOp, RepRevealOp, RepSetupOp, RepShareOp, RepSubOp, RepSumOp, RepTruncPrOp,
    ReplicatedPlacement,
};
use crate::error::{Error, Result};
use crate::kernels::{
    PlacementAdd, PlacementAdtToRep, PlacementDeriveSeed, PlacementDot, PlacementDotSetup,
    PlacementKeyGen, PlacementMean, PlacementMul, PlacementMulSetup, PlacementPlace,
    PlacementRepToAdt, PlacementReveal, PlacementRingMean, PlacementSampleUniform,
    PlacementSetupGen, PlacementShape, PlacementShareSetup, PlacementSub, PlacementSum,
    PlacementTruncPr, PlacementTruncPrProvider, PlacementZeros, Session,
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

pub type Replicated64Tensor = AbstractReplicatedTensor<Ring64Tensor>;

pub type Replicated128Tensor = AbstractReplicatedTensor<Ring128Tensor>;

pub type ReplicatedBitTensor = AbstractReplicatedTensor<BitTensor>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbstractReplicatedSetup<K> {
    pub keys: [[K; 2]; 3],
}

pub type ReplicatedSetup = AbstractReplicatedSetup<PrfKey>;

struct AbstractReplicatedShape<S> {
    shapes: [S; 3],
}

impl<R> Placed for AbstractReplicatedTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let AbstractReplicatedTensor {
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

impl<S: Session, R> PlacementPlace<S, AbstractReplicatedTensor<R>> for ReplicatedPlacement
where
    AbstractReplicatedTensor<R>: Placed<Placement = ReplicatedPlacement>,
    HostPlacement: PlacementPlace<S, R>,
{
    fn place(&self, sess: &S, x: AbstractReplicatedTensor<R>) -> AbstractReplicatedTensor<R> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                let AbstractReplicatedTensor {
                    shares: [[x00, x10], [x11, x21], [x22, x02]],
                } = x;

                let (player0, player1, player2) = self.host_placements();
                AbstractReplicatedTensor {
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

hybrid_kernel! {
    RepSetupOp,
    [
        (ReplicatedPlacement, () -> ReplicatedSetup => Self::kernel),
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

modelled!(PlacementShareSetup::share, ReplicatedPlacement, (ReplicatedSetup, Ring64Tensor) -> Replicated64Tensor, RepShareOp);
modelled!(PlacementShareSetup::share, ReplicatedPlacement, (ReplicatedSetup, Ring128Tensor) -> Replicated128Tensor, RepShareOp);
modelled!(PlacementShareSetup::share, ReplicatedPlacement, (ReplicatedSetup, BitTensor) -> ReplicatedBitTensor, RepShareOp);

hybrid_kernel! {
    RepShareOp,
    [
        (ReplicatedPlacement, (ReplicatedSetup, Ring64Tensor) -> Replicated64Tensor => Self::kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Ring128Tensor) -> Replicated128Tensor => Self::kernel),
        (ReplicatedPlacement, (ReplicatedSetup, BitTensor) -> ReplicatedBitTensor => Self::kernel),
    ]
}

impl RepShareOp {
    fn kernel<S: Session, ShapeT, SeedT, KeyT, RingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        setup: AbstractReplicatedSetup<KeyT>,
        x: RingT,
    ) -> AbstractReplicatedTensor<RingT>
    where
        RingT: Clone + Placed<Placement = HostPlacement>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        HostPlacement: PlacementSampleUniform<S, ShapeT, SeedT, RingT>,
        HostPlacement: PlacementZeros<S, ShapeT, RingT>,
        HostPlacement: PlacementDeriveSeed<S, KeyT, SeedT>,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementSub<S, RingT, RingT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, AbstractReplicatedTensor<RingT>>,
    {
        let x_player = x.placement().unwrap();

        let AbstractReplicatedSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = &setup;

        let (player0, player1, player2) = plc.host_placements();

        let shares = match () {
            _ if x_player == player0 => {
                let sync_key = RawNonce::generate();
                let shape = x_player.shape(sess, &x);

                let seed0 = player0.derive_seed(sess, sync_key.clone(), k00);
                let x00 = x_player.sample_uniform(sess, &shape, &seed0);
                let x10 = with_context!(x_player, sess, x - x00);

                let seed2 = player2.derive_seed(sess, sync_key, k02);
                let x22 = player2.zeros(sess, &shape);
                let x02 = player2.sample_uniform(sess, &shape, &seed2);

                let x11 = x10.clone();
                let x21 = player1.zeros(sess, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ if x_player == player1 => {
                let sync_key = RawNonce::generate();
                let shape = x_player.shape(sess, &x);

                let seed1 = player1.derive_seed(sess, sync_key.clone(), k11);
                let x11 = x_player.sample_uniform(sess, &shape, &seed1);
                let x21 = with_context!(x_player, sess, x - x11);

                let seed0 = player0.derive_seed(sess, sync_key, k10);
                let x00 = player0.zeros(sess, &shape);
                let x10 = player0.sample_uniform(sess, &shape, &seed0);

                let x22 = x21.clone();
                let x02 = player2.zeros(sess, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ if x_player == player2 => {
                let sync_key = RawNonce::generate();
                let shape = x_player.shape(sess, &x);

                let seed2 = player2.derive_seed(sess, sync_key.clone(), k22);
                let x22 = player2.sample_uniform(sess, &shape, &seed2);
                let x02 = with_context!(x_player, sess, x - x22);

                let seed1 = player1.derive_seed(sess, sync_key, k21);
                let x11 = player1.zeros(sess, &shape);
                let x21 = player1.sample_uniform(sess, &shape, &seed1);

                let x00 = x02.clone();
                let x10 = player0.zeros(sess, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ => {
                // in this case, where x_owner is _not_ among the replicated players,
                // we cannot use the zeros optimization trick but we can still make sure
                // that seeds are used as much as possible instead of dense random tensors;
                // however, we must make sure keys are not revealed to x_owner and only seeds

                let sync_key0 = RawNonce::generate();
                let sync_key1 = RawNonce::generate();
                let shape = x_player.shape(sess, &x);

                let seed00 = player0.derive_seed(sess, sync_key0.clone(), k00);
                let seed02 = player2.derive_seed(sess, sync_key0, k02);

                let seed11 = player1.derive_seed(sess, sync_key1.clone(), k11);
                let seed10 = player0.derive_seed(sess, sync_key1, k10);

                let x0 = x_player.sample_uniform(sess, &shape, &seed00);
                let x1 = x_player.sample_uniform(sess, &shape, &seed11);
                let x2 = with_context!(x_player, sess, x - x0 - x1);

                let x00 = player0.sample_uniform(sess, &shape, &seed00);
                let x10 = player0.sample_uniform(sess, &shape, &seed10);

                let x11 = player1.sample_uniform(sess, &shape, &seed11);
                let x21 = x2.clone();

                let x22 = x2;
                let x02 = player2.sample_uniform(sess, &shape, &seed02);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
        };

        plc.place(sess, AbstractReplicatedTensor { shares })
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
    fn kernel<S: Session, R: Clone>(
        sess: &S,
        receiver: &HostPlacement,
        xe: AbstractReplicatedTensor<R>,
    ) -> R
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, R, R, R>,
    {
        let AbstractReplicatedTensor {
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
    fn rep_rep_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedTensor<R>,
        y: AbstractReplicatedTensor<R>,
    ) -> AbstractReplicatedTensor<R>
    where
        HostPlacement: PlacementAdd<S, R, R, R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let AbstractReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, sess, x00 + y00);
        let z10 = with_context!(player0, sess, x10 + y10);

        let z11 = with_context!(player1, sess, x11 + y11);
        let z21 = with_context!(player1, sess, x21 + y21);

        let z22 = with_context!(player2, sess, x22 + y22);
        let z02 = with_context!(player2, sess, x02 + y02);

        AbstractReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn ring_rep_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: R,
        y: AbstractReplicatedTensor<R>,
    ) -> AbstractReplicatedTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, R, R, R>,
        ReplicatedPlacement: PlacementPlace<S, AbstractReplicatedTensor<R>>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let x_plc = x.placement().unwrap();

        let AbstractReplicatedTensor {
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

        rep.place(sess, AbstractReplicatedTensor { shares })
    }

    fn rep_ring_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedTensor<R>,
        y: R,
    ) -> AbstractReplicatedTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, R, R, R>,
        ReplicatedPlacement: PlacementPlace<S, AbstractReplicatedTensor<R>>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let y_plc = y.placement().unwrap();

        let AbstractReplicatedTensor {
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

        rep.place(sess, AbstractReplicatedTensor { shares })
    }
}

modelled!(PlacementSub::sub, ReplicatedPlacement, (Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (Ring64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (Ring128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (Replicated64Tensor, Ring64Tensor) -> Replicated64Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (Replicated128Tensor, Ring128Tensor) -> Replicated128Tensor, RepSubOp);
modelled!(PlacementSub::sub, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor, RepSubOp);

hybrid_kernel! {
    RepSubOp,
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

impl RepSubOp {
    fn rep_rep_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedTensor<R>,
        y: AbstractReplicatedTensor<R>,
    ) -> AbstractReplicatedTensor<R>
    where
        HostPlacement: PlacementSub<S, R, R, R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let AbstractReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, sess, x00 - y00);
        let z10 = with_context!(player0, sess, x10 - y10);

        let z11 = with_context!(player1, sess, x11 - y11);
        let z21 = with_context!(player1, sess, x21 - y21);

        let z22 = with_context!(player2, sess, x22 - y22);
        let z02 = with_context!(player2, sess, x02 - y02);

        AbstractReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn ring_rep_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: R,
        y: AbstractReplicatedTensor<R>,
    ) -> AbstractReplicatedTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<S, R, R, R>,
        ReplicatedPlacement: PlacementPlace<S, AbstractReplicatedTensor<R>>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let x_plc = x.placement().unwrap();

        let AbstractReplicatedTensor {
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

        rep.place(sess, AbstractReplicatedTensor { shares })
    }

    fn rep_ring_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractReplicatedTensor<R>,
        y: R,
    ) -> AbstractReplicatedTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<S, R, R, R>,
        ReplicatedPlacement: PlacementPlace<S, AbstractReplicatedTensor<R>>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let y_plc = y.placement().unwrap();

        let AbstractReplicatedTensor {
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

        rep.place(sess, AbstractReplicatedTensor { shares })
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

impl RepMulOp {
    fn rep_rep_kernel<S: Session, RingT, KeyT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        setup: AbstractReplicatedSetup<KeyT>,
        x: AbstractReplicatedTensor<RingT>,
        y: AbstractReplicatedTensor<RingT>,
    ) -> AbstractReplicatedTensor<RingT>
    where
        RingT: Clone,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        ReplicatedPlacement: ZeroShareGen<S, KeyT, RingT, ShapeT>,
        ReplicatedPlacement: PlacementPlace<S, AbstractReplicatedTensor<RingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let AbstractReplicatedTensor {
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
            AbstractReplicatedTensor {
                shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
            },
        )
    }

    fn ring_rep_kernel<S: Session, RingT, KeyT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<KeyT>,
        x: RingT,
        y: AbstractReplicatedTensor<RingT>,
    ) -> AbstractReplicatedTensor<RingT>
    where
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, sess, x * y00);
        let z10 = with_context!(player0, sess, x * y10);

        let z11 = with_context!(player1, sess, x * y11);
        let z21 = with_context!(player1, sess, x * y21);

        let z22 = with_context!(player2, sess, x * y22);
        let z02 = with_context!(player2, sess, x * y02);

        AbstractReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn rep_ring_kernel<S: Session, RingT, KeyT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<KeyT>,
        x: AbstractReplicatedTensor<RingT>,
        y: RingT,
    ) -> AbstractReplicatedTensor<RingT>
    where
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = with_context!(player0, sess, x00 * y);
        let z10 = with_context!(player0, sess, x10 * y);

        let z11 = with_context!(player1, sess, x11 * y);
        let z21 = with_context!(player1, sess, x21 * y);

        let z22 = with_context!(player2, sess, x22 * y);
        let z02 = with_context!(player2, sess, x02 * y);

        AbstractReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
}

modelled!(PlacementDotSetup::dot, ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepDotOp);
modelled!(PlacementDotSetup::dot, ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepDotOp);
modelled!(PlacementDotSetup::dot, ReplicatedPlacement, (ReplicatedSetup, Ring64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepDotOp);
modelled!(PlacementDotSetup::dot, ReplicatedPlacement, (ReplicatedSetup, Ring128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepDotOp);
modelled!(PlacementDotSetup::dot, ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Ring64Tensor) -> Replicated64Tensor, RepDotOp);
modelled!(PlacementDotSetup::dot, ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Ring128Tensor) -> Replicated128Tensor, RepDotOp);
// modelled!(PlacementDotSetup::dot, ReplicatedPlacement, (ReplicatedSetup, ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor, RepDotOp);

hybrid_kernel! {
    RepDotOp,
    [
        (ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor => Self::rep_rep_kernel),
        // (ReplicatedPlacement, (ReplicatedSetup, ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Ring64Tensor, Replicated64Tensor) -> Replicated64Tensor => Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Ring128Tensor, Replicated128Tensor) -> Replicated128Tensor => Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Ring64Tensor) -> Replicated64Tensor => Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Ring128Tensor) -> Replicated128Tensor => Self::rep_ring_kernel),
    ]
}

impl RepDotOp {
    fn rep_rep_kernel<S: Session, RingT, KeyT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        setup: AbstractReplicatedSetup<KeyT>,
        x: AbstractReplicatedTensor<RingT>,
        y: AbstractReplicatedTensor<RingT>,
    ) -> AbstractReplicatedTensor<RingT>
    where
        RingT: Clone,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementDot<S, RingT, RingT, RingT>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        ReplicatedPlacement: ZeroShareGen<S, KeyT, RingT, ShapeT>,
        ReplicatedPlacement: PlacementPlace<S, AbstractReplicatedTensor<RingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let AbstractReplicatedTensor {
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
            AbstractReplicatedTensor {
                shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
            },
        )
    }

    fn ring_rep_kernel<S: Session, RingT, KeyT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<KeyT>,
        x: RingT,
        y: AbstractReplicatedTensor<RingT>,
    ) -> AbstractReplicatedTensor<RingT>
    where
        HostPlacement: PlacementDot<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, sess, dot(&x, y00));
        let z10 = with_context!(player0, sess, dot(&x, y10));

        let z11 = with_context!(player1, sess, dot(&x, y11));
        let z21 = with_context!(player1, sess, dot(&x, y21));

        let z22 = with_context!(player2, sess, dot(&x, y22));
        let z02 = with_context!(player2, sess, dot(&x, y02));

        AbstractReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn rep_ring_kernel<S: Session, RingT, KeyT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<KeyT>,
        x: AbstractReplicatedTensor<RingT>,
        y: RingT,
    ) -> AbstractReplicatedTensor<RingT>
    where
        HostPlacement: PlacementDot<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = with_context!(player0, sess, dot(x00, &y));
        let z10 = with_context!(player0, sess, dot(x10, &y));

        let z11 = with_context!(player1, sess, dot(x11, &y));
        let z21 = with_context!(player1, sess, dot(x21, &y));

        let z22 = with_context!(player2, sess, dot(x22, &y));
        let z02 = with_context!(player2, sess, dot(x02, &y));

        AbstractReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
}

modelled!(PlacementMean::mean, ReplicatedPlacement, attributes[axis: Option<u32>, precision: u64] (Replicated64Tensor) -> Replicated64Tensor, RepMeanOp);
modelled!(PlacementMean::mean, ReplicatedPlacement, attributes[axis: Option<u32>, precision: u64] (Replicated128Tensor) -> Replicated128Tensor, RepMeanOp);

hybrid_kernel! {
    RepMeanOp,
    [
        (ReplicatedPlacement, (Replicated64Tensor) -> Replicated64Tensor => attributes[axis, precision] Self::kernel),
        (ReplicatedPlacement, (Replicated128Tensor) -> Replicated128Tensor => attributes[axis, precision] Self::kernel),
    ]
}

impl RepMeanOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: Option<u32>,
        precision: u64,
        x: AbstractReplicatedTensor<RingT>,
    ) -> AbstractReplicatedTensor<RingT>
    where
        HostPlacement: PlacementRingMean<S, RingT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, AbstractReplicatedTensor<RingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let precision: u32 = precision.try_into().unwrap();
        let z00 = player0.ring_mean(sess, axis, 2, precision, x00);
        let z10 = player0.ring_mean(sess, axis, 2, precision, x10);
        let z11 = player1.ring_mean(sess, axis, 2, precision, x11);
        let z21 = player1.ring_mean(sess, axis, 2, precision, x21);
        let z22 = player2.ring_mean(sess, axis, 2, precision, x22);
        let z02 = player2.ring_mean(sess, axis, 2, precision, x02);

        rep.place(
            sess,
            AbstractReplicatedTensor {
                shares: [[z00, z10], [z11, z21], [z22, z02]],
            },
        )
    }
}

modelled!(PlacementSum::sum, ReplicatedPlacement, attributes[axis: Option<u32>] (Replicated64Tensor) -> Replicated64Tensor, RepSumOp);
modelled!(PlacementSum::sum, ReplicatedPlacement, attributes[axis: Option<u32>] (Replicated128Tensor) -> Replicated128Tensor, RepSumOp);

hybrid_kernel! {
    RepSumOp,
    [
        (ReplicatedPlacement, (Replicated64Tensor) -> Replicated64Tensor => attributes[axis] Self::kernel),
        (ReplicatedPlacement, (Replicated128Tensor) -> Replicated128Tensor => attributes[axis] Self::kernel),
    ]
}

impl RepSumOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: Option<u32>,
        x: AbstractReplicatedTensor<RingT>,
    ) -> AbstractReplicatedTensor<RingT>
    where
        HostPlacement: PlacementSum<S, RingT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, AbstractReplicatedTensor<RingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let AbstractReplicatedTensor {
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
            AbstractReplicatedTensor {
                shares: [[z00, z10], [z11, z21], [z22, z02]],
            },
        )
    }
}

use std::convert::TryInto;

modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[amount: usize] (Replicated64Tensor) -> Replicated64Tensor, RepTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[amount: usize] (Replicated128Tensor) -> Replicated128Tensor, RepTruncPrOp);

hybrid_kernel! {
    RepTruncPrOp,
    [
        (ReplicatedPlacement,  (Replicated64Tensor) -> Replicated64Tensor => attributes[amount] Self::kernel),
        (ReplicatedPlacement,  (Replicated128Tensor) -> Replicated128Tensor => attributes[amount] Self::kernel),
    ]
}

impl RepTruncPrOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        amount: usize,
        xe: AbstractReplicatedTensor<RingT>,
    ) -> st!(AbstractReplicatedTensor<RingT>, S)
    where
        RingT: Clone,
        AbstractReplicatedTensor<RingT>: Into<st!(AbstractReplicatedTensor<RingT>)>,
        st!(AbstractAdditiveTensor<RingT>): TryInto<AbstractAdditiveTensor<RingT>>,
        AbstractAdditiveTensor<RingT>: Into<st!(AbstractAdditiveTensor<RingT>)>,
        st!(AbstractAdditiveTensor<RingT>): TryInto<AbstractAdditiveTensor<RingT>>,

        AbstractAdditiveTensor<RingT>: CanonicalType,
        <AbstractAdditiveTensor<RingT> as CanonicalType>::Type: KnownType<S>,
        AbstractReplicatedTensor<RingT>: CanonicalType,
        <AbstractReplicatedTensor<RingT> as CanonicalType>::Type: KnownType<S>,

        AdditivePlacement: PlacementRepToAdt<
            S,
            st!(AbstractReplicatedTensor<RingT>),
            st!(AbstractAdditiveTensor<RingT>),
        >,

        AdditivePlacement: PlacementTruncPrProvider<
            S,
            AbstractAdditiveTensor<RingT>,
            AbstractAdditiveTensor<RingT>,
        >,

        ReplicatedPlacement: PlacementAdtToRep<
            S,
            st!(AbstractAdditiveTensor<RingT>),
            st!(AbstractReplicatedTensor<RingT>),
        >,
    {
        let (player0, player1, player2) = rep.host_placements();

        let adt = AdditivePlacement {
            owners: [player0.owner, player1.owner],
        };
        let provider = player2;

        let x_adt = adt.rep_to_adt(sess, &xe.into()).try_into().ok().unwrap();
        let y_adt = adt.trunc_pr(sess, amount, &provider, &x_adt);
        rep.adt_to_rep(sess, &y_adt.into())
    }
}

pub trait CanonicalType {
    type Type;
}

use crate::ring::AbstractRingTensor;
use crate::symbolic::Symbolic;

impl<T> CanonicalType for AbstractRingTensor<T> {
    type Type = AbstractRingTensor<T>;
}

impl<T> CanonicalType for Symbolic<AbstractRingTensor<T>> {
    type Type = AbstractRingTensor<T>;
}

impl<RingT: CanonicalType> CanonicalType for AbstractAdditiveTensor<RingT> {
    type Type = AbstractAdditiveTensor<<RingT as CanonicalType>::Type>;
}

impl<RingT: CanonicalType + Placed<Placement = HostPlacement>> CanonicalType
    for Symbolic<AbstractAdditiveTensor<RingT>>
{
    type Type = AbstractAdditiveTensor<<RingT as CanonicalType>::Type>;
}

impl<RingT: CanonicalType> CanonicalType for AbstractReplicatedTensor<RingT> {
    type Type = AbstractReplicatedTensor<<RingT as CanonicalType>::Type>;
}

impl<RingT: CanonicalType + Placed<Placement = HostPlacement>> CanonicalType
    for Symbolic<AbstractReplicatedTensor<RingT>>
{
    type Type = AbstractReplicatedTensor<<RingT as CanonicalType>::Type>;
}

modelled!(PlacementAdtToRep::adt_to_rep, ReplicatedPlacement, (Additive64Tensor) -> Replicated64Tensor, AdtToRepOp);
modelled!(PlacementAdtToRep::adt_to_rep, ReplicatedPlacement, (Additive128Tensor) -> Replicated128Tensor, AdtToRepOp);

hybrid_kernel! {
    AdtToRepOp,
    [
        (ReplicatedPlacement, (Additive64Tensor) -> Replicated64Tensor => Self::kernel),
        (ReplicatedPlacement, (Additive128Tensor) -> Replicated128Tensor => Self::kernel),
    ]
}

impl AdtToRepOp {
    fn kernel<S: Session, ShapeT, SeedT, KeyT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractAdditiveTensor<RingT>,
    ) -> AbstractReplicatedTensor<RingT>
    where
        RingT: Placed<Placement = HostPlacement> + Clone,
        AbstractAdditiveTensor<RingT>: CanonicalType,
        <AbstractAdditiveTensor<RingT> as CanonicalType>::Type: KnownType<S>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        HostPlacement: PlacementKeyGen<S, KeyT>,
        HostPlacement: PlacementSampleUniform<S, ShapeT, SeedT, RingT>,
        HostPlacement: PlacementDeriveSeed<S, KeyT, SeedT>,
        AdditivePlacement: PlacementSub<
            S,
            st!(AbstractAdditiveTensor<RingT>, S),
            st!(AbstractAdditiveTensor<RingT>, S),
            st!(AbstractAdditiveTensor<RingT>, S),
        >,
        AbstractAdditiveTensor<RingT>: Into<st!(AbstractAdditiveTensor<RingT>, S)>,
        HostPlacement: PlacementReveal<S, st!(AbstractAdditiveTensor<RingT>, S), RingT>,
        ReplicatedPlacement: PlacementPlace<S, AbstractReplicatedTensor<RingT>>,
    {
        let AbstractAdditiveTensor { shares: [x0, x1] } = &x;

        let adt = x.placement().unwrap();
        let (adt_player0, adt_player1) = adt.host_placements();
        let (rep_player0, rep_player1, rep_player2) = rep.host_placements();

        let sync_key0 = RawNonce::generate();
        let sync_key1 = RawNonce::generate();
        let shape = adt_player0.shape(sess, x0);

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

        let k = provider.gen_key(sess);
        let seed1 = provider.derive_seed(sess, sync_key0, &k);
        let seed2 = provider.derive_seed(sess, sync_key1, &k);

        let y0_provider = provider.sample_uniform(sess, &shape, &seed1);
        let y1_provider = provider.sample_uniform(sess, &shape, &seed2);

        let y0 = adt_player0.sample_uniform(sess, &shape, &seed1);
        let y1 = adt_player1.sample_uniform(sess, &adt_player1.shape(sess, x1), &seed2);

        let y = AbstractAdditiveTensor {
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
        rep.place(sess, AbstractReplicatedTensor { shares })
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
        let sync_key0 = RawNonce::generate();
        let sync_key1 = RawNonce::generate();
        let sync_key2 = RawNonce::generate();

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

impl<S: Session, RingT> ZeroShareGen<S, cs!(PrfKey), RingT, cs!(Shape)> for ReplicatedPlacement
where
    PrfKey: KnownType<S>,
    Seed: KnownType<S>,
    Shape: KnownType<S>,
    HostPlacement: PlacementSampleUniform<S, cs!(Shape), cs!(Seed), RingT>,
    HostPlacement: PlacementSub<S, RingT, RingT, RingT>,
    ReplicatedPlacement: ReplicatedSeedsGen<S, cs!(PrfKey), cs!(Seed)>,
{
    fn gen_zero_share(
        &self,
        sess: &S,
        setup: &AbstractReplicatedSetup<cs!(PrfKey)>,
        shape: &AbstractReplicatedShape<cs!(Shape)>,
    ) -> AbstractReplicatedZeroShare<RingT> {
        let (player0, player1, player2) = self.host_placements();

        let AbstractReplicatedShape {
            shapes: [shape0, shape1, shape2],
        } = shape;

        let AbstractReplicatedSeeds {
            seeds: [[s00, s10], [s11, s21], [s22, s02]],
        } = &self.gen_seeds(sess, setup);

        let r00 = player0.sample_uniform(sess, shape0, s00);
        let r10 = player0.sample_uniform(sess, shape0, s10);
        let alpha0 = with_context!(player0, sess, r00 - r10);

        let r11 = player1.sample_uniform(sess, shape1, s11);
        let r21 = player1.sample_uniform(sess, shape1, s21);
        let alpha1 = with_context!(player1, sess, r11 - r21);

        let r22 = player2.sample_uniform(sess, shape2, s22);
        let r02 = player2.sample_uniform(sess, shape2, s02);
        let alpha2 = with_context!(player2, sess, r22 - r02);

        AbstractReplicatedZeroShare {
            alphas: [alpha0, alpha1, alpha2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::SyncSession;
    use crate::ring::AbstractRingTensor;
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

        let x1 = Additive64Tensor {
            shares: [
                AbstractRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "alice".into(),
                    },
                ),
                AbstractRingTensor::from_raw_plc(
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

        let x2 = Additive64Tensor {
            shares: [
                AbstractRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "bob".into(),
                    },
                ),
                AbstractRingTensor::from_raw_plc(
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

        let x3 = Additive64Tensor {
            shares: [
                AbstractRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "david".into(),
                    },
                ),
                AbstractRingTensor::from_raw_plc(
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

        let x4 = Additive64Tensor {
            shares: [
                AbstractRingTensor::from_raw_plc(
                    array![1, 2, 3],
                    HostPlacement {
                        owner: "alice".into(),
                    },
                ),
                AbstractRingTensor::from_raw_plc(
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
        let bob = HostPlacement {
            owner: "bob".into(),
        };
        let carole = HostPlacement {
            owner: "carole".into(),
        };

        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        fn ring_tensor(a: Array1<u64>, plc: &HostPlacement) -> Ring64Tensor {
            AbstractRingTensor::from_raw_plc(a, plc.clone())
        }

        let x1 = Replicated64Tensor {
            shares: [
                [
                    ring_tensor(array![1, 2, 3], &alice),
                    ring_tensor(array![11, 12, 13], &alice),
                ],
                [
                    ring_tensor(array![4, 5, 6], &bob),
                    ring_tensor(array![14, 15, 16], &bob),
                ],
                [
                    ring_tensor(array![7, 8, 9], &carole),
                    ring_tensor(array![17, 18, 19], &carole),
                ],
            ],
        };

        let sess = SyncSession::default();

        let res_rep = rep.mean(&sess, None, 0, &x1);
        let res = alice.reveal(&sess, &res_rep);
        println!("Result: {:?}", res);
        // TODO: Asserts
    }

    use ndarray::prelude::*;
    use rstest::rstest;

    #[test]
    fn test_rep_sum() {
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

        fn ring_tensor(a: Array1<u64>, plc: &HostPlacement) -> Ring64Tensor {
            AbstractRingTensor::from_raw_plc(a, plc.clone())
        }

        let x1 = Replicated64Tensor {
            shares: [
                [
                    ring_tensor(array![1, 2, 3], &alice),
                    ring_tensor(array![11, 12, 13], &alice),
                ],
                [
                    ring_tensor(array![4, 5, 6], &bob),
                    ring_tensor(array![14, 15, 16], &bob),
                ],
                [
                    ring_tensor(array![7, 8, 9], &carole),
                    ring_tensor(array![17, 18, 19], &carole),
                ],
            ],
        };

        let sess = SyncSession::default();

        let res_rep = rep.sum(&sess, None, &x1);
        let res = alice.reveal(&sess, &res_rep);
        println!("Result: {:?}", res);
        // TODO: Asserts
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

                let x = AbstractRingTensor::from_raw_plc(xs, alice.clone());
                let y = AbstractRingTensor::from_raw_plc(ys, alice.clone());

                let sess = SyncSession::default();
                let setup = rep.gen_setup(&sess);

                let x_shared = rep.share(&sess, &setup, &x);
                let y_shared = rep.share(&sess, &setup, &y);

                let sum = rep.add(&sess, &x_shared, &y_shared);
                let opened_sum = alice.reveal(&sess, &sum);
                assert_eq!(
                    opened_sum,
                    AbstractRingTensor::from_raw_plc(zs, alice.clone())
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

                let x = AbstractRingTensor::from_raw_plc(xs, alice.clone());
                let y = AbstractRingTensor::from_raw_plc(ys, alice.clone());

                let sess = SyncSession::default();
                let setup = rep.gen_setup(&sess);

                let x_shared = rep.share(&sess, &setup, &x);
                let y_shared = rep.share(&sess, &setup, &y);

                let sum = rep.$test_func(&sess, &setup, &x_shared, &y_shared);
                let opened_product = alice.reveal(&sess, &sum);
                assert_eq!(
                    opened_product,
                    AbstractRingTensor::from_raw_plc(zs, alice.clone())
                );
            }
        };
    }

    rep_binary_func_test!(test_rep_mul64, mul<u64>);
    rep_binary_func_test!(test_rep_mul128, mul<u128>);
    rep_binary_func_test!(test_rep_dot64, dot<u64>);
    rep_binary_func_test!(test_rep_dot128, dot<u128>);

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
                target = target + std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_rep_dot64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_dot128((a,b) in pairwise_same_length128())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target = target + std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_rep_dot128(a, b, target);
        }

    }

    macro_rules! rep_truncation_test {
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

                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let sess = SyncSession::default();
                let setup = rep.gen_setup(&sess);

                let alice_x1 = AbstractRingTensor::from_raw_plc(xs.clone(), alice.clone());
                let alice_rep = rep.share(&sess, &setup, &alice_x1);
                let alice_tr = rep.trunc_pr(&sess, amount, &alice_rep);
                let alice_open = alice.reveal(&sess, &alice_tr);

                let alice_y = AbstractRingTensor::from_raw_plc(ys.clone(), alice.clone());
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

                let bob_x1 = AbstractRingTensor::from_raw_plc(xs.clone(), bob.clone());
                let bob_rep = rep.share(&sess, &setup, &bob_x1);
                let bob_tr = rep.trunc_pr(&sess, amount, &bob_rep);
                let bob_open = bob.reveal(&sess, &bob_tr);

                let bob_y = AbstractRingTensor::from_raw_plc(ys.clone(), bob.clone());
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

                let carole_x1 = AbstractRingTensor::from_raw_plc(xs.clone(), carole.clone());
                let carole_rep = rep.share(&sess, &setup, &carole_x1);
                let carole_tr = rep.trunc_pr(&sess, amount, &carole_rep);
                let carole_open = carole.reveal(&sess, &carole_tr);

                let carole_y = AbstractRingTensor::from_raw_plc(ys.clone(), bob.clone());
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
        #[case] amount: usize,
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
        #[case] amount: usize,
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
        fn test_fuzzy_rep_trunc64(raw_vector in proptest::collection::vec(any_bounded_u64(), 1..5), amount in 0usize..62
        ) {
            let target = raw_vector.iter().map(|x| x >> amount).collect::<Vec<_>>();
            test_rep_truncation64(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), amount, Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }

        #[test]
        fn test_fuzzy_rep_trunc128(raw_vector in proptest::collection::vec(any_bounded_u128(), 1..5), amount in 0usize..126
        ) {
            let target = raw_vector.iter().map(|x| x >> amount).collect::<Vec<_>>();
            test_rep_truncation128(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), amount, Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }
    }
}
