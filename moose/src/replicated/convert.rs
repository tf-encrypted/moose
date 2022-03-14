//! Support for various conversion operators

use super::*;
use crate::additive::{AdditivePlacement, AdtTensor, DaBitProvider};
use crate::execution::SetupGeneration;
use crate::host::{AbstractHostAesKey, HostBitArray, HostFixedTensor, SyncKey};
use std::convert::TryInto;

impl ShareOp {
    pub(crate) fn aeskey_kernel<S: Session, HostBitArrayT, RepBitArrayT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        key: AbstractHostAesKey<HostBitArrayT>,
    ) -> Result<RepAesKey<RepBitArrayT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostBitArrayT, RepBitArrayT>,
    {
        let bit_array = plc.share(sess, &key.0);
        Ok(RepAesKey(bit_array))
    }

    pub(crate) fn fixed_kernel<S: Session, HostRingT, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: HostFixedTensor<HostRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementShare<S, HostRingT, RepRingT>,
    {
        Ok(RepFixedTensor {
            tensor: plc.share(sess, &x.tensor),
            fractional_precision: x.fractional_precision,
            integral_precision: x.integral_precision,
        })
    }

    pub(crate) fn array_kernel<S: Session, HostBitTensorT, RepBitTensorT, N: Const>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: HostBitArray<HostBitTensorT, N>,
    ) -> Result<RepBitArray<RepBitTensorT, N>>
    where
        ReplicatedPlacement: PlacementShare<S, HostBitTensorT, RepBitTensorT>,
    {
        let shared_tensor = plc.share(sess, &x.0);
        Ok(RepBitArray(shared_tensor, x.1))
    }

    pub(crate) fn ring_kernel<S: Session, ShapeT, SeedT, KeyT, RingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RingT,
    ) -> Result<RepTensor<RingT>>
    where
        S: SetupGeneration<ReplicatedPlacement, Setup = RepSetup<KeyT>>,
        RingT: Clone + Placed<Placement = HostPlacement>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        HostPlacement: PlacementSampleUniformSeeded<S, ShapeT, SeedT, RingT>,
        HostPlacement: PlacementZeros<S, ShapeT, RingT>,
        HostPlacement: PlacementDeriveSeed<S, KeyT, SeedT>,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementSub<S, RingT, RingT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTensor<RingT>>,
    {
        let x_player = x.placement()?;

        let setup = sess.setup(plc)?;
        let RepSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = setup.as_ref();

        let (player0, player1, player2) = plc.host_placements();

        let shares = match () {
            _ if x_player == player0 => {
                let sync_key = SyncKey::random();
                let shape = x_player.shape(sess, &x);

                let seed0 = player0.derive_seed(sess, sync_key.clone(), k00);
                let x00 = x_player.sample_uniform_seeded(sess, &shape, &seed0);
                let x10 = with_context!(x_player, sess, x - x00);

                let seed2 = player2.derive_seed(sess, sync_key, k02);
                let x22 = player2.zeros(sess, &shape);
                let x02 = player2.sample_uniform_seeded(sess, &shape, &seed2);

                let x11 = x10.clone();
                let x21 = player1.zeros(sess, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ if x_player == player1 => {
                let sync_key = SyncKey::random();
                let shape = x_player.shape(sess, &x);

                let seed1 = player1.derive_seed(sess, sync_key.clone(), k11);
                let x11 = x_player.sample_uniform_seeded(sess, &shape, &seed1);
                let x21 = with_context!(x_player, sess, x - x11);

                let seed0 = player0.derive_seed(sess, sync_key, k10);
                let x00 = player0.zeros(sess, &shape);
                let x10 = player0.sample_uniform_seeded(sess, &shape, &seed0);

                let x22 = x21.clone();
                let x02 = player2.zeros(sess, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ if x_player == player2 => {
                let sync_key = SyncKey::random();
                let shape = x_player.shape(sess, &x);

                let seed2 = player2.derive_seed(sess, sync_key.clone(), k22);
                let x22 = player2.sample_uniform_seeded(sess, &shape, &seed2);
                let x02 = with_context!(x_player, sess, x - x22);

                let seed1 = player1.derive_seed(sess, sync_key, k21);
                let x11 = player1.zeros(sess, &shape);
                let x21 = player1.sample_uniform_seeded(sess, &shape, &seed1);

                let x00 = x02.clone();
                let x10 = player0.zeros(sess, &shape);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
            _ => {
                // in this case, where x_owner is _not_ among the replicated players,
                // we cannot use the zeros optimization trick but we can still make sure
                // that seeds are used as much as possible instead of dense random tensors;
                // however, we must make sure keys are not revealed to x_owner and only seeds

                let sync_key0 = SyncKey::random();
                let sync_key1 = SyncKey::random();
                let shape = x_player.shape(sess, &x);

                let seed00 = player0.derive_seed(sess, sync_key0.clone(), k00);
                let seed02 = player2.derive_seed(sess, sync_key0, k02);

                let seed11 = player1.derive_seed(sess, sync_key1.clone(), k11);
                let seed10 = player0.derive_seed(sess, sync_key1, k10);

                let x0 = x_player.sample_uniform_seeded(sess, &shape, &seed00);
                let x1 = x_player.sample_uniform_seeded(sess, &shape, &seed11);
                let x2 = with_context!(x_player, sess, x - x0 - x1);

                let x00 = player0.sample_uniform_seeded(sess, &shape, &seed00);
                let x10 = player0.sample_uniform_seeded(sess, &shape, &seed10);

                let x11 = player1.sample_uniform_seeded(sess, &shape, &seed11);
                let x21 = x2.clone();

                let x22 = x2;
                let x02 = player2.sample_uniform_seeded(sess, &shape, &seed02);

                [[x00, x10], [x11, x21], [x22, x02]]
            }
        };

        Ok(plc.place(sess, RepTensor { shares }))
    }
}

impl RevealOp {
    pub(crate) fn shape_kernel<S: Session>(
        sess: &S,
        receiver: &HostPlacement,
        shape: ReplicatedShape,
    ) -> Result<HostShape>
    where
        HostPlacement: PlacementPlace<S, HostShape>,
    {
        let rep_plc = shape.placement()?;
        let (h0, h1, h2) = rep_plc.host_placements();
        if receiver == &h0 {
            Ok(shape.shapes[0].clone())
        } else if receiver == &h1 {
            Ok(shape.shapes[1].clone())
        } else if receiver == &h2 {
            Ok(shape.shapes[2].clone())
        } else {
            Ok(receiver.place(sess, shape.shapes[0].clone()))
        }
    }

    pub(crate) fn host_aeskey_kernel<S: Session, RepBitArrayT, HostBitArrayT>(
        sess: &S,
        receiver: &HostPlacement,
        key: RepAesKey<RepBitArrayT>,
    ) -> Result<AbstractHostAesKey<HostBitArrayT>>
    where
        HostPlacement: PlacementReveal<S, RepBitArrayT, HostBitArrayT>,
    {
        let bit_array = receiver.reveal(sess, &key.0);
        Ok(AbstractHostAesKey(bit_array))
    }

    pub(crate) fn host_fixed_kernel<S: Session, RepRingT, HostRingT>(
        sess: &S,
        receiver: &HostPlacement,
        xe: RepFixedTensor<RepRingT>,
    ) -> Result<HostFixedTensor<HostRingT>>
    where
        HostPlacement: PlacementReveal<S, RepRingT, HostRingT>,
    {
        let x = receiver.reveal(sess, &xe.tensor);
        Ok(HostFixedTensor {
            tensor: x,
            fractional_precision: xe.fractional_precision,
            integral_precision: xe.integral_precision,
        })
    }

    pub(crate) fn host_bit_array_kernel<S: Session, RepBitT, HostBitT, N>(
        sess: &S,
        receiver: &HostPlacement,
        xe: RepBitArray<RepBitT, N>,
    ) -> Result<HostBitArray<HostBitT, N>>
    where
        HostPlacement: PlacementReveal<S, RepBitT, HostBitT>,
    {
        let x = receiver.reveal(sess, &xe.0);
        Ok(HostBitArray(x, PhantomData))
    }

    pub(crate) fn host_uint64_kernel<S: Session, RepRingT>(
        sess: &S,
        receiver: &HostPlacement,
        xe: RepUintTensor<RepRingT>,
    ) -> Result<m!(HostUint64Tensor)>
    where
        HostRing64Tensor: KnownType<S>,
        HostUint64Tensor: KnownType<S>,
        HostPlacement: PlacementReveal<S, RepRingT, m!(HostRing64Tensor)>,
        HostPlacement: PlacementCast<S, m!(HostRing64Tensor), m!(HostUint64Tensor)>,
    {
        let x = receiver.reveal(sess, &xe.tensor);
        Ok(receiver.cast(sess, &x))
    }

    pub(crate) fn host_ring_kernel<S: Session, R: Clone>(
        sess: &S,
        receiver: &HostPlacement,
        xe: RepTensor<R>,
    ) -> Result<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, R, R, R>,
    {
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &xe;

        let (player0, player1, player2) = &xe.placement()?.host_placements();

        let res = match () {
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
        };
        Ok(res)
    }
}

impl RingInjectOp {
    pub(crate) fn rep_kernel<S: Session, HostBitT, HostRingT, HostShapeT, AdtRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        bit_idx: usize,
        x: RepTensor<HostBitT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        AdtTensor<HostRingT>: CanonicalType,
        <AdtTensor<HostRingT> as CanonicalType>::Type: KnownType<S>,
        AdtTensor<HostRingT>: Into<m!(c!(AdtTensor<HostRingT>))>,

        AdtTensor<HostBitT>: CanonicalType,
        <AdtTensor<HostBitT> as CanonicalType>::Type: KnownType<S>,
        AdtTensor<HostBitT>: Into<m!(c!(AdtTensor<HostBitT>))>,

        AdtTensor<HostRingT>: Into<AdtRingT>,
        m!(c!(AdtTensor<HostRingT>)): TryInto<AdtTensor<HostRingT>>,
        AdtRingT: TryInto<AdtTensor<HostRingT>>,

        HostPlacement: PlacementShape<S, HostBitT, HostShapeT>,
        ReplicatedPlacement: PlacementAdtToRep<S, AdtTensor<HostRingT>, RepTensor<HostRingT>>,
        AdditivePlacement: PlacementFill<S, HostShapeT, AdtRingT>,
        HostPlacement: PlacementFill<S, HostShapeT, HostRingT>,
        AdditivePlacement: DaBitProvider<S, HostShapeT, AdtTensor<HostRingT>, AdtTensor<HostBitT>>,
        AdditivePlacement: PlacementRepToAdt<S, RepTensor<HostBitT>, AdtTensor<HostBitT>>,
        AdditivePlacement:
            PlacementAdd<S, AdtTensor<HostBitT>, AdtTensor<HostBitT>, AdtTensor<HostBitT>>,
        AdditivePlacement: PlacementAdd<S, AdtRingT, HostRingT, AdtRingT>,
        AdditivePlacement: PlacementMul<S, AdtRingT, HostRingT, AdtRingT>,
        AdditivePlacement: PlacementSub<S, AdtRingT, AdtRingT, AdtRingT>,
        AdditivePlacement: PlacementShl<S, AdtRingT, AdtRingT>,
        HostPlacement: PlacementReveal<S, m!(c!(AdtTensor<HostBitT>)), HostBitT>,
        HostPlacement: PlacementRingInject<S, HostBitT, HostRingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let adt = AdditivePlacement {
            owners: [player0.owner.clone(), player1.owner],
        };
        let provider = player2;

        let RepTensor {
            shares: [[x00, _x10], [_x11, _x21], [x22, _x02]],
        } = &x;

        let shape_provider = provider.shape(sess, x22);
        let shape_player0 = player0.shape(sess, x00);

        // One could think to wrap this up into an additive shape for Hosts@(P0, P2)
        // but the additive placement that generates a dabit is Hosts@(P0, P1)
        // to avoid confusion the API corresponding gen_dabit takes two input shapes
        // 1) shape_provider - provider (dealer) shape
        // 2) shape_player0 - shape that corresponds to the party expanding the seeds received from provider.

        let (b_ring, b_bin) = adt.gen_dabit(sess, shape_provider, shape_player0, &provider);
        let x_adt = adt.rep_to_adt(sess, &x);

        // TODO(Morten) the following block would likely clean up nicely if we instead
        // revealed to a mirrored-2 placement, which would use only concrete kernels
        let c = with_context!(adt, sess, x_adt + b_bin);
        let c_open = player0.reveal(sess, &c.into());
        let c_ring = player0.ring_inject(sess, 0, &c_open);
        let b_ring = b_ring.into();
        let x_adt_ring = with_context!(
            adt,
            sess,
            b_ring + c_ring - b_ring * c_ring - b_ring * c_ring
        );
        let shifted_x_adt = adt.shl(sess, bit_idx, &x_adt_ring);
        let shifted_x_adt = shifted_x_adt.try_into().ok().unwrap();

        Ok(rep.adt_to_rep(sess, &shifted_x_adt))
    }
}

impl AdtToRepOp {
    pub(crate) fn kernel<S: Session, ShapeT, SeedT, KeyT, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AdtTensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement> + Clone,
        AdtTensor<HostRingT>: CanonicalType,
        <AdtTensor<HostRingT> as CanonicalType>::Type: KnownType<S>,
        HostPlacement: PlacementShape<S, HostRingT, ShapeT>,
        HostPlacement: PlacementKeyGen<S, KeyT>,
        HostPlacement: PlacementSampleUniformSeeded<S, ShapeT, SeedT, HostRingT>,
        HostPlacement: PlacementDeriveSeed<S, KeyT, SeedT>,
        AdditivePlacement:
            PlacementSub<S, AdtTensor<HostRingT>, AdtTensor<HostRingT>, AdtTensor<HostRingT>>,
        AdtTensor<HostRingT>: Into<m!(c!(AdtTensor<HostRingT>))>,
        HostPlacement: PlacementReveal<S, m!(c!(AdtTensor<HostRingT>)), HostRingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTensor<HostRingT>>,
    {
        let AdtTensor { shares: [x0, x1] } = &x;

        let adt = x.placement()?;
        let (adt_player0, adt_player1) = adt.host_placements();
        let (rep_player0, rep_player1, rep_player2) = rep.host_placements();
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
            _ => unimplemented!("protocol error in AdtToRep kernel"), // something is wrong in the protocol otherwise
        };

        let sync_key0 = SyncKey::random();
        let sync_key1 = SyncKey::random();

        let k = provider.gen_key(sess);
        let seed1 = provider.derive_seed(sess, sync_key0, &k);
        let seed2 = provider.derive_seed(sess, sync_key1, &k);

        let shape0 = adt_player0.shape(sess, x0);
        let shape1 = adt_player1.shape(sess, x1);

        let y0 = adt_player0.sample_uniform_seeded(sess, &shape0, &seed1);
        let y1 = adt_player1.sample_uniform_seeded(sess, &shape1, &seed2);

        let y0_provider = provider.sample_uniform_seeded(sess, &shape0, &seed1);
        let y1_provider = provider.sample_uniform_seeded(sess, &shape0, &seed2);

        let y = AdtTensor {
            shares: [y0.clone(), y1.clone()],
        };
        let c = adt_player0.reveal(sess, &adt.sub(sess, &x, &y).into());

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
        Ok(rep.place(sess, RepTensor { shares }))
    }
}

impl CastOp {
    pub(crate) fn rep_reduction_kernel<S: Session, HostT1, HostT2>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<HostT1>,
    ) -> Result<RepTensor<HostT2>>
    where
        HostPlacement: PlacementCast<S, HostT1, HostT2>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        Ok(RepTensor {
            shares: [
                [player0.cast(sess, x00), player0.cast(sess, x10)],
                [player1.cast(sess, x11), player1.cast(sess, x21)],
                [player2.cast(sess, x22), player2.cast(sess, x02)],
            ],
        })
    }
}
