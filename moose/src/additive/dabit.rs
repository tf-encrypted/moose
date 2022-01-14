//! DaBit generation for additive placements
use super::*;
use crate::computation::{HostPlacement, KnownType};
use crate::kernels::*;
use crate::host::SyncKey;
use crate::types::{PrfKey, Seed};
use macros::with_context;

/// Internal trait for DaBit generation
pub trait DaBitProvider<S: Session, HostShapeT, O1, O2> {
    fn gen_dabit(
        &self,
        sess: &S,
        shape_provider: HostShapeT,
        shape_player0: HostShapeT,
        provider: &HostPlacement,
    ) -> (O1, O2);
}

impl<S: Session, HostShapeT, HostRingT, HostBitT>
    DaBitProvider<S, HostShapeT, AdtTensor<HostRingT>, AdtTensor<HostBitT>> for AdditivePlacement
where
    HostRingT: Clone,
    Seed: KnownType<S>,
    PrfKey: KnownType<S>,
    HostPlacement: PlacementKeyGen<S, m!(PrfKey)>,
    HostPlacement: PlacementDeriveSeed<S, m!(PrfKey), m!(Seed)>,
    HostPlacement: PlacementSampleUniform<S, HostShapeT, HostBitT>,
    HostPlacement: PlacementSampleUniformSeeded<S, HostShapeT, m!(Seed), HostBitT>,
    HostPlacement: PlacementSampleUniformSeeded<S, HostShapeT, m!(Seed), HostRingT>,
    HostPlacement: PlacementSub<S, HostBitT, HostBitT, HostBitT>,
    HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
    HostPlacement: PlacementRingInject<S, HostBitT, HostRingT>,
    HostPlacement: PlacementPlace<S, HostRingT>,
    HostPlacement: PlacementPlace<S, HostBitT>,
{
    fn gen_dabit(
        &self,
        sess: &S,
        shape_provider: HostShapeT,
        shape_player0: HostShapeT,
        provider: &HostPlacement,
    ) -> (AdtTensor<HostRingT>, AdtTensor<HostBitT>) {
        let (player0, player1) = self.host_placements();
        assert!(*provider != player0);
        assert!(*provider != player1);

        let b: HostBitT = provider.sample_uniform(sess, &shape_provider);
        let br: HostRingT = provider.ring_inject(sess, 0, &b);

        let key = provider.gen_key(sess);
        let seed_b = provider.derive_seed(sess, SyncKey::random(), &key);
        let seed_br = provider.derive_seed(sess, SyncKey::random(), &key);

        let b0_provider: HostBitT = provider.sample_uniform_seeded(sess, &shape_provider, &seed_b);
        let b0: HostBitT = player0.sample_uniform_seeded(sess, &shape_player0, &seed_b);
        let b1: HostBitT = player1.place(sess, with_context!(provider, sess, b - b0_provider));

        let br0_provider: HostRingT =
            provider.sample_uniform_seeded(sess, &shape_provider, &seed_br);
        let br0: HostRingT = player0.sample_uniform_seeded(sess, &shape_player0, &seed_br);
        let br1: HostRingT = player1.place(sess, with_context!(provider, sess, br - br0_provider));

        let b_shared = AdtTensor { shares: [b0, b1] };
        let br_shared = AdtTensor { shares: [br0, br1] };

        (br_shared, b_shared)
    }
}
