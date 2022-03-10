//! Support for generating zero-shares

use super::*;
use crate::execution::SetupGeneration;
use crate::host::{HostPrfKey, HostSeed, SyncKey};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct RepZeroShare<HostRingT> {
    pub(crate) alphas: [HostRingT; 3],
}

pub(crate) trait ZeroShareGen<S: Session, ShapeT, RingT> {
    fn gen_zero_share(&self, sess: &S, shape: &RepShape<ShapeT>) -> Result<RepZeroShare<RingT>>;
}

impl<S: Session, RingT, ShapeT, SeedT, KeyT> ZeroShareGen<S, ShapeT, RingT> for ReplicatedPlacement
where
    S: SetupGeneration<ReplicatedPlacement, Setup = RepSetup<KeyT>>,
    HostPlacement: PlacementSampleUniformSeeded<S, ShapeT, SeedT, RingT>,
    HostPlacement: PlacementSub<S, RingT, RingT, RingT>,
    ReplicatedPlacement: SeedsGen<S, HostSeed = SeedT>,
{
    fn gen_zero_share(&self, sess: &S, shape: &RepShape<ShapeT>) -> Result<RepZeroShare<RingT>> {
        let (player0, player1, player2) = self.host_placements();

        let RepShape {
            shapes: [shape0, shape1, shape2],
        } = shape;

        let RepSeeds {
            seeds: [[s00, s10], [s11, s21], [s22, s02]],
        } = &self.gen_seeds(sess)?;

        let r00 = player0.sample_uniform_seeded(sess, shape0, s00);
        let r10 = player0.sample_uniform_seeded(sess, shape0, s10);
        let alpha0 = with_context!(player0, sess, r00 - r10);

        let r11 = player1.sample_uniform_seeded(sess, shape1, s11);
        let r21 = player1.sample_uniform_seeded(sess, shape1, s21);
        let alpha1 = with_context!(player1, sess, r11 - r21);

        let r22 = player2.sample_uniform_seeded(sess, shape2, s22);
        let r02 = player2.sample_uniform_seeded(sess, shape2, s02);
        let alpha2 = with_context!(player2, sess, r22 - r02);

        Ok(RepZeroShare {
            alphas: [alpha0, alpha1, alpha2],
        })
    }
}

pub(crate) struct RepSeeds<HostSeedT> {
    seeds: [[HostSeedT; 2]; 3],
}

pub(crate) trait SeedsGen<S: Session> {
    type HostSeed;
    fn gen_seeds(&self, sess: &S) -> Result<RepSeeds<Self::HostSeed>>;
}

impl<S: Session> SeedsGen<S> for ReplicatedPlacement
where
    HostPrfKey: KnownType<S>,
    HostSeed: KnownType<S>,
    HostPlacement: PlacementDeriveSeed<S, m!(HostPrfKey), m!(HostSeed)>,
    S: SetupGeneration<ReplicatedPlacement, Setup = RepSetup<m!(HostPrfKey)>>,
{
    type HostSeed = m!(HostSeed);

    fn gen_seeds(&self, sess: &S) -> Result<RepSeeds<m!(HostSeed)>> {
        let (player0, player1, player2) = self.host_placements();

        let setup = sess.setup(self)?;
        let RepSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = setup.as_ref();

        // NOTE for now we pick random sync_keys _at compile time_, which is okay from
        // a security perspective since the seeds depend on both the keys and the sid.
        // however, with sub-computations we could fix these as eg `0`, `1`, and `2`
        // and make compilation a bit more deterministic
        let sync_key0 = SyncKey::random();
        let sync_key1 = SyncKey::random();
        let sync_key2 = SyncKey::random();

        let s00 = player0.derive_seed(sess, sync_key0.clone(), k00);
        let s10 = player0.derive_seed(sess, sync_key1.clone(), k10);

        let s11 = player1.derive_seed(sess, sync_key1, k11);
        let s21 = player1.derive_seed(sess, sync_key2.clone(), k21);

        let s22 = player2.derive_seed(sess, sync_key2, k22);
        let s02 = player2.derive_seed(sess, sync_key0, k02);

        let seeds = [[s00, s10], [s11, s21], [s22, s02]];
        Ok(RepSeeds { seeds })
    }
}
