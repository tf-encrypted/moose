use crate::computation::{HostPlacement, Placed, PrimDeriveSeedOp, PrimPrfKeyGenOp, TAG_BYTES};
use crate::error::Result;
use crate::kernels::{NullaryKernel, PlacementDeriveSeed, PlacementKeyGen, PlacementPlace, RuntimeSession, Session};
use crate::prng::AesRng;
use crate::prng::{RngSeed, SEED_SIZE};
use crate::symbolic::Symbolic;
use serde::{Deserialize, Serialize};
use sodiumoxide::crypto::generichash;
use std::convert::TryFrom;

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawSeed(pub [u8; 16]);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Seed(pub RawSeed, pub HostPlacement);

moose_type!(Seed);

impl Placed for Seed {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl<S: Session> PlacementPlace<S, Seed> for HostPlacement {
    fn place(&self, _sess: &S, seed: Seed) -> Seed {
        match seed.placement() {
            Ok(place) if &place == self => seed,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                Seed(seed.0, self.clone())
            }
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawPrfKey(pub [u8; 16]);

impl RawPrfKey {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct PrfKey(pub RawPrfKey, pub HostPlacement);

moose_type!(PrfKey);

impl Placed for PrfKey {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl<S: Session> PlacementPlace<S, PrfKey> for HostPlacement {
    fn place(&self, _sess: &S, key: PrfKey) -> PrfKey {
        match key.placement() {
            Ok(place) if self == &place => key,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                PrfKey(key.0, self.clone())
            }
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct SyncKey([u8; TAG_BYTES]);

impl SyncKey {
    pub fn random() -> SyncKey {
        let mut raw_sync_key = [0u8; TAG_BYTES];
        sodiumoxide::init().expect("failed to initialize sodiumoxide");
        sodiumoxide::randombytes::randombytes_into(&mut raw_sync_key);
        SyncKey(raw_sync_key)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl TryFrom<Vec<u8>> for SyncKey {
    type Error = crate::error::Error;
    fn try_from(bytes: Vec<u8>) -> crate::error::Result<SyncKey> {
        Self::try_from(bytes.as_ref())
    }
}

impl TryFrom<&[u8]> for SyncKey {
    type Error = crate::error::Error;
    fn try_from(bytes: &[u8]) -> crate::error::Result<SyncKey> {
        if bytes.len() <= TAG_BYTES {
            // TODO easier way of doing this?
            let mut sync_key_bytes = [0; TAG_BYTES];
            for (idx, byte) in bytes.iter().enumerate() {
                sync_key_bytes[idx] = *byte;
            }
            Ok(SyncKey(sync_key_bytes))
        } else {
            Err(crate::error::Error::Unexpected) // TODO more helpful error message
        }
    }
}

modelled!(PlacementKeyGen::gen_key, HostPlacement, () -> PrfKey, PrimPrfKeyGenOp);

kernel! {
    PrimPrfKeyGenOp,
    [
        (HostPlacement, () -> PrfKey => [runtime] Self::kernel),
    ]
}

impl PrimPrfKeyGenOp {
    fn kernel<S: RuntimeSession>(_sess: &S, plc: &HostPlacement) -> PrfKey {
        let raw_key = RawPrfKey(AesRng::generate_random_key());
        PrfKey(raw_key, plc.clone())
    }
}

modelled!(PlacementDeriveSeed::derive_seed, HostPlacement, attributes[sync_key: SyncKey] (PrfKey) -> Seed, PrimDeriveSeedOp);

kernel! {
    PrimDeriveSeedOp,
    [
        (HostPlacement, (PrfKey) -> Seed => [runtime] attributes[sync_key] Self::kernel),
    ]
}

impl PrimDeriveSeedOp {
    fn kernel<S: RuntimeSession>(
        sess: &S,
        plc: &HostPlacement,
        sync_key: SyncKey,
        key: PrfKey,
    ) -> Seed {
        let sid = sess.session_id();
        let key_bytes = key.0.as_bytes();

        // compute seed as hash(sid || sync_key)[0..SEED_SIZE]
        let sid_bytes: [u8; TAG_BYTES] = sid.0;
        let sync_key_bytes: [u8; TAG_BYTES] = sync_key.0;
        let mut nonce: Vec<u8> = Vec::with_capacity(2 * TAG_BYTES);
        nonce.extend(&sid_bytes);
        nonce.extend(&sync_key_bytes);
        sodiumoxide::init().expect("failed to initialize sodiumoxide");
        let digest = generichash::hash(&nonce, Some(SEED_SIZE), Some(key_bytes)).unwrap();
        let mut raw_seed: RngSeed = [0u8; SEED_SIZE];
        raw_seed.copy_from_slice(digest.as_ref());

        Seed(RawSeed(raw_seed), plc.clone())
    }
}

// TODO deprecated
impl RawSeed {
    #[cfg_attr(
        feature = "exclude_old_framework",
        deprecated(
            note = "This function is only used by the old kernels, which are not aware of the placements."
        )
    )]
    pub fn from_prf(key: &RawPrfKey, sync_key: &SyncKey) -> RawSeed {
        let nonce = &sync_key.0;
        let raw_seed = crate::utils::derive_seed(&key.0, nonce);
        RawSeed(raw_seed)
    }
}

// TODO deprecated
impl RawPrfKey {
    #[cfg_attr(
        feature = "exclude_old_framework",
        deprecated(
            note = "This function is only used by the old kernels, which are not aware of the placements."
        )
    )]
    pub fn generate() -> RawPrfKey {
        let raw_key = AesRng::generate_random_key();
        RawPrfKey(raw_key)
    }
}
