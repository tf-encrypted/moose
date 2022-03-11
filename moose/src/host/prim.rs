use super::HostPlacement;
use crate::computation::{DeriveSeedOp, Placed, PrfKeyGenOp, TAG_BYTES};
use crate::error::{Error, Result};
use crate::execution::{RuntimeSession, Session};
use crate::kernels::PlacementPlace;
use crate::prng::AesRng;
use crate::prng::{RngSeed, SEED_SIZE};
use serde::{Deserialize, Serialize};
use sodiumoxide::crypto::generichash;
use std::convert::TryFrom;

#[derive(Serialize, Deserialize, PartialEq, Hash, Clone, Debug)]
pub struct RawSeed(pub [u8; 16]);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct HostSeed(pub RawSeed, pub HostPlacement);

impl Placed for HostSeed {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl<S: Session> PlacementPlace<S, HostSeed> for HostPlacement {
    fn place(&self, _sess: &S, seed: HostSeed) -> HostSeed {
        match seed.placement() {
            Ok(place) if &place == self => seed,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                HostSeed(seed.0, self.clone())
            }
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Hash, Clone, Debug)]
pub struct RawPrfKey(pub [u8; 16]);

impl RawPrfKey {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct HostPrfKey(pub RawPrfKey, pub HostPlacement);

impl Placed for HostPrfKey {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl<S: Session> PlacementPlace<S, HostPrfKey> for HostPlacement {
    fn place(&self, _sess: &S, key: HostPrfKey) -> HostPrfKey {
        match key.placement() {
            Ok(place) if self == &place => key,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                HostPrfKey(key.0, self.clone())
            }
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug)]
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

    pub fn from_bytes(bytes: [u8; TAG_BYTES]) -> Self {
        SyncKey(bytes)
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
            Err(crate::error::Error::Unexpected(None)) // TODO more helpful error message
        }
    }
}

impl PrfKeyGenOp {
    pub(crate) fn kernel<S: RuntimeSession>(_sess: &S, plc: &HostPlacement) -> Result<HostPrfKey> {
        let raw_key = RawPrfKey(AesRng::generate_random_key());
        Ok(HostPrfKey(raw_key, plc.clone()))
    }
}

impl DeriveSeedOp {
    pub(crate) fn kernel<S: RuntimeSession>(
        sess: &S,
        plc: &HostPlacement,
        sync_key: SyncKey,
        key: HostPrfKey,
    ) -> Result<HostSeed> {
        let sid = sess.session_id();
        let key_bytes = key.0.as_bytes();

        // compute seed as hash(sid || sync_key)[0..SEED_SIZE]
        let sid_bytes: [u8; TAG_BYTES] = *sid.as_bytes();
        let sync_key_bytes: [u8; TAG_BYTES] = sync_key.0;
        let mut nonce: Vec<u8> = Vec::with_capacity(2 * TAG_BYTES);
        nonce.extend(&sid_bytes);
        nonce.extend(&sync_key_bytes);
        sodiumoxide::init().expect("failed to initialize sodiumoxide");
        let digest = generichash::hash(&nonce, Some(SEED_SIZE), Some(key_bytes))
            .map_err(|_e| Error::KernelError("Failure to derive seed.".to_string()))?;
        let mut raw_seed: RngSeed = [0u8; SEED_SIZE];
        raw_seed.copy_from_slice(digest.as_ref());

        Ok(HostSeed(RawSeed(raw_seed), plc.clone()))
    }
}
