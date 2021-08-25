use crate::computation::{TAG_BYTES, HostPlacement, Placed, PrimDeriveSeedOp, PrimPrfKeyGenOp};
use crate::error::Result;
use crate::kernels::{
    NullaryKernel, PlacementDeriveSeed, PlacementKeyGen, PlacementPlace, RuntimeSession,
    SyncSession,
};
use crate::prng::AesRng;
use crate::symbolic::{Symbolic, SymbolicHandle, SymbolicSession};
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawSeed(pub [u8; 16]);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Seed(pub RawSeed, pub HostPlacement);

impl Placed for Seed {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl PlacementPlace<SyncSession, Seed> for HostPlacement {
    fn place(&self, _sess: &SyncSession, seed: Seed) -> Seed {
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

impl PlacementPlace<SymbolicSession, Symbolic<Seed>> for HostPlacement {
    fn place(&self, _sess: &SymbolicSession, x: Symbolic<Seed>) -> Symbolic<Seed> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                match x {
                    Symbolic::Concrete(seed) => {
                        // TODO insert Place ops?
                        Symbolic::Concrete(Seed(seed.0, self.clone()))
                    }
                    Symbolic::Symbolic(SymbolicHandle { op, plc: _ }) => {
                        // TODO insert `Place` ops here?
                        Symbolic::Symbolic(SymbolicHandle {
                            op,
                            plc: self.clone(),
                        })
                    }
                }
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

impl Placed for PrfKey {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.1.clone())
    }
}

impl PlacementPlace<SyncSession, PrfKey> for HostPlacement {
    fn place(&self, _sess: &SyncSession, key: PrfKey) -> PrfKey {
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

impl PlacementPlace<SymbolicSession, Symbolic<PrfKey>> for HostPlacement {
    fn place(&self, _sess: &SymbolicSession, x: Symbolic<PrfKey>) -> Symbolic<PrfKey> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                match x {
                    Symbolic::Concrete(key) => {
                        // TODO insert Place ops?
                        Symbolic::Concrete(PrfKey(key.0, self.clone()))
                    }
                    Symbolic::Symbolic(SymbolicHandle { op, plc: _ }) => {
                        // TODO insert `Place` ops here?
                        Symbolic::Symbolic(SymbolicHandle {
                            op,
                            plc: self.clone(),
                        })
                    }
                }
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
        SyncKey(raw_sync_key.into())
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
        (HostPlacement, () -> PrfKey => Self::kernel),
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
        (HostPlacement, (PrfKey) -> Seed => attributes[sync_key] Self::kernel),
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

        // TODO(Morten) how do we concatenate; describe motivations
        let mut nonce: Vec<u8> = vec![];
        nonce.extend(sid.as_bytes());
        nonce.extend(sync_key.0);

        // TODO(Morten) optimize below?
        use crate::prng::{RngSeed, SEED_SIZE};
        use sodiumoxide::crypto::generichash;
        sodiumoxide::init().expect("failed to initialize sodiumoxide");
        let mut hasher = generichash::State::new(Some(SEED_SIZE), Some(key_bytes)).expect("failed to initialize sodiumoxide hash function");
        hasher.update(&nonce).unwrap();
        let h = hasher.finalize().unwrap();
        let mut raw_seed: RngSeed = [0u8; SEED_SIZE];
        raw_seed.copy_from_slice(h.as_ref());
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
