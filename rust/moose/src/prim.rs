use crate::computation::{HostPlacement, Placed, PrimDeriveSeedOp, PrimPrfKeyGenOp};
use crate::kernels::{
    NewSyncSession, NullaryKernel, PlacementDeriveSeed, PlacementKeyGen, PlacementPlace,
    RuntimeSession,
};
use crate::prng::AesRng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawSeed(pub [u8; 16]);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Seed(pub RawSeed, pub HostPlacement);

impl Placed for Seed {
    type Placement = HostPlacement;

    fn placement(&self) -> Self::Placement {
        self.1.clone()
    }
}

impl PlacementPlace<NewSyncSession, Seed> for HostPlacement {
    fn place(&self, _sess: &NewSyncSession, seed: Seed) -> Seed {
        if self == &seed.placement() {
            seed
        } else {
            // TODO just updating the placement isn't enough,
            // we need this to eventually turn into Send + Recv
            Seed(seed.0, self.clone())
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawPrfKey(pub [u8; 16]);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct PrfKey(pub RawPrfKey, pub HostPlacement);

impl Placed for PrfKey {
    type Placement = HostPlacement;

    fn placement(&self) -> Self::Placement {
        self.1.clone()
    }
}

impl PlacementPlace<NewSyncSession, PrfKey> for HostPlacement {
    fn place(&self, _sess: &NewSyncSession, key: PrfKey) -> PrfKey {
        if self == &key.placement() {
            key
        } else {
            // TODO just updating the placement isn't enough,
            // we need this to eventually turn into Send + Recv
            PrfKey(key.0, self.clone())
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawNonce(pub Vec<u8>);

impl RawNonce {
    pub fn generate() -> RawNonce {
        let nonce = AesRng::generate_random_key();
        RawNonce(nonce.into())
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Nonce(pub RawNonce, pub HostPlacement);

impl Placed for Nonce {
    type Placement = HostPlacement;

    fn placement(&self) -> Self::Placement {
        self.1.clone()
    }
}

impl PlacementPlace<NewSyncSession, Nonce> for HostPlacement {
    fn place(&self, _sess: &NewSyncSession, nonce: Nonce) -> Nonce {
        if self == &nonce.placement() {
            nonce
        } else {
            // TODO just updating the placement isn't enough,
            // we need this to eventually turn into Send + Recv
            Nonce(nonce.0, self.clone())
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
    fn kernel(_sess: &NewSyncSession, plc: &HostPlacement) -> PrfKey {
        let raw_key = RawPrfKey(AesRng::generate_random_key());
        PrfKey(raw_key, plc.clone())
    }
}

modelled!(PlacementDeriveSeed::derive_seed, HostPlacement, attributes[sync_key: RawNonce] (PrfKey) -> Seed, PrimDeriveSeedOp);

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
        sync_key: RawNonce,
        key: PrfKey,
    ) -> Seed {
        let sid = sess.session_id();
        let raw_key = key.0;

        let mut nonce: Vec<u8> = vec![];
        nonce.extend(sid.as_bytes());
        nonce.extend(sync_key.0);

        let raw_seed = crate::utils::derive_seed(&raw_key.0, &nonce);
        Seed(RawSeed(raw_seed), plc.clone())
    }
}

// TODO deprecated
impl RawSeed {
    pub fn from_prf(key: &RawPrfKey, nonce: &RawNonce) -> RawSeed {
        let raw_seed = crate::utils::derive_seed(&key.0, &nonce.0);
        RawSeed(raw_seed)
    }
}

// TODO deprecated
impl RawPrfKey {
    pub fn generate() -> RawPrfKey {
        let raw_key = AesRng::generate_random_key();
        RawPrfKey(raw_key)
    }
}
