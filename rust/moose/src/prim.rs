use crate::computation::{HostPlacement, Placed, PrimPrfKeyGenOp};
use crate::kernels::{ConcreteContext, NullaryKernel, PlacementKeyGen};
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

modelled!(PlacementKeyGen::keygen, HostPlacement, () -> PrfKey, PrimPrfKeyGenOp);


kernel! {
    PrimPrfKeyGenOp,
    [
        (HostPlacement, () -> PrfKey => Self::kernel),
    ]
}

impl PrimPrfKeyGenOp {
    fn kernel(_ctx: &ConcreteContext, plc: &HostPlacement) -> PrfKey {
        let raw_key = RawPrfKey(AesRng::generate_random_key());
        PrfKey(raw_key, plc.clone())
    }
}

impl RawSeed {
    pub fn from_prf(key: &RawPrfKey, nonce: &RawNonce) -> RawSeed {
        let raw_seed = crate::utils::derive_seed(&key.0, &nonce.0);
        RawSeed(raw_seed)
    }
}

impl RawPrfKey {
    pub fn generate() -> RawPrfKey {
        let raw_key = AesRng::generate_random_key();
        RawPrfKey(raw_key)
    }
}
