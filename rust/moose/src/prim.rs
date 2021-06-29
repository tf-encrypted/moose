use crate::computation::HostPlacement;
use crate::prng::AesRng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawSeed(pub [u8; 16]);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Seed(pub RawSeed, pub HostPlacement);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawPrfKey(pub [u8; 16]);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct PrfKey(pub RawPrfKey, pub HostPlacement);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RawNonce(pub Vec<u8>);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Nonce(pub RawNonce, pub HostPlacement);

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
