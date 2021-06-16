use crate::prng::AesRng;
use serde::{Deserialize, Serialize};
use crate::computation::HostPlacement;

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Seed(pub [u8; 16]);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct PrfKey(pub [u8; 16], pub HostPlacement);

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Nonce(pub Vec<u8>);

impl Seed {
    pub fn from_prf(key: &PrfKey, nonce: &Nonce) -> Seed {
        let raw_seed = crate::utils::derive_seed(&key.0, &nonce.0);
        Seed(raw_seed)
    }
}

impl PrfKey {
    pub fn generate(plc: &HostPlacement) -> PrfKey {
        let raw_key = AesRng::generate_random_key();
        PrfKey(raw_key, plc.clone())
    }
}
