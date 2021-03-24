use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Seed(pub [u8; 16]);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrfKey(pub Vec<u8>);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Nonce(pub Vec<u8>);

impl Seed {
    pub fn from_prf(key: &PrfKey, nonce: &Nonce) -> Seed {
        Seed(crate::utils::derive_seed(&key.0, &nonce.0).into())
    }
}
