use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Seed(pub Vec<u8>);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrfKey(pub Vec<u8>);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Nonce(pub Vec<u8>);
