use sodiumoxide::crypto::generichash;

use crate::prng::{RngSeed, SEED_SIZE};

pub fn derive_seed(key: &[u8], nonce: &[u8]) -> RngSeed {
    match sodiumoxide::init() {
        Ok(()) => {
            let mut hasher = generichash::State::new(Some(SEED_SIZE), Some(key)).unwrap();
            hasher.update(nonce).unwrap();
            let h = hasher.finalize().unwrap();

            let mut output: RngSeed = [0u8; SEED_SIZE];
            output.copy_from_slice(h.as_ref());
            output
        }
        Err(()) => {
            // TODO: should this function return a Result<RngSeed> ?
            panic!("failed to initialize sodiumoxide");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_seed() {
        let key = [0u8; 16];
        let nonce = [0u8; 16];
        let seed = derive_seed(&key, &nonce);
        assert_eq!(seed.len(), SEED_SIZE);
    }
}
