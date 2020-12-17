use sodiumoxide::crypto::generichash;

use crate::prng::{PRNGSeed, SEED_SIZE};

pub fn derive_seed(key: &[u8], nonce: &[u8]) -> PRNGSeed {
    let _ = sodiumoxide::init();
    let mut hasher = generichash::State::new(SEED_SIZE, Some(key)).unwrap();
    hasher.update(nonce).unwrap();
    let h = hasher.finalize().unwrap();

    let mut output: PRNGSeed = [0u8; SEED_SIZE];
    output.copy_from_slice(h.as_ref());
    output
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
