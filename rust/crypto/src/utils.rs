use sodiumoxide::crypto::generichash;

// TODO(Dragos) replace the constant 16 with a seed_size
pub fn derive_seed(key: &[u8], nonce: &[u8]) -> [u8; 16] {
    let _ = sodiumoxide::init();
    let mut hasher = generichash::State::new(16, Some(&key)).unwrap();
    hasher.update(&nonce).unwrap();
    let h = hasher.finalize().unwrap();

    let mut output = [0u8; 16];
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
        assert_eq!(seed.len(), 16);
    }
}
