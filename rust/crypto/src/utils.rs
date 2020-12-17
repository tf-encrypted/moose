use sha3::{Digest, Sha3_256};

// TODO(Dragos) replace the constant 16 with a seed_size
pub fn derive_seed(key: &[u8], counter: u128) -> [u8;16] {
    // create a SHA3-256 object
    let mut hasher = Sha3_256::new();

    // write input message
    hasher.update(key);
    hasher.update(counter.to_le_bytes());

    // read hash digest and truncate to the first 16 bytes
    let mut output = [0u8; 16];
    output.copy_from_slice(&hasher.finalize()[..16]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_seed() {
        let key = [0u8; 16];
        let seed = derive_seed(&key, 0);
        assert_eq!(seed.len(), 16);
    }
}