use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockCipher, NewBlockCipher};
use aes::Aes128;


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {

        let key = GenericArray::from_slice(&[0u8; 16]);
        let mut block = GenericArray::clone_from_slice(&[0u8; 16]);
        let mut block8 = GenericArray::clone_from_slice(&[block; 8]);
        // Initialize cipher
        let cipher = Aes128::new(&key);

        let block_copy = block.clone();
        // Encrypt block in-place
        cipher.encrypt_block(&mut block);
        // And decrypt it back
        cipher.decrypt_block(&mut block);
        assert_eq!(block, block_copy);

        // We can encrypt 8 blocks simultaneously using
        // instruction-level parallelism
        let block8_copy = block8.clone();
        cipher.encrypt_blocks(&mut block8);
        cipher.decrypt_blocks(&mut block8);
        assert_eq!(block8, block8_copy);
    }
}

class PRNG {
    state
    
    init()
    get_random_bytes(<T>) // output sizeof(T) bytes
}