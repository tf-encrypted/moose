use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockCipher, NewBlockCipher};
use rand::{CryptoRng, Error, RngCore, SeedableRng};

const AES_BLK_SIZE: usize = 16;
const PIPELINES: usize = 8;
pub struct AesRngSeed(pub [u8; AES_BLK_SIZE]);
pub struct AesRngState(pub [u8; AES_BLK_SIZE * PIPELINES]);

struct AesRng {
    // state has same type as AesRngSeed
    state: AesRngState,
    seed: AesRngSeed,
}

impl Default for AesRngSeed {
    fn default() -> AesRngSeed {
        AesRngSeed([0; AES_BLK_SIZE])
    }
}

impl Default for AesRngState {
    fn default() -> AesRngState {
        AesRngState([0; AES_BLK_SIZE * PIPELINES])
    }
}

impl AsMut<[u8]> for AesRngSeed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl AsMut<[u8]> for AesRngState {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

fn inc_be(val: GenericArray<u8, U16>) -> () {
}

impl SeedableRng for AesRng {
            type Seed = AesRngSeed;

            #[inline]
            fn from_seed(seed: Self::Seed) -> Self {
                let mut block = GenericArray::clone_from_slice(&[0u8; 16]);
                let mut block8 = GenericArray::clone_from_slice(&[block; 8]);
                // let state = GenericArray::clone_from_slice(&[tmp; 8]);
                // for p in 0..PIPELINES {
                //     inc_be(state[p]);
                // }
                AesRng {
                    state: AesRngState([0;AES_BLK_SIZE*PIPELINES]),
                    seed: seed
                }
            }
    }

trait Next {
    fn getNext(&mut self) -> ();
}
impl Next for AesRng {
    fn getNext(&mut self) -> () {
        
    }
}

impl RngCore for AesRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32// TODO maybe there's a better way
    }

    fn next_u64(&mut self) -> u64 {
        0
        // TODO update state
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {}

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        Ok(self.fill_bytes(dest))
    }
        
}
impl CryptoRng for AesRng {}

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

// class PRNG {
//     state
    
//     init()
//     get_random_bytes(<T>) // output sizeof(T) bytes
// }