use std::fmt;
use aes::cipher::generic_array::{typenum::U8,typenum::U16,GenericArray};
use aes::cipher::{BlockCipher, NewBlockCipher};
use rand::{CryptoRng, Error, RngCore, SeedableRng};
use aes::Aes128;

const AES_BLK_SIZE: usize = 16;
const PIPELINES: u128 = 8;
const RAND_SIZE: usize = 8 * AES_BLK_SIZE;
pub struct AesRngSeed(pub [u8; AES_BLK_SIZE]);

type Block128 = GenericArray<u8, U16>;
type Block128x8 = GenericArray<GenericArray<u8, U16>, U8>;

pub struct AesRngState {
    state_bytes: [u8; 8 * 16],
    counter: u128,
    used_bytes: usize,
}

impl Default for AesRngState {
    fn default() -> Self {
       AesRngState::from_counter(0)
    }
}

impl AesRngState {
    fn from_counter(counter: u128) -> Self {
        let state_bytes = Block128x8::from_exact_iter(
        (0u128..PIPELINES).map(
            | p | {
                let block: Block128 = GenericArray::clone_from_slice(&(p + counter).to_le_bytes());
                block
            }
        )).unwrap();
        AesRngState {
            state_bytes,
            counter: counter + PIPELINES,
            used_bytes: 0
        }
    }
    fn next(&mut self) {
        let next_state = AesRngState::from_counter(self.counter);
        self.state_bytes = next_state.state_bytes;
        self.counter = next_state.counter;
        self.used_bytes = next_state.used_bytes;
    }
}

struct AesRng {
    // state has same type as AesRngSeed
    state: AesRngState,
    seed: AesRngSeed,
    cipher: Aes128,
}

impl Default for AesRngSeed {
    fn default() -> AesRngSeed {
        AesRngSeed([0; AES_BLK_SIZE])
    }
}

impl AsMut<[u8]> for AesRngSeed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl SeedableRng for AesRng {
    type Seed = AesRngSeed;

    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        // AES_{seed}(ctr), convert ctr to an AES input
        // 8 blocks of 128 bits, each block is divided
        // arrays [ [0, 0...., 0], [0, 0, .., 1], [0,...,0010], ... [0,...0111]]
        // TODO: Can we replace this with copy from slice?
        let key: Block128 = GenericArray::clone_from_slice(&(seed.0));
        AesRng {
            state: AesRngState::default(),
            seed: seed,
            cipher: Aes128::new(&key)
        }
    }
}

trait Next {
    fn getNext(&mut self) -> ();
}
impl Next for AesRng {
    fn getNext(&mut self) -> () {
        // can we do something like self.state = AesRngState::from_counter()?
        let counter = self.state.counter;
        self.state = AesRngState::from_counter(counter);
        self.cipher.encrypt_blocks(&mut self.state.state_bytes);
    }
}
 
trait Slicer {
    fn getSlice32(self, start: isize, end: isize) -> u32;
    fn getSlice64(self, start: isize, end: isize) -> u64;
}
impl Slicer for AesRng {
   fn getSlice64(self, start:isize, end: isize) -> u64 {

        let byte_chunk = [self.state.state_bytes[get_row(i)][get_col(i) 
            for i in range(start, end)]
        
        let array: [u8; 8] = 
        u64::from_le_bytes(byte_chunk);
    }
}

impl RngCore for AesRng {
    fn next_u32(&mut self) -> u32 {
        let mut state = self.state;
        if state.used_bytes >= RAND_SIZE-4 {
            self.getNext();
        }
        let counter = state.counter;
        let sliced_state = &state.state_bytes[..counter+4];
        let output = u32::from_le_bytes()
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
        let seed = AesRngSeed([0u8; 16]);
        let rng = AesRng::from_seed(seed);
        assert!(false);
    }
}