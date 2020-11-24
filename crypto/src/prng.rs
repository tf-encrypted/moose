use std::fmt;
use aes::cipher::generic_array::{typenum::U8,typenum::U16,GenericArray};
use aes::cipher::{BlockCipher, NewBlockCipher};
use rand::{CryptoRng, Error, RngCore, SeedableRng};
use aes::Aes128;
use byteorder::{ByteOrder, LittleEndian};

const AES_BLK_SIZE: usize = 16;
const PIPELINES_U128: u128 = 8;
const PIPELINES_USIZE: usize = 8;
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
    // this always allocates a new array. we should do this just once.
    fn from_counter(counter: u128) -> Self {
        let mut i: usize = 0;

        let mut state: [u8; 8 * 16] = [0u8; 8 * 16];
        for pipe_index in 0u128..PIPELINES_U128 {
            LittleEndian::write_u128(&mut state[i*AES_BLK_SIZE..(i+1)*AES_BLK_SIZE], counter+pipe_index);
            i += 1;
        }
        AesRngState {
            state_bytes: state,
            counter: counter + PIPELINES_U128,
            used_bytes: 0
        }
    }
    fn from_counter_no_alloc(counter: u128, state: &mut [u8; 8*16]) -> u128 {
        let mut i: usize = 0;
        for pipe_index in 0u128..PIPELINES_U128 {
            LittleEndian::write_u128(&mut state[i*AES_BLK_SIZE..(i+1)*AES_BLK_SIZE], counter+pipe_index);
            i += 1;
        };
        counter + PIPELINES_U128
    }
    fn next(&mut self) {
        let counter = AesRngState::from_counter_no_alloc(self.counter, &mut self.state_bytes);
        self.counter = counter;
        self.used_bytes = 0;
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
        let mut state = self.state.state_bytes;
        let mut state_in_blocks = Block128x8::from_exact_iter(
            (0..PIPELINES_USIZE).map(
                | p | {
                    let sliced_state = &mut state[p * 16..(p+1) * 16];
                    let block = GenericArray::from_mut_slice(sliced_state);
                    *block
                }
            )
        ).unwrap();
        self.state = AesRngState::from_counter(counter);
        self.cipher.encrypt_blocks(&mut state_in_blocks);
    }
}
 
impl RngCore for AesRng {
    fn next_u32(&mut self) -> u32 {
        if self.state.used_bytes >= RAND_SIZE-4 {
            self.getNext();
        }
        let used_bytes = self.state.used_bytes;
        self.state.used_bytes += 4; // update number of used bytes
        LittleEndian::read_u32(&mut self.state.state_bytes[used_bytes..used_bytes+4])
    }

    fn next_u64(&mut self) -> u64 {
        if self.state.used_bytes >= RAND_SIZE-8 {
            self.getNext();
        }
        let used_bytes = self.state.used_bytes;
        self.state.used_bytes += 8; // update number of used bytes
        LittleEndian::read_u64(&mut self.state.state_bytes[used_bytes..used_bytes+8])
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut index = 0;

        let mut read_len = RAND_SIZE - self.state.used_bytes;
        while read_len < dest.len() {
            for i in self.state.used_bytes..RAND_SIZE {
                dest[index] = self.state.state_bytes[i];
                index += 1;
            }
            self.getNext();
            read_len += RAND_SIZE;
        }
        let remaining = dest.len() - index;
        while remaining < dest.len() {
            dest[index] = self.state.state_bytes[index - remaining];
            index += 1;
        }
        self.state.used_bytes += remaining;
    }

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