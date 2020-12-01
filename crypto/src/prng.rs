use aes::cipher::generic_array::{typenum::U16, typenum::U8, GenericArray};
use aes::cipher::{BlockCipher, NewBlockCipher};
use aes::Aes128;
use byteorder::{ByteOrder, LittleEndian};
use rand::{CryptoRng, Error, RngCore, SeedableRng};
use std::mem;
use std::slice;

const AES_BLK_SIZE: usize = 16;
const PIPELINES_U128: u128 = 8;
const PIPELINES_USIZE: usize = 8;
const STATE_SIZE: usize = PIPELINES_USIZE * AES_BLK_SIZE;
pub const SEED_SIZE: usize = AES_BLK_SIZE;

type Block128 = GenericArray<u8, U16>;
type Block128x8 = GenericArray<Block128, U8>;

pub struct AesRngState {
    blocks: Block128x8,
    next_index: u128,
    used_bytes: usize,
}

impl Default for AesRngState {
    fn default() -> Self {
        AesRngState::init()
    }
}

impl AesRngState {
    fn as_mut_bytes(&mut self) -> &mut [u8] {
        #[allow(unsafe_code)]
        unsafe {
            slice::from_raw_parts_mut(&mut self.blocks as *mut Block128x8 as *mut u8, STATE_SIZE)
        }
    }

    fn init() -> Self {
        // TODO: This has to be done in manner similar to as_mut_bytes
        let mut state = [0_u8; STATE_SIZE];
        let blocks = Block128x8::from_exact_iter((0..PIPELINES_USIZE).map(|i| {
            LittleEndian::write_u128(
                &mut state[i * AES_BLK_SIZE..(i + 1) * AES_BLK_SIZE],
                i as u128,
            );
            let sliced_state = &mut state[i * AES_BLK_SIZE..(i + 1) * AES_BLK_SIZE];
            let block = GenericArray::from_mut_slice(sliced_state);
            *block
        }))
        .unwrap();

        AesRngState {
            blocks,
            next_index: PIPELINES_U128,
            used_bytes: 0,
        }
    }

    fn next(&mut self) {
        let counter = self.next_index;
        let blocks_bytes = self.as_mut_bytes();
        for i in 0..PIPELINES_USIZE {
            LittleEndian::write_u128(
                &mut blocks_bytes[i * AES_BLK_SIZE..(i + 1) * AES_BLK_SIZE],
                counter + i as u128,
            );
        }
        self.next_index += PIPELINES_U128;
        self.used_bytes = 0;
    }
}

pub struct AesRng {
    state: AesRngState,
    cipher: Aes128,
}

impl SeedableRng for AesRng {
    type Seed = [u8; SEED_SIZE];

    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        // AES_{seed}(ctr), convert ctr to an AES input
        // 8 blocks of 128 bits, each block is divided
        // arrays [ [0, 0...., 0], [0, 0, .., 1], [0,...,0010], ... [0,...0111]]
        // TODO: Can we replace this with copy from slice?
        let key: Block128 = GenericArray::clone_from_slice(&seed);
        let mut out = AesRng {
            state: AesRngState::default(),
            cipher: Aes128::new(&key),
        };
        out.init();
        out
    }
}

impl AesRng {
    fn init(&mut self) {
        self.cipher.encrypt_blocks(&mut self.state.blocks);
    }

    fn next(&mut self) {
        self.state.next();
        self.cipher.encrypt_blocks(&mut self.state.blocks);
    }
}

impl RngCore for AesRng {
    fn next_u32(&mut self) -> u32 {
        let u32_size = mem::size_of::<u32>();
        if self.state.used_bytes >= STATE_SIZE - u32_size {
            self.next();
        }
        let used_bytes = self.state.used_bytes;
        self.state.used_bytes += u32_size; // update number of used bytes
        let blocks_bytes = self.state.as_mut_bytes();
        LittleEndian::read_u32(&blocks_bytes[used_bytes..used_bytes + u32_size])
    }

    fn next_u64(&mut self) -> u64 {
        let u64_size = mem::size_of::<u64>();
        if self.state.used_bytes >= STATE_SIZE - u64_size {
            self.next();
        }
        let used_bytes = self.state.used_bytes;
        self.state.used_bytes += u64_size; // update number of used bytes
        LittleEndian::read_u64(&self.state.as_mut_bytes()[used_bytes..used_bytes + u64_size])
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut read_len = STATE_SIZE - self.state.used_bytes;
        let mut dest_start = 0;

        while read_len < dest.len() {
            let src_start = self.state.used_bytes;
            dest[dest_start..read_len]
                .copy_from_slice(&self.state.as_mut_bytes()[src_start..STATE_SIZE]);
            self.next();
            dest_start = read_len;
            read_len += STATE_SIZE;
        }

        let src_start = self.state.used_bytes;
        let remainder = dest.len() - dest_start;
        let dest_len = dest.len();

        dest[dest_start..dest_len]
            .copy_from_slice(&self.state.as_mut_bytes()[src_start..src_start + remainder]);
        self.state.used_bytes += remainder;
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for AesRng {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut rng = AesRng::from_seed([0u8; SEED_SIZE]);
        let mut out = [0u8; 16 * 8 * 2 + 1];
        rng.try_fill_bytes(&mut out).expect("");
        println!("out: {:?}", out);

        assert!(false);
    }
}
