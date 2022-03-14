//! Pure Rust implementation of a PRNG based on Advanced Encryption Standard
//!
//! The underlying implementation is based on the pipelined version of AES from
//! [`aes`] create.
//! The PRNG supports two version can take a seed either given by the user
//! or taken from /dev/random or dev/urandom using [`sodiumoxide`].
//!
//! By default the package is compiled with AES-NI implementation
//! for `i686`/`x86_64` target architectures with `target-feature=+aes`.
//!
//! The underlying algorithms are inspired from [MP-SPDZ] and [SCALE-MAMBA]
//! implementations which generate randomness in batches of 8 * 16 bytes
//! i.e. select a random key k and compute AES_k(0), ..., AES_k(7) giving out
//! 128 bytes of randomness as long as the key is random since AES acts as a [PRF].
//! At the next iteration AES_k(8), ..., AES_k(15) is computed and so on.
//!
//! # Usage example for already seeded PRNG
//! ```
//! use rand::{RngCore, SeedableRng};
//! use moose::prng::{AesRng};
//!
//! // initialize PRNG seed using sodiumoxide entropy pool
//! let mut rng: AesRng = AesRng::from_random_seed();
//! // gets 32 random bits
//! let output32 = rng.next_u32();
//! // gets 64 random bits
//! let output64 = rng.next_u64();
//! // gets 2 output bytes
//! let mut out = [0u8; 2];
//! rng.try_fill_bytes(&mut out).expect("");
//!
//! ```
//!
//! # Usage example for setting manually the PRNG seed
//! ```
//! use moose::prng::{AesRng, SEED_SIZE};
//! use rand::{RngCore, SeedableRng};
//! // seed is full of zeroes here, note this would be insecure
//! // in an actual implementation
//! let mut rng = AesRng::from_seed([0u8; SEED_SIZE]);
//!
//! // gets 32 random bits
//! let output32 = rng.next_u32();
//! // gets 64 random bits
//! let output64 = rng.next_u64();
//! ```
//!
//! For implementations of block cipher modes of operation see
//! [`block-modes`] crate.
//!
//! [fixslicing]: https://eprint.iacr.org/2020/1123.pdf
//! [AES-NI]: https://en.wikipedia.org/wiki/AES_instruction_set
//! [`block-modes`]: https://docs.rs/block-modes
//!
//! [`aes`]: aes
//! [`sodiumoxide`]: sodiumoxide
//! [MP-SPDZ]: https://github.com/data61/MP-SPDZ
//! [SCALE-MAMBA]: https://github.com/KULeuven-COSIC/SCALE-MAMBA
//! [PRF]: https://en.wikipedia.org/wiki/Pseudorandom_function_family
//!

use aes::cipher::generic_array::{typenum::U16, typenum::U8, GenericArray};
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use byteorder::{ByteOrder, LittleEndian};
use rand::{CryptoRng, Error, RngCore, SeedableRng};
use sodiumoxide::randombytes::randombytes_into;
use std::mem;
use std::slice;

const AES_BLK_SIZE: usize = 16;
const PIPELINES_U128: u128 = 8;
const PIPELINES_USIZE: usize = 8;
const STATE_SIZE: usize = PIPELINES_USIZE * AES_BLK_SIZE;
pub const SEED_SIZE: usize = AES_BLK_SIZE;
pub type RngSeed = [u8; SEED_SIZE];

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

/// Dumps 0, 1, ... PIPELINES_SIZE-1 into Block128 object
/// then unifies it into a Block128x8
/// this could probably be done faster in a similar manner to as_mut_bytes
fn create_init_state() -> Block128x8 {
    let mut state = [0_u8; STATE_SIZE];
    Block128x8::from_exact_iter((0..PIPELINES_USIZE).map(|i| {
        LittleEndian::write_u128(
            &mut state[i * AES_BLK_SIZE..(i + 1) * AES_BLK_SIZE],
            i as u128,
        );
        let sliced_state = &mut state[i * AES_BLK_SIZE..(i + 1) * AES_BLK_SIZE];
        let block = GenericArray::from_mut_slice(sliced_state);
        *block
    }))
    .unwrap()
}

impl AesRngState {
    /// Unsafe method which converts from a Block128x8 to a mutable u8 array
    fn as_mut_bytes(&mut self) -> &mut [u8] {
        #[allow(unsafe_code)]
        unsafe {
            slice::from_raw_parts_mut(&mut self.blocks as *mut Block128x8 as *mut u8, STATE_SIZE)
        }
    }

    /// Initialize state
    fn init() -> Self {
        AesRngState {
            blocks: create_init_state(),
            next_index: PIPELINES_U128,
            used_bytes: 0,
        }
    }

    /// Computes the next state by looking at the counter of the current state
    /// Writes the next counters into bytes over the u8 state array
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

/// Necessary data to compute randomness, a state and an initialized AES blockcipher.
pub struct AesRng {
    state: AesRngState,
    cipher: Aes128,
    n_cached_bits: usize,
    cached_bits: u64,
}

impl SeedableRng for AesRng {
    type Seed = RngSeed;
    /// Ideally this should be passed as a reference as we want
    /// to avoid copying the seed around. However this is probably going
    /// to be used few times, by default we should go with AesRng::from_random_seed
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        let key = GenericArray::clone_from_slice(&seed);
        let mut out = AesRng {
            state: AesRngState::default(),
            cipher: Aes128::new(&key),
            n_cached_bits: 0,
            cached_bits: 0,
        };
        out.init();
        out
    }
}

impl AesRng {
    /// useful for encrypting the initial state and obtain
    /// initial random byte output
    fn init(&mut self) {
        self.cipher.encrypt_blocks(&mut self.state.blocks);
    }

    /// To compute the next chunk of random bytes
    /// First we compute the next state according to its counter
    /// and then we encrypt it using AES.
    fn next(&mut self) {
        self.state.next();
        self.cipher.encrypt_blocks(&mut self.state.blocks);
    }

    pub fn generate_random_key() -> [u8; SEED_SIZE] {
        let mut seed = [0u8; SEED_SIZE];
        sodiumoxide::init().expect("failed to initialize sodiumoxide");
        randombytes_into(&mut seed);
        seed
    }

    /// Method to fetch a PRNG where its seed is taken from /dev/random
    /// or /dev/urandom if /dev/random doesn't have enough entropy
    /// The entropy selection is done automatically by sodiumoxide
    pub fn from_random_seed() -> Self {
        let seed = AesRng::generate_random_key();
        Self::from_seed(seed)
    }

    // Fetch a random bit from the cache
    pub fn get_bit(&mut self) -> u8 {
        if self.n_cached_bits == 0 {
            self.cached_bits = self.next_u64();
            self.n_cached_bits = 64;
        }
        self.n_cached_bits -= 1;
        let result: u8 = (self.cached_bits & 1) as u8;
        self.cached_bits >>= 1;
        result
    }
}

impl RngCore for AesRng {
    /// fetches 32 bits of randomness
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

    /// fetches 64 bits of randomness
    fn next_u64(&mut self) -> u64 {
        let u64_size = mem::size_of::<u64>();
        if self.state.used_bytes >= STATE_SIZE - u64_size {
            self.next();
        }
        let used_bytes = self.state.used_bytes;
        self.state.used_bytes += u64_size; // update number of used bytes
        LittleEndian::read_u64(&self.state.as_mut_bytes()[used_bytes..used_bytes + u64_size])
    }

    /// Fills in an array of bytes with randomness
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
    fn test_prng() {
        let seed = [0u8; SEED_SIZE];
        let key: Block128 = GenericArray::clone_from_slice(&seed);
        let cipher = Aes128::new(&key);

        let block0 =
            GenericArray::clone_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let block1 =
            GenericArray::clone_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let block2 =
            GenericArray::clone_from_slice(&[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let block3 =
            GenericArray::clone_from_slice(&[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let block4 =
            GenericArray::clone_from_slice(&[4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let block5 =
            GenericArray::clone_from_slice(&[5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let block6 =
            GenericArray::clone_from_slice(&[6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let block7 =
            GenericArray::clone_from_slice(&[7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        let mut blocks = Block128x8::clone_from_slice(&[
            block0, block1, block2, block3, block4, block5, block6, block7,
        ]);

        // create encryptions Enc_{seed}(0)...Enc_{seed}(7)
        cipher.encrypt_blocks(&mut blocks);

        let mut rng = AesRng::from_seed(seed);
        let mut out = [0u8; 16 * 8];
        rng.try_fill_bytes(&mut out).expect("");

        // counter works well
        assert_eq!(rng.state.used_bytes, 16 * 8);

        // encryptions produced initially match aes output
        assert_eq!(rng.state.blocks, blocks);

        let _ = rng.next_u32();
        // check used_bytes increments properly
        // after obtaining a fresh state
        assert_eq!(rng.state.used_bytes, 4);
    }

    #[test]
    fn test_seeded_prng() {
        let mut rng: AesRng = AesRng::from_random_seed();
        // test whether two consecutive calls can be done
        let _ = rng.next_u32();
        let _ = rng.next_u64();
    }
}
