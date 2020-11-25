use aes::cipher::generic_array::{typenum::U16, typenum::U8, GenericArray};
use aes::cipher::{BlockCipher, NewBlockCipher};
use aes::Aes128;
use byteorder::{ByteOrder, LittleEndian};
use rand::{CryptoRng, Error, RngCore, SeedableRng};
use std::mem;

const AES_BLK_SIZE: usize = 16;
const PIPELINES_U128: u128 = 8;
const PIPELINES_USIZE: usize = 8;
const RAND_SIZE: usize = PIPELINES_USIZE * AES_BLK_SIZE;

pub struct AesRngSeed(pub [u8; AES_BLK_SIZE]);
type Block128 = GenericArray<u8, U16>;
type Block128x8 = GenericArray<Block128, U8>;

pub struct AesRngState {
    bytes: [u8; 8 * 16],
    next_index: u128,
    used_bytes: usize,
}

impl Default for AesRngState {
    fn default() -> Self {
        AesRngState::from_counter(0)
    }
}

impl AesRngState {
    // this always allocates a new array. we should do this just once.
    fn write_to_state(counter: u128, state: &mut [u8; 8 * 16]) {
        for i in  0..PIPELINES_USIZE {
            LittleEndian::write_u128(
                &mut state[i * AES_BLK_SIZE..(i + 1) * AES_BLK_SIZE],
                counter + i as u128
            );
        }
    }
 
    fn from_counter(counter: u128) -> Self {
        let mut state: [u8; 8 * 16] = [0u8; 8 * 16];
        AesRngState::write_to_state(counter, &mut state);
        AesRngState {
            bytes: state,
            next_index: counter + PIPELINES_U128,
            used_bytes: 0,
        }
    }

    fn from_counter_no_alloc(counter: u128, state: &mut [u8; 8 * 16]) -> u128 {
       AesRngState::write_to_state(counter, state);
       counter + PIPELINES_U128
    }

    fn next(&mut self) {
        self.next_index = AesRngState::from_counter_no_alloc(self.next_index, &mut self.bytes);
        self.used_bytes = 0;
    }
}

struct AesRng {
    state: AesRngState,
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
        let mut out = AesRng {
            state: AesRngState::default(),
            cipher: Aes128::new(&key),
        };
        out.init();
        out
    }
}

trait Hash {
    fn next(&mut self);
    fn init(&mut self);
}
impl Hash for AesRng {
    fn init(&mut self) {
//        let mut state = self.state.bytes;
        let mut slice_b0: [u8;16] = [0; 16];
        let mut slice_b1: [u8;16] = [0; 16];
        let mut slice_b2: [u8;16] = [0; 16];
        let mut slice_b3: [u8;16] = [0; 16];
        let mut slice_b4: [u8;16] = [0; 16];
        let mut slice_b5: [u8;16] = [0; 16];

        let mut slice_b6: [u8;16] = [0; 16];
        let mut slice_b7: [u8;16] = [0; 16];

        let mut b0 = Block128::from_mut_slice(&mut slice_b0);
        let mut b1 = Block128::from_mut_slice(&mut slice_b1);
        let mut b2 = Block128::from_mut_slice(&mut slice_b2);
        let mut b3 = Block128::from_mut_slice(&mut slice_b3);
        let mut b4 = Block128::from_mut_slice(&mut slice_b4);
        let mut b5 = Block128::from_mut_slice(&mut slice_b5);
        let mut b6 = Block128::from_mut_slice(&mut slice_b6);
        let mut b7 = Block128::from_mut_slice(&mut slice_b7);

        println!("b0 ::: {:?}", b0);
        let mut array: [&Block128; 8] = [b0, b1, b2, b3, b4, b5, b6, b7];

        let mut block8 = Block128x8::from_mut_slice(&mut array[0..8]); // FIXME
        // array[0][0] += 1;

        // self.cipher.encrypt_blocks(&mut block8);
        println!("array[0] ::: {:?}", array[0]);
        println!("slice_b0 ::: {:?}", slice_b0);
    }
    fn next(&mut self) {
        self.state.next();
        let mut state_in_blocks = Block128x8::from_exact_iter((0..PIPELINES_USIZE).map(|p| {
            let sliced_state = &mut self.state.bytes[p * 16..(p + 1) * 16];
            let block = GenericArray::from_mut_slice(sliced_state);
            *block
        }))
        .unwrap();
        self.cipher.encrypt_blocks(&mut state_in_blocks);
    }
}

impl RngCore for AesRng {
    fn next_u32(&mut self) -> u32 {
        let u32_size = mem::size_of::<u32>();
        if self.state.used_bytes >= RAND_SIZE - u32_size {
            self.next();
        }
        let used_bytes = self.state.used_bytes;
        self.state.used_bytes += u32_size; // update number of used bytes
        LittleEndian::read_u32(&self.state.bytes[used_bytes..used_bytes + u32_size])
    }

    fn next_u64(&mut self) -> u64 {
        let u64_size = mem::size_of::<u32>();
        if self.state.used_bytes >= RAND_SIZE - u64_size {
            self.next();
        }
        let used_bytes = self.state.used_bytes;
        self.state.used_bytes += u64_size; // update number of used bytes
        LittleEndian::read_u64(&self.state.bytes[used_bytes..used_bytes + u64_size])
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut index = 0;
        let mut read_len = RAND_SIZE - self.state.used_bytes;

        println!("{:?}", read_len);
        while read_len < dest.len() {
            println!("Hey2");
            for i in self.state.used_bytes..RAND_SIZE {
                dest[index] = self.state.bytes[i];
                index += 1;
            }
            self.next();
            read_len += RAND_SIZE;
        }
        let mut start_index = 0;
        for i in index..dest.len() {
            dest[i] = self.state.bytes[start_index];
            println!("{:?}", dest[i]);
            start_index += 1;
            self.state.used_bytes += 1;
        }
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
        let seed = AesRngSeed([0u8; 16]);
        let mut rng = AesRng::from_seed(seed);
        let mut out = [0u8; 16];
        rng.try_fill_bytes(&mut out);
        println!("{:?}", out);
        assert!(false);
    }
}
