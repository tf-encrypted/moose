use criterion::{criterion_group, criterion_main, Criterion};
use moose::prng::{AesRng, SEED_SIZE};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn aes_rng(c: &mut Criterion) {
    c.bench_function("aes_rng_fill_bytes", |b| {
        let mut rng = AesRng::from_seed([0u8; SEED_SIZE]);
        let mut output = vec![0u8; 2 * 1024 * 1024];
        b.iter(|| {
            rng.try_fill_bytes(&mut output).unwrap();
        })
    });

    c.bench_function("aes_rng_next_u64", |b| {
        let mut rng = AesRng::from_seed([0u8; SEED_SIZE]);
        let n: u64 = 1000;
        b.iter(|| {
            for _ in 0..n {
                let _ = rng.next_u64();
            }
        })
    });
}

fn chacha_rng(c: &mut Criterion) {
    c.bench_function("chacha_rng_fill_bytes", |b| {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut output = vec![0u8; 2 * 1024 * 1024];
        b.iter(|| {
            rng.try_fill_bytes(&mut output).unwrap();
        })
    });

    c.bench_function("chacha_rng_next_u64", |b| {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n: u64 = 1000;
        b.iter(|| {
            for _ in 0..n {
                let _ = rng.next_u64();
            }
        })
    });
}

fn thread_rng(c: &mut Criterion) {
    c.bench_function("thread_rng_multiple_fill_16B", |b| {
        b.iter(|| {
            let mut output = vec![0u8; 16];
            for _i in 0..512 {
                let mut rng = rand::thread_rng();
                rng.try_fill_bytes(&mut output).unwrap();
            }
        })
    });
    c.bench_function("thread_rng_single_fill_16B", |b| {
        b.iter(|| {
            let mut output = vec![0u8; 16];
            let mut rng = rand::thread_rng();
            rng.try_fill_bytes(&mut output).unwrap();
        })
    });
}

fn getrandom_rng(c: &mut Criterion) {
    c.bench_function("get_random_rng_fill_16B", |b| {
        b.iter(|| {
            let mut output = vec![0u8; 16];
            for _i in 0..512 {
                getrandom::getrandom(&mut output).expect("failed to get randomness");
            }
        })
    });
    c.bench_function("get_random_rng_single_fill_16B", |b| {
        b.iter(|| {
            let mut output = vec![0u8; 16];
            getrandom::getrandom(&mut output).expect("failed to get randomness");
        })
    });
}

criterion_group!(benches, thread_rng, getrandom_rng, chacha_rng, aes_rng);
criterion_main!(benches);
