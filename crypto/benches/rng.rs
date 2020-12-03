use criterion::{black_box, criterion_group, criterion_main, Criterion};
use crypto::prng::{AesRng, SEED_SIZE};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn aes_rng(c: &mut Criterion) {
    c.bench_function("aes_rng_fill_bytes", |b| {
        let mut rng = AesRng::from_seed([0u8; SEED_SIZE]);
        let mut output = vec![0u8; 2 * 1024 * 1024];
        b.iter(|| {
            black_box(rng.try_fill_bytes(&mut output).unwrap());
        })
    });

    c.bench_function("aes_rng_next_u64", |b| {
        let mut rng = AesRng::from_seed([0u8; SEED_SIZE]);
        let n: u64 = 1000;
        b.iter(|| {
            black_box(for _ in 0..n {
                let _ = rng.next_u64();
            });
        })
    });
}

fn chacha_rng(c: &mut Criterion) {
    c.bench_function("chacha_rng_fill_bytes", |b| {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut output = vec![0u8; 2 * 1024 * 1024];
        b.iter(|| {
            black_box(rng.try_fill_bytes(&mut output).unwrap());
        })
    });

    c.bench_function("chacha_rng_next_u64", |b| {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n: u64 = 1000;
        b.iter(|| {
            black_box(for _ in 0..n {
                let _ = rng.next_u64();
            });
        })
    });
}

criterion_group!(benches, chacha_rng, aes_rng);
criterion_main!(benches);
