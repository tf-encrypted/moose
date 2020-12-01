use criterion::{black_box, criterion_group, criterion_main, Benchmark, Criterion};
use crypto::prng::AesRng;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub fn aes_benchmark(c: &mut Criterion) {
    let bench = Benchmark::new("bench_rng_aes", |b| {
        let mut rng = AesRng::from_seed([0u8; 16]);
        let mut output = vec![0u8; 2 * 1024 * 1024 * 1024];
        b.iter(|| {
            black_box(rng.try_fill_bytes(&mut output).unwrap());
        })
    });
    c.bench("bench_rng_aes", bench);
}
pub fn chacha_benchmark(c: &mut Criterion) {
    let bench = Benchmark::new("bench_rng_chacha", |b| {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut output = vec![0u8; 2 * 1024 * 1024 * 1024];
        b.iter(|| {
            black_box(rng.try_fill_bytes(&mut output).unwrap());
        })
    });
    c.bench("bench_rng_chacha", bench);
}

pub fn aes64_benchmark(c: &mut Criterion) {
    let bench = Benchmark::new("bench_rng_aes64", |b| {
        let mut rng = AesRng::from_seed([0u8; 16]);
        let n: u64 = 1000000000;
        b.iter(|| {
            black_box(for i in 0..n {
                let x = rng.next_u64();
            });
        })
    });
    c.bench("bench_rng_aes64", bench);
}

criterion_group!(benches, aes64_benchmark);
criterion_main!(benches);
