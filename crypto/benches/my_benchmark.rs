use criterion::{black_box, criterion_group, criterion_main, Benchmark, Criterion};
use crypto::prng::{AesRng, AesRngSeed};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub fn criterion_benchmark(c: &mut Criterion) {

    let bench = Benchmark::new("my_bench", |b| {
        let seed = AesRngSeed([0u8; 16]);
        let mut rng = AesRng::from_seed(seed);
        let mut output = vec![0u8; 2 * 1024 * 1024 * 10];
        b.iter(|| {
            black_box(rng.try_fill_bytes(&mut output).unwrap());
        })
    });
    c.bench("my_bench", bench);
}

pub fn criterion_benchmark_chacha(c: &mut Criterion) {

    let bench = Benchmark::new("my_bench_chacha", |b| {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut output = vec![0u8; 2 * 1024 * 1024 * 10];
        b.iter(|| {
            black_box(rng.try_fill_bytes(&mut output).unwrap());
        })
    });
    c.bench("my_bench_chacha", bench);
}



criterion_group!(benches, criterion_benchmark, criterion_benchmark_chacha);
criterion_main!(benches);
