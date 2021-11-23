use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::{collections::HashMap, sync::Arc};

fn networking_tall(c: &mut Criterion) {
    let mut group = c.benchmark_group("networking_tall");
    for size in [10, 100, 10_000, 100_000, 250_000, 500_000].iter() {
        let runtime = tokio::runtime::Runtime::new().unwrap();

        group.bench_function(BenchmarkId::new("oneshot", size), |b| {
            b.to_async(&runtime).iter(|| async {
                let (init_tx, mut cur_rx) = tokio::sync::oneshot::channel();

                let mut tasks = Vec::with_capacity(*size + 1);

                for _ in 0..*size {
                    let (new_tx, new_rx) = tokio::sync::oneshot::channel();

                    tasks.push(tokio::spawn(async move {
                        let value: u64 = cur_rx.await.expect("failed to receive value");
                        new_tx.send(value).expect("failed to pass on value");
                    }));

                    cur_rx = new_rx;
                }

                tasks.push(tokio::spawn(async move {
                    let res = cur_rx.await.unwrap();
                    black_box(res);
                }));

                init_tx.send(5).expect("failed to do initial send");

                for task in tasks {
                    task.await.unwrap()
                }
            });
        });

        group.bench_function(BenchmarkId::new("asynccell", size), |b| {
            b.to_async(&runtime).iter(|| async {
                let init_tx = async_cell::sync::AsyncCell::shared();
                let mut cur_rx = init_tx.clone();

                let mut tasks = Vec::with_capacity(*size + 1);

                for _ in 0..*size {
                    let new_tx = async_cell::sync::AsyncCell::shared();
                    let new_rx = new_tx.clone();

                    tasks.push(tokio::spawn(async move {
                        let value: u64 = cur_rx.get().await;
                        new_tx.set(value);
                    }));

                    cur_rx = new_rx;
                }

                tasks.push(tokio::spawn(async move {
                    let res = cur_rx.get().await;
                    black_box(res);
                }));

                init_tx.set(5);

                for task in tasks {
                    task.await.unwrap()
                }
            });
        });

        group.bench_function(
            BenchmarkId::new("hashmap_tokiorwlock_asynccell", size),
            |b| {
                b.to_async(&runtime).iter(|| async {
                    let store = Arc::<
                        tokio::sync::RwLock<HashMap<usize, Arc<async_cell::sync::AsyncCell<u64>>>>,
                    >::default();

                    let mut tasks = Vec::with_capacity(*size);

                    for i in 0..*size {
                        let store = Arc::clone(&store);
                        tasks.push(tokio::spawn(async move {
                            let value = {
                                let cell = {
                                    let mut store = store.write().await;
                                    store
                                        .entry(i)
                                        .or_insert_with(async_cell::sync::AsyncCell::shared)
                                        .clone()
                                };

                                cell.get().await
                            };

                            {
                                let cell = {
                                    let mut store = store.write().await;
                                    store
                                        .entry(i + 1)
                                        .or_insert_with(async_cell::sync::AsyncCell::shared)
                                        .clone()
                                };

                                cell.set(value);
                            }
                        }));
                    }

                    let cell = {
                        let mut store = store.write().await;
                        store
                            .entry(0)
                            .or_insert_with(async_cell::sync::AsyncCell::shared)
                            .clone()
                    };
                    cell.set(5);

                    for task in tasks {
                        task.await.unwrap()
                    }
                });
            },
        );

        group.bench_function(BenchmarkId::new("hashmap_stdrwlock_asynccell", size), |b| {
            b.to_async(&runtime).iter(|| async {
                let store = Arc::<
                    std::sync::RwLock<HashMap<usize, Arc<async_cell::sync::AsyncCell<u64>>>>,
                >::default();

                let mut tasks = Vec::with_capacity(*size);

                for i in 0..*size {
                    let store = Arc::clone(&store);
                    tasks.push(tokio::spawn(async move {
                        let value = {
                            let cell = {
                                let mut store = store.write().unwrap();
                                store
                                    .entry(i)
                                    .or_insert_with(async_cell::sync::AsyncCell::shared)
                                    .clone()
                            };

                            cell.get().await
                        };

                        {
                            let cell = {
                                let mut store = store.write().unwrap();
                                store
                                    .entry(i + 1)
                                    .or_insert_with(async_cell::sync::AsyncCell::shared)
                                    .clone()
                            };

                            cell.set(value);
                        }
                    }));
                }

                let cell = {
                    let mut store = store.write().unwrap();
                    store
                        .entry(0)
                        .or_insert_with(async_cell::sync::AsyncCell::shared)
                        .clone()
                };
                cell.set(5);

                for task in tasks {
                    task.await.unwrap()
                }
            });
        });

        group.bench_function(
            BenchmarkId::new("hashmap_parkingrwlock_asynccell", size),
            |b| {
                b.to_async(&runtime).iter(|| async {
                    let store = Arc::<
                        parking_lot::RwLock<HashMap<usize, Arc<async_cell::sync::AsyncCell<u64>>>>,
                    >::default();

                    let mut tasks = Vec::with_capacity(*size);

                    for i in 0..*size {
                        let store = Arc::clone(&store);
                        tasks.push(tokio::spawn(async move {
                            let value = {
                                let cell = {
                                    let mut store = store.write();
                                    store
                                        .entry(i)
                                        .or_insert_with(async_cell::sync::AsyncCell::shared)
                                        .clone()
                                };

                                cell.get().await
                            };

                            {
                                let cell = {
                                    let mut store = store.write();
                                    store
                                        .entry(i + 1)
                                        .or_insert_with(async_cell::sync::AsyncCell::shared)
                                        .clone()
                                };

                                cell.set(value);
                            }
                        }));
                    }

                    let cell = {
                        let mut store = store.write();
                        store
                            .entry(0)
                            .or_insert_with(async_cell::sync::AsyncCell::shared)
                            .clone()
                    };
                    cell.set(5);

                    for task in tasks {
                        task.await.unwrap()
                    }
                });
            },
        );

        group.bench_function(BenchmarkId::new("dashmap_asynccell", size), |b| {
            b.to_async(&runtime).iter(|| async {
                let store =
                    Arc::<dashmap::DashMap<usize, Arc<async_cell::sync::AsyncCell<u64>>>>::default(
                    );

                let mut tasks = Vec::with_capacity(*size);

                for i in 0..*size {
                    let store = Arc::clone(&store);
                    tasks.push(tokio::spawn(async move {
                        let value = {
                            let cell = store
                                .entry(i)
                                .or_insert_with(async_cell::sync::AsyncCell::shared)
                                .value()
                                .clone();

                            cell.get().await
                        };

                        {
                            let cell = store
                                .entry(i + 1)
                                .or_insert_with(async_cell::sync::AsyncCell::shared)
                                .value()
                                .clone();

                            cell.set(value);
                        }
                    }));
                }

                let cell = store
                    .entry(0)
                    .or_insert_with(async_cell::sync::AsyncCell::shared)
                    .value()
                    .clone();
                cell.set(5);

                for task in tasks {
                    task.await.unwrap()
                }
            });
        });
    }
}

fn networking_wide(c: &mut Criterion) {
    let mut group = c.benchmark_group("networking_wide");
    for size in [10_000, 100_000 /*, 250_000, 500_000*/].iter() {
        let runtime = tokio::runtime::Runtime::new().unwrap();

        group.bench_function(BenchmarkId::new("oneshot", size), |b| {
            b.to_async(&runtime).iter(|| async {
                let (init_tx, _rx) = tokio::sync::broadcast::channel(1);

                let mut tasks = Vec::with_capacity(*size * 2);

                for _ in 0..*size {
                    let mut init_rx = init_tx.subscribe();

                    let (tx, rx) = tokio::sync::oneshot::channel();

                    tasks.push(tokio::spawn(async move {
                        let value: u64 = init_rx.recv().await.unwrap();
                        tx.send(value).unwrap();
                    }));

                    tasks.push(tokio::spawn(async move {
                        let value = rx.await.unwrap();
                        black_box(value);
                    }));
                }

                init_tx.send(5).unwrap();

                for task in tasks {
                    task.await.unwrap()
                }
            });
        });

        group.bench_function(BenchmarkId::new("async_cell", size), |b| {
            b.to_async(&runtime).iter(|| async {
                let (init_tx, _rx) = tokio::sync::broadcast::channel(1);

                let mut tasks = Vec::with_capacity(*size * 2);

                for _ in 0..*size {
                    let mut init_rx = init_tx.subscribe();

                    let tx = async_cell::sync::AsyncCell::shared();
                    let rx = tx.clone();

                    tasks.push(tokio::spawn(async move {
                        let value: u64 = init_rx.recv().await.unwrap();
                        tx.set(value);
                    }));

                    tasks.push(tokio::spawn(async move {
                        let value = rx.get().await;
                        black_box(value);
                    }));
                }

                init_tx.send(5).unwrap();

                for task in tasks {
                    task.await.unwrap()
                }
            });
        });
    }
}

criterion_group!(networking, networking_tall, networking_wide);

criterion_main!(networking);
