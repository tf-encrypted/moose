use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moose::execution::Computation;
use rayon::prelude::*;
use std::collections::HashMap;

/// Benchmark iter vs par_iter for channel creation
/// Conclusion is that this never seems worth it.
fn par_channel(c: &mut Criterion) {
    c.bench_function("par_channel/rayon", |b| {
        b.iter(|| {
            let channels: Vec<_> = (0..250_000)
                .into_par_iter()
                .map(|_| tokio::sync::oneshot::channel::<u64>())
                .collect();
            black_box(channels);
        })
    });

    c.bench_function("par_channel/seq", |b| {
        use tokio::sync::oneshot::{Receiver, Sender};
        b.iter(|| {
            let channels: Vec<(Sender<_>, Receiver<_>)> = (0..250_000)
                .map(|_| tokio::sync::oneshot::channel::<u64>())
                .collect();
            black_box(channels);
        })
    });
}

/// Benchmark iter vs par_iter for spawning tasks.
/// Conclusion is that this never seems worth it.
fn par_spawn(c: &mut Criterion) {
    c.bench_function("par_spawn/rayon", |b| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        b.iter(|| {
            let channels: Vec<_> = (0..250_000)
                .into_par_iter()
                .map(|_| rt.spawn(async move { black_box(5) }))
                .collect();

            black_box(channels);
        })
    });

    c.bench_function("par_spawn/seq", |b| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        b.iter(|| {
            let channels: Vec<_> = (0..250_000)
                .map(|_| rt.spawn(async move { black_box(5) }))
                .collect();

            black_box(channels);
        })
    });
}

/// Benchmark iter vs par_iter for compiling operations.
/// Conclusion is that
fn par_compile(c: &mut Criterion) {
    use moose::execution::*;

    let operator = Operator::RingAdd(RingAddOp {
        lhs: Ty::Ring64TensorTy,
        rhs: Ty::Ring64TensorTy,
    });
    let operation = Operation {
        name: "y".into(),
        kind: operator,
        inputs: vec!["x".into(), "x".into()],
        placement: Placement::Host,
    };

    let mut group = c.benchmark_group("par_compile");
    for size in [10_000, 100_000, 250_000, 500_000, 1_000_000].iter() {
        group.bench_function(BenchmarkId::new("rayon", size), |b| {
            b.iter(|| {
                let compiled: Vec<_> = (0..*size)
                    .into_par_iter()
                    .map(|_| Compile::<CompiledAsyncOperation>::compile(&operation))
                    .collect();
                black_box(compiled);
            })
        });

        group.bench_function(BenchmarkId::new("seq", size), |b| {
            b.iter(|| {
                let compiled: Vec<_> = (0..*size)
                    .map(|_| Compile::<CompiledAsyncOperation>::compile(&operation))
                    .collect();
                black_box(compiled);
            })
        });
    }
}

criterion_group!(par, par_channel, par_spawn, par_compile);

/// This bench was used to figure out how to efficiently pass an Arc
/// that the recipient may or may not make use of.
fn prim_arc(c: &mut Criterion) {
    use std::sync::Arc;

    struct Session(u64);

    c.bench_function("prim_arc_direct", |b| {
        fn foo(s: &Session) -> u64 {
            3
        }

        let s = Session(5);
        b.iter(|| {
            black_box(foo(&s));
        })
    });

    c.bench_function("prim_arc_none", |b| {
        fn foo() -> u64 {
            3
        }

        let a = Arc::new(Session(5));
        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("prim_arc_clone", |b| {
        fn foo(x: Arc<Session>) -> u64 {
            3
        }

        let a = Arc::new(Session(5));
        b.iter(|| {
            black_box(foo(a.clone()));
        })
    });

    c.bench_function("prim_arc_ref", |b| {
        fn foo(x: &Arc<Session>) -> u64 {
            3
        }

        let a = Arc::new(Session(5));
        b.iter(|| {
            black_box(foo(&a));
        })
    });

    c.bench_function("prim_arc_ref_clone", |b| {
        fn foo(x: &Arc<Session>) -> u64 {
            let x = x.clone();
            3
        }

        let a = Arc::new(Session(5));
        b.iter(|| {
            black_box(foo(&a));
        })
    });
}

/// This bench was used to figure out overheads of using closures,
/// including any difference between Arc and Box, and using enums.
/// Conclusion seems to be that there is a non-trivial overhead
/// for typed closures, which unfortunately is needed, but enums do
/// not add anything, nor is there a difference between Arc and Box.
fn prim_closure(c: &mut Criterion) {
    use std::sync::Arc;

    c.bench_function("prim_closure_symbol", |b| {
        fn foo() -> u64 {
            3
        }

        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("prim_closure_fn", |b| {
        let foo = || 3;

        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("prim_closure_ArcFnUntyped", |b| {
        let x = 3;
        let foo: Arc<_> = Arc::new(move || x);

        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("prim_closure_BoxFnUntyped", |b| {
        let x = 3;
        let foo: Box<_> = Box::new(move || x);

        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("prim_closure_ArcFnTyped", |b| {
        let x = 3;
        let foo: Arc<dyn Fn() -> u64> = Arc::new(move || x);

        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("prim_closure_BoxFnTyped", |b| {
        let x = 3;
        let foo: Box<dyn Fn() -> u64> = Box::new(move || x);

        b.iter(|| {
            black_box(foo());
        })
    });

    pub enum Kernel {
        Function(fn() -> u64),
        ArcClosure(Arc<dyn Fn() -> u64>),
        BoxClosure(Box<dyn Fn() -> u64>),
    }

    impl Kernel {
        fn apply(&self) -> u64 {
            match self {
                Kernel::Function(f) => f(),
                Kernel::ArcClosure(f) => f(),
                Kernel::BoxClosure(f) => f(),
            }
        }
    }

    c.bench_function("prim_closure_enum_symbol", |b| {
        fn foo() -> u64 {
            3
        }

        let k = Kernel::Function(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("prim_closure_enum_fn", |b| {
        let foo = || 3;

        let k = Kernel::Function(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("prim_closure_enum_ArcFnUntyped", |b| {
        let x = 3;
        let foo: Arc<_> = Arc::new(move || x);

        let k = Kernel::ArcClosure(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("prim_closure_enum_BoxFnUntyped", |b| {
        let x = 3;
        let foo: Box<_> = Box::new(move || x);

        let k = Kernel::BoxClosure(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("prim_closure_enum_ArcFnTyped", |b| {
        let x = 3;
        let foo: Arc<dyn Fn() -> u64> = Arc::new(move || x);

        let k = Kernel::ArcClosure(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("prim_closure_enum_BoxFnTyped", |b| {
        let x = 3;
        let foo: Box<dyn Fn() -> u64> = Box::new(move || x);

        let k = Kernel::BoxClosure(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });
}

/// This bench was used to determine when to create kernels.
/// Surprisingly, rustc seems to do some magic that suggests late
/// creation ("inner"); however, this has a significant impact
/// for capturing closures.
fn prim_capture(c: &mut Criterion) {
    trait Compile<F: Fn(u64) -> u64> {
        fn foo(&self) -> F;
    }

    c.bench_function("prim_capture_symbol_outer", |b| {
        struct K;

        impl Compile<fn(u64) -> u64> for K {
            fn foo(&self) -> fn(u64) -> u64 {
                |x| x + 3
            }
        }

        let k = K;
        let f = k.foo();

        b.iter(|| {
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_symbol_inner", |b| {
        struct K;

        impl Compile<fn(u64) -> u64> for K {
            fn foo(&self) -> fn(u64) -> u64 {
                |x| x + 3
            }
        }

        let k = K;

        b.iter(|| {
            let f = k.foo();
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_outer", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn foo(&self) -> Box<dyn Fn(u64) -> u64> {
                Box::new(|x| x + 3)
            }
        }

        let k = K;
        let f = k.foo();

        b.iter(|| {
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_inner", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn foo(&self) -> Box<dyn Fn(u64) -> u64> {
                Box::new(|x| x + 3)
            }
        }

        let k = K;

        b.iter(|| {
            let f = k.foo();
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_move_empty_outer", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn foo(&self) -> Box<dyn Fn(u64) -> u64> {
                Box::new(move |x| x + 3)
            }
        }

        let k = K;
        let f = k.foo();

        b.iter(|| {
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_move_empty_inner", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn foo(&self) -> Box<dyn Fn(u64) -> u64> {
                Box::new(move |x| x + 3)
            }
        }

        let k = K;

        b.iter(|| {
            let f = k.foo();
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_move_nonempty_outer", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn foo(&self) -> Box<dyn Fn(u64) -> u64> {
                let c = 3;
                Box::new(move |x| x + c)
            }
        }

        let k = K;
        let f = k.foo();

        b.iter(|| {
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_move_nonempty_inner", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn foo(&self) -> Box<dyn Fn(u64) -> u64> {
                let c = 3;
                Box::new(move |x| x + c)
            }
        }

        let k = K;

        b.iter(|| {
            let f = k.foo();
            black_box(f(5));
        })
    });
}

criterion_group!(prim, prim_arc, prim_closure, prim_capture);

fn gen_sample_graph(size: usize) -> Computation {
    use moose::execution::*;
    use moose::ring::*;

    let operator = Operator::RingMul(RingMulOp {
        lhs: Ty::Ring64TensorTy,
        rhs: Ty::Ring64TensorTy,
    });

    let mut operations: Vec<_> = (0..size)
        .map(|i| Operation {
            name: format!("y{}", i),
            kind: operator.clone(),
            inputs: vec!["x".into(), "x".into()],
            placement: Placement::Host,
        })
        .collect();

    let raw_tensor: ndarray::ArrayD<u64> =
        ndarray::ArrayBase::from_shape_vec([10, 10], (0..100).collect())
            .unwrap()
            .into_dyn();

    operations.push(Operation {
        name: "x".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Ring64Tensor(Ring64Tensor::from(raw_tensor)),
        }),
        inputs: vec![],
        placement: Placement::Host,
    });

    Computation { operations }.toposort().unwrap()
}

fn compile(c: &mut Criterion) {
    use moose::execution::*;

    let operator = Operator::RingAdd(RingAddOp {
        lhs: Ty::Ring64TensorTy,
        rhs: Ty::Ring64TensorTy,
    });

    c.bench_function("compile_operator/sync", |b| {
        b.iter(|| {
            let kernel: SyncKernel = SyncCompile::compile(&operator).unwrap();
            black_box(kernel);
        })
    });

    c.bench_function("compile_operator/async", |b| {
        b.iter(|| {
            let kernel: AsyncKernel = AsyncCompile::compile(&operator).unwrap();
            black_box(kernel);
        })
    });

    let operation = Operation {
        name: "z".into(),
        kind: operator.clone(),
        inputs: vec!["x".into(), "y".into()],
        placement: Placement::Host,
    };

    c.bench_function("compile_operation/sync", |b| {
        b.iter(|| {
            let compiled: CompiledSyncOperation = operation.compile().unwrap();
            black_box(compiled);
        })
    });

    c.bench_function("compile_operation/async", |b| {
        b.iter(|| {
            let compiled: CompiledAsyncOperation = operation.compile().unwrap();
            black_box(compiled);
        })
    });

    let mut group = c.benchmark_group("compile_computation");
    for size in [10_000, 100_000, 250_000, 500_000, 1_000_000].iter() {
        let comp = gen_sample_graph(*size);

        group.bench_function(BenchmarkId::new("sync", size), |b| {
            b.iter(|| {
                let compiled: CompiledSyncComputation = comp.compile().unwrap();
                black_box(compiled);
            });
        });

        group.bench_function(BenchmarkId::new("async", size), |b| {
            b.iter(|| {
                let compiled: CompiledAsyncComputation = comp.compile().unwrap();
                black_box(compiled);
            });
        });
    }
}

fn execute(c: &mut Criterion) {
    use maplit::hashmap;
    use moose::execution::*;
    use std::sync::Arc;

    let mut group = c.benchmark_group("execute");
    for size in [10_000, 100_000, 250_000, 500_000].iter() {
        let comp = gen_sample_graph(*size);

        group.bench_function(BenchmarkId::new("sync_direct", size), |b| {
            let ctx = SyncContext {
                networking: Box::new(DummySyncNetworking),
            };

            b.iter(|| {
                let sid = 12345;
                let env = hashmap!();
                let res = comp.apply(&ctx, &sid, env);
                black_box(res);
            });
        });

        group.bench_function(BenchmarkId::new("sync_compiled", size), |b| {
            let comp_compiled: CompiledSyncComputation = comp.compile().unwrap();

            let ctx = SyncContext {
                networking: Box::new(DummySyncNetworking),
            };

            b.iter(|| {
                let sid = 12345;
                let env = HashMap::new();
                let res = comp_compiled.apply(&ctx, &sid, env);
                black_box(res);
            });
        });

        group.bench_function(BenchmarkId::new("async_compiled", size), |b| {
            let comp_compiled: CompiledAsyncComputation = comp.compile().unwrap();

            let ctx = Arc::new(AsyncContext {
                runtime: tokio::runtime::Runtime::new().unwrap(),
                networking: Box::new(DummyAsyncNetworking),
            });

            b.iter(|| {
                let sid = Arc::new(12345);
                let env = HashMap::new();
                let (sess, res) = comp_compiled.apply(&ctx, &sid, env).unwrap();
                ctx.join_session(sess).unwrap();
                black_box(res);
            });
        });
    }
}

criterion_group!(computations, compile, execute);

criterion_main!(computations, par, prim);
