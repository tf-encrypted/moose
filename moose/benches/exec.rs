use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moose::computation::Computation;
use rayon::prelude::*;
use std::convert::TryFrom;

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

criterion_group!(par, par_channel, par_spawn);

/// This bench was used to figure out how to efficiently pass an Arc
/// that the recipient may or may not make use of.
fn prim_arc(c: &mut Criterion) {
    use std::sync::Arc;

    struct Session(u64);

    c.bench_function("prim_arc_direct", |b| {
        fn f(_s: &Session) -> u64 {
            3
        }

        let s = Session(5);
        b.iter(|| {
            black_box(f(&s));
        })
    });

    c.bench_function("prim_arc_none", |b| {
        fn f() -> u64 {
            3
        }

        let _a = Arc::new(Session(5));
        b.iter(|| {
            black_box(f());
        })
    });

    c.bench_function("prim_arc_clone", |b| {
        fn f(_x: Arc<Session>) -> u64 {
            3
        }

        let a = Arc::new(Session(5));
        b.iter(|| {
            black_box(f(a.clone()));
        })
    });

    c.bench_function("prim_arc_ref", |b| {
        fn f(_x: &Arc<Session>) -> u64 {
            3
        }

        let a = Arc::new(Session(5));
        b.iter(|| {
            black_box(f(&a));
        })
    });

    c.bench_function("prim_arc_ref_clone", |b| {
        fn f(x: &Arc<Session>) -> u64 {
            let _x = x.clone();
            3
        }

        let a = Arc::new(Session(5));
        b.iter(|| {
            black_box(f(&a));
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
        fn f() -> u64 {
            3
        }

        b.iter(|| {
            black_box(f());
        })
    });

    c.bench_function("prim_closure_fn", |b| {
        let f = || 3;

        b.iter(|| {
            black_box(f());
        })
    });

    c.bench_function("prim_closure_ArcFnUntyped", |b| {
        let x = 3;
        let f: Arc<_> = Arc::new(move || x);

        b.iter(|| {
            black_box(f());
        })
    });

    c.bench_function("prim_closure_BoxFnUntyped", |b| {
        let x = 3;
        let f: Box<_> = Box::new(move || x);

        b.iter(|| {
            black_box(f());
        })
    });

    c.bench_function("prim_closure_ArcFnTyped", |b| {
        let x = 3;
        let f: Arc<dyn Fn() -> u64> = Arc::new(move || x);

        b.iter(|| {
            black_box(f());
        })
    });

    c.bench_function("prim_closure_BoxFnTyped", |b| {
        let x = 3;
        let f: Box<dyn Fn() -> u64> = Box::new(move || x);

        b.iter(|| {
            black_box(f());
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
        fn f() -> u64 {
            3
        }
        let k = Kernel::Function(f);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("prim_closure_enum_fn", |b| {
        let f = || 3;
        let k = Kernel::Function(f);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("prim_closure_enum_ArcFnUntyped", |b| {
        let x = 3;
        let f: Arc<_> = Arc::new(move || x);
        let k = Kernel::ArcClosure(f);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("prim_closure_enum_BoxFnUntyped", |b| {
        let x = 3;
        let f: Box<_> = Box::new(move || x);
        let k = Kernel::BoxClosure(f);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("prim_closure_enum_ArcFnTyped", |b| {
        let x = 3;
        let f: Arc<dyn Fn() -> u64> = Arc::new(move || x);
        let k = Kernel::ArcClosure(f);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("prim_closure_enum_BoxFnTyped", |b| {
        let x = 3;
        let f: Box<dyn Fn() -> u64> = Box::new(move || x);
        let k = Kernel::BoxClosure(f);

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
        fn compile(&self) -> F;
    }

    c.bench_function("prim_capture_symbol_outer", |b| {
        struct K;

        impl Compile<fn(u64) -> u64> for K {
            fn compile(&self) -> fn(u64) -> u64 {
                |x| x + 3
            }
        }

        let k = K;
        let f = k.compile();

        b.iter(|| {
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_symbol_inner", |b| {
        struct K;

        impl Compile<fn(u64) -> u64> for K {
            fn compile(&self) -> fn(u64) -> u64 {
                |x| x + 3
            }
        }

        let k = K;

        b.iter(|| {
            let f = k.compile();
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_outer", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn compile(&self) -> Box<dyn Fn(u64) -> u64> {
                Box::new(|x| x + 3)
            }
        }

        let k = K;
        let f = k.compile();

        b.iter(|| {
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_inner", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn compile(&self) -> Box<dyn Fn(u64) -> u64> {
                Box::new(|x| x + 3)
            }
        }

        let k = K;

        b.iter(|| {
            let f = k.compile();
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_move_empty_outer", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn compile(&self) -> Box<dyn Fn(u64) -> u64> {
                Box::new(move |x| x + 3)
            }
        }

        let k = K;
        let f = k.compile();

        b.iter(|| {
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_move_empty_inner", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn compile(&self) -> Box<dyn Fn(u64) -> u64> {
                Box::new(move |x| x + 3)
            }
        }

        let k = K;

        b.iter(|| {
            let f = k.compile();
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_move_nonempty_outer", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn compile(&self) -> Box<dyn Fn(u64) -> u64> {
                let c = 3;
                Box::new(move |x| x + c)
            }
        }

        let k = K;
        let f = k.compile();

        b.iter(|| {
            black_box(f(5));
        })
    });

    c.bench_function("prim_capture_Box_move_nonempty_inner", |b| {
        struct K;

        impl Compile<Box<dyn Fn(u64) -> u64>> for K {
            fn compile(&self) -> Box<dyn Fn(u64) -> u64> {
                let c = 3;
                Box::new(move |x| x + c)
            }
        }

        let k = K;

        b.iter(|| {
            let f = k.compile();
            black_box(f(5));
        })
    });
}

criterion_group!(prim, prim_arc, prim_closure, prim_capture);

fn gen_sample_graph(size: usize) -> Computation {
    use moose::computation::*;
    use moose::host::*;

    let operator = Operator::RingMul(RingMulOp {
        sig: Signature::binary(
            Ty::HostRing64Tensor,
            Ty::HostRing64Tensor,
            Ty::HostRing64Tensor,
        ),
    });

    let mut operations: Vec<_> = (0..size)
        .map(|i| Operation {
            name: format!("y{}", i),
            kind: operator.clone(),
            inputs: vec!["x".into(), "x".into()],
            placement: Placement::Host(HostPlacement {
                owner: Role("alice".into()),
            }),
        })
        .collect();

    let raw_tensor: ndarray::ArrayD<u64> =
        ndarray::ArrayBase::from_shape_vec([10, 10], (0..100).collect())
            .unwrap()
            .into_dyn();

    operations.push(Operation {
        name: "x".into(),
        kind: Operator::Constant(ConstantOp {
            sig: Signature::nullary(Ty::HostRing64Tensor),
            value: Constant::HostRing64Tensor(HostRing64Tensor::from(raw_tensor)),
        }),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            owner: Role("alice".into()),
        }),
    });

    Computation { operations }.toposort().unwrap()
}

criterion_main!(par, prim);
