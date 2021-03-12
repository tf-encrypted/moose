use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

fn par(c: &mut Criterion) {
    c.bench_function("par_channel_rayon", |b| {
        b.iter(|| {
            let channels: Vec<_> = (0..100_000)
                .into_par_iter()
                .map(|_| tokio::sync::oneshot::channel::<u64>())
                .collect();
            black_box(channels);
        })
    });

    c.bench_function("par_channel_seq", |b| {
        b.iter(|| {
            let channels: Vec<_> = (0..100_000)
                .map(|_| tokio::sync::oneshot::channel::<u64>())
                .collect();
            black_box(channels);
        })
    });

    c.bench_function("par_compile_rayon_closure", |b| {
        use moose::execution::*;

        let operator = Operator::RingShr(RingShrOp { amount: 1 });

        b.iter(|| {
            let compiled: Vec<_> = (0..1_000_000)
                .into_par_iter()
                .map(|_| operator.sync_kernel())
                .collect();
            black_box(compiled);
        })
    });

    c.bench_function("par_compile_seq_closure", |b| {
        use moose::execution::*;

        let operator = Operator::RingShr(RingShrOp { amount: 1 });

        b.iter(|| {
            let compiled: Vec<_> = (0..1_000_000)
                .map(|_| operator.sync_kernel())
                .collect();
            black_box(compiled);
        })
    });

    c.bench_function("par_compile_rayon_function", |b| {
        use moose::execution::*;

        let operator = Operator::RingAdd(RingAddOp {
            lhs: Ty::Ring64TensorTy,
            rhs: Ty::Ring64TensorTy,
        });

        b.iter(|| {
            let compiled: Vec<_> = (0..1_000_000)
                .into_par_iter()
                .map(|_| operator.sync_kernel())
                .collect();
            black_box(compiled);
        })
    });

    c.bench_function("par_compile_seq_function", |b| {
        use moose::execution::*;

        let operator = Operator::RingAdd(RingAddOp {
            lhs: Ty::Ring64TensorTy,
            rhs: Ty::Ring64TensorTy,
        });

        b.iter(|| {
            let compiled: Vec<_> = (0..1_000_000)
                .map(|_| operator.sync_kernel())
                .collect();
            black_box(compiled);
        })
    });
}

/// This bench is to figure out how to efficiently pass an Arc
/// that the recipient may or may not make use of.
fn arc(c: &mut Criterion) {
    use std::sync::Arc;

    struct Session(u64);

    c.bench_function("arc_direct", |b| {
        fn foo(s: &Session) -> u64 {
            3
        }

        let s = Session(5);
        b.iter(|| {
            black_box(foo(&s));
        })
    });

    c.bench_function("arc_none", |b| {
        fn foo() -> u64 {
            3
        }

        let a = Arc::new(Session(5));
        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("arc_clone", |b| {
        fn foo(x: Arc<Session>) -> u64 {
            3
        }

        let a = Arc::new(Session(5));
        b.iter(|| {
            black_box(foo(a.clone()));
        })
    });

    c.bench_function("arc_ref", |b| {
        fn foo(x: &Arc<Session>) -> u64 {
            3
        }

        let a = Arc::new(Session(5));
        b.iter(|| {
            black_box(foo(&a));
        })
    });

    c.bench_function("arc_ref_clone", |b| {
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

fn closure(c: &mut Criterion) {
    use std::sync::Arc;

    c.bench_function("closure_symbol", |b| {
        fn foo() -> u64 {
            3
        }

        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("closure_fn", |b| {
        let foo = || 3;

        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("closure_ArcFnUntyped", |b| {
        let x = 3;
        let foo: Arc<_> = Arc::new(move || x);

        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("closure_BoxFnUntyped", |b| {
        let x = 3;
        let foo: Box<_> = Box::new(move || x);

        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("closure_ArcFnTyped", |b| {
        let x = 3;
        let foo: Arc<dyn Fn() -> u64> = Arc::new(move || x);

        b.iter(|| {
            black_box(foo());
        })
    });

    c.bench_function("closure_BoxFnTyped", |b| {
        let x = 3;
        let foo: Box<dyn Fn() -> u64> = Box::new(move || x);

        b.iter(|| {
            black_box(foo());
        })
    });
}

fn enum_closure(c: &mut Criterion) {
    use std::sync::Arc;

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

    c.bench_function("enum_closure_symbol", |b| {
        fn foo() -> u64 {
            3
        }

        let k = Kernel::Function(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("enum_closure_fn", |b| {
        let foo = || 3;

        let k = Kernel::Function(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("enum_closure_ArcFnUntyped", |b| {
        let x = 3;
        let foo: Arc<_> = Arc::new(move || x);

        let k = Kernel::ArcClosure(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("enum_closure_BoxFnUntyped", |b| {
        let x = 3;
        let foo: Box<_> = Box::new(move || x);

        let k = Kernel::BoxClosure(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("enum_closure_ArcFnTyped", |b| {
        let x = 3;
        let foo: Arc<dyn Fn() -> u64> = Arc::new(move || x);

        let k = Kernel::ArcClosure(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });

    c.bench_function("enum_closure_BoxFnTyped", |b| {
        let x = 3;
        let foo: Box<dyn Fn() -> u64> = Box::new(move || x);

        let k = Kernel::BoxClosure(foo);

        b.iter(|| {
            black_box(k.apply());
        })
    });
}

fn ret(c: &mut Criterion) {
    trait Compile<F: Fn(u64) -> u64> {
        fn foo(&self) -> F;
    }

    c.bench_function("ret_symbol_outer", |b| {
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

    c.bench_function("ret_symbol_inner", |b| {
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

    c.bench_function("ret_Fn_outer", |b| {
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

    c.bench_function("ret_Fn_inner", |b| {
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

    c.bench_function("ret_Box_outer", |b| {
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

    c.bench_function("ret_Box_inner", |b| {
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

fn compile(c: &mut Criterion) {
    use moose::execution::*;

    // let operator = Operator::RingAdd(RingAddOp {
    //     lhs: Ty::Ring64TensorTy,
    //     rhs: Ty::Ring64TensorTy,
    // });

    let operator = Operator::RingShr(RingShrOp { amount: 1 });

    // let operator = Operator::RingMul(RingMulOp);

    c.bench_function("compile_operator_sync", |b| {
        b.iter(|| {
            let kernel: SyncKernel = operator.sync_kernel();
            black_box(kernel);
        })
    });

    c.bench_function("compile_operator_async", |b| {
        b.iter(|| {
            let kernel: AsyncKernel = operator.async_kernel();
            black_box(kernel);
        })
    });

    let operation = Operation {
        name: "z".into(),
        kind: operator,
        inputs: vec!["x".into(), "y".into()],
        placement: Placement::Host,
    };

    c.bench_function("compile_operation_sync", |b| {
        b.iter(|| {
            let compiled: CompiledOperation<Value> = operation.compile().unwrap();
            black_box(compiled);
        })
    });

    c.bench_function("compile_operation_async", |b| {
        b.iter(|| {
            let compiled: CompiledOperation<AsyncValue> = operation.compile().unwrap();
            black_box(compiled);
        })
    });
}

criterion_group!(benches, par, arc, closure, enum_closure, ret, compile);
criterion_main!(benches);
