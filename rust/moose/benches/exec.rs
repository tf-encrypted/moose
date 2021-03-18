use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::prelude::*;
use std::collections::HashMap;

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
        use tokio::sync::oneshot::{Receiver, Sender};
        b.iter(|| {
            let channels: Vec<(Sender<_>, Receiver<_>)> = (0..100_000)
                .map(|_| tokio::sync::oneshot::channel::<u64>())
                .collect();
            black_box(channels);
        })
    });

    c.bench_function("par_channel_seq_arc", |b| {
        use std::sync::Arc;

        let creator: Arc<dyn Fn() -> (_, _)> = Arc::new(|| tokio::sync::oneshot::channel::<u64>());

        b.iter(|| {
            let channels: Vec<_> = (0..100_000).map(|_| &creator).collect();
            black_box(channels);
        })
    });

    c.bench_function("par_spawn_rayon", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap();

            let channels: Vec<_> = (0..100_000)
                .into_par_iter()
                .map(|_| rt.spawn(async move { black_box(5) }))
                .collect();

            black_box(channels);
        })
    });

    c.bench_function("par_spawn_seq", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap();

            let channels: Vec<_> = (0..100_000)
                .map(|_| rt.spawn(async move { black_box(5) }))
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
                .map(|_| SyncCompile::compile(&operator))
                .collect();
            black_box(compiled);
        })
    });

    c.bench_function("par_compile_seq_closure", |b| {
        use moose::execution::*;

        let operator = Operator::RingShr(RingShrOp { amount: 1 });

        b.iter(|| {
            let compiled: Vec<_> = (0..1_000_000)
                .map(|_| SyncCompile::compile(&operator))
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
                .map(|_| SyncCompile::compile(&operator))
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
                .map(|_| SyncCompile::compile(&operator))
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
    use std::sync::Arc;

    let operator = Operator::RingAdd(RingAddOp {
        lhs: Ty::Ring64TensorTy,
        rhs: Ty::Ring64TensorTy,
    });

    // let operator = Operator::RingShr(RingShrOp { amount: 1 });
    // let context = Arc::new(KernelContext);

    // let operator = Operator::RingMul(RingMulOp);

    // c.bench_function("compile_operator_sync", |b| {
    //     b.iter(|| {
    //         // let kernel: SyncKernel = operator.new_sync_kernel();
    //         let kernel: SyncKernel = operator.sync_kernel();
    //         black_box(kernel);
    //     })
    // });

    // c.bench_function("compile_operator_async", |b| {
    //     b.iter(|| {
    //         let kernel: AsyncKernel = operator.async_kernel();
    //         black_box(kernel);
    //     })
    // });

    let operation = Operation {
        name: "z".into(),
        kind: operator,
        inputs: vec!["x".into(), "y".into()],
        placement: Placement::Host,
    };

    c.bench_function("compile_operation_sync", |b| {
        b.iter(|| {
            let compiled: CompiledSyncOperation = operation.compile().unwrap();
            black_box(compiled);
        })
    });

    c.bench_function("compile_operation_async", |b| {
        b.iter(|| {
            let compiled: CompiledAsyncOperation = operation.compile().unwrap();
            black_box(compiled);
        })
    });
}

fn exec(c: &mut Criterion) {
    use maplit::hashmap;
    use moose::execution::*;
    use moose::ring::*;
    use std::sync::Arc;

    let mut ops: Vec<_> = (0..500_000)
        .map(|i| Operation {
            name: format!("y{}", i),
            kind: Operator::RingAdd(RingAddOp {
                lhs: Ty::Ring64TensorTy,
                rhs: Ty::Ring64TensorTy,
            }),
            inputs: vec!["x".into(), "x".into()],
            placement: Placement::Host,
        })
        .collect();

    let x_op = Operation {
        name: "x".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Ring64Tensor(Ring64Tensor::from(vec![1, 2, 3, 4])),
        }),
        inputs: vec![],
        placement: Placement::Host,
    };

    ops.push(x_op);

    let comp = Computation { operations: ops }.toposort().unwrap();

    c.bench_function("exec_sync_compile", |b| {
        b.iter(|| {
            let compiled: CompiledSyncComputation = comp.compile().unwrap();
            black_box(compiled);
        })
    });

    c.bench_function("exec_async_compile", |b| {
        b.iter(|| {
            let compiled: CompiledAsyncComputation = comp.compile().unwrap();
            black_box(compiled);
        })
    });

    c.bench_function("exec_compiled_sync", |b| {
        let comp_compiled: CompiledSyncComputation = comp.compile().unwrap();

        b.iter(|| {
            let ctx = SyncContext {
                networking: Box::new(DummySyncNetworking),
            };

            let sid = 12345;

            let env = HashMap::new();
            // let env = HashMap::with_capacity(500_000);

            let res = comp_compiled.apply(&ctx, &sid, env);
            black_box(res);
        })
    });

    c.bench_function("exec_compiled_async", |b| {
        let comp_compiled: CompiledAsyncComputation = comp.compile().unwrap();

        b.iter(|| {
            let ctx = Arc::new(AsyncContext {
                runtime: tokio::runtime::Runtime::new().unwrap(),
                networking: Box::new(DummyAsyncNetworking),
            });

            let sid = Arc::new(12345);

            let env = HashMap::new();
            // let env = HashMap::with_capacity(500_000);

            let (sess, res) = comp_compiled.apply(&ctx, &sid, env).unwrap();
            ctx.join_session(sess).unwrap();
            black_box(ctx);
        })
    });

    c.bench_function("exec_sync_direct", |b| {
        b.iter(|| {
            let ctx = SyncContext {
                networking: Box::new(DummySyncNetworking),
            };

            let sid = 12345;

            let env = hashmap!();

            let res = comp.apply(&ctx, &sid, env);
            black_box(res);
        })
    });

    // c.bench_function("compile_computation_async", |b| {
    //     b.iter(|| {
    //         let compiled: CompiledAsyncOperation = operation.compile().unwrap();
    //         black_box(compiled);
    //     })
    // });
}

criterion_group!(benches, par, arc, closure, enum_closure, ret, compile, exec);
criterion_main!(benches);
