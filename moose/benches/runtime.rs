use std::{collections::HashMap, convert::TryInto, time::Duration};

use criterion::{criterion_group, criterion_main, Criterion};
use maplit::hashmap;
use moose::{
    compilation::{compile, Pass},
    computation::{Computation, Role, Value},
    execution::{AsyncTestRuntime, Identity},
};

fn runtime_simple_computation(c: &mut Criterion) {
    let source = r#"x = Input {arg_name = "x"}: () -> HostInt64Tensor @Host(alice)
    y = Input {arg_name = "y"}: () -> HostInt64Tensor @Host(alice)
    z = Add: (HostInt64Tensor, HostInt64Tensor) -> HostInt64Tensor (x, y) @Host(alice)
    output = Output: (HostInt64Tensor) -> HostInt64Tensor (z) @Host(alice)
    "#;
    let computation: Computation = source.try_into().unwrap();
    let x: Value = "HostInt64Tensor([5]) @Host(alice)".try_into().unwrap();
    let y: Value = "HostInt64Tensor([10]) @Host(alice)".try_into().unwrap();
    let arguments: HashMap<String, Value> = hashmap!("x".to_string() => x, "y".to_string()=> y);
    let storage_mapping: HashMap<String, HashMap<String, Value>> =
        hashmap!("alice".to_string() => hashmap!());
    let role_assignments: HashMap<Role, Identity> = hashmap!("alice".into() => "alice".into());

    c.bench_function("runtime_simple_computation", |b| {
        b.iter(|| {
            let storage_mapping = storage_mapping.clone();
            let role_assignments = role_assignments.clone();
            let arguments = arguments.clone();

            let mut executor = AsyncTestRuntime::new(storage_mapping);
            let _outputs = executor
                .evaluate_computation(&computation, role_assignments, arguments)
                .unwrap();
        })
    });
}

fn runtime_two_hosts(c: &mut Criterion) {
    let source = r#"
    x0 = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
    x1 = Constant{value=HostFloat32Tensor([[1.0, 0.0], [0.0, 1.0]])}: () -> HostFloat32Tensor @Host(bob)
    res = Dot: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x0, x1) @Host(alice)
    output = Output: (HostFloat32Tensor) -> HostFloat32Tensor (res) @Host(alice)
    "#;
    let computation: Computation = source.try_into().unwrap();
    let computation = compile(computation, Some(vec![Pass::Networking, Pass::Toposort])).unwrap();

    let arguments: HashMap<String, Value> = hashmap!();
    let storage_mapping: HashMap<String, HashMap<String, Value>> =
        hashmap!("alice".to_string() => hashmap!(), "bob".to_string()=>hashmap!());
    let role_assignments: HashMap<Role, Identity> =
        hashmap!("alice".into() => "alice".into(), "bob".into() => "bob".into());

    c.bench_function("runtime_two_hosts_dot", |b| {
        b.iter(|| {
            let storage_mapping = storage_mapping.clone();
            let role_assignments = role_assignments.clone();
            let arguments = arguments.clone();

            let mut executor = AsyncTestRuntime::new(storage_mapping);
            let _outputs = executor
                .evaluate_computation(&computation, role_assignments, arguments)
                .unwrap();
        })
    });
}

fn runtime_rep_computation(c: &mut Criterion) {
    let source = include_str!("./rep_computation.moose");
    let computation: Computation = source.try_into().unwrap();
    let computation = compile(computation, Some(vec![Pass::Networking, Pass::Toposort])).unwrap();

    let arguments: HashMap<String, Value> = hashmap!();
    let storage_mapping: HashMap<String, HashMap<String, Value>> = hashmap!(
            "alice".to_string() => hashmap!(),
            "bob".to_string()=>hashmap!(),
            "carole".to_string()=>hashmap!());
    let role_assignments: HashMap<Role, Identity> = hashmap!(
            "alice".into() => "alice".into(),
            "bob".into() => "bob".into(),
            "carole".into() => "carole".into());

    let mut group = c.benchmark_group("Slow Tests");
    group.measurement_time(Duration::new(10, 0));
    group.bench_function("runtime_replicated_computation", |b| {
        b.iter(|| {
            let storage_mapping = storage_mapping.clone();
            let role_assignments = role_assignments.clone();
            let arguments = arguments.clone();

            let mut executor = AsyncTestRuntime::new(storage_mapping);
            let _outputs = executor
                .evaluate_computation(&computation, role_assignments, arguments)
                .unwrap();
        })
    });
    group.finish();
}

criterion_group!(
    runtime,
    runtime_simple_computation,
    runtime_two_hosts,
    runtime_rep_computation
);
criterion_main!(runtime);
