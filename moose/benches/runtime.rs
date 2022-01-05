use std::{collections::HashMap, convert::TryInto, time::Duration};

use criterion::{criterion_group, criterion_main, Criterion};
use maplit::hashmap;
use moose::{
    compilation::{compile_passes, Pass},
    computation::{Computation, Role, Value},
    execution::{AsyncTestRuntime, Identity},
};

fn runtime_simple_computation(c: &mut Criterion) {
    let source = r#"x = Input {arg_name = "x"}: () -> Int64Tensor @Host(alice)
    y = Input {arg_name = "y"}: () -> Int64Tensor @Host(alice)
    z = HostAdd: (Int64Tensor, Int64Tensor) -> Int64Tensor (x, y) @Host(alice)
    output = Output: (Int64Tensor) -> Int64Tensor (z) @Host(alice)
    "#;
    let computation: Computation = source.try_into().unwrap();
    let x: Value = "Int64Tensor([5]) @Host(alice)".try_into().unwrap();
    let y: Value = "Int64Tensor([10]) @Host(alice)".try_into().unwrap();
    let arguments: HashMap<String, Value> = hashmap!("x".to_string() => x, "y".to_string()=> y);
    let storage_mapping: HashMap<String, HashMap<String, Value>> =
        hashmap!("alice".to_string() => hashmap!());
    let role_assignments: HashMap<String, String> =
        hashmap!("alice".to_string() => "alice".to_string());

    let valid_role_assignments = role_assignments
        .into_iter()
        .map(|arg| (Role::from(arg.1), Identity::from(arg.0)))
        .collect::<HashMap<Role, Identity>>();
    c.bench_function("runtime_simple_computation", |b| {
        b.iter(|| {
            let storage_mapping = storage_mapping.clone();
            let valid_role_assignments = valid_role_assignments.clone();
            let arguments = arguments.clone();

            let mut executor = AsyncTestRuntime::new(storage_mapping);
            let _outputs = executor
                .evaluate_computation(&computation, valid_role_assignments, arguments)
                .unwrap();
        })
    });
}

fn runtime_two_hosts(c: &mut Criterion) {
    let source = r#"
    x0 = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(alice)
    x1 = Constant{value=Float32Tensor([[1.0, 0.0], [0.0, 1.0]])}: () -> Float32Tensor @Host(bob)
    res = HostDot: (Float32Tensor, Float32Tensor) -> Float32Tensor (x0, x1) @Host(alice)
    output = Output: (Float32Tensor) -> Float32Tensor (res) @Host(alice)
    "#;
    let computation: Computation = source.try_into().unwrap();
    let computation = compile_passes(&computation, &[Pass::Networking, Pass::Toposort]).unwrap();

    let arguments: HashMap<String, Value> = hashmap!();
    let storage_mapping: HashMap<String, HashMap<String, Value>> =
        hashmap!("alice".to_string() => hashmap!(), "bob".to_string()=>hashmap!());
    let role_assignments: HashMap<String, String> = hashmap!("alice".to_string() => "alice".to_string(), "bob".to_string() => "bob".to_string());

    let valid_role_assignments = role_assignments
        .into_iter()
        .map(|arg| (Role::from(arg.1), Identity::from(arg.0)))
        .collect::<HashMap<Role, Identity>>();
    c.bench_function("runtime_two_hosts_dot", |b| {
        b.iter(|| {
            let storage_mapping = storage_mapping.clone();
            let valid_role_assignments = valid_role_assignments.clone();
            let arguments = arguments.clone();

            let mut executor = AsyncTestRuntime::new(storage_mapping);
            let _outputs = executor
                .evaluate_computation(&computation, valid_role_assignments, arguments)
                .unwrap();
        })
    });
}

fn runtime_rep_computation(c: &mut Criterion) {
    let source = include_str!("./rep_computation.moose");
    let computation: Computation = source.try_into().unwrap();
    let computation = compile_passes(&computation, &[Pass::Networking, Pass::Toposort]).unwrap();

    let arguments: HashMap<String, Value> = hashmap!();
    let storage_mapping: HashMap<String, HashMap<String, Value>> = hashmap!("alice".to_string() => hashmap!(), "bob".to_string()=>hashmap!(), "carole".to_string()=>hashmap!());
    let role_assignments: HashMap<String, String> = hashmap!("alice".to_string() => "alice".to_string(), "bob".to_string() => "bob".to_string(), "carole".to_string() => "carole".to_string());

    let valid_role_assignments = role_assignments
        .into_iter()
        .map(|arg| (Role::from(arg.1), Identity::from(arg.0)))
        .collect::<HashMap<Role, Identity>>();
    let mut group = c.benchmark_group("Slow Tests");
    group.measurement_time(Duration::new(10, 0));
    group.bench_function("runtime_replicated_computation", |b| {
        b.iter(|| {
            let storage_mapping = storage_mapping.clone();
            let valid_role_assignments = valid_role_assignments.clone();
            let arguments = arguments.clone();

            let mut executor = AsyncTestRuntime::new(storage_mapping);
            let _outputs = executor
                .evaluate_computation(&computation, valid_role_assignments, arguments)
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
