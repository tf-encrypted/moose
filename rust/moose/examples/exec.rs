use maplit::hashmap;
use moose::execution::*;

fn main() {
    let mut env = hashmap![];

    let key_op = Operation {
        name: "key".into(),
        kind: Operator::PrimGenPrfKey(PrimGenPrfKeyOp),
        inputs: vec![],
        placement: Placement::Host,
    };

    let x_seed_op = Operation {
        name: "x_seed".into(),
        kind: Operator::PrimDeriveSeed(PrimDeriveSeedOp {
            nonce: Nonce(vec![1, 2, 3]),
        }),
        inputs: vec!["key".into()],
        placement: Placement::Host,
    };

    let x_shape_op = Operation {
        name: "x_shape".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Shape(Shape(vec![2, 3])),
        }),
        inputs: vec![],
        placement: Placement::Host,
    };

    let x_op = Operation {
        name: "x".into(),
        kind: Operator::RingSample(RingSampleOp {
            output: Ty::Ring64TensorTy,
            max_value: None,
        }),
        inputs: vec!["x_shape".into(), "x_seed".into()],
        placement: Placement::Host,
    };

    println!("Constructing");
    let mut operations = vec![key_op, x_seed_op, x_shape_op, x_op];
    for i in 0..10_000_000 {
        operations.push(Operation {
            name: format!("y{}", i),
            kind: Operator::RingMul(RingMulOp), // { lhs: Ty::Ring64TensorTy, rhs: Ty::Ring64TensorTy }),
            inputs: vec!["x".into(), "x".into()],
            placement: Placement::Host,
        });
    }

    println!("Looping");
    let mut count = 0;
    for op in operations.iter() {
        if op.name == "not" {
            count += 1;
        }
    }

    println!("Computation");
    let comp = Computation { operations };

    println!("Executing");
    let executor = AsyncExecutor;
    let _ = executor.run_computation(&comp, env);

    // println!("Compiling");
    // let compiled_comp: CompiledComputation<Value> = comp.compile().unwrap();

    // // let rt = tokio::runtime::Builder::new_multi_thread()
    // //     .enable_all()
    // //     .build()
    // //     .unwrap();

    // // rt.block_on(async {
    //     println!("Launching");
    //     let env = compiled_comp.apply(env);
    //     println!("Running");
    //     // tokio::time::sleep(tokio::time::Duration::from_secs(120)).await;
    //     // let vals = futures::future::join_all(
    //     //     env.values().map(|op| op.clone()).collect::<Vec<_>>()).await;
    // // });
    // // rt.shutdown_timeout(tokio::time::Duration::from_secs(120));
    // println!("Done");
}
