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
            kind: Operator::RingMul(RingMulOp {
                lhs: Ty::Ring64TensorTy,
                rhs: Ty::Ring64TensorTy,
            }),
            inputs: vec!["x".into(), "x".into()],
            placement: Placement::Host,
        });
    }

    println!("Computation");
    let comp = Computation { operations };

    println!("Executing");
    // let executor = EagerExecutor;
    let executor = AsyncExecutor;
    let _ = executor.run_computation(&comp, 12345, env);
}
