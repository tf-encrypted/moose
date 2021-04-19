use maplit::hashmap;
use moose::computation::*;
use moose::execution::EagerExecutor;
use moose::prim::Nonce;
use moose::standard::Shape;

fn main() {
    let key_op = Operation {
        name: "key".into(),
        kind: Operator::PrimGenPrfKey(PrimGenPrfKeyOp),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            owner: Role("alice".into()),
        }),
    };

    let x_seed_op = Operation {
        name: "x_seed".into(),
        kind: Operator::PrimDeriveSeed(PrimDeriveSeedOp {
            nonce: Nonce(vec![1, 2, 3]),
        }),
        inputs: vec!["key".into()],
        placement: Placement::Host(HostPlacement {
            owner: Role("alice".into()),
        }),
    };

    let x_shape_op = Operation {
        name: "x_shape".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Shape(Shape(vec![2, 3])),
        }),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            owner: Role("alice".into()),
        }),
    };

    let x_op = Operation {
        name: "x".into(),
        kind: Operator::RingSample(RingSampleOp {
            output: Ty::Ring64TensorTy,
            max_value: None,
        }),
        inputs: vec!["x_shape".into(), "x_seed".into()],
        placement: Placement::Host(HostPlacement {
            owner: Role("alice".into()),
        }),
    };

    let mut operations = vec![key_op, x_seed_op, x_shape_op, x_op];
    for i in 0..10_000_000 {
        operations.push(Operation {
            name: format!("y{}", i),
            kind: Operator::RingMul(RingMulOp {
                lhs: Ty::Ring64TensorTy,
                rhs: Ty::Ring64TensorTy,
            }),
            inputs: vec!["x".into(), "x".into()],
            placement: Placement::Host(HostPlacement {
                owner: Role("alice".into()),
            }),
        });
    }

    let comp = Computation { operations };
    let args = hashmap![];
    let sid = 12345;

    let executor = EagerExecutor::new();
    // let executor = AsyncExecutor::new();
    let _ = executor.run_computation(&comp, sid, args);
}
