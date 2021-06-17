use moose::computation::*;
use moose::execution::*;
use moose::prim::Nonce;
use moose::standard::Shape;

fn main() {
    let key_op = Operation {
        name: "key".into(),
        kind: Operator::PrimGenPrfKey(PrimGenPrfKeyOp {
            sig: Signature::Nullary(NullarySignature { ret: Ty::PrfKeyTy }),
        }),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            owner: Role::from("alice"),
        }),
    };

    let x_seed_op = Operation {
        name: "x_seed".into(),
        kind: Operator::PrimDeriveSeed(PrimDeriveSeedOp {
            nonce: Nonce(vec![1, 2, 3]),
        }),
        inputs: vec!["key".into()],
        placement: Placement::Host(HostPlacement {
            owner: Role::from("alice"),
        }),
    };

    let x_shape_op = Operation {
        name: "x_shape".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Shape(Shape(vec![2, 3])),
        }),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            owner: Role::from("alice"),
        }),
    };

    let x_op = Operation {
        name: "x".into(),
        kind: Operator::RingSample(RingSampleOp {
            sig: Signature::Nullary(NullarySignature {
                ret: Ty::Ring64TensorTy,
            }),
            max_value: None,
        }),
        inputs: vec!["x_shape".into(), "x_seed".into()],
        placement: Placement::Host(HostPlacement {
            owner: Role::from("alice"),
        }),
    };

    let mut operations = vec![key_op, x_seed_op, x_shape_op, x_op];
    for i in 0..10_000_000 {
        operations.push(Operation {
            name: format!("y{}", i),
            kind: Operator::RingMul(RingMulOp {
                sig: Signature::Binary(BinarySignature {
                    arg0: Ty::Ring64TensorTy,
                    arg1: Ty::Ring64TensorTy,
                    ret: Ty::Ring64TensorTy,
                }),
            }),
            inputs: vec!["x".into(), "x".into()],
            placement: Placement::Host(HostPlacement {
                owner: Role::from("alice"),
            }),
        });
    }

    let comp = Computation { operations };

    let exec = TestExecutor::default();
    let outputs = exec.run_computation(&comp, SyncArgs::new()).unwrap();
    println!("Outputs: {:?}", outputs);
}
