use moose::computation::*;
use moose::execution::{SyncSession, TestSyncExecutor};
use moose::host::{HostPlacement, RawShape, SyncKey};
use std::convert::TryFrom;

fn main() {
    let key_op = Operation {
        name: "key".into(),
        kind: PrimPrfKeyGenOp {
            sig: Signature::nullary(Ty::PrfKey),
        }
        .into(),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            owner: Role::from("alice"),
        }),
    };

    let x_seed_op = Operation {
        name: "x_seed".into(),
        kind: PrimDeriveSeedOp {
            sig: Signature::unary(Ty::PrfKey, Ty::Seed),
            sync_key: SyncKey::try_from(vec![1, 2, 3]).unwrap(),
        }
        .into(),
        inputs: vec!["key".into()],
        placement: Placement::Host(HostPlacement {
            owner: Role::from("alice"),
        }),
    };

    let x_shape_op = Operation {
        name: "x_shape".into(),
        kind: ConstantOp {
            sig: Signature::nullary(Ty::HostShape),
            value: Constant::RawShape(RawShape(vec![2, 3])),
        }
        .into(),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            owner: Role::from("alice"),
        }),
    };

    let x_op = Operation {
        name: "x".into(),
        kind: RingSampleSeededOp {
            sig: Signature::binary(Ty::HostShape, Ty::Seed, Ty::HostRing64Tensor),
            max_value: None,
        }
        .into(),
        inputs: vec!["x_shape".into(), "x_seed".into()],
        placement: Placement::Host(HostPlacement {
            owner: Role::from("alice"),
        }),
    };

    let mut operations = vec![key_op, x_seed_op, x_shape_op, x_op];
    for i in 0..10_000_000 {
        operations.push(Operation {
            name: format!("y{}", i),
            kind: MulOp {
                sig: Signature::binary(
                    Ty::HostRing64Tensor,
                    Ty::HostRing64Tensor,
                    Ty::HostRing64Tensor,
                ),
            }
            .into(),
            inputs: vec!["x".into(), "x".into()],
            placement: Placement::Host(HostPlacement {
                owner: Role::from("alice"),
            }),
        });
    }

    let comp = Computation { operations };

    let executor = TestSyncExecutor::default();
    let session = SyncSession::from_session_id(SessionId::try_from("foobar").unwrap());
    let outputs = executor.run_computation(&comp, &session).unwrap();

    println!("Outputs: {:?}", outputs);
}
