use moose::additive::{AbstractAdditiveTensor, Additive128Tensor, Additive64Tensor};
use moose::computation::{AdditivePlacement, HostPlacement};
use moose::kernels::{PlacementReveal, PlacementTruncPrProvider};
use moose::ring::Ring64Tensor;
use proptest::prelude::*;

use moose::ring::FromRawPlc;

use moose::{kernels::SyncSession, ring::AbstractRingTensor};

pub fn tuple_gen() -> impl Strategy<Value = (Ring64Tensor, Ring64Tensor)> {
    (0usize..25)
        .prop_flat_map(|length| {
            (
                proptest::collection::vec(any::<u64>(), length),
                proptest::collection::vec(any::<u64>(), length),
            )
        })
        .prop_map(|(x, y)| (Ring64Tensor::from(x), Ring64Tensor::from(y)))
        .boxed()
}

fn any_bounded_u64() -> impl Strategy<Value = u64> {
    any::<u64>().prop_map(|x| (x >> 2) - 1)
}

proptest! {
    #[test]
    fn fuzzy_trunc(v in proptest::collection::vec(any_bounded_u64(), 1..5), amount in 0usize..62
    ) {

        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let bob = HostPlacement {
            owner: "bob".into(),
        };
        let carole = HostPlacement {
            owner: "carole".into(),
        };
        let adt = AdditivePlacement {
            owners: ["alice".into(), "bob".into()],
        };

        let x = Additive64Tensor {
            shares: [
                Ring64Tensor::from_vec_raw_plc(vec![0; v.len()], alice),
                Ring64Tensor::from_vec_raw_plc(v.clone(), bob)
            ],
        };

        let sess = SyncSession::default();
        let x_trunc = adt.trunc_pr(&sess, amount, &carole, &x);
        let _y = carole.reveal(&sess, &x_trunc);

        let t_vec = v.iter().map(|x| x >> amount).collect::<Vec<_>>();
        let target = AbstractRingTensor::from_vec_raw_plc(t_vec, carole);

        for (i, value) in _y.0.iter().enumerate() {
            let diff = value - target.0[i];
            assert!(
                diff == std::num::Wrapping(1)
                    || diff == std::num::Wrapping(u64::MAX)
                    || diff == std::num::Wrapping(0),
                "difference = {}, lhs = {}, rhs = {}",
                diff,
                value,
                target.0[i]
            );
        }
    }
}
