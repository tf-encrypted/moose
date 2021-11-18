use crate::computation::{LessThanOp, ReplicatedPlacement};
use crate::error::Result;
use crate::kernels::*;
use crate::replicated::{ReplicatedRing128Tensor, ReplicatedRing64Tensor};

modelled!(PlacementLessThan::less_than, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, LessThanOp);
modelled!(PlacementLessThan::less_than, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, LessThanOp);

impl LessThanOp {
    pub(crate) fn rep_kernel<S: Session, RepRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
        y: RepRingT,
    ) -> Result<RepRingT>
    where
        ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepRingT>,
    {
        let z = rep.sub(sess, &x, &y);
        Ok(rep.msb(sess, &z))
    }
}

mod tests {
    use crate::computation::{HostPlacement, ReplicatedPlacement};
    use crate::host::AbstractHostRingTensor;
    use crate::kernels::*;
    use ndarray::prelude::*;

    macro_rules! rep_binary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement {
                    owner: "alice".into(),
                };
                let rep = ReplicatedPlacement {
                    owners: ["alice".into(), "bob".into(), "carole".into()],
                };

                let x = AbstractHostRingTensor::from_raw_plc(xs, alice.clone());
                let y = AbstractHostRingTensor::from_raw_plc(ys, alice.clone());

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &x);
                let y_shared = rep.share(&sess, &y);

                let sum = rep.$test_func(&sess, &x_shared, &y_shared);
                let opened_product = alice.reveal(&sess, &sum);
                assert_eq!(
                    opened_product,
                    AbstractHostRingTensor::from_raw_plc(zs, alice.clone())
                );
            }
        };
    }

    rep_binary_func_test!(test_rep_lt64, less_than<u64>);
    rep_binary_func_test!(test_rep_lt128, less_than<u128>);

    #[test]
    fn test_rep_lt_64() {
        let x = array![0u64, 1, 2, -1_i64 as u64, -2_i64 as u64].into_dyn();
        let y = array![
            -1_i64 as u64,
            -2_i64 as u64,
            3_u64,
            -1_i64 as u64,
            -1_i64 as u64
        ]
        .into_dyn();
        let target = array![0, 0, 1, 0, 1].into_dyn();
        test_rep_lt64(x, y, target);
    }

    #[test]
    fn test_rep_lt_128() {
        let x = array![0u128, 1, 2, -1_i128 as u128, -2_i128 as u128].into_dyn();
        let y = array![
            -1_i128 as u128,
            -2_i128 as u128,
            3_u128,
            -1_i128 as u128,
            -1_i128 as u128
        ]
        .into_dyn();
        let target = array![0, 0, 1, 0, 1].into_dyn();
        test_rep_lt128(x, y, target);
    }
}
