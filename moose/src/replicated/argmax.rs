use super::*;
use crate::computation::ArgmaxOp;
use crate::error::Result;
use crate::execution::Session;

pub(crate) trait TreeReduceArgmax<S: Session, T, O> {
    fn reduce_argmax(&self, sess: &S, x: &[(T, T)]) -> (O, O);
}

impl<S: Session, T: Clone> TreeReduceArgmax<S, T, T> for ReplicatedPlacement
where
    ReplicatedPlacement: PlacementMul<S, T, T, T>,
    ReplicatedPlacement: PlacementLessThan<S, T, T, m!(ReplicatedBitTensor)>,
    ReplicatedBitTensor: KnownType<S>,
    ReplicatedPlacement: PlacementRingInject<S, m!(ReplicatedBitTensor), T>,
    ReplicatedPlacement: PlacementMux<S, T, T, T, T>,
{
    fn reduce_argmax(&self, sess: &S, x: &[(T, T)]) -> (T, T) {
        let elementwise_argmax =
            |rep: &ReplicatedPlacement, sess: &S, x: &(T, T), y: &(T, T)| -> (T, T) {
                let comp_bin = rep.less(sess, &x.1, &y.1);
                let comp_ring = rep.ring_inject(sess, 0, &comp_bin);
                (
                    rep.mux(sess, &comp_ring, &y.0, &x.0),
                    rep.mux(sess, &comp_ring, &y.1, &x.1),
                )
            };
        self.tree_reduce_argmax(sess, x, elementwise_argmax)
    }
}

impl ArgmaxOp {
    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: usize,
        upmost_index: usize,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<m!(ReplicatedRing64Tensor)>
    where
        ReplicatedRing64Tensor: KnownType<S>,
        ReplicatedPlacement: PlacementArgmax<S, RepRingT, m!(ReplicatedRing64Tensor)>,
    {
        Ok(rep.argmax(sess, axis, upmost_index, &x.tensor))
    }

    pub(crate) fn rep_ring_kernel<S: Session, RepRingT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: usize,
        upmost_index: usize,
        x: RepRingT,
    ) -> Result<m!(ReplicatedRing64Tensor)>
    where
        RepRingT: Clone,
        ReplicatedRing64Tensor: KnownType<S>,
        ReplicatedPlacement: PlacementIndexAxis<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShape<S, RepRingT, ShapeT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepRingT>,
        ReplicatedPlacement: TreeReduceArgmax<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShareReduction<S, RepRingT, m!(ReplicatedRing64Tensor)>,
    {
        let xs: Vec<_> = (0..upmost_index)
            .map(|index| rep.index_axis(sess, axis, index, &x))
            .collect();

        let x_pairs: Vec<(RepRingT, RepRingT)> = xs
            .iter()
            .enumerate()
            .map(|(i, item)| {
                (
                    rep.fill(sess, (i as u8).into(), &rep.shape(sess, item)),
                    item.clone(),
                )
            })
            .collect();

        // TODO(Dragos) here we can optimize at the first round of argmax, by doing it manually until we get replicated types all around
        let (secret_index, _max_value) = rep.reduce_argmax(sess, &x_pairs);

        // (x0 + x1 + x2) mod 2^128 = x , iff x in [0, 2^64)
        // (x0  mod 2^64 + x1 mod 2^64 + x2 mod 2^64) mod 2^64 = x
        // share trunc operation
        Ok(rep.share_reduction(sess, &secret_index))
    }
}


type RepTuple<R> = (R, R);

impl ReplicatedPlacement {
    pub(crate) fn tree_reduce_argmax<S, RepT>(
        &self,
        sess: &S,
        x: &[(RepT, RepT)],
        op: fn(&Self, &S, &(RepT, RepT), &(RepT, RepT)) -> (RepT, RepT),
    ) -> (RepT, RepT)
    where
        RepT: Clone,
    {
        let v_len = x.len();
        if v_len == 1 {
            (x[0].0.clone(), x[0].1.clone())
        } else {
            let chunk1 = &x[0..v_len / 2];
            let chunk2 = &x[v_len / 2..v_len];

            let op_res_chunk1 = self.tree_reduce(sess, chunk1, op);
            let op_res_chunk2 = self.tree_reduce(sess, chunk2, op);
            op(self, sess, &op_res_chunk1, &op_res_chunk2)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::host::FromRaw;
    use crate::host::HostRingTensor;
    use crate::kernels::*;
    use crate::prelude::*;
    use ndarray::array;
    use ndarray::prelude::*;

    macro_rules! rep_argmax_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(x: ArrayD<$tt>, y_target: ArrayD<u64>, axis: usize, upmost_index: usize) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let x: HostRingTensor<_> = alice.from_raw(x);
                let x_shared = rep.share(&sess, &x);
                let argmax = rep.$test_func(&sess, axis, upmost_index, &x_shared); // output is ReplicatedRing64Tensor

                let opened_argmax = alice.reveal(&sess, &argmax);
                let y_target: HostRing64Tensor = alice.from_raw(y_target);
                assert_eq!(y_target, opened_argmax);
            }
        };
    }

    rep_argmax_test!(test_rep_argmax64, argmax<u64>);
    rep_argmax_test!(test_rep_argmax128, argmax<u128>);

    #[test]
    fn test_argmax_64_1() {
        let x = array![1_i64, 2, -3, 4, 2, 2, 2, 3, 105].into_dyn();
        let expected_argmax = Array::from_elem([], 8_u64).into_dyn();
        test_rep_argmax64(x.mapv(|item| item as u64), expected_argmax, 0, 9);
    }

    #[test]
    fn test_argmax_64_2() {
        let x = array![
            [1231_i64, -323, -3, 12321],
            [93213, 12312321, -321, -3249],
            [3921, 4012, 3221, -321]
        ]
        .into_dyn();
        let expected_argmax = array![3_u64, 1, 1].into_dyn();
        test_rep_argmax64(x.mapv(|item| item as u64), expected_argmax, 1, 4);
    }

    #[test]
    fn test_argmax_64_3() {
        let x = array![
            [-3233_i64, 329423, 54843],
            [3994, 123, -31326],
            [-324, 55321, 23441]
        ]
        .into_dyn();
        let expected_argmax = array![1_u64, 0, 1].into_dyn();
        test_rep_argmax64(x.mapv(|item| item as u64), expected_argmax, 1, 3);
    }

    #[test]
    fn test_argmax_128_1() {
        let x = array![1_i128, 2, -3, 10000, 10000, 10000, 10000, 10000, 10000].into_dyn();
        let expected_argmax = Array::from_elem([], 3_u64).into_dyn();
        test_rep_argmax128(x.mapv(|item| item as u128), expected_argmax, 0, 9);
    }

    #[test]
    fn test_argmax_128_2() {
        let x = array![
            [1231_i128, -323, -3, 12321],
            [93213, 12312321, -321, -3249],
            [3921, 4012, 3221, -321]
        ]
        .into_dyn();
        let expected_argmax = array![3_u64, 1, 1].into_dyn();
        test_rep_argmax128(x.mapv(|item| item as u128), expected_argmax, 1, 4);
    }

    #[test]
    fn test_argmax_128_3() {
        let x = array![
            [-3233_i128, 329423, 54843],
            [3994, 123, -31326],
            [-324, 55321, 23441]
        ]
        .into_dyn();
        let expected_argmax = array![1_u64, 0, 1].into_dyn();
        test_rep_argmax128(x.mapv(|item| item as u128), expected_argmax, 1, 3);
    }
}
