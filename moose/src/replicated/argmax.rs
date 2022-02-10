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
    ReplicatedPlacement: PlacementGreaterThan<S, T, T, m!(ReplicatedBitTensor)>,
    ReplicatedBitTensor: KnownType<S>,
    ReplicatedPlacement: PlacementRingInject<S, m!(ReplicatedBitTensor), T>,
    ReplicatedPlacement: PlacementMux<S, T, T, T, T>,
{
    fn reduce_argmax(&self, sess: &S, x: &[(T, T)]) -> (T, T) {
        let elementwise_argmax =
            |rep: &ReplicatedPlacement, sess: &S, x: &(T, T), y: &(T, T)| -> (T, T) {
                let comp_bin = rep.greater_than(sess, &x.1, &y.1);
                let comp_ring = rep.ring_inject(sess, 0, &comp_bin);
                (
                    rep.mux(sess, &comp_ring, &x.0, &y.0),
                    rep.mux(sess, &comp_ring, &x.1, &y.1),
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
        ReplicatedPlacement: PlacementExpandDims<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShareReduction<S, RepRingT, m!(ReplicatedRing64Tensor)>,
    {
        let xs: Vec<_> = (0..upmost_index)
            .map(|index| rep.index_axis(sess, axis, index, &x))
            .collect();

        let x_pairs: Vec<(RepRingT, RepRingT)> = xs
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let result = (
                    rep.fill(sess, (i as u8).into(), &rep.shape(sess, item)),
                    item.clone(),
                );
                result
            })
            .collect();

        // TODO(Dragos) here we can optimize at the first round of argmax, by doing it manually until we get replicated types all around
        let (secret_index, _max_value) = rep.reduce_argmax(sess, &x_pairs);
        let expanded_index = rep.expand_dims(sess, [axis].to_vec(), &secret_index);

        // (x0 + x1 + x2) mod 2^128 = x , iff x in [0, 2^64)
        // (x0  mod 2^64 + x1 mod 2^64 + x2 mod 2^64) mod 2^64 = x
        // share trunc operation
        Ok(rep.share_reduction(sess, &expanded_index))
    }
}

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
    use ndarray::Zip;

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

    #[test]
    fn test_argmax_64() {
        let x = array![1, 2, -3_i64, 4, 2, 2, 2, 3, 105].into_dyn();
        let axis_index = 0_usize;

        let mut current_max = x.index_axis(Axis(axis_index), 0).to_owned();
        let mut current_pattern_max = current_max.mapv(|_x| 0_u64);

        for (index, subview) in x.axis_iter(Axis(axis_index)).enumerate() {
            let index = index as u64;
            Zip::from(&mut current_max)
                .and(&mut current_pattern_max)
                .and(&subview)
                .for_each(|max_entry, pattern_entry, &subview_entry| {
                    if *max_entry < subview_entry {
                        *max_entry = subview_entry;
                        *pattern_entry = index;
                    }
                });
        }
        // println!("x_max: {:?}", x_max);
        // println!("argmax: {:?}", expected_argmax.insert_axis(Axis(axis_index)));
        let expected_argmax = current_pattern_max.insert_axis(Axis(axis_index));
        test_rep_argmax64(x.mapv(|item| item as u64), expected_argmax, 0, 9);
    }
}
