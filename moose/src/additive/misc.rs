use crate::execution::Session;
use crate::host::HostPlacement;
use crate::kernels::{PlacementAdd, PlacementShl};
use crate::replicated::ReplicatedPlacement;

pub trait BitCompose<S: Session, R> {
    fn bit_compose(&self, sess: &S, bits: &[R]) -> R;
}

impl<S: Session, R> BitCompose<S, R> for HostPlacement
where
    R: Clone,
    HostPlacement: PlacementShl<S, R, R>,
    HostPlacement: TreeReduce<S, R>,
{
    fn bit_compose(&self, sess: &S, bits: &[R]) -> R {
        let shifted_bits: Vec<_> = (0..bits.len())
            .map(|i| self.shl(sess, i, &bits[i]))
            .collect();
        self.tree_reduce(sess, &shifted_bits)
    }
}

pub trait TreeReduce<S: Session, R> {
    fn tree_reduce(&self, sess: &S, sequence: &[R]) -> R;
}

impl<S: Session, R> TreeReduce<S, R> for HostPlacement
where
    R: Clone,
    HostPlacement: PlacementAdd<S, R, R, R>,
{
    fn tree_reduce(&self, sess: &S, sequence: &[R]) -> R {
        let n = sequence.len();
        if n == 1 {
            sequence[0].clone()
        } else {
            let mut reduced: Vec<_> = (0..n / 2)
                .map(|i| {
                    let x0: &R = &sequence[2 * i];
                    let x1: &R = &sequence[2 * i + 1];
                    self.add(sess, x0, x1)
                })
                .collect();
            if n % 2 == 1 {
                reduced.push(sequence[n - 1].clone());
            }
            self.tree_reduce(sess, &reduced)
        }
    }
}

impl ReplicatedPlacement {
    pub fn tree_reduce<S, RepT>(
        &self,
        sess: &S,
        x: &[RepT],
        op: fn(&Self, &S, &RepT, &RepT) -> RepT,
    ) -> RepT
    where
        RepT: Clone,
    {
        let v_len = x.len();
        if v_len == 1 {
            x[0].clone()
        } else {
            let chunk1 = &x[0..v_len / 2];
            let chunk2 = &x[v_len / 2..v_len];

            let op_res_chunk1 = self.tree_reduce(sess, chunk1, op);
            let op_res_chunk2 = self.tree_reduce(sess, chunk2, op);
            op(self, sess, &op_res_chunk1, &op_res_chunk2)
        }
    }
}
