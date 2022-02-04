use crate::replicated::ReplicatedPlacement;

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
