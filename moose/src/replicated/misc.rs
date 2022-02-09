use super::*;
use crate::mirrored::Mirrored3Placement;

impl ReplicatedPlacement {
    pub(crate) fn prefix_op<S, RepT>(
        &self,
        sess: &S,
        x: Vec<RepT>,
        op: fn(&Self, &S, &RepT, &RepT) -> RepT,
    ) -> Vec<RepT> {
        let v_len = x.len();

        let log_r = ((v_len as f64).log2().ceil()) as u32;

        let mut res = x;
        for i in 0..log_r {
            for j in 0..(2_i32.pow(log_r) / 2_i32.pow(i + 1)) {
                let y = (2_i32.pow(i) + j * 2_i32.pow(i + 1) - 1) as usize;
                let k_bound = (2_i32.pow(i) + 1) as usize;
                for k in 1..k_bound {
                    if y + k < v_len {
                        res[y + k] = op(self, sess, &res[y], &res[y + k]);
                    }
                }
            }
        }
        res
    }

    pub(crate) fn prefix_or<S: Session, RepT>(&self, sess: &S, x: Vec<RepT>) -> Vec<RepT>
    where
        ReplicatedPlacement: PlacementAnd<S, RepT, RepT, RepT>,
        ReplicatedPlacement: PlacementXor<S, RepT, RepT, RepT>,
    {
        let elementwise_or = |rep: &ReplicatedPlacement, sess: &S, x: &RepT, y: &RepT| -> RepT {
            rep.xor(sess, &rep.xor(sess, x, y), &rep.and(sess, x, y))
        };

        self.prefix_op(sess, x, elementwise_or)
    }

    #[allow(dead_code)]
    pub(crate) fn prefix_and<S: Session, RepT>(&self, sess: &S, x: Vec<RepT>) -> Vec<RepT>
    where
        ReplicatedPlacement: PlacementAnd<S, RepT, RepT, RepT>,
    {
        let elementwise_and = |rep: &ReplicatedPlacement, sess: &S, x: &RepT, y: &RepT| -> RepT {
            rep.and(sess, x, y)
        };

        self.prefix_op(sess, x, elementwise_and)
    }

    pub(crate) fn tree_reduce<S, RepT>(
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

/// Shift all shares of replicated secret to the right
///
/// It should be used carefully since `[x] >> amount` is *not* equal to
/// `[ [x0 >> amount, x1 >> amount], [x1 >> amount, x2 >> amount], [x2 >> amount, x0>>amount]`.
/// Used in conjunction with split operation so that we don't use the full
/// bit-decomposition in order to perform exact truncation.
pub(crate) trait PlacementShrRaw<S: Session, T, O> {
    fn shr_raw(&self, sess: &S, amount: usize, x: &T) -> O;
}

#[cfg(feature = "compilation")]
impl<S: Session, HostRingT: Placed<Placement = HostPlacement>>
    PlacementShrRaw<S, Symbolic<RepTensor<HostRingT>>, Symbolic<RepTensor<HostRingT>>>
    for ReplicatedPlacement
where
    ReplicatedPlacement: PlacementShrRaw<S, RepTensor<HostRingT>, RepTensor<HostRingT>>,
    RepTensor<HostRingT>: Into<Symbolic<RepTensor<HostRingT>>>,
{
    fn shr_raw(
        &self,
        sess: &S,
        amount: usize,
        x: &Symbolic<RepTensor<HostRingT>>,
    ) -> Symbolic<RepTensor<HostRingT>> {
        let concrete_x = match x {
            Symbolic::Concrete(x) => x,
            Symbolic::Symbolic(_) => {
                unimplemented!()
            }
        };
        let concrete_y = Self::shr_raw(self, sess, amount, concrete_x);
        concrete_y.into()
    }
}

impl<S: Session, HostRingT> PlacementShrRaw<S, RepTensor<HostRingT>, RepTensor<HostRingT>>
    for ReplicatedPlacement
where
    HostPlacement: PlacementShr<S, HostRingT, HostRingT>,
{
    fn shr_raw(&self, sess: &S, amount: usize, x: &RepTensor<HostRingT>) -> RepTensor<HostRingT> {
        let (player0, player1, player2) = self.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.shr(sess, amount, x00);
        let z10 = player0.shr(sess, amount, x10);

        let z11 = player1.shr(sess, amount, x11);
        let z21 = player1.shr(sess, amount, x21);

        let z22 = player2.shr(sess, amount, x22);
        let z02 = player2.shr(sess, amount, x02);

        RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
}

pub(crate) trait ShapeFill<S, TenT> {
    type Result;

    fn shape_fill<C: Into<Constant>>(
        &self,
        sess: &S,
        fill_value: C,
        shape_from: &TenT,
    ) -> Self::Result;
}

impl<S: Session, TenT> ShapeFill<S, TenT> for ReplicatedPlacement
where
    TenT: MirroredCounterpart,
    Self: PlacementShape<S, TenT, m!(ReplicatedShape)>,
    Mirrored3Placement: PlacementFill<S, m!(ReplicatedShape), TenT::MirroredType>,
    ReplicatedShape: KnownType<S>,
{
    type Result = TenT::MirroredType;

    fn shape_fill<C: Into<Constant>>(
        &self,
        sess: &S,
        fill_value: C,
        shape_from: &TenT,
    ) -> Self::Result {
        let shape = self.shape(sess, shape_from);

        let (player0, player1, player2) = self.host_placements();

        let mir = Mirrored3Placement {
            owners: [player0.owner, player1.owner, player2.owner],
        };

        mir.fill(sess, fill_value.into(), &shape)
    }
}

pub(crate) trait BinaryAdder<S: Session, RepBitT> {
    fn binary_adder(&self, sess: &S, x: &RepBitT, y: &RepBitT, ring_size: usize) -> RepBitT;
}

/// Binary addition protocol for tensors
impl<S: Session, RepBitT> BinaryAdder<S, RepBitT> for ReplicatedPlacement
where
    RepBitT: Clone,
    ReplicatedPlacement: PlacementAnd<S, RepBitT, RepBitT, RepBitT>,
    ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
    ReplicatedPlacement: PlacementShlDim<S, RepBitT, RepBitT>,
{
    fn binary_adder(&self, sess: &S, x: &RepBitT, y: &RepBitT, ring_size: usize) -> RepBitT {
        #![allow(clippy::many_single_char_names)]

        let rep = self;
        let log_r = (ring_size as f64).log2() as usize; // we know that R = 64/128

        // g is part of the generator set, p propagator set
        // A few helpful diagrams to understand what is happening here:
        // https://www.chessprogramming.org/Kogge-Stone_Algorithm or here: https://inst.eecs.berkeley.edu/~eecs151/sp19/files/lec20-adders.pdf

        // consider we have inputs a, b to the P,G computing gate
        // P = P_a and P_b
        // G = G_b xor (G_a and P_b)

        // P, G can be computed in a tree fashion, performing ops on chunks of len 2^i
        // Note the first level is computed as P0 = x ^ y, G0 = x & y;

        // Perform `g = x * y` for every tensor
        let mut g = rep.and(sess, x, y);

        // Perform `p_store = x + y` (just a helper to avoid compute xor() twice)
        let p_store = rep.xor(sess, x, y);
        let mut p = p_store.clone();

        // (Dragos) Note that in the future we might want to delete shl_dim op and replace it with
        // slice + stack op - however atm we can't do this. It can be unblocked after the following are implemented:
        // 1) slice tensors with unknown shape at compile time
        // 2) stack variable length of replicated tensors (variadic kernels + stack op)

        for i in 0..log_r {
            // computes p << (1<<i)
            // [ a[0], ... a[amount] ... a[ring_size - 1]
            // [ a[amount]...a[ring_size-1] 0 ... 0 ]
            let p1 = rep.shl_dim(sess, 1 << i, ring_size, &p);
            // computes g >> (1<<i)
            let g1 = rep.shl_dim(sess, 1 << i, ring_size, &g);

            // Note that the original algorithm had G_a and P_b, but we can have
            // G_a and P_a instead because the 1s in P_a do not matter in the final result
            // since they are cancelled out by the zeros in G_a
            let p_and_g = rep.and(sess, &p, &g1);

            // update `g = g xor p1 and g1`
            g = rep.xor(sess, &g, &p_and_g);

            // update `p = p * p1`
            p = rep.and(sess, &p, &p1);
        }

        // c is a copy of g with the first tensor (corresponding to the first bit) zeroed out
        let c = rep.shl_dim(sess, 1, ring_size, &g);

        // final result is `z = c xor p_store`
        rep.xor(sess, &c, &p_store)
    }
}
