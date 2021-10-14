use ndarray::ArrayD;

use crate::computation::{ReplicatedPlacement, RepSqrtOp};
use crate::error::Result;
use crate::kernels::*;
use crate::replicated::ReplicatedShape;

pub trait SimplifiedNormSq<S: Session, SetupT, BitT, RepT, N> {
    fn simp_norm_sq(&self, sess: &S, setup: &SetupT, x: RepT) -> (BitT, RepT);
}

impl<S: Session, SetupT, RepBitT, RepRingT, N> SimplifiedNormSq<S, SetupT, RepBitT, RepRingT, N> for ReplicatedPlacement
{
    fn simp_norm_sq(&self, sess: &S, rep: &ReplicatedPlacement, setup: &SetupT, x: RepRingT) -> (RepBitT, RepRingT) 
    where 
        ReplicatedPlacement: PlacementMsb<S, SetupT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementSum<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShape<S, RepRingT, ReplicatedShape>,
        ReplicatedPlacement: PlacementZeros<S, ReplicatedShape, RepRingT>
    {
        let z = rep.msb(sess, setup, &x);
        let bits_axis = Some(-1u32);
        // TODO choose this host placement randomly?
        let (p0, _, _) = rep.host_placements();
        let raw_index_vec = ArrayD::<RingT::Scalar>::from_vec(1..N);
        let index_vec = RingT::from_raw_plc(raw_index_vec, p0);
        let m = rep.dot(sess, &z, &index_vec);
        let m_shape = rep.shape(sess, &m);
        let m_odd = rep.zeros(sess, &m_shape);
        let zs: Vec<_> = (0..N - 1).map(|i| rep.index_axis(sess, -1usize, i, &z)).collect();
        for zi in zs.iter() {
            if i % 2 == 0 {
                m_odd = rep.add(sess, &m_odd, &zi);
            }
        }
        let ws: Vec<_> = (1..N / 2).map(|i| rep.add(sess, &zs[2*i - 1], &zs[2i])).collect();
        let w_shl = (1..N / 2 - 1).map(|i| rep.shl(sess, i, &ws[i - 1])).collect();
        // TODO replace with variadic add_n operation for replicated tensors
        let mut ring_res = rep.shl(sess, 0, &w_shl[0]);
        for wi in w_shl.iter().skip(1) {
            ring_res = rep.add(sess, &ring_res, wi);
        }
        (m_odd, ring_res)
    }
}

impl RepSqrtOp {
    fn rep_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT
    ) -> Result<RepRingT> {
        unimplemented!("")
    }
}
