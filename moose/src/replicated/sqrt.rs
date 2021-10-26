use std::cmp::max;
use std::convert::TryInto;

use ndarray::Array1;

use crate::computation::{CanonicalType, HostPlacement, KnownType, RepFixedpointSqrtOp, ReplicatedPlacement};
use crate::error::{Error, Result};
use crate::host::{AbstractHostRingTensor, HostRing64Tensor, FromRawPlc};
use crate::{Const, Ring, kernels::*};
use crate::replicated::{ReplicatedRing64Tensor, ReplicatedRing128Tensor, ReplicatedShape};

use super::AbstractReplicatedRingTensor;

pub trait SimplifiedNormSq<S: Session, SetupT, RepBitT, HostRingT> {
    fn simp_norm_sq_64(
        &self,
        sess: &S,
        setup: &SetupT,
        x: AbstractReplicatedRingTensor<HostRingT>,
        total_precision: usize,
        fractional_precision: usize
    ) -> (RepBitT, AbstractReplicatedRingTensor<HostRingT>);
    // TODO
    // fn simp_norm_sq_128(&self, sess: &S, setup: &SetupT, x: AbstractReplicatedRingTensor<HostRingT>) -> (RepBitT, AbstractReplicatedRingTensor<HostRingT>);
}

impl<S: Session, SetupT, RepBitT, HostRingT, N: Const> SimplifiedNormSq<S, SetupT, RepBitT, HostRingT> for ReplicatedPlacement
where
    HostRingT: Clone + Ring<BitLength = N> + Tensor<S> + From<HostRing64Tensor>,
    <HostRingT as Tensor<S>>::Scalar: From<usize>,
    AbstractReplicatedRingTensor<HostRingT>: CanonicalType,
    <AbstractReplicatedRingTensor<HostRingT> as CanonicalType>::Type: KnownType<S>,
    st!(AbstractReplicatedRingTensor<HostRingT>): TryInto<AbstractReplicatedRingTensor<HostRingT>>,
    ReplicatedShape: KnownType<S>,
    AbstractReplicatedRingTensor<HostRingT>: Clone + Into<st!(AbstractReplicatedRingTensor<HostRingT>)>,
    HostPlacement: PlacementPlace<S, HostRingT>,
    ReplicatedPlacement:
        PlacementMsb<S, SetupT, AbstractReplicatedRingTensor<HostRingT>, AbstractReplicatedRingTensor<HostRingT>>,
    ReplicatedPlacement: PlacementAdd<S, RepBitT, RepBitT, RepBitT>,
    ReplicatedPlacement:PlacementAdd<
        S,
        AbstractReplicatedRingTensor<HostRingT>,
        AbstractReplicatedRingTensor<HostRingT>,
        AbstractReplicatedRingTensor<HostRingT>
    >,
    ReplicatedPlacement: PlacementDotSetup<S, SetupT, AbstractReplicatedRingTensor<HostRingT>, HostRingT, AbstractReplicatedRingTensor<HostRingT>>,
    ReplicatedPlacement: PlacementShape<S, AbstractReplicatedRingTensor<HostRingT>, cs!(ReplicatedShape)>,
    ReplicatedPlacement: PlacementIndexAxis<S, AbstractReplicatedRingTensor<HostRingT>, AbstractReplicatedRingTensor<HostRingT>>,
    ReplicatedPlacement: PlacementZeros<S, cs!(ReplicatedShape), RepBitT>,
    ReplicatedPlacement: PlacementShl<S, AbstractReplicatedRingTensor<HostRingT>, AbstractReplicatedRingTensor<HostRingT>>,
{
    fn simp_norm_sq_64(
        &self,
        sess: &S,
        setup: &SetupT,
        x: AbstractReplicatedRingTensor<HostRingT>,
        total_precision: usize,
        fractional_precision: usize,
    ) -> (RepBitT, AbstractReplicatedRingTensor<HostRingT>)
    {
        let bit_length = HostRingT::BitLength::VALUE;
        let rep = self;
        // TODO op that gives masked bitdec msb
        let z = rep.msb(sess, setup, &x);
        // TODO write helper to determine last axis of n-dim tensor
        let bits_axis = Some(-1i32);
        // TODO choose this host placement randomly?
        let (p0, _, _) = rep.host_placements();
        let parity_indices = (1u64..bit_length as u64).collect();
        let raw_index_vec = Array1::from_vec(parity_indices);
        let index_vec = AbstractHostRingTensor::from_raw_plc(raw_index_vec, p0);
        // TODO add Constant kernels for HostRingTensor<R>
        // let v = p0.constant(sess, index_vec.into());
        let v = p0.place(sess, index_vec.into());
        let m = rep.dot_setup(sess, setup, &z, &v);
        let m_shape = rep.shape(sess, &m);
        let m_odd = rep.zeros(sess, &m_shape);
        let zs: Vec<_> = (0u64..bit_length as u64 - 1).map(|i| rep.index_axis(sess, 1, i.try_into().unwrap(), &z)).collect();
        for (i, zi) in zs.iter().enumerate() {
            if i % 2 == 0 {
                m_odd = rep.add(sess, &m_odd, zi);
            }
        }
        let ws: Vec<_> = (1..bit_length / 2).map(|i| rep.add(sess, &zs[2*i - 1], &zs[2*i])).collect();
        let w_shl = (1..bit_length / 2 - 1).map(|i| rep.shl(sess, i, &ws[i - 1])).collect();
        // TODO replace with variadic add_n operation for replicated tensors
        let mut ring_res = rep.shl(sess, 0, &w_shl[0]);
        for wi in w_shl.iter().skip(1) {
            ring_res = rep.add(sess, &ring_res, wi);
        }
        (m_odd, ring_res)
    }
}

modelled!(PlacementSqrtAsFixedpoint::sqrt_as_fixedpoint, ReplicatedPlacement, attributes[total_precision: usize, fractional_precision: usize] (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepFixedpointSqrtOp);
modelled!(PlacementSqrtAsFixedpoint::sqrt_as_fixedpoint, ReplicatedPlacement, attributes[total_precision: usize, fractional_precision: usize] (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepFixedpointSqrtOp);

kernel!{
    RepFixedpointSqrtOp,
    [
        (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] attributes[total_precision, fractional_precision] Self::rep_kernel_64),
        (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [transparent] attributes[total_precision, fractional_precision] Self::rep_kernel_128),
    ]
}

impl RepFixedpointSqrtOp {
    fn rep_kernel_64<S: Session, HostRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        total_precision: usize,
        fractional_precision: usize,
        x: AbstractReplicatedRingTensor<HostRingT>,
    ) -> Result<AbstractReplicatedRingTensor<HostRingT>>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: SimplifiedNormSq<S, S::ReplicatedSetup, RepBitT, HostRingT>,
    {
        let theta = (total_precision as f64).log2().ceil() as usize;
        theta = max(theta, 6);
        let setup = rep.gen_setup(sess);
        let (m_odd, w) = rep.simp_norm_sq_64(sess, &setup, x, total_precision, fractional_precision);
        unimplemented!("TODO")
    }

    fn rep_kernel_128<S: Session, HostRingT, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        total_precision: usize,
        fractional_precision: usize,
        x: AbstractReplicatedRingTensor<HostRingT>,
    ) -> Result<AbstractReplicatedRingTensor<HostRingT>>
    where
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: SimplifiedNormSq<S, S::ReplicatedSetup, RepBitT, HostRingT>,
    {
        unimplemented!("TODO")
    }
}
