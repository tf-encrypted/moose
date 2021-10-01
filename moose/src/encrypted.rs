use crate::circuits::bristol_fashion::aes;
use crate::computation::*;
use crate::fixedpoint::{Fixed128Tensor, FixedTensor};
use crate::host::{AbstractHostEncFixedTensor, HostBitTensor, HostEncFixed128Tensor, HostFixed128Tensor, HostRing128Tensor, HostShape};
use crate::kernels::{
    PlacementAdd, PlacementAnd, PlacementBitDec, PlacementFill, PlacementIndexAxis, PlacementNeg,
    PlacementRingInject, PlacementSetupGen, PlacementShape, PlacementShareSetup, PlacementXor,
    Session,
};
use crate::replicated::{AbstractReplicatedFixedTensor, AbstractReplicatedRingTensor, AbstractReplicatedShape, ReplicatedBitTensor, ReplicatedFixed128Tensor, ReplicatedRing128Tensor, ReplicatedShape};

impl AesDecryptOp {
    pub(crate) fn host_fixed_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        c: HostEncFixed128Tensor,
    ) -> HostFixed128Tensor
    where
        HostPlacement: PlacementBitDec<S, HostRing128Tensor, HostBitTensor>,
        HostPlacement: PlacementIndexAxis<S, HostBitTensor, HostBitTensor>,
        HostPlacement: PlacementRingInject<S, HostBitTensor, HostRing128Tensor>,
        HostPlacement: PlacementShape<S, HostEncFixed128Tensor, HostShape>,
        HostPlacement: PlacementFill<S, HostShape, HostRing128Tensor>,
        HostPlacement: PlacementAdd<S, HostRing128Tensor, HostRing128Tensor, HostRing128Tensor>,
        HostPlacement: PlacementXor<S, HostBitTensor, HostBitTensor, HostBitTensor>,
        HostPlacement: PlacementAnd<S, HostBitTensor, HostBitTensor, HostBitTensor>,
        HostPlacement: PlacementNeg<S, HostBitTensor, HostBitTensor>,
    {
        let shape = plc.shape(sess, &c);
        let c_decomposed = plc.bit_decompose(sess, &c.tensor);
        let c_bits: Vec<_> = (0..128)
            .map(|i| plc.index_axis(sess, 0, i, &c_decomposed))
            .collect();
        let m_bits = aes(sess, plc, c_bits.clone(), c_bits);
        let zero = plc.fill(sess, Constant::Ring128(0), &shape);
        let m = m_bits
            .iter()
            .enumerate()
            .map(|(i, b)| plc.ring_inject(sess, i, b))
            .fold(zero, |acc, x| plc.add(sess, &acc, &x));

        HostFixed128Tensor {
            tensor: m,
            fractional_precision: c.precision,
            integral_precision: 0, // TODO
        }
    }
}

impl AesDecryptOp {
    pub(crate) fn rep_fixed_kernel<S: Session, HostBitT, RepBitT, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        c: AbstractHostEncFixedTensor<HostRingT>,
    ) -> AbstractReplicatedFixedTensor<m!(AbstractReplicatedRingTensor<HostRingT>)>
    where
        AbstractHostEncFixedTensor<HostRingT>: KnownType<S>,
        Fixed128Tensor: KnownType<S>,
        HostFixed128Tensor: KnownType<S>,
        ReplicatedRing128Tensor: KnownType<S>,
        ReplicatedFixed128Tensor: KnownType<S>,
        RepBitT: Clone,
        HostRingT: Placed,
        HostRingT::Placement: Into<Placement>,

        HostShape: KnownType<S>,
        m!(HostShape): Clone,
        ReplicatedShape: KnownType<S>,
        AbstractReplicatedShape<m!(HostShape)>: Into<m!(ReplicatedShape)>,
        AbstractReplicatedRingTensor<HostRingT>: KnownType<S>,
        AbstractReplicatedFixedTensor<m!(ReplicatedRing128Tensor)>: Into<m!(ReplicatedFixed128Tensor)>,

        HostPlacement: PlacementBitDec<S, HostRingT, HostBitT>,
        HostPlacement: PlacementIndexAxis<S, HostBitT, HostBitT>,
        HostPlacement: PlacementShape<S, AbstractHostEncFixedTensor<HostRingT>, m!(HostShape)>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, HostBitT, RepBitT>,
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        ReplicatedPlacement: PlacementFill<S, m!(ReplicatedShape), m!(AbstractReplicatedRingTensor<HostRingT>)>,
        ReplicatedPlacement: PlacementAdd<
            S,
            m!(AbstractReplicatedRingTensor<HostRingT>),
            m!(AbstractReplicatedRingTensor<HostRingT>),
            m!(AbstractReplicatedRingTensor<HostRingT>),
        >,
        ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementAnd<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementNeg<S, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, m!(AbstractReplicatedRingTensor<HostRingT>)>,
    {
        let host = match c.placement() {
            Ok(Placement::Host(plc)) => plc,
            _ => unimplemented!(),
        };
        let setup = rep.gen_setup(sess);

        let host_shape: m!(HostShape) = host.shape(sess, &c);
        let shape: m!(ReplicatedShape) = AbstractReplicatedShape {
            shapes: [host_shape.clone(), host_shape.clone(), host_shape],
        }.into();

        // decompose into bits on host
        let c_decomposed = host.bit_decompose(sess, &c.tensor);
        let c_bits = (0..128).map(|i| host.index_axis(sess, 0, i, &c_decomposed));

        // sharing bits (could be public values but we don't have that yet)
        let c_bits_shared: Vec<RepBitT> = c_bits.map(|b| rep.share(sess, &setup, &b)).collect();

        // run AES
        let m_bits: Vec<RepBitT> = aes(sess, rep, c_bits_shared.clone(), c_bits_shared);

        // re-compose on rep
        let zero = rep.fill(sess, Constant::Ring128(0), &shape);
        let m = m_bits
            .iter()
            .enumerate()
            .map(|(i, b)| rep.ring_inject(sess, i, b))
            .fold(zero, |acc, x| rep.add(sess, &acc, &x));

        // wrap up as fixed-point tensor
        AbstractReplicatedFixedTensor {
            tensor: m,
            fractional_precision: c.precision,
            integral_precision: 0, // TODO
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_aes_decrypt_host() {}
}
