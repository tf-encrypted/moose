use crate::bristol_fashion::aes;
use crate::computation::*;
use crate::error::Result;
use crate::host::{
    AbstractHostEncFixedTensor, HostBitTensor, HostEncFixed128Tensor, HostFixed128Tensor,
    HostRing128Tensor, HostShape,
};
use crate::kernels::{
    PlacementAdd, PlacementAnd, PlacementBitDec, PlacementFill, PlacementIndexAxis, PlacementNeg,
    PlacementRingInject, PlacementSetupGen, PlacementShape, PlacementShareSetup, PlacementXor,
    Session,
};
use crate::replicated::{AbstractReplicatedFixedTensor, AbstractReplicatedShape, ReplicatedShape};

impl AesDecryptOp {
    pub(crate) fn host_fixed_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        c: HostEncFixed128Tensor,
    ) -> Result<HostFixed128Tensor>
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

        Ok(HostFixed128Tensor {
            tensor: m,
            fractional_precision: c.precision,
            integral_precision: 0, // TODO
        })
    }
}

impl AesDecryptOp {
    pub(crate) fn rep_fixed_kernel<S: Session, HostBitT, RepBitT, RepRingT, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        c: AbstractHostEncFixedTensor<HostRingT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        HostShape: KnownType<S>,
        ReplicatedShape: KnownType<S>,
        HostRingT: Placed<Placement = HostPlacement>,
        ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        HostPlacement: PlacementShape<S, HostRingT, m!(HostShape)>,
        m!(HostShape): Clone,
        HostPlacement: PlacementBitDec<S, HostRingT, HostBitT>,
        HostPlacement: PlacementIndexAxis<S, HostBitT, HostBitT>,
        ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, HostBitT, RepBitT>,
        RepBitT: Clone,
        ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementAnd<S, RepBitT, RepBitT, RepBitT>,
        ReplicatedPlacement: PlacementNeg<S, RepBitT, RepBitT>,
        AbstractReplicatedShape<m!(HostShape)>: Into<m!(ReplicatedShape)>,
        ReplicatedPlacement: PlacementFill<S, m!(ReplicatedShape), RepRingT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
    {
        let host = c.tensor.placement().unwrap();
        let setup = rep.gen_setup(sess);

        let host_shape = host.shape(sess, &c.tensor);
        let shape = AbstractReplicatedShape {
            shapes: [host_shape.clone(), host_shape.clone(), host_shape],
        }
        .into(); // TODO use macro to call `into` here? or update modelled! to call into?

        // decompose into bits on host
        let c_decomposed = host.bit_decompose(sess, &c.tensor);
        let c_bits = (0..128).map(|i| host.index_axis(sess, 0, i, &c_decomposed));

        // sharing bits (could be public values but we don't have that yet)
        let c_bits_shared: Vec<_> = c_bits.map(|b| rep.share(sess, &setup, &b)).collect();

        // run AES
        let m_bits: Vec<_> = aes(sess, rep, c_bits_shared.clone(), c_bits_shared);

        // re-compose on rep
        let zero = rep.fill(sess, Constant::Ring128(0), &shape.into());
        let m = m_bits
            .iter()
            .enumerate()
            .map(|(i, b)| rep.ring_inject(sess, i, b))
            .fold(zero, |acc, x| rep.add(sess, &acc, &x));

        // wrap up as fixed-point tensor
        Ok(AbstractReplicatedFixedTensor {
            tensor: m,
            fractional_precision: c.precision,
            integral_precision: 0, // TODO
        })
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_aes_decrypt_host() {}
}
