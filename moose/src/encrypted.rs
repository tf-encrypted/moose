use crate::bristol_fashion::aes;
use crate::computation::*;
use crate::error::Result;
use crate::fixedpoint::Fixed128Tensor;
use crate::host::{
    AbstractHostFixedAesTensor, HostAesKey, HostBitTensor, HostFixed128AesTensor,
    HostFixed128Tensor, HostRing128Tensor, HostShape,
};
use crate::kernels::{
    PlacementAdd, PlacementAnd, PlacementBitDec, PlacementDecrypt, PlacementFill,
    PlacementIndexAxis, PlacementNeg, PlacementRingInject, PlacementSetupGen, PlacementShape,
    PlacementShareSetup, PlacementXor, Session,
};
use crate::logical::Tensor;
use crate::replicated::{
    aes::AbstractReplicatedAesKey, aes::ReplicatedAesKey, AbstractReplicatedFixedTensor,
    AbstractReplicatedShape, ReplicatedFixed128Tensor, ReplicatedShape,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FixedAesTensor<HostFixedAesT> {
    Host(HostFixedAesT),
}

moose_type!(Fixed128AesTensor = FixedAesTensor<HostFixed128AesTensor>);

impl<HostFixedAesT> Placed for FixedAesTensor<HostFixedAesT>
where
    HostFixedAesT: Placed,
    HostFixedAesT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            FixedAesTensor::Host(x) => Ok(x.placement()?.into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AbstractAesTensor<Fixed128AesT> {
    Fixed128(Fixed128AesT),
}

moose_type!(AesTensor = AbstractAesTensor<Fixed128AesTensor>);

impl<Fixed128T> Placed for AbstractAesTensor<Fixed128T>
where
    Fixed128T: Placed,
    Fixed128T::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            AbstractAesTensor::Fixed128(x) => Ok(x.placement()?.into()),
        }
    }
}

modelled!(PlacementDecrypt::decrypt, HostPlacement, (HostAesKey, AesTensor) -> Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, HostPlacement, (HostAesKey, Fixed128AesTensor) -> Fixed128Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, HostPlacement, (HostAesKey, HostFixed128AesTensor) -> HostFixed128Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, ReplicatedPlacement, (ReplicatedAesKey, AesTensor) -> Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, ReplicatedPlacement, (ReplicatedAesKey, Fixed128AesTensor) -> Fixed128Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, ReplicatedPlacement, (ReplicatedAesKey, HostFixed128AesTensor) -> ReplicatedFixed128Tensor, AesDecryptOp);

kernel! {
    AesDecryptOp,
    [
        (HostPlacement, (HostAesKey, AesTensor) -> Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostAesKey, Fixed128AesTensor) -> Fixed128Tensor => [runtime] Self::host_fixed_kernel),
        (HostPlacement, (HostAesKey, HostFixed128AesTensor) -> HostFixed128Tensor => [runtime] Self::host_fixed_aes_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, AesTensor) -> Tensor => [runtime] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, Fixed128AesTensor) -> Fixed128Tensor => [runtime] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, HostFixed128AesTensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_fixed_aes_kernel),
    ]
}

impl AesDecryptOp {
    pub(crate) fn host_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        key: HostAesKey,
        ciphertext: AesTensor,
    ) -> Result<Tensor>
    where
        HostPlacement: PlacementDecrypt<S, HostAesKey, Fixed128AesTensor, Fixed128Tensor>,
    {
        match ciphertext {
            AesTensor::Fixed128(c) => {
                let x = plc.decrypt(sess, &key, &c);
                Ok(Tensor::Fixed128(x))
            }
        }
    }

    pub(crate) fn host_fixed_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        key: HostAesKey,
        ciphertext: Fixed128AesTensor,
    ) -> Result<Fixed128Tensor>
    where
        HostPlacement: PlacementDecrypt<S, HostAesKey, HostFixed128AesTensor, HostFixed128Tensor>,
    {
        match ciphertext {
            Fixed128AesTensor::Host(c) => {
                let x = plc.decrypt(sess, &key, &c);
                Ok(Fixed128Tensor::Host(x))
            }
        }
    }

    pub(crate) fn host_fixed_aes_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        key: HostAesKey,
        ciphertext: HostFixed128AesTensor,
    ) -> Result<HostFixed128Tensor>
where
        // HostPlacement: PlacementBitDec<S, HostRing128Tensor, HostBitTensor>,
        // HostPlacement: PlacementIndexAxis<S, HostBitTensor, HostBitTensor>,
        // HostPlacement: PlacementRingInject<S, HostBitTensor, HostRing128Tensor>,
        // HostPlacement: PlacementShape<S, HostFixed128AesTensor, HostShape>,
        // HostPlacement: PlacementFill<S, HostShape, HostRing128Tensor>,
        // HostPlacement: PlacementAdd<S, HostRing128Tensor, HostRing128Tensor, HostRing128Tensor>,
        // HostPlacement: PlacementXor<S, HostBitTensor, HostBitTensor, HostBitTensor>,
        // HostPlacement: PlacementAnd<S, HostBitTensor, HostBitTensor, HostBitTensor>,
        // HostPlacement: PlacementNeg<S, HostBitTensor, HostBitTensor>,
    {
        unimplemented!()

        // let shape = plc.shape(sess, &c);
        // // let c_decomposed = plc.bit_decompose(sess, &c.tensor);

        // let m_bits = aes(sess, plc, c_bits.clone(), c_bits);
        // let zero = plc.fill(sess, Constant::Ring128(0), &shape);
        // let m = m_bits
        //     .iter()
        //     .enumerate()
        //     .map(|(i, b)| plc.ring_inject(sess, i, b))
        //     .fold(zero, |acc, x| plc.add(sess, &acc, &x));

        // Ok(HostFixed128Tensor {
        //     tensor: m,
        //     fractional_precision: c.precision,
        //     integral_precision: 0, // TODO
        // })
    }
}

impl AesDecryptOp {
    pub(crate) fn rep_kernel<S: Session>(
        sess: &S,
        plc: &ReplicatedPlacement,
        key: ReplicatedAesKey,
        ciphertext: AesTensor,
    ) -> Result<Tensor>
    where
        ReplicatedPlacement:
            PlacementDecrypt<S, ReplicatedAesKey, Fixed128AesTensor, Fixed128Tensor>,
    {
        match ciphertext {
            AesTensor::Fixed128(c) => {
                let x = plc.decrypt(sess, &key, &c);
                Ok(Tensor::Fixed128(x))
            }
        }
    }

    pub(crate) fn rep_fixed_kernel<S: Session>(
        sess: &S,
        plc: &ReplicatedPlacement,
        key: ReplicatedAesKey,
        ciphertext: Fixed128AesTensor,
    ) -> Result<Fixed128Tensor>
    where
        ReplicatedPlacement:
            PlacementDecrypt<S, ReplicatedAesKey, HostFixed128AesTensor, ReplicatedFixed128Tensor>,
    {
        match ciphertext {
            Fixed128AesTensor::Host(c) => {
                let x = plc.decrypt(sess, &key, &c);
                Ok(Fixed128Tensor::Replicated(x))
            }
        }
    }

    pub(crate) fn rep_fixed_aes_kernel<S: Session, RepRingT, HostBitArray128T, HostBitArray256T>(
        sess: &S,
        rep: &ReplicatedPlacement,
        key: AbstractReplicatedAesKey<HostBitArray128T>,
        ciphertext: AbstractHostFixedAesTensor<HostBitArray256T>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
where
        // HostShape: KnownType<S>,
        // ReplicatedShape: KnownType<S>,
        // HostBitArrayT: Placed<Placement = HostPlacement>,
        // ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
        // HostPlacement: PlacementShape<S, HostBitArrayT, m!(HostShape)>,
        // m!(HostShape): Clone,
        // HostPlacement: PlacementBitDec<S, HostBitArrayT, HostBitT>,
        // HostPlacement: PlacementIndexAxis<S, HostBitT, HostBitT>,
        // ReplicatedPlacement: PlacementShareSetup<S, S::ReplicatedSetup, HostBitT, RepBitT>,
        // RepBitT: Clone,
        // ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
        // ReplicatedPlacement: PlacementAnd<S, RepBitT, RepBitT, RepBitT>,
        // ReplicatedPlacement: PlacementNeg<S, RepBitT, RepBitT>,
        // AbstractReplicatedShape<m!(HostShape)>: Into<m!(ReplicatedShape)>,
        // ReplicatedPlacement: PlacementFill<S, m!(ReplicatedShape), RepRingT>,
        // ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
        // ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
    {
        unimplemented!()

        // let host = c.tensor.placement().unwrap();
        // let setup = rep.gen_setup(sess);

        // let host_shape = host.shape(sess, &c.tensor);
        // let shape = AbstractReplicatedShape {
        //     shapes: [host_shape.clone(), host_shape.clone(), host_shape],
        // }
        // .into(); // TODO use macro to call `into` here? or update modelled! to call into?

        // // decompose into bits on host
        // let c_decomposed = host.bit_decompose(sess, &c.tensor);
        // let c_bits = (0..128).map(|i| host.index_axis(sess, 0, i, &c_decomposed));

        // // sharing bits (could be public values but we don't have that yet)
        // let c_bits_shared: Vec<_> = c_bits.map(|b| rep.share(sess, &setup, &b)).collect();

        // // run AES
        // let m_bits: Vec<_> = aes(sess, rep, c_bits_shared.clone(), c_bits_shared);

        // // re-compose on rep
        // let zero = rep.fill(sess, Constant::Ring128(0), &shape);
        // let m = m_bits
        //     .iter()
        //     .enumerate()
        //     .map(|(i, b)| rep.ring_inject(sess, i, b))
        //     .fold(zero, |acc, x| rep.add(sess, &acc, &x));

        // // wrap up as fixed-point tensor
        // Ok(AbstractReplicatedFixedTensor {
        //     tensor: m,
        //     fractional_precision: c.precision,
        //     integral_precision: 0, // TODO
        // })
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_aes_decrypt_host() {}
}
