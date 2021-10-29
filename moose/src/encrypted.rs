use crate::bristol_fashion::aes;
use crate::computation::*;
use crate::error::Result;
use crate::fixedpoint::{Fixed128Tensor, FixedTensor};
use crate::host::{
    AbstractHostFixedAesTensor, AbstractHostFixedTensor, HostAesKey, HostBitTensor,
    HostFixed128AesTensor, HostFixed128Tensor, HostRing128Tensor, HostShape,
};
use crate::kernels::{
    PlacementAdd, PlacementAnd, PlacementBitDec, PlacementDecrypt, PlacementFill,
    PlacementIndexAxis, PlacementNeg, PlacementRingInject, PlacementSetupGen, PlacementShape,
    PlacementShareSetup, PlacementXor, Session,
};
use crate::logical::{AbstractTensor, Tensor};
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
        (HostPlacement, (HostAesKey, Fixed128AesTensor) -> Fixed128Tensor => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostAesKey, HostFixed128AesTensor) -> HostFixed128Tensor => [hybrid] Self::host_fixed_aes_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, AesTensor) -> Tensor => [hybrid] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, Fixed128AesTensor) -> Fixed128Tensor => [hybrid] Self::rep_fixed_kernel),
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

    pub(crate) fn host_fixed_kernel<
        S: Session,
        HostAesKeyT,
        HostFixed128AesT,
        HostFixed128T,
        ReplicatedFixed128T,
    >(
        sess: &S,
        plc: &HostPlacement,
        key: HostAesKeyT,
        ciphertext: FixedAesTensor<HostFixed128AesT>,
    ) -> Result<FixedTensor<HostFixed128T, ReplicatedFixed128T>>
    where
        HostPlacement: PlacementDecrypt<S, HostAesKeyT, HostFixed128AesT, HostFixed128T>,
    {
        match ciphertext {
            FixedAesTensor::Host(c) => {
                let x = plc.decrypt(sess, &key, &c);
                Ok(FixedTensor::Host(x))
            }
        }
    }

    pub(crate) fn host_fixed_aes_kernel<S: Session, HostAesKeyT, HostBitArrayT, HostRing128T>(
        sess: &S,
        plc: &HostPlacement,
        key: HostAesKeyT,
        ciphertext: AbstractHostFixedAesTensor<HostBitArrayT>,
    ) -> Result<AbstractHostFixedTensor<HostRing128T>>
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
    pub(crate) fn rep_kernel<
        S: Session,
        ReplicatedAesKeyT,
        Fixed128AesT,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        key: ReplicatedAesKeyT,
        ciphertext: AbstractAesTensor<Fixed128AesT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>>
    where
        ReplicatedPlacement: PlacementDecrypt<S, ReplicatedAesKeyT, Fixed128AesT, Fixed128T>,
    {
        match ciphertext {
            AbstractAesTensor::Fixed128(c) => {
                let x = plc.decrypt(sess, &key, &c);
                Ok(AbstractTensor::Fixed128(x))
            }
        }
    }

    pub(crate) fn rep_fixed_kernel<
        S: Session,
        ReplicatedAesKeyT,
        HostFixed128AesT,
        HostFixed128T,
        ReplicatedFixed128T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        key: ReplicatedAesKeyT,
        ciphertext: FixedAesTensor<HostFixed128AesT>,
    ) -> Result<FixedTensor<HostFixed128T, ReplicatedFixed128T>>
    where
        ReplicatedPlacement:
            PlacementDecrypt<S, ReplicatedAesKeyT, HostFixed128AesT, ReplicatedFixed128T>,
    {
        match ciphertext {
            FixedAesTensor::Host(c) => {
                let x = plc.decrypt(sess, &key, &c);
                Ok(FixedTensor::Replicated(x))
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
