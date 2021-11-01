use crate::computation::*;
use crate::error::Result;
use crate::fixedpoint::{Fixed128Tensor, FixedTensor};
use crate::host::{AbstractHostAesKey, AbstractHostFixedAesTensor, AbstractHostFixedTensor, HostAesKey, HostBitArray128, HostBitTensor, HostFixed128AesTensor, HostFixed128Tensor, HostRing128Tensor, HostShape, PlacementToVec};
use crate::kernels::{PlacementAdd, PlacementAnd, PlacementBitDec, PlacementDecrypt, PlacementFill, PlacementIndex, PlacementIndexAxis, PlacementNeg, PlacementRingInject, PlacementShape, PlacementXor, Session};
use crate::logical::{AbstractTensor, Tensor};
use crate::replicated::{
    aes::AbstractReplicatedAesKey, aes::ReplicatedAesKey, AbstractReplicatedFixedTensor,
    ReplicatedFixed128Tensor,
};
use serde::{Deserialize, Serialize};
use crate::bristol_fashion::aes;

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
        (HostPlacement, (HostAesKey, AesTensor) -> Tensor => [hybrid] Self::host_kernel),
        (HostPlacement, (HostAesKey, Fixed128AesTensor) -> Fixed128Tensor => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostAesKey, HostFixed128AesTensor) -> HostFixed128Tensor => [hybrid] Self::host_fixed_aes_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, AesTensor) -> Tensor => [hybrid] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, Fixed128AesTensor) -> Fixed128Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, HostFixed128AesTensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_fixed_aes_kernel),
    ]
}

impl AesDecryptOp {
    pub(crate) fn host_kernel<S: Session, Fixed128AesT, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        key: m!(HostAesKey),
        ciphertext: AbstractAesTensor<Fixed128AesT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>>
    where
        HostAesKey: KnownType<S>,
        HostPlacement: PlacementDecrypt<S, m!(HostAesKey), Fixed128AesT, Fixed128T>,
    {
        match ciphertext {
            AbstractAesTensor::Fixed128(c) => {
                let x = plc.decrypt(sess, &key, &c);
                Ok(AbstractTensor::Fixed128(x))
            }
        }
    }

    pub(crate) fn host_fixed_kernel<
        S: Session,
        HostFixed128AesT,
        HostFixed128T,
        ReplicatedFixed128T,
    >(
        sess: &S,
        plc: &HostPlacement,
        key: m!(HostAesKey),
        ciphertext: FixedAesTensor<HostFixed128AesT>,
    ) -> Result<FixedTensor<HostFixed128T, ReplicatedFixed128T>>
    where
        HostAesKey: KnownType<S>,
        HostPlacement: PlacementDecrypt<S, m!(HostAesKey), HostFixed128AesT, HostFixed128T>,
    {
        match ciphertext {
            FixedAesTensor::Host(c) => {
                let x = plc.decrypt(sess, &key, &c);
                Ok(FixedTensor::Host(x))
            }
        }
    }

    pub(crate) fn host_fixed_aes_kernel<S: Session, HostBitArrayT>(
        sess: &S,
        plc: &HostPlacement,
        key: AbstractHostAesKey<HostBitArrayT>,
        ciphertext: AbstractHostFixedAesTensor<HostBitArrayT>,
    ) -> Result<AbstractHostFixedTensor<HostRing128Tensor>>
    where
        HostPlacement: PlacementToVec<S, HostBitArrayT, Item=HostBitTensor>,
        HostPlacement: PlacementShape<S, HostBitTensor, HostShape>,
        HostPlacement: PlacementFill<S, HostShape, HostBitTensor>,
        HostPlacement: PlacementFill<S, HostShape, HostRing128Tensor>,
        HostPlacement: PlacementXor<S, HostBitTensor, HostBitTensor, HostBitTensor>,
        HostPlacement: PlacementAnd<S, HostBitTensor, HostBitTensor, HostBitTensor>,
        HostPlacement: PlacementNeg<S, HostBitTensor, HostBitTensor>,
        HostPlacement: PlacementRingInject<S, HostBitTensor, HostRing128Tensor>,
        HostPlacement: PlacementAdd<S, HostRing128Tensor, HostRing128Tensor, HostRing128Tensor>,
        HostPlacement: PlacementAdd<S, HostRing128Tensor, HostRing128Tensor, HostRing128Tensor>,
    {
        let c_bits: Vec<HostBitTensor> = plc.to_vec(sess, &ciphertext.tensor);
        assert_eq!(c_bits.len(), 96 + 128);
        let k_bits: Vec<HostBitTensor> = plc.to_vec(sess, &key.0);
        assert_eq!(k_bits.len(), 128);

        let iv_bits = &c_bits[0..96];
        let encrypted_bits = &c_bits[96..224];

        let shape = plc.shape(sess, &iv_bits[0]);
        let bit_one: HostBitTensor = plc.fill(sess, Constant::Bit(1), &shape);
        let bit_zero: HostBitTensor = plc.fill(sess, Constant::Bit(0), &shape);
        let extended_iv: Vec<_> = (0..128).map(|i| {
            if i < 96 {
                iv_bits[i].clone()
            } else if i < 126 {
                bit_zero.clone()
            } else {
                bit_one.clone()
            }
        }).collect();

        let r_bits = aes(sess, plc, k_bits, extended_iv);
        let m_bits: Vec<_> = (0..128).map(|i| plc.xor(sess, &r_bits[i], &c_bits[i])).collect();

        // println!("m_bits: {:?}", m_bits);

        // TODO use BitCompose instead?
        let zero = plc.fill(sess, Constant::Ring128(0), &shape);
        let m = m_bits
            .iter()
            .enumerate()
            .map(|(i, b)| plc.ring_inject(sess, i, b))
            .fold(zero, |acc, x| plc.add(sess, &acc, &x));

        Ok(HostFixed128Tensor {
            tensor: m,
            fractional_precision: ciphertext.precision,
            integral_precision: 0, // TODO
        })
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
        _sess: &S,
        _rep: &ReplicatedPlacement,
        _key: AbstractReplicatedAesKey<HostBitArray128T>,
        _ciphertext: AbstractHostFixedAesTensor<HostBitArray256T>,
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
    use super::*;
    use aes_gcm::aead::{Aead, NewAead};
    use aes_gcm::{Aes128Gcm, Key, Nonce};
    // use crate::computation::HostPlacement;
    use crate::kernels::SyncSession;

    // use crate::host::{HostBitTensor, HostBitArray128};


    #[test]
    fn test_aes_decrypt_host() {

        let key = Key::from_slice(b"0000000000000000");
        let cipher = Aes128Gcm::new(key);
        let nonce = Nonce::from_slice(b"000000000000"); // 96-bits; unique per message

        let ciphertext = cipher
            .encrypt(nonce, b"0000000000000000".as_ref())
            .expect("encryption failure!"); // NOTE: handle this error to avoid panics!

        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let sess = SyncSession::default();
        let mut ct_bits: Vec<u8>;

        let inner_ct_bits: Vec<Vec<u8>> = ciphertext.iter().enumerate().map(|(i, item)| {
            let inner: Vec<u8> = (0..8).map(|j| {
                (item >> j) & 1
            }).collect();
            inner
        }).collect();

        let mut ct_bits: Vec<u8> = inner_ct_bits[0].clone();
        for i in 1..16 {
            ct_bits.extend_from_slice(inner_ct_bits[i].as_slice())
        }

        let ct_bit_ring: Vec<HostRing128Tensor> = (0..128).map(|i| {
            let bit: HostBitTensor = alice.fill(&sess, Constant::Bit(ct_bits[i]), &HostShape(vec![1], alice));
            alice.ring_inject(&sess, i, &bit);
        }).collect();

        let ct_ring_ten = alice.add_n(&sess, &ct_bit_ring);
        println!("ct_bits: {:?}", ct_bits);
        println!("total: {:?}", ct_ring_ten);

        assert_eq!(false, true);


        // let ct_ten = HostBitTensor::from_vec_plc(ciphertext, alice);
        // let key_ten = HostBitArray128::from_vec_plc(key_bits, alice);

        // let enc_fixed = HostFixed128Tensor {
        //     precision: 40,
        //     tensor: 
        // }
        // let plaintext = alice.decrypt(&sess, ciphertext, key_ten);
        // // let plaintext = cipher
        //     .decrypt(nonce, ciphertext.as_ref())
        //     .expect("decryption failure!"); // NOTE: handle this error to avoid panics!
        // assert_eq!(&plaintext, b"0000000000000000");

    }
}
