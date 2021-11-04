use crate::error::Result;
use crate::fixedpoint::{Fixed128Tensor, FixedTensor};
use crate::host::{
    AbstractHostAesKey, AbstractHostFixedAesTensor, AbstractHostFixedTensor, HostAesKey,
    HostFixed128AesTensor, HostFixed128Tensor,
};
use crate::kernels::{
    PlacementAdd, PlacementAnd, PlacementDecrypt, PlacementFill, PlacementIndex, PlacementNeg,
    PlacementRingInject, PlacementShape, PlacementXor, Session,
};
use crate::logical::{AbstractTensor, Tensor};
use crate::replicated::{
    aes::AbstractReplicatedAesKey, aes::ReplicatedAesKey, AbstractReplicatedFixedTensor,
    ReplicatedFixed128Tensor,
};
use crate::{computation::*, BitArray, N128, N224};
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
        (HostPlacement, (HostAesKey, AesTensor) -> Tensor => [hybrid] Self::host_kernel),
        (HostPlacement, (HostAesKey, Fixed128AesTensor) -> Fixed128Tensor => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostAesKey, HostFixed128AesTensor) -> HostFixed128Tensor => [hybrid] Self::host_fixed_aes_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, AesTensor) -> Tensor => [hybrid] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, Fixed128AesTensor) -> Fixed128Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, HostFixed128AesTensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_fixed_aes_kernel),
    ]
}

fn aesgcm<S: Session, P, BitTensorT, ShapeT>(
    sess: &S,
    plc: &P,
    key_bits: Vec<BitTensorT>,
    ciphertext_bits: Vec<BitTensorT>,
) -> Vec<BitTensorT>
where
    BitTensorT: Clone,
    P: PlacementShape<S, BitTensorT, ShapeT>,
    P: PlacementFill<S, ShapeT, BitTensorT>,
    P: PlacementXor<S, BitTensorT, BitTensorT, BitTensorT>,
    P: PlacementAnd<S, BitTensorT, BitTensorT, BitTensorT>,
    P: PlacementNeg<S, BitTensorT, BitTensorT>,
{
    assert_eq!(key_bits.len(), 128);
    assert_eq!(ciphertext_bits.len(), 96 + 128);

    // separate into nonce bits and masked plaintext
    let nonce_bits = &ciphertext_bits[0..96];
    let rm_bits = &ciphertext_bits[96..224];
    assert_eq!(nonce_bits.len(), 96);
    assert_eq!(rm_bits.len(), 128);

    // build full AES block with nonce and counter value of 2
    let shape = plc.shape(sess, &nonce_bits[0]);
    let one_bit: BitTensorT = plc.fill(sess, Constant::Bit(1), &shape);
    let zero_bit: BitTensorT = plc.fill(sess, Constant::Bit(0), &shape);
    let mut block_bits: Vec<_> = (0..128)
        .map(|i| {
            if i < nonce_bits.len() {
                nonce_bits[i].clone()
            } else {
                zero_bit.clone()
            }
        })
        .collect();
    block_bits[128 - 2] = one_bit;

    // apply AES to block to get mask
    let r_bits = crate::bristol_fashion::aes(sess, plc, key_bits, block_bits);

    // remove mask to recover plaintext
    let m_bits: Vec<BitTensorT> = rm_bits
        .iter()
        .zip(r_bits)
        .map(|(ci, ri)| plc.xor(sess, ci, &ri))
        .collect();

    m_bits
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

    pub(crate) fn host_fixed_aes_kernel<
        S: Session,
        KeyT,
        CiphertextT,
        ShapeT,
        HostRing128TensorT,
        HostBitTensorT,
    >(
        sess: &S,
        plc: &HostPlacement,
        key: AbstractHostAesKey<KeyT>,
        ciphertext: AbstractHostFixedAesTensor<CiphertextT>,
    ) -> Result<AbstractHostFixedTensor<HostRing128TensorT>>
    where
        KeyT: BitArray<Len = N128>,
        CiphertextT: BitArray<Len = N224>,
        HostBitTensorT: Clone,
        HostPlacement: PlacementIndex<S, KeyT, HostBitTensorT>,
        HostPlacement: PlacementIndex<S, CiphertextT, HostBitTensorT>,
        HostPlacement: PlacementShape<S, HostBitTensorT, ShapeT>,
        HostPlacement: PlacementFill<S, ShapeT, HostBitTensorT>,
        HostPlacement: PlacementRingInject<S, HostBitTensorT, HostRing128TensorT>,
        HostPlacement: PlacementFill<S, ShapeT, HostRing128TensorT>,
        HostPlacement: PlacementAdd<S, HostRing128TensorT, HostRing128TensorT, HostRing128TensorT>,
        HostPlacement: PlacementXor<S, HostBitTensorT, HostBitTensorT, HostBitTensorT>,
        HostPlacement: PlacementAnd<S, HostBitTensorT, HostBitTensorT, HostBitTensorT>,
        HostPlacement: PlacementNeg<S, HostBitTensorT, HostBitTensorT>,
    {
        // turn inputs into vectors
        let key_bits: Vec<HostBitTensorT> = (0..128).map(|i| plc.index(sess, i, &key.0)).collect();
        let ciphertext_bits: Vec<HostBitTensorT> = (0..224)
            .map(|i| plc.index(sess, i, &ciphertext.tensor))
            .collect();

        // perform AES-GCM decryption
        let m_bits = aesgcm(sess, plc, key_bits, ciphertext_bits);

        // bit compose plaintext to obtain ring values
        let shape = plc.shape(sess, &m_bits[0]);
        let zero_ring: HostRing128TensorT = plc.fill(sess, Constant::Ring128(0), &shape);
        let m = m_bits
            .iter()
            .enumerate()
            .map(|(i, b)| plc.ring_inject(sess, 127 - i, b))
            .fold(zero_ring, |acc, x| plc.add(sess, &acc, &x));

        Ok(AbstractHostFixedTensor {
            tensor: m,
            integral_precision: ciphertext.integral_precision,
            fractional_precision: ciphertext.fractional_precision,
        })
    }

    pub(crate) fn rep_fixed_aes_kernel<S: Session, KeyT, CiphertextT, RepRingT>(
        _sess: &S,
        _plc: &ReplicatedPlacement,
        _key: AbstractReplicatedAesKey<KeyT>,
        _ciphertext: AbstractHostFixedAesTensor<CiphertextT>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        KeyT: BitArray<Len = N128>,
        CiphertextT: BitArray<Len = N224>,
        //     HostBitTensorT: Clone,
        // ReplicatedPlacement: PlacementIndex<S, KeyT, RepBitTensorT>,
        //     ReplicatedPlacement: PlacementIndex<S, CiphertextT, HostBitTensorT>,
        //     ReplicatedPlacement: PlacementShape<S, HostBitTensorT, ShapeT>,
        //     ReplicatedPlacement: PlacementFill<S, ShapeT, HostBitTensorT>,
        //     ReplicatedPlacement: PlacementRingInject<S, HostBitTensorT, HostRing128TensorT>,
        //     ReplicatedPlacement: PlacementFill<S, ShapeT, HostRing128TensorT>,
        //     ReplicatedPlacement: PlacementAdd<S, HostRing128TensorT, HostRing128TensorT, HostRing128TensorT>,
        //     ReplicatedPlacement: PlacementXor<S, HostBitTensorT, HostBitTensorT, HostBitTensorT>,
        //     ReplicatedPlacement: PlacementAnd<S, HostBitTensorT, HostBitTensorT, HostBitTensorT>,
        //     ReplicatedPlacement: PlacementNeg<S, HostBitTensorT, HostBitTensorT>,
    {
        // // turn inputs into vectors
        // let key_bits: Vec<HostBitTensorT> = (0..128).map(|i| plc.index(sess, i, &key.0)).collect();
        // let ciphertext_bits: Vec<HostBitTensorT> = (0..224)
        //     .map(|i| plc.index(sess, i, &ciphertext.tensor))
        //     .collect();

        //     // perform AES-GCM decryption
        //     let m_bits = aesgcm(sess, plc, key_bits, ciphertext_bits);

        //     // bit compose plaintext to obtain ring values
        //     let shape = plc.shape(sess, &m_bits[0]);
        //     let zero_ring: HostRing128TensorT = plc.fill(sess, Constant::Ring128(0), &shape);
        //     let m = m_bits
        //         .iter()
        //         .enumerate()
        //         .map(|(i, b)| plc.ring_inject(sess, i, b))
        //         .fold(zero_ring, |acc, x| plc.add(sess, &acc, &x));

        //     Ok(AbstractHostFixedTensor {
        //         tensor: m,
        //         integral_precision: ciphertext.integral_precision,
        //         fractional_precision: ciphertext.fractional_precision,
        //     })
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{HostBitArray128, HostBitArray224};
    use crate::kernels::SyncSession;
    use aes::cipher::generic_array::sequence::Concat;
    use aes_gcm::{aead::NewAead, AeadInPlace};
    use ndarray::Array;

    #[test]
    fn test_aes_aesgcm() {
        let raw_key = [3; 16];
        let raw_nonce = [177; 12];
        let raw_plaintext = [132; 16];

        let expected_ciphertext = {
            let key = aes_gcm::Key::from_slice(&raw_key);
            let nonce = aes_gcm::Nonce::from_slice(&raw_nonce);

            // plaintext initially, then ciphertext
            let mut buffer = raw_plaintext;

            let cipher = aes_gcm::Aes128Gcm::new(key);
            let associated_data = vec![];
            let _tag = cipher
                .encrypt_in_place_detached(nonce, &associated_data, buffer.as_mut())
                .unwrap();

            buffer
        };

        let actual_ciphertext = {
            use aes::cipher::{generic_array::GenericArray, BlockEncrypt};
            use aes::NewBlockCipher;

            let mut raw_block = [0_u8; 16];
            // fill first bytes with nonce
            for (i, b) in raw_nonce.iter().enumerate() {
                raw_block[i] = *b;
            }
            // set counter to 2
            raw_block[15] = 2;

            let mut block = aes::Block::clone_from_slice(&raw_block);
            let key = GenericArray::from_slice(&raw_key);
            assert_eq!(block.len(), 16);
            assert_eq!(key.len(), 16);

            let cipher = aes::Aes128::new(key);
            cipher.encrypt_block(&mut block);

            block
                .iter()
                .zip(raw_plaintext)
                .map(|(r, m)| r ^ m)
                .collect::<Vec<_>>()
        };

        assert_eq!(&actual_ciphertext, &expected_ciphertext);
    }

    #[test]
    fn test_aes_decrypt_host() {
        let raw_key = [201; 16];
        let raw_nonce = [177; 12];
        let raw_plaintext = [132; 16];

        let alice = HostPlacement {
            owner: "alice".into(),
        };

        let ciphertext: HostFixed128AesTensor = {
            let key = aes_gcm::Key::from_slice(&raw_key);
            let nonce = aes_gcm::Nonce::from_slice(&raw_nonce);

            // plaintext initially, then ciphertext
            let mut buffer = raw_plaintext;

            let cipher = aes_gcm::Aes128Gcm::new(key);
            let associated_data = vec![];
            let _tag = cipher
                .encrypt_in_place_detached(nonce, &associated_data, buffer.as_mut())
                .unwrap();

            assert_eq!(nonce.len(), 12);
            assert_eq!(buffer.len(), 16);
            let raw_ciphertext = nonce.concat(buffer.into());

            let vec = crate::bristol_fashion::byte_vec_to_bit_vec_be(raw_ciphertext.as_ref());
            let array = Array::from_shape_vec((224, 1), vec).unwrap().into_dyn();
            let bit_array = HostBitArray224::from_raw_plc(array, alice.clone());

            HostFixed128AesTensor {
                integral_precision: 10,
                fractional_precision: 0,
                tensor: bit_array,
            }
        };

        let key: HostAesKey = {
            let vec = crate::bristol_fashion::byte_vec_to_bit_vec_be(raw_key.as_ref());
            let array = Array::from_shape_vec((128, 1), vec).unwrap().into_dyn();
            let bit_array = HostBitArray128::from_raw_plc(array, alice.clone());
            AbstractHostAesKey(bit_array)
        };

        let sess = SyncSession::default();
        let plaintext = alice.decrypt(&sess, &key, &ciphertext);

        let actual_plaintext = plaintext.tensor.0[0].0;
        let expected_plaintext = u128::from_be_bytes(raw_plaintext);
        assert_eq!(actual_plaintext, expected_plaintext);
    }
}
