use crate::bristol_fashion::aes;
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
        HostBitTensorT: std::fmt::Debug,
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
        let key_bits: Vec<_> = (0..128).map(|i| plc.index(sess, i, &key.0)).collect();
        println!("key_bits {:?}\n", key_bits);

        let ciphertext_bits: Vec<_> = (0..224)
            .map(|i| plc.index(sess, i, &ciphertext.tensor))
            .collect();

        let nonce_bits = &ciphertext_bits[0..96];
        assert_eq!(nonce_bits.len(), 96);
        // println!("nonce_bits {:?}\n", nonce_bits);
        let c_bits = &ciphertext_bits[96..224];
        assert_eq!(c_bits.len(), 128);
        // println!("c_bits {:?}\n", c_bits);

        let shape = plc.shape(sess, &nonce_bits[0]);
        let one_bit: HostBitTensorT = plc.fill(sess, Constant::Bit(1), &shape);
        let zero_bit: HostBitTensorT = plc.fill(sess, Constant::Bit(0), &shape);
        let mut block_bits: Vec<_> = (0..128)
            .map(|i| {
                // if i < nonce_bits.len() {
                //     nonce_bits[i].clone()
                // } else {
                //     zero_bit.clone()
                // }
                // if i >= 32 {
                //     nonce_bits[i - 32].clone()
                // } else {
                //     zero_bit.clone()
                // }
                zero_bit.clone()
            })
            .collect();
        block_bits[6] = one_bit;
        println!("block_bits {:?}\n", block_bits);

        let r_bits = aes(sess, plc, key_bits, block_bits);
        println!("r_bits {:?}\n", r_bits);
        let m_bits: Vec<_> = c_bits
            .iter()
            .zip(r_bits)
            .map(|(ci, ri)| plc.xor(sess, ci, &ri))
            .collect();
        println!("m_bits {:?}\n", m_bits);

        let zero_ring: HostRing128TensorT = plc.fill(sess, Constant::Ring128(0), &shape);
        let m = m_bits
            .iter()
            .enumerate()
            .map(|(i, b)| plc.ring_inject(sess, i, b))
            .fold(zero_ring, |acc, x| plc.add(sess, &acc, &x));

        Ok(AbstractHostFixedTensor {
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

    pub(crate) fn rep_fixed_aes_kernel<S: Session, RepRingT, HostBitArray128T, HostBitArray224T>(
        _sess: &S,
        _rep: &ReplicatedPlacement,
        _key: AbstractReplicatedAesKey<HostBitArray128T>,
        _ciphertext: AbstractHostFixedAesTensor<HostBitArray224T>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>> {
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
    use crate::host::{HostBitArray128, HostBitArray224, HostBitTensor};
    use crate::kernels::SyncSession;
    use aes::cipher::generic_array::sequence::Concat;
    use aes_gcm::{
        aead::{Aead, NewAead},
        AeadInPlace,
    };
    use ndarray::Array;

    #[test]
    fn test_aes_foo() {
        let raw_key = [3; 16];
        let raw_nonce = [177; 12];
        let raw_plaintext = [132; 16];

        let expected_ciphertext = {
            let key = aes_gcm::Key::from_slice(&raw_key);
            let nonce = aes_gcm::Nonce::from_slice(&raw_nonce);

            // plaintext initially, then ciphertext
            let mut buffer = raw_plaintext.clone();

            let cipher = aes_gcm::Aes128Gcm::new(key);
            let associated_data = vec![];
            let _tag = cipher
                .encrypt_in_place_detached(nonce, &associated_data, buffer.as_mut())
                .unwrap();

            buffer
        };

        let actual_ciphertext = {
            use aes::cipher::{generic_array::GenericArray, BlockEncrypt};
            use aes::{Aes128, Block, NewBlockCipher};

            let mut raw_block = [0_u8; 16];
            for (i, b) in raw_nonce.iter().enumerate() {
                raw_block[i] = *b;
            }
            // set counter value to 2
            raw_block[15] = 2;

            let mut block = Block::clone_from_slice(&raw_block);
            let key = GenericArray::from_slice(&raw_key);
            assert_eq!(block.len(), 16);
            assert_eq!(key.len(), 16);

            let cipher = Aes128::new(key);
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
        let raw_key = [0; 16];
        let raw_nonce = [0; 12];
        let raw_plaintext = [0; 16];

        let alice = HostPlacement {
            owner: "alice".into(),
        };

        let ciphertext: HostFixed128AesTensor = {
            let key = aes_gcm::Key::from_slice(&raw_key);
            let nonce = aes_gcm::Nonce::from_slice(&raw_nonce);

            // plaintext initially, then ciphertext
            let mut buffer = raw_plaintext.clone();

            println!("plaintext: {:?}\n", buffer);

            let cipher = aes_gcm::Aes128Gcm::new(key);
            let associated_data = vec![];
            let _tag = cipher
                .encrypt_in_place_detached(nonce, &associated_data, buffer.as_mut())
                .unwrap();

            assert_eq!(nonce.len(), 12);
            assert_eq!(buffer.len(), 16);
            let raw_ciphertext = nonce.concat(buffer.into());

            let vec = crate::bristol_fashion::byte_vec_to_bit_vec_be(raw_ciphertext.as_ref());
            // println!("vec: {:?}\n", vec);
            let array = Array::from_shape_vec((224, 1), vec).unwrap().into_dyn();
            let bit_array = HostBitArray224::from_raw_plc(array, alice.clone());

            HostFixed128AesTensor {
                precision: 0,
                tensor: bit_array,
            }
        };
        // println!("ciphertext: {:?}\n", ciphertext.tensor.0);

        let key: HostAesKey = {
            let vec = crate::bristol_fashion::byte_vec_to_bit_vec_be(raw_key.as_ref());
            let array = Array::from_shape_vec((128, 1), vec).unwrap().into_dyn();
            let bit_array = HostBitArray128::from_raw_plc(array, alice.clone());
            AbstractHostAesKey(bit_array)
        };

        let sess = SyncSession::default();
        let plaintext = alice.decrypt(&sess, &key, &ciphertext);
        println!("decrypted plaintext {:?}\n", plaintext.tensor.0);

        // let mut ct_bits: Vec<u8>;

        // let inner_ct_bits: Vec<Vec<u8>> = ciphertext.iter().enumerate().map(|(i, item)| {
        //     let inner: Vec<u8> = (0..8).map(|j| {
        //         (item >> j) & 1
        //     }).collect();
        //     inner
        // }).collect();

        // let mut ct_bits: Vec<u8> = inner_ct_bits[0].clone();
        // for i in 1..16 {
        //     ct_bits.extend_from_slice(inner_ct_bits[i].as_slice())
        // }

        // let ct_bit_ring: Vec<HostRing128Tensor> = (0..128).map(|i| {
        //     let bit: HostBitTensor = alice.fill(&sess, Constant::Bit(ct_bits[i]), &HostShape(vec![1], alice));
        //     alice.ring_inject(&sess, i, &bit);
        // }).collect();

        // let ct_ring_ten = alice.add_n(&sess, &ct_bit_ring);
        // println!("ct_bits: {:?}", ct_bits);
        // println!("total: {:?}", ct_ring_ten);

        // assert_eq!(false, true);

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
