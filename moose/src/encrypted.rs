use crate::error::Result;
use crate::fixedpoint::{Fixed128Tensor, FixedTensor};
use crate::host::{
    AbstractHostAesKey, AbstractHostFixedAesTensor, AbstractHostFixedTensor, HostAesKey,
    HostFixed128AesTensor, HostFixed128Tensor,
};
use crate::kernels::{
    PlacementAdd, PlacementAnd, PlacementDecrypt, PlacementFill, PlacementIndex, PlacementInput,
    PlacementNeg, PlacementReveal, PlacementRingInject, PlacementShape, PlacementShare,
    PlacementXor, Session,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AbstractAesKey<HostKeyT, RepKeyT> {
    Host(HostKeyT),
    Replicated(RepKeyT),
}

moose_type!(AesKey = AbstractAesKey<HostAesKey, ReplicatedAesKey>);

impl<HostKeyT, RepKeyT> Placed for AbstractAesKey<HostKeyT, RepKeyT>
where
    HostKeyT: Placed,
    HostKeyT::Placement: Into<Placement>,
    RepKeyT: Placed,
    RepKeyT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            AbstractAesKey::Host(x) => Ok(x.placement()?.into()),
            AbstractAesKey::Replicated(x) => Ok(x.placement()?.into()),
        }
    }
}

impl InputOp {
    pub(crate) fn aestensor<S: Session, Fixed128AesTensorT>(
        sess: &S,
        plc: &HostPlacement,
        arg_name: String,
    ) -> Result<AbstractAesTensor<Fixed128AesTensorT>>
    where
        HostPlacement: PlacementInput<S, Fixed128AesTensorT>,
    {
        let tensor = plc.input(sess, arg_name);
        Ok(AbstractAesTensor::Fixed128(tensor))
    }

    pub(crate) fn fixed_aestensor<S: Session, HostFixed128AesTensorT>(
        sess: &S,
        plc: &HostPlacement,
        arg_name: String,
    ) -> Result<FixedAesTensor<HostFixed128AesTensorT>>
    where
        HostPlacement: PlacementInput<S, HostFixed128AesTensorT>,
    {
        let tensor = plc.input(sess, arg_name);
        Ok(FixedAesTensor::Host(tensor))
    }

    pub(crate) fn host_fixed_aestensor<S: Session, HostBitArrayT>(
        sess: &S,
        plc: &HostPlacement,
        _sig: Signature,
        arg_name: String,
    ) -> Result<AbstractHostFixedAesTensor<HostBitArrayT>>
    where
        HostPlacement: PlacementInput<S, HostBitArrayT>,
    {
        let tensor = plc.input(sess, arg_name);
        Ok(AbstractHostFixedAesTensor {
            tensor,
            // TODO(Morten) extract precision from sig
            integral_precision: 46,
            fractional_precision: 40,
        })
    }

    pub(crate) fn aes_kernel_on_host<S: Session, HostAesKeyT, RepAesKeyT>(
        sess: &S,
        plc: &HostPlacement,
        arg_name: String,
    ) -> Result<AbstractAesKey<HostAesKeyT, RepAesKeyT>>
    where
        HostPlacement: PlacementInput<S, HostAesKeyT>,
    {
        let key = plc.input(sess, arg_name);
        Ok(AbstractAesKey::Host(key))
    }

    pub(crate) fn host_aes_kernel<S: Session, HostBitArrayT>(
        sess: &S,
        plc: &HostPlacement,
        arg_name: String,
    ) -> Result<AbstractHostAesKey<HostBitArrayT>>
    where
        HostPlacement: PlacementInput<S, HostBitArrayT>,
    {
        let bit_array = plc.input(sess, arg_name);
        Ok(AbstractHostAesKey(bit_array))
    }

    pub(crate) fn aes_kernel_on_replicated<S: Session, HostAesKeyT, RepAesKeyT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        arg_name: String,
    ) -> Result<AbstractAesKey<HostAesKeyT, RepAesKeyT>>
    where
        ReplicatedPlacement: PlacementInput<S, RepAesKeyT>,
    {
        let key = plc.input(sess, arg_name);
        Ok(AbstractAesKey::Replicated(key))
    }
}

modelled!(PlacementDecrypt::decrypt, HostPlacement, (AesKey, AesTensor) -> Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, HostPlacement, (HostAesKey, AesTensor) -> Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, HostPlacement, (HostAesKey, Fixed128AesTensor) -> Fixed128Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, HostPlacement, (HostAesKey, HostFixed128AesTensor) -> HostFixed128Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, ReplicatedPlacement, (AesKey, AesTensor) -> Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, ReplicatedPlacement, (ReplicatedAesKey, AesTensor) -> Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, ReplicatedPlacement, (ReplicatedAesKey, Fixed128AesTensor) -> Fixed128Tensor, AesDecryptOp);
modelled!(PlacementDecrypt::decrypt, ReplicatedPlacement, (ReplicatedAesKey, HostFixed128AesTensor) -> ReplicatedFixed128Tensor, AesDecryptOp);

kernel! {
    AesDecryptOp,
    [
        (HostPlacement, (AesKey, AesTensor) -> Tensor => [hybrid] Self::host_kernel),
        (HostPlacement, (HostAesKey, AesTensor) -> Tensor => [hybrid] Self::host_key_kernel),
        (HostPlacement, (HostAesKey, Fixed128AesTensor) -> Fixed128Tensor => [hybrid] Self::host_fixed_kernel),
        (HostPlacement, (HostAesKey, HostFixed128AesTensor) -> HostFixed128Tensor => [hybrid] Self::host_fixed_aes_kernel),
        (ReplicatedPlacement, (AesKey, AesTensor) -> Tensor => [hybrid] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, AesTensor) -> Tensor => [hybrid] Self::rep_key_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, Fixed128AesTensor) -> Fixed128Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedAesKey, HostFixed128AesTensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_fixed_aes_kernel),
    ]
}

impl AesDecryptOp {
    pub(crate) fn host_kernel<S: Session, HostAesKeyT, ReplicatedAesKeyT>(
        sess: &S,
        plc: &HostPlacement,
        key: AbstractAesKey<HostAesKeyT, ReplicatedAesKeyT>,
        ciphertext: m!(AesTensor),
    ) -> Result<m!(Tensor)>
    where
        AesTensor: KnownType<S>,
        Tensor: KnownType<S>,
        HostPlacement: PlacementDecrypt<S, HostAesKeyT, m!(AesTensor), m!(Tensor)>,
        HostPlacement: PlacementReveal<S, ReplicatedAesKeyT, HostAesKeyT>,
    {
        let host_key = match key {
            AbstractAesKey::Host(host_key) => host_key,
            AbstractAesKey::Replicated(replicated_key) => plc.reveal(sess, &replicated_key),
        };
        Ok(plc.decrypt(sess, &host_key, &ciphertext))
    }

    pub(crate) fn host_key_kernel<
        S: Session,
        Fixed128AesT,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
    >(
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

    pub(crate) fn rep_kernel<S: Session, HostAesKeyT, ReplicatedAesKeyT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        key: AbstractAesKey<HostAesKeyT, ReplicatedAesKeyT>,
        ciphertext: m!(AesTensor),
    ) -> Result<m!(Tensor)>
    where
        AesTensor: KnownType<S>,
        Tensor: KnownType<S>,
        ReplicatedPlacement: PlacementDecrypt<S, ReplicatedAesKeyT, m!(AesTensor), m!(Tensor)>,
        ReplicatedPlacement: PlacementShare<S, HostAesKeyT, ReplicatedAesKeyT>,
    {
        let replicated_key = match key {
            AbstractAesKey::Host(host_key) => plc.share(sess, &host_key),
            AbstractAesKey::Replicated(replicated_key) => replicated_key,
        };
        Ok(plc.decrypt(sess, &replicated_key, &ciphertext))
    }

    pub(crate) fn rep_key_kernel<
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
        HostBitArray128T,
        HostBitArray224T,
        ShapeT,
        HostRing128TensorT,
        HostBitTensorT,
    >(
        sess: &S,
        plc: &HostPlacement,
        key: AbstractHostAesKey<HostBitArray128T>,
        ciphertext: AbstractHostFixedAesTensor<HostBitArray224T>,
    ) -> Result<AbstractHostFixedTensor<HostRing128TensorT>>
    where
        HostBitArray128T: BitArray<Len = N128>,
        HostBitArray224T: BitArray<Len = N224>,
        HostBitTensorT: Clone,
        HostPlacement: PlacementIndex<S, HostBitArray128T, HostBitTensorT>,
        HostPlacement: PlacementIndex<S, HostBitArray224T, HostBitTensorT>,
        HostPlacement: PlacementShape<S, HostBitTensorT, ShapeT>,
        HostPlacement: PlacementFill<S, ShapeT, HostBitTensorT>,
        HostPlacement: PlacementRingInject<S, HostBitTensorT, HostRing128TensorT>,
        HostPlacement: PlacementFill<S, ShapeT, HostRing128TensorT>,
        HostPlacement: PlacementAdd<S, HostRing128TensorT, HostRing128TensorT, HostRing128TensorT>,
        HostPlacement: PlacementXor<S, HostBitTensorT, HostBitTensorT, HostBitTensorT>,
        HostPlacement: PlacementAnd<S, HostBitTensorT, HostBitTensorT, HostBitTensorT>,
        HostPlacement: PlacementNeg<S, HostBitTensorT, HostBitTensorT>,
        S: std::fmt::Debug,
    {
        let tensor = aesgcm(sess, plc, key.0, ciphertext.tensor);
        Ok(AbstractHostFixedTensor {
            tensor,
            integral_precision: ciphertext.integral_precision,
            fractional_precision: ciphertext.fractional_precision,
        })
    }

    pub(crate) fn rep_fixed_aes_kernel<
        S: Session,
        ShapeT,
        RepBitArray128T,
        RepBitArray224T,
        HostBitArray224T,
        RepBitTensorT,
        RepRing128TensorT,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        key: AbstractReplicatedAesKey<RepBitArray128T>,
        ciphertext: AbstractHostFixedAesTensor<HostBitArray224T>,
    ) -> Result<AbstractReplicatedFixedTensor<RepRing128TensorT>>
    where
        RepBitArray128T: BitArray<Len = N128>,
        HostBitArray224T: BitArray<Len = N224>,
        RepBitArray224T: BitArray<Len = N224>,
        RepBitTensorT: Clone,
        ReplicatedPlacement: PlacementIndex<S, RepBitArray128T, RepBitTensorT>,
        ReplicatedPlacement: PlacementIndex<S, RepBitArray224T, RepBitTensorT>,
        ReplicatedPlacement: PlacementShare<S, HostBitArray224T, RepBitArray224T>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitTensorT, RepRing128TensorT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepRing128TensorT>,
        ReplicatedPlacement:
            PlacementAdd<S, RepRing128TensorT, RepRing128TensorT, RepRing128TensorT>,
        ReplicatedPlacement: PlacementShape<S, RepBitTensorT, ShapeT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepBitTensorT>,
        ReplicatedPlacement: PlacementXor<S, RepBitTensorT, RepBitTensorT, RepBitTensorT>,
        ReplicatedPlacement: PlacementAnd<S, RepBitTensorT, RepBitTensorT, RepBitTensorT>,
        ReplicatedPlacement: PlacementNeg<S, RepBitTensorT, RepBitTensorT>,
        S: std::fmt::Debug,
    {
        let shared_ciphertext = plc.share(sess, &ciphertext.tensor);
        let tensor = aesgcm(sess, plc, key.0, shared_ciphertext);
        Ok(AbstractReplicatedFixedTensor {
            tensor,
            integral_precision: ciphertext.integral_precision,
            fractional_precision: ciphertext.fractional_precision,
        })
    }
}

/// Perform AES-GCM-128 decryption of a single 128 block
///
/// The key must be exactly 128 bits and the ciphertext exactly
/// 96+128 bits, where the former part is the nonce and the latter
/// the masked plaintext.
fn aesgcm<
    S: Session,
    P,
    KeyBitArrayT,
    CiphertextBitArrayT,
    PlaintextRingTensorT,
    BitTensorT,
    ShapeT,
>(
    sess: &S,
    plc: &P,
    key: KeyBitArrayT,
    ciphertext: CiphertextBitArrayT,
) -> PlaintextRingTensorT
where
    KeyBitArrayT: BitArray<Len = N128>,
    CiphertextBitArrayT: BitArray<Len = N224>,
    BitTensorT: Clone,
    P: PlacementIndex<S, KeyBitArrayT, BitTensorT>,
    P: PlacementIndex<S, CiphertextBitArrayT, BitTensorT>,
    P: PlacementShape<S, BitTensorT, ShapeT>,
    P: PlacementFill<S, ShapeT, BitTensorT>,
    P: PlacementRingInject<S, BitTensorT, PlaintextRingTensorT>,
    P: PlacementFill<S, ShapeT, PlaintextRingTensorT>,
    P: PlacementAdd<S, PlaintextRingTensorT, PlaintextRingTensorT, PlaintextRingTensorT>,
    P: PlacementXor<S, BitTensorT, BitTensorT, BitTensorT>,
    P: PlacementAnd<S, BitTensorT, BitTensorT, BitTensorT>,
    P: PlacementNeg<S, BitTensorT, BitTensorT>,
    S: std::fmt::Debug,
{
    // turn inputs into vectors
    let key_bits: Vec<BitTensorT> = (0..128).map(|i| plc.index(sess, i, &key)).collect();
    let ciphertext_bits: Vec<BitTensorT> =
        (0..224).map(|i| plc.index(sess, i, &ciphertext)).collect();

    assert_eq!(key_bits.len(), 128);
    assert_eq!(ciphertext_bits.len(), 96 + 128);

    // separate ciphertext into nonce bits and masked plaintext
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
    println!("number of ops BEFORE AES {:?}", sess);

    // apply AES to block to get mask
    let r_bits = crate::bristol_fashion::aes128(sess, plc, key_bits, block_bits);

    println!("number of ops AFTER AES {:?}", sess);

    // remove mask to recover plaintext
    let m_bits: Vec<BitTensorT> = rm_bits
        .iter()
        .zip(r_bits)
        .map(|(ci, ri)| plc.xor(sess, ci, &ri))
        .collect();

    // bit compose plaintext to obtain ring values
    let shape = plc.shape(sess, &m_bits[0]);
    let zero_ring: PlaintextRingTensorT = plc.fill(sess, Constant::Ring128(0), &shape);

    let res = m_bits
        .iter()
        .enumerate()
        .map(|(i, b)| plc.ring_inject(sess, 127 - i, b))
        .fold(zero_ring, |acc, x| plc.add(sess, &acc, &x));

    println!("number of ops at the end {:?}", sess);
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{HostBitArray128, HostBitArray224};
    use crate::kernels::PlacementReveal;
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

    #[test]
    fn test_aes_decrypt_replicated() {
        let raw_key = [201; 16];
        let raw_nonce = [177; 12];
        let raw_plaintext = [132; 16];

        let host = HostPlacement {
            owner: "host".into(),
        };

        let rep = ReplicatedPlacement {
            owners: [Role::from("alice"), Role::from("bob"), Role::from("carole")],
        };

        let sess = SyncSession::default();

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
            let bit_array = HostBitArray224::from_raw_plc(array, host.clone());

            HostFixed128AesTensor {
                integral_precision: 10,
                fractional_precision: 0,
                tensor: bit_array,
            }
        };

        let key: ReplicatedAesKey = {
            let vec = crate::bristol_fashion::byte_vec_to_bit_vec_be(raw_key.as_ref());
            let array = Array::from_shape_vec((128, 1), vec).unwrap().into_dyn();
            let bit_array = HostBitArray128::from_raw_plc(array, host.clone());
            let shared_bit_array = rep.share(&sess, &bit_array);
            AbstractReplicatedAesKey(shared_bit_array)
        };

        let shared_plaintext = rep.decrypt(&sess, &key, &ciphertext);
        let plaintext = host.reveal(&sess, &shared_plaintext);

        let actual_plaintext = plaintext.tensor.0[0].0;
        let expected_plaintext = u128::from_be_bytes(raw_plaintext);
        assert_eq!(actual_plaintext, expected_plaintext);
    }
}
