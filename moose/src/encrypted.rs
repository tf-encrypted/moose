use crate::bristol_fashion::aes;
use crate::computation::*;
use crate::error::Result;
use crate::host::{
    AbstractHostEncFixedTensor, HostBitArray128, HostBitTensor, HostEncFixed128Tensor,
    HostFixed128Tensor, HostRing128Tensor, HostShape,
};
use crate::kernels::{
    PlacementAdd, PlacementAnd, PlacementBitDec, PlacementFill, PlacementIndex, PlacementIndexAxis,
    PlacementNeg, PlacementRingInject, PlacementSetupGen, PlacementShape, PlacementShareSetup,
    PlacementXor, Session,
};
use crate::replicated::{
    AbstractReplicatedBitArray, AbstractReplicatedFixedTensor, AbstractReplicatedShape,
    ReplicatedShape,
};

impl AesDecryptOp {
    pub(crate) fn host_fixed_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        c: HostEncFixed128Tensor,
        k: HostBitArray128,
    ) -> Result<HostFixed128Tensor>
    where
        HostPlacement: PlacementBitDec<S, HostRing128Tensor, HostBitTensor>,
        HostPlacement: PlacementIndexAxis<S, HostBitTensor, HostBitTensor>,
        HostPlacement: PlacementRingInject<S, HostBitTensor, HostRing128Tensor>,
        HostPlacement: PlacementShape<S, HostEncFixed128Tensor, HostShape>,
        HostPlacement: PlacementShape<S, HostBitTensor, HostShape>,
        HostPlacement: PlacementFill<S, HostShape, HostRing128Tensor>,
        HostPlacement: PlacementFill<S, HostShape, HostBitTensor>,
        HostPlacement: PlacementAdd<S, HostRing128Tensor, HostRing128Tensor, HostRing128Tensor>,
        HostPlacement: PlacementXor<S, HostBitTensor, HostBitTensor, HostBitTensor>,
        HostPlacement: PlacementAnd<S, HostBitTensor, HostBitTensor, HostBitTensor>,
        HostPlacement: PlacementNeg<S, HostBitTensor, HostBitTensor>,
        HostPlacement: PlacementIndex<S, HostBitArray128, HostBitTensor>,
        HostPlacement: PlacementXor<S, HostBitTensor, HostBitTensor, HostBitTensor>,
    {
        let shape = plc.shape(sess, &c);
        let c_decomposed = plc.bit_decompose(sess, &c.tensor);
        let c_bits: Vec<_> = (0..128)
            .map(|i| plc.index_axis(sess, 0, i, &c_decomposed))
            .collect();
        let k_bits: Vec<_> = (0..128).map(|i| plc.index(sess, i, &k)).collect();


        let iv = plc.fill(sess, Constant::Ring128(0), &shape);
        let iv_decomposed = plc.bit_decompose(sess, &iv);
        let iv_bits: Vec<_> = (0..128).map(|i| plc.index_axis(sess, 0, i, &iv_decomposed)).collect();

        let iv_entry_shape = plc.shape(sess, &iv_bits[0]);
        let bit_one: HostBitTensor = plc.fill(sess, Constant::Bit(1), &iv_entry_shape);
        let bit_zero: HostBitTensor = plc.fill(sess, Constant::Bit(0), &iv_entry_shape);

        let complete_iv: Vec<_> = (0..128).map(|i| {
            if i < 96 {
                iv_bits[i].clone()
            } else if i < 126 || i == 127 {
                bit_zero.clone()
            } else {
                bit_one.clone()
            }
        }).collect();


        let buffer_bits = aes(sess, plc, k_bits, complete_iv);
        let m_bits: Vec<_> = (0..128).map(|i| plc.xor(sess, &buffer_bits[i], &c_bits[i])).collect();

        println!("m_bits: {:?}", m_bits);

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
    pub(crate) fn rep_fixed_kernel<S: Session, HostBitT, RepBitT, RepRingT, HostRingT, N>(
        sess: &S,
        rep: &ReplicatedPlacement,
        c: AbstractHostEncFixedTensor<HostRingT>,
        _k: AbstractReplicatedBitArray<RepBitT, N>,
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
        let zero = rep.fill(sess, Constant::Ring128(0), &shape);
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
