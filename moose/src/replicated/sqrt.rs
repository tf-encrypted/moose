use crate::computation::{CanonicalType, KnownType, ReplicatedPlacement};
use crate::{Const, Ring};
use crate::kernels::*;
use crate::replicated::division::SignFromMsb;
use crate::replicated::{AbstractReplicatedBitArray, AbstractReplicatedFixedTensor};

pub(crate) trait MsbIndex<S: RuntimeSession, SetupT, RepRingT, RepBitT> {
    fn msb_index(&self, sess: &S, setup: &SetupT, x: &AbstractReplicatedFixedTensor<RepRingT>) -> Vec<RepRingT>;
}

impl<S: RuntimeSession, SetupT, RepRingT, RepBitT, N: Const> MsbIndex<S, SetupT, RepRingT, RepBitT> for ReplicatedPlacement
where
    RepRingT: Ring<BitLength = N> + Clone,
    RepBitT: CanonicalType,
    AbstractReplicatedFixedTensor<RepRingT>: CanonicalType,
    <AbstractReplicatedFixedTensor<RepRingT> as CanonicalType>::Type: KnownType<S>,
    AbstractReplicatedFixedTensor<RepRingT>: Into<st!(AbstractReplicatedFixedTensor<RepRingT>)>,
    AbstractReplicatedBitArray<c!(RepBitT), N>: CanonicalType,
    AbstractReplicatedBitArray<c!(RepBitT), N>: KnownType<S>,
    ReplicatedPlacement: PlacementMsb<S, SetupT, RepRingT, RepRingT>,
    ReplicatedPlacement: SignFromMsb<S, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementMulSetup<S, SetupT, RepRingT, RepRingT, RepRingT>,
    ReplicatedPlacement: PlacementBitDecSetup<
        S,
        SetupT,
        RepRingT,
        m!(AbstractReplicatedBitArray<c!(RepBitT), N>),
    >,
    ReplicatedPlacement: PlacementIndex<S, m!(AbstractReplicatedBitArray<c!(RepBitT), N>), RepBitT>,
    ReplicatedPlacement: PlacementAndSetup<S, SetupT, RepBitT, RepBitT, RepBitT>,
    ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
    ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
    ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
{
    fn msb_index(&self, sess: &S, setup: &SetupT, x: &AbstractReplicatedFixedTensor<RepRingT>) -> Vec<RepRingT> {
        let rep = self;
        let total_precision = (x.integral_precision + x.fractional_precision) as usize;
        let ltz = rep.msb(sess, setup, &x.tensor);
        let sign = rep.sign_from_msb(sess, &ltz);
        let x_pos = rep.mul_setup(sess, setup, &sign, &x.tensor);

        let x_pos_binarray = rep.bit_decompose(sess, setup, &x_pos);
        // TODO: pull out the code that is identical in top_most into a common method
        let x_bits_rev = (0..total_precision).map(|i| rep.index(sess, total_precision- i - 1, &x_pos_binarray)).collect();

        let y_bits = rep.prefix_or(sess, setup, x_bits_rev);
        let mut y_top_bits: Vec<_> = y_bits
            .iter()
            .take(total_precision)
            .map(|e| rep.ring_inject(sess, 0, e))
            .collect();

        y_top_bits.reverse();
        let mut z: Vec<_> = (0..total_precision - 1)
            .map(|i| rep.sub(sess, &y_top_bits[i], &y_top_bits[i + 1]))
            .collect();
        z.push(y_top_bits[total_precision - 1].clone());

        z
    }
}

// pub trait SimplifiedNormSq<S: RuntimeSession, SetupT, RepBitT, HostRingT> {
//     fn simp_norm_sq_64(
//         &self,
//         sess: &S,
//         setup: &SetupT,
//         x: AbstractReplicatedRingTensor<HostRingT>,
//         total_precision: usize,
//         fractional_precision: usize
//     ) -> (RepBitT, AbstractReplicatedRingTensor<HostRingT>);
//     // TODO
//     // fn simp_norm_sq_128(&self, sess: &S, setup: &SetupT, x: AbstractReplicatedRingTensor<HostRingT>) -> (RepBitT, AbstractReplicatedRingTensor<HostRingT>);
// }

// impl<S: RuntimeSession, SetupT, RepBitT, HostRingT, N: Const> SimplifiedNormSq<S, SetupT, RepBitT, HostRingT> for ReplicatedPlacement
// where
//     HostRingT: Clone + Ring<BitLength = N> + Tensor<S> + From<HostRing64Tensor>,
//     <HostRingT as Tensor<S>>::Scalar: From<usize>,
//     AbstractReplicatedRingTensor<HostRingT>: CanonicalType,
//     <AbstractReplicatedRingTensor<HostRingT> as CanonicalType>::Type: KnownType<S>,
//     st!(AbstractReplicatedRingTensor<HostRingT>): TryInto<AbstractReplicatedRingTensor<HostRingT>>,
//     ReplicatedShape: KnownType<S>,
//     AbstractReplicatedRingTensor<HostRingT>: Clone + Into<st!(AbstractReplicatedRingTensor<HostRingT>)>,
//     HostPlacement: PlacementPlace<S, HostRingT>,
//     ReplicatedPlacement:
//         MsbIndex<S, AbstractReplicatedFixedTensor<HostRingT>, AbstractReplicatedRingTensor<HostRingT>>,
//     ReplicatedPlacement: PlacementAdd<S, RepBitT, RepBitT, RepBitT>,
//     ReplicatedPlacement:PlacementAdd<
//         S,
//         AbstractReplicatedRingTensor<HostRingT>,
//         AbstractReplicatedRingTensor<HostRingT>,
//         AbstractReplicatedRingTensor<HostRingT>
//     >,
//     ReplicatedPlacement: PlacementDotSetup<S, SetupT, AbstractReplicatedRingTensor<HostRingT>, HostRingT, AbstractReplicatedRingTensor<HostRingT>>,
//     ReplicatedPlacement: PlacementShape<S, AbstractReplicatedRingTensor<HostRingT>, cs!(ReplicatedShape)>,
//     ReplicatedPlacement: PlacementIndexAxis<S, AbstractReplicatedRingTensor<HostRingT>, AbstractReplicatedRingTensor<HostRingT>>,
//     ReplicatedPlacement: PlacementZeros<S, cs!(ReplicatedShape), RepBitT>,
//     ReplicatedPlacement: PlacementShl<S, AbstractReplicatedRingTensor<HostRingT>, AbstractReplicatedRingTensor<HostRingT>>,
// {
//     fn simp_norm_sq_64(
//         &self,
//         sess: &S,
//         setup: &SetupT,
//         x: AbstractReplicatedRingTensor<HostRingT>,
//         total_precision: usize,
//         fractional_precision: usize,
//     ) -> (RepBitT, AbstractReplicatedRingTensor<HostRingT>)
//     {
//         let bit_length = HostRingT::BitLength::VALUE;
//         let rep = self;
//         // TODO op that gives masked bitdec msb
//         let z = rep.msb_index(sess, &x);
//         // TODO write helper to determine last axis of n-dim tensor
//         let bits_axis = Some(-1i32);
//         // TODO choose this host placement randomly?
//         let (p0, _, _) = rep.host_placements();
//         let parity_indices = (1u64..bit_length as u64).collect();
//         let raw_index_vec = Array1::from_vec(parity_indices);
//         let index_vec = AbstractHostRingTensor::from_raw_plc(raw_index_vec, p0);
//         // TODO add Constant kernels for HostRingTensor<R>
//         // let v = p0.constant(sess, index_vec.into());
//         let v = p0.place(sess, index_vec.into());
//         let m = rep.dot_setup(sess, setup, &z, &v);
//         let m_shape = rep.shape(sess, &m);
//         let m_odd = rep.zeros(sess, &m_shape);
//         let zs: Vec<_> = (0u64..bit_length as u64 - 1).map(|i| rep.index_axis(sess, 1, i.try_into().unwrap(), &z)).collect();
//         for (i, zi) in zs.iter().enumerate() {
//             if i % 2 == 0 {
//                 m_odd = rep.add(sess, &m_odd, zi);
//             }
//         }
//         let ws: Vec<_> = (1..bit_length / 2).map(|i| rep.add(sess, &zs[2*i - 1], &zs[2*i])).collect();
//         let w_shl = (1..bit_length / 2 - 1).map(|i| rep.shl(sess, i, &ws[i - 1])).collect();
//         // TODO replace with variadic add_n operation for replicated tensors
//         let mut ring_res = rep.shl(sess, 0, &w_shl[0]);
//         for wi in w_shl.iter().skip(1) {
//             ring_res = rep.add(sess, &ring_res, wi);
//         }
//         (m_odd, ring_res)
//     }
// }

// modelled!(PlacementSqrtAsFixedpoint::sqrt_as_fixedpoint, ReplicatedPlacement, attributes[total_precision: usize, fractional_precision: usize] (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, RepFixedpointSqrtOp);
// modelled!(PlacementSqrtAsFixedpoint::sqrt_as_fixedpoint, ReplicatedPlacement, attributes[total_precision: usize, fractional_precision: usize] (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, RepFixedpointSqrtOp);

// kernel!{
//     RepFixedpointSqrtOp,
//     [
//         (ReplicatedPlacement, (ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] attributes[total_precision, fractional_precision] Self::rep_kernel_64),
//         (ReplicatedPlacement, (ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [transparent] attributes[total_precision, fractional_precision] Self::rep_kernel_128),
//     ]
// }

// impl RepFixedpointSqrtOp {
//     fn rep_kernel_64<S: RuntimeSession, HostRingT, RepBitT>(
//         sess: &S,
//         rep: &ReplicatedPlacement,
//         total_precision: usize,
//         fractional_precision: usize,
//         x: AbstractReplicatedRingTensor<HostRingT>,
//     ) -> Result<AbstractReplicatedRingTensor<HostRingT>>
//     where
//         ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
//         ReplicatedPlacement: SimplifiedNormSq<S, S::ReplicatedSetup, RepBitT, HostRingT>,
//     {
//         let theta = (total_precision as f64).log2().ceil() as usize;
//         theta = max(theta, 6);
//         let setup = rep.gen_setup(sess);
//         let (m_odd, w) = rep.simp_norm_sq_64(sess, &setup, x, total_precision, fractional_precision);
//         unimplemented!("TODO")
//     }

//     fn rep_kernel_128<S: RuntimeSession, HostRingT, RepRingT, RepBitT>(
//         sess: &S,
//         rep: &ReplicatedPlacement,
//         total_precision: usize,
//         fractional_precision: usize,
//         x: AbstractReplicatedRingTensor<HostRingT>,
//     ) -> Result<AbstractReplicatedRingTensor<HostRingT>>
//     where
//         ReplicatedPlacement: PlacementSetupGen<S, S::ReplicatedSetup>,
//         ReplicatedPlacement: SimplifiedNormSq<S, S::ReplicatedSetup, RepBitT, HostRingT>,
//     {
//         unimplemented!("TODO")
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::computation::{HostPlacement, ReplicatedPlacement};
    use crate::host::{AbstractHostFixedTensor, AbstractHostRingTensor, HostRing64Tensor};
    use crate::kernels::SyncSession;
    use ndarray::array;

    #[test]
    fn test_msb_index() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x_ring = AbstractHostRingTensor::from_raw_plc(array![7u64], alice.clone());
        let ip = 3;
        let fp = 5;
        let tp = ip + fp;
        let x = AbstractHostFixedTensor{tensor: x_ring, integral_precision: ip, fractional_precision: fp};

        let sess = SyncSession::default();
        let setup = rep.gen_setup(&sess);

        let x_shared = rep.share(&sess, &setup, &x);

        let z_bits = rep.msb_index(&sess, &setup, &x_shared);

        let revealed_bits: Vec<HostRing64Tensor> = (0..tp as usize - 1).map(|i| alice.reveal(&sess, &z_bits[i])).collect();

        let expected: Vec<HostRing64Tensor> = vec![
            HostRing64Tensor::from_raw_plc(array!(0u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(0u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(1u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(0u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(0u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(0u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(0u64), alice),
        ];
        assert!(revealed_bits == expected);
    }
}
