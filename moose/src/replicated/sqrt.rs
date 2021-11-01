use crate::computation::{CanonicalType, KnownType, ReplicatedPlacement};
use crate::kernels::*;
use crate::replicated::division::SignFromMsb;
use crate::replicated::{AbstractReplicatedBitArray, AbstractReplicatedFixedTensor};
use crate::{Const, Ring};

// Alternative to PlacementMsb suited for replicated fixedpoint tensors.
// For a fixedpoint tensor `x`, produces an array of bits of length
// `x.integral_precision + x.fractional_precision` such that every bit is 0
// except for the index at which `x` takes its MSB, which is 1.
// Example:
//      msb_index(AbstractFixedpointTensor(7, 3, 5) = [0 0 1 0 0 0 0 0]
pub(crate) trait MsbIndex<S: Session, SetupT, RepRingT, RepBitT> {
    fn msb_index(
        &self,
        sess: &S,
        setup: &SetupT,
        x: &AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Vec<RepRingT>;
}

impl<S: Session, SetupT, RepRingT, RepBitT, N: Const> MsbIndex<S, SetupT, RepRingT, RepBitT>
    for ReplicatedPlacement
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
    ReplicatedPlacement:
        PlacementBitDecSetup<S, SetupT, RepRingT, m!(AbstractReplicatedBitArray<c!(RepBitT), N>)>,
    ReplicatedPlacement: PlacementIndex<S, m!(AbstractReplicatedBitArray<c!(RepBitT), N>), RepBitT>,
    ReplicatedPlacement: PlacementAndSetup<S, SetupT, RepBitT, RepBitT, RepBitT>,
    ReplicatedPlacement: PlacementXor<S, RepBitT, RepBitT, RepBitT>,
    ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
    ReplicatedPlacement: PlacementSub<S, RepRingT, RepRingT, RepRingT>,
{
    fn msb_index(
        &self,
        sess: &S,
        setup: &SetupT,
        x: &AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Vec<RepRingT> {
        let rep = self;
        let total_precision = (x.integral_precision + x.fractional_precision) as usize;
        let ltz = rep.msb(sess, setup, &x.tensor);
        let sign = rep.sign_from_msb(sess, &ltz);
        let x_pos = rep.mul_setup(sess, setup, &sign, &x.tensor);

        let x_pos_binarray = rep.bit_decompose(sess, setup, &x_pos);
        // TODO: pull out the code that is identical in top_most into a common method
        let x_bits_rev = (0..total_precision)
            .map(|i| rep.index(sess, total_precision - i - 1, &x_pos_binarray))
            .collect();

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
        let x = AbstractHostFixedTensor {
            tensor: x_ring,
            integral_precision: ip,
            fractional_precision: fp,
        };

        let sess = SyncSession::default();
        let setup = rep.gen_setup(&sess);

        let x_shared = rep.share(&sess, &setup, &x);

        let z_bits = rep.msb_index(&sess, &setup, &x_shared);

        let revealed_bits: Vec<HostRing64Tensor> = (0..tp as usize)
            .map(|i| alice.reveal(&sess, &z_bits[i]))
            .collect();

        let expected: Vec<HostRing64Tensor> = vec![
            HostRing64Tensor::from_raw_plc(array!(0u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(0u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(1u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(0u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(0u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(0u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(0u64), alice.clone()),
            HostRing64Tensor::from_raw_plc(array!(0u64), alice),
        ];
        assert!(revealed_bits == expected);
    }
}
