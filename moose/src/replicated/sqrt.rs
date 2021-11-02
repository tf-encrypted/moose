use crate::computation::{CanonicalType, KnownType, ReplicatedPlacement};
use crate::kernels::*;
use crate::replicated::division::SignFromMsb;
use crate::replicated::{AbstractReplicatedBitArray, AbstractReplicatedFixedTensor};
use crate::{Const, Ring};

use super::division::TopMost;

// Alternative to TopMost suited for replicated fixedpoint tensors.
// For a fixedpoint tensor `x`, produces an array of bits of length
// `x.integral_precision + x.fractional_precision` such that every bit is 0
// except for the largest index at which `x` takes a nonzero bit, which is 1.
// Example:
//      rep.fixed_top_most(
//          sess,
//          setup,
//          AbstractReplicatedFixedpointTensor{
//              tensor: AbstractReplicatedHostTensor(7),
//              integral_precision: 3,
//              fractional_precision: 5,
//          }
//      ) = [0 0 1 0 0 0 0 0]
pub(crate) trait FixedTopMost<S: Session, SetupT, RepRingT, RepBitT> {
    fn fixed_top_most(
        &self,
        sess: &S,
        setup: &SetupT,
        x: &AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Vec<RepBitT>;
}

impl<S: Session, SetupT, RepRingT, RepBitT, N: Const> FixedTopMost<S, SetupT, RepRingT, RepBitT>
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
    ReplicatedPlacement: TopMost<S, SetupT, RepRingT, RepBitT>,
{
    fn fixed_top_most(
        &self,
        sess: &S,
        setup: &SetupT,
        x: &AbstractReplicatedFixedTensor<RepRingT>,
    ) -> Vec<RepBitT> {
        let rep = self;
        let total_precision = (x.integral_precision + x.fractional_precision) as usize;
        let gtz = rep.msb(sess, setup, &x.tensor);
        let sign = rep.sign_from_msb(sess, &gtz);
        let x_pos = rep.mul_setup(sess, setup, &sign, &x.tensor);
        rep.top_most(sess, setup, total_precision, &x_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::computation::{HostPlacement, ReplicatedPlacement};
    use crate::host::{AbstractHostFixedTensor, AbstractHostRingTensor, HostBitTensor};
    use crate::kernels::SyncSession;
    use ndarray::array;

    #[test]
    fn test_fixed_top_most() {
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

        let z_bits = rep.fixed_top_most(&sess, &setup, &x_shared);

        let revealed_bits: Vec<HostBitTensor> = (0..tp as usize)
            .map(|i| alice.reveal(&sess, &z_bits[i]))
            .collect();

        let expected: Vec<HostBitTensor> = vec![
            HostBitTensor::from_raw_plc(array!(0u8).into_dyn(), alice.clone()),
            HostBitTensor::from_raw_plc(array!(0u8).into_dyn(), alice.clone()),
            HostBitTensor::from_raw_plc(array!(1u8).into_dyn(), alice.clone()),
            HostBitTensor::from_raw_plc(array!(0u8).into_dyn(), alice.clone()),
            HostBitTensor::from_raw_plc(array!(0u8).into_dyn(), alice.clone()),
            HostBitTensor::from_raw_plc(array!(0u8).into_dyn(), alice.clone()),
            HostBitTensor::from_raw_plc(array!(0u8).into_dyn(), alice.clone()),
            HostBitTensor::from_raw_plc(array!(0u8).into_dyn(), alice),
        ];
        assert!(revealed_bits == expected);
    }
}
