//! Truncation for additive placements
use super::*;
use crate::computation::{CanonicalType, KnownType};
use crate::execution::Session;
use crate::host::{HostPlacement, HostPrfKey, HostSeed, HostShape, SyncKey};
use crate::kernels::*;
use crate::replicated::RepTensor;
use crate::{Const, Ring};
use macros::with_context;
use std::convert::TryInto;

/// Trait for truncation mask generation
pub trait TruncMaskGen<S: Session, ShapeT, RingT> {
    fn gen_trunc_mask(
        &self,
        sess: &S,
        amount: usize,
        shape: &ShapeT,
    ) -> (AdtTensor<RingT>, AdtTensor<RingT>, AdtTensor<RingT>);
}

impl<S: Session, HostShapeT, HostRingT> TruncMaskGen<S, HostShapeT, HostRingT> for HostPlacement
where
    HostPrfKey: KnownType<S>,
    HostSeed: KnownType<S>,
    HostRingT: Ring + Clone,
    HostPlacement: PlacementDeriveSeed<S, m!(HostPrfKey), m!(HostSeed)>,
    HostPlacement: PlacementSampleUniform<S, HostShapeT, HostRingT>,
    HostPlacement: PlacementSampleUniformSeeded<S, HostShapeT, m!(HostSeed), HostRingT>,
    HostPlacement: PlacementKeyGen<S, m!(HostPrfKey)>,
    HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
    HostPlacement: PlacementShr<S, HostRingT, HostRingT>,
    HostPlacement: PlacementShl<S, HostRingT, HostRingT>,
{
    fn gen_trunc_mask(
        &self,
        sess: &S,
        amount: usize,
        shape: &HostShapeT, // TODO(Morten) take AdditiveShape instead?
    ) -> (
        AdtTensor<HostRingT>,
        AdtTensor<HostRingT>,
        AdtTensor<HostRingT>,
    ) {
        let r = self.sample_uniform(sess, shape);
        let r_msb = self.shr(sess, HostRingT::BitLength::VALUE - 1, &r);
        let r_top = self.shr(sess, amount + 1, &self.shl(sess, 1, &r));

        let key = self.gen_key(sess);
        let share = |x| {
            // TODO(Dragos) this could be optimized by instead sending the key (or seeds) to p0
            let sync_key = SyncKey::random();
            let seed = self.derive_seed(sess, sync_key, &key);
            let x0 = self.sample_uniform_seeded(sess, shape, &seed);
            let x1 = self.sub(sess, x, &x0);
            AdtTensor { shares: [x0, x1] }
        };

        let r_shared = share(&r);
        let r_top_shared = share(&r_top);
        let r_msb_shared = share(&r_msb);

        (r_shared, r_top_shared, r_msb_shared)
    }
}

pub(crate) trait TruncPrProvider<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: usize, provider: &HostPlacement, x: &T) -> O;
}

impl<S: Session, HostRingT> TruncPrProvider<S, AdtTensor<HostRingT>, AdtTensor<HostRingT>>
    for AdditivePlacement
where
    AdtTensor<HostRingT>: CanonicalType,
    <AdtTensor<HostRingT> as CanonicalType>::Type: KnownType<S>,
    RepTensor<HostRingT>: CanonicalType,
    <RepTensor<HostRingT> as CanonicalType>::Type: KnownType<S>,
    HostRingT: Ring,
    HostShape: KnownType<S>,
    HostPlacement: TruncMaskGen<S, m!(HostShape), HostRingT>,
    HostPlacement: PlacementReveal<S, m!(c!(AdtTensor<HostRingT>)), HostRingT>,
    HostPlacement: PlacementOnes<S, m!(HostShape), HostRingT>,
    HostPlacement: PlacementShape<S, HostRingT, m!(HostShape)>,
    HostPlacement: PlacementShl<S, HostRingT, HostRingT>,
    HostPlacement: PlacementShr<S, HostRingT, HostRingT>,
    AdtTensor<HostRingT>: Clone + Into<m!(c!(AdtTensor<HostRingT>))>,
    m!(c!(AdtTensor<HostRingT>)): TryInto<AdtTensor<HostRingT>>,
    AdditivePlacement:
        PlacementAdd<S, m!(c!(AdtTensor<HostRingT>)), HostRingT, m!(c!(AdtTensor<HostRingT>))>,
    AdditivePlacement: PlacementAdd<
        S,
        m!(c!(AdtTensor<HostRingT>)),
        m!(c!(AdtTensor<HostRingT>)),
        m!(c!(AdtTensor<HostRingT>)),
    >,
    AdditivePlacement:
        PlacementAdd<S, AdtTensor<HostRingT>, AdtTensor<HostRingT>, AdtTensor<HostRingT>>,
    AdditivePlacement:
        PlacementSub<S, HostRingT, m!(c!(AdtTensor<HostRingT>)), m!(c!(AdtTensor<HostRingT>))>,
    AdditivePlacement:
        PlacementMul<S, m!(c!(AdtTensor<HostRingT>)), HostRingT, m!(c!(AdtTensor<HostRingT>))>,
    AdditivePlacement: PlacementShl<S, m!(c!(AdtTensor<HostRingT>)), m!(c!(AdtTensor<HostRingT>))>,
    AdditivePlacement: PlacementSub<
        S,
        m!(c!(AdtTensor<HostRingT>)),
        m!(c!(AdtTensor<HostRingT>)),
        m!(c!(AdtTensor<HostRingT>)),
    >,
    AdditivePlacement:
        PlacementSub<S, AdtTensor<HostRingT>, AdtTensor<HostRingT>, AdtTensor<HostRingT>>,
    AdditivePlacement:
        PlacementSub<S, m!(c!(AdtTensor<HostRingT>)), HostRingT, m!(c!(AdtTensor<HostRingT>))>,
{
    fn trunc_pr(
        &self,
        sess: &S,
        amount: usize,
        provider: &HostPlacement,
        x: &AdtTensor<HostRingT>,
    ) -> AdtTensor<HostRingT> {
        #![allow(clippy::many_single_char_names)]

        let (player0, player1) = self.host_placements();
        assert!(*provider != player0);
        assert!(*provider != player1);

        let AdtTensor { shares: [x0, _x1] } = x;

        let shape = player0.shape(sess, x0);

        let (r, r_top, r_msb) = provider.gen_trunc_mask(sess, amount, &shape);
        // NOTE we consider input is always signed, and the following positive
        // conversion would be optional for unsigned numbers
        // NOTE we assume that input numbers are in range -2^{k-2} <= x < 2^{k-2}
        // so that 0 <= x + 2^{k-2} < 2^{k-1}
        // TODO we could insert debug_assert! to check above conditions
        let k = HostRingT::BitLength::VALUE - 1;
        let ones = player0.ones(sess, &shape);
        let upshifter = player0.shl(sess, k - 1, &ones);
        let downshifter = player0.shl(sess, k - amount - 1, &ones);

        // TODO(Morten) think the rest of this would clean up nicely if we instead revealed to a mirrored placement
        let x_positive: AdtTensor<HostRingT> = self
            .add(sess, &x.clone().into(), &upshifter)
            .try_into()
            .ok()
            .unwrap();
        let masked = self.add(sess, &x_positive, &r);
        let c = player0.reveal(sess, &masked.into());
        let c_no_msb = player0.shl(sess, 1, &c);
        // also called shifted
        let c_top = player0.shr(sess, amount + 1, &c_no_msb);
        let c_msb = player0.shr(sess, HostRingT::BitLength::VALUE - 1, &c);

        // OK
        let overflow = with_context!(
            self,
            sess,
            r_msb.clone().into() + c_msb - r_msb.clone().into() * c_msb - r_msb.into() * c_msb
        ); // a xor b = a+b-2ab
        let shifted_overflow = self.shl(sess, k - amount, &overflow);
        // shifted - upper + overflow << (k - m)
        let y_positive = with_context!(self, sess, c_top - r_top.into() + shifted_overflow);

        with_context!(self, sess, y_positive - downshifter)
            .try_into()
            .ok()
            .unwrap()
    }
}

#[cfg(feature = "sync_execute")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::HostRingTensor;
    use crate::prelude::*;
    use ndarray::prelude::*;
    use proptest::prelude::*;

    #[test]
    fn test_trunc() {
        let alice = HostPlacement::from("alice");
        let bob = HostPlacement::from("bob");
        let carole = HostPlacement::from("carole");
        let adt = AdditivePlacement::from(["alice", "bob"]);

        let sess = SyncSession::default();

        let x = AdditiveRing64Tensor {
            shares: [
                alice.from_raw(array![0_u64, 0, 0]),
                bob.from_raw(array![
                    4611686018427387903,
                    -1152921504606846976_i64 as u64,
                    1152921504606846975
                ]),
            ],
        };

        let x_trunc = adt.trunc_pr(&sess, 60, &carole, &x);
        let _y = carole.reveal(&sess, &x_trunc);

        let target: HostRing64Tensor = carole.from_raw(array![3, -1_i64 as u64, 0]);

        // probabilistic truncation can be off by 1
        for (i, value) in _y.0.iter().enumerate() {
            let diff = value - target.0[i];
            assert!(
                diff == std::num::Wrapping(1)
                    || diff == std::num::Wrapping(u64::MAX)
                    || diff == std::num::Wrapping(0),
                "difference = {}, lhs = {}, rhs = {}",
                diff,
                value,
                target.0[i]
            );
        }
    }

    fn any_bounded_u64() -> impl Strategy<Value = u64> {
        any::<u64>().prop_map(|x| (x >> 2) - 1)
    }

    fn any_bounded_u128() -> impl Strategy<Value = u128> {
        any::<u128>().prop_map(|x| (x >> 2) - 1)
    }

    macro_rules! adt_truncation_test {
        ($func_name:ident, $tt: ident) => {
            fn $func_name(xs: ArrayD<$tt>, amount: usize, ys: ArrayD<$tt>) {
                let alice = HostPlacement::from("alice");
                let bob = HostPlacement::from("bob");
                let carole = HostPlacement::from("carole");
                let adt = AdditivePlacement::from(["alice", "bob"]);

                let sess = SyncSession::default();

                // creates an additive sharing of xs
                let zero =
                    Array::from_shape_vec(IxDyn(&[xs.len()]), vec![0 as $tt; xs.len()]).unwrap();
                let x = AdtTensor {
                    shares: [
                        HostRingTensor::from_raw_plc(zero, alice),
                        HostRingTensor::from_raw_plc(xs.clone(), bob),
                    ],
                };

                let x_trunc = adt.trunc_pr(&sess, amount, &carole, &x);
                let y = carole.reveal(&sess, &x_trunc);

                let target_y: HostRingTensor<_> = carole.from_raw(ys.clone());
                for (i, value) in y.0.iter().enumerate() {
                    let diff = value - target_y.0[i];
                    assert!(
                        diff == std::num::Wrapping(1)
                            || diff == std::num::Wrapping($tt::MAX)
                            || diff == std::num::Wrapping(0),
                        "difference = {}, lhs = {}, rhs = {}",
                        diff,
                        value,
                        target_y.0[i]
                    );
                }
            }
        };
    }

    adt_truncation_test!(test_adt_trunc64, u64);
    adt_truncation_test!(test_adt_trunc128, u128);

    proptest! {
        #[test]
        fn test_fuzzy_adt_trunc64(raw_vector in proptest::collection::vec(any_bounded_u64(), 1..5), amount in 0usize..62
        ) {
            let target = raw_vector.iter().map(|x| x >> amount).collect::<Vec<_>>();
            test_adt_trunc64(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), amount, Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }

        #[test]
        fn test_fuzzy_adt_trunc128(raw_vector in proptest::collection::vec(any_bounded_u128(), 1..5), amount in 0usize..126
        ) {
            let target = raw_vector.iter().map(|x| x >> amount).collect::<Vec<_>>();
            test_adt_trunc128(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), amount, Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }
    }
}
