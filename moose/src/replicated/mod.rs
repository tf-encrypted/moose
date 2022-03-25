//! Placements backed by three-party replicated secret sharing

use crate::computation::*;
use crate::error::{Error, Result};
#[cfg(feature = "compile")]
use crate::execution::symbolic::Symbolic;
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::kernels::*;
use crate::mirrored::Mir3Tensor;
use crate::types::*;
use crate::{BitArray, Const, MirroredCounterpart, Ring, Underlying};
use macros::with_context;
use serde::{Deserialize, Serialize};
#[cfg(feature = "compile")]
use std::convert::TryFrom;
use std::marker::PhantomData;

mod aes;
mod argmax;
mod arith;
mod bits;
mod compare;
mod control_flow;
mod convert;
mod division;
mod exp;
mod fixedpoint;
mod input;
mod log;
mod misc;
mod ops;
mod setup;
mod softmax;
mod zero_share;
pub use self::aes::RepAesKey;
pub use self::fixedpoint::RepFixedTensor;
pub(crate) use self::misc::{BinaryAdder, ShapeFill};
pub use self::setup::RepSetup;
use self::zero_share::{RepZeroShare, ZeroShareGen};

/// Placement type for three-party replicated secret sharing
#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct ReplicatedPlacement {
    pub owners: [Role; 3],
}

impl ReplicatedPlacement {
    pub fn host_placements(&self) -> (HostPlacement, HostPlacement, HostPlacement) {
        let player0 = HostPlacement {
            owner: self.owners[0].clone(),
        };
        let player1 = HostPlacement {
            owner: self.owners[1].clone(),
        };
        let player2 = HostPlacement {
            owner: self.owners[2].clone(),
        };
        (player0, player1, player2)
    }
}

impl<R: Into<Role>> From<[R; 3]> for ReplicatedPlacement {
    fn from(roles: [R; 3]) -> ReplicatedPlacement {
        let [role0, role1, role2] = roles;
        ReplicatedPlacement {
            owners: [role0.into(), role1.into(), role2.into()],
        }
    }
}

/// Secret tensor used by replicated placements
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepTensor<HostRingT> {
    pub(crate) shares: [[HostRingT; 2]; 3],
}

impl<HostTenT> Placed for RepTensor<HostTenT>
where
    HostTenT: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = self;

        let owner0 = x00.placement()?.owner;
        let owner1 = x11.placement()?.owner;
        let owner2 = x22.placement()?.owner;

        if x10.placement()?.owner == owner0
            && x21.placement()?.owner == owner1
            && x02.placement()?.owner == owner2
        {
            let owners = [owner0, owner1, owner2];
            Ok(ReplicatedPlacement { owners })
        } else {
            Err(Error::MalformedPlacement)
        }
    }
}

impl<S: Session, HostRingT> PlacementPlace<S, RepTensor<HostRingT>> for ReplicatedPlacement
where
    RepTensor<HostRingT>: Placed<Placement = ReplicatedPlacement>,
    HostPlacement: PlacementPlace<S, HostRingT>,
{
    fn place(&self, sess: &S, x: RepTensor<HostRingT>) -> RepTensor<HostRingT> {
        match x.placement() {
            Ok(place) if &place == self => x,
            _ => {
                let RepTensor {
                    shares: [[x00, x10], [x11, x21], [x22, x02]],
                } = x;

                let (player0, player1, player2) = self.host_placements();
                RepTensor {
                    shares: [
                        [player0.place(sess, x00), player0.place(sess, x10)],
                        [player1.place(sess, x11), player1.place(sess, x21)],
                        [player2.place(sess, x22), player2.place(sess, x02)],
                    ],
                }
            }
        }
    }
}

impl<HostRingT> Underlying for RepTensor<HostRingT> {
    type TensorType = HostRingT;
}

impl<HostRingT: Ring> Ring for RepTensor<HostRingT> {
    type BitLength = HostRingT::BitLength;
}

impl<HostRingT> MirroredCounterpart for RepTensor<HostRingT> {
    type MirroredType = Mir3Tensor<HostRingT>;
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RepUintTensor<RepRingT> {
    pub tensor: RepRingT,
}

impl<RepRingT: Placed> Placed for RepUintTensor<RepRingT> {
    type Placement = RepRingT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.tensor.placement()
    }
}

impl<S: Session, RepRingT> PlacementPlace<S, RepUintTensor<RepRingT>> for ReplicatedPlacement
where
    RepUintTensor<RepRingT>: Placed<Placement = ReplicatedPlacement>,
    ReplicatedPlacement: PlacementPlace<S, RepRingT>,
{
    fn place(&self, sess: &S, x: RepUintTensor<RepRingT>) -> RepUintTensor<RepRingT> {
        match x.placement() {
            Ok(place) if self == &place => x,
            _ => {
                // TODO just updating the placement isn't enough,
                // we need this to eventually turn into Send + Recv
                RepUintTensor {
                    tensor: self.place(sess, x.tensor),
                }
            }
        }
    }
}

/// Public shape for replicated placements
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepShape<HostShapeT> {
    pub shapes: [HostShapeT; 3],
}

impl<HostShapeT> Placed for RepShape<HostShapeT>
where
    HostShapeT: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let RepShape {
            shapes: [s0, s1, s2],
        } = self;

        let owner0 = s0.placement()?.owner;
        let owner1 = s1.placement()?.owner;
        let owner2 = s2.placement()?.owner;

        let owners = [owner0, owner1, owner2];
        Ok(ReplicatedPlacement { owners })
    }
}

/// Replicated bit array of fixed size
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepBitArray<RepBitTensorT, N>(RepBitTensorT, PhantomData<N>);

impl<RepBitTensorT, N: Const> BitArray for RepBitArray<RepBitTensorT, N> {
    type Len = N;
}

#[cfg(feature = "compile")]
impl<RepBitTensorT: Placed, N: Const> BitArray for Symbolic<RepBitArray<RepBitTensorT, N>> {
    type Len = N;
}

impl<RepBitTensorT: Placed, N> Placed for RepBitArray<RepBitTensorT, N> {
    type Placement = RepBitTensorT::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        self.0.placement()
    }
}

// TODO implement using moose_type macro

#[cfg(feature = "compile")]
impl<N> PartiallySymbolicType for RepBitArray<ReplicatedBitTensor, N> {
    type Type = RepBitArray<<ReplicatedBitTensor as SymbolicType>::Type, N>;
}

impl<N> CanonicalType for RepBitArray<ReplicatedBitTensor, N> {
    type Type = Self;
}

#[cfg(feature = "compile")]
impl<N> CanonicalType for RepBitArray<<ReplicatedBitTensor as SymbolicType>::Type, N> {
    type Type = RepBitArray<ReplicatedBitTensor, N>;
}

#[cfg(feature = "compile")]
impl<N> CanonicalType for Symbolic<RepBitArray<<ReplicatedBitTensor as SymbolicType>::Type, N>> {
    type Type = RepBitArray<ReplicatedBitTensor, N>;
}

#[cfg(feature = "compile")]
impl<RepBitT: Placed, N> From<RepBitArray<RepBitT, N>> for Symbolic<RepBitArray<RepBitT, N>>
where
    RepBitT: Placed<Placement = ReplicatedPlacement>,
{
    fn from(x: RepBitArray<RepBitT, N>) -> Self {
        Symbolic::Concrete(x)
    }
}

#[cfg(feature = "compile")]
impl<RepBitT, N> TryFrom<Symbolic<RepBitArray<RepBitT, N>>> for RepBitArray<RepBitT, N>
where
    RepBitT: Placed<Placement = ReplicatedPlacement>,
{
    type Error = Error;
    fn try_from(v: Symbolic<RepBitArray<RepBitT, N>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected(None)), // TODO err message
        }
    }
}

#[cfg(feature = "sync_execute")]
#[cfg(test)]
mod tests {
    use super::{PhantomData, RepBitArray, RepTensor};
    use crate::host::{HostRingTensor, RawShape};
    use crate::mirrored::Mir3Tensor;

    use crate::prelude::*;
    use crate::{N128, N64};
    use ndarray::prelude::*;
    use proptest::prelude::*;
    use rstest::rstest;

    #[test]
    fn test_ring_identity() {
        let alice = HostPlacement::from("alice");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        let sess = SyncSession::default();

        let x: HostRing64Tensor = alice.from_raw(array![1u64, 2, 3]);
        let x_shared = rep.share(&sess, &x);
        let iden = rep.identity(&sess, &x_shared);
        let opened_result = alice.reveal(&sess, &iden);

        assert_eq!(opened_result, x);
    }

    #[test]
    fn test_identity_diff_plc() {
        let alice0 = HostPlacement::from("alice-0");
        let rep0 = ReplicatedPlacement::from(["alice-0", "bob-0", "carole-0"]);
        let rep1 = ReplicatedPlacement::from(["alice-1", "bob-1", "carole-1"]);

        let sess = SyncSession::default();

        let x: HostRing64Tensor = alice0.from_raw(array![1, 2, 3]);
        let x_shared = rep0.share(&sess, &x);
        let iden = rep1.identity(&sess, &x_shared);
        let opened_result = alice0.reveal(&sess, &iden);
        assert_eq!(opened_result, x);
    }

    #[test]
    fn test_adt_to_rep() {
        let alice = HostPlacement::from("alice");
        let bob = HostPlacement::from("bob");
        let carole = HostPlacement::from("carole");
        let dave = HostPlacement::from("dave");
        let eric = HostPlacement::from("eric");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        let sess = SyncSession::default();

        let x1 = AdditiveRing64Tensor {
            shares: [
                alice.from_raw(array![1, 2, 3]),
                bob.from_raw(array![4, 5, 6]),
            ],
        };

        let x1_rep = rep.adt_to_rep(&sess, &x1);
        assert_eq!(alice.reveal(&sess, &x1_rep), alice.reveal(&sess, &x1));
        assert_eq!(bob.reveal(&sess, &x1_rep), bob.reveal(&sess, &x1));
        assert_eq!(carole.reveal(&sess, &x1_rep), carole.reveal(&sess, &x1));

        let x2 = AdditiveRing64Tensor {
            shares: [
                bob.from_raw(array![1, 2, 3]),
                alice.from_raw(array![4, 5, 6]),
            ],
        };

        let x2_rep = rep.adt_to_rep(&sess, &x2);
        assert_eq!(alice.reveal(&sess, &x2_rep), alice.reveal(&sess, &x2));
        assert_eq!(bob.reveal(&sess, &x2_rep), bob.reveal(&sess, &x2));
        assert_eq!(carole.reveal(&sess, &x2_rep), carole.reveal(&sess, &x2));

        let x3 = AdditiveRing64Tensor {
            shares: [
                dave.from_raw(array![1, 2, 3]),
                eric.from_raw(array![4, 5, 6]),
            ],
        };

        let x3_rep = rep.adt_to_rep(&sess, &x3);
        assert_eq!(alice.reveal(&sess, &x3_rep), alice.reveal(&sess, &x3));
        assert_eq!(bob.reveal(&sess, &x3_rep), bob.reveal(&sess, &x3));
        assert_eq!(carole.reveal(&sess, &x3_rep), carole.reveal(&sess, &x3));

        let x4 = AdditiveRing64Tensor {
            shares: [
                alice.from_raw(array![1, 2, 3]),
                eric.from_raw(array![4, 5, 6]),
            ],
        };

        let x4_rep = rep.adt_to_rep(&sess, &x4);
        assert_eq!(alice.reveal(&sess, &x4_rep), alice.reveal(&sess, &x4));
        assert_eq!(bob.reveal(&sess, &x4_rep), bob.reveal(&sess, &x4));
        assert_eq!(carole.reveal(&sess, &x4_rep), carole.reveal(&sess, &x4));
    }

    #[test]
    fn test_rep_mean() {
        let alice = HostPlacement::from("alice");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        let sess = SyncSession::default();

        let scaling_base = 2;
        let scaling_exp = 24;

        let x: HostFloat64Tensor = alice.from_raw(array![1.0, 2.0, 3.0]);
        let x = alice.fixedpoint_ring_encode(&sess, scaling_base, scaling_exp, &x);
        let x_shared = rep.share(&sess, &x);

        let mean = rep.mean_as_fixedpoint(&sess, None, scaling_base, scaling_exp, &x_shared);
        let mean = rep.trunc_pr(&sess, scaling_exp, &mean);
        let opened_result = alice.reveal(&sess, &mean);
        let decoded_result =
            alice.fixedpoint_ring_decode(&sess, scaling_base, scaling_exp, &opened_result);

        assert!(num_traits::abs(2.0 - decoded_result.0[[]]) < 0.01);
    }

    #[test]
    fn test_rep_add_n() {
        let alice = HostPlacement::from("alice");
        let bob = HostPlacement::from("bob");
        let carole = HostPlacement::from("carole");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        let sess = SyncSession::default();

        // 64 bit
        let a: HostRing64Tensor = alice.from_raw(array![1, 2, 3]);
        let b: HostRing64Tensor = bob.from_raw(array![2, 3, 4]);
        let c: HostRing64Tensor = carole.from_raw(array![5, 12, 13]);
        let expected: HostRing64Tensor = alice.from_raw(array![8, 17, 20]);
        let inputs = vec![a, b, c];
        let shares: Vec<_> = inputs.into_iter().map(|x| rep.share(&sess, &x)).collect();
        let sum = rep.add_n(&sess, &shares);
        let opened_result = alice.reveal(&sess, &sum);
        assert_eq!(expected, opened_result);

        // 128 bit
        let a: HostRing128Tensor = alice.from_raw(array![[1, 2, 3], [2, 3, 4]]);
        let b: HostRing128Tensor = bob.from_raw(array![[2, 3, 4], [2, 3, 4]]);
        let c: HostRing128Tensor = carole.from_raw(array![[5, 12, 13], [1, 2, 3]]);
        let expected: HostRing128Tensor = alice.from_raw(array![[8, 17, 20], [5, 8, 11]]);
        let inputs = vec![a, b, c];
        let shares: Vec<_> = inputs.into_iter().map(|x| rep.share(&sess, &x)).collect();
        let sum = rep.add_n(&sess, &shares);
        let opened_result = alice.reveal(&sess, &sum);
        assert_eq!(expected, opened_result);
    }

    #[test]
    fn test_rep_sum() {
        let alice = HostPlacement::from("alice");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        let sess = SyncSession::default();

        let x: HostRing64Tensor = alice.from_raw(array![1, 2, 3]);
        let x_shared = rep.share(&sess, &x);
        let sum = rep.sum(&sess, None, &x_shared);
        let opened_result = alice.reveal(&sess, &sum);
        let expected: HostRing64Tensor = alice.from_raw(Array::from_elem([], 6));
        assert_eq!(opened_result, expected);
    }

    macro_rules! diag_op_test {
        ($func_name:ident, $tt:ty) => {
            fn $func_name() {
                let x = array![[1, 2], [3, 4]];
                let exp = array![1, 4];

                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let xr: $tt = alice.from_raw(x);
                let x_shared = rep.share(&sess, &xr);
                let diag = rep.diag(&sess, &x_shared);
                let opened_diag = alice.reveal(&sess, &diag);
                assert_eq!(opened_diag, alice.from_raw(exp))
            }
        };
    }

    diag_op_test!(rep_diag_bit, HostBitTensor);
    diag_op_test!(rep_diag_ring64, HostRing64Tensor);
    diag_op_test!(rep_diag_ring128, HostRing128Tensor);

    #[test]
    fn test_rep_diag_bit() {
        rep_diag_bit()
    }

    #[test]
    fn test_rep_diag_ring64() {
        rep_diag_ring64()
    }

    #[test]
    fn test_rep_diag_ring128() {
        rep_diag_ring128()
    }

    macro_rules! index_axis_op_test {
        ($func_name:ident, $tt:ident) => {
            fn $func_name() {
                let x = array![[[1, 2], [3, 4]], [[4, 5], [6, 7]]];
                let exp = array![[4, 5], [6, 7]];

                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let xr: $tt = alice.from_raw(x);
                let x_shared = rep.share(&sess, &xr);
                let index_axis = rep.index_axis(&sess, 0, 1, &x_shared);
                let opened_index_axis = alice.reveal(&sess, &index_axis);
                assert_eq!(opened_index_axis, alice.from_raw(exp))
            }
        };
    }

    index_axis_op_test!(rep_index_axis_bit, HostBitTensor);
    index_axis_op_test!(rep_index_axis_ring64, HostRing64Tensor);
    index_axis_op_test!(rep_index_axis_ring128, HostRing128Tensor);

    #[test]
    fn test_rep_index_axis_bit() {
        rep_index_axis_bit()
    }

    #[test]
    fn test_rep_index_axis_ring64() {
        rep_index_axis_ring64()
    }

    #[test]
    fn test_rep_index_axis_ring128() {
        rep_index_axis_ring128()
    }

    macro_rules! index_op_test {
        ($func_name:ident, $tt:ident, $n:ty) => {
            fn $func_name() {
                let x = array![[1, 2], [3, 4]];
                let exp = array![1, 2];

                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let xr: HostBitTensor = alice.from_raw(x);
                let x_shared = rep.share(&sess, &xr);
                let x_shared_bit_array =
                    RepBitArray::<ReplicatedBitTensor, $n>(x_shared, PhantomData);
                let index = rep.index(&sess, 0, &x_shared_bit_array);
                let opened_index = alice.reveal(&sess, &index);
                assert_eq!(opened_index, alice.from_raw(exp))
            }
        };
    }

    index_op_test!(rep_index_bit64, HostBitTensor, N64);
    index_op_test!(rep_index_bit128, HostBitTensor, N128);

    #[test]
    fn test_rep_index_bit64() {
        rep_index_bit64()
    }

    #[test]
    fn test_rep_index_bit128() {
        rep_index_bit128()
    }

    macro_rules! rep_add_test {
        ($func_name:ident, $tt: ident) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let x: HostRingTensor<_> = alice.from_raw(xs);
                let y: HostRingTensor<_> = alice.from_raw(ys);

                let x_shared = rep.share(&sess, &x);
                let y_shared = rep.share(&sess, &y);

                let sum = rep.add(&sess, &x_shared, &y_shared);
                let opened_sum = alice.reveal(&sess, &sum);
                assert_eq!(opened_sum, alice.from_raw(zs));
            }
        };
    }

    rep_add_test!(test_rep_add64, u64);
    rep_add_test!(test_rep_add128, u128);

    #[rstest]
    #[case(array![1_u64, 2, 3].into_dyn(),
        array![1_u64, 2, 3].into_dyn(),
        array![2_u64, 4, 6].into_dyn())
    ]
    #[case(array![-1_i64 as u64, -2_i64 as u64, -3_i64 as u64].into_dyn(),
        array![1_u64, 2, 3].into_dyn(),
        array![0_u64, 0, 0].into_dyn())
    ]
    fn test_rep_add_64(#[case] x: ArrayD<u64>, #[case] y: ArrayD<u64>, #[case] z: ArrayD<u64>) {
        test_rep_add64(x, y, z);
    }

    #[rstest]
    #[case(array![1_u128, 2, 3].into_dyn(),
        array![1_u128, 2, 3].into_dyn(),
        array![2_u128, 4, 6].into_dyn())
    ]
    #[case(array![-1_i128 as u128, -2_i128 as u128, -3_i128 as u128].into_dyn(),
        array![1_u128, 2, 3].into_dyn(),
        array![0_u128, 0, 0].into_dyn())
    ]
    fn test_rep_add_128(#[case] x: ArrayD<u128>, #[case] y: ArrayD<u128>, #[case] z: ArrayD<u128>) {
        test_rep_add128(x, y, z);
    }

    macro_rules! rep_binary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let x: HostRingTensor<_> = alice.from_raw(xs);
                let y: HostRingTensor<_> = alice.from_raw(ys);

                let x_shared = rep.share(&sess, &x);
                let y_shared = rep.share(&sess, &y);

                let sum = rep.$test_func(&sess, &x_shared, &y_shared);
                let opened_product = alice.reveal(&sess, &sum);
                assert_eq!(opened_product, alice.from_raw(zs));
            }
        };
    }

    rep_binary_func_test!(test_rep_mul64, mul<u64>);
    rep_binary_func_test!(test_rep_mul128, mul<u128>);
    rep_binary_func_test!(test_rep_dot64, dot<u64>);
    rep_binary_func_test!(test_rep_dot128, dot<u128>);

    macro_rules! pairwise_same_length {
        ($func_name:ident, $tt: ident) => {
            fn $func_name() -> impl Strategy<Value = (ArrayD<$tt>, ArrayD<$tt>)> {
                (1usize..25)
                    .prop_flat_map(|length| {
                        (
                            proptest::collection::vec(any::<$tt>(), length),
                            proptest::collection::vec(any::<$tt>(), length),
                        )
                    })
                    .prop_map(|(x, y)| {
                        let a = Array::from_shape_vec(IxDyn(&[x.len()]), x).unwrap();
                        let b = Array::from_shape_vec(IxDyn(&[y.len()]), y).unwrap();
                        (a, b)
                    })
                    .boxed()
            }
        };
    }

    pairwise_same_length!(pairwise_same_length64, u64);
    pairwise_same_length!(pairwise_same_length128, u128);

    proptest! {
        #[test]
        fn test_fuzzy_rep_mul64((a,b) in pairwise_same_length64())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u64; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_rep_mul64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_mul128((a,b) in pairwise_same_length128())
        {
            let mut target = Array::from_shape_vec(IxDyn(&[a.len()]), vec![0u128; a.len()]).unwrap();
            for i in 0..a.len() {
                target[i] = (std::num::Wrapping(a[i]) * std::num::Wrapping(b[i])).0;
            }
            test_rep_mul128(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_dot64((a,b) in pairwise_same_length64())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_rep_dot64(a, b, target);
        }

        #[test]
        fn test_fuzzy_rep_dot128((a,b) in pairwise_same_length128())
        {
            let mut target = std::num::Wrapping(0);
            for i in 0..a.len() {
                target += std::num::Wrapping(a[i]) * std::num::Wrapping(b[i]);
            }
            let target = Array::from_shape_vec(IxDyn(&[]), vec![target.0]).unwrap();
            test_rep_dot128(a, b, target);
        }

    }

    macro_rules! rep_mir_binary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, ys: $tt, zs_mir: ArrayD<$tt>, zmir_s: ArrayD<$tt>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);
                let mir = Mirrored3Placement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let x: HostRingTensor<_> = alice.from_raw(xs);
                let target_rep_mir: HostRingTensor<_> = alice.from_raw(zs_mir);
                let target_mir_rep: HostRingTensor<_> = alice.from_raw(zmir_s);

                let x_shared = rep.share(&sess, &x);
                let y_mir: Mir3Tensor<HostRingTensor<$tt>> =
                    mir.fill(&sess, ys.into(), &rep.shape(&sess, &x_shared));

                let result_rep_mir = rep.$test_func(&sess, &x_shared, &y_mir);
                let opened_result = alice.reveal(&sess, &result_rep_mir);
                assert_eq!(opened_result, target_rep_mir);

                let result_mir_rep = rep.$test_func(&sess, &y_mir, &x_shared);
                let opened_result = alice.reveal(&sess, &result_mir_rep);
                assert_eq!(opened_result, target_mir_rep);
            }
        };
    }

    rep_mir_binary_func_test!(test_rep_mir_add64, add<u64>);
    rep_mir_binary_func_test!(test_rep_mir_add128, add<u128>);
    rep_mir_binary_func_test!(test_rep_mir_sub64, sub<u64>);
    rep_mir_binary_func_test!(test_rep_mir_sub128, sub<u128>);

    #[test]
    fn test_rep_mir_add_64() {
        let x = array![0u64, 1, 2].into_dyn();
        let y = 2u64;
        let target_rep_mir = array![2u64, 3, 4].into_dyn();
        let target_mir_rep = array![2u64, 3, 4].into_dyn();
        test_rep_mir_add64(x, y, target_rep_mir, target_mir_rep);
    }

    #[test]
    fn test_rep_mir_add_128() {
        let x = array![0u128, 1, 2].into_dyn();
        let y = 2u128;
        let target_rep_mir = array![2u128, 3, 4].into_dyn();
        let target_mir_rep = array![2u128, 3, 4].into_dyn();
        test_rep_mir_add128(x, y, target_rep_mir, target_mir_rep);
    }

    #[test]
    fn test_rep_mir_sub_64() {
        let x = array![2u64, 3, 4].into_dyn();
        let y = 2u64;
        let target_rep_mir = array![0u64, 1, 2].into_dyn();
        let target_mir_rep = array![0u64, 18446744073709551615, 18446744073709551614].into_dyn();
        test_rep_mir_sub64(x, y, target_rep_mir, target_mir_rep);
    }

    #[test]
    fn test_rep_mir_sub_128() {
        let x = array![2u128, 3, 4].into_dyn();
        let y = 2u128;
        let target_rep_mir = array![0u128, 1, 2].into_dyn();
        let target_mir_rep = array![
            0u128,
            340282366920938463463374607431768211455,
            340282366920938463463374607431768211454
        ]
        .into_dyn();
        test_rep_mir_sub128(x, y, target_rep_mir, target_mir_rep);
    }

    macro_rules! rep_mir_mul_setup_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, ys: $tt, zs: ArrayD<$tt>) {
                let alice = HostPlacement::from("alice");
                let bob = HostPlacement::from("bob");
                let carole = HostPlacement::from("carole");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);
                let mir3 = Mirrored3Placement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let x: HostRingTensor<_> = alice.from_raw(xs);
                let target: HostRingTensor<_> = alice.from_raw(zs);

                let x_shared = rep.share(&sess, &x);

                let s0 = alice.from_raw(RawShape(vec![1]));
                let s1 = bob.from_raw(RawShape(vec![1]));
                let s2 = carole.from_raw(RawShape(vec![1]));

                let mir_shape = ReplicatedShape {
                    shapes: [s0, s1, s2],
                };

                let y_mir: Mir3Tensor<HostRingTensor<$tt>> =
                    mir3.fill(&sess, ys.into(), &mir_shape);

                let result_rep_mir = rep.$test_func(&sess, &x_shared, &y_mir);
                let opened_result = alice.reveal(&sess, &result_rep_mir);
                assert_eq!(opened_result, target);

                let result_mir_rep = rep.$test_func(&sess, &y_mir, &x_shared);
                let opened_result = alice.reveal(&sess, &result_mir_rep);
                assert_eq!(opened_result, target);
            }
        };
    }

    rep_mir_mul_setup_func_test!(test_rep_mir_mul64, mul<u64>);
    rep_mir_mul_setup_func_test!(test_rep_mir_mul128, mul<u128>);

    #[test]
    fn test_rep_mir_mul_64() {
        let x = array![0u64, 1, 2].into_dyn();
        let y = 2u64;
        let target = array![0u64, 2, 4].into_dyn();
        test_rep_mir_mul64(x, y, target);
    }

    #[test]
    fn test_rep_mir_mul_128() {
        let x = array![0u128, 1, 2].into_dyn();
        let y = 2u128;
        let target = array![0u128, 2, 4].into_dyn();
        test_rep_mir_mul128(x, y, target);
    }

    macro_rules! rep_truncation_test {
        ($func_name:ident, $tt: ident) => {
            fn $func_name(xs: ArrayD<$tt>, amount: u32, ys: ArrayD<$tt>) {
                let alice = HostPlacement::from("alice");
                let bob = HostPlacement::from("bob");
                let carole = HostPlacement::from("carole");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let alice_x1: HostRingTensor<_> = alice.from_raw(xs.clone());
                let alice_rep = rep.share(&sess, &alice_x1);
                let alice_tr = rep.trunc_pr(&sess, amount, &alice_rep);
                let alice_open = alice.reveal(&sess, &alice_tr);

                let alice_y: HostRingTensor<_> = alice.from_raw(ys.clone());
                assert_eq!(alice_open.1, alice_y.1); // make sure placements are equal

                // truncation can be off by 1
                for (i, value) in alice_y.0.iter().enumerate() {
                    let diff = value - &alice_open.0[i];
                    assert!(
                        diff == std::num::Wrapping(1 as $tt)
                            || diff == std::num::Wrapping($tt::MAX)
                            || diff == std::num::Wrapping(0),
                        "difference = {}, lhs = {}, rhs = {}",
                        diff,
                        value,
                        &alice_open.0[i]
                    );
                }

                let bob_x1: HostRingTensor<_> = bob.from_raw(xs.clone());
                let bob_rep = rep.share(&sess, &bob_x1);
                let bob_tr = rep.trunc_pr(&sess, amount, &bob_rep);
                let bob_open = bob.reveal(&sess, &bob_tr);

                let bob_y: HostRingTensor<_> = bob.from_raw(ys.clone());
                assert_eq!(bob_open.1, bob);

                for (i, value) in bob_y.0.iter().enumerate() {
                    let diff = value - &bob_open.0[i];
                    assert!(
                        diff == std::num::Wrapping(1 as $tt)
                            || diff == std::num::Wrapping($tt::MAX)
                            || diff == std::num::Wrapping(0),
                        "difference = {}, lhs = {}, rhs = {}",
                        diff,
                        value,
                        &bob_open.0[i]
                    );
                }

                let carole_x1: HostRingTensor<_> = carole.from_raw(xs.clone());
                let carole_rep = rep.share(&sess, &carole_x1);
                let carole_tr = rep.trunc_pr(&sess, amount, &carole_rep);
                let carole_open = carole.reveal(&sess, &carole_tr);

                let carole_y: HostRingTensor<_> = bob.from_raw(ys.clone());
                assert_eq!(carole_open.1, carole);

                for (i, value) in carole_y.0.iter().enumerate() {
                    let diff = value - &carole_open.0[i];
                    assert!(
                        diff == std::num::Wrapping(1 as $tt)
                            || diff == std::num::Wrapping($tt::MAX)
                            || diff == std::num::Wrapping(0),
                        "difference = {}, lhs = {}, rhs = {}",
                        diff,
                        value,
                        &carole_open.0[i]
                    );
                }
            }
        };
    }

    rep_truncation_test!(test_rep_truncation64, u64);
    rep_truncation_test!(test_rep_truncation128, u128);

    #[rstest]
    #[case(array![1_u64, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        1,
        array![0_u64, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952].into_dyn())
    ]
    #[case(array![1_u64, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        62,
        array![0_u64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1].into_dyn())
    ]
    #[case(array![1_u64, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        61,
        array![0_u64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2].into_dyn())
    ]
    #[case(array![-10_i64 as u64].into_dyn(), 1, array![-5_i64 as u64].into_dyn())]
    #[case(array![-10_i64 as u64].into_dyn(), 0, array![-10_i64 as u64].into_dyn())]
    #[case(array![-1152921504606846976_i64 as u64].into_dyn(), 60, array![-1_i64 as u64].into_dyn())]
    fn test_rep_truncation_64(
        #[case] x: ArrayD<u64>,
        #[case] amount: u32,
        #[case] target: ArrayD<u64>,
    ) {
        test_rep_truncation64(x, amount, target);
    }

    #[rstest]
    #[case(array![1_u128, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        1,
        array![0_u128, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952].into_dyn())
    ]
    #[case(array![1_u128, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        62,
        array![0_u128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1].into_dyn())
    ]
    #[case(array![1_u128, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904].into_dyn(),
        61,
        array![0_u128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2].into_dyn())
    ]
    #[case(array![-10_i128 as u128].into_dyn(), 1, array![-5_i128 as u128].into_dyn())]
    #[case(array![-10_i128 as u128].into_dyn(), 0, array![-10_i128 as u128].into_dyn())]
    #[case(array![-1152921504606846976_i128 as u128].into_dyn(), 60, array![-1_i128 as u128].into_dyn())]
    fn test_rep_truncation_128(
        #[case] x: ArrayD<u128>,
        #[case] amount: u32,
        #[case] target: ArrayD<u128>,
    ) {
        test_rep_truncation128(x, amount, target);
    }

    fn any_bounded_u64() -> impl Strategy<Value = u64> {
        any::<u64>().prop_map(|x| (x >> 2) - 1)
    }

    fn any_bounded_u128() -> impl Strategy<Value = u128> {
        any::<u128>().prop_map(|x| (x >> 2) - 1)
    }

    proptest! {

        #[test]
        fn test_fuzzy_rep_trunc64(raw_vector in proptest::collection::vec(any_bounded_u64(), 1..5), amount in 0u32..62
        ) {
            let target = raw_vector.iter().map(|x| x >> amount).collect::<Vec<_>>();
            test_rep_truncation64(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), amount, Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }

        #[test]
        fn test_fuzzy_rep_trunc128(raw_vector in proptest::collection::vec(any_bounded_u128(), 1..5), amount in 0u32..126
        ) {
            let target = raw_vector.iter().map(|x| x >> amount).collect::<Vec<_>>();
            test_rep_truncation128(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), amount, Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }
    }

    macro_rules! rep_unary_func_test {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, zs: ArrayD<$tt>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let x: HostRingTensor<_> = alice.from_raw(xs);
                let x_shared = rep.share(&sess, &x);
                let result: RepTensor<HostRingTensor<$tt>> = rep.$test_func(&sess, &x_shared);
                let opened_result = alice.reveal(&sess, &result);
                assert_eq!(opened_result, alice.from_raw(zs));
            }
        };
    }

    rep_unary_func_test!(test_rep_msb64, msb<u64>);
    rep_unary_func_test!(test_rep_msb128, msb<u128>);

    #[rstest]
    #[case(array![-10_i64 as u64, -100_i64 as u64, -200000_i64 as u64, 0, 1].into_dyn(), array![1_u64, 1, 1, 0, 0].into_dyn())]
    fn test_rep_msb_64(#[case] x: ArrayD<u64>, #[case] target: ArrayD<u64>) {
        test_rep_msb64(x, target);
    }

    #[rstest]
    #[case(array![-10_i128 as u128, -100_i128 as u128, -200000_i128 as u128, 0, 1].into_dyn(), array![1_u128, 1, 1, 0, 0].into_dyn())]
    fn test_rep_msb_128(#[case] x: ArrayD<u128>, #[case] target: ArrayD<u128>) {
        test_rep_msb128(x, target);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn test_fuzzy_rep_msb64(raw_vector in proptest::collection::vec(any::<i64>().prop_map(|x| x as u64), 1..5)) {
            let target = raw_vector.iter().map(|x|
                (*x as i64).is_negative() as u64
            ).collect::<Vec<_>>();
            test_rep_msb64(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }

        #[test]
        fn test_fuzzy_rep_msb128(raw_vector in proptest::collection::vec(any::<i128>().prop_map(|x| x as u128), 1..5)) {
            let target = raw_vector.iter().map(|x|
                (*x as i128).is_negative() as u128
            ).collect::<Vec<_>>();
            test_rep_msb128(Array::from_shape_vec(IxDyn(&[raw_vector.len()]), raw_vector).unwrap(), Array::from_shape_vec(IxDyn(&[target.len()]), target).unwrap());
        }
    }

    #[rstest]
    #[case(array![0_u8, 1, 0].into_dyn())]
    fn test_ring_inject(#[case] xs: ArrayD<u8>) {
        let alice = HostPlacement::from("alice");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        let sess = SyncSession::default();

        let x: HostBitTensor = alice.from_raw(xs.clone());
        let x_shared = rep.share(&sess, &x);

        let x_ring64: ReplicatedRing64Tensor = rep.ring_inject(&sess, 0, &x_shared);
        let x_ring128: ReplicatedRing128Tensor = rep.ring_inject(&sess, 0, &x_shared);

        let target64: HostRing64Tensor = alice.from_raw(xs.map(|x| *x as u64));
        let target128: HostRing128Tensor = alice.from_raw(xs.map(|x| *x as u128));

        assert_eq!(alice.reveal(&sess, &x_ring64), target64);
        assert_eq!(alice.reveal(&sess, &x_ring128), target128);

        let shifted_x_ring64: ReplicatedRing64Tensor = rep.ring_inject(&sess, 20, &x_shared);
        assert_eq!(
            alice.reveal(&sess, &shifted_x_ring64),
            alice.shl(&sess, 20, &target64)
        );
    }

    rep_unary_func_test!(test_rep_abs64, abs_as_fixedpoint<u64>);
    rep_unary_func_test!(test_rep_abs128, abs_as_fixedpoint<u128>);

    #[rstest]
    #[case(array![-10_i64 as u64, -100_i64 as u64, -200000_i64 as u64, 0, 1000].into_dyn(), array![10_u64, 100, 200000, 0, 1000].into_dyn())]
    fn test_rep_abs_64(#[case] x: ArrayD<u64>, #[case] target: ArrayD<u64>) {
        test_rep_abs64(x, target);
    }

    #[rstest]
    #[case(array![-10_i128 as u128, -100_i128 as u128, -200000_i128 as u128, 0, 1000].into_dyn(), array![10_u128, 100, 200000, 0, 1000].into_dyn())]
    fn test_rep_abs_128(#[case] x: ArrayD<u128>, #[case] target: ArrayD<u128>) {
        test_rep_abs128(x, target);
    }

    fn test_rep_bit_dec64(xs: ArrayD<u64>, zs: ArrayD<u8>) {
        let alice = HostPlacement::from("alice");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        let sess = SyncSession::default();

        let x: HostRing64Tensor = alice.from_raw(xs);
        let x_shared = rep.share(&sess, &x);
        let result: ReplicatedBitArray64 = rep.bit_decompose(&sess, &x_shared);
        let opened_result = alice.reveal(&sess, &result);
        assert_eq!(opened_result, alice.from_raw(zs));
    }

    #[rstest]
    #[case(array![1073741823].into_dyn(),
        array![
            [1_u8],[1],[1],[1],[1],[1],[1],[1],
            [1],[1],[1],[1],[1],[1],[1],[1],
            [1],[1],[1],[1],[1],[1],[1],[1],
            [1],[1],[1],[1],[1],[1],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],
        ].into_dyn()
    )]
    fn test_rep_bit_dec_64(#[case] x: ArrayD<u64>, #[case] y: ArrayD<u8>) {
        test_rep_bit_dec64(x, y);
    }

    macro_rules! rep_bit_compose_test {
        ($func_name:ident, $tt:ty) => {
            fn $func_name(xs: ArrayD<$tt>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let expected = xs.clone();

                let x: HostRingTensor<_> = alice.from_raw(xs);
                let x_shared = rep.share(&sess, &x);
                let decomposed = rep.bit_decompose(&sess, &x_shared);
                let composed = rep.bit_compose(&sess, &decomposed);
                let opened_result = alice.reveal(&sess, &composed);
                assert_eq!(opened_result, alice.from_raw(expected));
            }
        };
    }

    rep_bit_compose_test!(test_rep_bit_compose64, u64);
    rep_bit_compose_test!(test_rep_bit_compose128, u128);

    #[rstest]
    #[case(array![1073741823i128 as u128, 0, 6].into_dyn())]
    fn test_rep_bit_compose_128(#[case] xs: ArrayD<u128>) {
        test_rep_bit_compose128(xs);
    }

    #[rstest]
    #[case(array![1073741823, 0, 6].into_dyn())]
    fn test_rep_bit_compose_64(#[case] x: ArrayD<u64>) {
        test_rep_bit_compose64(x);
    }

    #[test]
    fn test_bit_dec_different_plc() {
        let xs = array![1073741823].into_dyn();
        let zs = array![
            [1_u8],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        ]
        .into_dyn();

        let alice = HostPlacement::from("alice");
        let bob = HostPlacement::from("bob");
        let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

        let sess = SyncSession::default();

        let x: HostRing64Tensor = bob.from_raw(xs);
        let x_shared = rep.share(&sess, &x);
        let result: ReplicatedBitArray64 = rep.bit_decompose(&sess, &x_shared);
        let opened_result = alice.reveal(&sess, &result);
        assert_eq!(opened_result, alice.from_raw(zs));
    }

    macro_rules! rep_prefix_op_bit_test {
        ($func_name:ident, $test_func: ident) => {
            fn $func_name(x: ArrayD<u64>, y_target: Vec<u8>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let sess = SyncSession::default();

                let x: HostRing64Tensor = alice.from_raw(x);
                let x_shared = rep.share(&sess, &x);
                let x_bits: ReplicatedBitArray64 = rep.bit_decompose(&sess, &x_shared);
                let x_bits_vec: Vec<_> = (0..64).map(|i| rep.index(&sess, i, &x_bits)).collect();

                let out = rep.$test_func(&sess, x_bits_vec);

                for (i, el) in out.iter().enumerate() {
                    let b = alice.reveal(&sess, el);
                    assert_eq!(b.0.data[0] as u8, y_target[i]);
                }
            }
        };
    }

    rep_prefix_op_bit_test!(test_rep_prefix_or, prefix_or);
    rep_prefix_op_bit_test!(test_rep_prefix_and, prefix_and);

    #[test]
    fn test_prefix_or() {
        let x = array![1024u64].into_dyn();
        let y_target = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
        ];
        test_rep_prefix_or(x, y_target);
    }

    #[test]
    fn test_prefix_and() {
        let x = array![7u64].into_dyn();
        let y_target = vec![
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];
        test_rep_prefix_and(x, y_target);
    }

    macro_rules! rep_binary_func_test_bit {
        ($func_name:ident, $test_func: ident<$tt: ty>) => {
            fn $func_name(xs: ArrayD<$tt>, ys: ArrayD<$tt>, zs: ArrayD<u8>) {
                let alice = HostPlacement::from("alice");
                let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

                let x: HostRingTensor<_> = alice.from_raw(xs);
                let y: HostRingTensor<_> = alice.from_raw(ys);

                let sess = SyncSession::default();

                let x_shared = rep.share(&sess, &x);
                let y_shared = rep.share(&sess, &y);

                let sum = rep.$test_func(&sess, &x_shared, &y_shared);
                let opened_product = alice.reveal(&sess, &sum);
                assert_eq!(opened_product, alice.from_raw(zs));
            }
        };
    }

    rep_binary_func_test_bit!(test_rep_lt64, less<u64>);
    rep_binary_func_test_bit!(test_rep_lt128, less<u128>);

    #[test]
    fn test_rep_lt_64() {
        let x = array![0u64, 1, 2, -1_i64 as u64, -2_i64 as u64].into_dyn();
        let y = array![
            -1_i64 as u64,
            -2_i64 as u64,
            3_u64,
            -1_i64 as u64,
            -1_i64 as u64
        ]
        .into_dyn();
        let target = array![0, 0, 1, 0, 1].into_dyn();
        test_rep_lt64(x, y, target);
    }

    #[test]
    fn test_rep_lt_128() {
        let x = array![0u128, 1, 2, -1_i128 as u128, -2_i128 as u128].into_dyn();
        let y = array![
            -1_i128 as u128,
            -2_i128 as u128,
            3_u128,
            -1_i128 as u128,
            -1_i128 as u128
        ]
        .into_dyn();
        let target = array![0, 0, 1, 0, 1].into_dyn();
        test_rep_lt128(x, y, target);
    }

    rep_binary_func_test_bit!(test_rep_gt64, greater_than<u64>);
    rep_binary_func_test_bit!(test_rep_gt128, greater_than<u128>);

    #[test]
    fn test_rep_gt_64() {
        let x = array![0u64, 1, 2, -1_i64 as u64, -2_i64 as u64, 2u64.pow(62)].into_dyn();
        let y = array![
            -1_i64 as u64,
            -2_i64 as u64,
            3_u64,
            -1_i64 as u64,
            -1_i64 as u64,
            (-4611686018427387904_i64 + 1) as u64 // -2^62+1
        ]
        .into_dyn();
        let target = array![1, 1, 0, 0, 0, 1].into_dyn();
        test_rep_gt64(x, y, target);
    }

    #[test]
    fn test_rep_gt_128() {
        let x = array![0u128, 1, 2, -1_i128 as u128, -2_i128 as u128].into_dyn();
        let y = array![
            -1_i128 as u128,
            -2_i128 as u128,
            3_u128,
            -1_i128 as u128,
            -1_i128 as u128
        ]
        .into_dyn();
        let target = array![1, 1, 0, 0, 0].into_dyn();
        test_rep_gt128(x, y, target);
    }
}
