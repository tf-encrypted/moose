//! Various operations for additive placements
use super::*;
use crate::computation::{
    AdtAddOp, AdtFillOp, AdtMulOp, AdtRevealOp, AdtShlOp, AdtSubOp, Constant, HostPlacement,
    Placed, ShapeOp,
};
use crate::error::Result;
use crate::kernels::*;
use crate::types::*;
use macros::with_context;

impl ShapeOp {
    pub(crate) fn adt_kernel<S: Session, HostT, ShapeT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<HostT>,
    ) -> Result<AdtShape<ShapeT>>
    where
        HostPlacement: PlacementShape<S, HostT, ShapeT>,
    {
        let (player0, player1) = adt.host_placements();
        let AdtTensor { shares: [x0, x1] } = &x;
        Ok(AdtShape {
            shapes: [player0.shape(sess, x0), player1.shape(sess, x1)],
        })
    }
}

modelled_kernel! {
    PlacementFill::fill, AdtFillOp{value: Constant},
    [
        (AdditivePlacement, (HostShape) -> AdditiveRing64Tensor => [hybrid] Self::host_kernel),
        (AdditivePlacement, (HostShape) -> AdditiveRing128Tensor => [hybrid] Self::host_kernel),
        (AdditivePlacement, (AdditiveShape) -> AdditiveRing64Tensor => [concrete] Self::adt_kernel),
        (AdditivePlacement, (AdditiveShape) -> AdditiveRing128Tensor => [concrete] Self::adt_kernel),
    ]
}

impl AdtFillOp {
    fn host_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        plc: &AdditivePlacement,
        value: Constant,
        shape: ShapeT,
    ) -> Result<AdtTensor<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return Mirrored2Tensor, but we don't have that type yet

        let (player0, player1) = plc.host_placements();

        let shares = [
            player0.fill(sess, value, &shape),
            player1.fill(sess, Constant::Ring64(0), &shape),
        ];
        Ok(AdtTensor { shares })
    }

    fn adt_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        plc: &AdditivePlacement,
        value: Constant,
        shape: AdtShape<ShapeT>,
    ) -> Result<AdtTensor<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return Mirrored2Tensor, but we don't have that type yet

        let AdtShape {
            shapes: [shape0, shape1],
        } = &shape;

        let (player0, player1) = plc.host_placements();

        let shares = [
            player0.fill(sess, value, shape0),
            player1.fill(sess, Constant::Ring64(0), shape1),
        ];
        Ok(AdtTensor { shares })
    }
}

modelled_kernel! {
    PlacementReveal::reveal, AdtRevealOp,
    [
        (HostPlacement, (AdditiveRing64Tensor) -> HostRing64Tensor => [hybrid] Self::kernel),
        (HostPlacement, (AdditiveRing128Tensor) -> HostRing128Tensor => [hybrid] Self::kernel),
        (HostPlacement, (AdditiveBitTensor) -> HostBitTensor => [hybrid] Self::kernel),
    ]
}

impl AdtRevealOp {
    fn kernel<S: Session, RingT>(
        sess: &S,
        plc: &HostPlacement,
        xe: AdtTensor<RingT>,
    ) -> Result<RingT>
    where
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
    {
        let AdtTensor { shares: [x0, x1] } = &xe;
        Ok(with_context!(plc, sess, x0 + x1))
    }
}

modelled_kernel! {
    PlacementAdd::add, AdtAddOp,
    [
        (AdditivePlacement, (AdditiveRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [concrete] Self::adt_adt_kernel),
        // TODO(Morten) replace host tensors with mirrored tensors in the below
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (AdditiveBitTensor, HostBitTensor) -> AdditiveBitTensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::host_adt_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::host_adt_kernel),
        (AdditivePlacement, (HostBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [hybrid] Self::host_adt_kernel),
    ]
}

impl AdtAddOp {
    fn adt_adt_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<HostRingT>,
        y: AdtTensor<HostRingT>,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, player1) = adt.host_placements();

        let AdtTensor { shares: [x0, x1] } = &x;
        let AdtTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x0 + y0);
        let z1 = with_context!(player1, sess, x1 + y1);

        Ok(AdtTensor { shares: [z0, z1] })
    }

    fn adt_host_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<HostRingT>,
        y: HostRingT,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
        AdditivePlacement: PlacementPlace<S, AdtTensor<HostRingT>>,
    {
        let (player0, player1) = adt.host_placements();
        let y_plc = y.placement()?;

        let AdtTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, sess, x0 + y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, sess, x1 + y)],
            _ => [with_context!(player0, sess, x0 + y), x1],
        };
        Ok(adt.place(sess, AdtTensor { shares }))
    }

    fn host_adt_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: HostRingT,
        y: AdtTensor<HostRingT>,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
        AdditivePlacement: PlacementPlace<S, AdtTensor<HostRingT>>,
    {
        let (player0, player1) = adt.host_placements();
        let x_plc = x.placement()?;

        let AdtTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, sess, y0 + x), y1],
            _ if x_plc == player1 => [y0, with_context!(player1, sess, x + y1)],
            _ => [with_context!(player0, sess, x + y0), y1],
        };
        Ok(adt.place(sess, AdtTensor { shares }))
    }
}

modelled_kernel! {
    PlacementSub::sub, AdtSubOp,
    [
        (AdditivePlacement, (AdditiveRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [concrete] Self::adt_adt_kernel),
        // TODO(Morten) replace host tensors with mirrored tensors in the below
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::host_adt_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::host_adt_kernel),
    ]
}

impl AdtSubOp {
    fn adt_adt_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<HostRingT>,
        y: AdtTensor<HostRingT>,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, player1) = adt.host_placements();

        let AdtTensor { shares: [x0, x1] } = &x;
        let AdtTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x0 - y0);
        let z1 = with_context!(player1, sess, x1 - y1);

        Ok(AdtTensor { shares: [z0, z1] })
    }

    fn adt_host_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<HostRingT>,
        y: HostRingT,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
        AdditivePlacement: PlacementPlace<S, AdtTensor<HostRingT>>,
    {
        let (player0, player1) = adt.host_placements();
        let y_plc = y.placement()?;

        let AdtTensor { shares: [x0, x1] } = x;

        let shares = match () {
            _ if y_plc == player0 => [with_context!(player0, sess, x0 - y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, sess, x1 - y)],
            _ => [with_context!(player0, sess, x0 - y), x1],
        };
        Ok(adt.place(sess, AdtTensor { shares }))
    }

    fn host_adt_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: HostRingT,
        y: AdtTensor<HostRingT>,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<S, HostRingT, HostRingT, HostRingT>,
        HostPlacement: PlacementNeg<S, HostRingT, HostRingT>,
        AdditivePlacement: PlacementPlace<S, AdtTensor<HostRingT>>,
    {
        let (player0, player1) = adt.host_placements();
        let x_plc = x.placement()?;

        let AdtTensor { shares: [y0, y1] } = y;

        let shares = match () {
            _ if x_plc == player0 => [with_context!(player0, sess, x - y0), player1.neg(sess, &y1)],
            _ if x_plc == player1 => [player0.neg(sess, &y0), with_context!(player1, sess, x - y1)],
            _ => [with_context!(player0, sess, x - y0), player1.neg(sess, &y1)],
        };
        Ok(adt.place(sess, AdtTensor { shares }))
    }
}

modelled_kernel! {
    PlacementMul::mul, AdtMulOp,
    [
        // TODO(Morten) replace host tensors with mirrored tensors in the below
        (AdditivePlacement, (HostRing64Tensor, AdditiveRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::host_adt_kernel),
        (AdditivePlacement, (AdditiveRing64Tensor, HostRing64Tensor) -> AdditiveRing64Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (AdditiveRing128Tensor, HostRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (HostRing128Tensor, AdditiveRing128Tensor) -> AdditiveRing128Tensor => [hybrid] Self::host_adt_kernel),
        (AdditivePlacement, (AdditiveBitTensor, HostBitTensor) -> AdditiveBitTensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (HostBitTensor, AdditiveBitTensor) -> AdditiveBitTensor => [hybrid] Self::host_adt_kernel),

    ]
}

impl AdtMulOp {
    fn host_adt_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: HostRingT,
        y: AdtTensor<HostRingT>,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, player1) = adt.host_placements();

        let AdtTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x * y0);
        let z1 = with_context!(player1, sess, x * y1);

        Ok(AdtTensor { shares: [z0, z1] })
    }

    fn adt_host_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<HostRingT>,
        y: HostRingT,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostRingT: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, player1) = adt.host_placements();

        let AdtTensor { shares: [x0, x1] } = &x;

        let z0 = with_context!(player0, sess, x0 * y);
        let z1 = with_context!(player1, sess, x1 * y);

        Ok(AdtTensor { shares: [z0, z1] })
    }
}

modelled_kernel! {
    PlacementShl::shl, AdtShlOp{amount: usize},
    [
        (AdditivePlacement, (AdditiveRing64Tensor) -> AdditiveRing64Tensor => [concrete] Self::kernel),
        (AdditivePlacement, (AdditiveRing128Tensor) -> AdditiveRing128Tensor => [concrete] Self::kernel),
    ]
}

impl AdtShlOp {
    fn kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &AdditivePlacement,
        amount: usize,
        x: AdtTensor<HostRingT>,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostPlacement: PlacementShl<S, HostRingT, HostRingT>,
    {
        let (player0, player1) = plc.host_placements();
        let AdtTensor { shares: [x0, x1] } = &x;
        let z0 = player0.shl(sess, amount, x0);
        let z1 = player1.shl(sess, amount, x1);
        Ok(AdtTensor { shares: [z0, z1] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        computation::{KnownType, Operation, Operator, Placement, RingAddOp},
        host::AbstractHostRingTensor,
        symbolic::{Symbolic, SymbolicHandle, SymbolicSession},
    };
    use ndarray::array;

    #[test]
    fn test_add() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let bob = HostPlacement {
            owner: "bob".into(),
        };
        let adt = AdditivePlacement {
            owners: ["alice".into(), "bob".into()],
        };

        let x = AdditiveRing64Tensor {
            shares: [
                AbstractHostRingTensor::from_raw_plc(array![1, 2, 3], alice.clone()),
                AbstractHostRingTensor::from_raw_plc(array![4, 5, 6], bob.clone()),
            ],
        };

        let y = AdditiveRing64Tensor {
            shares: [
                AbstractHostRingTensor::from_raw_plc(array![7, 8, 9], alice.clone()),
                AbstractHostRingTensor::from_raw_plc(array![1, 2, 3], bob.clone()),
            ],
        };

        let sess = SyncSession::default();
        let AdtTensor { shares: [z0, z1] } = adt.add(&sess, &x, &y);

        assert_eq!(
            z0,
            AbstractHostRingTensor::from_raw_plc(array![1 + 7, 2 + 8, 3 + 9], alice.clone())
        );
        assert_eq!(
            z1,
            AbstractHostRingTensor::from_raw_plc(array![4 + 1, 5 + 2, 6 + 3], bob.clone())
        );

        let r_alice = AbstractHostRingTensor::from_raw_plc(array![7, 8, 9], alice.clone());
        let AdtTensor { shares: [zr0, zr1] } = adt.add(&sess, &x, &r_alice);

        assert_eq!(
            zr0,
            AbstractHostRingTensor::from_raw_plc(array![1 + 7, 2 + 8, 3 + 9], alice.clone())
        );
        assert_eq!(
            zr1,
            AbstractHostRingTensor::from_raw_plc(array![4, 5, 6], bob.clone())
        );

        let r_bob = AbstractHostRingTensor::from_raw_plc(array![7, 8, 9], bob.clone());
        let AdtTensor {
            shares: [zrb0, zrb1],
        } = adt.add(&sess, &x, &r_bob);

        assert_eq!(
            zrb0,
            AbstractHostRingTensor::from_raw_plc(array![1, 2, 3], alice)
        );
        assert_eq!(
            zrb1,
            AbstractHostRingTensor::from_raw_plc(array![4 + 7, 5 + 8, 6 + 9], bob)
        );
    }

    #[test]
    fn test_symbolic_add() {
        let adt = AdditivePlacement {
            owners: ["alice".into(), "bob".into()],
        };

        let x: <AdditiveRing64Tensor as KnownType<SymbolicSession>>::Type =
            Symbolic::Symbolic(SymbolicHandle {
                op: "x".into(),
                plc: adt.clone(),
            });

        let y: <AdditiveRing64Tensor as KnownType<SymbolicSession>>::Type =
            Symbolic::Symbolic(SymbolicHandle {
                op: "x".into(),
                plc: adt.clone(),
            });

        let sess = SymbolicSession::default();
        let z = adt.add(&sess, &x, &y);

        let op_name = match z {
            Symbolic::Symbolic(handle) => {
                assert_eq!("op_0", handle.op);
                handle.op
            }
            _ => panic!("Expected a symbolic result from the symbolic addition"),
        };

        sess.ops_iter(|mut iter| match iter.find(|o| o.name == op_name) {
            None => panic!("Newly created operation was not placed on graph"),
            Some(op) => assert!(matches!(
                op,
                Operation {
                    kind: Operator::AdtAdd(AdtAddOp { sig: _ }),
                    ..
                }
            )),
        });
    }

    #[test]
    fn test_concrete_symbolic_add() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let bob = HostPlacement {
            owner: "bob".into(),
        };

        let adt = AdditivePlacement {
            owners: ["alice".into(), "bob".into()],
        };

        let x: <AdditiveRing64Tensor as KnownType<SymbolicSession>>::Type =
            Symbolic::Concrete(AdtTensor {
                shares: [
                    Symbolic::Symbolic(SymbolicHandle {
                        op: "x0".into(),
                        plc: alice.clone(),
                    }),
                    Symbolic::Symbolic(SymbolicHandle {
                        op: "x1".into(),
                        plc: bob.clone(),
                    }),
                ],
            });

        let y: <AdditiveRing64Tensor as KnownType<SymbolicSession>>::Type =
            Symbolic::Concrete(AdtTensor {
                shares: [
                    Symbolic::Symbolic(SymbolicHandle {
                        op: "y0".into(),
                        plc: alice,
                    }),
                    Symbolic::Symbolic(SymbolicHandle {
                        op: "y1".into(),
                        plc: bob,
                    }),
                ],
            });

        let sess = SymbolicSession::default();
        let z = adt.add(&sess, &x, &y);

        match &z {
            Symbolic::Concrete(AdtTensor { shares: [z0, z1] }) => {
                match z0 {
                    Symbolic::Symbolic(handle) => {
                        assert_eq!("op_0", handle.op);
                    }
                    _ => panic!("Expected a symbolic result from the symbolic addition"),
                }
                match z1 {
                    Symbolic::Symbolic(handle) => {
                        assert_eq!("op_1", handle.op);
                    }
                    _ => panic!("Expected a symbolic result from the symbolic addition"),
                }
            }
            _ => {
                panic!("Expected a concrete result from the symbolic addition on a concrete value")
            }
        }

        sess.ops_iter(|mut iter| {
            assert!(iter.any(|o| matches!(o,
                Operation {
                    name,
                    kind: Operator::RingAdd(RingAddOp { sig: _ }),
                    inputs,
                    placement: Placement::Host(HostPlacement { owner }),
                    ..
                }
                if name == "op_0" && inputs == &vec!["x0", "y0"] && owner.0 == "alice"
            )));
        });

        sess.ops_iter(|mut iter| {
            assert!(iter.any(|o| matches!(o,
                Operation {
                    name,
                    kind: Operator::RingAdd(RingAddOp { sig: _ }),
                    inputs,
                    placement: Placement::Host(HostPlacement { owner }),
                    ..
                }
                if name == "op_1" && inputs == &vec!["x1", "y1"] && owner.0 == "bob"
            )));
        });
    }
}
