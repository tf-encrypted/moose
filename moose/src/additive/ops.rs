//! Various operations for additive placements
use super::*;
use crate::computation::{AddOp, Constant, FillOp, MulOp, Placed, RevealOp, ShapeOp, ShlOp, SubOp};
use crate::error::Result;
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::kernels::*;
use moose_macros::with_context;

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

impl FillOp {
    pub(crate) fn adt_host_kernel<S: Session, ShapeT, RingT>(
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

    pub(crate) fn adt_adt_kernel<S: Session, ShapeT, RingT>(
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

impl RevealOp {
    pub(crate) fn host_adt_kernel<S: Session, RingT>(
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

impl AddOp {
    pub(crate) fn adt_adt_kernel<S: Session, HostRingT>(
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

    pub(crate) fn adt_host_kernel<S: Session, HostRingT>(
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

    pub(crate) fn host_adt_kernel<S: Session, HostRingT>(
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

impl SubOp {
    pub(crate) fn adt_adt_kernel<S: Session, HostRingT>(
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

    pub(crate) fn adt_host_kernel<S: Session, HostRingT>(
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

    pub(crate) fn host_adt_kernel<S: Session, HostRingT>(
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

impl MulOp {
    pub(crate) fn host_adt_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: HostRingT,
        y: AdtTensor<HostRingT>,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostPlacement: PlacementMul<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, player1) = adt.host_placements();

        let AdtTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, sess, x * y0);
        let z1 = with_context!(player1, sess, x * y1);

        Ok(AdtTensor { shares: [z0, z1] })
    }

    pub(crate) fn adt_host_kernel<S: Session, HostRingT>(
        sess: &S,
        adt: &AdditivePlacement,
        x: AdtTensor<HostRingT>,
        y: HostRingT,
    ) -> Result<AdtTensor<HostRingT>>
    where
        HostPlacement: PlacementMul<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, player1) = adt.host_placements();

        let AdtTensor { shares: [x0, x1] } = &x;

        let z0 = with_context!(player0, sess, x0 * y);
        let z1 = with_context!(player1, sess, x1 * y);

        Ok(AdtTensor { shares: [z0, z1] })
    }
}

impl ShlOp {
    pub(crate) fn adt_kernel<S: Session, HostRingT>(
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
    #[cfg(feature = "compile")]
    use crate::computation::{KnownType, Operation, Operator, Placement};
    #[cfg(feature = "compile")]
    use crate::execution::symbolic::{Symbolic, SymbolicHandle, SymbolicSession};
    use crate::prelude::*;
    use crate::types::*;
    use ndarray::prelude::*;

    #[cfg(feature = "sync_execute")]
    #[test]
    fn test_add() {
        let alice = HostPlacement::from("alice");
        let bob = HostPlacement::from("bob");
        let adt = AdditivePlacement::from(["alice", "bob"]);

        let x = AdditiveRing64Tensor {
            shares: [
                alice.from_raw(array![1, 2, 3]),
                bob.from_raw(array![4, 5, 6]),
            ],
        };

        let y = AdditiveRing64Tensor {
            shares: [
                alice.from_raw(array![7, 8, 9]),
                bob.from_raw(array![1, 2, 3]),
            ],
        };

        let sess = SyncSession::default();
        let AdtTensor { shares: [z0, z1] } = adt.add(&sess, &x, &y);

        assert_eq!(z0, alice.from_raw(array![1 + 7, 2 + 8, 3 + 9]));
        assert_eq!(z1, bob.from_raw(array![4 + 1, 5 + 2, 6 + 3]));

        let r_alice: HostRing64Tensor = alice.from_raw(array![7, 8, 9]);
        let AdtTensor { shares: [zr0, zr1] } = adt.add(&sess, &x, &r_alice);

        assert_eq!(zr0, alice.from_raw(array![1 + 7, 2 + 8, 3 + 9]));
        assert_eq!(zr1, bob.from_raw(array![4, 5, 6]));

        let r_bob: HostRing64Tensor = bob.from_raw(array![7, 8, 9]);
        let AdtTensor {
            shares: [zrb0, zrb1],
        } = adt.add(&sess, &x, &r_bob);

        assert_eq!(zrb0, alice.from_raw(array![1, 2, 3]));
        assert_eq!(zrb1, bob.from_raw(array![4 + 7, 5 + 8, 6 + 9]));
    }

    #[cfg(feature = "compile")]
    #[test]
    fn test_symbolic_add() {
        let adt = AdditivePlacement::from(["alice", "bob"]);

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
                    kind: Operator::Add(AddOp { sig: _ }),
                    ..
                }
            )),
        });
    }

    #[cfg(feature = "compile")]
    #[test]
    fn test_concrete_symbolic_add() {
        let alice = HostPlacement::from("alice");
        let bob = HostPlacement::from("bob");
        let adt = AdditivePlacement::from(["alice", "bob"]);

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
                    kind: Operator::Add(AddOp { sig: _ }),
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
                    kind: Operator::Add(AddOp { sig: _ }),
                    inputs,
                    placement: Placement::Host(HostPlacement { owner }),
                    ..
                }
                if name == "op_1" && inputs == &vec!["x1", "y1"] && owner.0 == "bob"
            )));
        });
    }
}
