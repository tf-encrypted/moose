//! Support for arithmetic operators

use super::*;
use crate::fixedpoint::FixedpointTensor;
use crate::mirrored::Mir3Tensor;

impl AddOp {
    pub(crate) fn rep_rep_kernel<S: Session, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<HostRingT>,
        y: RepTensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let RepTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, sess, x00 + y00);
        let z10 = with_context!(player0, sess, x10 + y10);

        let z11 = with_context!(player1, sess, x11 + y11);
        let z21 = with_context!(player1, sess, x21 + y21);

        let z22 = with_context!(player2, sess, x22 + y22);
        let z02 = with_context!(player2, sess, x02 + y02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn mir_rep_kernel<S: Session, HostRingT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: Mir3Tensor<HostRingT>,
        y: RepTensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementShape<S, HostRingT, ShapeT>,
        HostPlacement: PlacementBroadcast<S, ShapeT, HostRingT, HostRingT>,
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let Mir3Tensor {
            values: [x0, x1, x2],
        } = x;

        let RepTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = y;

        // required ops to do mirrored/replicated ops
        let z00 = with_context!(player0, sess, x0 + y00);
        let z02 = with_context!(player2, sess, x2 + y02);

        // sometimes shapes could be different, so we make sure to do a broadcast for remaining shares.
        let s0 = player0.shape(sess, &z00);
        let s1 = player1.shape(sess, &with_context!(player1, sess, x1 + y11));
        let s2 = player2.shape(sess, &z02);

        let z10 = player0.broadcast(sess, &s0, &y10);
        let z11 = player1.broadcast(sess, &s1, &y11);
        let z21 = player1.broadcast(sess, &s1, &y21);
        let z22 = player2.broadcast(sess, &s2, &y22);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn rep_mir_kernel<S: Session, ShapeT, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<HostRingT>,
        y: Mir3Tensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,
        HostPlacement: PlacementShape<S, HostRingT, ShapeT>,
        HostPlacement: PlacementBroadcast<S, ShapeT, HostRingT, HostRingT>,
    {
        // the shapes of shares need to be the same.
        let (player0, player1, player2) = rep.host_placements();

        let Mir3Tensor {
            values: [y0, y1, y2],
        } = y;

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        // required ops to do mirrored/replicated ops
        let z00 = with_context!(player0, sess, x00 + y0);
        let z02 = with_context!(player2, sess, x02 + y2);

        // sometimes shapes could be different, so we make sure to do a broadcast for remaining shares.
        let s0 = player0.shape(sess, &z00);
        let s1 = player1.shape(sess, &with_context!(player1, sess, x11 + y1));
        let s2 = player2.shape(sess, &z02);

        let z10 = player0.broadcast(sess, &s0, &x10);
        let z11 = player1.broadcast(sess, &s1, &x11);
        let z21 = player1.broadcast(sess, &s1, &x21);
        let z22 = player2.broadcast(sess, &s2, &x22);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl AddNOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        xs: &[RepTensor<HostRingT>],
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementAddN<S, HostRingT, HostRingT>,
        HostRingT: Clone,
    {
        let (player0, player1, player2) = rep.host_placements();

        let mut z00s: Vec<HostRingT> = Vec::new();
        let mut z10s: Vec<HostRingT> = Vec::new();
        let mut z11s: Vec<HostRingT> = Vec::new();
        let mut z21s: Vec<HostRingT> = Vec::new();
        let mut z22s: Vec<HostRingT> = Vec::new();
        let mut z02s: Vec<HostRingT> = Vec::new();
        for x in xs.iter() {
            let RepTensor {
                shares: [[x00, x10], [x11, x21], [x22, x02]],
            } = &x;

            z00s.push(x00.clone());
            z10s.push(x10.clone());
            z11s.push(x11.clone());
            z21s.push(x21.clone());
            z22s.push(x22.clone());
            z02s.push(x02.clone());
        }

        let z00 = player0.add_n(sess, &z00s);
        let z10 = player0.add_n(sess, &z10s);
        let z11 = player1.add_n(sess, &z11s);
        let z21 = player1.add_n(sess, &z21s);
        let z22 = player2.add_n(sess, &z22s);
        let z02 = player2.add_n(sess, &z02s);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl SumOp {
    pub(crate) fn rep_ring_kernel<S: Session, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: Option<usize>,
        x: RepTensor<RingT>,
    ) -> Result<RepTensor<RingT>>
    where
        HostPlacement: PlacementSum<S, RingT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTensor<RingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.sum(sess, axis, x00);
        let z10 = player0.sum(sess, axis, x10);
        let z11 = player1.sum(sess, axis, x11);
        let z21 = player1.sum(sess, axis, x21);
        let z22 = player2.sum(sess, axis, x22);
        let z02 = player2.sum(sess, axis, x02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl SubOp {
    pub(crate) fn rep_rep_kernel<S: Session, R>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<R>,
        y: RepTensor<R>,
    ) -> Result<RepTensor<R>>
    where
        HostPlacement: PlacementSub<S, R, R, R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let RepTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, sess, x00 - y00);
        let z10 = with_context!(player0, sess, x10 - y10);

        let z11 = with_context!(player1, sess, x11 - y11);
        let z21 = with_context!(player1, sess, x21 - y21);

        let z22 = with_context!(player2, sess, x22 - y22);
        let z02 = with_context!(player2, sess, x02 - y02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn mir_rep_kernel<S: Session, R, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: Mir3Tensor<R>,
        y: RepTensor<R>,
    ) -> Result<RepTensor<R>>
    where
        HostPlacement: PlacementSub<S, R, R, R>,
        HostPlacement: PlacementShape<S, R, ShapeT>,
        HostPlacement: PlacementNeg<S, R, R>,
        HostPlacement: PlacementBroadcast<S, ShapeT, R, R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let Mir3Tensor {
            values: [x0, x1, x2],
        } = &x;

        let RepTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        // required ops to do mirrored/replicated op
        let z00 = with_context!(player0, sess, x0 - y00);
        let z10 = player0.neg(sess, y10);
        let z11 = player1.neg(sess, y11);
        let z21 = player1.neg(sess, y21);
        let z22 = player2.neg(sess, y22);
        let z02 = with_context!(player2, sess, x2 - y02);

        // sometimes shapes could be different, so we make sure to do a broadcast for remaining shares.
        let s0 = player0.shape(sess, &z00);
        let s1 = player1.shape(sess, &with_context!(player1, sess, x1 - y11));
        let s2 = player2.shape(sess, &z02);

        let z10 = player0.broadcast(sess, &s0, &z10);
        let z11 = player1.broadcast(sess, &s1, &z11);
        let z21 = player1.broadcast(sess, &s1, &z21);
        let z22 = player2.broadcast(sess, &s2, &z22);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn rep_mir_kernel<S: Session, R, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<R>,
        y: Mir3Tensor<R>,
    ) -> Result<RepTensor<R>>
    where
        HostPlacement: PlacementSub<S, R, R, R>,
        HostPlacement: PlacementShape<S, R, ShapeT>,
        HostPlacement: PlacementBroadcast<S, ShapeT, R, R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let Mir3Tensor {
            values: [y0, y1, y2],
        } = y;

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        // required ops to do mirrored/replicated ops
        let z00 = with_context!(player0, sess, x00 - y0);
        let z02 = with_context!(player2, sess, x02 - y2);

        // sometimes shapes could be different, so we make sure to do a broadcast for remaining shares.
        let s0 = player0.shape(sess, &z00);
        let s1 = player1.shape(sess, &with_context!(player1, sess, x11 - y1));
        let s2 = player2.shape(sess, &z02);

        let z10 = player0.broadcast(sess, &s0, &x10);
        let z11 = player1.broadcast(sess, &s1, &x11);
        let z21 = player1.broadcast(sess, &s1, &x21);
        let z22 = player2.broadcast(sess, &s2, &x22);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl MulOp {
    pub(crate) fn rep_rep_kernel<S: Session, RingT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<RingT>,
        y: RepTensor<RingT>,
    ) -> Result<RepTensor<RingT>>
    where
        RingT: Clone,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        ReplicatedPlacement: ZeroShareGen<S, ShapeT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTensor<RingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let RepTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let v0 = with_context!(player0, sess, { x00 * y00 + x00 * y10 + x10 * y00 });
        let v1 = with_context!(player1, sess, { x11 * y11 + x11 * y21 + x21 * y11 });
        let v2 = with_context!(player2, sess, { x22 * y22 + x22 * y02 + x02 * y22 });

        let s0 = player0.shape(sess, &v0);
        let s1 = player1.shape(sess, &v1);
        let s2 = player2.shape(sess, &v2);
        let zero_shape = RepShape {
            shapes: [s0, s1, s2],
        };

        let RepZeroShare {
            alphas: [a0, a1, a2],
        } = rep.gen_zero_share(sess, &zero_shape)?;

        let z0 = with_context!(player0, sess, { v0 + a0 });
        let z1 = with_context!(player1, sess, { v1 + a1 });
        let z2 = with_context!(player2, sess, { v2 + a2 });

        Ok(rep.place(
            sess,
            RepTensor {
                shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
            },
        ))
    }

    pub(crate) fn mir_rep_kernel<S: Session, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: Mir3Tensor<RingT>,
        y: RepTensor<RingT>,
    ) -> Result<RepTensor<RingT>>
    where
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let Mir3Tensor {
            values: [x0, x1, x2],
        } = &x;

        let z00 = with_context!(player0, sess, x0 * y00);
        let z10 = with_context!(player0, sess, x0 * y10);

        let z11 = with_context!(player1, sess, x1 * y11);
        let z21 = with_context!(player1, sess, x1 * y21);

        let z22 = with_context!(player2, sess, x2 * y22);
        let z02 = with_context!(player2, sess, x2 * y02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn rep_mir_kernel<S: Session, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<RingT>,
        y: Mir3Tensor<RingT>,
    ) -> Result<RepTensor<RingT>>
    where
        HostPlacement: PlacementMul<S, RingT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let Mir3Tensor {
            values: [y0, y1, y2],
        } = &y;

        let z00 = with_context!(player0, sess, x00 * y0);
        let z10 = with_context!(player0, sess, x10 * y0);

        let z11 = with_context!(player1, sess, x11 * y1);
        let z21 = with_context!(player1, sess, x21 * y1);

        let z22 = with_context!(player2, sess, x22 * y2);
        let z02 = with_context!(player2, sess, x02 * y2);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl DotOp {
    pub(crate) fn rep_rep_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<RingT>,
        y: RepTensor<RingT>,
    ) -> Result<RepTensor<RingT>>
    where
        RingT: Clone,
        HostPlacement: PlacementAdd<S, RingT, RingT, RingT>,
        HostPlacement: PlacementDot<S, RingT, RingT, RingT>,
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
        ReplicatedPlacement: ZeroShareGen<S, ShapeT, RingT>,
        ReplicatedPlacement: PlacementPlace<S, RepTensor<RingT>>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let RepTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let v0 = with_context!(player0, sess, {
            dot(x00, y00) + dot(x00, y10) + dot(x10, y00)
        });
        let v1 = with_context!(player1, sess, {
            dot(x11, y11) + dot(x11, y21) + dot(x21, y11)
        });
        let v2 = with_context!(player2, sess, {
            dot(x22, y22) + dot(x22, y02) + dot(x02, y22)
        });

        let s0 = player0.shape(sess, &v0);
        let s1 = player1.shape(sess, &v1);
        let s2 = player2.shape(sess, &v2);
        let zero_shape = RepShape {
            shapes: [s0, s1, s2],
        };

        let RepZeroShare {
            alphas: [a0, a1, a2],
        } = rep.gen_zero_share(sess, &zero_shape)?;

        let z0 = with_context!(player0, sess, { v0 + a0 });
        let z1 = with_context!(player1, sess, { v1 + a1 });
        let z2 = with_context!(player2, sess, { v2 + a2 });

        Ok(rep.place(
            sess,
            RepTensor {
                shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
            },
        ))
    }
}

impl AndOp {
    pub(crate) fn rep_kernel<S: Session, RepT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepT,
        y: RepT,
    ) -> Result<RepT>
    where
        ReplicatedPlacement: PlacementMul<S, RepT, RepT, RepT>,
    {
        // and = mul in Z2
        Ok(rep.mul(sess, &x, &y))
    }
}

impl XorOp {
    pub(crate) fn rep_kernel<S: Session, X1, X2, Y>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: X1,
        y: X2,
    ) -> Result<Y>
    where
        ReplicatedPlacement: PlacementAdd<S, X1, X2, Y>,
    {
        // add = xor in Z2
        Ok(rep.add(sess, &x, &y))
    }
}

impl NegOp {
    pub(crate) fn rep_bit_kernel<S: Session, HostBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<HostBitT>,
    ) -> Result<RepTensor<HostBitT>>
    where
        HostPlacement: PlacementNeg<S, HostBitT, HostBitT>,
    {
        let (player0, _player1, player2) = rep.host_placements();

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        // TODO(Morten)
        // we could choose share to change at random
        // to more fairly distribute compute load
        let y00 = player0.neg(sess, &x00);
        let y10 = x10;
        let y11 = x11;
        let y21 = x21;
        let y22 = x22;
        let y02 = player2.neg(sess, &x02);

        Ok(RepTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        })
    }

    pub(crate) fn rep_rep_kernel<S: Session, HostRepT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<HostRepT>,
    ) -> Result<RepTensor<HostRepT>>
    where
        HostPlacement: PlacementNeg<S, HostRepT, HostRepT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let y00 = player0.neg(sess, &x00);
        let y10 = player0.neg(sess, &x10);
        let y11 = player1.neg(sess, &x11);
        let y21 = player1.neg(sess, &x21);
        let y22 = player2.neg(sess, &x22);
        let y02 = player2.neg(sess, &x02);

        Ok(RepTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        })
    }
}

impl ShlOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        amount: usize,
        x: RepTensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementShl<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;
        let z00 = player0.shl(sess, amount, x00);
        let z10 = player0.shl(sess, amount, x10);

        let z11 = player1.shl(sess, amount, x11);
        let z21 = player1.shl(sess, amount, x21);

        let z22 = player2.shl(sess, amount, x22);
        let z02 = player2.shl(sess, amount, x02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl MsbOp {
    pub(crate) fn rep_bit_kernel<S: Session, RepRingT, RepBitT, RepBitArrayT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
    ) -> Result<RepBitT>
    where
        RepRingT: Ring<BitLength = N>,
        RepBitArrayT: BitArray<Len = N>,
        ReplicatedPlacement: PlacementBitDecompose<S, RepRingT, RepBitArrayT>,
        ReplicatedPlacement: PlacementIndex<S, RepBitArrayT, RepBitT>,
    {
        let bits = rep.bit_decompose(sess, &x);
        Ok(rep.index(sess, N::VALUE - 1, &bits))
    }

    pub(crate) fn rep_ring_kernel<S: Session, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
    ) -> Result<RepRingT>
    where
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
    {
        let x_bin = rep.msb(sess, &x);
        Ok(rep.ring_inject(sess, 0, &x_bin))
    }

    pub(crate) fn rep_bit_dec_kernel<S: Session, RepBitArrayT, RepRingT, RepBitT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        bits: RepBitArrayT,
    ) -> Result<RepRingT>
    where
        RepBitArrayT: BitArray<Len = N>,
        ReplicatedPlacement: PlacementIndex<S, RepBitArrayT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
    {
        let msb = rep.index(sess, N::VALUE - 1, &bits);
        Ok(rep.ring_inject(sess, 0, &msb))
    }
}

impl RingFixedpointAbsOp {
    pub(crate) fn rep_ring_kernel<S: Session, RepRingT, MirRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
    ) -> Result<RepRingT>
    where
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShl<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementSub<S, MirRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: ShapeFill<S, RepRingT, Result = MirRingT>,
    {
        let msb_ring = rep.msb(sess, &x);
        let double = rep.shl(sess, 1, &msb_ring);
        let ones = rep.shape_fill(sess, Constant::Ring64(1), &msb_ring);
        let sign = rep.sub(sess, &ones, &double);
        Ok(rep.mul(sess, &sign, &x))
    }
}

impl RingFixedpointReluOp {
    pub(crate) fn rep_ring_kernel<S: Session, RepRingT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
    ) -> Result<RepRingT>
    where
        ReplicatedPlacement: PlacementMsb<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepRingT>,
        ReplicatedPlacement: PlacementShape<S, RepRingT, ShapeT>,
        ReplicatedPlacement: PlacementMux<S, RepRingT, RepRingT, RepRingT, RepRingT>,
    {
        let sign_bit = rep.msb(sess, &x);
        let zeros = rep.fill(sess, 0_u8.into(), &rep.shape(sess, &x));

        Ok(rep.mux(sess, &sign_bit, &zeros, &x))
    }
}

impl SigmoidOp {
    pub(crate) fn rep_rep_kernel<S: Session, RepFixedT, ShapeT, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepFixedT,
    ) -> Result<RepFixedT>
    where
        RepRingT: Clone,
        RepFixedT: FixedpointTensor,
        RepFixedTensor<RepRingT>: Into<RepFixedT>,
        ReplicatedPlacement: PlacementShape<S, RepFixedT, ShapeT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementDiv<S, RepFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementExp<S, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementNeg<S, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementGreaterThan<S, RepFixedT, RepFixedT, RepBitT>,
        ReplicatedPlacement: PlacementMux<S, RepRingT, RepFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
    {
        // TODO [Yann]: revisit once we support mixed arithmetic for division
        let ones = 1.0_f64.as_fixedpoint(x.fractional_precision() as usize);

        let ones_fill = rep.fill(sess, ones.into(), &rep.shape(sess, &x));
        let zeros_fill = rep.fill(sess, 0_u8.into(), &rep.shape(sess, &x));

        let ones_rep = RepFixedTensor {
            tensor: ones_fill,
            integral_precision: x.integral_precision(),
            fractional_precision: x.fractional_precision(),
        }
        .into();
        let zeros_rep = RepFixedTensor {
            tensor: zeros_fill,
            integral_precision: x.integral_precision(),
            fractional_precision: x.fractional_precision(),
        }
        .into();

        let denominator = rep.add(sess, &ones_rep, &rep.exp(sess, &rep.neg(sess, &x)));
        let output = rep.div(sess, &ones_rep, &denominator);

        // input sanitization
        let max_val = 2_f64
            .powf((x.integral_precision() - 1).into())
            .ln()
            .as_fixedpoint(x.fractional_precision() as usize);
        let max_val_fill = rep.fill(sess, max_val.into(), &rep.shape(sess, &x));
        let max_val_rep = RepFixedTensor {
            tensor: max_val_fill,
            integral_precision: x.integral_precision(),
            fractional_precision: x.fractional_precision(),
        }
        .into();

        // compute upper bound
        let upper = rep.greater_than(sess, &x, &max_val_rep); // x > max_val?
        let upper_ring = rep.ring_inject(sess, 0, &upper);
        let upper_wall = rep.mux(sess, &upper_ring, &ones_rep, &output);

        // compute lower bound
        let lower = rep.greater_than(sess, &rep.neg(sess, &max_val_rep), &x); // -max_val > x?
        let lower_ring = rep.ring_inject(sess, 0, &lower);
        let res = rep.mux(sess, &lower_ring, &zeros_rep, &upper_wall);

        Ok(res)
    }
}