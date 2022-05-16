//! Support for various operators that do not fit elsewhere

use super::*;
use crate::host::SliceInfo;
use crate::mirrored::{Mir3Tensor, Mirrored3Placement};

impl IdentityOp {
    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementIdentity<S, RepRingT, RepRingT>,
    {
        let tensor = rep.identity(sess, &x.tensor);
        Ok(RepFixedTensor {
            tensor,
            integral_precision: x.integral_precision,
            fractional_precision: x.fractional_precision,
        })
    }

    pub(crate) fn rep_inner_kernel<S: Session, HostT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<HostT>,
    ) -> Result<RepTensor<HostT>>
    where
        HostPlacement: PlacementIdentity<S, HostT, HostT>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;
        let y00 = player0.identity(sess, x00);
        let y10 = player0.identity(sess, x10);
        let y11 = player1.identity(sess, x11);
        let y21 = player1.identity(sess, x21);
        let y22 = player2.identity(sess, x22);
        let y02 = player2.identity(sess, x02);
        Ok(RepTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        })
    }
}

impl ConcatOp {
    pub(crate) fn rep_rep_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: u32,
        xs: &[RepTensor<HostRingT>],
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementConcatenate<S, HostRingT, HostRingT>,
        HostRingT: Clone,
    {
        let mut z00s: Vec<HostRingT> = Vec::new();
        let mut z10s: Vec<HostRingT> = Vec::new();
        let mut z11s: Vec<HostRingT> = Vec::new();
        let mut z21s: Vec<HostRingT> = Vec::new();
        let mut z22s: Vec<HostRingT> = Vec::new();
        let mut z02s: Vec<HostRingT> = Vec::new();

        let (player0, player1, player2) = plc.host_placements();
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
        let z00 = player0.concatenate(sess, axis, &z00s);
        let z10 = player0.concatenate(sess, axis, &z10s);
        let z11 = player1.concatenate(sess, axis, &z11s);
        let z21 = player1.concatenate(sess, axis, &z21s);
        let z22 = player2.concatenate(sess, axis, &z22s);
        let z02 = player2.concatenate(sess, axis, &z02s);
        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl FillOp {
    pub(crate) fn rep_ring64_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        value: u64,
        rep_shape: RepShape<ShapeT>,
    ) -> Result<RepTensor<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return PublicReplicatedTensor, but we don't have that type yet
        let (player0, player1, player2) = rep.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &rep_shape;

        let shares = [
            [
                player0.fill(sess, Constant::Ring64(value), s0),
                player0.fill(sess, Constant::Ring64(0_u64), s0),
            ],
            [
                player1.fill(sess, Constant::Ring64(0_u64), s1),
                player1.fill(sess, Constant::Ring64(0_u64), s1),
            ],
            [
                player2.fill(sess, Constant::Ring64(0_u64), s2),
                player2.fill(sess, Constant::Ring64(value), s2),
            ],
        ];

        Ok(RepTensor { shares })
    }

    pub(crate) fn rep_ring128_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        value: u128,
        rep_shape: RepShape<ShapeT>,
    ) -> Result<RepTensor<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return PublicReplicatedTensor, but we don't have that type yet
        let (player0, player1, player2) = rep.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &rep_shape;

        let shares = [
            [
                player0.fill(sess, Constant::Ring128(value), s0),
                player0.fill(sess, Constant::Ring128(0_u128), s0),
            ],
            [
                player1.fill(sess, Constant::Ring128(0_u128), s1),
                player1.fill(sess, Constant::Ring128(0_u128), s1),
            ],
            [
                player2.fill(sess, Constant::Ring128(0_u128), s2),
                player2.fill(sess, Constant::Ring128(value), s2),
            ],
        ];

        Ok(RepTensor { shares })
    }

    pub(crate) fn mir_ring64_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        mir: &Mirrored3Placement,
        value: u64,
        rep_shape: RepShape<ShapeT>,
    ) -> Result<Mir3Tensor<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        let (player0, player1, player2) = mir.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &rep_shape;

        let values = [
            player0.fill(sess, Constant::Ring64(value), s0),
            player1.fill(sess, Constant::Ring64(value), s1),
            player2.fill(sess, Constant::Ring64(value), s2),
        ];

        Ok(Mir3Tensor { values })
    }

    pub(crate) fn mir_ring128_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        mir: &Mirrored3Placement,
        value: u128,
        rep_shape: RepShape<ShapeT>,
    ) -> Result<Mir3Tensor<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        let (player0, player1, player2) = mir.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &rep_shape;

        let values = [
            player0.fill(sess, Constant::Ring128(value), s0),
            player1.fill(sess, Constant::Ring128(value), s1),
            player2.fill(sess, Constant::Ring128(value), s2),
        ];

        Ok(Mir3Tensor { values })
    }

    pub(crate) fn rep_bit_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        value: u8,
        rep_shape: RepShape<ShapeT>,
    ) -> Result<RepTensor<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        // TODO should really return PublicReplicatedTensor, but we don't have that type yet
        let (player0, player1, player2) = rep.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &rep_shape;

        let shares = [
            [
                player0.fill(sess, Constant::Bit(value), s0),
                player0.fill(sess, Constant::Bit(0_u8), s0),
            ],
            [
                player1.fill(sess, Constant::Bit(0_u8), s1),
                player1.fill(sess, Constant::Bit(0_u8), s1),
            ],
            [
                player2.fill(sess, Constant::Bit(0_u8), s2),
                player2.fill(sess, Constant::Bit(value), s2),
            ],
        ];

        Ok(RepTensor { shares })
    }

    pub(crate) fn mir_bit_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        mir: &Mirrored3Placement,
        value: u8,
        shape: RepShape<ShapeT>,
    ) -> Result<Mir3Tensor<RingT>>
    where
        HostPlacement: PlacementFill<S, ShapeT, RingT>,
    {
        let (player0, player1, player2) = mir.host_placements();

        let RepShape {
            shapes: [s0, s1, s2],
        } = &shape;

        let values = [
            player0.fill(sess, Constant::Bit(value), s0),
            player1.fill(sess, Constant::Bit(value), s1),
            player2.fill(sess, Constant::Bit(value), s2),
        ];

        Ok(Mir3Tensor { values })
    }
}

impl ExpandDimsOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Vec<usize>,
        x: RepTensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementExpandDims<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.expand_dims(sess, axis.clone(), x00);
        let z10 = player0.expand_dims(sess, axis.clone(), x10);

        let z11 = player1.expand_dims(sess, axis.clone(), x11);
        let z21 = player1.expand_dims(sess, axis.clone(), x21);

        let z22 = player2.expand_dims(sess, axis.clone(), x22);
        let z02 = player2.expand_dims(sess, axis, x02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl ReshapeOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT, HostShapeT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepTensor<HostRingT>,
        shape: RepShape<HostShapeT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementReshape<S, HostRingT, HostShapeT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let RepShape {
            shapes: [s0, s1, s2],
        } = &shape;

        let z00 = player0.reshape(sess, x00, s0);
        let z10 = player0.reshape(sess, x10, s0);

        let z11 = player1.reshape(sess, x11, s1);
        let z21 = player1.reshape(sess, x21, s1);

        let z22 = player2.reshape(sess, x22, s2);
        let z02 = player2.reshape(sess, x02, s2);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl IndexAxisOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        index: usize,
        x: RepTensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementIndexAxis<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.index_axis(sess, axis, index, x00);
        let z10 = player0.index_axis(sess, axis, index, x10);

        let z11 = player1.index_axis(sess, axis, index, x11);
        let z21 = player1.index_axis(sess, axis, index, x21);

        let z22 = player2.index_axis(sess, axis, index, x22);
        let z02 = player2.index_axis(sess, axis, index, x02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl IndexOp {
    pub(crate) fn rep_kernel<S: Session, RepBitT, N>(
        sess: &S,
        plc: &ReplicatedPlacement,
        index: usize,
        x: RepBitArray<RepBitT, N>,
    ) -> Result<RepBitT>
    where
        ReplicatedPlacement: PlacementIndexAxis<S, RepBitT, RepBitT>,
    {
        // TODO until we have HostBitArrays we simply delegate to IndexAxis operations
        let stacked_tensor = x.0;
        Ok(plc.index_axis(sess, 0, index, &stacked_tensor))
    }
}

impl DiagOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: RepTensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementDiag<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.diag(sess, x00);
        let z10 = player0.diag(sess, x10);

        let z11 = player1.diag(sess, x11);
        let z21 = player1.diag(sess, x21);

        let z22 = player2.diag(sess, x22);
        let z02 = player2.diag(sess, x02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl SliceOp {
    pub(crate) fn rep_shape_kernel<S: Session, ShapeT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        slice: SliceInfo,
        shape: RepShape<ShapeT>,
    ) -> Result<RepShape<ShapeT>>
    where
        HostPlacement: PlacementSlice<S, ShapeT, ShapeT>,
    {
        let (player0, player1, player2) = plc.host_placements();

        let RepShape {
            shapes: [shape0, shape1, shape2],
        } = shape;

        let new_shape0 = player0.slice(sess, slice.clone(), &shape0);
        let new_shape1 = player1.slice(sess, slice.clone(), &shape1);
        let new_shape2 = player2.slice(sess, slice, &shape2);

        Ok(RepShape {
            shapes: [new_shape0, new_shape1, new_shape2],
        })
    }

    pub(crate) fn rep_ring_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        info: SliceInfo,
        x: RepTensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementSlice<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.slice(sess, info.clone(), x00);
        let z10 = player0.slice(sess, info.clone(), x10);

        let z11 = player1.slice(sess, info.clone(), x11);
        let z21 = player1.slice(sess, info.clone(), x21);

        let z22 = player2.slice(sess, info.clone(), x22);
        let z02 = player2.slice(sess, info, x02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn rep_uint_kernel<S: Session, RepRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        info: SliceInfo,
        x: RepUintTensor<RepRingT>,
    ) -> Result<RepUintTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementSlice<S, RepRingT, RepRingT>,
    {
        Ok(RepUintTensor {
            tensor: rep.slice(sess, info, &x.tensor),
        })
    }
}

impl ShlDimOp {
    pub(crate) fn rep_bit_kernel<S: Session, HostBitT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        amount: usize,
        bit_length: usize,
        x: RepTensor<HostBitT>,
    ) -> Result<RepTensor<HostBitT>>
    where
        HostPlacement: PlacementShlDim<S, HostBitT, HostBitT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let z00 = player0.shl_dim(sess, amount, bit_length, &x00);
        let z10 = player0.shl_dim(sess, amount, bit_length, &x10);

        let z11 = player1.shl_dim(sess, amount, bit_length, &x11);
        let z21 = player1.shl_dim(sess, amount, bit_length, &x21);

        let z22 = player2.shl_dim(sess, amount, bit_length, &x22);
        let z02 = player2.shl_dim(sess, amount, bit_length, &x02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }
}

impl ShapeOp {
    pub(crate) fn rep_kernel<S: Session, RingT, ShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepTensor<RingT>,
    ) -> Result<RepShape<ShapeT>>
    where
        HostPlacement: PlacementShape<S, RingT, ShapeT>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let RepTensor {
            shares: [[x00, _x10], [x11, _x21], [x22, _x02]],
        } = &x;
        Ok(RepShape {
            shapes: [
                player0.shape(sess, x00),
                player1.shape(sess, x11),
                player2.shape(sess, x22),
            ],
        })
    }

    pub(crate) fn rep_repfixed_kernel<S: Session, RepRingT, RepShapeT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<RepShapeT>
    where
        ReplicatedPlacement: PlacementShape<S, RepRingT, RepShapeT>,
    {
        Ok(rep.shape(sess, &x.tensor))
    }
}

impl BroadcastOp {
    pub(crate) fn rep_ring_kernel<S: Session, ShapeT, RingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        shape: RepShape<ShapeT>,
        x: RepTensor<RingT>,
    ) -> Result<RepTensor<RingT>>
    where
        HostPlacement: PlacementBroadcast<S, ShapeT, RingT, RingT>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let RepShape {
            shapes: [s0, s1, s2],
        } = &shape;

        Ok(RepTensor {
            shares: [
                [
                    player0.broadcast(sess, s0, x00),
                    player0.broadcast(sess, s0, x10),
                ],
                [
                    player1.broadcast(sess, s1, x11),
                    player1.broadcast(sess, s1, x21),
                ],
                [
                    player2.broadcast(sess, s2, x22),
                    player2.broadcast(sess, s2, x02),
                ],
            ],
        })
    }
}

impl SqueezeOp {
    pub(crate) fn rep_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<usize>,
        x: RepTensor<HostRingT>,
    ) -> Result<RepTensor<HostRingT>>
    where
        HostPlacement: PlacementSqueeze<S, HostRingT, HostRingT>,
    {
        let (player0, player1, player2) = plc.host_placements();
        let RepTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = player0.squeeze(sess, axis, x00);
        let z10 = player0.squeeze(sess, axis, x10);

        let z11 = player1.squeeze(sess, axis, x11);
        let z21 = player1.squeeze(sess, axis, x21);

        let z22 = player2.squeeze(sess, axis, x22);
        let z02 = player2.squeeze(sess, axis, x02);

        Ok(RepTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn rep_uint_kernel<S: Session, RepRingT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: Option<usize>,
        x: RepUintTensor<RepRingT>,
    ) -> Result<RepUintTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementSqueeze<S, RepRingT, RepRingT>,
    {
        Ok(RepUintTensor {
            tensor: rep.squeeze(sess, axis, &x.tensor),
        })
    }
}
