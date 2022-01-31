use super::*;

pub trait PlacementFill<S: Session, ShapeT, O> {
    fn fill(&self, sess: &S, value: Constant, shape: &ShapeT) -> O;
}

modelled!(PlacementFill::fill, ReplicatedPlacement, attributes[value: Constant] (ReplicatedShape) -> ReplicatedRing64Tensor, FillOp);
modelled!(PlacementFill::fill, ReplicatedPlacement, attributes[value: Constant] (ReplicatedShape) -> ReplicatedRing128Tensor, FillOp);
modelled!(PlacementFill::fill, ReplicatedPlacement, attributes[value: Constant] (ReplicatedShape) -> ReplicatedBitTensor, FillOp);
modelled!(PlacementFill::fill, Mirrored3Placement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3Ring64Tensor, FillOp);
modelled!(PlacementFill::fill, Mirrored3Placement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3Ring128Tensor, FillOp);
modelled!(PlacementFill::fill, Mirrored3Placement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3BitTensor, FillOp);
modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> HostBitTensor, FillOp);
modelled!(PlacementFill::fill, Mirrored3Placement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3Fixed64Tensor, FillOp);
modelled!(PlacementFill::fill, Mirrored3Placement, attributes[value: Constant] (ReplicatedShape) -> Mirrored3Fixed128Tensor, FillOp);

kernel! {
    FillOp,
    [
        (HostPlacement, (HostShape) -> HostBitTensor => [runtime] custom |op| {
            use std::convert::TryInto;
            let value: u8 = match op.value {
                Constant::Bit(v) => v,
                Constant::Ring64(v) => v.try_into().map_err(|_| {
                    Error::KernelError("Cannot fill HostBitTensor with non-binary value.".to_string())
                })?,
                Constant::Ring128(v) => v.try_into().map_err(|_| {
                    Error::KernelError("Cannot fill HostBitTensor with non-binary value.".to_string())
                })?,
                _ => {
                    return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a HostBitTensor", op.value.ty())))
                }
            };
            if !(value == 0 || value == 1) {
                return Err(Error::KernelError(
                    "Cannot fill HostBitTensor with non-binary value.".to_string(),
                ));
            }
            assert!(value == 0 || value == 1);
            Ok(Box::new(move |sess, host, host_shape| {
                Self::bit_kernel(sess, host, value, host_shape)
            }))
        }),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedRing64Tensor => [concrete] custom |op| {
                let value: u64 = match op.value {
                    Constant::Bit(v) => v as u64,
                    Constant::Ring64(v) => v,
                    Constant::Float64(v) => v as u64,
                    Constant::Fixed(FixedpointConstant {
                        value, precision
                    }) => {
                        ((value * ((1u64 << precision) as f64)) as i64) as u64
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a ReplicatedRing64Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::ring64_kernel(sess, rep, value, rep_shape)
                }))
            }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Ring64Tensor => [concrete] custom |op| {
                let value: u64 = match op.value {
                    Constant::Bit(v) => v as u64,
                    Constant::Ring64(v) => v,
                    Constant::Float64(v) => v as u64,
                    Constant::Fixed(FixedpointConstant {
                        value, precision
                    }) => {
                        ((value * ((1u64 << precision) as f64)) as i64) as u64
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Ring64Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_ring64_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedRing128Tensor => [concrete] custom |op| {
                let value: u128 = match op.value {
                    Constant::Bit(v) => v as u128,
                    Constant::Ring64(v) => v as u128,
                    Constant::Ring128(v) => v,
                    Constant::Float64(v) => v as u128,
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                        ((value * ((1u128 << precision) as f64)) as i128) as u128
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a ReplicatedRing128Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::ring128_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Ring128Tensor => [concrete] custom |op| {
                let value: u128 = match op.value {
                    Constant::Bit(v) => v as u128,
                    Constant::Ring64(v) => v as u128,
                    Constant::Ring128(v) => v,
                    Constant::Float64(v) => v as u128,
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                        ((value * ((1u128 << precision) as f64)) as i128) as u128
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Ring128Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_ring128_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (ReplicatedPlacement, (ReplicatedShape) -> ReplicatedBitTensor => [concrete] custom |op| {
                let value: u8 = match op.value {
                    Constant::Bit(v) => v,
                    Constant::Ring64(v) => v as u8,
                    Constant::Ring128(v) => v as u8,
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a ReplicatedBitTensor", op.value.ty()))),
                };
                if value != 0 && value != 1 {
                    return Err(Error::InvalidArgument(format!("Could only support 0 and 1 for the bit tensor fill, got {}", value)));
                }
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::rep_bit_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3BitTensor => [concrete] custom |op| {
                let value: u8 = match op.value {
                    Constant::Bit(v) => v,
                    Constant::Ring64(v) => v as u8,
                    Constant::Ring128(v) => v as u8,
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3BitTensor", op.value.ty()))),
                };
                if value != 0 && value != 1 {
                    return Err(Error::InvalidArgument(format!("Could only support 0 and 1 for the bit tensor fill, got {}", value)));
                }
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_bit_kernel(sess, rep, value, rep_shape)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Fixed64Tensor => [hybrid] custom |op| {
                let (ring_value, fractional_precision, integral_precision) = match op.value {
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                        let ring_value: u64 = ((value * ((1u64 << precision) as f64)) as i64) as u64;
                        let fractional_precision = precision as u32;
                        let integral_precision = value.log2().ceil() as u32;
                        (ring_value, fractional_precision, integral_precision)
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Fixed64Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_fixed_kernel(sess, rep, Constant::Ring64(ring_value), rep_shape, fractional_precision, integral_precision)
                }))
        }),
        (Mirrored3Placement, (ReplicatedShape) -> Mirrored3Fixed128Tensor => [hybrid] custom |op| {
                let (ring_value, fractional_precision, integral_precision) = match op.value {
                    Constant::Fixed(FixedpointConstant{value, precision}) => {
                        let ring_value: u128 = ((value * ((1u128 << precision) as f64)) as i128) as u128;
                        let fractional_precision = precision as u32;
                        let integral_precision = value.log2().ceil() as u32;
                        (ring_value, fractional_precision, integral_precision)
                    },
                    _ => return Err(Error::UnimplementedOperator(
                        format!("Cannot fill from {:?} into a Mirrored3Fixed128Tensor", op.value.ty()))),
                };
                Ok(Box::new(move |sess, rep, rep_shape| {
                    Self::mir_fixed_kernel(sess, rep, Constant::Ring128(ring_value), rep_shape, fractional_precision, integral_precision)
                }))
        }),
    ]
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

modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> HostRing64Tensor, RingFillOp);
modelled!(PlacementFill::fill, HostPlacement, attributes[value: Constant] (HostShape) -> HostRing128Tensor, RingFillOp);

kernel! {
    RingFillOp,
    [
        (HostPlacement, (HostShape) -> HostRing64Tensor => [runtime] attributes[value: Ring64] Self::ring64_kernel),
        (HostPlacement, (HostShape) -> HostRing128Tensor => [runtime] attributes[value: Ring128] Self::ring128_kernel),
    ]
}

pub trait PlacementZeros<S: Session, ShapeT, O> {
    fn zeros(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementZeros<S, ShapeT, O> for P
where
    P: PlacementFill<S, ShapeT, O>,
    O: TensorLike<S>,
    O::Scalar: Into<Constant>,
    O::Scalar: From<u8>,
{
    fn zeros(&self, sess: &S, shape: &ShapeT) -> O {
        let value = O::Scalar::from(0).into();
        self.fill(sess, value, shape)
    }
}

pub trait PlacementOnes<S: Session, ShapeT, O> {
    fn ones(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementOnes<S, ShapeT, O> for P
where
    P: PlacementFill<S, ShapeT, O>,
    O: TensorLike<S>,
    O::Scalar: Into<Constant>,
    O::Scalar: From<u8>,
{
    fn ones(&self, sess: &S, shape: &ShapeT) -> O {
        let value = O::Scalar::from(1).into();
        self.fill(sess, value, shape)
    }
}

modelled_kernel! {
    PlacementOnes::ones, FloatingpointOnesOp,
    [
        (HostPlacement, (HostShape) -> Float64Tensor => [hybrid] Self::float_host_kernel),
    ]
}

modelled_kernel! {
    PlacementOnes::ones, HostOnesOp,
    [
        (HostPlacement, (HostShape) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostInt64Tensor => [runtime] Self::kernel),
    ]
}

modelled_kernel! {
    PlacementOnes::ones, OnesOp,
    [
        (HostPlacement, (HostShape) -> Tensor => [hybrid] Self::host_kernel),
        // We do not support the ReplicatedPlacement: PlacementFill yet, hence we do not support Ones.
        // Also, logical Tensor can only hold Host tensors at the moment.
        // (ReplicatedPlacement, (HostShape) -> Tensor => [hybrid] Self::rep_kernel),
    ]
}

pub trait PlacementConstant<S: Session, O> {
    fn constant(&self, sess: &S, value: Constant) -> O;
}

macro_rules! constant_kernels {
    ($($val:ident),+) => {
        $(
            modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> $val, ConstantOp);
        )+
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> HostString, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> HostShape, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> PrfKey, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Seed, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Float32Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, HostPlacement, attributes[value: Constant] () -> Float64Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, Mirrored3Placement, attributes[value: Constant] () -> Float32Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, Mirrored3Placement, attributes[value: Constant] () -> Float64Tensor, ConstantOp);
        modelled!(PlacementConstant::constant, Mirrored3Placement, attributes[value: Constant] () -> Tensor, ConstantOp);


        kernel! {
            ConstantOp, [
                $(
                    (HostPlacement, () -> $val => [runtime] attributes[value: $val] Self::kernel),
                )+
                (HostPlacement, () -> HostString => [runtime] attributes[value: String] Self::string_kernel),
                (HostPlacement, () -> HostShape => [runtime] attributes[value: RawShape] Self::shape_kernel),
                (HostPlacement, () -> PrfKey => [runtime] attributes[value: RawPrfKey] Self::prf_key_kernel),
                (HostPlacement, () -> Seed => [runtime] attributes[value: RawSeed] Self::seed_kernel),
                (HostPlacement, () -> Tensor => [concrete] attributes[sig, value] Self::logical_kernel),
                (HostPlacement, () -> Float32Tensor => [concrete] attributes[value] Self::float_kernel),
                (HostPlacement, () -> Float64Tensor => [concrete] attributes[value] Self::float_kernel),
                (Mirrored3Placement, () -> Tensor => [concrete] attributes[sig, value] Self::mir3_logical_kernel),
                (Mirrored3Placement, () -> Float32Tensor => [concrete] attributes[value] Self::mir3_float_kernel),
                (Mirrored3Placement, () -> Float64Tensor => [concrete] attributes[value] Self::mir3_float_kernel),

            ]
        }
    };
}

constant_kernels![
    HostRing64Tensor,
    HostRing128Tensor,
    HostFloat32Tensor,
    HostFloat64Tensor,
    HostInt8Tensor,
    HostInt16Tensor,
    HostInt32Tensor,
    HostInt64Tensor,
    HostUint8Tensor,
    HostUint16Tensor,
    HostUint32Tensor,
    HostUint64Tensor
];
