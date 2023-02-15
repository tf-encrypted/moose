use super::*;
use crate::host::{RawPrfKey, RawSeed, RawShape};
use crate::TensorLike;

pub trait PlacementFill<S: Session, ShapeT, O> {
    fn fill(&self, sess: &S, value: Constant, shape: &ShapeT) -> O;
}

modelled_kernel! {
    PlacementFill::fill, FillOp{value: Constant},
    [
        (HostPlacement, (HostShape) -> HostRing64Tensor => [runtime] custom |op| {
            let value = match op.value {
                Constant::Ring64(v) => v,
                _ => return Err(Error::KernelError(
                    "Cannot fill HostRing64Tensor with non-Ring64 value.".to_string(),
                ))
            };
            Ok(Box::new(move |sess, host, shape| {
                Self::host_ring64_kernel(sess, host, value, shape)
            }))
        }),
        (HostPlacement, (HostShape) -> HostRing128Tensor => [runtime] custom |op| {
            let value = match op.value {
                Constant::Ring128(v) => v,
                _ => return Err(Error::KernelError(
                    "Cannot fill HostRing64Tensor with non-Ring128 value.".to_string(),
                ))
            };
            Ok(Box::new(move |sess, host, shape| {
                Self::host_ring128_kernel(sess, host, value, shape)
            }))
        }),
        (AdditivePlacement, (HostShape) -> AdditiveRing64Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (HostShape) -> AdditiveRing128Tensor => [hybrid] Self::adt_host_kernel),
        (AdditivePlacement, (AdditiveShape) -> AdditiveRing64Tensor => [concrete] Self::adt_adt_kernel),
        (AdditivePlacement, (AdditiveShape) -> AdditiveRing128Tensor => [concrete] Self::adt_adt_kernel),
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
                Self::host_bit_kernel(sess, host, value, host_shape)
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
                    Self::rep_ring64_kernel(sess, rep, value, rep_shape)
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
                    Self::rep_ring128_kernel(sess, rep, value, rep_shape)
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
                    return Err(Error::InvalidArgument(format!("Could only support 0 and 1 for the bit tensor fill, got {value}")));
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
                    return Err(Error::InvalidArgument(format!("Could only support 0 and 1 for the bit tensor fill, got {value}")));
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

pub trait PlacementZeros<S: Session, ShapeT, O> {
    fn zeros(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementZeros<S, ShapeT, O> for P
where
    P: PlacementFill<S, ShapeT, O>,
    O: TensorLike,
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
    O: TensorLike,
    O::Scalar: Into<Constant>,
    O::Scalar: From<u8>,
{
    fn ones(&self, sess: &S, shape: &ShapeT) -> O {
        let value = O::Scalar::from(1).into();
        self.fill(sess, value, shape)
    }
}

modelled_kernel! {
    PlacementOnes::ones, OnesOp,
    [
        (HostPlacement, (Shape) -> Tensor => [concrete] custom |op| {
            use crate::logical::{AbstractTensor, TensorDType};
            match op.sig.ret() {
                Ty::Tensor(TensorDType::Float32) => Ok(Box::new(move |sess, plc, shape| {
                    Self::logical_host_kernel::<_, Float32Tensor, _, _>(sess, plc, shape).map(AbstractTensor::Float32)
                })),
                Ty::Tensor(TensorDType::Float64) => Ok(Box::new(move |sess, plc, shape| {
                    Self::logical_host_kernel::<_, Float64Tensor, _, _>(sess, plc, shape).map(AbstractTensor::Float64)
                })),
                other => {
                    Err(Error::UnimplementedOperator(
                        format!("Cannot build ones of type {other:?}")))
                },
            }
        }),
        // We do not support the ReplicatedPlacement: PlacementFill yet, hence we do not support Ones.
        // Also, logical Tensor can only hold Host tensors at the moment.
        // (ReplicatedPlacement, (HostShape) -> Tensor => [hybrid] Self::logical_rep_kernel),
        (HostPlacement, (HostShape) -> Float32Tensor => [hybrid] Self::host_float_kernel),
        (HostPlacement, (HostShape) -> Float64Tensor => [hybrid] Self::host_float_kernel),
        (HostPlacement, (HostShape) -> HostFloat32Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostShape) -> HostFloat64Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostShape) -> HostInt8Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostShape) -> HostInt16Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostShape) -> HostInt32Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostShape) -> HostInt64Tensor => [runtime] Self::host_kernel),
    ]
}

modelled_kernel! {
    PlacementZeros::zeros, ZerosOp,
    [
        (HostPlacement, (Shape) -> Tensor => [concrete] custom |op| {
            use crate::logical::{AbstractTensor, TensorDType};
            match op.sig.ret() {
                Ty::Tensor(TensorDType::Float32) => Ok(Box::new(move |sess, plc, shape| {
                    Self::logical_host_kernel::<_, Float32Tensor, _, _>(sess, plc, shape).map(AbstractTensor::Float32)
                })),
                Ty::Tensor(TensorDType::Float64) => Ok(Box::new(move |sess, plc, shape| {
                    Self::logical_host_kernel::<_, Float64Tensor, _, _>(sess, plc, shape).map(AbstractTensor::Float64)
                })),
                other => {
                    Err(Error::UnimplementedOperator(
                        format!("Cannot build zeros of type {other:?}")))
                },
            }
        }),
        // We do not support the ReplicatedPlacement: PlacementFill yet, hence we do not support Zeros.
        // Also, logical Tensor can only hold Host tensors at the moment.
        // (ReplicatedPlacement, (HostShape) -> Tensor => [hybrid] Self::logical_rep_kernel),
        (HostPlacement, (HostShape) -> Float32Tensor => [hybrid] Self::host_float_kernel),
        (HostPlacement, (HostShape) -> Float64Tensor => [hybrid] Self::host_float_kernel),
        (HostPlacement, (HostShape) -> HostFloat32Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostShape) -> HostFloat64Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostShape) -> HostInt8Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostShape) -> HostInt16Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostShape) -> HostInt32Tensor => [runtime] Self::host_kernel),
        (HostPlacement, (HostShape) -> HostInt64Tensor => [runtime] Self::host_kernel),
    ]
}

pub trait PlacementConstant<S: Session, O> {
    fn constant(&self, sess: &S, value: Constant) -> O;
}

// TODO(Morten) Since other work is updating the kernel macros I did not want to change
// any of these right now to match the need for individual `attributes[value: T]`.
// If we want we can get rid of `unwrapper` in the near future by updating the macros.

macro_rules! unwrapper {
    ($op:ident, $t:ident, $k:path) => {{
        let value: $t = match &$op.value {
            Constant::$t(v) => Ok(v.clone()),
            _ => Err(crate::error::Error::UnimplementedOperator("".to_string())),
        }?;
        Ok(Box::new(move |sess, plc| $k(sess, plc, value.clone())))
    }};
}

modelled_kernel! {
    PlacementConstant::constant, ConstantOp{value: Constant},
    [
        (HostPlacement, () -> HostRing64Tensor => [runtime] custom |op| unwrapper!(op, HostRing64Tensor, Self::kernel)),
        (HostPlacement, () -> HostRing128Tensor => [runtime] custom |op| unwrapper!(op, HostRing128Tensor, Self::kernel)),
        (HostPlacement, () -> HostFloat32Tensor => [runtime] custom |op| unwrapper!(op, HostFloat32Tensor, Self::kernel)),
        (HostPlacement, () -> HostFloat64Tensor => [runtime] custom |op| unwrapper!(op, HostFloat64Tensor, Self::kernel)),
        (HostPlacement, () -> HostInt8Tensor => [runtime] custom |op| unwrapper!(op, HostInt8Tensor, Self::kernel)),
        (HostPlacement, () -> HostInt16Tensor => [runtime] custom |op| unwrapper!(op, HostInt16Tensor, Self::kernel)),
        (HostPlacement, () -> HostInt32Tensor => [runtime] custom |op| unwrapper!(op, HostInt32Tensor, Self::kernel)),
        (HostPlacement, () -> HostInt64Tensor => [runtime] custom |op| unwrapper!(op, HostInt64Tensor, Self::kernel)),
        (HostPlacement, () -> HostUint8Tensor => [runtime] custom |op| unwrapper!(op, HostUint8Tensor, Self::kernel)),
        (HostPlacement, () -> HostUint16Tensor => [runtime] custom |op| unwrapper!(op, HostUint16Tensor, Self::kernel)),
        (HostPlacement, () -> HostUint32Tensor => [runtime] custom |op| unwrapper!(op, HostUint32Tensor, Self::kernel)),
        (HostPlacement, () -> HostUint64Tensor => [runtime] custom |op| unwrapper!(op, HostUint64Tensor, Self::kernel)),
        (HostPlacement, () -> HostBitTensor => [runtime] custom |op| unwrapper!(op, HostBitTensor, Self::kernel)),
        (HostPlacement, () -> HostString => [runtime] custom |op| unwrapper!(op, String, Self::string_kernel)),
        (HostPlacement, () -> HostPrfKey => [runtime] custom |op| unwrapper!(op, RawPrfKey, Self::prf_key_kernel)),
        (HostPlacement, () -> HostSeed => [runtime] custom |op| unwrapper!(op, RawSeed, Self::seed_kernel)),
        (HostPlacement, () -> Tensor => [concrete] custom |op| {
            let sig = op.sig;
            let value = op.value.clone();
            Ok(Box::new(move |sess, plc| {
                Self::logical_kernel(sess, plc, sig, value.clone())
            }))
        }),
        (HostPlacement, () -> Float32Tensor => [concrete] Self::float_kernel),
        (HostPlacement, () -> Float64Tensor => [concrete] Self::float_kernel),
        (HostPlacement, () -> Uint64Tensor => [concrete] Self::u64_kernel),
        (HostPlacement, () -> BooleanTensor => [concrete] Self::bool_kernel),
        (Mirrored3Placement, () -> Tensor => [concrete] custom |op| {
            let sig = op.sig;
            let value = op.value.clone();
            Ok(Box::new(move |sess, plc| {
                Self::mir3_logical_kernel(sess, plc, sig, value.clone())
            }))
        }),
        (Mirrored3Placement, () -> Float32Tensor => [concrete] Self::mir3_float_kernel),
        (Mirrored3Placement, () -> Float64Tensor => [concrete] Self::mir3_float_kernel),
        (HostPlacement, () -> Shape => [concrete] custom |op| {
            let sig = op.sig;
            let value = op.value.clone();
            Ok(Box::new(move |sess, plc| {
                Self::shape_logical_kernel(sess, plc, sig, value.clone())
            }))
        }),
        (HostPlacement, () -> HostShape => [runtime] custom |op| unwrapper!(op, RawShape, Self::shape_kernel)),
    ]
}
