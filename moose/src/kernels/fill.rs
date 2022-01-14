use crate::computation::*;
use crate::types::*;
use crate::replicated::ReplicatedPlacement;
use crate::mirrored::Mirrored3Placement;
use crate::Error;
use super::traits::*;

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
                        (value * ((1u64 << precision) as f64)) as u64
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
                        (value * ((1u64 << precision) as f64)) as u64
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
                            (value * ((1u128 << precision) as f64)) as u128
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
                            (value * ((1u128 << precision) as f64)) as u128
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
                        let ring_value: u64 = (value * ((1u64 << precision) as f64)) as u64;
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
                        let ring_value: u128 = (value * ((1u128 << precision) as f64)) as u128;
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
