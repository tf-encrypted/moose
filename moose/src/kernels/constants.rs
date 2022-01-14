use crate::computation::*;
use crate::types::*;
use crate::replicated::ReplicatedPlacement;
use crate::mirrored::Mirrored3Placement;
use crate::Error;
use super::traits::*;
use super::*;

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
