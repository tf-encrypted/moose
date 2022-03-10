//! Moose recognized types

use crate::additive::{AdtShape, AdtTensor};
use crate::boolean::BoolTensor;
use crate::encrypted::{AbstractAesKey, AbstractAesTensor, FixedAesTensor};
use crate::fixedpoint::FixedTensor;
use crate::floatingpoint::FloatTensor;
use crate::host::{
    AbstractHostAesKey, HostBitArray, HostFixedAesTensor, HostFixedTensor, HostRingTensor,
    HostTensor,
};
use crate::integer::AbstractUint64Tensor;
use crate::logical::AbstractTensor;
use crate::mirrored::{Mir3Tensor, MirFixedTensor};
use crate::replicated::{
    RepAesKey, RepBitArray, RepFixedTensor, RepShape, RepTensor, RepUintTensor,
};
pub use crate::{
    host::{HostPrfKey, HostSeed},
    host::{HostShape, HostString},
};
use crate::{N128, N224, N256, N64};
// Logical types

pub type Tensor = AbstractTensor<
    Fixed64Tensor,
    Fixed128Tensor,
    Float32Tensor,
    Float64Tensor,
    BooleanTensor,
    Uint64Tensor,
>;

moose_type!(Fixed64Tensor = FixedTensor<HostFixed64Tensor, Mirrored3Fixed64Tensor, ReplicatedFixed64Tensor>);
moose_type!(HostFixed64Tensor = HostFixedTensor<HostRing64Tensor>);
moose_type!(Mirrored3Fixed64Tensor = MirFixedTensor<Mirrored3Ring64Tensor>);
moose_type!(ReplicatedFixed64Tensor = RepFixedTensor<ReplicatedRing64Tensor>);

moose_type!(Fixed128Tensor = FixedTensor<HostFixed128Tensor, Mirrored3Fixed128Tensor, ReplicatedFixed128Tensor>);
moose_type!(HostFixed128Tensor = HostFixedTensor<HostRing128Tensor>);
moose_type!(Mirrored3Fixed128Tensor = MirFixedTensor<Mirrored3Ring128Tensor>);
moose_type!(ReplicatedFixed128Tensor = RepFixedTensor<ReplicatedRing128Tensor>);

moose_type!(Float32Tensor = FloatTensor<HostFloat32Tensor, Mirrored3Float32>);
moose_type!(HostFloat32Tensor = [atomic] HostTensor<f32>);
moose_type!(Mirrored3Float32 = Mir3Tensor<HostFloat32Tensor>);

moose_type!(Float64Tensor = FloatTensor<HostFloat64Tensor, Mirrored3Float64>);
moose_type!(HostFloat64Tensor = [atomic] HostTensor<f64>);
moose_type!(Mirrored3Float64 = Mir3Tensor<HostFloat64Tensor>);

moose_type!(BooleanTensor = BoolTensor<HostBitTensor, ReplicatedBitTensor>);
pub use crate::host::HostBitTensor;
moose_type!(ReplicatedBitTensor = RepTensor<HostBitTensor>);

moose_type!(ReplicatedUint64Tensor = RepUintTensor<ReplicatedRing64Tensor>);
moose_type!(HostUint64Tensor = [atomic] HostTensor<u64>);
moose_type!(Uint64Tensor = AbstractUint64Tensor<HostUint64Tensor, ReplicatedUint64Tensor>);
// Encrypted types

moose_type!(AesTensor = AbstractAesTensor<Fixed128AesTensor>);
moose_type!(Fixed128AesTensor = FixedAesTensor<HostFixed128AesTensor>);
moose_type!(HostFixed128AesTensor = HostFixedAesTensor<HostBitArray224>);

moose_type!(AesKey = AbstractAesKey<HostAesKey, ReplicatedAesKey>);
moose_type!(HostAesKey = AbstractHostAesKey<HostBitArray128>);
moose_type!(ReplicatedAesKey = RepAesKey<ReplicatedBitArray128>);

// Ring types

moose_type!(ReplicatedRing64Tensor = RepTensor<HostRing64Tensor>);
moose_type!(AdditiveRing64Tensor = AdtTensor<HostRing64Tensor>);
moose_type!(Mirrored3Ring64Tensor = Mir3Tensor<HostRing64Tensor>);
moose_type!(HostRing64Tensor = [atomic] HostRingTensor<u64>);

moose_type!(ReplicatedRing128Tensor = RepTensor<HostRing128Tensor>);
moose_type!(AdditiveRing128Tensor = AdtTensor<HostRing128Tensor>);
moose_type!(Mirrored3Ring128Tensor = Mir3Tensor<HostRing128Tensor>);
moose_type!(HostRing128Tensor = [atomic] HostRingTensor<u128>);

// Misc mirrored types

moose_type!(Mirrored3BitTensor = Mir3Tensor<HostBitTensor>);

// Mist additive types

moose_type!(AdditiveBitTensor = AdtTensor<HostBitTensor>);
moose_type!(AdditiveShape = AdtShape<HostShape>);

// Misc replicated types

moose_type!(ReplicatedShape = RepShape<HostShape>);
pub type ReplicatedBitArray64 = RepBitArray<ReplicatedBitTensor, N64>;
pub type ReplicatedBitArray128 = RepBitArray<ReplicatedBitTensor, N128>;
pub type ReplicatedBitArray224 = RepBitArray<ReplicatedBitTensor, N224>;

// Misc host types

moose_type!(HostInt8Tensor = [atomic] HostTensor<i8>);
moose_type!(HostInt16Tensor = [atomic] HostTensor<i16>);
moose_type!(HostInt32Tensor = [atomic] HostTensor<i32>);
moose_type!(HostInt64Tensor = [atomic] HostTensor<i64>);
moose_type!(HostUint8Tensor = [atomic] HostTensor<u8>);
moose_type!(HostUint16Tensor = [atomic] HostTensor<u16>);
moose_type!(HostUint32Tensor = [atomic] HostTensor<u32>);
pub type HostBitArray64 = HostBitArray<HostBitTensor, N64>;
pub type HostBitArray128 = HostBitArray<HostBitTensor, N128>;
pub type HostBitArray224 = HostBitArray<HostBitTensor, N224>;
pub type HostBitArray256 = HostBitArray<HostBitTensor, N256>;

moose_type!(HostPrfKey);
moose_type!(HostSeed);
moose_type!(HostBitTensor);
moose_type!(HostString);
moose_type!(HostShape);
