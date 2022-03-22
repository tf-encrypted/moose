use super::*;

pub trait PlacementSend<S: Session, T, O> {
    fn send(&self, sess: &S, rendezvous_key: RendezvousKey, receiver: Role, x: &T) -> O;
}

pub trait PlacementReceive<S: Session, O> {
    fn receive(&self, sess: &S, rendezvous_key: RendezvousKey, sender: Role) -> O;
}

pub trait PlacementInput<S: Session, O> {
    fn input(&self, sess: &S, arg_name: String) -> O;
}

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> $value, InputOp);
    )*
)}

modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Float32Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Float64Tensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostBitArray64, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostBitArray128, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostBitArray224, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostAesKey, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitTensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedRing64Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedRing128Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedFixed64Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedFixed128Tensor, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitArray64, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitArray128, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedBitArray224, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> ReplicatedAesKey, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> AesKey, InputOp);
modelled!(PlacementInput::input, ReplicatedPlacement, attributes[arg_name: String] () -> AesKey, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> AesTensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> Fixed128AesTensor, InputOp);
modelled!(PlacementInput::input, HostPlacement, attributes[arg_name: String] () -> HostFixed128AesTensor, InputOp);

kernel! {
    InputOp, [
        (HostPlacement, () -> HostString => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUnit => [runtime] attributes[arg_name] Self::missing_kernel),
        (HostPlacement, () -> HostShape => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostSeed => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostPrfKey => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostBitArray64 => [concrete] attributes[arg_name] Self::host_bitarray64),
        (HostPlacement, () -> HostBitArray128 => [concrete] attributes[arg_name] Self::host_bitarray128),
        (HostPlacement, () -> HostBitArray224 => [concrete] attributes[arg_name] Self::host_bitarray224),
        (HostPlacement, () -> HostBitTensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostRing64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostRing128Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostFloat32Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostFloat64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt8Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt16Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt32Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostInt64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint8Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint16Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint32Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostUint64Tensor => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> HostFixed64Tensor => [runtime] attributes[arg_name] Self::missing_kernel),
        (HostPlacement, () -> HostFixed128Tensor => [runtime] attributes[arg_name] Self::missing_kernel),
        (HostPlacement, () -> Tensor => [concrete] attributes[sig, arg_name] Self::logical_kernel),
        (HostPlacement, () -> Float32Tensor => [concrete] attributes[arg_name] Self::float_kernel),
        (HostPlacement, () -> Float64Tensor => [concrete] attributes[arg_name] Self::float_kernel),
        (HostPlacement, () -> AesKey => [concrete] attributes[arg_name] Self::aes_kernel_on_host),
        (HostPlacement, () -> HostAesKey => [concrete] attributes[arg_name] Self::host_aes_kernel),
        (HostPlacement, () -> AesTensor => [concrete] attributes[arg_name] Self::aestensor),
        (HostPlacement, () -> Fixed128AesTensor => [concrete] attributes[arg_name] Self::fixed_aestensor),
        (HostPlacement, () -> HostFixed128AesTensor => [concrete] attributes[sig, arg_name] Self::host_fixed_aestensor),
        (ReplicatedPlacement, () -> ReplicatedBitTensor => [concrete] attributes[arg_name] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedRing64Tensor => [concrete] attributes[arg_name] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedRing128Tensor => [concrete] attributes[arg_name] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedFixed64Tensor => [concrete] attributes[sig, arg_name] Self::replicated_fixed_kernel),
        (ReplicatedPlacement, () -> ReplicatedFixed128Tensor => [concrete] attributes[sig, arg_name] Self::replicated_fixed_kernel),
        (ReplicatedPlacement, () -> ReplicatedBitArray64 => [concrete] attributes[arg_name] Self::replicated_bitarray64),
        (ReplicatedPlacement, () -> ReplicatedBitArray128 => [concrete] attributes[arg_name] Self::replicated_bitarray128),
        (ReplicatedPlacement, () -> ReplicatedBitArray224 => [concrete] attributes[arg_name] Self::replicated_bitarray224),
        (ReplicatedPlacement, () -> AesKey => [concrete] attributes[arg_name] Self::aes_kernel_on_replicated),
        (ReplicatedPlacement, () -> ReplicatedAesKey => [concrete] attributes[arg_name] Self::replicated_aes_kernel),
    ]
}

pub trait PlacementOutput<S: Session, T, O> {
    fn output(&self, sess: &S, x: &T) -> O;
}

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementOutput::output, HostPlacement, ($value) -> $value, OutputOp);
    )*
)}

modelled!(PlacementOutput::output, HostPlacement, (Tensor) -> Tensor, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (Float32Tensor) -> Float32Tensor, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (Float64Tensor) -> Float64Tensor, OutputOp);
modelled!(PlacementOutput::output, HostPlacement, (BooleanTensor) -> BooleanTensor, OutputOp);

kernel! {
    OutputOp, [
        (HostPlacement, (HostUnit) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (HostSeed) -> HostSeed => [runtime] Self::kernel),
        (HostPlacement, (HostPrfKey) -> HostPrfKey => [runtime] Self::kernel),
        (HostPlacement, (HostString) -> HostString => [runtime] Self::kernel),
        (HostPlacement, (HostBitTensor) -> HostBitTensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing64Tensor) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostRing128Tensor) -> HostRing128Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat32Tensor) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFloat64Tensor) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt8Tensor) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt16Tensor) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt32Tensor) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostInt64Tensor) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint8Tensor) -> HostUint8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint16Tensor) -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint32Tensor) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostUint64Tensor) -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostFixed64Tensor) -> HostFixed64Tensor => [runtime] Self::non_placing_kernel),
        (HostPlacement, (HostFixed128Tensor) -> HostFixed128Tensor => [runtime] Self::non_placing_kernel),
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::logical_kernel),
        (HostPlacement, (BooleanTensor) -> BooleanTensor => [hybrid] Self::bool_kernel),
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_kernel),
    ]
}

pub trait PlacementLoad<S: Session, KeyT, QueryT, O> {
    fn load(&self, sess: &S, key: &KeyT, query: &QueryT) -> O;
}

modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> HostFloat32Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> HostFloat64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> Float32Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> Float64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> Tensor, LoadOp);

kernel! {
    LoadOp, [
        (HostPlacement, (HostString, HostString) -> HostUnit => [runtime] Self::missing_kernel),
        (HostPlacement, (HostString, HostString) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostSeed => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostPrfKey => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostString => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostBitTensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostRing128Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint8Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostFixed64Tensor => [runtime] Self::missing_kernel),
        (HostPlacement, (HostString, HostString) -> HostFixed128Tensor => [runtime] Self::missing_kernel),
        (HostPlacement, (HostString, HostString) -> Float32Tensor => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, HostString) -> Float64Tensor => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, HostString) -> Tensor => [hybrid] attributes[sig] Self::logical_kernel),
    ]
}

pub trait PlacementSave<S: Session, KeyT, T, O> {
    fn save(&self, sess: &S, key: &KeyT, x: &T) -> O;
}

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementSave::save, HostPlacement, (HostString, $value) -> HostUnit, SaveOp);
    )*
)}

modelled!(PlacementSave::save, HostPlacement, (HostString, Tensor) -> HostUnit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, Float32Tensor) -> HostUnit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, Float64Tensor) -> HostUnit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, BooleanTensor) -> HostUnit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, Uint64Tensor) -> HostUnit, SaveOp);

kernel! {
    SaveOp, [
        (HostPlacement, (HostString, HostUnit) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostShape) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostSeed) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostPrfKey) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostBitTensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostRing64Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostRing128Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFloat32Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFloat64Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt8Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt16Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt32Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt64Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint8Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint16Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint32Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint64Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFixed64Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFixed128Tensor) -> HostUnit => [runtime] Self::kernel),
        (HostPlacement, (HostString, Tensor) -> HostUnit => [hybrid] Self::logical_kernel),
        (HostPlacement, (HostString, Float32Tensor) -> HostUnit => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, Float64Tensor) -> HostUnit => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, BooleanTensor) -> HostUnit => [hybrid] Self::bool_kernel),
        (HostPlacement, (HostString, Uint64Tensor) -> HostUnit => [hybrid] Self::u64_kernel),
    ]
}
