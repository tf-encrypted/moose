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
        (HostPlacement, () -> Unit => [runtime] attributes[arg_name] Self::missing_kernel),
        (HostPlacement, () -> HostShape => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> Seed => [runtime] attributes[arg_name] Self::kernel),
        (HostPlacement, () -> PrfKey => [runtime] attributes[arg_name] Self::kernel),
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
        (HostPlacement, (Unit) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostShape) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (Seed) -> Seed => [runtime] Self::kernel),
        (HostPlacement, (PrfKey) -> PrfKey => [runtime] Self::kernel),
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

modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> HostFloat64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> Float64Tensor, LoadOp);
modelled!(PlacementLoad::load, HostPlacement, (HostString, HostString) -> Tensor, LoadOp);

kernel! {
    LoadOp, [
        (HostPlacement, (HostString, HostString) -> Unit => [runtime] Self::missing_kernel),
        (HostPlacement, (HostString, HostString) -> HostShape => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> Seed => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> PrfKey => [runtime] Self::kernel),
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
        (HostPlacement, (HostString, HostString) -> Float64Tensor => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, HostString) -> Tensor => [hybrid] Self::logical_kernel),
    ]
}

pub trait PlacementSave<S: Session, KeyT, T, O> {
    fn save(&self, sess: &S, key: &KeyT, x: &T) -> O;
}

for_all_values! {( $($value:ty),* ) => (
    $(
        modelled!(PlacementSave::save, HostPlacement, (HostString, $value) -> Unit, SaveOp);
    )*
)}

modelled!(PlacementSave::save, HostPlacement, (HostString, Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, Float32Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, Float64Tensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, BooleanTensor) -> Unit, SaveOp);
modelled!(PlacementSave::save, HostPlacement, (HostString, Uint64Tensor) -> Unit, SaveOp);

kernel! {
    SaveOp, [
        (HostPlacement, (HostString, Unit) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostShape) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, Seed) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, PrfKey) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostString) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostBitTensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostRing64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostRing128Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFloat32Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFloat64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt8Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt16Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt32Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostInt64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint8Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint16Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint32Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostUint64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFixed64Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, HostFixed128Tensor) -> Unit => [runtime] Self::kernel),
        (HostPlacement, (HostString, Tensor) -> Unit => [hybrid] Self::logical_kernel),
        (HostPlacement, (HostString, Float32Tensor) -> Unit => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, Float64Tensor) -> Unit => [hybrid] Self::float_kernel),
        (HostPlacement, (HostString, BooleanTensor) -> Unit => [hybrid] Self::bool_kernel),
        (HostPlacement, (HostString, Uint64Tensor) -> Unit => [hybrid] Self::u64_kernel),
    ]
}
