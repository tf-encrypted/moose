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

modelled_kernel! {
    PlacementInput::input, InputOp{arg_name: String},
    [
        (HostPlacement, () -> HostString => [runtime] Self::kernel),
        (HostPlacement, () -> HostUnit => [runtime] Self::missing_kernel),
        (HostPlacement, () -> HostShape => [runtime] Self::kernel),
        (HostPlacement, () -> HostSeed => [runtime] Self::kernel),
        (HostPlacement, () -> HostPrfKey => [runtime] Self::kernel),
        (HostPlacement, () -> HostBitArray64 => [concrete] Self::host_bitarray64),
        (HostPlacement, () -> HostBitArray128 => [concrete] Self::host_bitarray128),
        (HostPlacement, () -> HostBitArray224 => [concrete] Self::host_bitarray224),
        (HostPlacement, () -> HostBitTensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostRing64Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostRing128Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostFloat32Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostFloat64Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostInt8Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostInt16Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostInt32Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostInt64Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostUint8Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostUint16Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostUint32Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostUint64Tensor => [runtime] Self::kernel),
        (HostPlacement, () -> HostFixed64Tensor => [runtime] Self::missing_kernel),
        (HostPlacement, () -> HostFixed128Tensor => [runtime] Self::missing_kernel),
        (HostPlacement, () -> Tensor => [concrete] custom |op| {
            let sig = op.sig.clone();
            let arg_name = op.arg_name.clone();
            Ok(Box::new(move |sess, plc| {
                Self::logical_kernel(sess, plc, sig.clone(), arg_name.clone())
            }))
        }),
        (HostPlacement, () -> Float32Tensor => [concrete] Self::float_kernel),
        (HostPlacement, () -> Float64Tensor => [concrete] Self::float_kernel),
        (HostPlacement, () -> AesKey => [concrete] Self::aes_kernel_on_host),
        (HostPlacement, () -> HostAesKey => [concrete] Self::host_aes_kernel),
        (HostPlacement, () -> AesTensor => [concrete] Self::aestensor),
        (HostPlacement, () -> Fixed128AesTensor => [concrete] Self::fixed_aestensor),
        (HostPlacement, () -> HostFixed128AesTensor => [concrete] custom |op| {
            let sig = op.sig.clone();
            let arg_name = op.arg_name.clone();
            Ok(Box::new(move |sess, plc| {
                Self::host_fixed_aestensor(sess, plc, sig.clone(), arg_name.clone())
            }))
        }),
        (ReplicatedPlacement, () -> ReplicatedBitTensor => [concrete] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedRing64Tensor => [concrete] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedRing128Tensor => [concrete] Self::replicated_ring_kernel),
        (ReplicatedPlacement, () -> ReplicatedFixed64Tensor => [concrete] custom |op| {
            let sig = op.sig.clone();
            let arg_name = op.arg_name.clone();
            Ok(Box::new(move |sess, plc| {
                Self::replicated_fixed_kernel(sess, plc, sig.clone(), arg_name.clone())
            }))
        }),
        (ReplicatedPlacement, () -> ReplicatedFixed128Tensor => [concrete] custom |op| {
            let sig = op.sig.clone();
            let arg_name = op.arg_name.clone();
            Ok(Box::new(move |sess, plc| {
                Self::replicated_fixed_kernel(sess, plc, sig.clone(), arg_name.clone())
            }))
        }),
        (ReplicatedPlacement, () -> ReplicatedBitArray64 => [concrete] Self::replicated_bitarray64),
        (ReplicatedPlacement, () -> ReplicatedBitArray128 => [concrete] Self::replicated_bitarray128),
        (ReplicatedPlacement, () -> ReplicatedBitArray224 => [concrete] Self::replicated_bitarray224),
        (ReplicatedPlacement, () -> AesKey => [concrete] Self::aes_kernel_on_replicated),
        (ReplicatedPlacement, () -> ReplicatedAesKey => [concrete] Self::replicated_aes_kernel),
    ]
}

pub trait PlacementOutput<S: Session, T, O> {
    fn output(&self, sess: &S, x: &T) -> O;
}

modelled_kernel! {
    PlacementOutput::output, OutputOp,
    [
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
        (HostPlacement, (HostString, HostString) -> Tensor => [hybrid] custom |op| {
            use crate::logical::{AbstractTensor, TensorDType};
            match op.sig.ret() {
                Ty::Tensor(TensorDType::Float32) => Ok(Box::new(move |sess, plc, key, query| {
                    Self::logical_kernel::<_, Float32Tensor>(sess, plc, key, query).map(AbstractTensor::Float32)
                })),
                Ty::Tensor(TensorDType::Float64) => Ok(Box::new(move |sess, plc, key, query| {
                    Self::logical_kernel::<_, Float64Tensor>(sess, plc, key, query).map(AbstractTensor::Float64)
                })),
                other => {
                    return Err(Error::UnimplementedOperator(
                        format!("Cannot load tensor of type {:?}", other)))
                },
            }
        }),
    ]
}

pub trait PlacementSave<S: Session, KeyT, T, O> {
    fn save(&self, sess: &S, key: &KeyT, x: &T) -> O;
}

modelled_kernel! {
    PlacementSave::save, SaveOp,
    [
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
