use super::*;

pub trait PlacementDeriveSeed<S: Session, KeyT, SeedT> {
    fn derive_seed(&self, sess: &S, sync_key: SyncKey, key: &KeyT) -> SeedT;
}

modelled_kernel! {
    PlacementDeriveSeed::derive_seed, PrimDeriveSeedOp{sync_key: SyncKey},
    [
        (HostPlacement, (PrfKey) -> Seed => [runtime] Self::kernel),
    ]
}

pub trait PlacementSample<S: Session, ShapeT, O> {
    fn sample(&self, sess: &S, max_value: Option<u64>, shape: &ShapeT) -> O;
}

modelled!(PlacementSample::sample, HostPlacement, attributes[max_value: Option<u64>] (HostShape) -> HostRing64Tensor, RingSampleOp);
modelled!(PlacementSample::sample, HostPlacement, attributes[max_value: Option<u64>] (HostShape) -> HostRing128Tensor, RingSampleOp);

kernel! {
    RingSampleOp,
    [
        (HostPlacement, (HostShape) -> HostRing64Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_uniform_u64(ctx, plc, shape)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_bits_u64(ctx, plc, shape)
                })),
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleOp with max_value != 1".to_string()
                )),
            }
        }),
        (HostPlacement, (HostShape) -> HostRing128Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_uniform_u128(ctx, plc, shape)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape| {
                    Self::kernel_bits_u128(ctx, plc, shape)
                })),
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleOp with max_value != 1".to_string()
                )),
            }
        }),
    ]
}

pub trait PlacementSampleUniform<S: Session, ShapeT, O> {
    fn sample_uniform(&self, sess: &S, shape: &ShapeT) -> O;
}

modelled_kernel! {
    PlacementSampleUniform::sample_uniform, BitSampleOp,
    [
        (HostPlacement, (HostShape) -> HostBitTensor => [runtime] Self::kernel),
    ]
}

impl<S: Session, ShapeT, O, P> PlacementSampleUniform<S, ShapeT, O> for P
where
    P: PlacementSample<S, ShapeT, O>,
{
    fn sample_uniform(&self, sess: &S, shape: &ShapeT) -> O {
        self.sample(sess, None, shape)
    }
}

pub trait PlacementSampleBits<S: Session, ShapeT, O> {
    fn sample_bits(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementSampleBits<S, ShapeT, O> for P
where
    P: PlacementSample<S, ShapeT, O>,
{
    fn sample_bits(&self, sess: &S, shape: &ShapeT) -> O {
        self.sample(sess, Some(1), shape)
    }
}

pub trait PlacementSampleSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_seeded(&self, sess: &S, max_value: Option<u64>, shape: &ShapeT, seed: &SeedT) -> O;
}

modelled!(PlacementSampleSeeded::sample_seeded, HostPlacement, attributes[max_value: Option<u64>] (HostShape, Seed) -> HostRing64Tensor, RingSampleSeededOp);
modelled!(PlacementSampleSeeded::sample_seeded, HostPlacement, attributes[max_value: Option<u64>] (HostShape, Seed) -> HostRing128Tensor, RingSampleSeededOp);

kernel! {
    RingSampleSeededOp,
    [
        (HostPlacement, (HostShape, Seed) -> HostRing64Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_uniform_u64(ctx, plc, shape, seed)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_bits_u64(ctx, plc, shape, seed)
                })),
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleSeededOp with max_value != 1".to_string()
                )),
            }
        }),
        (HostPlacement, (HostShape, Seed) -> HostRing128Tensor => [runtime] custom |op| {
            match op.max_value {
                None => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_uniform_u128(ctx, plc, shape, seed)
                })),
                Some(max_value) if max_value == 1 => Ok(Box::new(|ctx, plc, shape, seed| {
                    Self::kernel_bits_u128(ctx, plc, shape, seed)
                })),
                _ => Err(Error::UnimplementedOperator(
                    "RingSampleSeededOp with max_value != 1".to_string()
                )),
            }
        }),
    ]
}

pub trait PlacementSampleUniformSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_uniform_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

modelled_kernel! {
    PlacementSampleUniformSeeded::sample_uniform_seeded, BitSampleSeededOp,
    [
        (HostPlacement, (HostShape, Seed) -> HostBitTensor => [runtime] Self::kernel),
    ]
}

impl<S: Session, ShapeT, SeedT, O, P> PlacementSampleUniformSeeded<S, ShapeT, SeedT, O> for P
where
    P: PlacementSampleSeeded<S, ShapeT, SeedT, O>,
{
    fn sample_uniform_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O {
        self.sample_seeded(sess, None, shape, seed)
    }
}

pub trait PlacementSampleBitsSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_bits_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

impl<S: Session, ShapeT, SeedT, O, P> PlacementSampleBitsSeeded<S, ShapeT, SeedT, O> for P
where
    P: PlacementSampleSeeded<S, ShapeT, SeedT, O>,
{
    fn sample_bits_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O {
        self.sample_seeded(sess, Some(1), shape, seed)
    }
}
