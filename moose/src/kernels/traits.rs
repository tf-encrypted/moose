use super::*;

pub trait PlacementShape<S: Session, T, ShapeT> {
    fn shape(&self, sess: &S, x: &T) -> ShapeT;
}

pub trait PlacementReshape<S: Session, T, ShapeT, O> {
    fn reshape(&self, sess: &S, x: &T, shape: &ShapeT) -> O;
}

pub trait PlacementDecrypt<S: Session, KeyT, C, O> {
    fn decrypt(&self, sess: &S, key: &KeyT, ciphertext: &C) -> O;
}

pub trait PlacementKeyGen<S: Session, KeyT> {
    fn gen_key(&self, sess: &S) -> KeyT;
}

pub trait PlacementSetupGen<S: Session, SetupT> {
    fn gen_setup(&self, sess: &S) -> SetupT;
}

pub trait PlacementDeriveSeed<S: Session, KeyT, SeedT> {
    fn derive_seed(&self, sess: &S, sync_key: SyncKey, key: &KeyT) -> SeedT;
}

pub trait PlacementAdd<S: Session, T, U, O> {
    fn add(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementSub<S: Session, T, U, O> {
    fn sub(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementNeg<S: Session, T, O> {
    fn neg(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementMul<S: Session, T, U, O> {
    fn mul(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementDiv<S: Session, T, U, O> {
    fn div(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementDot<S: Session, T, U, O> {
    fn dot(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementShl<S: Session, T, O> {
    fn shl(&self, sess: &S, amount: usize, x: &T) -> O;
}

pub trait PlacementShr<S: Session, T, O> {
    fn shr(&self, sess: &S, amount: usize, x: &T) -> O;
}

pub trait PlacementXor<S: Session, T, U, O> {
    fn xor(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementAnd<S: Session, T, U, O> {
    fn and(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementOr<S: Session, T, U, O> {
    fn or(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementBitExtract<S: Session, T, O> {
    fn bit_extract(&self, sess: &S, bit_idx: usize, x: &T) -> O;
}

pub trait PlacementBitDec<S: Session, T, O> {
    fn bit_decompose(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementBitCompose<S: Session, T, O> {
    fn bit_compose(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementRingInject<S: Session, T, O> {
    fn ring_inject(&self, sess: &S, bit_idx: usize, x: &T) -> O;
}

pub trait PlacementShare<S: Session, T, O> {
    fn share(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementReveal<S: Session, T, O> {
    fn reveal(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementFill<S: Session, ShapeT, O> {
    fn fill(&self, sess: &S, value: Constant, shape: &ShapeT) -> O;
}

pub trait PlacementZeros<S: Session, ShapeT, O> {
    fn zeros(&self, sess: &S, shape: &ShapeT) -> O;
}

pub trait PlacementMean<S: Session, T, O> {
    fn mean(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

pub trait PlacementMeanAsFixedpoint<S: Session, T, O> {
    fn mean_as_fixedpoint(
        &self,
        sess: &S,
        axis: Option<u32>,
        scaling_base: u64,
        scaling_exp: u32,
        x: &T,
    ) -> O;
}

pub trait PlacementSqrt<S: Session, T, O> {
    fn sqrt(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementAddN<S: Session, T, O> {
    fn add_n(&self, sess: &S, x: &[T]) -> O;
}

pub trait PlacementSum<S: Session, T, O> {
    fn sum(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

pub trait PlacementEqual<S: Session, T, U, O> {
    fn equal(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementMux<S: Session, T, U, V, O> {
    fn mux(&self, sess: &S, s: &T, x: &U, y: &V) -> O;
}

pub trait PlacementPow2<S: Session, T, O> {
    fn pow2(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementExp<S: Session, T, O> {
    fn exp(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementSigmoid<S: Session, T, O> {
    fn sigmoid(&self, sess: &S, x: &T) -> O;
}
pub trait PlacementLessThan<S: Session, T, U, O> {
    fn less(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementGreaterThan<S: Session, T, U, O> {
    fn greater_than(&self, sess: &S, x: &T, y: &U) -> O;
}

pub trait PlacementDemirror<S: Session, T, O> {
    fn demirror(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementMirror<S: Session, T, O> {
    fn mirror(&self, sess: &S, x: &T) -> O;
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

pub trait PlacementSample<S: Session, ShapeT, O> {
    fn sample(&self, sess: &S, max_value: Option<u64>, shape: &ShapeT) -> O;
}

pub trait PlacementSampleUniform<S: Session, ShapeT, O> {
    fn sample_uniform(&self, sess: &S, shape: &ShapeT) -> O;
}

pub trait PlacementSampleBits<S: Session, ShapeT, O> {
    fn sample_bits(&self, sess: &S, shape: &ShapeT) -> O;
}

impl<S: Session, ShapeT, O, P> PlacementSampleUniform<S, ShapeT, O> for P
where
    P: PlacementSample<S, ShapeT, O>,
{
    fn sample_uniform(&self, sess: &S, shape: &ShapeT) -> O {
        self.sample(sess, None, shape)
    }
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

pub trait PlacementSampleUniformSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_uniform_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

pub trait PlacementSampleBitsSeeded<S: Session, ShapeT, SeedT, O> {
    fn sample_bits_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O;
}

impl<S: Session, ShapeT, SeedT, O, P> PlacementSampleUniformSeeded<S, ShapeT, SeedT, O> for P
where
    P: PlacementSampleSeeded<S, ShapeT, SeedT, O>,
{
    fn sample_uniform_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O {
        self.sample_seeded(sess, None, shape, seed)
    }
}

impl<S: Session, ShapeT, SeedT, O, P> PlacementSampleBitsSeeded<S, ShapeT, SeedT, O> for P
where
    P: PlacementSampleSeeded<S, ShapeT, SeedT, O>,
{
    fn sample_bits_seeded(&self, sess: &S, shape: &ShapeT, seed: &SeedT) -> O {
        self.sample_seeded(sess, Some(1), shape, seed)
    }
}

pub trait PlacementRepToAdt<S: Session, T, O> {
    fn rep_to_adt(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementAdtToRep<S: Session, T, O> {
    fn adt_to_rep(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementTruncPr<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: u32, x: &T) -> O;
}

pub trait TruncPrProvider<S: Session, T, O> {
    fn trunc_pr(&self, sess: &S, amount: usize, provider: &HostPlacement, x: &T) -> O;
}

pub trait PlacementAbs<S: Session, T, O> {
    fn abs(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementMsb<S: Session, T, O> {
    fn msb(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementSign<S: Session, T, O> {
    fn sign(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementPlace<S: Session, T> {
    fn place(&self, sess: &S, x: T) -> T;
}

pub trait PlacementConstant<S: Session, O> {
    fn constant(&self, sess: &S, value: Constant) -> O;
}

pub trait PlacementIdentity<S: Session, T, O> {
    fn identity(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementInput<S: Session, O> {
    fn input(&self, sess: &S, arg_name: String) -> O;
}

pub trait PlacementOutput<S: Session, T, O> {
    fn output(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementLoad<S: Session, KeyT, QueryT, O> {
    fn load(&self, sess: &S, key: &KeyT, query: &QueryT) -> O;
}

pub trait PlacementSave<S: Session, KeyT, T, O> {
    fn save(&self, sess: &S, key: &KeyT, x: &T) -> O;
}

pub trait PlacementSend<S: Session, T, O> {
    fn send(&self, sess: &S, rendezvous_key: RendezvousKey, receiver: Role, x: &T) -> O;
}

pub trait PlacementReceive<S: Session, O> {
    fn receive(&self, sess: &S, rendezvous_key: RendezvousKey, sender: Role) -> O;
}

pub trait PlacementAtLeast2D<S: Session, T, O> {
    fn at_least_2d(&self, sess: &S, to_column_vector: bool, x: &T) -> O;
}

pub trait PlacementRingFixedpointEncode<S: Session, T, O> {
    fn fixedpoint_ring_encode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
}

pub trait PlacementRingFixedpointDecode<S: Session, T, O> {
    fn fixedpoint_ring_decode(&self, sess: &S, scaling_base: u64, scaling_exp: u32, x: &T) -> O;
}

pub trait PlacementFixedpointEncode<S: Session, T, O> {
    fn fixedpoint_encode(
        &self,
        sess: &S,
        fractional_precision: u32,
        integral_precision: u32,
        x: &T,
    ) -> O;
}

pub trait PlacementFixedpointDecode<S: Session, T, O> {
    fn fixedpoint_decode(&self, sess: &S, precision: u32, x: &T) -> O;
}

pub trait PlacementExpandDims<S: Session, T, O> {
    fn expand_dims(&self, sess: &S, axis: Vec<u32>, x: &T) -> O;
}

pub trait PlacementSqueeze<S: Session, T, O> {
    fn squeeze(&self, sess: &S, axis: Option<u32>, x: &T) -> O;
}

pub trait PlacementConcatenate<S: Session, TS, O> {
    fn concatenate(&self, sess: &S, axis: u32, xs: &[TS]) -> O;
}

pub trait PlacementTranspose<S: Session, T, O> {
    fn transpose(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementInverse<S: Session, T, O> {
    fn inverse(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementCast<S: Session, T, O> {
    fn cast(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementSlice<S: Session, T, O> {
    fn slice(&self, sess: &S, slice_info: SliceInfo, x: &T) -> O;
}

pub trait PlacementDiag<S: Session, T, O> {
    fn diag(&self, sess: &S, x: &T) -> O;
}

pub trait PlacementIndexAxis<S: Session, T, O> {
    fn index_axis(&self, sess: &S, axis: usize, index: usize, x: &T) -> O;
}

pub trait PlacementIndex<S: Session, T, O> {
    fn index(&self, sess: &S, index: usize, x: &T) -> O;
}

pub trait PlacementShlDim<S: Session, T, O> {
    fn shl_dim(&self, sess: &S, amount: usize, ring_size: usize, x: &T) -> O;
}
