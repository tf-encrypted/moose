use crate::bit::BitTensor;
use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::{
    map_receive_error, map_send_result, AsyncKernel, CompilationContext, Compile, Kernel,
    SyncKernel,
};
use crate::prim::{PrfKey, RawPrfKey, RawSeed, Seed};
use crate::replicated::ReplicatedSetup;
use crate::ring::{Ring128Tensor, Ring64Tensor};
use crate::standard::{
    Float32Tensor, Float64Tensor, Int32Tensor, Int64Tensor, Shape, Uint32Tensor, Uint64Tensor,
};
use crate::{closure_kernel, function_kernel};
use std::convert::TryFrom;
use std::sync::Arc;

pub trait Context {
    type Value;
    fn execute(&self, op: Operator, plc: &Placement, operands: Vec<Self::Value>) -> Self::Value;

    type ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup;
}

pub struct ConcreteContext {
    // replicated_keys: HashMap<ReplicatedPlacement, ReplicatedSetup>, // TODO
}

impl Context for ConcreteContext {
    type Value = Value;

    fn execute(&self, op: Operator, plc: &Placement, operands: Vec<Value>) -> Value {
        unimplemented!() // TODO
    }

    type ReplicatedSetup = ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup {
        unimplemented!() // TODO
    }
}

pub trait DispatchKernel<C: Context> {
    fn compile<'c>(
        &self,
        ctx: &'c C,
        plc: &Placement,
    ) -> Box<dyn Fn(Vec<C::Value>) -> C::Value + 'c>;
}

// TODO if rustc can't figure out how to optimize Box<dyn Fn...> for
// function kernels then we could consider returning an enum over
// fn.. and Box<dyn Fn...> in the traits below instead

pub trait NullaryKernel<C: Context, P, Y> {
    fn compile(&self, ctx: &C, plc: &P) -> Box<dyn Fn(&C, &P) -> Y>;
}

pub trait UnaryKernel<C: Context, P, X0, Y> {
    fn compile(&self, ctx: &C, plc: &P) -> Box<dyn Fn(&C, &P, X0) -> Y>;
}

pub trait BinaryKernel<C: Context, P, X0, X1, Y> {
    fn compile(&self, ctx: &C, plc: &P) -> Box<dyn Fn(&C, &P, X0, X1) -> Y>;
}

pub trait TernaryKernel<C: Context, P, X0, X1, X2, Y> {
    fn compile(&self, ctx: &C, plc: &P) -> Box<dyn Fn(&C, &P, X0, X1, X2) -> Y>;
}

pub(crate) trait NullaryKernelCheck<C: Context, P, Y>
where
    Self: NullaryKernel<C, P, Y>,
{
}

pub(crate) trait UnaryKernelCheck<C: Context, P, X0, Y>
where
    Self: UnaryKernel<C, P, X0, Y>,
{
}

pub(crate) trait BinaryKernelCheck<C: Context, P, X0, X1, Y>
where
    Self: BinaryKernel<C, P, X0, X1, Y>,
{
}

pub(crate) trait TernaryKernelCheck<C: Context, P, X0, X1, X2, Y>
where
    Self: TernaryKernel<C, P, X0, X1, X2, Y>,
{
}

pub trait PlacementKeyGen<C: Context, K> {
    fn keygen(&self, ctx: &C) -> K;
}

pub trait PlacementAdd<C: Context, T, U, O> {
    fn add(&self, ctx: &C, x: &T, y: &U) -> O;
}

pub trait PlacementSub<C: Context, T, U, O> {
    fn sub(&self, ctx: &C, x: &T, y: &U) -> O;
}

pub trait PlacementMul<C: Context, T, U, O> {
    fn mul(&self, ctx: &C, x: &T, y: &U) -> O;
}
pub trait PlacementShl<C: Context, T, O> {
    fn shl(&self, ctx: &C, amount: usize, x: &T) -> O;
}

pub trait PlacementShr<C: Context, T, O> {
    fn shr(&self, ctx: &C, amount: usize, x: &T) -> O;
}

pub trait PlacementXor<C: Context, T, U, O> {
    fn xor(&self, ctx: &C, x: &T, y: &U) -> O;
}

pub trait PlacementAnd<C: Context, T, U, O> {
    fn and(&self, ctx: &C, x: &T, y: &U) -> O;
}

pub trait PlacementMulSetup<C: Context, S, T, U, O> {
    fn mul(&self, ctx: &C, s: &S, x: &T, y: &U) -> O;
}

pub trait PlacementShare<C: Context, T, O> {
    fn share(&self, ctx: &C, x: &T) -> O;
}

pub trait PlacementReveal<C: Context, T, O> {
    fn reveal(&self, ctx: &C, x: &T) -> O;
}

pub trait PlacementSample<C: Context, O> {
    fn sample(&self, ctx: &C) -> O;
}

pub trait PlacementRepToAdd<C: Context, T, O> {
    fn rep_to_add(&self, ctx: &C, x: &T) -> O;
}

fn check_type(v: &Value, expected: Ty) -> Result<()> {
    if v.ty() == expected {
        Ok(())
    } else {
        Err(Error::TypeMismatch {
            expected: format!("{:?}", expected),
            found: v.ty(),
        })
    }
}

impl Compile<SyncKernel> for Operator {
    fn compile(&self, ctx: &CompilationContext) -> Result<SyncKernel> {
        use Operator::*;
        match self {
            Identity(op) => Compile::<SyncKernel>::compile(op, ctx),
            Load(op) => Compile::<SyncKernel>::compile(op, ctx),
            Save(op) => Compile::<SyncKernel>::compile(op, ctx),
            Send(op) => Compile::<SyncKernel>::compile(op, ctx),
            Receive(op) => Compile::<SyncKernel>::compile(op, ctx),
            Input(op) => Compile::<SyncKernel>::compile(op, ctx),
            Output(op) => Compile::<SyncKernel>::compile(op, ctx),
            Constant(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdAdd(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdSub(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdMul(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdDiv(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdDot(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdMean(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdOnes(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdConcatenate(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdExpandDims(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdReshape(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdAtLeast2D(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdShape(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdSlice(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdSum(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdTranspose(op) => Compile::<SyncKernel>::compile(op, ctx),
            StdInverse(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingAdd(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSub(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingMul(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingDot(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSum(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingShape(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingSample(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingFill(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingShl(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingShr(op) => Compile::<SyncKernel>::compile(op, ctx),
            RingInject(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitExtract(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitSample(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitFill(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitXor(op) => Compile::<SyncKernel>::compile(op, ctx),
            BitAnd(op) => Compile::<SyncKernel>::compile(op, ctx),
            PrimDeriveSeed(op) => Compile::<SyncKernel>::compile(op, ctx),
            PrimGenPrfKey(op) => Compile::<SyncKernel>::compile(op, ctx),
            FixedpointRingEncode(op) => Compile::<SyncKernel>::compile(op, ctx),
            FixedpointRingDecode(op) => Compile::<SyncKernel>::compile(op, ctx),
            FixedpointRingMean(op) => Compile::<SyncKernel>::compile(op, ctx),
            op => unimplemented!("deprecated, should not impl for {:?}", op),
        }
    }
}

impl Compile<AsyncKernel> for Operator {
    fn compile(&self, ctx: &CompilationContext) -> Result<AsyncKernel> {
        use Operator::*;
        match self {
            Identity(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Load(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Save(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Send(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Receive(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Input(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Output(op) => Compile::<AsyncKernel>::compile(op, ctx),
            Constant(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdAdd(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdSub(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdMul(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdDiv(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdDot(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdMean(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdOnes(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdConcatenate(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdExpandDims(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdReshape(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdAtLeast2D(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdShape(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdSlice(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdSum(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdTranspose(op) => Compile::<AsyncKernel>::compile(op, ctx),
            StdInverse(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingAdd(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSub(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingMul(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingDot(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSum(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingShape(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingSample(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingFill(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingShl(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingShr(op) => Compile::<AsyncKernel>::compile(op, ctx),
            RingInject(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitExtract(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitSample(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitFill(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitXor(op) => Compile::<AsyncKernel>::compile(op, ctx),
            BitAnd(op) => Compile::<AsyncKernel>::compile(op, ctx),
            PrimDeriveSeed(op) => Compile::<AsyncKernel>::compile(op, ctx),
            PrimGenPrfKey(op) => Compile::<AsyncKernel>::compile(op, ctx),
            FixedpointRingEncode(op) => Compile::<AsyncKernel>::compile(op, ctx),
            FixedpointRingDecode(op) => Compile::<AsyncKernel>::compile(op, ctx),
            FixedpointRingMean(op) => Compile::<AsyncKernel>::compile(op, ctx),
            op => unimplemented!("deprecated, should not impl for {:?}", op),
        }
    }
}

macro_rules! signature {
    (() -> $ret: pat) => {
        Signature::Nullary(NullarySignature { ret: $ret })
    };
    (($t0: pat) -> $ret: pat) => {
        Signature::Unary(UnarySignature {
            arg0: $t0,
            ret: $ret,
        })
    };
    (($t0: pat, $t1: pat) -> $ret: pat) => {
        Signature::Binary(BinarySignature {
            arg0: $t0,
            arg1: $t1,
            ret: $ret,
        })
    };
    (($t0: pat, $t1: pat, $t2: pat) -> $ret: pat) => {
        Signature::Ternary(TernarySignature {
            arg0: $t0,
            arg1: $t1,
            arg2: $t2,
            ret: $ret,
        })
    };
}

macro_rules! std_unary_kernel {
    ($op:ty, $k:expr) => {
        impl Compile<Kernel> for $op {
            fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
                match self.sig {
                    signature![(Ty::Float32Tensor) -> _] => {
                        function_kernel!(Float32Tensor, $k)
                    }
                    signature![(Ty::Float64Tensor) -> _] => {
                        function_kernel!(Float64Tensor, $k)
                    }
                    signature![(Ty::Int32Tensor) -> _] => {
                        function_kernel!(Int32Tensor, $k)
                    }
                    signature![(Ty::Int64Tensor) -> _] => {
                        function_kernel!(Int64Tensor, $k)
                    }
                    signature![(Ty::Uint32Tensor) -> _] => {
                        function_kernel!(Uint32Tensor, $k)
                    }
                    signature![(Ty::Uint64Tensor) -> _] => {
                        function_kernel!(Uint64Tensor, $k)
                    }
                    _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
                }
            }
        }
    };
}

macro_rules! std_binary_kernel {
    ($op:ty, $k:expr) => {
        impl Compile<Kernel> for $op {
            fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
                match self.sig {
                    signature![(Ty::Float32Tensor, Ty::Float32Tensor) -> _] => {
                        function_kernel!(Float32Tensor, Float32Tensor, $k)
                    }
                    signature![(Ty::Float64Tensor, Ty::Float64Tensor) -> _] => {
                        function_kernel!(Float64Tensor, Float64Tensor, $k)
                    }
                    signature![(Ty::Int32Tensor, Ty::Int32Tensor) -> _] => {
                        function_kernel!(Int32Tensor, Int32Tensor, $k)
                    }
                    signature![(Ty::Int64Tensor, Ty::Int64Tensor) -> _] => {
                        function_kernel!(Int64Tensor, Int64Tensor, $k)
                    }
                    signature![(Ty::Uint32Tensor, Ty::Uint32Tensor) -> _] => {
                        function_kernel!(Uint32Tensor, Uint32Tensor, $k)
                    }
                    signature![(Ty::Uint64Tensor, Ty::Uint64Tensor) -> _] => {
                        function_kernel!(Uint64Tensor, Uint64Tensor, $k)
                    }
                    _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
                }
            }
        }
    };
}

std_binary_kernel!(StdAddOp, |x, y| x + y);
std_binary_kernel!(StdSubOp, |x, y| x - y);
std_binary_kernel!(StdMulOp, |x, y| x * y);
std_binary_kernel!(StdDivOp, |x, y| x / y);
std_binary_kernel!(StdDotOp, |x, y| x.dot(y));
std_unary_kernel!(StdShapeOp, |x| x.shape());
std_unary_kernel!(StdTransposeOp, |x| x.transpose());

impl Compile<Kernel> for StdInverseOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                closure_kernel!(Float32Tensor, |x| x.inv())
            }
            signature![(_) -> Ty::Float64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.inv())
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdMeanOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|x| x as usize);
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                closure_kernel!(Float32Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::Float64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::Int32Tensor] => {
                closure_kernel!(Int32Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::Int64Tensor] => {
                closure_kernel!(Int64Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::Uint32Tensor] => {
                closure_kernel!(Uint32Tensor, |x| x.mean(axis))
            }
            signature![(_) -> Ty::Uint64Tensor] => {
                closure_kernel!(Uint64Tensor, |x| x.mean(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdOnesOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                function_kernel!(Shape, |shape| Float32Tensor::ones(shape))
            }
            signature![(_) -> Ty::Float64Tensor] => {
                function_kernel!(Shape, |shape| Float64Tensor::ones(shape))
            }
            signature![(_) -> Ty::Int32Tensor] => {
                function_kernel!(Shape, |shape| Int32Tensor::ones(shape))
            }
            signature![(_) -> Ty::Int64Tensor] => {
                function_kernel!(Shape, |shape| Int64Tensor::ones(shape))
            }
            signature![(_) -> Ty::Uint32Tensor] => {
                function_kernel!(Shape, |shape| Uint32Tensor::ones(shape))
            }
            signature![(_) -> Ty::Uint64Tensor] => {
                function_kernel!(Shape, |shape| Uint64Tensor::ones(shape))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdConcatenateOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        use crate::standard::concatenate;
        let axis = self.axis as usize;
        match self.sig {
            signature![(_, _) -> Ty::Float32Tensor] => {
                closure_kernel!(vec[Float32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::Float64Tensor] => {
                closure_kernel!(vec[Float64Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::Int32Tensor] => {
                closure_kernel!(vec[Int32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::Int64Tensor] => {
                closure_kernel!(vec[Int64Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::Uint32Tensor] => {
                closure_kernel!(vec[Uint32Tensor], |xs| concatenate(axis, &xs))
            }
            signature![(_, _) -> Ty::Uint64Tensor] => {
                closure_kernel!(vec[Uint64Tensor], |xs| concatenate(axis, &xs))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdExpandDimsOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis as usize;
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                closure_kernel!(Float32Tensor, |x| x.expand_dims(axis))
            }
            signature![(_) -> Ty::Float64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.expand_dims(axis))
            }
            signature![(_) -> Ty::Int32Tensor] => {
                closure_kernel!(Int32Tensor, |x| x.expand_dims(axis))
            }
            signature![(_) -> Ty::Int64Tensor] => {
                closure_kernel!(Int64Tensor, |x| x.expand_dims(axis))
            }
            signature![(_) -> Ty::Uint32Tensor] => {
                closure_kernel!(Uint32Tensor, |x| x.expand_dims(axis))
            }
            signature![(_) -> Ty::Uint64Tensor] => {
                closure_kernel!(Uint64Tensor, |x| x.expand_dims(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdReshapeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(_, _) -> Ty::Float32Tensor] => {
                function_kernel!(Float32Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            signature![(_, _) -> Ty::Float64Tensor] => {
                function_kernel!(Float64Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            signature![(_, _) -> Ty::Int32Tensor] => {
                function_kernel!(Int32Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            signature![(_, _) -> Ty::Int64Tensor] => {
                function_kernel!(Int64Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            signature![(_, _) -> Ty::Uint32Tensor] => {
                function_kernel!(Uint32Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            signature![(_, _) -> Ty::Uint64Tensor] => {
                function_kernel!(Uint64Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdAtLeast2DOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let tcv = self.to_column_vector;
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::Float64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::Int32Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::Int64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::Uint32Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            signature![(_) -> Ty::Uint64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdSliceOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let start = self.start as usize;
        let end = self.end as usize;
        match self.sig {
            signature![(_) -> Ty::Shape] => {
                closure_kernel!(Shape, |x| Shape(x.0.slice(start, end), x.1))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdSumOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.sig {
            signature![(_) -> Ty::Float32Tensor] => {
                closure_kernel!(Float32Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::Float64Tensor] => {
                closure_kernel!(Float64Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::Int32Tensor] => {
                closure_kernel!(Int32Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::Int64Tensor] => {
                closure_kernel!(Int64Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::Uint32Tensor] => {
                closure_kernel!(Uint32Tensor, |x| x.sum(axis))
            }
            signature![(_) -> Ty::Uint64Tensor] => {
                closure_kernel!(Uint64Tensor, |x| x.sum(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for PrimDeriveSeedOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let nonce = self.nonce.clone();
        closure_kernel!(PrfKey, |key| Seed(
            RawSeed::from_prf(&key.0, &nonce),
            HostPlacement {
                owner: "TODO".into()
            }
        ))
    }
}

impl Compile<Kernel> for PrimGenPrfKeyOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(|| PrfKey(
            RawPrfKey::generate(),
            HostPlacement {
                owner: "TODO".into()
            }
        ))
    }
}

impl Compile<Kernel> for RingAddOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::Ring64Tensor, Ty::Ring64Tensor) -> _] => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x + y)
            }
            signature![(Ty::Ring128Tensor, Ty::Ring128Tensor) -> _] => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x + y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSubOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::Ring64Tensor, Ty::Ring64Tensor) -> _] => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x - y)
            }
            signature![(Ty::Ring128Tensor, Ty::Ring128Tensor) -> _] => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x - y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingMulOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::Ring64Tensor, Ty::Ring64Tensor) -> _] => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x * y)
            }
            signature![(Ty::Ring128Tensor, Ty::Ring128Tensor) -> _] => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x * y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingDotOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::Ring64Tensor, Ty::Ring64Tensor) -> _] => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x.dot(y))
            }
            signature![(Ty::Ring128Tensor, Ty::Ring128Tensor) -> _] => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x.dot(y))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSumOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.sig {
            signature![(_) -> Ty::Ring64Tensor] => closure_kernel!(Ring64Tensor, |x| x.sum(axis)),
            signature![(_) -> Ty::Ring128Tensor] => {
                closure_kernel!(Ring128Tensor, |x| x.sum(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingShapeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            signature![(Ty::Ring64Tensor) -> Ty::Shape] => {
                function_kernel!(Ring64Tensor, |x| x.shape())
            }
            signature![(Ty::Ring128Tensor) -> Ty::Shape] => {
                function_kernel!(Ring128Tensor, |x| x.shape())
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingFillOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (&self.sig, self.value.clone()) {
            (signature![(_) -> Ty::Ring64Tensor], Primitive::Ring64(value)) => {
                closure_kernel!(Shape, |shape| Ring64Tensor::fill(&shape.0, value))
            }
            (signature![(_) -> Ty::Ring128Tensor], Primitive::Ring128(value)) => {
                closure_kernel!(Shape, |shape| Ring128Tensor::fill(&shape.0, value))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSampleOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (&self.sig, self.max_value) {
            (signature![(_, _) -> Ty::Ring64Tensor], None) => {
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_uniform(
                    &shape.0, &seed.0
                ))
            }
            (signature!((_, _) -> Ty::Ring64Tensor), Some(max_value)) if max_value == 1 => {
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_bits(
                    &shape.0, &seed.0
                ))
            }
            (signature![(_, _) -> Ty::Ring128Tensor], None) => {
                function_kernel!(Shape, Seed, |shape, seed| Ring128Tensor::sample_uniform(
                    &shape.0, &seed.0
                ))
            }
            (signature![(_, _) -> Ty::Ring128Tensor], Some(max_value)) if max_value == 1 => {
                function_kernel!(Shape, Seed, |shape, seed| Ring128Tensor::sample_bits(
                    &shape.0, &seed.0
                ))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingShlOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let amount = self.amount;
        match self.sig {
            signature![(_) -> Ty::Ring64Tensor] => {
                closure_kernel!(Ring64Tensor, |x| x << amount)
            }
            signature![(_) -> Ty::Ring128Tensor] => {
                closure_kernel!(Ring128Tensor, |x| x << amount)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingShrOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let amount = self.amount;
        match self.sig {
            signature![(_) -> Ty::Ring64Tensor] => {
                closure_kernel!(Ring64Tensor, |x| x >> amount)
            }
            signature![(_) -> Ty::Ring128Tensor] => {
                closure_kernel!(Ring128Tensor, |x| x >> amount)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingInjectOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let bit_idx = self.bit_idx;
        match self.sig {
            signature![(_) -> Ty::Ring64Tensor] => {
                closure_kernel!(BitTensor, |x| Ring64Tensor::from(x) << bit_idx)
            }
            signature![(_) -> Ty::Ring128Tensor] => {
                closure_kernel!(BitTensor, |x| Ring128Tensor::from(x) << bit_idx)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for BitExtractOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let bit_idx = self.bit_idx;
        match self.sig {
            signature![(Ty::Ring64Tensor) -> _] => {
                closure_kernel!(Ring64Tensor, |x| x.bit_extract(bit_idx))
            }
            signature![(Ty::Ring128Tensor) -> _] => {
                closure_kernel!(Ring128Tensor, |x| x.bit_extract(bit_idx))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for BitSampleOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(Shape, Seed, |shape, seed| BitTensor::sample_uniform(
            &shape.0, &seed.0
        ))
    }
}

impl Compile<Kernel> for BitFillOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let value = self.value;
        closure_kernel!(Shape, |shape| BitTensor::fill(&shape.0, value))
    }
}

impl Compile<Kernel> for BitXorOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(BitTensor, BitTensor, |x, y| x ^ y)
    }
}

impl Compile<Kernel> for BitAndOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(BitTensor, BitTensor, |x, y| x & y)
    }
}

impl Compile<Kernel> for FixedpointRingEncodeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        use crate::fixedpoint::Convert;
        match self.sig {
            signature![(Ty::Float64Tensor) -> Ty::Ring64Tensor] => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(Float64Tensor, |x| Ring64Tensor::encode(&x, scaling_factor))
            }
            signature![(Ty::Float64Tensor) -> Ty::Ring128Tensor] => {
                let scaling_factor = u128::pow(self.scaling_base as u128, self.scaling_exp);
                closure_kernel!(Float64Tensor, |x| Ring128Tensor::encode(&x, scaling_factor))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for FixedpointRingDecodeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        use crate::fixedpoint::Convert;
        match self.sig {
            signature![(Ty::Ring64Tensor) -> _] => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(Ring64Tensor, |x| Ring64Tensor::decode(&x, scaling_factor))
            }
            signature![(Ty::Ring128Tensor) -> _] => {
                let scaling_factor = u128::pow(self.scaling_base as u128, self.scaling_exp);
                closure_kernel!(Ring128Tensor, |x| Ring128Tensor::decode(&x, scaling_factor))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for FixedpointRingMeanOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis;
        match self.sig {
            signature![(_) -> Ty::Ring64Tensor] => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(Ring64Tensor, |x| Ring64Tensor::ring_mean(
                    x,
                    axis,
                    scaling_factor
                ))
            }
            signature![(_) -> Ty::Ring128Tensor] => {
                let scaling_factor = u128::pow(self.scaling_base as u128, self.scaling_exp);
                closure_kernel!(Ring128Tensor, |x| Ring128Tensor::ring_mean(
                    x,
                    axis,
                    scaling_factor
                ))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for ConstantOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let value = self.value.clone();
        Ok(Kernel::NullaryClosure(Arc::new(move || {
            Ok(value.place(&HostPlacement {
                owner: "TODO".into(),
            }))
        })))
    }
}

impl Compile<SyncKernel> for SendOp {
    fn compile(&self, ctx: &CompilationContext) -> Result<SyncKernel> {
        let rendezvous_key = self.rendezvous_key.clone();
        let receiver_id = ctx
            .role_assignment
            .get(&self.receiver)
            .cloned()
            .ok_or_else(|| {
                Error::Compilation(format!(
                    "missing identity assignment for '{}'",
                    &self.receiver
                ))
            })?;

        Ok(SyncKernel::Unary(Box::new(move |sess, v| {
            sess.networking
                .send(&v, &receiver_id, &rendezvous_key, &sess.sid)?;
            Ok(Value::Unit)
        })))
    }
}

impl Compile<AsyncKernel> for SendOp {
    fn compile(&self, ctx: &CompilationContext) -> Result<AsyncKernel> {
        let rendezvous_key = Arc::new(self.rendezvous_key.clone());
        let receiver_id = Arc::new(
            ctx.role_assignment
                .get(&self.receiver)
                .cloned()
                .ok_or_else(|| {
                    Error::Compilation(format!(
                        "missing identity assignment for '{}'",
                        &self.receiver
                    ))
                })?,
        );

        Ok(AsyncKernel::Unary(Box::new(move |sess, v, sender| {
            let sess = Arc::clone(sess);
            let rendezvous_key = Arc::clone(&rendezvous_key);
            let receiver_id = Arc::clone(&receiver_id);

            tokio::spawn(async move {
                let v: Value = v.await.map_err(map_receive_error)?;
                sess.networking
                    .send(&v, &receiver_id, &rendezvous_key, &sess.sid)
                    .await?;
                map_send_result(sender.send(Value::Unit))
            })
        })))
    }
}

impl Compile<SyncKernel> for ReceiveOp {
    fn compile(&self, ctx: &CompilationContext) -> Result<SyncKernel> {
        let expected_ty = self.sig.ret();
        let rendezvous_key = self.rendezvous_key.clone();
        let sender_id = ctx
            .role_assignment
            .get(&self.sender)
            .cloned()
            .ok_or_else(|| {
                Error::Compilation(format!(
                    "missing identity assignment for '{}'",
                    &self.sender
                ))
            })?;

        Ok(SyncKernel::Nullary(Box::new(move |sess| {
            let v: Value = sess
                .networking
                .receive(&sender_id, &rendezvous_key, &sess.sid)?;
            check_type(&v, expected_ty)?;
            Ok(v)
        })))
    }
}

impl Compile<AsyncKernel> for ReceiveOp {
    fn compile(&self, ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.sig.ret();
        let rendezvous_key = Arc::new(self.rendezvous_key.clone());
        let sender_id = Arc::new(ctx.role_assignment.get(&self.sender).cloned().ok_or_else(
            || {
                Error::Compilation(format!(
                    "missing identity assignment for '{}'",
                    &self.sender
                ))
            },
        )?);

        Ok(AsyncKernel::Nullary(Box::new(move |sess, sender| {
            let sess = Arc::clone(sess);
            let rendezvous_key = Arc::clone(&rendezvous_key);
            let sender_id = Arc::clone(&sender_id);

            tokio::spawn(async move {
                let v: Value = sess
                    .networking
                    .receive(&sender_id, &rendezvous_key, &sess.sid)
                    .await?;
                check_type(&v, expected_ty)?;
                map_send_result(sender.send(v))
            })
        })))
    }
}

impl Compile<SyncKernel> for IdentityOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        let expected_ty = self.sig.ret();

        Ok(SyncKernel::Unary(Box::new(move |_sess, v| {
            check_type(&v, expected_ty)?;
            Ok(v)
        })))
    }
}

impl Compile<AsyncKernel> for IdentityOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.sig.ret();

        Ok(AsyncKernel::Unary(Box::new(move |_sess, v, sender| {
            tokio::spawn(async move {
                let v: Value = v.await.map_err(map_receive_error)?;
                check_type(&v, expected_ty)?;
                map_send_result(sender.send(v))
            })
        })))
    }
}

impl Compile<SyncKernel> for InputOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        let arg_name = self.arg_name.clone();
        let expected_ty = self.sig.ret();

        Ok(SyncKernel::Nullary(Box::new(move |sess| {
            let arg = sess
                .arguments
                .get(&arg_name)
                .cloned()
                .ok_or_else(|| Error::MissingArgument(arg_name.clone()))?;
            check_type(&arg, expected_ty)?;
            Ok(arg)
        })))
    }
}

impl Compile<AsyncKernel> for InputOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.sig.ret();
        let arg_name = Arc::new(self.arg_name.clone());

        Ok(AsyncKernel::Nullary(Box::new(move |sess, sender| {
            let sess = Arc::clone(sess);
            let arg_name = Arc::clone(&arg_name);

            tokio::spawn(async move {
                let arg = sess
                    .arguments
                    .get(arg_name.as_ref())
                    .cloned()
                    .ok_or_else(|| Error::MissingArgument(arg_name.as_ref().clone()))?;
                check_type(&arg, expected_ty)?;
                map_send_result(sender.send(arg))
            })
        })))
    }
}

impl Compile<SyncKernel> for OutputOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        Ok(SyncKernel::Unary(Box::new(move |_sess, x0| Ok(x0))))
    }
}

impl Compile<AsyncKernel> for OutputOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        Ok(AsyncKernel::Unary(Box::new(move |_sess, x0, sender| {
            tokio::spawn(async move {
                let val = x0.await.map_err(map_receive_error)?;
                map_send_result(sender.send(val))
            })
        })))
    }
}

impl Compile<SyncKernel> for SaveOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        let expected_ty = self.sig.arg(1)?;

        Ok(SyncKernel::Binary(Box::new(move |sess, key, val| {
            let key = String::try_from(key)?;
            check_type(&val, expected_ty)?;
            sess.storage.save(&key, &sess.sid, &val)?;
            Ok(Value::Unit)
        })))
    }
}

impl Compile<AsyncKernel> for SaveOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.sig.ret();

        Ok(AsyncKernel::Binary(Box::new(
            move |sess, key, val, sender| {
                let sess = Arc::clone(sess);

                tokio::spawn(async move {
                    let key = String::try_from(key.await.map_err(map_receive_error)?)?;
                    let val = val.await.map_err(map_receive_error)?;
                    check_type(&val, expected_ty)?;
                    sess.storage.save(&key, &sess.sid, &val).await?;
                    map_send_result(sender.send(Value::Unit))
                })
            },
        )))
    }
}

impl Compile<SyncKernel> for LoadOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<SyncKernel> {
        let expected_ty = self.sig.ret();

        Ok(SyncKernel::Binary(Box::new(move |sess, key, query| {
            let key = String::try_from(key)?;
            let _query = String::try_from(query)?;
            let val = sess
                .storage
                .load(&key, &sess.sid, Some(expected_ty), &_query)?;
            check_type(&val, expected_ty)?;
            Ok(val)
        })))
    }
}

impl Compile<AsyncKernel> for LoadOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.sig.ret();

        Ok(AsyncKernel::Binary(Box::new(
            move |sess, key, query, sender| {
                let sess = Arc::clone(sess);

                tokio::spawn(async move {
                    let key = String::try_from(key.await.map_err(map_receive_error)?)?;
                    let _query = String::try_from(query.await.map_err(map_receive_error)?)?;
                    let val = sess
                        .storage
                        .load(&key, &sess.sid, Some(expected_ty), &_query)
                        .await?;
                    check_type(&val, expected_ty)?;
                    map_send_result(sender.send(val))
                })
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use crate::execution::*;
    use std::convert::TryInto;

    #[test]
    fn test_standard_shape_ops() -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Constant{value = Float32Tensor([[1.0, 2.0], [3.0, 4.0]])} @Host(alice)
        shape = StdShape: (Float32Tensor) -> Float32Tensor (x) @Host(alice)
        expand_dims = StdExpandDims {axis = 2}: (Float32Tensor) -> Float32Tensor (x) @Host(alice)
        transpose = StdTranspose : (Float32Tensor) -> Float32Tensor (x) @Host(alice)"#;

        let exec = TestExecutor::default();
        let _outputs = exec.run_computation(&source.try_into()?, SyncArgs::new())?;
        Ok(())
    }
}
