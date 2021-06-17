use crate::bit::BitTensor;
use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::{
    map_receive_error, map_send_result, AsyncKernel, CompilationContext, Compile, Kernel,
    SyncKernel,
};
use crate::prim::{PrfKey, Seed};
use crate::ring::{Ring128Tensor, Ring64Tensor};
use crate::standard::{
    Float32Tensor, Float64Tensor, Int32Tensor, Int64Tensor, Shape, Uint32Tensor, Uint64Tensor,
};
use crate::{closure_kernel, function_kernel};
use std::convert::TryFrom;
use std::sync::Arc;

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
            Identity(op) => op.compile(ctx),
            Load(op) => op.compile(ctx),
            Save(op) => op.compile(ctx),
            Send(op) => op.compile(ctx),
            Receive(op) => op.compile(ctx),
            Input(op) => op.compile(ctx),
            Output(op) => op.compile(ctx),
            Constant(op) => op.compile(ctx),
            StdAdd(op) => op.compile(ctx),
            StdSub(op) => op.compile(ctx),
            StdMul(op) => op.compile(ctx),
            StdDiv(op) => op.compile(ctx),
            StdDot(op) => op.compile(ctx),
            StdMean(op) => op.compile(ctx),
            StdOnes(op) => op.compile(ctx),
            StdConcatenate(op) => op.compile(ctx),
            StdExpandDims(op) => op.compile(ctx),
            StdReshape(op) => op.compile(ctx),
            StdAtLeast2D(op) => op.compile(ctx),
            StdShape(op) => op.compile(ctx),
            StdSlice(op) => op.compile(ctx),
            StdSum(op) => op.compile(ctx),
            StdTranspose(op) => op.compile(ctx),
            StdInverse(op) => op.compile(ctx),
            RingAdd(op) => op.compile(ctx),
            RingSub(op) => op.compile(ctx),
            RingMul(op) => op.compile(ctx),
            RingDot(op) => op.compile(ctx),
            RingSum(op) => op.compile(ctx),
            RingShape(op) => op.compile(ctx),
            RingSample(op) => op.compile(ctx),
            RingFill(op) => op.compile(ctx),
            RingShl(op) => op.compile(ctx),
            RingShr(op) => op.compile(ctx),
            RingInject(op) => op.compile(ctx),
            BitExtract(op) => op.compile(ctx),
            BitSample(op) => op.compile(ctx),
            BitFill(op) => op.compile(ctx),
            BitXor(op) => op.compile(ctx),
            BitAnd(op) => op.compile(ctx),
            PrimDeriveSeed(op) => op.compile(ctx),
            PrimGenPrfKey(op) => op.compile(ctx),
            FixedpointRingEncode(op) => op.compile(ctx),
            FixedpointRingDecode(op) => op.compile(ctx),
            FixedpointRingMean(op) => op.compile(ctx),
        }
    }
}

impl Compile<AsyncKernel> for Operator {
    fn compile(&self, ctx: &CompilationContext) -> Result<AsyncKernel> {
        use Operator::*;
        match self {
            Identity(op) => op.compile(ctx),
            Load(op) => op.compile(ctx),
            Save(op) => op.compile(ctx),
            Send(op) => op.compile(ctx),
            Receive(op) => op.compile(ctx),
            Input(op) => op.compile(ctx),
            Output(op) => op.compile(ctx),
            Constant(op) => op.compile(ctx),
            StdAdd(op) => op.compile(ctx),
            StdSub(op) => op.compile(ctx),
            StdMul(op) => op.compile(ctx),
            StdDiv(op) => op.compile(ctx),
            StdDot(op) => op.compile(ctx),
            StdMean(op) => op.compile(ctx),
            StdOnes(op) => op.compile(ctx),
            StdConcatenate(op) => op.compile(ctx),
            StdExpandDims(op) => op.compile(ctx),
            StdReshape(op) => op.compile(ctx),
            StdAtLeast2D(op) => op.compile(ctx),
            StdShape(op) => op.compile(ctx),
            StdSlice(op) => op.compile(ctx),
            StdSum(op) => op.compile(ctx),
            StdTranspose(op) => op.compile(ctx),
            StdInverse(op) => op.compile(ctx),
            RingAdd(op) => op.compile(ctx),
            RingSub(op) => op.compile(ctx),
            RingMul(op) => op.compile(ctx),
            RingDot(op) => op.compile(ctx),
            RingSum(op) => op.compile(ctx),
            RingShape(op) => op.compile(ctx),
            RingSample(op) => op.compile(ctx),
            RingFill(op) => op.compile(ctx),
            RingShl(op) => op.compile(ctx),
            RingShr(op) => op.compile(ctx),
            RingInject(op) => op.compile(ctx),
            BitExtract(op) => op.compile(ctx),
            BitSample(op) => op.compile(ctx),
            BitFill(op) => op.compile(ctx),
            BitXor(op) => op.compile(ctx),
            BitAnd(op) => op.compile(ctx),
            PrimDeriveSeed(op) => op.compile(ctx),
            PrimGenPrfKey(op) => op.compile(ctx),
            FixedpointRingEncode(op) => op.compile(ctx),
            FixedpointRingDecode(op) => op.compile(ctx),
            FixedpointRingMean(op) => op.compile(ctx),
        }
    }
}

macro_rules! std_unary_kernel {
    ($op:ty, $k:expr) => {
        impl Compile<Kernel> for $op {
            fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
                match self.ty {
                    Ty::Float32TensorTy => {
                        function_kernel!(Float32Tensor, $k)
                    }
                    Ty::Float64TensorTy => {
                        function_kernel!(Float64Tensor, $k)
                    }
                    Ty::Int32TensorTy => {
                        function_kernel!(Int32Tensor, $k)
                    }
                    Ty::Int64TensorTy => {
                        function_kernel!(Int64Tensor, $k)
                    }
                    Ty::Uint32TensorTy => {
                        function_kernel!(Uint32Tensor, $k)
                    }
                    Ty::Uint64TensorTy => {
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
                match (self.lhs, self.rhs) {
                    (Ty::Float32TensorTy, Ty::Float32TensorTy) => {
                        function_kernel!(Float32Tensor, Float32Tensor, $k)
                    }
                    (Ty::Float64TensorTy, Ty::Float64TensorTy) => {
                        function_kernel!(Float64Tensor, Float64Tensor, $k)
                    }
                    (Ty::Int32TensorTy, Ty::Int32TensorTy) => {
                        function_kernel!(Int32Tensor, Int32Tensor, $k)
                    }
                    (Ty::Int64TensorTy, Ty::Int64TensorTy) => {
                        function_kernel!(Int64Tensor, Int64Tensor, $k)
                    }
                    (Ty::Uint32TensorTy, Ty::Uint32TensorTy) => {
                        function_kernel!(Uint32Tensor, Uint32Tensor, $k)
                    }
                    (Ty::Uint64TensorTy, Ty::Uint64TensorTy) => {
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
        match self.ty {
            Ty::Float32TensorTy => {
                closure_kernel!(Float32Tensor, |x| x.inv())
            }
            Ty::Float64TensorTy => {
                closure_kernel!(Float64Tensor, |x| x.inv())
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdMeanOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|x| x as usize);
        match self.ty {
            Ty::Float32TensorTy => {
                closure_kernel!(Float32Tensor, |x| x.mean(axis))
            }
            Ty::Float64TensorTy => {
                closure_kernel!(Float64Tensor, |x| x.mean(axis))
            }
            Ty::Int32TensorTy => {
                closure_kernel!(Int32Tensor, |x| x.mean(axis))
            }
            Ty::Int64TensorTy => {
                closure_kernel!(Int64Tensor, |x| x.mean(axis))
            }
            Ty::Uint32TensorTy => {
                closure_kernel!(Uint32Tensor, |x| x.mean(axis))
            }
            Ty::Uint64TensorTy => {
                closure_kernel!(Uint64Tensor, |x| x.mean(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdOnesOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.ty {
            Ty::Float32TensorTy => {
                function_kernel!(Shape, |shape| Float32Tensor::ones(shape))
            }
            Ty::Float64TensorTy => {
                function_kernel!(Shape, |shape| Float64Tensor::ones(shape))
            }
            Ty::Int32TensorTy => {
                function_kernel!(Shape, |shape| Int32Tensor::ones(shape))
            }
            Ty::Int64TensorTy => {
                function_kernel!(Shape, |shape| Int64Tensor::ones(shape))
            }
            Ty::Uint32TensorTy => {
                function_kernel!(Shape, |shape| Uint32Tensor::ones(shape))
            }
            Ty::Uint64TensorTy => {
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
        match self.ty {
            Ty::Float32TensorTy => {
                closure_kernel!(vec[Float32Tensor], |xs| concatenate(axis, &xs))
            }
            Ty::Float64TensorTy => {
                closure_kernel!(vec[Float64Tensor], |xs| concatenate(axis, &xs))
            }
            Ty::Int32TensorTy => closure_kernel!(vec[Int32Tensor], |xs| concatenate(axis, &xs)),
            Ty::Int64TensorTy => closure_kernel!(vec[Int64Tensor], |xs| concatenate(axis, &xs)),
            Ty::Uint32TensorTy => {
                closure_kernel!(vec[Uint32Tensor], |xs| concatenate(axis, &xs))
            }
            Ty::Uint64TensorTy => {
                closure_kernel!(vec[Uint64Tensor], |xs| concatenate(axis, &xs))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdExpandDimsOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis as usize;
        match self.ty {
            Ty::Float32TensorTy => {
                closure_kernel!(Float32Tensor, |x| x.expand_dims(axis))
            }
            Ty::Float64TensorTy => {
                closure_kernel!(Float64Tensor, |x| x.expand_dims(axis))
            }
            Ty::Int32TensorTy => {
                closure_kernel!(Int32Tensor, |x| x.expand_dims(axis))
            }
            Ty::Int64TensorTy => {
                closure_kernel!(Int64Tensor, |x| x.expand_dims(axis))
            }
            Ty::Uint32TensorTy => {
                closure_kernel!(Uint32Tensor, |x| x.expand_dims(axis))
            }
            Ty::Uint64TensorTy => {
                closure_kernel!(Uint64Tensor, |x| x.expand_dims(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdReshapeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.ty {
            Ty::Float32TensorTy => {
                function_kernel!(Float32Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            Ty::Float64TensorTy => {
                function_kernel!(Float64Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            Ty::Int32TensorTy => {
                function_kernel!(Int32Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            Ty::Int64TensorTy => {
                function_kernel!(Int64Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            Ty::Uint32TensorTy => {
                function_kernel!(Uint32Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            Ty::Uint64TensorTy => {
                function_kernel!(Uint64Tensor, Shape, |x, newshape| x.reshape(newshape))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdAtLeast2DOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let tcv = self.to_column_vector;
        match self.ty {
            Ty::Float32TensorTy => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            Ty::Float64TensorTy => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            Ty::Int32TensorTy => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            Ty::Int64TensorTy => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            Ty::Uint32TensorTy => {
                closure_kernel!(Float64Tensor, |x| x.atleast_2d(tcv))
            }
            Ty::Uint64TensorTy => {
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
        match self.ty {
            Ty::ShapeTy => closure_kernel!(Shape, |x| x.slice(start, end)),
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for StdSumOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.ty {
            Ty::Float32TensorTy => {
                closure_kernel!(Float32Tensor, |x| x.sum(axis))
            }
            Ty::Float64TensorTy => {
                closure_kernel!(Float64Tensor, |x| x.sum(axis))
            }
            Ty::Int32TensorTy => {
                closure_kernel!(Int32Tensor, |x| x.sum(axis))
            }
            Ty::Int64TensorTy => {
                closure_kernel!(Int64Tensor, |x| x.sum(axis))
            }
            Ty::Uint32TensorTy => {
                closure_kernel!(Uint32Tensor, |x| x.sum(axis))
            }
            Ty::Uint64TensorTy => {
                closure_kernel!(Uint64Tensor, |x| x.sum(axis))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for PrimDeriveSeedOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let nonce = self.nonce.clone();
        closure_kernel!(PrfKey, |key| Seed::from_prf(&key, &nonce))
    }
}

impl Compile<Kernel> for PrimGenPrfKeyOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(|| PrfKey::generate(&HostPlacement {
            owner: "localhost".into()
        }))
    }
}

impl Compile<Kernel> for RingAddOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            Signature::Binary(BinarySignature {
                arg0: Ty::Ring64TensorTy,
                arg1: Ty::Ring64TensorTy,
                ..
            }) => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x + y)
            }
            Signature::Binary(BinarySignature {
                arg0: Ty::Ring128TensorTy,
                arg1: Ty::Ring128TensorTy,
                ..
            }) => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x + y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSubOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            Signature::Binary(BinarySignature {
                arg0: Ty::Ring64TensorTy,
                arg1: Ty::Ring64TensorTy,
                ..
            }) => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x - y)
            }
            Signature::Binary(BinarySignature {
                arg0: Ty::Ring128TensorTy,
                arg1: Ty::Ring128TensorTy,
                ..
            }) => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x - y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingMulOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.sig {
            Signature::Binary(BinarySignature {
                arg0: Ty::Ring64TensorTy,
                arg1: Ty::Ring64TensorTy,
                ..
            }) => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x * y)
            }
            Signature::Binary(BinarySignature {
                arg0: Ty::Ring128TensorTy,
                arg1: Ty::Ring128TensorTy,
                ..
            }) => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x * y)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingDotOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x.dot(y))
            }
            (Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x.dot(y))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSumOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.ty {
            Ty::Ring64TensorTy => closure_kernel!(Ring64Tensor, |x| x.sum(axis)),
            Ty::Ring128TensorTy => closure_kernel!(Ring128Tensor, |x| x.sum(axis)),
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingShapeOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match self.ty {
            Ty::Ring64TensorTy => function_kernel!(Ring64Tensor, |x| x.shape()),
            Ty::Ring128TensorTy => function_kernel!(Ring128Tensor, |x| x.shape()),
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingFillOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (self.ty, self.value.clone()) {
            (Ty::Ring64TensorTy, Value::Ring64(value)) => {
                closure_kernel!(Shape, |shape| Ring64Tensor::fill(&shape, value))
            }
            (Ty::Ring128TensorTy, Value::Ring128(value)) => {
                closure_kernel!(Shape, |shape| Ring128Tensor::fill(&shape, value))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingSampleOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        match (self.sig, self.max_value) {
            (
                Signature::Nullary(NullarySignature {
                    ret: Ty::Ring64TensorTy,
                }),
                None,
            ) => {
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_uniform(
                    &shape, &seed
                ))
            }
            (
                Signature::Nullary(NullarySignature {
                    ret: Ty::Ring64TensorTy,
                }),
                Some(max_value),
            ) if max_value == 1 => {
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_bits(
                    &shape, &seed
                ))
            }
            (
                Signature::Nullary(NullarySignature {
                    ret: Ty::Ring128TensorTy,
                }),
                None,
            ) => {
                function_kernel!(Shape, Seed, |shape, seed| Ring128Tensor::sample_uniform(
                    &shape, &seed
                ))
            }
            (
                Signature::Nullary(NullarySignature {
                    ret: Ty::Ring128TensorTy,
                }),
                Some(max_value),
            ) if max_value == 1 => {
                function_kernel!(Shape, Seed, |shape, seed| Ring128Tensor::sample_bits(
                    &shape, &seed
                ))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingShlOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let amount = self.amount;
        match self.ty {
            Ty::Ring64TensorTy => {
                closure_kernel!(Ring64Tensor, |x| x << amount)
            }
            Ty::Ring128TensorTy => {
                closure_kernel!(Ring128Tensor, |x| x << amount)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingShrOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let amount = self.amount;
        match self.ty {
            Ty::Ring64TensorTy => {
                closure_kernel!(Ring64Tensor, |x| x >> amount)
            }
            Ty::Ring128TensorTy => {
                closure_kernel!(Ring128Tensor, |x| x >> amount)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for RingInjectOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let bit_idx = self.bit_idx;
        match self.output {
            Ty::Ring64TensorTy => {
                closure_kernel!(BitTensor, |x| Ring64Tensor::from(x) << bit_idx)
            }
            Ty::Ring128TensorTy => {
                closure_kernel!(BitTensor, |x| Ring128Tensor::from(x) << bit_idx)
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for BitExtractOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let bit_idx = self.bit_idx;
        match self.ring_type {
            Ty::Ring64TensorTy => {
                closure_kernel!(Ring64Tensor, |x| x.bit_extract(bit_idx))
            }
            Ty::Ring128TensorTy => {
                closure_kernel!(Ring128Tensor, |x| x.bit_extract(bit_idx))
            }
            _ => Err(Error::UnimplementedOperator(format!("{:?}", self))),
        }
    }
}

impl Compile<Kernel> for BitSampleOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        function_kernel!(Shape, Seed, |shape, seed| BitTensor::sample_uniform(
            &shape, &seed
        ))
    }
}

impl Compile<Kernel> for BitFillOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<Kernel> {
        let value = self.value;
        closure_kernel!(Shape, |shape| BitTensor::fill(&shape, value))
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
        match self.ty {
            Ty::Ring64TensorTy => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(Float64Tensor, |x| Ring64Tensor::encode(&x, scaling_factor))
            }
            Ty::Ring128TensorTy => {
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
        match self.input_ty {
            Ty::Ring64TensorTy => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(Ring64Tensor, |x| Ring64Tensor::decode(&x, scaling_factor))
            }
            Ty::Ring128TensorTy => {
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
        match self.ty {
            Ty::Ring64TensorTy => {
                let scaling_factor = u64::pow(self.scaling_base, self.scaling_exp);
                closure_kernel!(Ring64Tensor, |x| Ring64Tensor::ring_mean(
                    x,
                    axis,
                    scaling_factor
                ))
            }
            Ty::Ring128TensorTy => {
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
        Ok(Kernel::NullaryClosure(Arc::new(move || Ok(value.clone()))))
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
        let expected_ty = self.ty;
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
        let expected_ty = self.ty;
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
        let expected_ty = self.ty;

        Ok(SyncKernel::Unary(Box::new(move |_sess, v| {
            check_type(&v, expected_ty)?;
            Ok(v)
        })))
    }
}

impl Compile<AsyncKernel> for IdentityOp {
    fn compile(&self, _ctx: &CompilationContext) -> Result<AsyncKernel> {
        let expected_ty = self.ty;

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
        let expected_ty = self.ty;

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
        let expected_ty = self.ty;
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
        let expected_ty = self.ty;

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
        let expected_ty = self.ty;

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
        let expected_ty = self.ty;

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
        let expected_ty = self.ty;

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
