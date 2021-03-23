use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::{
    map_receive_error, map_send_error, AsyncKernel, Compile, Kernel, SyncKernel,
};
use crate::prim::{PrfKey, Seed};
use crate::ring::{Ring128Tensor, Ring64Tensor};
use crate::standard::{
    Float32Tensor, Float64Tensor, Int32Tensor, Int64Tensor, Shape, Uint32Tensor, Uint64Tensor,
};
use crate::{closure_kernel, function_kernel};

impl Compile<SyncKernel> for Operator {
    fn compile(&self) -> Result<SyncKernel> {
        use Operator::*;
        match self {
            Constant(op) => Compile::<SyncKernel>::compile(op),
            StdAdd(op) => Compile::<SyncKernel>::compile(op),
            StdSub(op) => Compile::<SyncKernel>::compile(op),
            StdMul(op) => Compile::<SyncKernel>::compile(op),
            StdDiv(op) => Compile::<SyncKernel>::compile(op),
            StdReshape(op) => Compile::<SyncKernel>::compile(op),
            StdSum(op) => Compile::<SyncKernel>::compile(op),
            RingAdd(op) => Compile::<SyncKernel>::compile(op),
            RingSub(op) => Compile::<SyncKernel>::compile(op),
            RingMul(op) => Compile::<SyncKernel>::compile(op),
            RingDot(op) => Compile::<SyncKernel>::compile(op),
            RingSum(op) => Compile::<SyncKernel>::compile(op),
            RingShape(op) => Compile::<SyncKernel>::compile(op),
            RingSample(op) => Compile::<SyncKernel>::compile(op),
            RingFill(op) => Compile::<SyncKernel>::compile(op),
            RingShl(op) => Compile::<SyncKernel>::compile(op),
            RingShr(op) => Compile::<SyncKernel>::compile(op),
            PrimDeriveSeed(op) => Compile::<SyncKernel>::compile(op),
            PrimGenPrfKey(op) => Compile::<SyncKernel>::compile(op),
            Send(op) => Compile::<SyncKernel>::compile(op),
            Receive(op) => Compile::<SyncKernel>::compile(op),
            FixedpointRingEncode(op) => Compile::<SyncKernel>::compile(op),
            FixedpointRingDecode(op) => Compile::<SyncKernel>::compile(op),
            FixedpointRingMean(op) => Compile::<SyncKernel>::compile(op),
        }
    }
}

impl Compile<AsyncKernel> for Operator {
    fn compile(&self) -> Result<AsyncKernel> {
        use Operator::*;
        match self {
            Constant(op) => Compile::<AsyncKernel>::compile(op),
            StdAdd(op) => Compile::<AsyncKernel>::compile(op),
            StdSub(op) => Compile::<AsyncKernel>::compile(op),
            StdMul(op) => Compile::<AsyncKernel>::compile(op),
            StdDiv(op) => Compile::<AsyncKernel>::compile(op),
            StdReshape(op) => Compile::<AsyncKernel>::compile(op),
            StdSum(op) => Compile::<AsyncKernel>::compile(op),
            RingAdd(op) => Compile::<AsyncKernel>::compile(op),
            RingSub(op) => Compile::<AsyncKernel>::compile(op),
            RingMul(op) => Compile::<AsyncKernel>::compile(op),
            RingDot(op) => Compile::<AsyncKernel>::compile(op),
            RingSum(op) => Compile::<AsyncKernel>::compile(op),
            RingShape(op) => Compile::<AsyncKernel>::compile(op),
            RingSample(op) => Compile::<AsyncKernel>::compile(op),
            RingFill(op) => Compile::<AsyncKernel>::compile(op),
            RingShl(op) => Compile::<AsyncKernel>::compile(op),
            RingShr(op) => Compile::<AsyncKernel>::compile(op),
            PrimDeriveSeed(op) => Compile::<AsyncKernel>::compile(op),
            PrimGenPrfKey(op) => Compile::<AsyncKernel>::compile(op),
            Send(op) => Compile::<AsyncKernel>::compile(op),
            Receive(op) => Compile::<AsyncKernel>::compile(op),
            FixedpointRingEncode(op) => Compile::<AsyncKernel>::compile(op),
            FixedpointRingDecode(op) => Compile::<AsyncKernel>::compile(op),
            FixedpointRingMean(op) => Compile::<AsyncKernel>::compile(op),
        }
    }
}

impl Compile<Kernel> for ConstantOp {
    fn compile(&self) -> Result<Kernel> {
        use std::sync::Arc;
        let value = self.value.clone();
        Ok(Kernel::NullaryClosure(Arc::new(move || Ok(value.clone()))))
    }
}

impl Compile<Kernel> for StdAddOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Float32TensorTy, Ty::Float32TensorTy) => {
                function_kernel!(Float32Tensor, Float32Tensor, |x, y| x + y)
            }
            (Ty::Float64TensorTy, Ty::Float64TensorTy) => {
                function_kernel!(Float64Tensor, Float64Tensor, |x, y| x + y)
            }
            (Ty::Int32TensorTy, Ty::Int32TensorTy) => {
                function_kernel!(Int32Tensor, Int32Tensor, |x, y| x + y)
            }
            (Ty::Int64TensorTy, Ty::Int64TensorTy) => {
                function_kernel!(Int64Tensor, Int64Tensor, |x, y| x + y)
            }
            (Ty::Uint32TensorTy, Ty::Uint32TensorTy) => {
                function_kernel!(Uint32Tensor, Uint32Tensor, |x, y| x + y)
            }
            (Ty::Uint64TensorTy, Ty::Uint64TensorTy) => {
                function_kernel!(Uint64Tensor, Uint64Tensor, |x, y| x + y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for StdSubOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Float32TensorTy, Ty::Float32TensorTy) => {
                function_kernel!(Float32Tensor, Float32Tensor, |x, y| x - y)
            }
            (Ty::Float64TensorTy, Ty::Float64TensorTy) => {
                function_kernel!(Float64Tensor, Float64Tensor, |x, y| x - y)
            }
            (Ty::Int32TensorTy, Ty::Int32TensorTy) => {
                function_kernel!(Int32Tensor, Int32Tensor, |x, y| x - y)
            }
            (Ty::Int64TensorTy, Ty::Int64TensorTy) => {
                function_kernel!(Int64Tensor, Int64Tensor, |x, y| x - y)
            }
            (Ty::Uint32TensorTy, Ty::Uint32TensorTy) => {
                function_kernel!(Uint32Tensor, Uint32Tensor, |x, y| x - y)
            }
            (Ty::Uint64TensorTy, Ty::Uint64TensorTy) => {
                function_kernel!(Uint64Tensor, Uint64Tensor, |x, y| x - y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for StdMulOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Float32TensorTy, Ty::Float32TensorTy) => {
                function_kernel!(Float32Tensor, Float32Tensor, |x, y| x * y)
            }
            (Ty::Float64TensorTy, Ty::Float64TensorTy) => {
                function_kernel!(Float64Tensor, Float64Tensor, |x, y| x * y)
            }
            (Ty::Int32TensorTy, Ty::Int32TensorTy) => {
                function_kernel!(Int32Tensor, Int32Tensor, |x, y| x * y)
            }
            (Ty::Int64TensorTy, Ty::Int64TensorTy) => {
                function_kernel!(Int64Tensor, Int64Tensor, |x, y| x * y)
            }
            (Ty::Uint32TensorTy, Ty::Uint32TensorTy) => {
                function_kernel!(Uint32Tensor, Uint32Tensor, |x, y| x * y)
            }
            (Ty::Uint64TensorTy, Ty::Uint64TensorTy) => {
                function_kernel!(Uint64Tensor, Uint64Tensor, |x, y| x * y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for StdDivOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Float32TensorTy, Ty::Float32TensorTy) => {
                function_kernel!(Float32Tensor, Float32Tensor, |x, y| x / y)
            }
            (Ty::Float64TensorTy, Ty::Float64TensorTy) => {
                function_kernel!(Float64Tensor, Float64Tensor, |x, y| x / y)
            }
            (Ty::Int32TensorTy, Ty::Int32TensorTy) => {
                function_kernel!(Int32Tensor, Int32Tensor, |x, y| x / y)
            }
            (Ty::Int64TensorTy, Ty::Int64TensorTy) => {
                function_kernel!(Int64Tensor, Int64Tensor, |x, y| x / y)
            }
            (Ty::Uint32TensorTy, Ty::Uint32TensorTy) => {
                function_kernel!(Uint32Tensor, Uint32Tensor, |x, y| x / y)
            }
            (Ty::Uint64TensorTy, Ty::Uint64TensorTy) => {
                function_kernel!(Uint64Tensor, Uint64Tensor, |x, y| x / y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for StdReshapeOp {
    fn compile(&self) -> Result<Kernel> {
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
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for StdSumOp {
    fn compile(&self) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.ty {
            Ty::Float32TensorTy => {
                closure_kernel!(Float32Tensor, |x: Float32Tensor| x.sum(axis))
            }
            Ty::Float64TensorTy => {
                closure_kernel!(Float64Tensor, |x: Float64Tensor| x.sum(axis))
            }
            Ty::Int32TensorTy => {
                closure_kernel!(Int32Tensor, |x: Int32Tensor| x.sum(axis))
            }
            Ty::Int64TensorTy => {
                closure_kernel!(Int64Tensor, |x: Int64Tensor| x.sum(axis))
            }
            Ty::Uint32TensorTy => {
                closure_kernel!(Uint32Tensor, |x: Uint32Tensor| x.sum(axis))
            }
            Ty::Uint64TensorTy => {
                closure_kernel!(Uint64Tensor, |x: Uint64Tensor| x.sum(axis))
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for PrimDeriveSeedOp {
    fn compile(&self) -> Result<Kernel> {
        let nonce = self.nonce.0.clone();
        closure_kernel!(PrfKey, |key: PrfKey| {
            // TODO(Morten) pass key as-is without unwrapping
            Seed(crate::utils::derive_seed(&key.0, &nonce).into())
        })
    }
}

impl Compile<Kernel> for PrimGenPrfKeyOp {
    fn compile(&self) -> Result<Kernel> {
        use crate::prng::AesRng;
        function_kernel!(|| {
            // TODO(Morten) we shouldn't have core logic directly in kernels
            let raw_key = AesRng::generate_random_key();
            Value::PrfKey(PrfKey(raw_key.into()))
        })
    }
}

impl Compile<Kernel> for RingAddOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x + y)
            }
            (Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x + y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for RingSubOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x - y)
            }
            (Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                function_kernel!(Ring128Tensor, Ring128Tensor, |x, y| x - y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for RingMulOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x * y)
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for RingDotOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.lhs, self.rhs) {
            (Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                use crate::ring::Dot;
                function_kernel!(Ring64Tensor, Ring64Tensor, |x, y| x.dot(y))
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for RingSumOp {
    fn compile(&self) -> Result<Kernel> {
        let axis = self.axis.map(|a| a as usize);
        match self.ty {
            Ty::Ring64TensorTy => closure_kernel!(Ring64Tensor, |x: Ring64Tensor| x.sum(axis)),
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for RingShapeOp {
    fn compile(&self) -> Result<Kernel> {
        match self.ty {
            Ty::Ring64TensorTy => function_kernel!(Ring64Tensor, |x| {
                Shape(x.0.shape().into()) // TODO(Morten) wrapping should not happen here
            }),
            Ty::Ring128TensorTy => function_kernel!(Ring128Tensor, |x| {
                Shape(x.0.shape().into()) // TODO(Morten) wrapping should not happen here
            }),
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for RingFillOp {
    fn compile(&self) -> Result<Kernel> {
        let value = self.value;
        // TODO(Morten) should not call .0 here
        closure_kernel!(Shape, |shape: Shape| Ring64Tensor::fill(&shape.0, value))
    }
}

impl Compile<Kernel> for RingSampleOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.output, self.max_value) {
            (Ty::Ring64TensorTy, None) => {
                use crate::ring::Sample;
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_uniform(
                    &shape.0, &seed.0
                ))
            }
            (Ty::Ring64TensorTy, Some(max_value)) if max_value == 1 => {
                use crate::ring::Sample;
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_bits(
                    &shape.0, &seed.0
                ))
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for RingShlOp {
    fn compile(&self) -> Result<Kernel> {
        let amount = self.amount;
        closure_kernel!(Ring64Tensor, |x| x << amount)
    }
}

impl Compile<Kernel> for RingShrOp {
    fn compile(&self) -> Result<Kernel> {
        let amount = self.amount;
        closure_kernel!(Ring64Tensor, |x| x >> amount)
    }
}

impl Compile<Kernel> for FixedpointRingEncodeOp {
    fn compile(&self) -> Result<Kernel> {
        use crate::fixedpoint::Convert;
        let scaling_factor = self.scaling_factor;
        closure_kernel!(Float64Tensor, |x| Ring64Tensor::encode(&x, scaling_factor))
    }
}

impl Compile<Kernel> for FixedpointRingDecodeOp {
    fn compile(&self) -> Result<Kernel> {
        use crate::fixedpoint::Convert;
        let scaling_factor = self.scaling_factor;
        closure_kernel!(Ring64Tensor, |x| Ring64Tensor::decode(&x, scaling_factor))
    }
}

impl Compile<Kernel> for FixedpointRingMeanOp {
    fn compile(&self) -> Result<Kernel> {
        let axis = self.axis;
        let scaling_factor = self.scaling_factor;
        closure_kernel!(Ring64Tensor, |x| Ring64Tensor::ring_mean(
            x,
            axis,
            scaling_factor
        ))
    }
}

impl Compile<SyncKernel> for SendOp {
    fn compile(&self) -> Result<SyncKernel> {
        let rdv = self.rendezvous_key.clone();
        Ok(SyncKernel::Unary(Box::new(move |ctx, sid, v| {
            ctx.networking.send(&v, &rdv, &sid);
            Ok(Value::Unit)
        })))
    }
}

impl Compile<AsyncKernel> for SendOp {
    fn compile(&self) -> Result<AsyncKernel> {
        use std::sync::Arc;
        let rdv = Arc::new(self.rendezvous_key.clone());
        Ok(AsyncKernel::Unary(Box::new(move |ctx, sid, v, sender| {
            let ctx = Arc::clone(ctx);
            let sid = Arc::clone(sid);
            let rdv = Arc::clone(&rdv);
            tokio::spawn(async move {
                let v: Value = v.await.map_err(map_receive_error)?;
                ctx.networking.send(&v, &rdv, &sid).await;
                sender.send(Value::Unit).map_err(map_send_error)
            })
        })))
    }
}

impl Compile<SyncKernel> for ReceiveOp {
    fn compile(&self) -> Result<SyncKernel> {
        let rdv = self.rendezvous_key.clone();
        Ok(SyncKernel::Nullary(Box::new(move |ctx, sid| {
            ctx.networking.receive(&rdv, sid)
        })))
    }
}

impl Compile<AsyncKernel> for ReceiveOp {
    fn compile(&self) -> Result<AsyncKernel> {
        use std::sync::Arc;
        let rdv = Arc::new(self.rendezvous_key.clone());
        Ok(AsyncKernel::Nullary(Box::new(move |ctx, sid, sender| {
            let ctx = Arc::clone(ctx);
            let sid = Arc::clone(sid);
            let rdv = Arc::clone(&rdv);
            tokio::spawn(async move {
                let v: Value = ctx.networking.receive(&rdv, &sid).await?;
                sender.send(v).map_err(map_send_error)
            })
        })))
    }
}
