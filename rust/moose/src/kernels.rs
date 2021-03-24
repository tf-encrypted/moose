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
            Constant(op) => op.compile(),
            StdAdd(op) => op.compile(),
            StdSub(op) => op.compile(),
            StdMul(op) => op.compile(),
            StdDiv(op) => op.compile(),
            StdReshape(op) => op.compile(),
            StdSum(op) => op.compile(),
            RingAdd(op) => op.compile(),
            RingSub(op) => op.compile(),
            RingMul(op) => op.compile(),
            RingDot(op) => op.compile(),
            RingSum(op) => op.compile(),
            RingShape(op) => op.compile(),
            RingSample(op) => op.compile(),
            RingFill(op) => op.compile(),
            RingShl(op) => op.compile(),
            RingShr(op) => op.compile(),
            PrimDeriveSeed(op) => op.compile(),
            PrimGenPrfKey(op) => op.compile(),
            Send(op) => op.compile(),
            Receive(op) => op.compile(),
            FixedpointRingEncode(op) => op.compile(),
            FixedpointRingDecode(op) => op.compile(),
            FixedpointRingMean(op) => op.compile(),
        }
    }
}

impl Compile<AsyncKernel> for Operator {
    fn compile(&self) -> Result<AsyncKernel> {
        use Operator::*;
        match self {
            Constant(op) => op.compile(),
            StdAdd(op) => op.compile(),
            StdSub(op) => op.compile(),
            StdMul(op) => op.compile(),
            StdDiv(op) => op.compile(),
            StdReshape(op) => op.compile(),
            StdSum(op) => op.compile(),
            RingAdd(op) => op.compile(),
            RingSub(op) => op.compile(),
            RingMul(op) => op.compile(),
            RingDot(op) => op.compile(),
            RingSum(op) => op.compile(),
            RingShape(op) => op.compile(),
            RingSample(op) => op.compile(),
            RingFill(op) => op.compile(),
            RingShl(op) => op.compile(),
            RingShr(op) => op.compile(),
            PrimDeriveSeed(op) => op.compile(),
            PrimGenPrfKey(op) => op.compile(),
            Send(op) => op.compile(),
            Receive(op) => op.compile(),
            FixedpointRingEncode(op) => op.compile(),
            FixedpointRingDecode(op) => op.compile(),
            FixedpointRingMean(op) => op.compile(),
        }
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
            Ty::Ring64TensorTy => function_kernel!(Ring64Tensor, |x| x.shape()),
            Ty::Ring128TensorTy => function_kernel!(Ring128Tensor, |x| x.shape()),
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for RingFillOp {
    fn compile(&self) -> Result<Kernel> {
        let value = self.value;
        closure_kernel!(Shape, |shape: Shape| Ring64Tensor::fill(shape, value))
    }
}

impl Compile<Kernel> for RingSampleOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.output, self.max_value) {
            (Ty::Ring64TensorTy, None) => {
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_uniform(
                    &shape.0, &seed.0
                ))
            }
            (Ty::Ring64TensorTy, Some(max_value)) if max_value == 1 => {
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

impl Compile<Kernel> for ConstantOp {
    fn compile(&self) -> Result<Kernel> {
        use std::sync::Arc;
        let value = self.value.clone();
        Ok(Kernel::NullaryClosure(Arc::new(move || Ok(value.clone()))))
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
