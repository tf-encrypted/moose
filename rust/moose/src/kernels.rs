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
            Identity(op) => op.compile(),
            Send(op) => op.compile(),
            Receive(op) => op.compile(),
            Input(op) => op.compile(),
            Output(op) => op.compile(),
            Constant(op) => op.compile(),
            StdAdd(op) => op.compile(),
            StdSub(op) => op.compile(),
            StdMul(op) => op.compile(),
            StdDiv(op) => op.compile(),
            StdDot(op) => op.compile(),
            StdMean(op) => op.compile(),
            StdOnes(op) => op.compile(),
            StdConcatenate(op) => op.compile(),
            StdExpandDims(op) => op.compile(),
            StdReshape(op) => op.compile(),
            StdShape(op) => op.compile(),
            StdSum(op) => op.compile(),
            StdTranspose(op) => op.compile(),
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
            Identity(op) => op.compile(),
            Send(op) => op.compile(),
            Receive(op) => op.compile(),
            Input(op) => op.compile(),
            Output(op) => op.compile(),
            Constant(op) => op.compile(),
            StdAdd(op) => op.compile(),
            StdSub(op) => op.compile(),
            StdMul(op) => op.compile(),
            StdDiv(op) => op.compile(),
            StdDot(op) => op.compile(),
            StdMean(op) => op.compile(),
            StdOnes(op) => op.compile(),
            StdConcatenate(op) => op.compile(),
            StdExpandDims(op) => op.compile(),
            StdReshape(op) => op.compile(),
            StdShape(op) => op.compile(),
            StdSum(op) => op.compile(),
            StdTranspose(op) => op.compile(),
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
            FixedpointRingEncode(op) => op.compile(),
            FixedpointRingDecode(op) => op.compile(),
            FixedpointRingMean(op) => op.compile(),
        }
    }
}

macro_rules! std_binary_kernel {
    ($op:ty, $k:expr) => {
        impl Compile<Kernel> for $op {
            fn compile(&self) -> Result<Kernel> {
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
                    _ => Err(Error::UnimplementedOperator),
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

impl Compile<Kernel> for StdMeanOp {
    fn compile(&self) -> Result<Kernel> {
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
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for StdOnesOp {
    fn compile(&self) -> Result<Kernel> {
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

            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for StdConcatenateOp {
    fn compile(&self) -> Result<Kernel> {
        let axis = self.axis as usize;
        match self.ty {
            Ty::Float32TensorTy => {
                closure_kernel!(Vec::<Float32Tensor>, |xs| standard::concatenate(axis, &xs[..]))
            },
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for StdExpandDimsOp {
    fn compile(&self) -> Result<Kernel> {
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

impl Compile<Kernel> for StdShapeOp {
    fn compile(&self) -> Result<Kernel> {
        match self.ty {
            Ty::Float32TensorTy => {
                function_kernel!(Float32Tensor, |x| x.shape())
            }
            Ty::Float64TensorTy => {
                function_kernel!(Float64Tensor, |x| x.shape())
            }
            Ty::Int32TensorTy => {
                function_kernel!(Int32Tensor, |x| x.shape())
            }
            Ty::Int64TensorTy => {
                function_kernel!(Int64Tensor, |x| x.shape())
            }
            Ty::Uint32TensorTy => {
                function_kernel!(Uint32Tensor, |x| x.shape())
            }
            Ty::Uint64TensorTy => {
                function_kernel!(Uint64Tensor, |x| x.shape())
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
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for StdTransposeOp {
    fn compile(&self) -> Result<Kernel> {
        match self.ty {
            Ty::Float32TensorTy => {
                function_kernel!(Float32Tensor, |x| x.transpose())
            }
            Ty::Float64TensorTy => {
                function_kernel!(Float64Tensor, |x| x.transpose())
            }
            Ty::Int32TensorTy => {
                function_kernel!(Int32Tensor, |x| x.transpose())
            }
            Ty::Int64TensorTy => {
                function_kernel!(Int64Tensor, |x| x.transpose())
            }
            Ty::Uint32TensorTy => {
                function_kernel!(Uint32Tensor, |x| x.transpose())
            }
            Ty::Uint64TensorTy => {
                function_kernel!(Uint64Tensor, |x| x.transpose())
            }
            _ => Err(Error::UnimplementedOperator),
        }
    }
}

impl Compile<Kernel> for PrimDeriveSeedOp {
    fn compile(&self) -> Result<Kernel> {
        let nonce = self.nonce.clone();
        closure_kernel!(PrfKey, |key| Seed::from_prf(&key, &nonce))
    }
}

impl Compile<Kernel> for PrimGenPrfKeyOp {
    fn compile(&self) -> Result<Kernel> {
        function_kernel!(PrfKey::generate)
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
            Ty::Ring64TensorTy => closure_kernel!(Ring64Tensor, |x| x.sum(axis)),
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
        closure_kernel!(Shape, |shape| Ring64Tensor::fill(&shape, value))
    }
}

impl Compile<Kernel> for RingSampleOp {
    fn compile(&self) -> Result<Kernel> {
        match (self.output, self.max_value) {
            (Ty::Ring64TensorTy, None) => {
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_uniform(
                    &shape, &seed
                ))
            }
            (Ty::Ring64TensorTy, Some(max_value)) if max_value == 1 => {
                function_kernel!(Shape, Seed, |shape, seed| Ring64Tensor::sample_bits(
                    &shape, &seed
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
        Ok(SyncKernel::Unary(Box::new(move |sess, v| {
            sess.networking.send(&v, &rdv, &sess.sid)?;
            Ok(Value::Unit)
        })))
    }
}

impl Compile<AsyncKernel> for SendOp {
    fn compile(&self) -> Result<AsyncKernel> {
        use std::sync::Arc;
        let rdv = Arc::new(self.rendezvous_key.clone());
        Ok(AsyncKernel::Unary(Box::new(move |sess, v, sender| {
            let sess = Arc::clone(sess);
            let rdv = Arc::clone(&rdv);
            tokio::spawn(async move {
                let v: Value = v.await.map_err(map_receive_error)?;
                sess.networking.send(&v, &rdv, &sess.sid).await?;
                sender.send(Value::Unit).map_err(map_send_error)
            })
        })))
    }
}

impl Compile<SyncKernel> for ReceiveOp {
    fn compile(&self) -> Result<SyncKernel> {
        let rdv = self.rendezvous_key.clone();
        let expected_ty = self.ty;
        Ok(SyncKernel::Nullary(Box::new(move |sess| {
            let val = sess.networking.receive(&rdv, &sess.sid)?;
            if val.ty() == expected_ty {
                Ok(val)
            } else {
                Err(Error::TypeMismatch)
            }
        })))
    }
}

impl Compile<AsyncKernel> for ReceiveOp {
    fn compile(&self) -> Result<AsyncKernel> {
        use std::sync::Arc;
        let rdv = Arc::new(self.rendezvous_key.clone());
        let expected_ty = self.ty;
        Ok(AsyncKernel::Nullary(Box::new(move |sess, sender| {
            let sess = Arc::clone(sess);
            let rdv = Arc::clone(&rdv);
            tokio::spawn(async move {
                let val: Value = sess.networking.receive(&rdv, &sess.sid).await?;
                if val.ty() == expected_ty {
                    sender.send(val).map_err(map_send_error)
                } else {
                    Err(Error::TypeMismatch)
                }
            })
        })))
    }
}

impl Compile<SyncKernel> for IdentityOp {
    fn compile(&self) -> Result<SyncKernel> {
        let expected_ty = self.ty;
        Ok(SyncKernel::Unary(Box::new(move |_sess, val| {
            if val.ty() == expected_ty {
                Ok(val)
            } else {
                Err(Error::TypeMismatch)
            }
        })))
    }
}

impl Compile<AsyncKernel> for IdentityOp {
    fn compile(&self) -> Result<AsyncKernel> {
        let expected_ty = self.ty;
        Ok(AsyncKernel::Unary(Box::new(move |_sess, val, sender| {
            tokio::spawn(async move {
                let val: Value = val.await.map_err(map_receive_error)?;
                if val.ty() == expected_ty {
                    sender.send(val).map_err(map_send_error)
                } else {
                    Err(Error::TypeMismatch)
                }
            })
        })))
    }
}

impl Compile<SyncKernel> for InputOp {
    fn compile(&self) -> Result<SyncKernel> {
        let expected_ty = self.ty;
        let arg_name = self.arg_name.clone();
        Ok(SyncKernel::Nullary(Box::new(move |sess| {
            let arg = sess
                .args
                .get(&arg_name)
                .cloned()
                .ok_or(Error::MalformedEnvironment)?;
            if arg.ty() == expected_ty {
                Ok(arg)
            } else {
                Err(Error::TypeMismatch)
            }
        })))
    }
}

impl Compile<AsyncKernel> for InputOp {
    fn compile(&self) -> Result<AsyncKernel> {
        use std::sync::Arc;
        let expected_ty = self.ty;
        let arg_name = Arc::new(self.arg_name.clone());
        Ok(AsyncKernel::Nullary(Box::new(move |sess, sender| {
            let sess = Arc::clone(sess);
            let arg_name = Arc::clone(&arg_name);
            tokio::spawn(async move {
                let async_arg = sess
                    .args
                    .get(arg_name.as_ref())
                    .cloned()
                    .ok_or(Error::MalformedEnvironment)?;
                let arg: Value = async_arg.await.map_err(map_receive_error)?;
                if arg.ty() == expected_ty {
                    sender.send(arg).map_err(map_send_error)
                } else {
                    Err(Error::TypeMismatch)
                }
            })
        })))
    }
}

impl Compile<SyncKernel> for OutputOp {
    fn compile(&self) -> Result<SyncKernel> {
        Ok(SyncKernel::Unary(Box::new(move |_sess, x0| Ok(x0))))
    }
}

impl Compile<AsyncKernel> for OutputOp {
    fn compile(&self) -> Result<AsyncKernel> {
        Ok(AsyncKernel::Unary(Box::new(move |_sess, x0, sender| {
            tokio::spawn(async move {
                let val = x0.await.map_err(map_receive_error)?;
                sender.send(val).map_err(map_send_error)
            })
        })))
    }
}

#[test]
fn test_standard_shape_ops() {
    use crate::execution::EagerExecutor;
    use crate::standard::Float32Tensor;
    use maplit::hashmap;
    use ndarray::prelude::*;

    let env = hashmap![];
    let x = Float32Tensor::from(
        array![[1.0, 2.0], [3.0, 4.0]]
            .into_dimensionality::<IxDyn>()
            .unwrap(),
    );
    let x_op = Operation {
        name: "x".into(),
        kind: Operator::Constant(ConstantOp {
            value: Value::Float32Tensor(x),
        }),
        inputs: vec![],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };
    let shape_op = Operation {
        name: "shape".into(),
        kind: Operator::StdShape(StdShapeOp {
            ty: Ty::Float32TensorTy,
        }),
        inputs: vec!["x".into()],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };
    let expand_dims_op = Operation {
        name: "expand_dims".into(),
        kind: Operator::StdExpandDims(StdExpandDimsOp {
            ty: Ty::Float32TensorTy,
            axis: 2,
        }),
        inputs: vec!["x".into()],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };
    let transpose_op = Operation {
        name: "transpose".into(),
        kind: Operator::StdTranspose(StdTransposeOp {
            ty: Ty::Float32TensorTy,
        }),
        inputs: vec!["x".into()],
        placement: Placement::Host(HostPlacement {
            name: "alice".into(),
        }),
    };
    let operations = vec![x_op, shape_op, expand_dims_op, transpose_op];
    let comp = Computation { operations }.toposort().unwrap();

    let exec = EagerExecutor::new();
    exec.run_computation(&comp, 12345, env).ok();
}
