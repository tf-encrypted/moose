use super::*;
use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::kernels::*;
use crate::mirrored::Mirrored3Placement;
use crate::replicated::ReplicatedPlacement;

impl IdentityOp {
    pub(crate) fn logical_host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementIdentity<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementIdentity<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementIdentity<S, Float32T, Float32T>,
        HostPlacement: PlacementIdentity<S, Float64T, Float64T>,
        HostPlacement: PlacementIdentity<S, BoolT, BoolT>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.identity(sess, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.identity(sess, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            AbstractTensor::Float32(x) => {
                let result = plc.identity(sess, &x);
                Ok(AbstractTensor::Float32(result))
            }
            AbstractTensor::Float64(x) => {
                let result = plc.identity(sess, &x);
                Ok(AbstractTensor::Float64(result))
            }
            AbstractTensor::Bool(x) => {
                let result = plc.identity(sess, &x);
                Ok(AbstractTensor::Bool(result))
            }
        }
    }

    pub(crate) fn logical_rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementIdentity<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementIdentity<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = rep.identity(sess, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = rep.identity(sess, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            x => Err(Error::UnimplementedOperator(format!(
                "Missing rep identity op for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}

impl AddOp {
    pub(crate) fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementAdd<S, Fixed64T, Fixed64T, Fixed64T>,
        HostPlacement: PlacementAdd<S, Fixed128T, Fixed128T, Fixed128T>,
        HostPlacement: PlacementAdd<S, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementAdd<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.add(sess, &x, &y);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.add(sess, &x, &y);
                Ok(AbstractTensor::Fixed128(result))
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.add(sess, &x, &y);
                Ok(AbstractTensor::Float32(result))
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.add(sess, &x, &y);
                Ok(AbstractTensor::Float64(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing host add op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }

    pub(crate) fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementAdd<S, Fixed64T, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementAdd<S, Fixed128T, Fixed128T, Fixed128T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.add(sess, &x, &y);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.add(sess, &x, &y);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated add op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }
}

impl AddNOp {
    pub(crate) fn host_logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        xs: &[AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>],
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementAddN<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementAddN<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementAddN<S, Float32T, Float32T>,
        HostPlacement: PlacementAddN<S, Float64T, Float64T>,
        Fixed64T: Clone,
        Fixed128T: Clone,
        Float32T: Clone,
        Float64T: Clone,
    {
        if xs.is_empty() {
            Err(Error::InvalidArgument(
                "cannot add_n on empty array of tensors".to_string(),
            ))
        } else {
            let x = &xs[0];
            match x {
                AbstractTensor::Fixed64(_) => {
                    let vec: Vec<Fixed64T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            AbstractTensor::Fixed64(x) => x.clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(AbstractTensor::Fixed64(result))
                }
                AbstractTensor::Fixed128(_) => {
                    let vec: Vec<Fixed128T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            AbstractTensor::Fixed128(x) => x.clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(AbstractTensor::Fixed128(result))
                }
                AbstractTensor::Float32(_) => {
                    let vec: Vec<Float32T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            AbstractTensor::Float32(x) => x.clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(AbstractTensor::Float32(result))
                }
                AbstractTensor::Float64(_) => {
                    let vec: Vec<Float64T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            AbstractTensor::Float64(x) => x.clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(AbstractTensor::Float64(result))
                }
                x => Err(Error::UnimplementedOperator(format!(
                    "Missing host add_n op for {:?}",
                    &x.ty_desc(),
                ))),
            }
        }
    }

    pub(crate) fn logical_rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        xs: &[AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>],
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementAddN<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementAddN<S, Fixed128T, Fixed128T>,
        Fixed64T: Clone,
        Fixed128T: Clone,
    {
        if xs.is_empty() {
            Err(Error::InvalidArgument(
                "cannot add_n on empty array of tensors".to_string(),
            ))
        } else {
            let x = &xs[0];
            match x {
                AbstractTensor::Fixed64(_) => {
                    let vec: Vec<Fixed64T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            AbstractTensor::Fixed64(x) => (*x).clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(AbstractTensor::Fixed64(result))
                }
                AbstractTensor::Fixed128(_) => {
                    let vec: Vec<Fixed128T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            AbstractTensor::Fixed128(x) => (*x).clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(AbstractTensor::Fixed128(result))
                }
                x => Err(Error::UnimplementedOperator(format!(
                    "Missing replicated add_n op for {:?}",
                    &x.ty_desc(),
                ))),
            }
        }
    }
}

impl SubOp {
    pub(crate) fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementSub<S, Fixed64T, Fixed64T, Fixed64T>,
        HostPlacement: PlacementSub<S, Fixed128T, Fixed128T, Fixed128T>,
        HostPlacement: PlacementSub<S, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementSub<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.sub(sess, &x, &y);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.sub(sess, &x, &y);
                Ok(AbstractTensor::Fixed128(result))
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.sub(sess, &x, &y);
                Ok(AbstractTensor::Float32(result))
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.sub(sess, &x, &y);
                Ok(AbstractTensor::Float64(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing host sub op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }

    pub(crate) fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementSub<S, Fixed64T, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementSub<S, Fixed128T, Fixed128T, Fixed128T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.sub(sess, &x, &y);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.sub(sess, &x, &y);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated sub op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }
}

impl MulOp {
    pub(crate) fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementMul<S, Fixed64T, Fixed64T, Fixed64T>,
        HostPlacement: PlacementMul<S, Fixed128T, Fixed128T, Fixed128T>,
        HostPlacement: PlacementTruncPr<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementTruncPr<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementMul<S, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementMul<S, Float64T, Float64T, Float64T>,
    {
        // TODO(Morten)
        // we should probably use a trait bound on Fixed64T
        // and Fixed128T to extract precision instead
        let precision = match sig.arg(0) {
            Ok(Ty::Tensor(TensorDType::Fixed64 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            Ok(Ty::Tensor(TensorDType::Fixed128 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            _ => None,
        };
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.mul(sess, &x, &y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.mul(sess, &x, &y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed128(result))
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.mul(sess, &x, &y);
                Ok(AbstractTensor::Float32(result))
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.mul(sess, &x, &y);
                Ok(AbstractTensor::Float64(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing host mul op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }

    pub(crate) fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementMul<S, Fixed64T, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementMul<S, Fixed128T, Fixed128T, Fixed128T>,
        ReplicatedPlacement: PlacementTruncPr<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementTruncPr<S, Fixed128T, Fixed128T>,
    {
        let precision = match sig.arg(0) {
            Ok(Ty::Tensor(TensorDType::Fixed64 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            Ok(Ty::Tensor(TensorDType::Fixed128 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            _ => None,
        };

        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.mul(sess, &x, &y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.mul(sess, &x, &y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated mul op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }
}

impl DivOp {
    pub(crate) fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementDiv<S, Fixed64T, Fixed64T, Fixed64T>,
        HostPlacement: PlacementDiv<S, Fixed128T, Fixed128T, Fixed128T>,
        HostPlacement: PlacementDiv<S, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementDiv<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.div(sess, &x, &y);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.div(sess, &x, &y);
                Ok(AbstractTensor::Fixed128(result))
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.div(sess, &x, &y);
                Ok(AbstractTensor::Float32(result))
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.div(sess, &x, &y);
                Ok(AbstractTensor::Float64(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing host div op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }

    pub(crate) fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementDiv<S, Fixed64T, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementDiv<S, Fixed128T, Fixed128T, Fixed128T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.div(sess, &x, &y);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.div(sess, &x, &y);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated div for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }
}

impl DotOp {
    pub(crate) fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementDot<S, Fixed64T, Fixed64T, Fixed64T>,
        HostPlacement: PlacementDot<S, Fixed128T, Fixed128T, Fixed128T>,
        HostPlacement: PlacementTruncPr<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementTruncPr<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementDot<S, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementDot<S, Float64T, Float64T, Float64T>,
    {
        // TODO(Morten) same, use trait bound to extract
        let precision = match sig.arg(0) {
            Ok(Ty::Tensor(TensorDType::Fixed64 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            Ok(Ty::Tensor(TensorDType::Fixed128 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            _ => None,
        };
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.dot(sess, &x, &y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.dot(sess, &x, &y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed128(result))
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.dot(sess, &x, &y);
                Ok(AbstractTensor::Float32(result))
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.dot(sess, &x, &y);
                Ok(AbstractTensor::Float64(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing host dot op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }

    pub(crate) fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementDot<S, Fixed64T, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementDot<S, Fixed128T, Fixed128T, Fixed128T>,
        ReplicatedPlacement: PlacementTruncPr<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementTruncPr<S, Fixed128T, Fixed128T>,
    {
        let precision = match sig.arg(0) {
            Ok(Ty::Tensor(TensorDType::Fixed64 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            Ok(Ty::Tensor(TensorDType::Fixed128 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            _ => None,
        };
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.dot(sess, &x, &y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.dot(sess, &x, &y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated dot op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }
}

impl LessOp {
    pub(crate) fn logical_host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementLessThan<S, Fixed64T, Fixed64T, BoolT>,
        HostPlacement: PlacementLessThan<S, Fixed128T, Fixed128T, BoolT>,
        HostPlacement: PlacementLessThan<S, Float32T, Float32T, BoolT>,
        HostPlacement: PlacementLessThan<S, Float64T, Float64T, BoolT>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.less(sess, &x, &y);
                Ok(AbstractTensor::Bool(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.less(sess, &x, &y);
                Ok(AbstractTensor::Bool(result))
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.less(sess, &x, &y);
                Ok(AbstractTensor::Bool(result))
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.less(sess, &x, &y);
                Ok(AbstractTensor::Bool(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing host less op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }

    pub(crate) fn logical_rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementLessThan<S, Fixed64T, Fixed64T, BoolT>,
        ReplicatedPlacement: PlacementLessThan<S, Fixed128T, Fixed128T, BoolT>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.less(sess, &x, &y);
                Ok(AbstractTensor::Bool(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.less(sess, &x, &y);
                Ok(AbstractTensor::Bool(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing host less op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }
}

impl MuxOp {
    pub(crate) fn logical_rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        s: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementMux<S, BoolT, Fixed64T, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementMux<S, BoolT, Fixed128T, Fixed128T, Fixed128T>,
    {
        match (s, x, y) {
            (AbstractTensor::Bool(s), AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.mux(sess, &s, &x, &y);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Bool(s), AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.mux(sess, &s, &x, &y);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (s, x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated mux op for {:?}, {:?} and {:?}",
                &s.ty_desc(),
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }
}

impl CastOp {
    pub(crate) fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementFixedpointDecode<S, Fixed64T, Float32T>,
        HostPlacement: PlacementFixedpointDecode<S, Fixed128T, Float64T>,
        HostPlacement: PlacementFixedpointEncode<S, Float32T, Fixed64T>,
        HostPlacement: PlacementFixedpointEncode<S, Float64T, Fixed128T>,
    {
        let arg0_precision = match sig.arg(0) {
            Ok(Ty::Tensor(TensorDType::Fixed64 {
                integral_precision,
                fractional_precision,
            })) => Some((integral_precision, fractional_precision)),
            Ok(Ty::Tensor(TensorDType::Fixed128 {
                integral_precision,
                fractional_precision,
            })) => Some((integral_precision, fractional_precision)),
            _ => None,
        };

        match (x, sig.ret()) {
            (AbstractTensor::Fixed64(x), Ty::Tensor(TensorDType::Float32)) => {
                let (_, fractional_precision) = arg0_precision.unwrap();
                let inner = plc.fixedpoint_decode(sess, fractional_precision, &x);
                Ok(AbstractTensor::Float32(inner))
            }
            (AbstractTensor::Fixed128(x), Ty::Tensor(TensorDType::Float64)) => {
                let (_, fractional_precision) = arg0_precision.unwrap();
                let inner = plc.fixedpoint_decode(sess, fractional_precision, &x);
                Ok(AbstractTensor::Float64(inner))
            }
            (
                AbstractTensor::Float32(x),
                Ty::Tensor(TensorDType::Fixed64 {
                    fractional_precision,
                    integral_precision,
                }),
            ) => {
                let inner =
                    plc.fixedpoint_encode(sess, fractional_precision, integral_precision, &x);
                Ok(AbstractTensor::Fixed64(inner))
            }
            (
                AbstractTensor::Float64(x),
                Ty::Tensor(TensorDType::Fixed128 {
                    fractional_precision,
                    integral_precision,
                }),
            ) => {
                let inner =
                    plc.fixedpoint_encode(sess, fractional_precision, integral_precision, &x);
                Ok(AbstractTensor::Fixed128(inner))
            }
            (x, ret) => Err(Error::UnimplementedOperator(format!(
                "Cast operator does not support casting of {:?} to {:?}",
                &x.ty_desc(),
                &ret
            ))),
        }
    }

    pub(crate) fn mir_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &Mirrored3Placement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        Mirrored3Placement: PlacementFixedpointDecode<S, Fixed64T, Float32T>,
        Mirrored3Placement: PlacementFixedpointDecode<S, Fixed128T, Float64T>,
        Mirrored3Placement: PlacementFixedpointEncode<S, Float32T, Fixed64T>,
        Mirrored3Placement: PlacementFixedpointEncode<S, Float64T, Fixed128T>,
    {
        let arg0_precision = match sig.arg(0) {
            Ok(Ty::Tensor(TensorDType::Fixed64 {
                integral_precision,
                fractional_precision,
            })) => Some((integral_precision, fractional_precision)),
            Ok(Ty::Tensor(TensorDType::Fixed128 {
                integral_precision,
                fractional_precision,
            })) => Some((integral_precision, fractional_precision)),
            _ => None,
        };

        match (x, sig.ret()) {
            (AbstractTensor::Fixed64(x), Ty::Tensor(TensorDType::Float32)) => {
                let (_, fractional_precision) = arg0_precision.unwrap();
                let inner = plc.fixedpoint_decode(sess, fractional_precision, &x);
                Ok(AbstractTensor::Float32(inner))
            }
            (AbstractTensor::Fixed128(x), Ty::Tensor(TensorDType::Float64)) => {
                let (_, fractional_precision) = arg0_precision.unwrap();
                let inner = plc.fixedpoint_decode(sess, fractional_precision, &x);
                Ok(AbstractTensor::Float64(inner))
            }
            (
                AbstractTensor::Float32(x),
                Ty::Tensor(TensorDType::Fixed64 {
                    fractional_precision,
                    integral_precision,
                }),
            ) => {
                let inner =
                    plc.fixedpoint_encode(sess, fractional_precision, integral_precision, &x);
                Ok(AbstractTensor::Fixed64(inner))
            }
            (
                AbstractTensor::Float64(x),
                Ty::Tensor(TensorDType::Fixed128 {
                    fractional_precision,
                    integral_precision,
                }),
            ) => {
                let inner =
                    plc.fixedpoint_encode(sess, fractional_precision, integral_precision, &x);
                Ok(AbstractTensor::Fixed128(inner))
            }
            (x, ret) => Err(Error::UnimplementedOperator(format!(
                "Cast operator does not support casting of {:?} to {:?}",
                &x.ty_desc(),
                &ret
            ))),
        }
    }
}

impl AtLeast2DOp {
    pub(crate) fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        to_column_vector: bool,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        // HostPlacement: PlacementAtLeast2D<S, Fixed64T, Fixed64T>,
        // HostPlacement: PlacementAtLeast2D<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementAtLeast2D<S, Float32T, Float32T>,
        HostPlacement: PlacementAtLeast2D<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Fixed64(_x) => {
                unimplemented!()
                // let z = plc.at_least_2d(sess, to_column_vector, &x);
                // Ok(AbstractTensor::Fixed64(z))
            }
            AbstractTensor::Fixed128(_x) => {
                unimplemented!()
                // let z = plc.at_least_2d(sess, to_column_vector, &x);
                // AbstractTensor::Fixed128(z)
            }
            AbstractTensor::Float32(x) => {
                let z = plc.at_least_2d(sess, to_column_vector, &x);
                Ok(AbstractTensor::Float32(z))
            }
            AbstractTensor::Float64(x) => {
                let z = plc.at_least_2d(sess, to_column_vector, &x);
                Ok(AbstractTensor::Float64(z))
            }
            AbstractTensor::Bool(_) => {
                unimplemented!()
            }
        }
    }
}

impl MeanOp {
    pub(crate) fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        axis: Option<u32>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementMean<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementMean<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementMean<S, Float32T, Float32T>,
        HostPlacement: PlacementMean<S, Float64T, Float64T>,
        HostPlacement: PlacementTruncPr<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementTruncPr<S, Fixed128T, Fixed128T>,
    {
        let precision = match sig.arg(0) {
            Ok(Ty::Tensor(TensorDType::Fixed64 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            Ok(Ty::Tensor(TensorDType::Fixed128 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            _ => None,
        };
        match x {
            AbstractTensor::Fixed64(x) => {
                let z = plc.mean(sess, axis, &x);
                let z = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed64(z))
            }
            AbstractTensor::Fixed128(x) => {
                let z = plc.mean(sess, axis, &x);
                let z = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed128(z))
            }
            AbstractTensor::Float32(x) => {
                let z = plc.mean(sess, axis, &x);
                Ok(AbstractTensor::Float32(z))
            }
            AbstractTensor::Float64(x) => {
                let z = plc.mean(sess, axis, &x);
                Ok(AbstractTensor::Float64(z))
            }
            AbstractTensor::Bool(_) => {
                unimplemented!()
            }
        }
    }

    pub(crate) fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        sig: Signature,
        axis: Option<u32>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementMean<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementMean<S, Fixed128T, Fixed128T>,
        ReplicatedPlacement: PlacementTruncPr<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementTruncPr<S, Fixed128T, Fixed128T>,
    {
        let precision = match sig.arg(0) {
            Ok(Ty::Tensor(TensorDType::Fixed64 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            Ok(Ty::Tensor(TensorDType::Fixed128 {
                fractional_precision: precision,
                ..
            })) => Some(precision),
            _ => None,
        };
        match x {
            AbstractTensor::Fixed64(x) => {
                let z = plc.mean(sess, axis, &x);
                let z = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed64(z))
            }
            AbstractTensor::Fixed128(x) => {
                let z = plc.mean(sess, axis, &x);
                let z = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed128(z))
            }
            x => Err(Error::UnimplementedOperator(format!(
                "Replicated mean is unsupported for {:?}.",
                x.ty_desc()
            ))),
        }
    }
}

impl SumOp {
    pub(crate) fn logical_host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<usize>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementSum<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementSum<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementSum<S, Float32T, Float32T>,
        HostPlacement: PlacementSum<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(AbstractTensor::Fixed64(z))
            }
            AbstractTensor::Fixed128(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(AbstractTensor::Fixed128(z))
            }
            AbstractTensor::Float32(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(AbstractTensor::Float32(z))
            }
            AbstractTensor::Float64(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(AbstractTensor::Float64(z))
            }
            AbstractTensor::Bool(_) => {
                unimplemented!()
            }
        }
    }

    pub(crate) fn logical_rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<usize>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementSum<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementSum<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(AbstractTensor::Fixed64(z))
            }
            AbstractTensor::Fixed128(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(AbstractTensor::Fixed128(z))
            }
            x => Err(Error::UnimplementedOperator(format!(
                "Replicated sum is unsupported for {:?}.",
                x.ty_desc()
            ))),
        }
    }
}

impl OnesOp {
    #[allow(clippy::type_complexity)]
    pub(crate) fn host_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        shape: cs!(HostShape),
    ) -> Result<
        AbstractTensor<
            cs!(Fixed64Tensor),
            cs!(Fixed128Tensor),
            cs!(Float32Tensor),
            cs!(Float64Tensor),
            cs!(BooleanTensor),
        >,
    >
    where
        HostShape: KnownType<S>,
        Fixed64Tensor: KnownType<S>,
        Fixed128Tensor: KnownType<S>,
        Float32Tensor: KnownType<S>,
        Float64Tensor: KnownType<S>,
        BooleanTensor: KnownType<S>,
        HostPlacement: PlacementOnes<S, cs!(HostShape), cs!(Float64Tensor)>,
    {
        let result = plc.ones(sess, &shape);
        Ok(AbstractTensor::Float64(result))
    }
}

impl ExpandDimsOp {
    pub(crate) fn logical_host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<usize>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementExpandDims<S, Float32T, Float32T>,
        HostPlacement: PlacementExpandDims<S, Float64T, Float64T>,
        HostPlacement: PlacementExpandDims<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementExpandDims<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementExpandDims<S, BoolT, BoolT>,
    {
        match x {
            AbstractTensor::Float64(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Float64(z))
            }
            AbstractTensor::Float32(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Float32(z))
            }
            AbstractTensor::Fixed64(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Fixed64(z))
            }
            AbstractTensor::Fixed128(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Fixed128(z))
            }
            AbstractTensor::Bool(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Bool(z))
            }
        }
    }
}

impl ExpandDimsOp {
    pub(crate) fn logical_rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Vec<usize>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementExpandDims<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementExpandDims<S, Fixed128T, Fixed128T>,
        ReplicatedPlacement: PlacementExpandDims<S, BoolT, BoolT>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            AbstractTensor::Bool(x) => {
                let result = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Bool(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated expand_dims for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}

impl IndexAxisOp {
    pub(crate) fn logical_host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementIndexAxis<S, Float32T, Float32T>,
        HostPlacement: PlacementIndexAxis<S, Float64T, Float64T>,
        HostPlacement: PlacementIndexAxis<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementIndexAxis<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementIndexAxis<S, BoolT, BoolT>,
    {
        match x {
            AbstractTensor::Float32(x) => {
                let z = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Float32(z))
            }
            AbstractTensor::Float64(x) => {
                let z = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Float64(z))
            }
            AbstractTensor::Fixed64(x) => {
                let z = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Fixed64(z))
            }
            AbstractTensor::Fixed128(x) => {
                let z = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Fixed128(z))
            }
            AbstractTensor::Bool(x) => {
                let z = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Bool(z))
            }
        }
    }
}

impl IndexAxisOp {
    pub(crate) fn logical_rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        index: usize,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementIndexAxis<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementIndexAxis<S, Fixed128T, Fixed128T>,
        ReplicatedPlacement: PlacementIndexAxis<S, BoolT, BoolT>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            AbstractTensor::Bool(x) => {
                let result = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Bool(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated index_axis for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}

impl ConcatOp {
    pub(crate) fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        axis: u32,
        xs: &[AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>],
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementConcatenate<S, Float32T, Float32T>,
        HostPlacement: PlacementConcatenate<S, Float64T, Float64T>,
        Float32T: Clone,
        Float64T: Clone,
    {
        match xs[0] {
            // (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
            //     let result = plc.concatenate(sess, axis, &x, &y);
            //     AbstractTensor::Fixed64(result)
            // }
            // (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
            //     let result = plc.concatenate(sess, axis, &x, &y);
            //     AbstractTensor::Fixed128(result)
            // }
            AbstractTensor::Float32(_) => {
                let xs: Vec<Float32T> = xs
                    .iter()
                    .map(|x| match x {
                        AbstractTensor::Float32(x) => x.clone(),
                        _ => {
                            unimplemented!(
                                "ConcatOp can not concatenate tensors of different kinds"
                            )
                        }
                    })
                    .collect();
                let result = plc.concatenate(sess, axis, &xs);
                Ok(AbstractTensor::Float32(result))
            }
            AbstractTensor::Float64(_) => {
                let xs: Vec<Float64T> = xs
                    .iter()
                    .map(|x| match x {
                        AbstractTensor::Float64(x) => x.clone(),
                        _ => {
                            unimplemented!(
                                "ConcatOp can not concatenate tensors of different kinds"
                            )
                        }
                    })
                    .collect();
                let result = plc.concatenate(sess, axis, &xs);
                Ok(AbstractTensor::Float64(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(
                "ConcatOp missing an implementation.".to_string(),
            )),
        }
    }

    pub(crate) fn logical_rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: u32,
        xs: &[AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>],
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementConcatenate<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementConcatenate<S, Fixed128T, Fixed128T>,
        Fixed64T: Clone,
        Fixed128T: Clone,
    {
        if xs.is_empty() {
            Err(Error::InvalidArgument(
                "cannot concatenate on empty array of tensors".to_string(),
            ))
        } else {
            let x = &xs[0];
            match x {
                AbstractTensor::Fixed64(_) => {
                    let vec: Result<Vec<Fixed64T>> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            AbstractTensor::Fixed64(x) => Ok(x.clone()),
                            _ => Err(Error::InvalidArgument(
                                "concat does not support mixed tensor types".to_string(),
                            )),
                        })
                        .collect();
                    let result = plc.concatenate(sess, axis, &vec?);
                    Ok(AbstractTensor::Fixed64(result))
                }
                AbstractTensor::Fixed128(_) => {
                    let vec: Result<Vec<Fixed128T>> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            AbstractTensor::Fixed128(x) => Ok(x.clone()),
                            _ => Err(Error::InvalidArgument(
                                "concat does not support mixed tensor types".to_string(),
                            )),
                        })
                        .collect();
                    let result = plc.concatenate(sess, axis, &vec?);
                    Ok(AbstractTensor::Fixed128(result))
                }
                x => Err(Error::UnimplementedOperator(format!(
                    "Missing replicated concatenate op for {:?}",
                    &x.ty_desc(),
                ))),
            }
        }
    }
}

impl TransposeOp {
    pub(crate) fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        // HostPlacement: PlacementTranspose<S, Float32T, Float32T>,
        HostPlacement: PlacementTranspose<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Fixed64(_x) => {
                unimplemented!()
                // let z = plc.transpose(sess, &x);
                // AbstractTensor::Fixed64(z)
            }
            AbstractTensor::Fixed128(_x) => {
                unimplemented!()
                // let z = plc.transpose(sess, &x);
                // AbstractTensor::Fixed128(z)
            }
            AbstractTensor::Float32(_x) => {
                unimplemented!()
                // let z = plc.transpose(sess, &x);
                // AbstractTensor::Float32(z)
            }
            AbstractTensor::Float64(x) => {
                let z = plc.transpose(sess, &x);
                Ok(AbstractTensor::Float64(z))
            }
            AbstractTensor::Bool(_) => {
                unimplemented!()
            }
        }
    }
}

impl InverseOp {
    pub(crate) fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        // HostPlacement: PlacementInverse<S, Float32T, Float32T>,
        HostPlacement: PlacementInverse<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Fixed64(_x) => {
                unimplemented!()
                // let z = plc.inverse(sess, &x);
                // AbstractTensor::Fixed64(z)
            }
            AbstractTensor::Fixed128(_x) => {
                unimplemented!()
                // let z = plc.inverse(sess, &x);
                // AbstractTensor::Fixed128(z)
            }
            AbstractTensor::Float32(_x) => {
                unimplemented!()
                // let z = plc.inverse(sess, &x);
                // AbstractTensor::Float32(z)
            }
            AbstractTensor::Float64(x) => {
                let z = plc.inverse(sess, &x);
                Ok(AbstractTensor::Float64(z))
            }
            AbstractTensor::Bool(_) => {
                unimplemented!()
            }
        }
    }
}

impl LoadOp {
    #[allow(clippy::type_complexity)]
    pub(crate) fn logical_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        key: cs!(HostString),
        query: cs!(HostString),
    ) -> Result<
        AbstractTensor<
            cs!(Fixed64Tensor),
            cs!(Fixed128Tensor),
            cs!(Float32Tensor),
            cs!(Float64Tensor),
            cs!(BooleanTensor),
        >,
    >
    where
        HostString: KnownType<S>,
        Fixed64Tensor: KnownType<S>,
        Fixed128Tensor: KnownType<S>,
        Float32Tensor: KnownType<S>,
        Float64Tensor: KnownType<S>,
        BooleanTensor: KnownType<S>,
        HostPlacement: PlacementLoad<S, cs!(HostString), cs!(HostString), cs!(Float64Tensor)>,
    {
        let z = plc.load(sess, &key, &query);
        Ok(AbstractTensor::Float64(z))
    }
}

impl SaveOp {
    pub(crate) fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        key: cs!(HostString),
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<cs!(Unit)>
    where
        HostString: KnownType<S>,
        Unit: KnownType<S>,
        // HostPlacement: PlacementSave<S, cs!(HostString), Fixed64T, cs!(Unit)>,
        // HostPlacement: PlacementSave<S, cs!(HostString), Fixed128T, cs!(Unit)>,
        HostPlacement: PlacementSave<S, cs!(HostString), Float32T, cs!(Unit)>,
        HostPlacement: PlacementSave<S, cs!(HostString), Float64T, cs!(Unit)>,
        HostPlacement: PlacementSave<S, cs!(HostString), BoolT, cs!(Unit)>,
    {
        match x {
            AbstractTensor::Fixed64(_x) => {
                unimplemented!()
                // plc.save(sess, &key, &x)
            }
            AbstractTensor::Fixed128(_x) => {
                unimplemented!()
                // plc.save(sess, &key, &x)
            }
            AbstractTensor::Bool(x) => Ok(plc.save(sess, &key, &x)),
            AbstractTensor::Float32(x) => Ok(plc.save(sess, &key, &x)),
            AbstractTensor::Float64(x) => Ok(plc.save(sess, &key, &x)),
        }
    }
}

impl ShapeOp {
    pub(crate) fn host_logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        HostShapeT,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<HostShapeT>
    where
        HostPlacement: PlacementShape<S, Float32T, HostShapeT>,
        HostPlacement: PlacementShape<S, Float64T, HostShapeT>,
        HostPlacement: PlacementShape<S, Fixed64T, HostShapeT>,
        HostPlacement: PlacementShape<S, Fixed128T, HostShapeT>,
    {
        match x {
            AbstractTensor::Float32(x) => Ok(plc.shape(sess, &x)),
            AbstractTensor::Float64(x) => Ok(plc.shape(sess, &x)),
            AbstractTensor::Fixed64(x) => Ok(plc.shape(sess, &x)),
            AbstractTensor::Fixed128(x) => Ok(plc.shape(sess, &x)),
            AbstractTensor::Bool(_) => unimplemented!(),
        }
    }

    pub(crate) fn rep_logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        RepShapeT,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<RepShapeT>
    where
        ReplicatedPlacement: PlacementShape<S, Fixed64T, RepShapeT>,
        ReplicatedPlacement: PlacementShape<S, Fixed128T, RepShapeT>,
    {
        match x {
            AbstractTensor::Fixed64(x) => Ok(plc.shape(sess, &x)),
            AbstractTensor::Fixed128(x) => Ok(plc.shape(sess, &x)),
            _ => Err(Error::UnimplementedOperator(
                "Floating point ops not supported on ReplicatedPlacement.".to_string(),
            )),
        }
    }
}

impl ConstantOp {
    pub(crate) fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        value: Constant,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementConstant<S, Float32T>,
        HostPlacement: PlacementConstant<S, Float64T>,
    {
        match sig.ret() {
            Ty::Tensor(TensorDType::Float32) => {
                let z = plc.constant(sess, value);
                Ok(AbstractTensor::Float32(z))
            }
            Ty::Tensor(TensorDType::Float64) => {
                let z = plc.constant(sess, value);
                Ok(AbstractTensor::Float64(z))
            }
            ret => Err(Error::UnimplementedOperator(format!(
                "ConstantOp can not produce tensors of type {:?} yet",
                ret
            ))),
        }
    }

    pub(crate) fn mir3_logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &Mirrored3Placement,
        sig: Signature,
        value: Constant,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        Mirrored3Placement: PlacementConstant<S, Float32T>,
        Mirrored3Placement: PlacementConstant<S, Float64T>,
    {
        match sig.ret() {
            Ty::Tensor(TensorDType::Float32) => {
                let z = plc.constant(sess, value);
                Ok(AbstractTensor::Float32(z))
            }
            Ty::Tensor(TensorDType::Float64) => {
                let z = plc.constant(sess, value);
                Ok(AbstractTensor::Float64(z))
            }
            ret => Err(Error::UnimplementedOperator(format!(
                "ConstantOp can not produce tensors of type {:?} yet",
                ret
            ))),
        }
    }
}

impl InputOp {
    pub(crate) fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        arg_name: String,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementInput<S, Float32T>,
        HostPlacement: PlacementInput<S, Float64T>,
    {
        match sig.ret() {
            Ty::Tensor(TensorDType::Float32) => {
                let z = plc.input(sess, arg_name);
                Ok(AbstractTensor::Float32(z))
            }
            Ty::Tensor(TensorDType::Float64) => {
                let z = plc.input(sess, arg_name);
                Ok(AbstractTensor::Float64(z))
            }
            ret => Err(Error::UnimplementedOperator(format!(
                "InputOp can not produce tensors of type {:?} yet",
                ret
            ))),
        }
    }
}

impl OutputOp {
    pub(crate) fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementOutput<S, Float32T, Float32T>,
        HostPlacement: PlacementOutput<S, Float64T, Float64T>,
        HostPlacement: PlacementOutput<S, BoolT, BoolT>,
    {
        match x {
            AbstractTensor::Fixed64(_x) => Err(Error::UnimplementedOperator(
                "OutputOp missing a Fixed64 implementation.".to_string(),
            )),
            AbstractTensor::Fixed128(_x) => Err(Error::UnimplementedOperator(
                "OutputOp missing a Fixed128 implementation.".to_string(),
            )),
            AbstractTensor::Bool(x) => Ok(AbstractTensor::Bool(plc.output(sess, &x))),
            AbstractTensor::Float32(x) => Ok(AbstractTensor::Float32(plc.output(sess, &x))),
            AbstractTensor::Float64(x) => Ok(AbstractTensor::Float64(plc.output(sess, &x))),
        }
    }
}

impl ExpOp {
    pub(crate) fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementExp<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementExp<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.exp(sess, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.exp(sess, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated exp for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}

impl SigmoidOp {
    pub(crate) fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementSigmoid<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementSigmoid<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.sigmoid(sess, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.sigmoid(sess, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated sigmoid for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}

impl LnOp {
    pub(crate) fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementLn<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementLn<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.ln(sess, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.ln(sess, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated natural logarithm for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}

impl BitOrOp {
    pub(crate) fn logical_host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementOr<S, BoolT, BoolT, BoolT>,
    {
        match (x, y) {
            (AbstractTensor::Bool(x), AbstractTensor::Bool(y)) => {
                let result = plc.or(sess, &x, &y);
                Ok(AbstractTensor::Bool(result))
            }
            (x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing host less op for {:?} and {:?}",
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }
}

impl MaximumOp {
    pub(crate) fn rep_logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: &[AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>],
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementMaximum<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementMaximum<S, Fixed128T, Fixed128T>,
        Fixed64T: Clone,
        Fixed128T: Clone,
    {
        if x.is_empty() {
            return Err(Error::InvalidArgument(
                "maximum op needs a non-empty array of tensors".to_string(),
            ));
        }
        for entry in x {
            if entry.ty_desc() != x[0].ty_desc() {
                return Err(Error::InvalidArgument(
                    "maximum op all args to have same types".to_string(),
                ));
            }
        }

        let out = match x[0] {
            AbstractTensor::Fixed64(_) => {
                let xv: Vec<Fixed64T> = x
                    .iter()
                    .filter_map(|entry| match entry {
                        AbstractTensor::Fixed64(v) => Some(v.clone()),
                        _ => None,
                    })
                    .collect();
                if xv.len() != x.len() {
                    return Err(Error::Unexpected(Some(
                        "maximum op all args to have same types".to_string(),
                    )));
                }
                AbstractTensor::Fixed64(plc.maximum(sess, &xv))
            }
            AbstractTensor::Fixed128(_) => {
                let xv: Vec<Fixed128T> = x
                    .iter()
                    .filter_map(|entry| match entry {
                        AbstractTensor::Fixed128(v) => Some(v.clone()),
                        _ => None, // never going to be reached
                    })
                    .collect();
                if xv.len() != x.len() {
                    return Err(Error::Unexpected(Some(
                        "maximum op all args to have same types".to_string(),
                    )));
                }
                AbstractTensor::Fixed128(plc.maximum(sess, &xv))
            }
            _ => unimplemented!(),
        };
        Ok(out)
    }
}

impl SoftmaxOp {
    pub fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        upmost_index: usize,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        ReplicatedPlacement: PlacementSoftmax<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementSoftmax<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.softmax(sess, axis, upmost_index, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.softmax(sess, axis, upmost_index, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated softmax for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}
