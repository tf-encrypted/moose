use super::*;
use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::{Operands, Session};
use crate::host::HostPlacement;
use crate::kernels::*;
use crate::mirrored::Mirrored3Placement;
use crate::replicated::ReplicatedPlacement;

impl IdentityOp {
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        HostPlacement: PlacementIdentity<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementIdentity<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementIdentity<S, Float32T, Float32T>,
        HostPlacement: PlacementIdentity<S, Float64T, Float64T>,
        HostPlacement: PlacementIdentity<S, BoolT, BoolT>,
        HostPlacement: PlacementIdentity<S, Uint64T, Uint64T>,
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
            AbstractTensor::Uint64(x) => {
                let result = plc.identity(sess, &x);
                Ok(AbstractTensor::Uint64(result))
            }
        }
    }

    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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

    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn host_logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        xs: &[AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>],
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
                    let vec: Operands<Fixed64T> = xs
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
                    let vec: Operands<Fixed128T> = xs
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
                    let vec: Operands<Float32T> = xs
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
                    let vec: Operands<Float64T> = xs
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

    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        xs: &[AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>],
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
                    let vec: Operands<Fixed64T> = xs
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
                    let vec: Operands<Fixed128T> = xs
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
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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

    pub(crate) fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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

    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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

    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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

    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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

impl LessThanOp {
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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

    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        s: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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

    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        s: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        HostPlacement: PlacementMux<S, BoolT, Fixed64T, Fixed64T, Fixed64T>,
        HostPlacement: PlacementMux<S, BoolT, Fixed128T, Fixed128T, Fixed128T>,
        HostPlacement: PlacementMux<S, BoolT, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementMux<S, BoolT, Float64T, Float64T, Float64T>,
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
            (AbstractTensor::Bool(s), AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.mux(sess, &s, &x, &y);
                Ok(AbstractTensor::Float32(result))
            }
            (AbstractTensor::Bool(s), AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.mux(sess, &s, &x, &y);
                Ok(AbstractTensor::Float64(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            (s, x, y) => Err(Error::UnimplementedOperator(format!(
                "Missing host mux op for {:?}, {:?} and {:?}",
                &s.ty_desc(),
                &x.ty_desc(),
                &y.ty_desc()
            ))),
        }
    }
}

impl CastOp {
    pub(crate) fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        HostPlacement: PlacementCast<S, BoolT, Float32T>,
        HostPlacement: PlacementCast<S, BoolT, Float64T>,
        HostPlacement: PlacementCast<S, BoolT, Uint64T>,
        HostPlacement: PlacementCast<S, Float32T, BoolT>,
        HostPlacement: PlacementCast<S, Float32T, Float64T>,
        HostPlacement: PlacementCast<S, Float32T, Uint64T>,
        HostPlacement: PlacementCast<S, Float64T, BoolT>,
        HostPlacement: PlacementCast<S, Float64T, Float32T>,
        HostPlacement: PlacementCast<S, Float64T, Uint64T>,
        HostPlacement: PlacementCast<S, Uint64T, BoolT>,
        HostPlacement: PlacementCast<S, Uint64T, Float32T>,
        HostPlacement: PlacementCast<S, Uint64T, Float64T>,
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
            // standard casts
            // from bool
            (AbstractTensor::Bool(x), Ty::Tensor(TensorDType::Float32)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Float32(res))
            }
            (AbstractTensor::Bool(x), Ty::Tensor(TensorDType::Float64)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Float64(res))
            }
            (AbstractTensor::Bool(x), Ty::Tensor(TensorDType::Uint64)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Uint64(res))
            }
            // from float
            (AbstractTensor::Float32(x), Ty::Tensor(TensorDType::Bool)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Bool(res))
            }
            (AbstractTensor::Float32(x), Ty::Tensor(TensorDType::Float64)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Float64(res))
            }
            (AbstractTensor::Float32(x), Ty::Tensor(TensorDType::Uint64)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Uint64(res))
            }
            (AbstractTensor::Float64(x), Ty::Tensor(TensorDType::Bool)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Bool(res))
            }
            (AbstractTensor::Float64(x), Ty::Tensor(TensorDType::Float32)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Float32(res))
            }
            (AbstractTensor::Float64(x), Ty::Tensor(TensorDType::Uint64)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Uint64(res))
            }
            // from int
            (AbstractTensor::Uint64(x), Ty::Tensor(TensorDType::Float32)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Float32(res))
            }
            (AbstractTensor::Uint64(x), Ty::Tensor(TensorDType::Float64)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Float64(res))
            }
            (AbstractTensor::Uint64(x), Ty::Tensor(TensorDType::Bool)) => {
                let res = plc.cast(sess, &x);
                Ok(AbstractTensor::Bool(res))
            }
            // fixedpoint casts
            // fixedpoint decoding
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
            // fixedpoint encoding
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
                "Cast operator does not support casting of {} to {:?}",
                &x.ty_desc(),
                &ret
            ))),
        }
    }

    pub(crate) fn mir_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>(
        sess: &S,
        plc: &Mirrored3Placement,
        sig: Signature,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        to_column_vector: bool,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
            AbstractTensor::Uint64(_) => {
                unimplemented!()
            }
        }
    }
}

impl MeanOp {
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        axis: Option<u32>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
            x => Err(Error::UnimplementedOperator(format!(
                "Mean op (Host) is unsupported for {:?}.",
                x.ty_desc()
            ))),
        }
    }

    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        sig: Signature,
        axis: Option<u32>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<usize>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
            x => Err(Error::UnimplementedOperator(format!(
                "Sum op (Host) is unsupported for {:?}.",
                x.ty_desc()
            ))),
        }
    }

    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<usize>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_host_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        shape: m!(HostShape),
    ) -> Result<
        AbstractTensor<
            m!(Fixed64Tensor),
            m!(Fixed128Tensor),
            m!(Float32Tensor),
            m!(Float64Tensor),
            m!(BooleanTensor),
            m!(Uint64Tensor),
        >,
    >
    where
        HostShape: KnownType<S>,
        Fixed64Tensor: KnownType<S>,
        Fixed128Tensor: KnownType<S>,
        Float32Tensor: KnownType<S>,
        Float64Tensor: KnownType<S>,
        BooleanTensor: KnownType<S>,
        Uint64Tensor: KnownType<S>,
        HostPlacement: PlacementOnes<S, m!(HostShape), m!(Float64Tensor)>,
    {
        let result = plc.ones(sess, &shape);
        Ok(AbstractTensor::Float64(result))
    }
}

impl ExpandDimsOp {
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<usize>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
            x => Err(Error::UnimplementedOperator(format!(
                "Expand dims op (host) is unsupported for {:?}.",
                x.ty_desc()
            ))),
        }
    }
}

impl ExpandDimsOp {
    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Vec<usize>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
            x => Err(Error::UnimplementedOperator(format!(
                "Index axis op (host) is unsupported for {:?}.",
                x.ty_desc()
            ))),
        }
    }
}

impl IndexAxisOp {
    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        index: usize,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        axis: u32,
        xs: &[AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>],
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
                let xs: Operands<Float32T> = xs
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
                let xs: Operands<Float64T> = xs
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

    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: u32,
        x: &[AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>],
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        ReplicatedPlacement: PlacementConcatenate<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementConcatenate<S, Fixed128T, Fixed128T>,
        ReplicatedPlacement: PlacementConcatenate<S, BoolT, BoolT>,
        Fixed64T: Clone,
        Fixed128T: Clone,
        BoolT: Clone,
    {
        if x.is_empty() {
            return Err(Error::InvalidArgument(
                "Concat op needs a non-empty array of tensors".to_string(),
            ));
        }

        for entry in x {
            if entry.ty_desc() != x[0].ty_desc() {
                return Err(Error::InvalidArgument(
                    "concat op all args to have same types".to_string(),
                ));
            }
        }

        let out = match x[0] {
            AbstractTensor::Fixed64(_) => {
                let xv: Operands<Fixed64T> = x
                    .iter()
                    .filter_map(|entry| match entry {
                        AbstractTensor::Fixed64(v) => Some(v.clone()),
                        _ => None,
                    })
                    .collect();
                if xv.len() != x.len() {
                    return Err(Error::Unexpected(Some(
                        "in concat op all args must have same types".to_string(),
                    )));
                }
                AbstractTensor::Fixed64(plc.concatenate(sess, axis, &xv))
            }
            AbstractTensor::Fixed128(_) => {
                let xv: Operands<Fixed128T> = x
                    .iter()
                    .filter_map(|entry| match entry {
                        AbstractTensor::Fixed128(v) => Some(v.clone()),
                        _ => None, // never going to be reached
                    })
                    .collect();
                if xv.len() != x.len() {
                    return Err(Error::Unexpected(Some(
                        "in concat op all args must have same types".to_string(),
                    )));
                }
                AbstractTensor::Fixed128(plc.concatenate(sess, axis, &xv))
            }
            AbstractTensor::Bool(_) => {
                let xv: Operands<BoolT> = x
                    .iter()
                    .filter_map(|entry| match entry {
                        AbstractTensor::Bool(v) => Some(v.clone()),
                        _ => None, // never going to be reached
                    })
                    .collect();
                if xv.len() != x.len() {
                    return Err(Error::Unexpected(Some(
                        "in concat op all args must have same types".to_string(),
                    )));
                }
                AbstractTensor::Bool(plc.concatenate(sess, axis, &xv))
            }
            _ => {
                return Err(Error::UnimplementedOperator(format!(
                    "Missing replicated concat op for {:?}",
                    &x[0].ty_desc(),
                )))
            }
        };
        Ok(out)
    }
}

impl TransposeOp {
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        HostPlacement: PlacementTranspose<S, Float32T, Float32T>,
        HostPlacement: PlacementTranspose<S, Float64T, Float64T>,
    {
        use AbstractTensor::*;
        match x {
            Float32(x) => {
                let z = plc.transpose(sess, &x);
                Ok(Float32(z))
            }
            Float64(x) => {
                let z = plc.transpose(sess, &x);
                Ok(Float64(z))
            }
            Fixed64(_) | Fixed128(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Transpose op (host) is unsupported for {:?}.", x.ty_desc()),
            )),
        }
    }
}

impl InverseOp {
    pub(crate) fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        // HostPlacement: PlacementInverse<S, Float32T, Float32T>,
        HostPlacement: PlacementInverse<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Float64(x) => {
                let z = plc.inverse(sess, &x);
                Ok(AbstractTensor::Float64(z))
            }
            x => Err(Error::UnimplementedOperator(format!(
                "Inverse op (host) is unsupported for {:?}.",
                x.ty_desc()
            ))),
        }
    }
}

impl LoadOp {
    #[allow(clippy::type_complexity)]
    pub(crate) fn logical_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        key: m!(HostString),
        query: m!(HostString),
    ) -> Result<
        AbstractTensor<
            m!(Fixed64Tensor),
            m!(Fixed128Tensor),
            m!(Float32Tensor),
            m!(Float64Tensor),
            m!(BooleanTensor),
            m!(Uint64Tensor),
        >,
    >
    where
        HostString: KnownType<S>,
        Fixed64Tensor: KnownType<S>,
        Fixed128Tensor: KnownType<S>,
        Float32Tensor: KnownType<S>,
        Float64Tensor: KnownType<S>,
        BooleanTensor: KnownType<S>,
        Uint64Tensor: KnownType<S>,
        HostPlacement: PlacementLoad<S, m!(HostString), m!(HostString), m!(Float64Tensor)>,
    {
        let z = plc.load(sess, &key, &query);
        Ok(AbstractTensor::Float64(z))
    }
}

impl SaveOp {
    pub(crate) fn logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        key: m!(HostString),
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<m!(HostUnit)>
    where
        HostString: KnownType<S>,
        HostUnit: KnownType<S>,
        // HostPlacement: PlacementSave<S, m!(HostString), Fixed64T, m!(HostUnit)>,
        // HostPlacement: PlacementSave<S, m!(HostString), Fixed128T, m!(HostUnit)>,
        HostPlacement: PlacementSave<S, m!(HostString), Float32T, m!(HostUnit)>,
        HostPlacement: PlacementSave<S, m!(HostString), Float64T, m!(HostUnit)>,
        HostPlacement: PlacementSave<S, m!(HostString), BoolT, m!(HostUnit)>,
        HostPlacement: PlacementSave<S, m!(HostString), Uint64T, m!(HostUnit)>,
    {
        match x {
            AbstractTensor::Bool(x) => Ok(plc.save(sess, &key, &x)),
            AbstractTensor::Float32(x) => Ok(plc.save(sess, &key, &x)),
            AbstractTensor::Float64(x) => Ok(plc.save(sess, &key, &x)),
            AbstractTensor::Uint64(x) => Ok(plc.save(sess, &key, &x)),
            x => Err(Error::UnimplementedOperator(format!(
                "Save op (host) is unsupported for {:?}.",
                x.ty_desc()
            ))),
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
        Uint64T,
        HostShapeT,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
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
            x => Err(Error::UnimplementedOperator(format!(
                "Shape op (host) is unsupported for {:?}.",
                x.ty_desc()
            ))),
        }
    }

    pub(crate) fn rep_logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
        RepShapeT,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<RepShapeT>
    where
        ReplicatedPlacement: PlacementShape<S, Fixed64T, RepShapeT>,
        ReplicatedPlacement: PlacementShape<S, Fixed128T, RepShapeT>,
    {
        match x {
            AbstractTensor::Fixed64(x) => Ok(plc.shape(sess, &x)),
            AbstractTensor::Fixed128(x) => Ok(plc.shape(sess, &x)),
            _ => Err(Error::UnimplementedOperator(
                "Shape op (Rep) op not supported on ReplicatedPlacement.".to_string(),
            )),
        }
    }
}

impl ConstantOp {
    pub(crate) fn logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        value: Constant,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        HostPlacement: PlacementConstant<S, Float32T>,
        HostPlacement: PlacementConstant<S, Float64T>,
        HostPlacement: PlacementConstant<S, Uint64T>,
        HostPlacement: PlacementConstant<S, BoolT>,
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
            Ty::Tensor(TensorDType::Uint64) => {
                let z = plc.constant(sess, value);
                Ok(AbstractTensor::Uint64(z))
            }
            Ty::Tensor(TensorDType::Bool) => {
                let z = plc.constant(sess, value);
                Ok(AbstractTensor::Bool(z))
            }
            ret => Err(Error::UnimplementedOperator(format!(
                "ConstantOp can not produce tensors of type {:?} yet",
                ret
            ))),
        }
    }

    pub(crate) fn mir3_logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &Mirrored3Placement,
        sig: Signature,
        value: Constant,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        arg_name: String,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        HostPlacement: PlacementOutput<S, Float32T, Float32T>,
        HostPlacement: PlacementOutput<S, Float64T, Float64T>,
        HostPlacement: PlacementOutput<S, BoolT, BoolT>,
    {
        match x {
            AbstractTensor::Bool(x) => Ok(AbstractTensor::Bool(plc.output(sess, &x))),
            AbstractTensor::Float32(x) => Ok(AbstractTensor::Float32(plc.output(sess, &x))),
            AbstractTensor::Float64(x) => Ok(AbstractTensor::Float64(plc.output(sess, &x))),
            x => Err(Error::UnimplementedOperator(format!(
                "Output op (host) is unsupported for {:?}.",
                x.ty_desc()
            ))),
        }
    }
}

impl ExpOp {
    pub(crate) fn logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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

impl LogOp {
    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        ReplicatedPlacement: PlacementLog<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementLog<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.log(sess, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.log(sess, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated natural logarithm for {:?}",
                &x.ty_desc(),
            ))),
        }
    }

    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        HostPlacement: PlacementLog<S, Float32T, Float32T>,
        HostPlacement: PlacementLog<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Float64(x) => {
                let result = plc.log(sess, &x);
                Ok(AbstractTensor::Float64(result))
            }
            AbstractTensor::Float32(x) => {
                let result = plc.log(sess, &x);
                Ok(AbstractTensor::Float32(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated natural logarithm for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}

impl Log2Op {
    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        ReplicatedPlacement: PlacementLog2<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementLog2<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.log2(sess, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.log2(sess, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated logarithm base 2 for {:?}",
                &x.ty_desc(),
            ))),
        }
    }

    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        HostPlacement: PlacementLog2<S, Float32T, Float32T>,
        HostPlacement: PlacementLog2<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Float64(x) => {
                let result = plc.log2(sess, &x);
                Ok(AbstractTensor::Float64(result))
            }
            AbstractTensor::Float32(x) => {
                let result = plc.log2(sess, &x);
                Ok(AbstractTensor::Float32(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated logarithm base 2 for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}

impl OrOp {
    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
    pub(crate) fn rep_logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: &[AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>],
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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
                let xv: Operands<Fixed64T> = x
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
                let xv: Operands<Fixed128T> = x
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
            _ => {
                return Err(Error::UnimplementedOperator(format!(
                    "Missing replicated maximum op for {:?}",
                    &x[0].ty_desc(),
                )))
            }
        };
        Ok(out)
    }
}

impl SoftmaxOp {
    pub fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        upmost_index: usize,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
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

impl ArgmaxOp {
    pub(crate) fn logical_rep_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        upmost_index: usize,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        ReplicatedPlacement: PlacementArgmax<S, Fixed64T, Uint64T>,
        ReplicatedPlacement: PlacementArgmax<S, Fixed128T, Uint64T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.argmax(sess, axis, upmost_index, &x);
                Ok(AbstractTensor::Uint64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.argmax(sess, axis, upmost_index, &x);
                Ok(AbstractTensor::Uint64(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated argmax for {:?}",
                &x.ty_desc(),
            ))),
        }
    }

    pub(crate) fn logical_host_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
    >(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        upmost_index: usize,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        HostPlacement: PlacementArgmax<S, Fixed64T, Uint64T>,
        HostPlacement: PlacementArgmax<S, Fixed128T, Uint64T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.argmax(sess, axis, upmost_index, &x);
                Ok(AbstractTensor::Uint64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.argmax(sess, axis, upmost_index, &x);
                Ok(AbstractTensor::Uint64(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated argmax for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}
