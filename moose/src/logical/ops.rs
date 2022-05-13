use super::*;
use crate::computation::*;
use crate::error::{Error, Result};
use crate::execution::{Operands, Session};
use crate::host::{HostPlacement, SliceInfo};
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.identity(sess, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.identity(sess, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            Float32(x) => {
                let result = plc.identity(sess, &x);
                Ok(AbstractTensor::Float32(result))
            }
            Float64(x) => {
                let result = plc.identity(sess, &x);
                Ok(AbstractTensor::Float64(result))
            }
            Bool(x) => {
                let result = plc.identity(sess, &x);
                Ok(AbstractTensor::Bool(result))
            }
            Uint64(x) => {
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = rep.identity(sess, &x);
                Ok(Fixed64(result))
            }
            Fixed128(x) => {
                let result = rep.identity(sess, &x);
                Ok(Fixed128(result))
            }
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Missing rep identity op for {:?}", &x.ty_desc(),),
            )),
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
        use AbstractTensor::*;
        match (&x, &y) {
            (Fixed64(x), Fixed64(y)) => {
                let result = plc.add(sess, x, y);
                Ok(Fixed64(result))
            }
            (Fixed128(x), Fixed128(y)) => {
                let result = plc.add(sess, x, y);
                Ok(Fixed128(result))
            }
            (Float32(x), Float32(y)) => {
                let result = plc.add(sess, x, y);
                Ok(Float32(result))
            }
            (Float64(x), Float64(y)) => {
                let result = plc.add(sess, x, y);
                Ok(Float64(result))
            }
            (Fixed64(_), _)
            | (Fixed128(_), _)
            | (Float32(_), _)
            | (Float64(_), _)
            | (Uint64(_), _)
            | (Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing host add op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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
        use AbstractTensor::*;
        match (&x, &y) {
            (Fixed64(x), Fixed64(y)) => {
                let result = plc.add(sess, x, y);
                Ok(AbstractTensor::Fixed64(result))
            }
            (Fixed128(x), Fixed128(y)) => {
                let result = plc.add(sess, x, y);
                Ok(Fixed128(result))
            }
            (Fixed64(_), _)
            | (Fixed128(_), _)
            | (Float32(_), _)
            | (Float64(_), _)
            | (Uint64(_), _)
            | (Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated add op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
            ))),
        }
    }
}

impl AbsOp {
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
        HostPlacement: PlacementAbs<S, Float32T, Float32T>,
        HostPlacement: PlacementAbs<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Float32(x) => {
                let result = plc.abs(sess, &x);
                Ok(AbstractTensor::Float32(result))
            }
            AbstractTensor::Float64(x) => {
                let result = plc.abs(sess, &x);
                Ok(AbstractTensor::Float64(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing host abs for {:?}",
                &x.ty_desc(),
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
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        ReplicatedPlacement: PlacementAbs<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementAbs<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.abs(sess, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.abs(sess, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            // TODO(Morten) would be nice to catch statically; perhaps if custom kernel?!
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated abs for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}

impl ReluOp {
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
        HostPlacement: PlacementRelu<S, Float32T, Float32T>,
        HostPlacement: PlacementRelu<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Float32(x) => {
                let result = plc.relu(sess, &x);
                Ok(AbstractTensor::Float32(result))
            }
            AbstractTensor::Float64(x) => {
                let result = plc.relu(sess, &x);
                Ok(AbstractTensor::Float64(result))
            }
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing host relu for {:?}",
                &x.ty_desc(),
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
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        ReplicatedPlacement: PlacementRelu<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementRelu<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let result = plc.relu(sess, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            AbstractTensor::Fixed128(x) => {
                let result = plc.relu(sess, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing replicated relu for {:?}",
                &x.ty_desc(),
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
            use AbstractTensor::*;
            let x = &xs[0];
            match x {
                Fixed64(_) => {
                    let vec: Operands<Fixed64T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            Fixed64(x) => x.clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(Fixed64(result))
                }
                Fixed128(_) => {
                    let vec: Operands<Fixed128T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            Fixed128(x) => x.clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(Fixed128(result))
                }
                Float32(_) => {
                    let vec: Operands<Float32T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            Float32(x) => x.clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(Float32(result))
                }
                Float64(_) => {
                    let vec: Operands<Float64T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            Float64(x) => x.clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(Float64(result))
                }
                Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(format!(
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
            use AbstractTensor::*;
            let x = &xs[0];
            match x {
                Fixed64(_) => {
                    let vec: Operands<Fixed64T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            Fixed64(x) => (*x).clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(Fixed64(result))
                }
                Fixed128(_) => {
                    let vec: Operands<Fixed128T> = xs
                        .iter()
                        .map(|abstract_tensor| match abstract_tensor {
                            Fixed128(x) => (*x).clone(),
                            _ => unimplemented!("mixed types in tensor"),
                        })
                        .collect();
                    let result = plc.add_n(sess, &vec);
                    Ok(Fixed128(result))
                }
                Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                    format!("Missing replicated add_n op for {:?}", &x.ty_desc(),),
                )),
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
        use AbstractTensor::*;
        match (&x, &y) {
            (Fixed64(x), Fixed64(y)) => {
                let result = plc.sub(sess, x, y);
                Ok(Fixed64(result))
            }
            (Fixed128(x), Fixed128(y)) => {
                let result = plc.sub(sess, x, y);
                Ok(Fixed128(result))
            }
            (Float32(x), Float32(y)) => {
                let result = plc.sub(sess, x, y);
                Ok(Float32(result))
            }
            (Float64(x), Float64(y)) => {
                let result = plc.sub(sess, x, y);
                Ok(Float64(result))
            }
            (Fixed64(_), _)
            | (Fixed128(_), _)
            | (Float32(_), _)
            | (Float64(_), _)
            | (Uint64(_), _)
            | (Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing host sub op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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
        use AbstractTensor::*;
        match (&x, &y) {
            (Fixed64(x), Fixed64(y)) => {
                let result = plc.sub(sess, x, y);
                Ok(Fixed64(result))
            }
            (Fixed128(x), Fixed128(y)) => {
                let result = plc.sub(sess, x, y);
                Ok(Fixed128(result))
            }
            (Fixed64(_), _)
            | (Fixed128(_), _)
            | (Float32(_), _)
            | (Float64(_), _)
            | (Uint64(_), _)
            | (Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated sub op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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

        match (&x, &y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.mul(sess, x, y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.mul(sess, x, y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed128(result))
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.mul(sess, x, y);
                Ok(AbstractTensor::Float32(result))
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.mul(sess, x, y);
                Ok(AbstractTensor::Float64(result))
            }
            (AbstractTensor::Fixed64(_), _)
            | (AbstractTensor::Fixed128(_), _)
            | (AbstractTensor::Float32(_), _)
            | (AbstractTensor::Float64(_), _)
            | (AbstractTensor::Uint64(_), _)
            | (AbstractTensor::Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing host mul op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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

        match (&x, &y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.mul(sess, x, y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.mul(sess, x, y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed128(result))
            }
            (AbstractTensor::Fixed64(_), _)
            | (AbstractTensor::Fixed128(_), _)
            | (AbstractTensor::Float32(_), _)
            | (AbstractTensor::Float64(_), _)
            | (AbstractTensor::Uint64(_), _)
            | (AbstractTensor::Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated mul op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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
        use AbstractTensor::*;
        match (&x, &y) {
            (Fixed64(x), Fixed64(y)) => {
                let result = plc.div(sess, x, y);
                Ok(Fixed64(result))
            }
            (Fixed128(x), Fixed128(y)) => {
                let result = plc.div(sess, x, y);
                Ok(Fixed128(result))
            }
            (Float32(x), Float32(y)) => {
                let result = plc.div(sess, x, y);
                Ok(Float32(result))
            }
            (Float64(x), Float64(y)) => {
                let result = plc.div(sess, x, y);
                Ok(Float64(result))
            }
            (Fixed64(_), _)
            | (Fixed128(_), _)
            | (Float32(_), _)
            | (Float64(_), _)
            | (Uint64(_), _)
            | (Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing host div op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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
        use AbstractTensor::*;
        match (&x, &y) {
            (Fixed64(x), Fixed64(y)) => {
                let result = plc.div(sess, x, y);
                Ok(Fixed64(result))
            }
            (Fixed128(x), Fixed128(y)) => {
                let result = plc.div(sess, x, y);
                Ok(Fixed128(result))
            }
            (Fixed64(_), _)
            | (Fixed128(_), _)
            | (Float32(_), _)
            | (Float64(_), _)
            | (Uint64(_), _)
            | (Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated div for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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

        match (&x, &y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.dot(sess, x, y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.dot(sess, x, y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed128(result))
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.dot(sess, x, y);
                Ok(AbstractTensor::Float32(result))
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.dot(sess, x, y);
                Ok(AbstractTensor::Float64(result))
            }
            (AbstractTensor::Fixed64(_), _)
            | (AbstractTensor::Fixed128(_), _)
            | (AbstractTensor::Float32(_), _)
            | (AbstractTensor::Float64(_), _)
            | (AbstractTensor::Uint64(_), _)
            | (AbstractTensor::Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing host dot op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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

        match (&x, &y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.dot(sess, x, y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed64(result))
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.dot(sess, x, y);
                let result = plc.trunc_pr(sess, precision.unwrap(), &z);
                Ok(AbstractTensor::Fixed128(result))
            }
            (AbstractTensor::Fixed64(_), _)
            | (AbstractTensor::Fixed128(_), _)
            | (AbstractTensor::Float32(_), _)
            | (AbstractTensor::Float64(_), _)
            | (AbstractTensor::Uint64(_), _)
            | (AbstractTensor::Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated dot op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
            ))),
        }
    }
}

impl LessOp {
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
        HostPlacement: PlacementLess<S, Fixed64T, Fixed64T, BoolT>,
        HostPlacement: PlacementLess<S, Fixed128T, Fixed128T, BoolT>,
        HostPlacement: PlacementLess<S, Float32T, Float32T, BoolT>,
        HostPlacement: PlacementLess<S, Float64T, Float64T, BoolT>,
    {
        use AbstractTensor::*;
        match (&x, &y) {
            (Fixed64(x), Fixed64(y)) => {
                let result = plc.less(sess, x, y);
                Ok(Bool(result))
            }
            (Fixed128(x), Fixed128(y)) => {
                let result = plc.less(sess, x, y);
                Ok(Bool(result))
            }
            (Float32(x), Float32(y)) => {
                let result = plc.less(sess, x, y);
                Ok(Bool(result))
            }
            (Float64(x), Float64(y)) => {
                let result = plc.less(sess, x, y);
                Ok(Bool(result))
            }
            (Fixed64(_), _)
            | (Fixed128(_), _)
            | (Float32(_), _)
            | (Float64(_), _)
            | (Uint64(_), _)
            | (Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing host less op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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
        ReplicatedPlacement: PlacementLess<S, Fixed64T, Fixed64T, BoolT>,
        ReplicatedPlacement: PlacementLess<S, Fixed128T, Fixed128T, BoolT>,
    {
        use AbstractTensor::*;
        match (&x, &y) {
            (Fixed64(x), Fixed64(y)) => {
                let result = plc.less(sess, x, y);
                Ok(Bool(result))
            }
            (Fixed128(x), Fixed128(y)) => {
                let result = plc.less(sess, x, y);
                Ok(Bool(result))
            }
            (Fixed64(_), _)
            | (Fixed128(_), _)
            | (Float32(_), _)
            | (Float64(_), _)
            | (Uint64(_), _)
            | (Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing host less op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
            ))),
        }
    }
}

impl GreaterOp {
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
        HostPlacement: PlacementGreater<S, Fixed64T, Fixed64T, BoolT>,
        HostPlacement: PlacementGreater<S, Fixed128T, Fixed128T, BoolT>,
        HostPlacement: PlacementGreater<S, Float32T, Float32T, BoolT>,
        HostPlacement: PlacementGreater<S, Float64T, Float64T, BoolT>,
    {
        use AbstractTensor::*;
        match (&x, &y) {
            (Fixed64(x), Fixed64(y)) => {
                let result = plc.greater(sess, x, y);
                Ok(Bool(result))
            }
            (Fixed128(x), Fixed128(y)) => {
                let result = plc.greater(sess, x, y);
                Ok(Bool(result))
            }
            (Float32(x), Float32(y)) => {
                let result = plc.greater(sess, x, y);
                Ok(Bool(result))
            }
            (Float64(x), Float64(y)) => {
                let result = plc.greater(sess, x, y);
                Ok(Bool(result))
            }
            (Fixed64(_), _)
            | (Fixed128(_), _)
            | (Float32(_), _)
            | (Float64(_), _)
            | (Uint64(_), _)
            | (Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing host greater op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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
        ReplicatedPlacement: PlacementGreater<S, Fixed64T, Fixed64T, BoolT>,
        ReplicatedPlacement: PlacementGreater<S, Fixed128T, Fixed128T, BoolT>,
    {
        use AbstractTensor::*;
        match (&x, &y) {
            (Fixed64(x), Fixed64(y)) => {
                let result = plc.greater(sess, x, y);
                Ok(Bool(result))
            }
            (Fixed128(x), Fixed128(y)) => {
                let result = plc.greater(sess, x, y);
                Ok(Bool(result))
            }
            (Fixed64(_), _)
            | (Fixed128(_), _)
            | (Float32(_), _)
            | (Float64(_), _)
            | (Uint64(_), _)
            | (Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing host greater op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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
        use AbstractTensor::*;
        match (&s, &x, &y) {
            (Bool(s), Fixed64(x), Fixed64(y)) => {
                let result = plc.mux(sess, s, x, y);
                Ok(Fixed64(result))
            }
            (Bool(s), Fixed128(x), Fixed128(y)) => {
                let result = plc.mux(sess, s, x, y);
                Ok(Fixed128(result))
            }
            (Fixed64(_), _, _)
            | (Fixed128(_), _, _)
            | (Float32(_), _, _)
            | (Float64(_), _, _)
            | (Uint64(_), _, _)
            | (Bool(_), _, _) => Err(Error::UnimplementedOperator(format!(
                "Missing replicated mux op for {:?}, {:?} and {:?}",
                s.ty_desc(),
                x.ty_desc(),
                y.ty_desc()
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
        use AbstractTensor::*;
        match (&s, &x, &y) {
            (Bool(s), Fixed64(x), Fixed64(y)) => {
                let result = plc.mux(sess, s, x, y);
                Ok(Fixed64(result))
            }
            (Bool(s), Fixed128(x), Fixed128(y)) => {
                let result = plc.mux(sess, s, x, y);
                Ok(Fixed128(result))
            }
            (Bool(s), Float32(x), Float32(y)) => {
                let result = plc.mux(sess, s, x, y);
                Ok(Float32(result))
            }
            (Bool(s), Float64(x), Float64(y)) => {
                let result = plc.mux(sess, s, x, y);
                Ok(Float64(result))
            }
            (Fixed64(_), _, _)
            | (Fixed128(_), _, _)
            | (Float32(_), _, _)
            | (Float64(_), _, _)
            | (Uint64(_), _, _)
            | (Bool(_), _, _) => Err(Error::UnimplementedOperator(format!(
                "Missing host mux op for {:?}, {:?} and {:?}",
                s.ty_desc(),
                x.ty_desc(),
                y.ty_desc()
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

        match (&x, sig.ret()) {
            // standard casts
            // from bool
            (AbstractTensor::Bool(x), Ty::Tensor(TensorDType::Float32)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Float32(res))
            }
            (AbstractTensor::Bool(x), Ty::Tensor(TensorDType::Float64)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Float64(res))
            }
            (AbstractTensor::Bool(x), Ty::Tensor(TensorDType::Uint64)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Uint64(res))
            }
            // from float
            (AbstractTensor::Float32(x), Ty::Tensor(TensorDType::Bool)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Bool(res))
            }
            (AbstractTensor::Float32(x), Ty::Tensor(TensorDType::Float64)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Float64(res))
            }
            (AbstractTensor::Float32(x), Ty::Tensor(TensorDType::Uint64)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Uint64(res))
            }
            (AbstractTensor::Float64(x), Ty::Tensor(TensorDType::Bool)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Bool(res))
            }
            (AbstractTensor::Float64(x), Ty::Tensor(TensorDType::Float32)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Float32(res))
            }
            (AbstractTensor::Float64(x), Ty::Tensor(TensorDType::Uint64)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Uint64(res))
            }
            // from int
            (AbstractTensor::Uint64(x), Ty::Tensor(TensorDType::Float32)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Float32(res))
            }
            (AbstractTensor::Uint64(x), Ty::Tensor(TensorDType::Float64)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Float64(res))
            }
            (AbstractTensor::Uint64(x), Ty::Tensor(TensorDType::Bool)) => {
                let res = plc.cast(sess, x);
                Ok(AbstractTensor::Bool(res))
            }
            // fixedpoint casts
            // fixedpoint decoding
            (AbstractTensor::Fixed64(x), Ty::Tensor(TensorDType::Float32)) => {
                let (_, fractional_precision) = arg0_precision.unwrap();
                let inner = plc.fixedpoint_decode(sess, fractional_precision, x);
                Ok(AbstractTensor::Float32(inner))
            }
            (AbstractTensor::Fixed128(x), Ty::Tensor(TensorDType::Float64)) => {
                let (_, fractional_precision) = arg0_precision.unwrap();
                let inner = plc.fixedpoint_decode(sess, fractional_precision, x);
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
                    plc.fixedpoint_encode(sess, fractional_precision, integral_precision, x);
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
                    plc.fixedpoint_encode(sess, fractional_precision, integral_precision, x);
                Ok(AbstractTensor::Fixed128(inner))
            }
            (AbstractTensor::Float32(_), ret)
            | (AbstractTensor::Float64(_), ret)
            | (AbstractTensor::Fixed64(_), ret)
            | (AbstractTensor::Fixed128(_), ret)
            | (AbstractTensor::Uint64(_), ret)
            | (AbstractTensor::Bool(_), ret) => Err(Error::UnimplementedOperator(format!(
                "Cast operator does not support casting of {} to {:?}",
                x.ty_desc(),
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

        match (&x, sig.ret()) {
            (AbstractTensor::Fixed64(x), Ty::Tensor(TensorDType::Float32)) => {
                let (_, fractional_precision) = arg0_precision.unwrap();
                let inner = plc.fixedpoint_decode(sess, fractional_precision, x);
                Ok(AbstractTensor::Float32(inner))
            }
            (AbstractTensor::Fixed128(x), Ty::Tensor(TensorDType::Float64)) => {
                let (_, fractional_precision) = arg0_precision.unwrap();
                let inner = plc.fixedpoint_decode(sess, fractional_precision, x);
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
                    plc.fixedpoint_encode(sess, fractional_precision, integral_precision, x);
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
                    plc.fixedpoint_encode(sess, fractional_precision, integral_precision, x);
                Ok(AbstractTensor::Fixed128(inner))
            }
            (AbstractTensor::Float32(_), ret)
            | (AbstractTensor::Float64(_), ret)
            | (AbstractTensor::Fixed64(_), ret)
            | (AbstractTensor::Fixed128(_), ret)
            | (AbstractTensor::Uint64(_), ret)
            | (AbstractTensor::Bool(_), ret) => Err(Error::UnimplementedOperator(format!(
                "Cast operator does not support casting of {:?} to {:?}",
                x.ty_desc(),
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
        HostPlacement: PlacementAtLeast2D<S, Float32T, Float32T>,
        HostPlacement: PlacementAtLeast2D<S, Float64T, Float64T>,
    {
        use AbstractTensor::*;
        match x {
            Float32(x) => {
                let z = plc.at_least_2d(sess, to_column_vector, &x);
                Ok(Float32(z))
            }
            Float64(x) => {
                let z = plc.at_least_2d(sess, to_column_vector, &x);
                Ok(Float64(z))
            }
            Fixed64(_) | Fixed128(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Missing replicated at_least_2d for {:?}", &x.ty_desc(),),
            )),
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
            AbstractTensor::Bool(_) | AbstractTensor::Uint64(_) => {
                Err(Error::UnimplementedOperator(format!(
                    "Mean op (Host) is unsupported for {:?}.",
                    x.ty_desc()
                )))
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
            AbstractTensor::Float32(_)
            | AbstractTensor::Float64(_)
            | AbstractTensor::Bool(_)
            | AbstractTensor::Uint64(_) => Err(Error::UnimplementedOperator(format!(
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(Fixed64(z))
            }
            Fixed128(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(Fixed128(z))
            }
            Float32(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(Float32(z))
            }
            Float64(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(Float64(z))
            }
            Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(format!(
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(Fixed64(z))
            }
            Fixed128(x) => {
                let z = plc.sum(sess, axis, &x);
                Ok(Fixed128(z))
            }
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Replicated sum is unsupported for {:?}.", x.ty_desc()),
            )),
        }
    }
}

impl OnesOp {
    #[allow(clippy::type_complexity)]
    pub(crate) fn logical_host_kernel<S: Session, TensorT, HostS, RepS>(
        sess: &S,
        plc: &HostPlacement,
        shape: AbstractShape<HostS, RepS>,
    ) -> Result<m!(TensorT)>
    where
        TensorT: KnownType<S>,
        HostPlacement: PlacementOnes<S, HostS, m!(TensorT)>,
        HostPlacement: PlacementReveal<S, RepS, HostS>,
    {
        match shape {
            AbstractShape::Host(sh) => Ok(plc.ones(sess, &sh)),
            AbstractShape::Replicated(sh) => {
                let sh = plc.reveal(sess, &sh);
                Ok(plc.ones(sess, &sh))
            }
        }
    }
}

impl ZerosOp {
    #[allow(clippy::type_complexity)]
    pub(crate) fn logical_host_kernel<S: Session, TensorT, HostS, RepS>(
        sess: &S,
        plc: &HostPlacement,
        shape: AbstractShape<HostS, RepS>,
    ) -> Result<m!(TensorT)>
    where
        TensorT: KnownType<S>,
        HostPlacement: PlacementZeros<S, HostS, m!(TensorT)>,
        HostPlacement: PlacementReveal<S, RepS, HostS>,
    {
        match shape {
            AbstractShape::Host(sh) => Ok(plc.zeros(sess, &sh)),
            AbstractShape::Replicated(sh) => {
                let sh = plc.reveal(sess, &sh);
                Ok(plc.zeros(sess, &sh))
            }
        }
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
        use AbstractTensor::*;
        match x {
            Float64(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Float64(z))
            }
            Float32(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Float32(z))
            }
            Fixed64(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Fixed64(z))
            }
            Fixed128(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Fixed128(z))
            }
            Bool(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Bool(z))
            }
            Uint64(_) => Err(Error::UnimplementedOperator(format!(
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            Bool(x) => {
                let result = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Bool(result))
            }
            Float32(_) | Float64(_) | Uint64(_) => Err(Error::UnimplementedOperator(format!(
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
        use AbstractTensor::*;
        match x {
            Float32(x) => {
                let z = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Float32(z))
            }
            Float64(x) => {
                let z = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Float64(z))
            }
            Fixed64(x) => {
                let z = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Fixed64(z))
            }
            Fixed128(x) => {
                let z = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Fixed128(z))
            }
            Bool(x) => {
                let z = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Bool(z))
            }
            Uint64(_) => Err(Error::UnimplementedOperator(format!(
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Fixed128(result))
            }
            Bool(x) => {
                let result = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Bool(result))
            }
            Float32(_) | Float64(_) | Uint64(_) => Err(Error::UnimplementedOperator(format!(
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
        use AbstractTensor::*;
        match xs[0] {
            Float32(_) => {
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
                Ok(Float32(result))
            }
            Float64(_) => {
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
                Ok(Float64(result))
            }
            Fixed64(_) | Fixed128(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
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

        use AbstractTensor::*;
        let out = match x[0] {
            Fixed64(_) => {
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
                Fixed64(plc.concatenate(sess, axis, &xv))
            }
            Fixed128(_) => {
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
                Fixed128(plc.concatenate(sess, axis, &xv))
            }
            Bool(_) => {
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
                Bool(plc.concatenate(sess, axis, &xv))
            }
            Float32(_) | Float64(_) | Uint64(_) => {
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
        HostPlacement: PlacementInverse<S, Float32T, Float32T>,
        HostPlacement: PlacementInverse<S, Float64T, Float64T>,
    {
        use AbstractTensor::*;
        match x {
            Float32(x) => {
                let z = plc.inverse(sess, &x);
                Ok(Float32(z))
            }
            Float64(x) => {
                let z = plc.inverse(sess, &x);
                Ok(Float64(z))
            }
            Fixed64(_) | Fixed128(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Inverse op (host) is unsupported for {:?}.", x.ty_desc()),
            )),
        }
    }
}

impl LoadOp {
    #[allow(clippy::type_complexity)]
    pub(crate) fn logical_kernel<S: Session, TensorT>(
        sess: &S,
        plc: &HostPlacement,
        key: m!(HostString),
        query: m!(HostString),
    ) -> Result<m!(TensorT)>
    where
        HostString: KnownType<S>,
        TensorT: KnownType<S>,
        HostPlacement: PlacementLoad<S, m!(HostString), m!(HostString), m!(TensorT)>,
    {
        let z = plc.load(sess, &key, &query);
        Ok(z)
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
        HostPlacement: PlacementSave<S, m!(HostString), Float32T, m!(HostUnit)>,
        HostPlacement: PlacementSave<S, m!(HostString), Float64T, m!(HostUnit)>,
        HostPlacement: PlacementSave<S, m!(HostString), BoolT, m!(HostUnit)>,
        HostPlacement: PlacementSave<S, m!(HostString), Uint64T, m!(HostUnit)>,
    {
        use AbstractTensor::*;
        match x {
            Bool(x) => Ok(plc.save(sess, &key, &x)),
            Float32(x) => Ok(plc.save(sess, &key, &x)),
            Float64(x) => Ok(plc.save(sess, &key, &x)),
            Uint64(x) => Ok(plc.save(sess, &key, &x)),
            Fixed64(_) | Fixed128(_) => Err(Error::UnimplementedOperator(format!(
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
        RepShapeT,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractShape<HostShapeT, RepShapeT>>
    where
        HostPlacement: PlacementShape<S, Float32T, HostShapeT>,
        HostPlacement: PlacementShape<S, Float64T, HostShapeT>,
        HostPlacement: PlacementShape<S, Fixed64T, HostShapeT>,
        HostPlacement: PlacementShape<S, Fixed128T, HostShapeT>,
    {
        use AbstractShape::*;
        use AbstractTensor::*;
        match x {
            Float32(x) => Ok(Host(plc.shape(sess, &x))),
            Float64(x) => Ok(Host(plc.shape(sess, &x))),
            Fixed64(x) => Ok(Host(plc.shape(sess, &x))),
            Fixed128(x) => Ok(Host(plc.shape(sess, &x))),
            Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(format!(
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
        HostShapeT,
        RepShapeT,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractShape<HostShapeT, RepShapeT>>
    where
        ReplicatedPlacement: PlacementShape<S, Fixed64T, RepShapeT>,
        ReplicatedPlacement: PlacementShape<S, Fixed128T, RepShapeT>,
    {
        use AbstractShape::*;
        use AbstractTensor::*;
        match x {
            Fixed64(x) => Ok(Replicated(plc.shape(sess, &x))),
            Fixed128(x) => Ok(Replicated(plc.shape(sess, &x))),
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                "Shape op (Rep) op not supported on ReplicatedPlacement.".to_string(),
            )),
        }
    }
}

impl ReshapeOp {
    pub(crate) fn host_logical_kernel<
        S: Session,
        Fixed64T,
        Fixed128T,
        Float32T,
        Float64T,
        BoolT,
        Uint64T,
        HostS,
        RepS,
    >(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        shape: AbstractShape<HostS, RepS>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        HostPlacement: PlacementReshape<S, Float32T, HostS, Float32T>,
        HostPlacement: PlacementReshape<S, Float64T, HostS, Float64T>,
        HostPlacement: PlacementReveal<S, RepS, HostS>,
    {
        let sh = match shape {
            AbstractShape::Host(sh) => sh,
            AbstractShape::Replicated(sh) => plc.reveal(sess, &sh),
        };

        use AbstractTensor::*;
        match x {
            Float32(x) => Ok(Float32(plc.reshape(sess, &x, &sh))),
            Float64(x) => Ok(Float64(plc.reshape(sess, &x, &sh))),
            _ => Err(Error::UnimplementedOperator(format!(
                "Save op (host) is unsupported for {:?}.",
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
        HostShapeT,
        RepShapeT,
    >(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
        shape: AbstractShape<HostShapeT, RepShapeT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        ReplicatedPlacement: PlacementShare<S, HostShapeT, RepShapeT>,
        ReplicatedPlacement: PlacementReshape<S, Fixed64T, RepShapeT, Fixed64T>,
        ReplicatedPlacement: PlacementReshape<S, Fixed128T, RepShapeT, Fixed128T>,
    {
        let sh = match shape {
            AbstractShape::Host(sh) => plc.share(sess, &sh),
            AbstractShape::Replicated(sh) => sh,
        };

        use AbstractTensor::*;
        match x {
            Fixed64(x) => Ok(Fixed64(plc.reshape(sess, &x, &sh))),
            Fixed128(x) => Ok(Fixed128(plc.reshape(sess, &x, &sh))),
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                "Reshape op (Rep) op not supported on ReplicatedPlacement.".to_string(),
            )),
        }
    }
}

impl SliceOp {
    pub(crate) fn logical_host_shape<S: Session, HostS, RepS>(
        sess: &S,
        plc: &HostPlacement,
        slice: SliceInfo,
        shape: AbstractShape<HostS, RepS>,
    ) -> Result<AbstractShape<HostS, RepS>>
    where
        HostPlacement: PlacementSlice<S, HostS, HostS>,
        HostPlacement: PlacementReveal<S, RepS, HostS>,
    {
        use AbstractShape::*;
        match shape {
            Host(x) => Ok(Host(plc.slice(sess, slice, &x))),
            Replicated(x) => {
                let sh = plc.reveal(sess, &x);
                Ok(Host(plc.slice(sess, slice, &sh)))
            }
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
        slice: SliceInfo,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        HostPlacement: PlacementSlice<S, Float32T, Float32T>,
        HostPlacement: PlacementSlice<S, Float64T, Float64T>,
        HostPlacement: PlacementSlice<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementSlice<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementSlice<S, BoolT, BoolT>,
        HostPlacement: PlacementSlice<S, Uint64T, Uint64T>,
    {
        use AbstractTensor::*;
        match x {
            Float32(x) => {
                let result = plc.slice(sess, slice, &x);
                Ok(Float32(result))
            }
            Float64(x) => {
                let result = plc.slice(sess, slice, &x);
                Ok(Float64(result))
            }
            Fixed64(x) => {
                let result = plc.slice(sess, slice, &x);
                Ok(Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.slice(sess, slice, &x);
                Ok(Fixed128(result))
            }
            Bool(x) => {
                let result = plc.slice(sess, slice, &x);
                Ok(Bool(result))
            }
            Uint64(x) => {
                let result = plc.slice(sess, slice, &x);
                Ok(Uint64(result))
            }
        }
    }

    pub(crate) fn logical_rep_shape<S: Session, HostS, RepS>(
        sess: &S,
        plc: &ReplicatedPlacement,
        slice: SliceInfo,
        shape: AbstractShape<HostS, RepS>,
    ) -> Result<AbstractShape<HostS, RepS>>
    where
        ReplicatedPlacement: PlacementSlice<S, RepS, RepS>,
        ReplicatedPlacement: PlacementShare<S, HostS, RepS>,
    {
        use AbstractShape::*;
        match shape {
            Replicated(x) => Ok(Replicated(plc.slice(sess, slice, &x))),
            Host(x) => {
                let sh = plc.share(sess, &x);
                Ok(Replicated(plc.slice(sess, slice, &sh)))
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
        info: SliceInfo,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        ReplicatedPlacement: PlacementSlice<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementSlice<S, Fixed128T, Fixed128T>,
        ReplicatedPlacement: PlacementSlice<S, BoolT, BoolT>,
        ReplicatedPlacement: PlacementSlice<S, Uint64T, Uint64T>,
    {
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.slice(sess, info, &x);
                Ok(Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.slice(sess, info, &x);
                Ok(Fixed128(result))
            }
            Bool(x) => {
                let result = plc.slice(sess, info, &x);
                Ok(Bool(result))
            }
            Uint64(x) => {
                let result = plc.slice(sess, info, &x);
                Ok(Uint64(result))
            }
            Float32(_) | Float64(_) => Err(Error::UnimplementedOperator(format!(
                "Missing rep slice for {:?}",
                &x.ty_desc(),
            ))),
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

    pub(crate) fn shape_logical_kernel<S: Session, HostShapeT, RepShapeT>(
        sess: &S,
        plc: &HostPlacement,
        sig: Signature,
        value: Constant,
    ) -> Result<AbstractShape<HostShapeT, RepShapeT>>
    where
        HostPlacement: PlacementConstant<S, HostShapeT>,
    {
        match sig.ret() {
            Ty::Shape(TensorShape::Host) => {
                let z = plc.constant(sess, value);
                Ok(AbstractShape::Host(z))
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
        use AbstractTensor::*;
        match x {
            Bool(x) => Ok(Bool(plc.output(sess, &x))),
            Float32(x) => Ok(Float32(plc.output(sess, &x))),
            Float64(x) => Ok(Float64(plc.output(sess, &x))),
            Fixed64(_) | Fixed128(_) | Uint64(_) => Err(Error::UnimplementedOperator(format!(
                "Output op (host) is unsupported for {:?}.",
                x.ty_desc()
            ))),
        }
    }
}

impl ExpOp {
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
        ReplicatedPlacement: PlacementExp<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementExp<S, Fixed128T, Fixed128T>,
    {
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.exp(sess, &x);
                Ok(Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.exp(sess, &x);
                Ok(Fixed128(result))
            }
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Missing replicated exp for {:?}", &x.ty_desc(),),
            )),
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
        HostPlacement: PlacementExp<S, Float32T, Float32T>,
        HostPlacement: PlacementExp<S, Float64T, Float64T>,
        HostPlacement: PlacementExp<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementExp<S, Fixed128T, Fixed128T>,
    {
        use AbstractTensor::*;
        match x {
            Float32(x) => {
                let result = plc.exp(sess, &x);
                Ok(Float32(result))
            }
            Float64(x) => {
                let result = plc.exp(sess, &x);
                Ok(Float64(result))
            }
            Fixed64(x) => {
                let result = plc.exp(sess, &x);
                Ok(Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.exp(sess, &x);
                Ok(Fixed128(result))
            }
            Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(format!(
                "Missing host exp for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}

impl SqrtOp {
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
        HostPlacement: PlacementSqrt<S, Float32T, Float32T>,
        HostPlacement: PlacementSqrt<S, Float64T, Float64T>,
    {
        use AbstractTensor::*;
        match x {
            Float32(x) => {
                let z = plc.sqrt(sess, &x);
                Ok(Float32(z))
            }
            Float64(x) => {
                let z = plc.sqrt(sess, &x);
                Ok(Float64(z))
            }
            Fixed64(_) | Fixed128(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Sqrt op (host) is unsupported for {:?}.", x.ty_desc()),
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
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>>
    where
        ReplicatedPlacement: PlacementSqrt<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementSqrt<S, Fixed128T, Fixed128T>,
    {
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.sqrt(sess, &x);
                Ok(Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.sqrt(sess, &x);
                Ok(Fixed128(result))
            }
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Missing replicated sqrt for {:?}", &x.ty_desc(),),
            )),
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.sigmoid(sess, &x);
                Ok(Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.sigmoid(sess, &x);
                Ok(Fixed128(result))
            }
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Missing replicated sigmoid for {:?}", &x.ty_desc(),),
            )),
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.log(sess, &x);
                Ok(Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.log(sess, &x);
                Ok(Fixed128(result))
            }
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => {
                Err(Error::UnimplementedOperator(format!(
                    "Missing replicated natural logarithm for {:?}",
                    &x.ty_desc(),
                )))
            }
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
        use AbstractTensor::*;
        match x {
            Float32(x) => {
                let result = plc.log(sess, &x);
                Ok(Float32(result))
            }
            Float64(x) => {
                let result = plc.log(sess, &x);
                Ok(Float64(result))
            }
            Fixed64(_) | Fixed128(_) | Bool(_) | Uint64(_) => {
                Err(Error::UnimplementedOperator(format!(
                    "Missing replicated natural logarithm for {:?}",
                    &x.ty_desc(),
                )))
            }
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.log2(sess, &x);
                Ok(Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.log2(sess, &x);
                Ok(Fixed128(result))
            }
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Missing replicated logarithm base 2 for {:?}", &x.ty_desc(),),
            )),
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
        use AbstractTensor::*;
        match x {
            Float64(x) => {
                let result = plc.log2(sess, &x);
                Ok(Float64(result))
            }
            Float32(x) => {
                let result = plc.log2(sess, &x);
                Ok(Float32(result))
            }
            Fixed64(_) | Fixed128(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Missing replicated logarithm base 2 for {:?}", &x.ty_desc(),),
            )),
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
        use AbstractTensor::*;
        match (&x, &y) {
            (Bool(x), Bool(y)) => {
                let result = plc.or(sess, x, y);
                Ok(Bool(result))
            }
            (Fixed64(_), _)
            | (Fixed128(_), _)
            | (Float32(_), _)
            | (Float64(_), _)
            | (Uint64(_), _)
            | (Bool(_), _) => Err(Error::UnimplementedOperator(format!(
                "Missing host less op for {:?} and {:?}",
                x.ty_desc(),
                y.ty_desc()
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

        use AbstractTensor::*;
        let out = match x[0] {
            Fixed64(_) => {
                let xv: Operands<Fixed64T> = x
                    .iter()
                    .filter_map(|entry| match entry {
                        Fixed64(v) => Some(v.clone()),
                        _ => None,
                    })
                    .collect();
                if xv.len() != x.len() {
                    return Err(Error::Unexpected(Some(
                        "maximum op all args to have same types".to_string(),
                    )));
                }
                Fixed64(plc.maximum(sess, &xv))
            }
            Fixed128(_) => {
                let xv: Operands<Fixed128T> = x
                    .iter()
                    .filter_map(|entry| match entry {
                        Fixed128(v) => Some(v.clone()),
                        _ => None, // never going to be reached
                    })
                    .collect();
                if xv.len() != x.len() {
                    return Err(Error::Unexpected(Some(
                        "maximum op all args to have same types".to_string(),
                    )));
                }
                Fixed128(plc.maximum(sess, &xv))
            }
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => {
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
    pub fn logical_rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT, Uint64T>(
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.softmax(sess, axis, upmost_index, &x);
                Ok(Fixed64(result))
            }
            Fixed128(x) => {
                let result = plc.softmax(sess, axis, upmost_index, &x);
                Ok(Fixed128(result))
            }
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Missing replicated softmax for {:?}", &x.ty_desc(),),
            )),
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
        HostPlacement: PlacementSoftmax<S, Float32T, Float32T>,
        HostPlacement: PlacementSoftmax<S, Float64T, Float64T>,
    {
        use AbstractTensor::*;
        match x {
            Float32(x) => {
                let result = plc.softmax(sess, axis, upmost_index, &x);
                Ok(Float32(result))
            }
            Float64(x) => {
                let result = plc.softmax(sess, axis, upmost_index, &x);
                Ok(Float64(result))
            }
            Fixed64(_) | Fixed128(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Missing replicated softmax for {:?}", &x.ty_desc(),),
            )),
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.argmax(sess, axis, upmost_index, &x);
                Ok(Uint64(result))
            }
            Fixed128(x) => {
                let result = plc.argmax(sess, axis, upmost_index, &x);
                Ok(Uint64(result))
            }
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Missing replicated argmax for {:?}", &x.ty_desc(),),
            )),
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
        use AbstractTensor::*;
        match x {
            Fixed64(x) => {
                let result = plc.argmax(sess, axis, upmost_index, &x);
                Ok(Uint64(result))
            }
            Fixed128(x) => {
                let result = plc.argmax(sess, axis, upmost_index, &x);
                Ok(Uint64(result))
            }
            Float32(_) | Float64(_) | Bool(_) | Uint64(_) => Err(Error::UnimplementedOperator(
                format!("Missing replicated argmax for {:?}", &x.ty_desc(),),
            )),
        }
    }
}
