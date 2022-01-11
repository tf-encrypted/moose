use crate::boolean::BooleanTensor;
use crate::computation::*;
use crate::error::{Error, Result};
use crate::fixedpoint::{Fixed128Tensor, Fixed64Tensor};
use crate::floatingpoint::{Float32Tensor, Float64Tensor};
use crate::host::{HostShape, HostString};
use crate::kernels::*;
use crate::replicated::{
    AbstractReplicatedFixedTensor, RepTen, ReplicatedFixed128Tensor, ReplicatedFixed64Tensor,
    ReplicatedRing128Tensor, ReplicatedRing64Tensor, ReplicatedShape,
};
use crate::symbolic::Symbolic;
use derive_more::Display;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Shape {
    Host(HostShape),
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Copy, Clone, Debug, Display)]
pub enum TensorDType {
    #[display(fmt = "Fixed64({}, {})", integral_precision, fractional_precision)]
    Fixed64 {
        integral_precision: u32,
        fractional_precision: u32,
    },
    #[display(fmt = "Fixed128({}, {})", integral_precision, fractional_precision)]
    Fixed128 {
        integral_precision: u32,
        fractional_precision: u32,
    },
    Float32,
    Float64,
    Bool,
    Unknown,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT> {
    Fixed64(Fixed64T),
    Fixed128(Fixed128T),
    Float32(Float32T),
    Float64(Float64T),
    Bool(BoolT),
}

pub type Tensor =
    AbstractTensor<Fixed64Tensor, Fixed128Tensor, Float32Tensor, Float64Tensor, BooleanTensor>;

impl<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>
    AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>
{
    pub fn ty_desc(&self) -> String {
        match self {
            AbstractTensor::Fixed64(_) => "Tensor(Fixed64)",
            AbstractTensor::Fixed128(_) => "Tensor(Fixed128)",
            AbstractTensor::Float32(_) => "Tensor(Float32)",
            AbstractTensor::Float64(_) => "Tensor(Float64)",
            AbstractTensor::Bool(_) => "Tensor(Bool)",
        }
        .to_string()
    }
}

impl<Fixed64T, Fixed128T, Float32T, Float64T, BoolT> Placed
    for AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>
where
    Fixed64T: Placed,
    Fixed64T::Placement: Into<Placement>,
    Fixed128T: Placed,
    Fixed128T::Placement: Into<Placement>,
    Float32T: Placed,
    Float32T::Placement: Into<Placement>,
    Float64T: Placed,
    Float64T::Placement: Into<Placement>,
    BoolT: Placed,
    BoolT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            AbstractTensor::Fixed64(x) => Ok(x.placement()?.into()),
            AbstractTensor::Fixed128(x) => Ok(x.placement()?.into()),
            AbstractTensor::Float32(x) => Ok(x.placement()?.into()),
            AbstractTensor::Float64(x) => Ok(x.placement()?.into()),
            AbstractTensor::Bool(x) => Ok(x.placement()?.into()),
        }
    }
}

impl PartiallySymbolicType for Tensor {
    #[allow(clippy::type_complexity)]
    type Type = AbstractTensor<
        <Fixed64Tensor as SymbolicType>::Type,
        <Fixed128Tensor as SymbolicType>::Type,
        <Float32Tensor as SymbolicType>::Type,
        <Float64Tensor as SymbolicType>::Type,
        <BooleanTensor as SymbolicType>::Type,
    >;
}

impl<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>
    From<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    for Symbolic<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
where
    Fixed64T: Placed<Placement = Placement>,
    Fixed128T: Placed<Placement = Placement>,
    Float32T: Placed<Placement = Placement>,
    Float64T: Placed<Placement = Placement>,
    BoolT: Placed<Placement = Placement>,
{
    fn from(x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>
    TryFrom<Symbolic<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>>
    for AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>
where
    Fixed64T: Placed<Placement = Placement>,
    Fixed128T: Placed<Placement = Placement>,
    Float32T: Placed<Placement = Placement>,
    Float64T: Placed<Placement = Placement>,
    BoolT: Placed<Placement = Placement>,
{
    type Error = ();
    fn try_from(
        v: Symbolic<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>,
    ) -> std::result::Result<Self, ()> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(()),
        }
    }
}

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

modelled_kernel! {
    PlacementAdd::add, AddOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::rep_kernel),
    ]
}

impl AddOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

modelled_kernel! {
    PlacementSub::sub, SubOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::rep_kernel),
    ]
}

impl SubOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

modelled!(PlacementMul::mul, HostPlacement, (Tensor, Tensor) -> Tensor, MulOp);

kernel! {
    MulOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] attributes[sig] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] attributes[sig] Self::rep_kernel),
    ]
}

impl MulOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

modelled_kernel! {
    PlacementDiv::div, DivOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] Self::rep_kernel),
    ]
}

impl DivOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

modelled!(PlacementDot::dot, HostPlacement, (Tensor, Tensor) -> Tensor, DotOp);

kernel! {
    DotOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [concrete] attributes[sig] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [concrete] attributes[sig] Self::rep_kernel),
    ]
}

impl DotOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

modelled!(PlacementLessThan::less, HostPlacement, (Tensor, Tensor) -> Tensor, LessOp);
modelled!(PlacementLessThan::less, ReplicatedPlacement, (Tensor, Tensor) -> Tensor, LessOp);

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

modelled!(PlacementMux::mux, ReplicatedPlacement, (Tensor, Tensor, Tensor) -> Tensor, MuxOp);

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

modelled!(PlacementCast::cast, HostPlacement, (Tensor) -> Tensor, CastOp);
modelled!(PlacementCast::cast, Mirrored3Placement, (Tensor) -> Tensor, CastOp);

kernel! {
    CastOp,
    [
        (HostPlacement, (Tensor) -> Tensor => [concrete] attributes[sig] Self::kernel),
        (Mirrored3Placement, (Tensor) -> Tensor => [concrete] attributes[sig] Self::mir_kernel),
    ]
}

impl CastOp {
    fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

    fn mir_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

kernel! {
    AtLeast2DOp, [
        (HostPlacement, (Tensor) -> Tensor => [concrete] attributes[to_column_vector] Self::host_kernel),
        // (ReplicatedPlacement, (Tensor) -> Tensor => [hybrid] attributes[to_column_vector] Self::rep_kernel),
    ]
}

impl AtLeast2DOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

kernel! {
    MeanOp, [
        (HostPlacement, (Tensor) -> Tensor => [concrete] attributes[sig, axis] Self::host_kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] attributes[sig, axis] Self::rep_kernel),
    ]
}

impl MeanOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

kernel! {
    SumOp, [
        (HostPlacement, (Tensor) -> Tensor => [concrete] attributes[axis] Self::host_kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [concrete] attributes[axis] Self::rep_kernel),
    ]
}

impl SumOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
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

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
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

modelled_kernel! {
    PlacementOnes::ones, OnesOp,
    [
        (HostPlacement, (HostShape) -> Tensor => [hybrid] Self::host_kernel),
        // We do not support the ReplicatedPlacement: PlacementFill yet, hence we do not support Ones.
        // Also, logical Tensor can only hold Host tensors at the moment.
        // (ReplicatedPlacement, (HostShape) -> Tensor => [hybrid] Self::rep_kernel),
    ]
}

impl OnesOp {
    #[allow(clippy::type_complexity)]
    fn host_kernel<S: Session>(
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

modelled!(PlacementExpandDims::expand_dims, HostPlacement, attributes[axis: Vec<u32>] (Tensor) -> Tensor, ExpandDimsOp);

kernel! {
    ExpandDimsOp,
    [
        (HostPlacement, (Tensor) -> Tensor => [concrete] attributes[axis] Self::host_kernel),
    ]
}

impl ExpandDimsOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<u32>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementExpandDims<S, Float32T, Float32T>,
        HostPlacement: PlacementExpandDims<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Fixed64(_x) => {
                unimplemented!()
                // let z = plc.expand_dims(sess, axis, &x);
                // AbstractTensor::Fixed64(z)
            }
            AbstractTensor::Fixed128(_x) => {
                unimplemented!()
                // let z = plc.expand_dims(sess, axis, &x);
                // AbstractTensor::Fixed128(z)
            }
            AbstractTensor::Float32(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Float32(z))
            }
            AbstractTensor::Float64(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                Ok(AbstractTensor::Float64(z))
            }
            AbstractTensor::Bool(_x) => {
                unimplemented!()
            }
        }
    }
}

impl IndexAxisOp {
    pub fn logical_host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>,
    ) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT>>
    where
        HostPlacement: PlacementIndexAxis<S, Float32T, Float32T>,
        HostPlacement: PlacementIndexAxis<S, Float64T, Float64T>,
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
            AbstractTensor::Bool(x) => {
                let z = plc.index_axis(sess, axis, index, &x);
                Ok(AbstractTensor::Bool(z))
            }
            _ => Err(Error::UnimplementedOperator(format!(
                "Missing host index_axis for {:?}",
                &x.ty_desc(),
            ))),
        }
    }
}

impl IndexAxisOp {
    pub fn logical_rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[Tensor] -> Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[ReplicatedRing64Tensor] -> ReplicatedRing64Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[ReplicatedRing128Tensor] -> ReplicatedRing128Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[ReplicatedFixed64Tensor] -> ReplicatedFixed64Tensor, ConcatOp);
modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[ReplicatedFixed128Tensor] -> ReplicatedFixed128Tensor, ConcatOp);
//modelled!(PlacementConcatenate::concatenate, ReplicatedPlacement, attributes[axis: u32] vec[Tensor] -> Tensor, ConcatOp);

kernel! {
    ConcatOp, [
        (HostPlacement, vec[Tensor] -> Tensor => [concrete] attributes[axis] Self::host_kernel),
        (ReplicatedPlacement, vec[ReplicatedRing64Tensor] -> ReplicatedRing64Tensor => [concrete] attributes[axis] Self::rep_rep_kernel),
        (ReplicatedPlacement, vec[ReplicatedRing128Tensor] -> ReplicatedRing128Tensor => [concrete] attributes[axis] Self::rep_rep_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed64Tensor] -> ReplicatedFixed64Tensor => [concrete] attributes[axis] Self::rep_fixed_kernel),
        (ReplicatedPlacement, vec[ReplicatedFixed128Tensor] -> ReplicatedFixed128Tensor => [concrete] attributes[axis] Self::rep_fixed_kernel),
        //(ReplicatedPlacement, vec[Tensor] -> Tensor => [concrete] attributes[axis] Self::rep_logical_kernel),
    ]
}

impl ConcatOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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
                        AbstractTensor::Float32(x) => (*x).clone(),
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
                        AbstractTensor::Float64(x) => (*x).clone(),
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

    fn rep_rep_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: u32,
        xs: &[RepTen<HostRingT>],
    ) -> Result<RepTen<HostRingT>>
    where
        HostPlacement: PlacementConcatenate<S, HostRingT, HostRingT>,
        HostRingT: Clone,
    {
        let mut z00s: Vec<HostRingT> = Vec::new();
        let mut z10s: Vec<HostRingT> = Vec::new();
        let mut z11s: Vec<HostRingT> = Vec::new();
        let mut z21s: Vec<HostRingT> = Vec::new();
        let mut z22s: Vec<HostRingT> = Vec::new();
        let mut z02s: Vec<HostRingT> = Vec::new();

        let (player0, player1, player2) = plc.host_placements();
        for x in xs.iter() {
            let RepTen {
                shares: [[x00, x10], [x11, x21], [x22, x02]],
            } = &x;

            z00s.push(x00.clone());
            z10s.push(x10.clone());
            z11s.push(x11.clone());
            z21s.push(x21.clone());
            z22s.push(x22.clone());
            z02s.push(x02.clone());
        }
        let z00 = player0.concatenate(sess, axis, &z00s);
        let z10 = player0.concatenate(sess, axis, &z10s);
        let z11 = player1.concatenate(sess, axis, &z11s);
        let z21 = player1.concatenate(sess, axis, &z21s);
        let z22 = player2.concatenate(sess, axis, &z22s);
        let z02 = player2.concatenate(sess, axis, &z02s);
        Ok(RepTen {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        })
    }

    pub(crate) fn rep_fixed_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: u32,
        xs: &[AbstractReplicatedFixedTensor<RepRingT>],
    ) -> Result<AbstractReplicatedFixedTensor<RepRingT>>
    where
        ReplicatedPlacement: PlacementConcatenate<S, RepRingT, RepRingT>,
        RepRingT: Clone,
    {
        if xs.is_empty() {
            Err(Error::InvalidArgument(
                "cannot concat on empty array of tensors".to_string(),
            ))
        } else {
            let mut tensors = Vec::new();
            let fractional_precision = xs[0].fractional_precision;
            let integral_precision = xs[0].integral_precision;
            for x in xs.iter() {
                if (x.integral_precision != integral_precision)
                    || (x.fractional_precision != fractional_precision)
                {
                    return Err(Error::InvalidArgument(
                        "precisions of tensors must match when concatenating".to_string(),
                    ));
                }
                tensors.push(x.tensor.clone());
            }
            let tensor = plc.concatenate(sess, axis, &tensors);
            Ok(AbstractReplicatedFixedTensor {
                tensor,
                fractional_precision,
                integral_precision,
            })
        }
    }

    //pub(crate) fn rep_logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
    //    sess: &S,
    //    plc: &ReplicatedPlacement,
    //    axis: u32,
    //    x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    //) -> Result<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>>
    //where
    //    ReplicatedPlacement: PlacementExp<S, Fixed64T, Fixed64T>,
    //    ReplicatedPlacement: PlacementExp<S, Fixed128T, Fixed128T>,

    //{
    //    //match x {
    //    //    AbstractTensor::Fixed64(x) => {
    //    //        let result = plc.exp(sess, &x);
    //    //        Ok(AbstractTensor::Fixed64(result))
    //    //    }
    //    //    AbstractTensor::Fixed128(x) => {
    //    //        let result = plc.exp(sess, &x);
    //    //        Ok(AbstractTensor::Fixed128(result))
    //    //    }
    //    //    _ => Err(Error::UnimplementedOperator(format!(
    //    //        "Missing replicated concat for {:?}",
    //    //        &x.ty_desc(),
    //    //    ))),
    //    //}
    //    unimplemented!("rep_logical_kernel TODO")
    //}
}

modelled!(PlacementTranspose::transpose, HostPlacement, (Tensor) -> Tensor, TransposeOp);

kernel! {
    TransposeOp, [
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::kernel),
    ]
}

impl TransposeOp {
    pub fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

modelled_kernel! {
    PlacementInverse::inverse, InverseOp,
    [
        (HostPlacement, (Tensor) -> Tensor => [concrete] Self::kernel),
    ]
}

impl InverseOp {
    pub fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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
    pub fn logical_kernel<S: Session>(
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
    pub fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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
    pub fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

    pub fn mir3_logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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
    pub fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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
    pub fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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
    pub fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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
    pub fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T, BoolT>(
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

modelled!(PlacementOr::or, HostPlacement, (Tensor, Tensor) -> Tensor, BitOrOp);

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
