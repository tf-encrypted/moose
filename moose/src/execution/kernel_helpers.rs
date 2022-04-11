use crate::computation::{
    Operator, PartiallySymbolicType, Placed, Placement, SymbolicType, SymbolicValue, Value,
};
use crate::error::Result;
use crate::execution::symbolic::{Symbolic, SymbolicSession};
use crate::execution::SyncSession;
use crate::kernels::NgKernel;
use std::convert::{TryFrom, TryInto};

#[allow(dead_code)]
pub(crate) fn sync_nullary_fn<U, P>(
    _op: &Operator,
    kf: fn(&SyncSession, &P) -> Result<U>,
) -> Result<NgKernel<SyncSession>>
where
    U: 'static,
    P: 'static,

    Placement: TryInto<P, Error = crate::Error>,
    Value: From<U>,
{
    Ok(NgKernel::Nullary {
        closure: Box::new(move |sess: &SyncSession, plc: &Placement| {
            let plc: P = Placement::try_into(plc.clone())?;
            let y = kf(sess, &plc)?;
            Ok(Value::from(y))
        }),
    })
}

pub(crate) fn symbolic_nullary_runtime<U, P>(op: Operator) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error>,
    Placement: From<P>,

    U: PartiallySymbolicType,
    <U as PartiallySymbolicType>::Type: Placed<Placement = P>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(crate::kernels::NgKernel::Nullary {
        closure: Box::new(
            move |sess: &crate::execution::SymbolicSession, plc: &crate::computation::Placement| {
                let plc = P::try_from(plc.clone())?;
                let h = sess.add_operation(&op, &[], &plc);
                let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                Ok(SymbolicValue::from(h))
            },
        ),
    })
}

// TODO(Morten) can we merge sync_unary_box and sync_unary_fn and still
// be certain that we only get two copies? What are the correct trait bounds?

pub(crate) fn sync_unary_box<T0, U, P>(
    _op: Operator,
    kf: Box<dyn Fn(&SyncSession, &P, T0) -> Result<U>>,
) -> Result<NgKernel<SyncSession>>
where
    T0: 'static,
    U: 'static,
    P: 'static,

    Placement: TryInto<P, Error = crate::Error>,
    Value: TryInto<T0, Error = crate::Error>,
    Value: From<U>,
{
    Ok(NgKernel::Unary {
        closure: Box::new(move |sess: &SyncSession, plc: &Placement, v0: Value| {
            let plc: P = Placement::try_into(plc.clone())?;
            let x0: T0 = Value::try_into(v0)?;
            let y = kf(sess, &plc, x0)?;
            Ok(Value::from(y))
        }),
    })
}

pub(crate) fn sync_unary_fn<T0, U, P>(
    _op: Operator,
    kf: fn(&SyncSession, &P, T0) -> Result<U>,
) -> Result<NgKernel<SyncSession>>
where
    T0: 'static,
    U: 'static,
    P: 'static,
    Placement: TryInto<P, Error = crate::Error>,
    Value: TryInto<T0, Error = crate::Error>,
    Value: From<U>,
{
    Ok(NgKernel::Unary {
        closure: Box::new(move |sess: &SyncSession, plc: &Placement, v0: Value| {
            let plc: P = Placement::try_into(plc.clone())?;
            let x0: T0 = Value::try_into(v0)?;
            let y = kf(sess, &plc, x0)?;
            Ok(Value::from(y))
        }),
    })
}

pub(crate) fn symbolic_unary_runtime<T0, U, P>(op: Operator) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error>,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    U: PartiallySymbolicType,

    <T0 as PartiallySymbolicType>::Type: Placed,
    <U as PartiallySymbolicType>::Type: Placed<Placement = P>,

    SymbolicValue: TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(crate::kernels::NgKernel::Unary {
        closure: Box::new(
            move |sess: &crate::execution::SymbolicSession,
                  plc: &crate::computation::Placement,
                  v0: crate::computation::SymbolicValue| {
                let plc = P::try_from(plc.clone())?;
                let v0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;

                match v0 {
                    Symbolic::Symbolic(x0) => {
                        let h = sess.add_operation(&op, &[&x0.op], &plc);
                        let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                        Ok(SymbolicValue::from(h))
                    }
                    _ => unimplemented!(),
                }
            },
        ),
    })
}

pub(crate) fn symbolic_unary_concrete_box<T0, U, P>(
    op: Operator,
    kf: Box<
        dyn Fn(
            &SymbolicSession,
            &P,
            <T0 as PartiallySymbolicType>::Type,
        ) -> Result<<U as PartiallySymbolicType>::Type>,
    >,
) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    U: PartiallySymbolicType, // TODO use SymbolicType here?

    <T0 as PartiallySymbolicType>::Type: Placed + 'static,
    <U as PartiallySymbolicType>::Type: Placed + 'static,
    // TODO(Morten) shouldn't need this, we should have Placed<Placement = P> wrt U
    <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,
    SymbolicValue: TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Unary {
        closure: Box::new(
            move |sess: &SymbolicSession, plc: &Placement, v0: SymbolicValue| {
                let plc = P::try_from(plc.clone())?;
                let v0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;

                match v0 {
                    Symbolic::Concrete(x0) => {
                        let y = kf(sess, &plc, x0)?;
                        Ok(SymbolicValue::from(Symbolic::Concrete(y)))
                    }
                    Symbolic::Symbolic(h0) => {
                        let h = sess.add_operation(&op, &[&h0.op], &plc);
                        let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                        Ok(SymbolicValue::from(h))
                    }
                }
            },
        ),
    })
}

pub(crate) fn symbolic_unary_concrete_fn<T0, U, P>(
    op: Operator,
    kf: fn(
        &SymbolicSession,
        &P,
        <T0 as PartiallySymbolicType>::Type,
    ) -> Result<<U as PartiallySymbolicType>::Type>,
) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    U: PartiallySymbolicType, // TODO use SymbolicType here?

    <T0 as PartiallySymbolicType>::Type: Placed + 'static,
    <U as PartiallySymbolicType>::Type: Placed + 'static,
    // TODO(Morten) shouldn't need this, we should have Placed<Placement = P> wrt U
    <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,
    SymbolicValue: TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Unary {
        closure: Box::new(
            move |sess: &SymbolicSession, plc: &Placement, v0: SymbolicValue| {
                let plc = P::try_from(plc.clone())?;
                let v0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;

                match v0 {
                    Symbolic::Concrete(x0) => {
                        let y = kf(sess, &plc, x0)?;
                        Ok(SymbolicValue::from(Symbolic::Concrete(y)))
                    }
                    Symbolic::Symbolic(h0) => {
                        let h = sess.add_operation(&op, &[&h0.op], &plc);
                        let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                        Ok(SymbolicValue::from(h))
                    }
                }
            },
        ),
    })
}
// v = SymbolicValue::RepTensor(Symbolic<RepTensor>::Concrete(x))

// concrete:
//     y: (concrete) = kf(x)
//     SymbolicValue::from(Symbolic::Concrete(y))

// transparent:
//     y: Symbolic = kf(Symbolic::Concrete(x))
//     SymbolicValue::from(y)

pub(crate) fn symbolic_unary_transparent_box<T0, U, P>(
    _op: Operator,
    kf: Box<
        dyn Fn(
            &SymbolicSession,
            &P,
            <T0 as SymbolicType>::Type,
        ) -> Result<<U as SymbolicType>::Type>,
    >,
) -> Result<NgKernel<SymbolicSession>>
where
    P: 'static,
    Placement: TryInto<P, Error = crate::Error>,

    T0: PartiallySymbolicType,
    <T0 as PartiallySymbolicType>::Type: Placed,
    <T0 as PartiallySymbolicType>::Type: 'static,
    SymbolicValue: TryInto<<T0 as SymbolicType>::Type, Error = crate::Error>,

    U: PartiallySymbolicType,
    <U as PartiallySymbolicType>::Type: Placed<Placement = P>,
    <U as PartiallySymbolicType>::Type: 'static,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Unary {
        closure: Box::new(
            move |sess: &SymbolicSession, plc: &Placement, v0: SymbolicValue| {
                let plc: P = Placement::try_into(plc.clone())?;
                let x0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;
                let y: <U as SymbolicType>::Type = kf(sess, &plc, x0)?;
                Ok(SymbolicValue::from(y))
            },
        ),
    })
}

pub(crate) fn symbolic_unary_transparent_fn<T0, U, P>(
    _op: Operator,
    kf: fn(&SymbolicSession, &P, <T0 as SymbolicType>::Type) -> Result<<U as SymbolicType>::Type>,
) -> Result<NgKernel<SymbolicSession>>
where
    P: 'static,
    Placement: TryInto<P, Error = crate::Error>,

    T0: PartiallySymbolicType,
    <T0 as PartiallySymbolicType>::Type: Placed,
    <T0 as PartiallySymbolicType>::Type: 'static,
    SymbolicValue: TryInto<<T0 as SymbolicType>::Type, Error = crate::Error>,

    U: PartiallySymbolicType,
    <U as PartiallySymbolicType>::Type: Placed<Placement = P>,
    <U as PartiallySymbolicType>::Type: 'static,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Unary {
        closure: Box::new(
            move |sess: &SymbolicSession, plc: &Placement, v0: SymbolicValue| {
                let plc: P = Placement::try_into(plc.clone())?;
                let x0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;
                let y: <U as SymbolicType>::Type = kf(sess, &plc, x0)?;
                Ok(SymbolicValue::from(y))
            },
        ),
    })
}

pub(crate) fn symbolic_unary_hybrid_fn<T0, U, X0, Y, P>(
    op: Operator,
    kf: fn(&SymbolicSession, &P, X0) -> Result<Y>,
) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    U: PartiallySymbolicType,

    Symbolic<<T0 as PartiallySymbolicType>::Type>: Clone + TryInto<X0>,
    Y: Into<<U as SymbolicType>::Type>,

    X0: 'static,
    Y: 'static,

    <T0 as PartiallySymbolicType>::Type: Placed + 'static,
    <U as PartiallySymbolicType>::Type: Placed + 'static,
    // TODO(Morten) shouldn't need this, we should have Placed<Placement = P> wrt U
    <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,

    SymbolicValue: TryInto<<T0 as SymbolicType>::Type, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Unary {
        closure: Box::new(
            move |sess: &SymbolicSession, plc: &Placement, v0: SymbolicValue| {
                let plc = P::try_from(plc.clone())?;

                let vs0: Symbolic<<T0 as PartiallySymbolicType>::Type> =
                    SymbolicValue::try_into(v0)?;

                let v0 = vs0.clone().try_into();

                match v0 {
                    Ok(v0) => {
                        let y = kf(sess, &plc, v0)?;
                        let y: <U as SymbolicType>::Type = y.into();
                        Ok(SymbolicValue::from(y))
                    }
                    _ => match vs0 {
                        Symbolic::Symbolic(h0) => {
                            let h = sess.add_operation(&op, &[&h0.op], &plc);
                            let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                            Ok(SymbolicValue::from(h))
                        }
                        _ => unimplemented!(),
                    },
                }
            },
        ),
    })
}

pub(crate) fn symbolic_unary_hybrid_box<T0, U, X0, Y, P>(
    op: Operator,
    kf: Box<dyn Fn(&SymbolicSession, &P, X0) -> Result<Y>>,
) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    U: PartiallySymbolicType,

    Symbolic<<T0 as PartiallySymbolicType>::Type>: Clone + TryInto<X0>,
    Y: Into<<U as SymbolicType>::Type>,

    X0: 'static,
    Y: 'static,

    <T0 as PartiallySymbolicType>::Type: Placed + 'static,
    <U as PartiallySymbolicType>::Type: Placed + 'static,
    // TODO(Morten) shouldn't need this, we should have Placed<Placement = P> wrt U
    <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,

    SymbolicValue: TryInto<<T0 as SymbolicType>::Type, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Unary {
        closure: Box::new(
            move |sess: &SymbolicSession, plc: &Placement, v0: SymbolicValue| {
                let plc = P::try_from(plc.clone())?;

                let vs0: Symbolic<<T0 as PartiallySymbolicType>::Type> =
                    SymbolicValue::try_into(v0)?;

                let v0 = vs0.clone().try_into();

                match v0 {
                    Ok(v0) => {
                        let y = kf(sess, &plc, v0)?;
                        let y: <U as SymbolicType>::Type = y.into();
                        Ok(SymbolicValue::from(y))
                    }
                    _ => match vs0 {
                        Symbolic::Symbolic(h0) => {
                            let h = sess.add_operation(&op, &[&h0.op], &plc);
                            let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                            Ok(SymbolicValue::from(h))
                        }
                        _ => unimplemented!(),
                    },
                }
            },
        ),
    })
}

pub(crate) fn sync_binary_box<T0, T1, U, P>(
    _op: Operator,
    kf: Box<dyn Fn(&SyncSession, &P, T0, T1) -> Result<U>>,
) -> Result<NgKernel<SyncSession>>
where
    T0: 'static,
    T1: 'static,
    U: 'static,
    P: 'static,

    Placement: TryInto<P, Error = crate::Error>,
    Value: TryInto<T0, Error = crate::Error>,
    Value: TryInto<T1, Error = crate::Error>,
    Value: From<U>,
{
    Ok(NgKernel::Binary {
        closure: Box::new(
            move |sess: &SyncSession, plc: &Placement, v0: Value, v1: Value| {
                let plc: P = Placement::try_into(plc.clone())?;
                let x0: T0 = Value::try_into(v0)?;
                let x1: T1 = Value::try_into(v1)?;
                let y = kf(sess, &plc, x0, x1)?;
                Ok(Value::from(y))
            },
        ),
    })
}

pub(crate) fn sync_binary_fn<T0, T1, U, P>(
    _op: Operator,
    kf: fn(&SyncSession, &P, T0, T1) -> Result<U>,
) -> Result<NgKernel<SyncSession>>
where
    T0: 'static,
    T1: 'static,
    U: 'static,
    P: 'static,
    Placement: TryInto<P, Error = crate::Error>,
    Value: TryInto<T0, Error = crate::Error>,
    Value: TryInto<T1, Error = crate::Error>,
    Value: From<U>,
{
    Ok(NgKernel::Binary {
        closure: Box::new(
            move |sess: &SyncSession, plc: &Placement, v0: Value, v1: Value| {
                let plc: P = Placement::try_into(plc.clone())?;
                let x0: T0 = Value::try_into(v0)?;
                let x1: T1 = Value::try_into(v1)?;
                let y = kf(sess, &plc, x0, x1)?;
                Ok(Value::from(y))
            },
        ),
    })
}

pub(crate) fn symbolic_binary_runtime<T0, T1, U, P>(
    op: Operator,
) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error>,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    T1: PartiallySymbolicType,
    U: PartiallySymbolicType,

    <T0 as PartiallySymbolicType>::Type: Placed,
    <T1 as PartiallySymbolicType>::Type: Placed,
    <U as PartiallySymbolicType>::Type: Placed<Placement = P>,

    SymbolicValue: TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: TryInto<Symbolic<<T1 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(crate::kernels::NgKernel::Binary {
        closure: Box::new(
            move |sess: &crate::execution::SymbolicSession,
                  plc: &crate::computation::Placement,
                  v0: crate::computation::SymbolicValue,
                  v1: crate::computation::SymbolicValue| {
                let plc = P::try_from(plc.clone())?;
                let v0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;
                let v1: <T1 as SymbolicType>::Type = SymbolicValue::try_into(v1)?;

                match (v0, v1) {
                    (Symbolic::Symbolic(x0), Symbolic::Symbolic(x1)) => {
                        let h = sess.add_operation(&op, &[&x0.op, &x1.op], &plc);
                        let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                        Ok(SymbolicValue::from(h))
                    }
                    _ => unimplemented!(),
                }
            },
        ),
    })
}

pub(crate) fn symbolic_binary_concrete_box<T0, T1, U, P>(
    op: Operator,
    kf: Box<
        dyn Fn(
            &SymbolicSession,
            &P,
            <T0 as PartiallySymbolicType>::Type,
            <T1 as PartiallySymbolicType>::Type,
        ) -> Result<<U as PartiallySymbolicType>::Type>,
    >,
) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    T1: PartiallySymbolicType,
    U: PartiallySymbolicType, // TODO use SymbolicType here?

    <T0 as PartiallySymbolicType>::Type: Placed + 'static,
    <T1 as PartiallySymbolicType>::Type: Placed + 'static,
    <U as PartiallySymbolicType>::Type: Placed + 'static,
    // TODO(Morten) shouldn't need this, we should have Placed<Placement = P> wrt U
    <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,
    SymbolicValue: TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: TryInto<Symbolic<<T1 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Binary {
        closure: Box::new(
            move |sess: &SymbolicSession, plc: &Placement, v0: SymbolicValue, v1: SymbolicValue| {
                let plc = P::try_from(plc.clone())?;
                let v0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;
                let v1: <T1 as SymbolicType>::Type = SymbolicValue::try_into(v1)?;

                match (v0, v1) {
                    (Symbolic::Concrete(x0), Symbolic::Concrete(x1)) => {
                        let y = kf(sess, &plc, x0, x1)?;
                        Ok(SymbolicValue::from(Symbolic::Concrete(y)))
                    }
                    (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                        let h = sess.add_operation(&op, &[&h0.op, &h1.op], &plc);
                        let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                        Ok(SymbolicValue::from(h))
                    }
                    _ => Err(crate::error::Error::Unexpected(Some(
                        "Mixed symbolic and concrete value during compilation".to_string(),
                    ))),
                }
            },
        ),
    })
}

pub(crate) fn symbolic_binary_concrete_fn<T0, T1, U, P>(
    op: Operator,
    kf: fn(
        &SymbolicSession,
        &P,
        <T0 as PartiallySymbolicType>::Type,
        <T1 as PartiallySymbolicType>::Type,
    ) -> Result<<U as PartiallySymbolicType>::Type>,
) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    T1: PartiallySymbolicType,
    U: PartiallySymbolicType, // TODO use SymbolicType here?

    <T0 as PartiallySymbolicType>::Type: Placed + 'static,
    <T1 as PartiallySymbolicType>::Type: Placed + 'static,
    <U as PartiallySymbolicType>::Type: Placed + 'static,
    // TODO(Morten) shouldn't need this, we should have Placed<Placement = P> wrt U
    <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,
    SymbolicValue: TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: TryInto<Symbolic<<T1 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Binary {
        closure: Box::new(
            move |sess: &SymbolicSession, plc: &Placement, v0: SymbolicValue, v1: SymbolicValue| {
                let plc = P::try_from(plc.clone())?;
                let v0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;
                let v1: <T1 as SymbolicType>::Type = SymbolicValue::try_into(v1)?;

                match (v0, v1) {
                    (Symbolic::Concrete(x0), Symbolic::Concrete(x1)) => {
                        let y = kf(sess, &plc, x0, x1)?;
                        Ok(SymbolicValue::from(Symbolic::Concrete(y)))
                    }
                    (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                        let h = sess.add_operation(&op, &[&h0.op, &h1.op], &plc);
                        let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                        Ok(SymbolicValue::from(h))
                    }
                    _ => Err(crate::error::Error::Unexpected(Some(
                        "Mixed symbolic and concrete value during compilation".to_string(),
                    ))),
                }
            },
        ),
    })
}

pub(crate) fn symbolic_binary_transparent_fn<T0, T1, U, P>(
    _op: Operator,
    kf: fn(
        &SymbolicSession,
        &P,
        <T0 as SymbolicType>::Type,
        <T1 as SymbolicType>::Type,
    ) -> Result<<U as SymbolicType>::Type>,
) -> Result<NgKernel<SymbolicSession>>
where
    P: 'static,
    Placement: TryInto<P, Error = crate::Error>,

    T0: PartiallySymbolicType,
    <T0 as PartiallySymbolicType>::Type: Placed,
    <T0 as PartiallySymbolicType>::Type: 'static,
    SymbolicValue: TryInto<<T0 as SymbolicType>::Type, Error = crate::Error>,

    T1: PartiallySymbolicType,
    <T1 as PartiallySymbolicType>::Type: Placed,
    <T1 as PartiallySymbolicType>::Type: 'static,
    SymbolicValue: TryInto<<T1 as SymbolicType>::Type, Error = crate::Error>,

    U: PartiallySymbolicType,
    <U as PartiallySymbolicType>::Type: Placed<Placement = P>,
    <U as PartiallySymbolicType>::Type: 'static,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Binary {
        closure: Box::new(
            move |sess: &SymbolicSession, plc: &Placement, v0: SymbolicValue, v1: SymbolicValue| {
                let plc: P = Placement::try_into(plc.clone())?;
                let x0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;
                let x1: <T1 as SymbolicType>::Type = SymbolicValue::try_into(v1)?;
                let y: <U as SymbolicType>::Type = kf(sess, &plc, x0, x1)?;
                Ok(SymbolicValue::from(y))
            },
        ),
    })
}

pub(crate) fn symbolic_binary_hybrid_fn<T0, T1, U, X0, X1, Y, P>(
    op: Operator,
    kf: fn(&SymbolicSession, &P, X0, X1) -> Result<Y>,
) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    T1: PartiallySymbolicType,
    U: PartiallySymbolicType,

    Symbolic<<T0 as PartiallySymbolicType>::Type>: Clone + TryInto<X0>,
    Symbolic<<T1 as PartiallySymbolicType>::Type>: Clone + TryInto<X1>,
    Y: Into<<U as SymbolicType>::Type>,

    X0: 'static,
    X1: 'static,
    Y: 'static,

    <T0 as PartiallySymbolicType>::Type: Placed + 'static,
    <T1 as PartiallySymbolicType>::Type: Placed + 'static,
    <U as PartiallySymbolicType>::Type: Placed + 'static,
    // TODO(Morten) shouldn't need this, we should have Placed<Placement = P> wrt U
    <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,

    SymbolicValue: TryInto<<T0 as SymbolicType>::Type, Error = crate::Error>,
    SymbolicValue: TryInto<<T1 as SymbolicType>::Type, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Binary {
        closure: Box::new(
            move |sess: &SymbolicSession, plc: &Placement, v0: SymbolicValue, v1: SymbolicValue| {
                let plc = P::try_from(plc.clone())?;

                let vs0: Symbolic<<T0 as PartiallySymbolicType>::Type> =
                    SymbolicValue::try_into(v0)?;
                let vs1: Symbolic<<T1 as PartiallySymbolicType>::Type> =
                    SymbolicValue::try_into(v1)?;

                let v0 = vs0.clone().try_into();
                let v1 = vs1.clone().try_into();

                match (v0, v1) {
                    (Ok(v0), Ok(v1)) => {
                        let y = kf(sess, &plc, v0, v1)?;
                        let y: <U as SymbolicType>::Type = y.into();
                        Ok(SymbolicValue::from(y))
                    }
                    _ => match (vs0, vs1) {
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                            let h = sess.add_operation(&op, &[&h0.op, &h1.op], &plc);
                            let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                            Ok(SymbolicValue::from(h))
                        }
                        _ => unimplemented!(),
                    },
                }
            },
        ),
    })
}

pub(crate) fn symbolic_ternary_runtime<T0, T1, T2, U, P>(
    op: Operator,
) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error>,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    T1: PartiallySymbolicType,
    T2: PartiallySymbolicType,
    U: PartiallySymbolicType,

    <T0 as PartiallySymbolicType>::Type: Placed,
    <T1 as PartiallySymbolicType>::Type: Placed,
    <T2 as PartiallySymbolicType>::Type: Placed,
    <U as PartiallySymbolicType>::Type: Placed<Placement = P>,

    SymbolicValue: TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: TryInto<Symbolic<<T1 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: TryInto<Symbolic<<T2 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(crate::kernels::NgKernel::Ternary {
        closure: Box::new(
            move |sess: &crate::execution::SymbolicSession,
                  plc: &crate::computation::Placement,
                  v0: crate::computation::SymbolicValue,
                  v1: crate::computation::SymbolicValue,
                  v2: crate::computation::SymbolicValue| {
                let plc = P::try_from(plc.clone())?;
                let v0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;
                let v1: <T1 as SymbolicType>::Type = SymbolicValue::try_into(v1)?;
                let v2: <T2 as SymbolicType>::Type = SymbolicValue::try_into(v2)?;

                match (v0, v1, v2) {
                    (Symbolic::Symbolic(x0), Symbolic::Symbolic(x1), Symbolic::Symbolic(x2)) => {
                        let h = sess.add_operation(&op, &[&x0.op, &x1.op, &x2.op], &plc);
                        let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                        Ok(SymbolicValue::from(h))
                    }
                    _ => unimplemented!(),
                }
            },
        ),
    })
}

pub(crate) fn symbolic_ternary_concrete<T0, T1, T2, U, P>(
    op: Operator,
    kf: fn(
        &SymbolicSession,
        &P,
        <T0 as PartiallySymbolicType>::Type,
        <T1 as PartiallySymbolicType>::Type,
        <T2 as PartiallySymbolicType>::Type,
    ) -> Result<<U as PartiallySymbolicType>::Type>,
) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    T1: PartiallySymbolicType,
    T2: PartiallySymbolicType,
    U: PartiallySymbolicType, // TODO use SymbolicType here?

    <T0 as PartiallySymbolicType>::Type: Placed + 'static,
    <T1 as PartiallySymbolicType>::Type: Placed + 'static,
    <T2 as PartiallySymbolicType>::Type: Placed + 'static,
    <U as PartiallySymbolicType>::Type: Placed + 'static,
    // TODO(Morten) shouldn't need this, we should have Placed<Placement = P> wrt U
    <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,

    SymbolicValue: TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: TryInto<Symbolic<<T1 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: TryInto<Symbolic<<T2 as PartiallySymbolicType>::Type>, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Ternary {
        closure: Box::new(
            move |sess: &SymbolicSession,
                  plc: &Placement,
                  v0: SymbolicValue,
                  v1: SymbolicValue,
                  v2: SymbolicValue| {
                let plc = P::try_from(plc.clone())?;
                let v0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;
                let v1: <T1 as SymbolicType>::Type = SymbolicValue::try_into(v1)?;
                let v2: <T2 as SymbolicType>::Type = SymbolicValue::try_into(v2)?;

                match (v0, v1, v2) {
                    (Symbolic::Concrete(x0), Symbolic::Concrete(x1), Symbolic::Concrete(x2)) => {
                        let y = kf(sess, &plc, x0, x1, x2)?;
                        Ok(SymbolicValue::from(Symbolic::Concrete(y)))
                    }
                    (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
                        let h = sess.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc);
                        let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                        Ok(SymbolicValue::from(h))
                    }
                    _ => Err(crate::error::Error::Unexpected(Some(
                        "Mixed symbolic and concrete value during compilation".to_string(),
                    ))),
                }
            },
        ),
    })
}

pub(crate) fn symbolic_ternary_hybrid<T0, T1, T2, U, X0, X1, X2, Y, P>(
    op: Operator,
    kf: fn(&SymbolicSession, &P, X0, X1, X2) -> Result<Y>,
) -> Result<NgKernel<SymbolicSession>>
where
    P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
    Placement: From<P>,

    T0: PartiallySymbolicType,
    T1: PartiallySymbolicType,
    T2: PartiallySymbolicType,
    U: PartiallySymbolicType,

    Symbolic<<T0 as PartiallySymbolicType>::Type>: Clone + TryInto<X0>,
    Symbolic<<T1 as PartiallySymbolicType>::Type>: Clone + TryInto<X1>,
    Symbolic<<T2 as PartiallySymbolicType>::Type>: Clone + TryInto<X2>,
    Y: Into<<U as SymbolicType>::Type>,

    X0: 'static,
    X1: 'static,
    X2: 'static,
    Y: 'static,

    <T0 as PartiallySymbolicType>::Type: Placed + 'static,
    <T1 as PartiallySymbolicType>::Type: Placed + 'static,
    <T2 as PartiallySymbolicType>::Type: Placed + 'static,
    <U as PartiallySymbolicType>::Type: Placed + 'static,
    // TODO(Morten) shouldn't need this, we should have Placed<Placement = P> wrt U
    <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,

    SymbolicValue: TryInto<<T0 as SymbolicType>::Type, Error = crate::Error>,
    SymbolicValue: TryInto<<T1 as SymbolicType>::Type, Error = crate::Error>,
    SymbolicValue: TryInto<<T2 as SymbolicType>::Type, Error = crate::Error>,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Ternary {
        closure: Box::new(
            move |sess: &SymbolicSession,
                  plc: &Placement,
                  v0: SymbolicValue,
                  v1: SymbolicValue,
                  v2: SymbolicValue| {
                let plc = P::try_from(plc.clone())?;

                let vs0: Symbolic<<T0 as PartiallySymbolicType>::Type> =
                    SymbolicValue::try_into(v0)?;
                let vs1: Symbolic<<T1 as PartiallySymbolicType>::Type> =
                    SymbolicValue::try_into(v1)?;
                let vs2: Symbolic<<T2 as PartiallySymbolicType>::Type> =
                    SymbolicValue::try_into(v2)?;

                let v0 = vs0.clone().try_into();
                let v1 = vs1.clone().try_into();
                let v2 = vs2.clone().try_into();

                match (v0, v1, v2) {
                    (Ok(v0), Ok(v1), Ok(v2)) => {
                        let y = kf(sess, &plc, v0, v1, v2)?;
                        let y: <U as SymbolicType>::Type = y.into();
                        Ok(SymbolicValue::from(y))
                    }
                    _ => match (vs0, vs1, vs2) {
                        (
                            Symbolic::Symbolic(h0),
                            Symbolic::Symbolic(h1),
                            Symbolic::Symbolic(h2),
                        ) => {
                            let h = sess.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc);
                            let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                            Ok(SymbolicValue::from(h))
                        }
                        _ => unimplemented!(),
                    },
                }
            },
        ),
    })
}

pub(crate) fn _symbolic_ternary_transparent_fn<T0, T1, T2, U, P>(
    _op: Operator,
    kf: fn(
        &SymbolicSession,
        &P,
        <T0 as SymbolicType>::Type,
        <T1 as SymbolicType>::Type,
        <T2 as SymbolicType>::Type,
    ) -> Result<<U as SymbolicType>::Type>,
) -> Result<NgKernel<SymbolicSession>>
where
    P: 'static,
    Placement: TryInto<P, Error = crate::Error>,

    T0: PartiallySymbolicType,
    <T0 as PartiallySymbolicType>::Type: Placed,
    <T0 as PartiallySymbolicType>::Type: 'static,
    SymbolicValue: TryInto<<T0 as SymbolicType>::Type, Error = crate::Error>,

    T1: PartiallySymbolicType,
    <T1 as PartiallySymbolicType>::Type: Placed,
    <T1 as PartiallySymbolicType>::Type: 'static,
    SymbolicValue: TryInto<<T1 as SymbolicType>::Type, Error = crate::Error>,

    T2: PartiallySymbolicType,
    <T2 as PartiallySymbolicType>::Type: Placed,
    <T2 as PartiallySymbolicType>::Type: 'static,
    SymbolicValue: TryInto<<T2 as SymbolicType>::Type, Error = crate::Error>,

    U: PartiallySymbolicType,
    <U as PartiallySymbolicType>::Type: Placed<Placement = P>,
    <U as PartiallySymbolicType>::Type: 'static,
    SymbolicValue: From<<U as SymbolicType>::Type>,
{
    Ok(NgKernel::Ternary {
        closure: Box::new(
            move |sess: &SymbolicSession,
                  plc: &Placement,
                  v0: SymbolicValue,
                  v1: SymbolicValue,
                  v2: SymbolicValue| {
                let plc: P = Placement::try_into(plc.clone())?;
                let x0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;
                let x1: <T1 as SymbolicType>::Type = SymbolicValue::try_into(v1)?;
                let x2: <T2 as SymbolicType>::Type = SymbolicValue::try_into(v2)?;
                let y: <U as SymbolicType>::Type = kf(sess, &plc, x0, x1, x2)?;
                Ok(SymbolicValue::from(y))
            },
        ),
    })
}
