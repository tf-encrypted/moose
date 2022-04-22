use crate::computation::{
    Operator, PartiallySymbolicType, Placed, Placement, SymbolicType, SymbolicValue, Value,
};
use crate::error::Result;
use crate::execution::symbolic::{Symbolic, SymbolicSession};
use crate::execution::{Operands, Session};
use crate::kernels::NgKernel;
use std::convert::{TryFrom, TryInto};

pub(crate) fn nullary<S, U, P, F>(kf: F) -> Result<NgKernel<S, Value>>
where
    F: Fn(&S, &P) -> Result<U> + Send + Sync + 'static,
    S: Session + 'static,
    P: TryFrom<Placement, Error = crate::Error> + 'static,
    U: Into<Value> + 'static,
{
    Ok(NgKernel::Nullary {
        closure: Box::new(move |sess: &S, plc: &Placement| {
            let plc = P::try_from(plc.clone())?;
            let y = kf(sess, &plc)?;
            Ok(y.into())
        }),
    })
}

pub(crate) fn unary<S, T0, U, P, F>(kf: F) -> Result<NgKernel<S, Value>>
where
    F: Fn(&S, &P, T0) -> Result<U> + Send + Sync + 'static,
    S: Session + 'static,
    P: TryFrom<Placement, Error = crate::Error> + 'static,
    T0: TryFrom<Value, Error = crate::Error> + 'static,
    U: Into<Value> + 'static,
{
    Ok(NgKernel::Unary {
        closure: Box::new(move |sess: &S, plc: &Placement, v0: Value| {
            let plc = P::try_from(plc.clone())?;
            let x0 = T0::try_from(v0)?;
            let y = kf(sess, &plc, x0)?;
            Ok(y.into())
        }),
    })
}

pub(crate) fn binary<S, T0, T1, U, P, F>(kf: F) -> Result<NgKernel<S, Value>>
where
    F: Fn(&S, &P, T0, T1) -> Result<U> + Send + Sync + 'static,
    S: Session + 'static,
    P: TryFrom<Placement, Error = crate::Error> + 'static,
    T0: TryFrom<Value, Error = crate::Error> + 'static,
    T1: TryFrom<Value, Error = crate::Error> + 'static,
    U: Into<Value> + 'static,
{
    Ok(NgKernel::Binary {
        closure: Box::new(move |sess: &S, plc: &Placement, v0: Value, v1: Value| {
            let plc = P::try_from(plc.clone())?;
            let x0 = T0::try_from(v0)?;
            let x1 = T1::try_from(v1)?;
            let y: U = kf(sess, &plc, x0, x1)?;
            Ok(y.into())
        }),
    })
}

pub(crate) fn ternary<S, T0, T1, T2, U, P, F>(kf: F) -> Result<NgKernel<S, Value>>
where
    F: Fn(&S, &P, T0, T1, T2) -> Result<U> + Send + Sync + 'static,
    S: Session + 'static,
    P: TryFrom<Placement, Error = crate::Error> + 'static,
    T0: TryFrom<Value, Error = crate::Error> + 'static,
    T1: TryFrom<Value, Error = crate::Error> + 'static,
    T2: TryFrom<Value, Error = crate::Error> + 'static,
    U: Into<Value> + 'static,
{
    Ok(NgKernel::Ternary {
        closure: Box::new(
            move |sess: &S, plc: &Placement, v0: Value, v1: Value, v2: Value| {
                let plc = P::try_from(plc.clone())?;
                let x0 = T0::try_from(v0)?;
                let x1 = T1::try_from(v1)?;
                let x2 = T2::try_from(v2)?;
                let y = kf(sess, &plc, x0, x1, x2)?;
                Ok(y.into())
            },
        ),
    })
}

pub(crate) fn variadic<S, TS, U, P, F>(kf: F) -> Result<NgKernel<S, Value>>
where
    F: Fn(&S, &P, &[TS]) -> Result<U> + Send + Sync + 'static,
    S: Session + 'static,
    P: TryFrom<Placement, Error = crate::Error> + 'static,
    TS: TryFrom<Value, Error = crate::Error> + 'static,
    U: Into<Value> + 'static,
{
    Ok(NgKernel::Variadic {
        closure: Box::new(move |sess: &S, plc: &Placement, vs: Operands<Value>| {
            let plc = P::try_from(plc.clone())?;
            let xs: Operands<TS> = vs
                .into_iter()
                .map(|xi| TS::try_from(xi.clone()))
                .collect::<Result<_>>()?;
            let y = kf(sess, &plc, &xs)?;
            Ok(y.into())
        }),
    })
}

pub(crate) mod symbolic {
    use super::*;

    pub(crate) mod runtime {
        use super::*;

        pub(crate) fn nullary<U, P>(
            op: Operator,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            P: TryFrom<Placement, Error = crate::Error> + Clone,
            U: PartiallySymbolicType,
            <U as PartiallySymbolicType>::Type: Placed<Placement = P>,
            SymbolicValue: From<Symbolic<<U as PartiallySymbolicType>::Type>>,
            Placement: From<P>,
        {
            Ok(NgKernel::Nullary {
                closure: Box::new(move |sess: &SymbolicSession, plc: &Placement| {
                    let plc = P::try_from(plc.clone())?;
                    let h = sess.add_operation(&op, &[], &plc);
                    let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                    Ok(SymbolicValue::from(h))
                }),
            })
        }

        pub(crate) fn unary<T0, U, P>(
            op: Operator,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            P: Clone + TryFrom<Placement, Error = crate::Error>,
            Placement: From<P>,
            T0: PartiallySymbolicType,
            U: PartiallySymbolicType,
            <T0 as PartiallySymbolicType>::Type: Placed,
            <U as PartiallySymbolicType>::Type: Placed<Placement = P>,
            SymbolicValue:
                TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
            SymbolicValue: From<<U as SymbolicType>::Type>,
        {
            Ok(crate::kernels::NgKernel::Unary {
                closure: Box::new(
                    move |sess: &SymbolicSession, plc: &Placement, v0: SymbolicValue| {
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

        pub(crate) fn binary<T0, T1, U, P>(
            op: Operator,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            P: Clone + TryFrom<Placement, Error = crate::Error>,
            Placement: From<P>,

            T0: PartiallySymbolicType,
            T1: PartiallySymbolicType,
            U: PartiallySymbolicType,

            <T0 as PartiallySymbolicType>::Type: Placed,
            <T1 as PartiallySymbolicType>::Type: Placed,
            <U as PartiallySymbolicType>::Type: Placed<Placement = P>,

            SymbolicValue:
                TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
            SymbolicValue:
                TryInto<Symbolic<<T1 as PartiallySymbolicType>::Type>, Error = crate::Error>,
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

        pub(crate) fn ternary<T0, T1, T2, U, P>(
            op: Operator,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
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

            SymbolicValue:
                TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
            SymbolicValue:
                TryInto<Symbolic<<T1 as PartiallySymbolicType>::Type>, Error = crate::Error>,
            SymbolicValue:
                TryInto<Symbolic<<T2 as PartiallySymbolicType>::Type>, Error = crate::Error>,
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
                            (
                                Symbolic::Symbolic(x0),
                                Symbolic::Symbolic(x1),
                                Symbolic::Symbolic(x2),
                            ) => {
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

        pub(crate) fn variadic<TS, U, P>(
            op: Operator,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            P: Clone + TryFrom<Placement, Error = crate::Error>,
            Placement: From<P>,

            TS: PartiallySymbolicType,
            U: PartiallySymbolicType,

            <TS as PartiallySymbolicType>::Type: Placed,
            <U as PartiallySymbolicType>::Type: Placed<Placement = P>,

            SymbolicValue:
                TryInto<Symbolic<<TS as PartiallySymbolicType>::Type>, Error = crate::Error>,
            SymbolicValue: From<<U as SymbolicType>::Type>,
        {
            Ok(crate::kernels::NgKernel::Variadic {
                closure: Box::new(
                    move |sess: &SymbolicSession, plc: &Placement, vs: Operands<SymbolicValue>| {
                        let plc = P::try_from(plc.clone())?;
                        let ts_vals: Operands<<TS as SymbolicType>::Type> = vs
                            .iter()
                            .map(|xi| SymbolicValue::try_into(xi.clone()))
                            .collect::<Result<_>>()?;

                        let kernel_vals: Vec<_> = ts_vals
                            .iter()
                            .filter_map(Symbolic::symbolic_handle)
                            .map(|v| v.op.as_str())
                            .collect();

                        if kernel_vals.len() == vs.len() {
                            let h = sess.add_operation(&op, &kernel_vals, &plc);
                            let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                            Ok(SymbolicValue::from(h))
                        } else {
                            unimplemented!()
                        }
                    },
                ),
            })
        }
    }

    pub(crate) mod concrete {
        use super::*;

        pub(crate) fn nullary<U, P, F>(kf: F) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            F: Fn(&SymbolicSession, &P) -> Result<<U as PartiallySymbolicType>::Type>
                + Send
                + Sync
                + 'static,
            P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
            Placement: From<P>,
            U: PartiallySymbolicType,
            <U as PartiallySymbolicType>::Type: Placed + 'static,
            <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,
            SymbolicValue: From<<U as SymbolicType>::Type>,
        {
            Ok(NgKernel::Nullary {
                closure: Box::new(move |sess: &SymbolicSession, plc: &Placement| {
                    let plc = P::try_from(plc.clone())?;
                    let y = kf(sess, &plc)?;
                    Ok(SymbolicValue::from(Symbolic::Concrete(y)))
                }),
            })
        }

        pub(crate) fn unary<T0, U, P, F>(
            op: Operator,
            kf: F,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            F: Fn(
                    &SymbolicSession,
                    &P,
                    <T0 as PartiallySymbolicType>::Type,
                ) -> Result<<U as PartiallySymbolicType>::Type>
                + Send
                + Sync
                + 'static,
            P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
            Placement: From<P>,
            T0: PartiallySymbolicType,
            U: PartiallySymbolicType,
            <T0 as PartiallySymbolicType>::Type: Placed + 'static,
            <U as PartiallySymbolicType>::Type: Placed + 'static,
            // TODO(Morten) shouldn't need this, we should have Placed<Placement = P> wrt U
            <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,
            SymbolicValue:
                TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
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

        pub(crate) fn binary<T0, T1, U, P, F>(
            op: Operator,
            kf: F,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            F: Fn(
                &SymbolicSession,
                &P,
                <T0 as PartiallySymbolicType>::Type,
                <T1 as PartiallySymbolicType>::Type,
            ) -> Result<<U as PartiallySymbolicType>::Type>,
            F: Send + Sync + 'static,
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
            SymbolicValue:
                TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
            SymbolicValue:
                TryInto<Symbolic<<T1 as PartiallySymbolicType>::Type>, Error = crate::Error>,
            SymbolicValue: From<<U as SymbolicType>::Type>,
        {
            Ok(NgKernel::Binary {
                closure: Box::new(
                    move |sess: &SymbolicSession,
                          plc: &Placement,
                          v0: SymbolicValue,
                          v1: SymbolicValue| {
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

        pub(crate) fn ternary<T0, T1, T2, U, P>(
            op: Operator,
            kf: fn(
                &SymbolicSession,
                &P,
                <T0 as PartiallySymbolicType>::Type,
                <T1 as PartiallySymbolicType>::Type,
                <T2 as PartiallySymbolicType>::Type,
            ) -> Result<<U as PartiallySymbolicType>::Type>,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
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

            SymbolicValue:
                TryInto<Symbolic<<T0 as PartiallySymbolicType>::Type>, Error = crate::Error>,
            SymbolicValue:
                TryInto<Symbolic<<T1 as PartiallySymbolicType>::Type>, Error = crate::Error>,
            SymbolicValue:
                TryInto<Symbolic<<T2 as PartiallySymbolicType>::Type>, Error = crate::Error>,
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
                            (
                                Symbolic::Concrete(x0),
                                Symbolic::Concrete(x1),
                                Symbolic::Concrete(x2),
                            ) => {
                                let y = kf(sess, &plc, x0, x1, x2)?;
                                Ok(SymbolicValue::from(Symbolic::Concrete(y)))
                            }
                            (
                                Symbolic::Symbolic(h0),
                                Symbolic::Symbolic(h1),
                                Symbolic::Symbolic(h2),
                            ) => {
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

        pub(crate) fn variadic<TS, U, P, F>(
            op: Operator,
            kf: F,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            F: Fn(
                    &SymbolicSession,
                    &P,
                    &[<TS as PartiallySymbolicType>::Type],
                ) -> Result<<U as PartiallySymbolicType>::Type>
                + Send
                + Sync
                + 'static,
            P: Clone + TryFrom<Placement, Error = crate::Error> + 'static,
            Placement: From<P>,

            TS: PartiallySymbolicType + Clone,
            U: PartiallySymbolicType, // TODO use SymbolicType here?

            <TS as PartiallySymbolicType>::Type: Placed + 'static + Clone,
            <U as PartiallySymbolicType>::Type: Placed + 'static,
            // TODO(Morten) shouldn't need this, we should have Placed<Placement = P> wrt U
            <<U as PartiallySymbolicType>::Type as Placed>::Placement: From<P>,
            SymbolicValue:
                TryInto<Symbolic<<TS as PartiallySymbolicType>::Type>, Error = crate::Error>,
            SymbolicValue: From<<U as SymbolicType>::Type>,
        {
            Ok(NgKernel::Variadic {
                closure: Box::new(
                    move |sess: &SymbolicSession, plc: &Placement, vs: Operands<SymbolicValue>| {
                        let plc = P::try_from(plc.clone())?;

                        let ts_vals: Vec<<TS as SymbolicType>::Type> = vs
                            .iter()
                            .map(|xi| {
                                let x: Result<<TS as SymbolicType>::Type> =
                                    SymbolicValue::try_into(xi.clone());
                                x
                            })
                            .collect::<Result<_>>()?;

                        let kernel_vals: Operands<_> = ts_vals
                            .iter()
                            .filter_map(|xi| match xi {
                                Symbolic::Concrete(v) => Some(v.clone()),
                                Symbolic::Symbolic(_) => None,
                            })
                            .collect();

                        if kernel_vals.len() == vs.len() {
                            let y = kf(sess, &plc, kernel_vals.as_slice())?;
                            Ok(SymbolicValue::from(Symbolic::Concrete(y)))
                        } else {
                            let handles: Vec<_> = ts_vals
                                .iter()
                                .filter_map(Symbolic::symbolic_handle)
                                .map(|v| v.op.as_str())
                                .collect();
                            if handles.len() == vs.len() {
                                // success; we can record in graph
                                let h = sess.add_operation(&op, &handles, &plc);
                                let h: <U as SymbolicType>::Type = Symbolic::Symbolic(h);
                                Ok(SymbolicValue::from(h))
                            } else {
                                Err(crate::error::Error::Unexpected(Some("Variadic concrete flavor found mixed symbolic and concrete value during compilation.".to_string())))
                            }
                        }
                    },
                ),
            })
        }
    }

    pub(crate) mod transparent {
        use super::*;

        pub(crate) fn unary<T0, U, P, F>(kf: F) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            F: Fn(
                    &SymbolicSession,
                    &P,
                    <T0 as SymbolicType>::Type,
                ) -> Result<<U as SymbolicType>::Type>
                + Send
                + Sync
                + 'static,
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

        pub(crate) fn binary<T0, T1, U, P, F>(
            kf: F,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            F: Fn(
                &SymbolicSession,
                &P,
                <T0 as SymbolicType>::Type,
                <T1 as SymbolicType>::Type,
            ) -> Result<<U as SymbolicType>::Type>,
            F: Send + Sync + 'static,
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
                    move |sess: &SymbolicSession,
                          plc: &Placement,
                          v0: SymbolicValue,
                          v1: SymbolicValue| {
                        let plc: P = Placement::try_into(plc.clone())?;
                        let x0: <T0 as SymbolicType>::Type = SymbolicValue::try_into(v0)?;
                        let x1: <T1 as SymbolicType>::Type = SymbolicValue::try_into(v1)?;
                        let y: <U as SymbolicType>::Type = kf(sess, &plc, x0, x1)?;
                        Ok(SymbolicValue::from(y))
                    },
                ),
            })
        }

        pub(crate) fn _ternary<T0, T1, T2, U, P, F>(
            kf: F,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            F: Fn(
                &SymbolicSession,
                &P,
                <T0 as SymbolicType>::Type,
                <T1 as SymbolicType>::Type,
                <T2 as SymbolicType>::Type,
            ) -> Result<<U as SymbolicType>::Type>,
            F: Send + Sync + 'static,
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

        pub(crate) fn variadic<TS, U, P, F>(
            kf: F,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            F: Fn(
                    &SymbolicSession,
                    &P,
                    &[<TS as SymbolicType>::Type],
                ) -> Result<<U as SymbolicType>::Type>
                + Send
                + Sync
                + 'static,
            P: 'static,
            Placement: TryInto<P, Error = crate::Error>,

            TS: PartiallySymbolicType,
            <TS as PartiallySymbolicType>::Type: Placed,
            <TS as PartiallySymbolicType>::Type: 'static,
            SymbolicValue: TryInto<<TS as SymbolicType>::Type, Error = crate::Error>,

            U: PartiallySymbolicType,
            <U as PartiallySymbolicType>::Type: Placed<Placement = P>,
            <U as PartiallySymbolicType>::Type: 'static,
            SymbolicValue: From<<U as SymbolicType>::Type>,
        {
            Ok(NgKernel::Variadic {
                closure: Box::new(
                    move |sess: &SymbolicSession, plc: &Placement, vs: Operands<SymbolicValue>| {
                        let plc: P = Placement::try_into(plc.clone())?;
                        let ts_vals: Vec<<TS as SymbolicType>::Type> = vs
                            .iter()
                            .map(|xi| {
                                let x: Result<<TS as SymbolicType>::Type> =
                                    SymbolicValue::try_into(xi.clone());
                                x
                            })
                            .collect::<Result<_>>()?;

                        let y: <U as SymbolicType>::Type = kf(sess, &plc, &ts_vals)?;
                        Ok(SymbolicValue::from(y))
                    },
                ),
            })
        }
    }

    pub(crate) mod hybrid {
        use super::*;

        pub(crate) fn unary<T0, U, X0, Y, P, F>(
            op: Operator,
            kf: F,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            F: Fn(&SymbolicSession, &P, X0) -> Result<Y> + Send + Sync + 'static,
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

        pub(crate) fn binary<T0, T1, U, X0, X1, Y, P, F>(
            op: Operator,
            kf: F,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
        where
            F: Fn(&SymbolicSession, &P, X0, X1) -> Result<Y> + Send + Sync + 'static,
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
                    move |sess: &SymbolicSession,
                          plc: &Placement,
                          v0: SymbolicValue,
                          v1: SymbolicValue| {
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

        pub(crate) fn ternary<T0, T1, T2, U, X0, X1, X2, Y, P>(
            op: Operator,
            kf: fn(&SymbolicSession, &P, X0, X1, X2) -> Result<Y>,
        ) -> Result<NgKernel<SymbolicSession, SymbolicValue>>
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
                                    let h =
                                        sess.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc);
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
    }
}
