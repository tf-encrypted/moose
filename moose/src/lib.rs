// Returns the context-specific type for the given basic type.
macro_rules! cs {
    ($t:ty) => {
        <$t as KnownType<S>>::Type
    };
}

macro_rules! st {
    ($t:ty) => {
        <<$t as crate::computation::CanonicalType>::Type as KnownType<S>>::Type
    };
    ($t:ty, $s:ty) => {
        <<$t as crate::computation::CanonicalType>::Type as KnownType<$s>>::Type
    };
}

/// Map a type to its canonical type
///
/// Using this macro requires adding the following trait bound:
///   $t: CanonicalType
///
/// Examples:
/// - c!(RepTen<HostBitTen>) -> RepTen<HostBiTTen>
/// - c!(RepTen<Sym<HostBitTen>>) -> RepTen<HostBiTTen>
/// - c!(Sym<RepTen<Sym<HostBitTen>>>) -> RepTen<HostBiTTen>
macro_rules! c {
    ($t:ty) => {
        <$t as crate::computation::CanonicalType>::Type
    };
}

/// Map a _canonical_ type to its session-specific type
///
/// Using this macro required adding the following trait bound:
///  $t: KnownType<S>
///
/// Note also that this is sometimes useful in conjection with `c!`:
///   m!(c!(RepTen<HostRingT>))
/// which then requires adding trait bounds:
///   $t: CanonicalType
///   <$t as CanonicalType>::Type: KnownType<S>
///
/// Examples:
/// - m!(RepTen<HostBitTen>) in SyncSession -> RepTen<HostBitTen>
/// - m!(RepTen<HostBitTen>) in SymbSession -> Sym<RepTen<Sym<HostBitTen>>>
macro_rules! m {
    ($t:ty) => {
        <$t as KnownType<S>>::Type
    };
}

macro_rules! derive_runtime_kernel {
    (nullary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> crate::error::Result<Box<dyn Fn(&_, &_,) -> _>> = &|$op| $kf;
            kf($self)
        }
    };
    (unary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> crate::error::Result<Box<dyn Fn(&_, &_, _) -> _>> = &|$op| $kf;
            kf($self)
        }
    };
    (binary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> crate::error::Result<Box<dyn Fn(&_, &_, _, _) -> _>> = &|$op| $kf;
            kf($self)
        }
    };
    (ternary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> crate::error::Result<Box<dyn Fn(&_, &_, _, _, _) -> _>> = &|$op| $kf;
            kf($self)
        }
    };

    (variadic, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> Box<dyn Fn(&_, &_, Vec<_>) -> _> = &|$op| $kf;
            kf($self)
        }
    };

    (nullary, attributes[$($attr:ident$(: $prim_ty:ident)?),+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
                // The following block applies the optional Constant type restriction to the attribute and unwraps it
                $(
                    let $attr = match $attr {
                        Constant::$prim_ty(v) => v,
                        _ => return Err(crate::error::Error::TypeMismatch{
                            expected: stringify!($prim_ty).to_string(),
                            found: $attr.ty(),
                        })
                    };
                )?
            )+
            crate::error::Result::<Box<dyn Fn(&_, &_) -> _>>::Ok(Box::new(move |sess, plc| {
                $k(sess, plc, $($attr.clone()),+)
            }))
        }
    };
    (unary, attributes[$($attr:ident$(: $prim_ty:ident)?),+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
                // The following block applies the optional Constant type restriction to the attribute and unwraps it
                $(
                    let $attr = match $attr {
                        Constant::$prim_ty(v) => v,
                        _ => return Err(crate::error::Error::TypeMismatch{
                            expected: stringify!($prim_ty).to_string(),
                            found: $attr.ty(),
                        })
                    };
                )?
            )+
            {
                crate::error::Result::<Box<dyn Fn(&_, &_, _) -> _>>::Ok(
                    Box::new(move |sess, plc, x0| {
                        $k(sess, plc, $($attr.clone()),+, x0)
                    })
                )
            }
        }
    };
    (binary, attributes[$($attr:ident$(: $prim_ty:ident)?),+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
                // The following block applies the optional Constant type restriction to the attribute and unwraps it
                $(
                    let $attr = match $attr {
                        Constant::$prim_ty(v) => v,
                        _ => return Err(crate::error::Error::TypeMismatch{
                            expected: stringify!($prim_ty).to_string(),
                            found: $attr.ty(),
                        })
                    };
                )?
            )+
            crate::error::Result::<Box<dyn Fn(&_, &_, _, _) -> _>>::Ok(Box::new(move |sess, plc, x0, x1| {
                $k(sess, plc, $($attr.clone()),+, x0, x1)
            }))
        }
    };
    (ternary, attributes[$($attr:ident$(: $prim_ty:ident)?),+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
                // The following block applies the optional Constant type restriction to the attribute and unwraps it
                $(
                    let $attr = match $attr {
                        Constant::$prim_ty(v) => v,
                        _ => return Err(crate::error::Error::TypeMismatch{
                            expected: stringify!($prim_ty).to_string(),
                            found: $attr.ty(),
                        })
                    };
                )?
            )+
            crate::error::Result::<Box<dyn Fn(&_, &_, _, _, _) -> crate::error::Result<_>>>::Ok(Box::new(move |sess, plc, x0, x1, x2| {
                $k(sess, plc, $($attr.clone()),+), x0, x1, x2
            }))
        }
    };

    (variadic, attributes[$($attr:ident$(: $prim_ty:ident)?),+] $k:expr, $self:ident) => {
        {
            $(
                let $attr = $self.$attr.clone();
                    // The following block applies the optional Constant type restriction to the attribute and unwraps it
                    $(
                        let $attr = match $attr {
                            Constant::$prim_ty(v) => v,
                            _ => return Err(crate::error::Error::TypeMismatch{
                                expected: stringify!($prim_ty).to_string(),
                                found: $attr.ty(),
                            })
                        };
                    )?
                )+
                {
                    crate::error::Result::<Box<dyn Fn(&_, &_, Vec<_>) -> _>>::Ok(
                        Box::new(move |sess, plc, xs| {
                            $k(sess, plc, $($attr.clone()),+, &xs)
                        })
                    )
                }
        }
    };

    (nullary, $k:expr, $self:ident) => {
        crate::error::Result::<Box<dyn Fn(&_, &_,) -> _>>::Ok(Box::new($k))
    };
    (unary, $k:expr, $self:ident) => {
        crate::error::Result::<Box<dyn Fn(&_, &_, _) -> _>>::Ok(Box::new($k))
    };
    (binary, $k:expr, $self:ident) => {
        crate::error::Result::<Box<dyn Fn(&_, &_, _, _) -> _>>::Ok(Box::new($k))
    };
    (ternary, $k:expr, $self:ident) => {
        crate::error::Result::<Box<dyn Fn(&_, &_, _, _, _) -> _>>::Ok(Box::new($k))
    };
    (variadic, $k:expr, $self:ident) => {
        crate::error::Result::<Box<dyn Fn(&_, &_, Vec<_>) -> _>>::Ok(Box::new($k))
    };
}

macro_rules! concrete_dispatch_kernel {

    /*
    Nullary
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::kernels::SyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<Box<dyn Fn(&crate::kernels::SyncSession, Vec<crate::computation::Value>) -> crate::error::Result<crate::computation::Value>>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, NullarySignature};
                use crate::kernels::{SyncSession, NullaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature{
                                ret: <$u as KnownType<SyncSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;

                            let k = <$op as NullaryKernel<SyncSession, $plc, $u>>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands: Vec<crate::computation::Value>| {
                                assert_eq!(operands.len(), 0);

                                let y: $u = k(sess, &plc)?;
                                debug_assert_eq!(y.placement()?, plc.clone().into());
                                Ok(y.into())
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }
    };

    /*
    Unary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::kernels::SyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<Box<dyn Fn(&crate::kernels::SyncSession, Vec<crate::computation::Value>) -> crate::error::Result<crate::computation::Value>>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, UnarySignature, Value};
                use crate::kernels::{SyncSession, UnaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Unary(UnarySignature {
                                arg0: <$t0 as KnownType<SyncSession>>::TY,
                                ret: <$u as KnownType<SyncSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;

                            let k = <$op as UnaryKernel<SyncSession, $plc, $t0, $u>>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands: Vec<Value>| {
                                assert_eq!(operands.len(), 1);

                                let x0: $t0 = operands.get(0).unwrap().clone().try_into()?;

                                let y: $u = k(sess, &plc, x0)?;
                                debug_assert_eq!(y.placement()?, plc.clone().into());
                                Ok(y.into())
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }
    };

    /*
    Binary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::kernels::SyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<Box<dyn Fn(&crate::kernels::SyncSession, Vec<crate::computation::Value>) -> crate::error::Result<crate::computation::Value>>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, BinarySignature};
                use crate::kernels::{SyncSession, BinaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Binary(BinarySignature{
                                arg0: <$t0 as KnownType<SyncSession>>::TY,
                                arg1: <$t1 as KnownType<SyncSession>>::TY,
                                ret: <$u as KnownType<SyncSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;

                            let k = <$op as BinaryKernel<
                                SyncSession,
                                $plc,
                                $t0,
                                $t1,
                                $u
                            >>::compile(self, &plc)?;

                            Ok(Box::new(
                                move |sess, operands: Vec<crate::computation::Value>| {
                                assert_eq!(operands.len(), 2);

                                let x0: $t0 = operands.get(0).unwrap().clone().try_into()?;
                                let x1: $t1 = operands.get(1).unwrap().clone().try_into()?;

                                let y: $u = k(sess, &plc, x0, x1)?;
                                debug_assert_eq!(y.placement()?, plc.clone().into());
                                Ok(y.into())
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }
    };

    /*
    Ternary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::kernels::SyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<Box<dyn Fn(&crate::kernels::SyncSession, Vec<crate::computation::Value>) -> crate::error::Result<crate::computation::Value>>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, TernarySignature};
                use crate::kernels::{SyncSession, TernaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Ternary(TernarySignature{
                                arg0: <$t0 as KnownType<SyncSession>>::TY,
                                arg1: <$t1 as KnownType<SyncSession>>::TY,
                                arg2: <$t2 as KnownType<SyncSession>>::TY,
                                ret: <$u as KnownType<SyncSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;

                            let k = <$op as TernaryKernel<SyncSession, $plc, $t0, $t1, $t2, $u>>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands| {
                                assert_eq!(operands.len(), 3);

                                let x0: $t0 = operands.get(0).unwrap().clone().try_into()?;
                                let x1: $t1 = operands.get(1).unwrap().clone().try_into()?;
                                let x2: $t2 = operands.get(2).unwrap().clone().try_into()?;

                                let y: $u = k(sess, &plc, x0, x1, x2)?;
                                debug_assert_eq!(y.placement()?, plc.clone().into());
                                Ok(y.into())
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }
    };

    /*
    Variadic
    */

    ($op:ty, [$( ($plc:ty, vec[$ts:ty] -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::kernels::SyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<Box<dyn Fn(&crate::kernels::SyncSession, Vec<crate::computation::Value>) -> crate::error::Result<crate::computation::Value>>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, VariadicSignature, Value};
                use crate::kernels::{SyncSession, VariadicKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Variadic(VariadicSignature {
                                args: <$ts as KnownType<SyncSession>>::TY,
                                ret: <$u as KnownType<SyncSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();

                            let k = <$op as VariadicKernel<SyncSession, $plc, $ts, $u>>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands: Vec<Value>| {
                                let xs: Vec<$ts> = operands.into_iter().map(|xi| xi.try_into().unwrap()).collect();

                                let y: $u = k(sess, &plc, xs)?;
                                debug_assert_eq!(y.placement()?, plc.clone().into());
                                Ok(y.into())
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }
    };
}

macro_rules! symbolic_dispatch_kernel {

    /*
    Nullary
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::symbolic::SymbolicSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                Vec<crate::computation::SymbolicValue>
            ) -> crate::error::Result<crate::computation::SymbolicValue>>> {
                use crate::computation::{KnownPlacement, Signature, NullarySignature, KnownType};
                use crate::kernels::{NullaryKernel};
                use crate::symbolic::SymbolicSession;
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature {
                                ret: <$u as KnownType<SymbolicSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();

                            let k = <$op as NullaryKernel<
                                SymbolicSession,
                                $plc,
                                <$u as KnownType<SymbolicSession>>::Type,
                            >>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands| {
                                assert_eq!(operands.len(), 0);

                                let y: <$u as KnownType<SymbolicSession>>::Type = k(sess, &plc)?;
                                Ok(y.into())
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }
    };

    /*
    Unary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::symbolic::SymbolicSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                Vec<crate::computation::SymbolicValue>
            ) -> crate::error::Result<crate::computation::SymbolicValue>>> {
                use crate::computation::{KnownPlacement, Signature, UnarySignature, KnownType};
                use crate::kernels::{UnaryKernel};
                use crate::symbolic::SymbolicSession;
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Unary(UnarySignature {
                                arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                                ret: <$u as KnownType<SymbolicSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;

                            let k = <$op as UnaryKernel<
                                SymbolicSession,
                                $plc,
                                <$t0 as KnownType<SymbolicSession>>::Type,
                                <$u as KnownType<SymbolicSession>>::Type,
                            >>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands| {
                                assert_eq!(operands.len(), 1);

                                let x0: <$t0 as KnownType<SymbolicSession>>::Type = operands.get(0).unwrap().clone().try_into()?;

                                let y: <$u as KnownType<SymbolicSession>>::Type = k(sess, &plc, x0)?;
                                Ok(y.into())
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }
    };

    // /*
    // Binary
    // */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::symbolic::SymbolicSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                Vec<crate::computation::SymbolicValue>
            ) -> crate::error::Result<crate::computation::SymbolicValue>>> {
                use crate::computation::{KnownPlacement, Signature, BinarySignature, KnownType};
                use crate::kernels::{BinaryKernel};
                use crate::symbolic::SymbolicSession;
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Binary(BinarySignature {
                                arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                                arg1: <$t1 as KnownType<SymbolicSession>>::TY,
                                ret: <$u as KnownType<SymbolicSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;

                            let k = <$op as BinaryKernel<
                                SymbolicSession,
                                $plc,
                                <$t0 as KnownType<SymbolicSession>>::Type,
                                <$t1 as KnownType<SymbolicSession>>::Type,
                                <$u as KnownType<SymbolicSession>>::Type,
                            >>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands| {
                                assert_eq!(operands.len(), 2);

                                let x0: <$t0 as KnownType<SymbolicSession>>::Type = operands.get(0).unwrap().clone().try_into()?;
                                let x1: <$t1 as KnownType<SymbolicSession>>::Type = operands.get(1).unwrap().clone().try_into()?;

                                let y: <$u as KnownType<SymbolicSession>>::Type = k(sess, &plc, x0, x1)?;
                                Ok(y.into())
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }
    };

    // /*
    // Ternary
    // */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::symbolic::SymbolicSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                Vec<crate::computation::SymbolicValue>
            ) -> crate::error::Result<crate::computation::SymbolicValue>>> {
                use crate::computation::{KnownPlacement, Signature, TernarySignature, KnownType};
                use crate::kernels::{TernaryKernel};
                use crate::symbolic::SymbolicSession;
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Ternary(TernarySignature {
                                arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                                arg1: <$t1 as KnownType<SymbolicSession>>::TY,
                                arg2: <$t2 as KnownType<SymbolicSession>>::TY,
                                ret: <$u as KnownType<SymbolicSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;

                            let k = <$op as TernaryKernel<
                                SymbolicSession,
                                $plc,
                                <$t0 as KnownType<SymbolicSession>>::Type,
                                <$t1 as KnownType<SymbolicSession>>::Type,
                                <$t2 as KnownType<SymbolicSession>>::Type,
                                <$u as KnownType<SymbolicSession>>::Type,
                            >>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands| {
                                assert_eq!(operands.len(), 3);

                                let x0: <$t0 as KnownType<SymbolicSession>>::Type = operands.get(0).unwrap().clone().try_into()?;
                                let x1: <$t1 as KnownType<SymbolicSession>>::Type = operands.get(1).unwrap().clone().try_into()?;
                                let x2: <$t2 as KnownType<SymbolicSession>>::Type = operands.get(2).unwrap().clone().try_into()?;

                                let y: <$u as KnownType<SymbolicSession>>::Type = k(sess, &plc, x0, x1, x2)?;
                                Ok(y.into())
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }
    };

    /*
    Variadic
    */

    ($op:ty, [$( ($plc:ty, vec[$ts:ty] -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::symbolic::SymbolicSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            )-> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                Vec<crate::computation::SymbolicValue>
            ) -> crate::error::Result<crate::computation::SymbolicValue>>> {
                use crate::computation::{KnownPlacement, Signature, VariadicSignature, KnownType};
                use crate::kernels::{VariadicKernel};
                use crate::symbolic::SymbolicSession;
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Variadic(VariadicSignature {
                                args: <$ts as KnownType<SymbolicSession>>::TY,
                                ret: <$u as KnownType<SymbolicSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;

                            let k = <$op as VariadicKernel<
                                SymbolicSession,
                                $plc,
                                <$ts as KnownType<SymbolicSession>>::Type,
                                <$u as KnownType<SymbolicSession>>::Type,
                            >>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands| {
                                let xs: Vec<<$ts as KnownType<SymbolicSession>>::Type> = operands.into_iter().map(|xi| xi.try_into().unwrap()).collect();

                                let y: <$u as KnownType<SymbolicSession>>::Type = k(sess, &plc, xs)?;
                                Ok(y.into())
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }
    };
}

/// Macros to define kernels for operators.
///
/// Sample definition would be in this form:
/// `
/// kernel! {
///   MySuperOp,
///   [
///     (HostPlacement, (HostFixed64Tensor, HostFixed64Tensor) -> HostFixed64Tensor => [runtime] attributes[axis, precision] Self::host_kernel),
///     (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor => [hybrid] attributes[axis, precision] Self::rep_kernel),
///   ]
/// }
/// `
///
/// The following kernel flavours are supported. Note that the flavour only
/// affects behaviour in symbolic sessions (and not in eg sync sessions)
/// where it is used to determined whether the kernel function should be called
/// or whether the operation should be added to the graph.
/// - "runtime": never call the kernel function
/// - "hybrid": use TryInto/Into to determine if kernel function should be called
/// - "transparent": always call kernel function
/// - "concrete": call kernel function on Symbolic::Concrete values
macro_rules! kernel {

    /*
    Nullary
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, () -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, () -> $u), )+]);

        $(
            impl crate::kernels::NullaryKernel<
                crate::kernels::SyncSession,
                $plc,
                <$u as crate::computation::KnownType<crate::kernels::SyncSession>>::Type
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<Box<dyn Fn(
                    &crate::kernels::SyncSession,
                    &$plc)
                    -> crate::error::Result<
                        <$u as crate::computation::KnownType<crate::kernels::SyncSession>>::Type>>
                    >
                {
                    derive_runtime_kernel![nullary, $($kp)+, self]
                }
            }
        )+

        $(
            kernel!(__nullary $flavour, $op, $plc, () -> $u => $($kp)+);
        )+
    };

    (__nullary hybrid, $op:ty, $plc:ty, () -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::NullaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc
            ) -> crate::error::Result<
                <$u as KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                use crate::symbolic::SymbolicSession;

                let k = derive_runtime_kernel![nullary, $($kp)+, self]?;

                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                | {
                    let y = k(sess, plc)?;
                    Ok(y.into())
                }))
            }
        }
    };

    (__nullary transparent, $op:ty, $plc:ty, () -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::NullaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc
            ) -> crate::error::Result<<$u as KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                derive_runtime_kernel![nullary, $($kp)+, self]
            }
        }
    };

    (__nullary runtime, $op:ty, $plc:ty, () -> $u:ty => $($kp:tt)+) => {
        impl NullaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc)
                -> crate::error::Result<
                    <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
                >
            >> {
                use crate::symbolic::{SymbolicSession, SymbolicHandle, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                | {
                    let op_name = sess.add_operation(&op, &[], &plc.clone().into());
                    Ok(Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() }))
                }))
            }
        }
    };

    /*
    Unary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0) -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, ($t0) -> $u), )+]);

        $(
            impl crate::kernels::UnaryKernel<
                crate::kernels::SyncSession,
                $plc,
                $t0,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::kernels::SyncSession, &$plc, $t0) -> crate::error::Result<$u>>
                > {
                    derive_runtime_kernel![unary, $($kp)+, self]
                }
            }
        )+

        $(
            kernel!(__unary $flavour, $op, $plc, ($t0) -> $u => $($kp)+);
        )+
    };

    (__unary hybrid, $op:ty, $plc:ty, ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::UnaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                use crate::symbolic::{Symbolic, SymbolicSession, SymbolicHandle};
                use std::convert::TryInto;

                let op = self.clone();

                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![unary, $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)

                    let v0 = x0.clone().try_into();

                    match v0 {
                        Ok(v0) => {
                            let y = k(sess, plc, v0);
                            y.map(|x| x.into())
                        }
                        _ => match x0 {
                            Symbolic::Symbolic(h0) => {
                                let op_name = sess.add_operation(op, &[&h0.op], &plc.clone().into());
                                Ok(Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() }))
                            }
                            _ => unimplemented!() // ok
                        }
                    }
                }))
            }
        }
    };

    (__unary transparent, $op:ty, $plc:ty, ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::UnaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                derive_runtime_kernel![unary, $($kp)+, self]
            }
        }
    };

    (__unary runtime, $op:ty, $plc:ty, ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::UnaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type)
                -> crate::error::Result<
                    <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
                >>>
            {
                use crate::computation::{KnownType};
                use crate::symbolic::{SymbolicSession, SymbolicHandle, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type
                | {
                    match x0 {
                        Symbolic::Symbolic(h0) => {
                            let op_name = sess.add_operation(&op, &[&h0.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() }))
                        }
                        _ => unimplemented!()
                    }
                }))
            }
        }
    };

    /*
    Binary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0, $t1) -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, ($t0, $t1) -> $u), )+]);

        $(
            impl crate::kernels::BinaryKernel<
                crate::kernels::SyncSession,
                $plc,
                $t0,
                $t1,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::kernels::SyncSession, &$plc, $t0, $t1) -> crate::error::Result<$u>>
                > {
                    derive_runtime_kernel![binary, $($kp)+, self]
                }
            }
        )+

        $(
            kernel!(__binary $flavour, $op, $plc, ($t0, $t1) -> $u => $($kp)+);
        )+
    };

    (__binary hybrid, $op:ty, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::BinaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                use crate::symbolic::{Symbolic, SymbolicSession, SymbolicHandle};
                use std::convert::TryInto;

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as KnownType<SymbolicSession>>::Type,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![binary, $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)

                    let v0 = x0.clone().try_into();
                    let v1 = x1.clone().try_into();

                    match (v0, v1) {
                        (Ok(v0), Ok(v1)) => {
                            let y = k(sess, plc, v0, v1)?;
                            Ok(y.into())
                        }
                        _ => match (x0, x1) {
                            (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                                let op_name = sess.add_operation(op, &[&h0.op, &h1.op], &plc.clone().into());
                                Ok(Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() }))
                            }
                            _ => unimplemented!() // ok
                        }
                    }
                }))
            }
        }
    };

    (__binary concrete, $op:ty, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::BinaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                use crate::symbolic::{Symbolic, SymbolicSession, SymbolicHandle};
                use std::convert::TryInto;

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as KnownType<SymbolicSession>>::Type,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![binary, $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)

                    match (x0, x1) {
                        (Symbolic::Concrete(v0), Symbolic::Concrete(v1)) => {
                            let y = k(sess, plc, v0, v1)?;
                            Ok(y)
                        }
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                            let op_name = sess.add_operation(op, &[&h0.op, &h1.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() }))
                        }
                        _ => unimplemented!() // ok
                    }
                }))
            }
        }
    };

    (__binary transparent, $op:ty, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::BinaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                derive_runtime_kernel![binary, $($kp)+, self]
            }
        }
    };

    (__binary runtime, $op:ty, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::BinaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                use crate::computation::{KnownType};
                use crate::symbolic::{SymbolicSession, SymbolicHandle, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as KnownType<SymbolicSession>>::Type
                | {
                    match (x0, x1) {
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                            let op_name = sess.add_operation(&op, &[&h0.op, &h1.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() }))
                        }
                        _ => unimplemented!()
                    }
                }))
            }
        }
    };

    /*
    Ternary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u), )+]);

        $(
            impl crate::kernels::TernaryKernel<
                crate::kernels::SyncSession,
                $plc,
                $t0,
                $t1,
                $t2,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<Box<
                    dyn Fn(&crate::kernels::SyncSession, &$plc, $t0, $t1, $t2) -> crate::error::Result<$u>
                >> {
                    derive_runtime_kernel![ternary, $($kp)+, self]
                }
            }
        )+

        $(
            kernel!(__ternary $flavour, $op, $plc, ($t0, $t1, $t2) -> $u => $($kp)+);
        )+
    };

    (__ternary transparent, $op:ty, $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::TernaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t2 as KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                derive_runtime_kernel![ternary, $($kp)+, self]
            }
        }
    };

    (__ternary hybrid, $op:ty, $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::TernaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t2 as KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                use crate::symbolic::{Symbolic, SymbolicSession, SymbolicHandle};
                use std::convert::TryInto;

                let k = derive_runtime_kernel![ternary, $($kp)+, self]?;

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as KnownType<SymbolicSession>>::Type,
                    x2: <$t2 as KnownType<SymbolicSession>>::Type,
                | {
                    let v0 = x0.clone().try_into();
                    let v1 = x1.clone().try_into();
                    let v2 = x2.clone().try_into();

                    match (v0, v1, v2) {
                        (Ok(v0), Ok(v1), Ok(v2)) => {
                            let y = k(sess, plc, v0, v1, v2)?;
                            Ok(y.into())
                        }
                        _ => match (x0, x1, x2) {
                            (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
                                let op_name = sess.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
                                Ok(Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() }))
                            }
                            _ => unimplemented!() // ok
                        }
                    }
                }))
            }
        }
    };

    (__ternary runtime, $op:ty, $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::TernaryKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t2 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> crate::error::Result<
                <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type>>
            > {
                use crate::computation::{KnownType};
                use crate::symbolic::{SymbolicSession, SymbolicHandle, Symbolic};

                let op = self.clone();
                Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as KnownType<SymbolicSession>>::Type,
                    x2: <$t2 as KnownType<SymbolicSession>>::Type
                | {
                    match (x0, x1, x2) {
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
                            let op_name = sess.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() }))
                        }
                        _ => unimplemented!()
                    }
                })
            }
        }
    };

    /*
    Variadic
    */

    ($op:ty, [$( ($plc:ty, vec[$ts:ty] -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, vec[$ts] -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, vec[$ts] -> $u), )+]);

        $(
            impl crate::kernels::VariadicKernel<
                crate::kernels::SyncSession,
                $plc,
                $ts,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<Box<
                    dyn Fn(&crate::kernels::SyncSession, &$plc, Vec<$ts>) -> crate::error::Result<$u>
                >> {
                    derive_runtime_kernel![variadic, $($kp)+, self]
                }
            }
        )+

        $(
            kernel!(__variadic $flavour, $op, $plc, vec[$ts] -> $u => $($kp)+);
        )+
    };

    (__variadic transparent, $op:ty, $plc:ty, vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::VariadicKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                Vec<<$ts as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type>
            ) -> crate::error::Result<<$u as KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                derive_runtime_kernel![variadic, $($kp)+, self]
            }
        }
    };

    (__variadic hybrid, $op:ty, $plc:ty, vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::VariadicKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                Vec<<$ts as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type>
            ) -> crate::error::Result<<$u as KnownType<crate::symbolic::SymbolicSession>>::Type>>>
            {
                use crate::symbolic::{Symbolic, SymbolicSession, SymbolicHandle};
                use std::convert::TryInto;

                let op = self.clone();

                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    xs: Vec<<$ts as KnownType<SymbolicSession>>::Type>,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![variadic, $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)

                    // attempt to convert operands to match kernel
                    let kernel_vals: Vec<_> = xs.iter().cloned().filter_map(|x| x.try_into().ok()).collect();
                    if kernel_vals.len() == xs.len() {
                        // success; we can apply kernel
                        let y = k(sess, plc, kernel_vals)?;
                        Ok(y.into())
                    } else {
                        // operands did not match kernel so record in graph instead
                        let handles: Vec<_> = xs.iter().filter_map(Symbolic::symbolic_handle).map(|h| h.op.as_str()).collect();
                        if handles.len() == xs.len() {
                            // success; we can record in graph
                            let op_name = sess.add_operation(op, &handles, &plc.clone().into());
                            return Ok(Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() }));
                        } else {
                            // unexpected
                            unimplemented!()
                        }
                    }
                }))
            }
        }
    };

    (__variadic runtime, $op:ty, $plc:ty, vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::VariadicKernel<
            crate::symbolic::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::symbolic::SymbolicSession,
                &$plc,
                Vec<<$ts as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type>
            ) -> crate::error::Result<
                <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type>>
            > {
                use crate::computation::{KnownType};
                use crate::symbolic::{SymbolicSession, SymbolicHandle, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    xs: Vec<<$ts as KnownType<SymbolicSession>>::Type>
                | {
                    let res: Vec<&str> = xs.iter().filter_map(|x| {
                        match x {
                            Symbolic::Symbolic(h0) => {
                                Some(&h0.op[..])
                            }
                            _ => None
                        }
                    }).collect();

                    if  res.len() == xs.len() {
                        let op_name = sess.add_operation(&op, &res, &plc.clone().into());
                        return Ok(Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() }));
                    }

                    unimplemented!()
                }))
            }
        }
    };
}

macro_rules! modelled {
    /*
    Nullary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? () -> $u:ty, $op:ident) => {
        impl crate::kernels::NullaryKernelCheck<crate::kernels::SyncSession, $plc, $u> for $op {}

        impl $t<crate::kernels::SyncSession, $u> for $plc {
            fn $f(&self, sess: &crate::kernels::SyncSession, $($($attr_id:$attr_ty),*)?) -> $u {
                use crate::computation::{KnownType, NullarySignature};
                use crate::kernels::{Session, SyncSession};
                use std::convert::TryInto;

                let sig = NullarySignature {
                    ret: <$u as KnownType<SyncSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                sess.execute(op.into(), &self.into(), vec![])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }

        impl crate::kernels::NullaryKernelCheck<
            crate::symbolic::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
        > for $op {}

        impl $t<
            crate::symbolic::SymbolicSession,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $plc {
            fn $f(
                &self,
                sess: &crate::symbolic::SymbolicSession,
                $($($attr_id:$attr_ty),*)?
            ) -> <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type {
                use crate::computation::{KnownType, NullarySignature};
                use crate::kernels::{Session};
                use crate::symbolic::{SymbolicSession};
                use std::convert::TryInto;

                let sig = NullarySignature {
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                sess.execute(op.into(), &self.into(), vec![])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    /*
    Unary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty) -> $u:ty, $op:ident) => {
        impl crate::kernels::UnaryKernelCheck<crate::kernels::SyncSession, $plc, $t0, $u> for $op {}

        impl $t<
            crate::kernels::SyncSession,
            $t0,
            $u
        > for $plc {
            fn $f(
                &self,
                sess: &crate::kernels::SyncSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &$t0
            ) -> $u {
                use crate::computation::{KnownType, UnarySignature};
                use crate::kernels::{Session, SyncSession};
                use std::convert::TryInto;

                let sig = UnarySignature {
                    arg0: <$t0 as KnownType<SyncSession>>::TY,
                    ret: <$u as KnownType<SyncSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                sess.execute(op.into(), &self.into(), vec![x0.clone().into()])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }

        impl crate::kernels::UnaryKernelCheck<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
        > for $op {}

        impl $t<
            crate::symbolic::SymbolicSession,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $plc {
            fn $f(
                &self,
                sess: &crate::symbolic::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type {
                use crate::computation::{KnownType, UnarySignature};
                use crate::kernels::{Session};
                use crate::symbolic::{SymbolicSession};
                use std::convert::TryInto;

                let sig = UnarySignature {
                    arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                sess.execute(op.into(), &self.into(), vec![x0.clone().into()])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    /*
    Binary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty, $t1:ty) -> $u:ty, $op:ident) => {
        impl crate::kernels::BinaryKernelCheck<crate::kernels::SyncSession, $plc, $t0, $t1, $u> for $op {}

        impl $t<crate::kernels::SyncSession, $t0, $t1, $u> for $plc {
            fn $f(&self, sess: &crate::kernels::SyncSession, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1) -> $u {
                use crate::computation::{KnownType, BinarySignature};
                use crate::kernels::{Session, SyncSession};
                use std::convert::TryInto;

                let sig = BinarySignature {
                    arg0: <$t0 as KnownType<SyncSession>>::TY,
                    arg1: <$t1 as KnownType<SyncSession>>::TY,
                    ret: <$u as KnownType<SyncSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                sess.execute(
                    op.into(),
                    &self.into(),
                    vec![x0.clone().into(), x1.clone().into()],
                )
                .unwrap()
                .try_into()
                .unwrap()
            }
        }

        impl crate::kernels::BinaryKernelCheck<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
        > for $op {}

        impl $t<
            crate::symbolic::SymbolicSession,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $plc {
            fn $f(
                &self,
                sess: &crate::symbolic::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                x1: &<$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type {
                use crate::computation::{KnownType, BinarySignature};
                use crate::kernels::{Session};
                use crate::symbolic::{SymbolicSession};
                use std::convert::TryInto;

                let sig = BinarySignature {
                    arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                    arg1: <$t1 as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                sess.execute(op.into(), &self.into(), vec![x0.clone().into(), x1.clone().into()])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    /*
    Ternary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $op:ident) => {
        impl crate::kernels::TernaryKernelCheck<crate::kernels::SyncSession, $plc, $t0, $t1, $t2, $u> for $op {}

        impl $t<crate::kernels::SyncSession, $t0, $t1, $t2, $u> for $plc {
            fn $f(&self, sess: &crate::kernels::SyncSession, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1, x2: &$t2) -> $u {
                use crate::computation::{KnownType, TernarySignature};
                use crate::kernels::{Session, SyncSession};
                use std::convert::TryInto;

                let sig = TernarySignature {
                    arg0: <$t0 as KnownType<SyncSession>>::TY,
                    arg1: <$t1 as KnownType<SyncSession>>::TY,
                    arg2: <$t2 as KnownType<SyncSession>>::TY,
                    ret: <$u as KnownType<SyncSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                sess.execute(
                    op.into(),
                    &self.into(),
                    vec![x0.clone().into(), x1.clone().into(), x2.clone().into()],
                )
                .unwrap()
                .try_into()
                .unwrap()
            }
        }

        impl crate::kernels::TernaryKernelCheck<
            crate::symbolic::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
        > for $op {}

        impl $t<
            crate::symbolic::SymbolicSession,
            <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $plc {
            fn $f(
                &self,
                sess: &crate::symbolic::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                x1: &<$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                x2: &<$t2 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
            ) -> <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type {
                use crate::computation::{KnownType, TernarySignature};
                use crate::kernels::{Session};
                use crate::symbolic::{SymbolicSession};
                use std::convert::TryInto;

                let sig = TernarySignature {
                    arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                    arg1: <$t1 as KnownType<SymbolicSession>>::TY,
                    arg2: <$t2 as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                sess.execute(op.into(), &self.into(), vec![x0.clone().into(), x1.clone().into(), x2.clone().into()])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };
    /*
    Variadic
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? vec[$ts:ty] -> $u:ty, $op:ident) => {
        impl crate::kernels::VariadicKernelCheck<crate::kernels::SyncSession, $plc, $ts, $u> for $op {}

        impl $t<
            crate::kernels::SyncSession,
            $ts,
            $u
        > for $plc {
            fn $f(
                &self,
                sess: &crate::kernels::SyncSession,
                $($($attr_id:$attr_ty),*,)?
                xs: &[$ts]
            ) -> $u {
                use crate::computation::{KnownType, VariadicSignature};
                use crate::kernels::{Session, SyncSession};
                use std::convert::TryInto;

                let sig = VariadicSignature {
                    args: <$ts as KnownType<SyncSession>>::TY,
                    ret: <$u as KnownType<SyncSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                let vs: Vec<Value> = xs.iter().map(|x| x.clone().into()).collect();
                sess.execute(op.into(), &self.into(), vs)
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }

        impl crate::kernels::VariadicKernelCheck<
            crate::symbolic::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
        > for $op {}

        impl $t<
            crate::symbolic::SymbolicSession,
            <$ts as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type
        > for $plc {
            fn $f(
                &self,
                sess: &crate::symbolic::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                xs: &[<$ts as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type]
            ) -> <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type {
                use crate::computation::{KnownType, VariadicSignature};
                use crate::kernels::{Session};
                use crate::symbolic::{SymbolicSession};
                use std::convert::TryInto;

                let sig = VariadicSignature {
                    args: <$ts as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                let vs: Vec<SymbolicValue> = xs.iter().map(|x| x.clone().into()).collect();
                sess.execute(op.into(), &self.into(), vs)
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };
}

macro_rules! modelled_alias {
    /*
    Binary
    */
    ($src_t:ident::$src_f:ident, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $dst_t:ident::$dst_f:ident) => {
        impl $src_t<crate::kernels::SyncSession, $t0, $t1, $u> for $plc {
            fn $src_f(&self, sess: &crate::kernels::SyncSession, x0: &$t0, x1: &$t1) -> $u {
                $dst_t::$dst_f(self, sess, x0, x1)
            }
        }

        impl
            $src_t<
                crate::symbolic::SymbolicSession,
                <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            > for $plc
        {
            fn $src_f(
                &self,
                ctx: &crate::symbolic::SymbolicSession,
                x0: &<$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                x1: &<$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            ) -> <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type {
                $dst_t::$dst_f(self, ctx, x0, x1)
            }
        }
    };

    /*
    Ternary
    */
    ($src_t:ident::$src_f:ident, $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $dst_t:ident::$dst_f:ident) => {
        impl $src_t<crate::kernels::SyncSession, $t0, $t1, $t2, $u> for $plc {
            fn $src_f(
                &self,
                sess: &crate::kernels::SyncSession,
                x0: &$t0,
                x1: &$t1,
                x2: &$t2,
            ) -> $u {
                $dst_t::$dst_f(self, sess, x0, x1, x2)
            }
        }

        impl
            $src_t<
                crate::symbolic::SymbolicSession,
                <$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$t2 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            > for $plc
        {
            fn $src_f(
                &self,
                ctx: &crate::symbolic::SymbolicSession,
                x0: &<$t0 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                x1: &<$t1 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
                x2: &<$t2 as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type,
            ) -> <$u as crate::computation::KnownType<crate::symbolic::SymbolicSession>>::Type {
                $dst_t::$dst_f(self, ctx, x0, x1, x2)
            }
        }
    };
}

macro_rules! moose_type {
    // Use this for unparameterised types that are already defined
    ($atomic:ident) => {
        impl crate::computation::SymbolicType for $atomic {
            type Type = crate::symbolic::Symbolic<$atomic>;
        }

        impl crate::computation::CanonicalType for $atomic {
            type Type = $atomic;
        }

        impl crate::computation::CanonicalType for crate::symbolic::Symbolic<$atomic> {
            type Type = $atomic;
        }
    };

    // Use this for undefined parameterised types that may be wrapping non-Moose types
    ($combined:ident = [atomic] $t:ty) => {
        pub type $combined = $t;

        impl crate::computation::SymbolicType for $combined {
            type Type = crate::symbolic::Symbolic<$combined>;
        }

        impl crate::computation::CanonicalType for $combined {
            type Type = $combined;
        }

        impl crate::computation::CanonicalType for crate::symbolic::Symbolic<$combined> {
            type Type = $combined;
        }
    };

    // Use this for undefined parameterised types that are wrapping a single Moose types
    ($combined:ident = $outer:ident<$inner:ident>) => {
        pub type $combined = $outer<$inner>;

        impl crate::computation::SymbolicType for $outer<$inner> {
            type Type = crate::symbolic::Symbolic<
                $outer<<$inner as crate::computation::SymbolicType>::Type>,
            >;
        }

        impl crate::computation::CanonicalType for $outer<$inner> {
            type Type = $outer<<$inner as crate::computation::CanonicalType>::Type>;
        }

        impl crate::computation::CanonicalType
            for $outer<<$inner as crate::computation::SymbolicType>::Type>
        {
            type Type = $outer<<$inner as crate::computation::CanonicalType>::Type>;
        }

        impl crate::computation::CanonicalType
            for crate::symbolic::Symbolic<
                $outer<<$inner as crate::computation::SymbolicType>::Type>,
            >
        {
            type Type = $outer<<$inner as crate::computation::CanonicalType>::Type>;
        }

        // The kernel macro uses this to map (partially) concrete outputs to symbolic values
        impl From<$outer<<$inner as crate::computation::SymbolicType>::Type>>
            for <$combined as crate::computation::SymbolicType>::Type
        {
            fn from(x: $outer<<$inner as crate::computation::SymbolicType>::Type>) -> Self {
                crate::symbolic::Symbolic::Concrete(x)
            }
        }

        // The kernel macros uses this to determine whether to invoke kernels, and
        // if so, to map symbolic values to (partially) concrete inputs
        impl TryFrom<<$combined as crate::computation::SymbolicType>::Type>
            for $outer<<$inner as crate::computation::SymbolicType>::Type>
        {
            type Error = crate::error::Error;

            fn try_from(
                v: <$combined as crate::computation::SymbolicType>::Type,
            ) -> crate::error::Result<Self> {
                match v {
                    crate::symbolic::Symbolic::Concrete(x) => Ok(x),
                    _ => Err(crate::error::Error::Unexpected(None)), // TODO err message
                }
            }
        }
    };

    // Use this for undefined parameterised types that are wrapping two Moose types
    ($combined:ident = $outer:ident<$inner1:ident, $inner2:ident>) => {
        pub type $combined = $outer<$inner1, $inner2>;

        impl crate::computation::SymbolicType for $outer<$inner1, $inner2> {
            type Type = crate::symbolic::Symbolic<
                $outer<
                    <$inner1 as crate::computation::SymbolicType>::Type,
                    <$inner2 as crate::computation::SymbolicType>::Type,
                >,
            >;
        }

        impl crate::computation::CanonicalType for $outer<$inner1, $inner2> {
            type Type = $outer<
                <$inner1 as crate::computation::CanonicalType>::Type,
                <$inner2 as crate::computation::CanonicalType>::Type,
            >;
        }

        impl crate::computation::CanonicalType
            for $outer<
                <$inner1 as crate::computation::SymbolicType>::Type,
                <$inner2 as crate::computation::SymbolicType>::Type,
            >
        {
            type Type = $outer<
                <$inner1 as crate::computation::CanonicalType>::Type,
                <$inner2 as crate::computation::CanonicalType>::Type,
            >;
        }

        impl crate::computation::CanonicalType
            for crate::symbolic::Symbolic<
                $outer<
                    <$inner1 as crate::computation::SymbolicType>::Type,
                    <$inner2 as crate::computation::SymbolicType>::Type,
                >,
            >
        {
            type Type = $outer<
                <$inner1 as crate::computation::CanonicalType>::Type,
                <$inner2 as crate::computation::CanonicalType>::Type,
            >;
        }

        // The kernel macro uses this to map (partially) concrete outputs to symbolic values
        impl
            From<
                $outer<
                    <$inner1 as crate::computation::SymbolicType>::Type,
                    <$inner2 as crate::computation::SymbolicType>::Type,
                >,
            > for <$combined as crate::computation::SymbolicType>::Type
        {
            fn from(
                x: $outer<
                    <$inner1 as crate::computation::SymbolicType>::Type,
                    <$inner2 as crate::computation::SymbolicType>::Type,
                >,
            ) -> Self {
                crate::symbolic::Symbolic::Concrete(x)
            }
        }

        // The kernel macros uses this to determine whether to invoke kernels, and
        // if so, to map symbolic values to (partially) concrete inputs
        impl TryFrom<<$combined as crate::computation::SymbolicType>::Type>
            for $outer<
                <$inner1 as crate::computation::SymbolicType>::Type,
                <$inner2 as crate::computation::SymbolicType>::Type,
            >
        {
            type Error = crate::error::Error;

            fn try_from(
                v: <$combined as crate::computation::SymbolicType>::Type,
            ) -> crate::error::Result<Self> {
                match v {
                    crate::symbolic::Symbolic::Concrete(x) => Ok(x),
                    _ => Err(crate::error::Error::Unexpected(None)), // TODO err message
                }
            }
        }
    };
}

// NOTE const generics is currently not mature in stable
// so we go the old route for now;
// see https://github.com/rust-lang/rust/issues/60551

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct N64;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct N128;

pub trait Const {
    const VALUE: usize;
}

impl Const for N64 {
    const VALUE: usize = 64;
}

impl Const for N128 {
    const VALUE: usize = 128;
}

pub trait Ring {
    type BitLength: Const;
}

macro_rules! unmodelled {
    /*
    Nullary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? () -> $u:ty, $op:ident) => {
        impl crate::kernels::NullaryKernelCheck<crate::kernels::SyncSession, $plc, $u> for $op {}
    };

    /*
    Unary
    */
    ($plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty) -> $u:ty, $op:ident) => {
        impl crate::kernels::UnaryKernelCheck<crate::kernels::SyncSession, $plc, $t0, $u> for $op {}
    };

    /*
    Binary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty, $t1:ty) -> $u:ty, $op:ident) => {
        impl crate::kernels::BinaryKernelCheck<crate::kernels::SyncSession, $plc, $t0, $t1, $u>
            for $op
        {
        }
    };

    /*
    Ternary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $op:ident) => {
        impl
            crate::kernels::TernaryKernelCheck<crate::kernels::SyncSession, $plc, $t0, $t1, $t2, $u>
            for $op
        {
        }
    };
}

pub mod additive;
pub mod common;
pub mod compilation;
pub mod computation;
pub mod error;
pub mod execution;
pub mod fixedpoint;
pub mod floatingpoint;
pub mod host;
pub mod kernels;
pub mod logical;
pub mod networking;
pub mod prim;
pub mod prng;
pub mod python_computation;
pub mod replicated;
pub mod storage;
pub mod symbolic;
pub mod text_computation;
pub mod utils;
