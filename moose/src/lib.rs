// Returns the context-specific type for the given basic type.
macro_rules! cs {
    ($canonical_type:ty) => {
        <$canonical_type as KnownType<S>>::Type
    };
}

macro_rules! st {
    ($partially_canonical_type:ty) => {
        <<$partially_canonical_type as crate::computation::CanonicalType>::Type as KnownType<S>>::Type
    };
    ($partially_canonical_type:ty, $s:ty) => {
        <<$partially_canonical_type as crate::computation::CanonicalType>::Type as KnownType<$s>>::Type
    };
}

/// Map a type to its canonical type
///
/// Using this macro requires adding the following trait bound:
///   $t: CanonicalType
///
/// Examples:
/// - c!(RepTensor<HostBitTen>) -> RepTensor<HostBitTen>
/// - c!(RepTensor<Sym<HostBitTen>>) -> RepTensor<HostBitTen>
/// - c!(Sym<RepTensor<Sym<HostBitTen>>>) -> RepTensor<HostBitTen>
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
///   m!(c!(RepTensor<HostRingT>))
/// which then requires adding trait bounds:
///   $t: CanonicalType
///   <$t as CanonicalType>::Type: KnownType<S>
///
/// Examples:
/// - m!(RepTensor<HostBitTen>) in SyncSession -> RepTensor<HostBitTen>
/// - m!(RepTensor<HostBitTen>) in SymbSession -> Sym<RepTensor<Sym<HostBitTen>>>
macro_rules! m {
    ($t:ty) => {
        <$t as KnownType<S>>::Type
    };
}

macro_rules! derive_runtime_kernel {
    (nullary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> crate::error::Result<Box<dyn Fn(&_, &_,) -> _ + Send>> = &|$op| $kf;
            kf($self)
        }
    };
    (unary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> crate::error::Result<Box<dyn Fn(&_, &_, _) -> _ + Send>> = &|$op| $kf;
            kf($self)
        }
    };
    (binary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> crate::error::Result<Box<dyn Fn(&_, &_, _, _) -> _ + Send>> = &|$op| $kf;
            kf($self)
        }
    };
    (ternary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> crate::error::Result<Box<dyn Fn(&_, &_, _, _, _) -> _ + Send>> = &|$op| $kf;
            kf($self)
        }
    };

    (variadic, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> crate::error::Result<Box<dyn Fn(&_, &_, Vec<_>) -> _ + Send>> = &|$op| $kf;
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
            crate::error::Result::<Box<dyn Fn(&_, &_) -> _ + Send>>::Ok(Box::new(move |sess, plc| {
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
                crate::error::Result::<Box<dyn Fn(&_, &_, _) -> _ + Send>>::Ok(
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
            crate::error::Result::<Box<dyn Fn(&_, &_, _, _) -> _ + Send>>::Ok(Box::new(move |sess, plc, x0, x1| {
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
            crate::error::Result::<Box<dyn Fn(&_, &_, _, _, _) -> crate::error::Result<_> + Send>>::Ok(Box::new(move |sess, plc, x0, x1, x2| {
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
                    crate::error::Result::<Box<dyn Fn(&_, &_, Vec<_>) -> _ + Send>>::Ok(
                        Box::new(move |sess, plc, xs| {
                            $k(sess, plc, $($attr.clone()),+, &xs)
                        })
                    )
                }
        }
    };

    (nullary, $k:expr, $self:ident) => {
        crate::error::Result::<Box<dyn Fn(&_, &_,) -> _ + Send>>::Ok(Box::new($k))
    };
    (unary, $k:expr, $self:ident) => {
        crate::error::Result::<Box<dyn Fn(&_, &_, _) -> _ + Send>>::Ok(Box::new($k))
    };
    (binary, $k:expr, $self:ident) => {
        crate::error::Result::<Box<dyn Fn(&_, &_, _, _) -> _ + Send>>::Ok(Box::new($k))
    };
    (ternary, $k:expr, $self:ident) => {
        crate::error::Result::<Box<dyn Fn(&_, &_, _, _, _) -> _ + Send>>::Ok(Box::new($k))
    };
    (variadic, $k:expr, $self:ident) => {
        crate::error::Result::<Box<dyn Fn(&_, &_, Vec<_>) -> _ + Send>>::Ok(
            Box::new(move |sess, plc, xs| {
                $k(sess, plc, &xs)
            })
        )
    };
}

macro_rules! concrete_dispatch_kernel {

    /*
    Nullary
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::execution::SyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::SyncSession>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, NullarySignature};
                use crate::execution::{SyncSession};
                use crate::kernels::{NullaryKernel};
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
                            let op = self.clone();

                            let k = <$op as NullaryKernel<SyncSession, $plc, $u>>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands: Vec<crate::computation::Value>| {
                                assert_eq!(operands.len(), 0);

                                let y: $u = k(sess, &plc)?;
                                if y.placement()? == plc.clone().into() {
                                    Ok(y.into())
                                } else {
                                    Err(crate::error::Error::KernelError(format!("Placement mismatch after running {:?}. Expected {:?} got {:?}", op, plc, y.placement())))
                                }
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        impl crate::kernels::DispatchKernel<crate::execution::AsyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::AsyncSession>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, NullarySignature, AsyncValue};
                use crate::execution::{AsyncSession};
                use crate::kernels::{NullaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature {
                                ret: <$u as KnownType<AsyncSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;
                            // TODO: Do we want to be deriving the kernel inside? Probably not...
                            let op = self.clone();

                            Ok(Box::new(move |sess, operands: Vec<AsyncValue>| {
                                assert_eq!(operands.len(), 0);
                                let sess = sess.clone();
                                let plc = plc.clone();
                                let k = <$op as NullaryKernel<AsyncSession, $plc, $u>>::compile(&op, &plc)?;
                                let (sender, result) = crate::computation::new_async_value(); // This creates a channel
                                let op = op.clone(); // Needed for the error message for KernelError
                                let tasks = std::sync::Arc::clone(&sess.tasks);
                                let task: tokio::task::JoinHandle<crate::error::Result<()>> = tokio::spawn(async move {
                                    let y: $u = k(&sess, &plc)?;
                                    if y.placement()? == plc.clone().into() {
                                        crate::execution::map_send_result(sender.send(y.into()))?;
                                        Ok(())
                                    } else {
                                        Err(crate::error::Error::KernelError(format!("Placement mismatch after running {:?}. Expected {:?} got {:?}", op, plc, y.placement())))
                                    }
                                });
                                let mut tasks = tasks.write().unwrap();
                                tasks.push(task);

                                Ok(result)
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
        impl crate::kernels::DispatchKernel<crate::execution::SyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::SyncSession>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, UnarySignature, Value};
                use crate::execution::{SyncSession};
                use crate::kernels::{UnaryKernel};
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
                            let op = self.clone();

                            Ok(Box::new(move |sess, operands: Vec<Value>| {
                                assert_eq!(operands.len(), 1);

                                let x0: $t0 = operands.get(0).unwrap().clone().try_into()?;

                                let y: $u = k(sess, &plc, x0)?;
                                if y.placement()? == plc.clone().into() {
                                    Ok(y.into())
                                } else {
                                    Err(crate::error::Error::KernelError(format!("Placement mismatch after running {:?}. Expected {:?} got {:?}", op, plc, y.placement())))
                                }
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        impl crate::kernels::DispatchKernel<crate::execution::AsyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::AsyncSession>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, UnarySignature, AsyncValue};
                use crate::execution::{AsyncSession};
                use crate::kernels::{UnaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Unary(UnarySignature {
                                arg0: <$t0 as KnownType<AsyncSession>>::TY,
                                ret: <$u as KnownType<AsyncSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;
                            // TODO: Do we want to be deriving the kernel inside? Probably not...
                            let op = self.clone();
                            // let k = <$op as UnaryKernel<AsyncSession, $plc, $t0, $u>>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands: Vec<AsyncValue>| {
                                assert_eq!(operands.len(), 1);
                                let sess = sess.clone();
                                let plc = plc.clone();
                                let k = <$op as UnaryKernel<AsyncSession, $plc, $t0, $u>>::compile(&op, &plc)?;
                                let (sender, result) = crate::computation::new_async_value(); // This creates a channel
                                let op = op.clone(); // Needed for the error message for KernelError
                                let tasks = std::sync::Arc::clone(&sess.tasks);
                                let task: tokio::task::JoinHandle<crate::error::Result<()>> = tokio::spawn(async move {
                                    let operands = futures::future::join_all(operands).await;
                                    let x0: $t0 = operands
                                            .get(0)
                                            .ok_or_else(|| crate::error::Error::MalformedEnvironment(format!("Argument {} is missing", 0)))?
                                            .clone()
                                            .map_err(crate::execution::map_receive_error)?
                                            .try_into()?;
                                    let y: $u = k(&sess, &plc, x0)?;
                                    if y.placement()? == plc.clone().into() {
                                        crate::execution::map_send_result(sender.send(y.into()))?;
                                        Ok(())
                                    } else {
                                        Err(crate::error::Error::KernelError(format!("Placement mismatch after running {:?}. Expected {:?} got {:?}", op, plc, y.placement())))
                                    }
                                });
                                let mut tasks = tasks.write().unwrap();
                                tasks.push(task);

                                Ok(result)
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
        impl crate::kernels::DispatchKernel<crate::execution::SyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::SyncSession>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, BinarySignature};
                use crate::execution::{SyncSession};
                use crate::kernels::{BinaryKernel};
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
                            let op = self.clone();

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
                                if y.placement()? == plc.clone().into() {
                                    Ok(y.into())
                                } else {
                                    Err(crate::error::Error::KernelError(format!("Placement mismatch after running {:?}. Expected {:?} got {:?}", op, plc, y.placement())))
                                }
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        impl crate::kernels::DispatchKernel<crate::execution::AsyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::AsyncSession>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, BinarySignature, AsyncValue};
                use crate::execution::{AsyncSession};
                use crate::kernels::{BinaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Binary(BinarySignature {
                                arg0: <$t0 as KnownType<AsyncSession>>::TY,
                                arg1: <$t1 as KnownType<AsyncSession>>::TY,
                                ret: <$u as KnownType<AsyncSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;
                            // TODO: Do we want to be deriving the kernel inside? Probably not...
                            let op = self.clone();

                            Ok(Box::new(move |sess, operands: Vec<AsyncValue>| {
                                assert_eq!(operands.len(), 2);
                                let sess = sess.clone();
                                let plc = plc.clone();
                                let k = <$op as BinaryKernel<AsyncSession, $plc, $t0, $t1, $u>>::compile(&op, &plc)?;
                                let (sender, result) = crate::computation::new_async_value(); // This creates a channel
                                let op = op.clone(); // Needed for the error message for KernelError
                                let tasks = std::sync::Arc::clone(&sess.tasks);
                                let task: tokio::task::JoinHandle<crate::error::Result<()>> = tokio::spawn(async move {
                                    let operands = futures::future::join_all(operands).await;
                                    let x0: $t0 = operands
                                            .get(0)
                                            .ok_or_else(|| crate::error::Error::MalformedEnvironment(format!("Argument {} is missing", 0)))?
                                            .clone()
                                            .map_err(crate::execution::map_receive_error)?
                                            .try_into()?;
                                    let x1: $t1 = operands
                                            .get(1)
                                            .ok_or_else(|| crate::error::Error::MalformedEnvironment(format!("Argument {} is missing", 1)))?
                                            .clone()
                                            .map_err(crate::execution::map_receive_error)?
                                            .try_into()?;
                                    let y: $u = k(&sess, &plc, x0, x1)?;
                                    if y.placement()? == plc.clone().into() {
                                        crate::execution::map_send_result(sender.send(y.into()))?;
                                        Ok(())
                                    } else {
                                        Err(crate::error::Error::KernelError(format!("Placement mismatch after running {:?}. Expected {:?} got {:?}", op, plc, y.placement())))
                                    }
                                });
                                let mut tasks = tasks.write().unwrap();
                                tasks.push(task);

                                Ok(result)
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
        impl crate::kernels::DispatchKernel<crate::execution::SyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::SyncSession>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, TernarySignature};
                use crate::execution::{SyncSession};
                use crate::kernels::{TernaryKernel};
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
                            let op = self.clone();

                            let k = <$op as TernaryKernel<SyncSession, $plc, $t0, $t1, $t2, $u>>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands| {
                                assert_eq!(operands.len(), 3);

                                let x0: $t0 = operands.get(0).unwrap().clone().try_into()?;
                                let x1: $t1 = operands.get(1).unwrap().clone().try_into()?;
                                let x2: $t2 = operands.get(2).unwrap().clone().try_into()?;

                                let y: $u = k(sess, &plc, x0, x1, x2)?;
                                if y.placement()? == plc.clone().into() {
                                    Ok(y.into())
                                } else {
                                    Err(crate::error::Error::KernelError(format!("Placement mismatch after running {:?}. Expected {:?} got {:?}", op, plc, y.placement())))
                                }
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        impl crate::kernels::DispatchKernel<crate::execution::AsyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::AsyncSession>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, TernarySignature, AsyncValue};
                use crate::execution::{AsyncSession};
                use crate::kernels::{TernaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Ternary(TernarySignature {
                                arg0: <$t0 as KnownType<AsyncSession>>::TY,
                                arg1: <$t1 as KnownType<AsyncSession>>::TY,
                                arg2: <$t2 as KnownType<AsyncSession>>::TY,
                                ret: <$u as KnownType<AsyncSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;
                            // TODO: Do we want to be deriving the kernel inside? Probably not...
                            let op = self.clone();

                            Ok(Box::new(move |sess, operands: Vec<AsyncValue>| {
                                assert_eq!(operands.len(), 3);
                                let sess = sess.clone();
                                let plc = plc.clone();
                                let k = <$op as TernaryKernel<AsyncSession, $plc, $t0, $t1, $t2, $u>>::compile(&op, &plc)?;
                                let (sender, result) = crate::computation::new_async_value(); // This creates a channel
                                let op = op.clone(); // Needed for the error message for KernelError
                                let tasks = std::sync::Arc::clone(&sess.tasks);
                                let task: tokio::task::JoinHandle<crate::error::Result<()>> = tokio::spawn(async move {
                                    let operands = futures::future::join_all(operands).await;
                                    let x0: $t0 = operands
                                            .get(0)
                                            .ok_or_else(|| crate::error::Error::MalformedEnvironment(format!("Argument {} is missing", 0)))?
                                            .clone()
                                            .map_err(crate::execution::map_receive_error)?
                                            .try_into()?;
                                    let x1: $t1 = operands
                                            .get(1)
                                            .ok_or_else(|| crate::error::Error::MalformedEnvironment(format!("Argument {} is missing", 1)))?
                                            .clone()
                                            .map_err(crate::execution::map_receive_error)?
                                            .try_into()?;
                                    let x2: $t2 = operands
                                            .get(2)
                                            .ok_or_else(|| crate::error::Error::MalformedEnvironment(format!("Argument {} is missing", 2)))?
                                            .clone()
                                            .map_err(crate::execution::map_receive_error)?
                                            .try_into()?;
                                    let y: $u = k(&sess, &plc, x0, x1, x2)?;
                                    if y.placement()? == plc.clone().into() {
                                        crate::execution::map_send_result(sender.send(y.into()))?;
                                        Ok(())
                                    } else {
                                        Err(crate::error::Error::KernelError(format!("Placement mismatch after running {:?}. Expected {:?} got {:?}", op, plc, y.placement())))
                                    }
                                });
                                let mut tasks = tasks.write().unwrap();
                                tasks.push(task);

                                Ok(result)
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
        impl crate::kernels::DispatchKernel<crate::execution::SyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::SyncSession>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, VariadicSignature, Value};
                use crate::execution::{SyncSession};
                use crate::kernels::{VariadicKernel};
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
                            let op = self.clone();

                            let k = <$op as VariadicKernel<SyncSession, $plc, $ts, $u>>::compile(self, &plc)?;

                            Ok(Box::new(move |sess, operands: Vec<Value>| {
                                let xs: crate::error::Result<Vec<$ts>> = operands.into_iter().map(|xi| xi.try_into()).collect();

                                let y: $u = k(sess, &plc, xs?)?;
                                debug_assert_eq!(y.placement()?, plc.clone().into());
                                if y.placement()? == plc.clone().into() {
                                    Ok(y.into())
                                } else {
                                    Err(crate::error::Error::KernelError(format!("Placement mismatch after running {:?}. Expected {:?} got {:?}", op, plc, y.placement())))
                                }
                            }))
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        impl crate::kernels::DispatchKernel<crate::execution::AsyncSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::AsyncSession>>
            {
                use crate::computation::{KnownPlacement, KnownType, Signature, VariadicSignature, AsyncValue};
                use crate::execution::{AsyncSession};
                use crate::kernels::{VariadicKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Variadic(VariadicSignature {
                                args: <$ts as KnownType<AsyncSession>>::TY,
                                ret: <$u as KnownType<AsyncSession>>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into()?;
                            // TODO: Do we want to be deriving the kernel inside? Probably not...
                            let op = self.clone();

                            Ok(Box::new(move |sess, operands: Vec<AsyncValue>| {
                                let sess = sess.clone();
                                let plc = plc.clone();
                                let k = <$op as VariadicKernel<AsyncSession, $plc, $ts, $u>>::compile(&op, &plc)?;
                                let (sender, result) = crate::computation::new_async_value(); // This creates a channel
                                let op = op.clone(); // Needed for the error message for KernelError
                                let tasks = std::sync::Arc::clone(&sess.tasks);
                                let task: tokio::task::JoinHandle<crate::error::Result<()>> = tokio::spawn(async move {
                                    // A bit of involved way of going from a vector of futures to a vector of concrete values extracted
                                    let xs = futures::future::join_all(operands).await;
                                    let xs: std::result::Result<Vec<crate::computation::Value>, _> = xs.into_iter().collect();
                                    let xs = xs.map_err(crate::execution::map_receive_error)?;
                                    let xs: crate::error::Result<Vec<$ts>> = xs.into_iter().map(|xi| xi.try_into()).collect();
                                    let y: $u = k(&sess, &plc, xs?)?;
                                    if y.placement()? == plc.clone().into() {
                                        crate::execution::map_send_result(sender.send(y.into()))?;
                                        Ok(())
                                    } else {
                                        Err(crate::error::Error::KernelError(format!("Placement mismatch after running {:?}. Expected {:?} got {:?}", op, plc, y.placement())))
                                    }
                                });
                                let mut tasks = tasks.write().unwrap();
                                tasks.push(task);

                                Ok(result)
                            }))                        }
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
        impl crate::kernels::DispatchKernel<crate::execution::SymbolicSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::SymbolicSession>>
            {
                use crate::computation::{KnownPlacement, Signature, NullarySignature, KnownType};
                use crate::kernels::{NullaryKernel};
                use crate::execution::SymbolicSession;
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
        impl crate::kernels::DispatchKernel<crate::execution::SymbolicSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::SymbolicSession>>
            {
                use crate::computation::{KnownPlacement, Signature, UnarySignature, KnownType};
                use crate::kernels::{UnaryKernel};
                use crate::execution::SymbolicSession;
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
        impl crate::kernels::DispatchKernel<crate::execution::SymbolicSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::SymbolicSession>>
            {
                use crate::computation::{KnownPlacement, Signature, BinarySignature, KnownType};
                use crate::kernels::{BinaryKernel};
                use crate::execution::SymbolicSession;
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
        impl crate::kernels::DispatchKernel<crate::execution::SymbolicSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::computation::CompiledKernel<crate::execution::SymbolicSession>>
            {
                use crate::computation::{KnownPlacement, Signature, TernarySignature, KnownType};
                use crate::kernels::{TernaryKernel};
                use crate::execution::SymbolicSession;
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
        impl crate::kernels::DispatchKernel<crate::execution::SymbolicSession> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            )-> crate::error::Result<crate::computation::CompiledKernel<crate::execution::SymbolicSession>>
            {
                use crate::computation::{KnownPlacement, Signature, VariadicSignature, KnownType};
                use crate::kernels::{VariadicKernel};
                use crate::execution::SymbolicSession;
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
                crate::execution::SyncSession,
                $plc,
                <$u as crate::computation::KnownType<crate::execution::SyncSession>>::Type
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<Box<dyn Fn(
                    &crate::execution::SyncSession,
                    &$plc)
                    -> crate::error::Result<
                        <$u as crate::computation::KnownType<crate::execution::SyncSession>>::Type> + Send>
                    >
                {
                    derive_runtime_kernel![nullary, $($kp)+, self]
                }
            }
        )+

        $(
            impl crate::kernels::NullaryKernel<
                crate::execution::AsyncSession,
                $plc,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::AsyncSession, &$plc) -> crate::error::Result<$u> + Send>
                > {
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
            crate::execution::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc
            ) -> crate::error::Result<
                <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::SymbolicSession;

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

    (__nullary concrete, $op:ty, $plc:ty, () -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::NullaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc
            ) -> crate::error::Result<
                <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};

                let k = derive_runtime_kernel![nullary, $($kp)+, self]?;

                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                | {
                    let y = k(sess, plc)?;
                    Ok(Symbolic::Concrete(y))
                }))
            }
        }
    };

    (__nullary transparent, $op:ty, $plc:ty, () -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::NullaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                derive_runtime_kernel![nullary, $($kp)+, self]
            }
        }
    };

    (__nullary runtime, $op:ty, $plc:ty, () -> $u:ty => $($kp:tt)+) => {
        impl NullaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc)
                -> crate::error::Result<
                    <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
                > + Send
            >> {
                use crate::execution::symbolic::{SymbolicSession, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                | {
                    let h = sess.add_operation(&op, &[], &plc.clone().into());
                    Ok(Symbolic::Symbolic(h))
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
                crate::execution::SyncSession,
                $plc,
                $t0,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::SyncSession, &$plc, $t0) -> crate::error::Result<$u> + Send>
                > {
                    derive_runtime_kernel![unary, $($kp)+, self]
                }
            }
        )+

        $(
            impl crate::kernels::UnaryKernel<
                crate::execution::AsyncSession,
                $plc,
                $t0,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::AsyncSession, &$plc, $t0) -> crate::error::Result<$u> + Send>
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
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};
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
                                let h = sess.add_operation(op, &[&h0.op], &plc.clone().into());
                                Ok(Symbolic::Symbolic(h))
                            }
                            x0 => Err(crate::error::Error::Unexpected(Some(format!("Unary hybrid kernel flavor encountered Concrete argument in Symbolic-only case {:?}.", x0))))
                        }
                    }
                }))
            }
        }
    };

    (__unary concrete, $op:ty, $plc:ty, ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::UnaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};

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

                    match x0 {
                        Symbolic::Concrete(v0) => {
                            let y = k(sess, plc, v0)?;
                            Ok(Symbolic::Concrete(y))
                        }
                        Symbolic::Symbolic(h0) => {
                            let h = sess.add_operation(op, &[&h0.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                    }
                }))
            }
        }
    };

    (__unary transparent, $op:ty, $plc:ty, ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::UnaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                derive_runtime_kernel![unary, $($kp)+, self]
            }
        }
    };

    (__unary runtime, $op:ty, $plc:ty, ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::UnaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type)
                -> crate::error::Result<
                    <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
                > + Send>>
            {
                use crate::computation::{KnownType};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type
                | {
                    match x0 {
                        Symbolic::Symbolic(h0) => {
                            let h = sess.add_operation(&op, &[&h0.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                        x0 => Err(crate::error::Error::Unexpected(Some(format!("Unary runtime kernel encountered Concrete argument: {:?}.", x0))))
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
                crate::execution::SyncSession,
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
                    Box<dyn Fn(&crate::execution::SyncSession, &$plc, $t0, $t1) -> crate::error::Result<$u> + Send>
                > {
                    derive_runtime_kernel![binary, $($kp)+, self]
                }
            }
        )+

        $(
            impl crate::kernels::BinaryKernel<
                crate::execution::AsyncSession,
                $plc,
                $t0,
                $t1,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::AsyncSession, &$plc, $t0, $t1) -> crate::error::Result<$u> + Send>
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
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};
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
                                let h = sess.add_operation(op, &[&h0.op, &h1.op], &plc.clone().into());
                                Ok(Symbolic::Symbolic(h))
                            }
                            _ => {
                                Err(crate::error::Error::Unexpected(Some("Mixed symbolic and concrete value during compilation".to_string())))
                            }
                        }
                    }
                }))
            }
        }
    };

    (__binary concrete, $op:ty, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::BinaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};

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
                            Ok(Symbolic::Concrete(y))
                        }
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                            let h = sess.add_operation(op, &[&h0.op, &h1.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                        _ => Err(crate::error::Error::Unexpected(Some("Mixed symbolic and concrete value during compilation".to_string())))
                    }
                }))
            }
        }
    };

    (__binary transparent, $op:ty, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::BinaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                derive_runtime_kernel![binary, $($kp)+, self]
            }
        }
    };

    (__binary runtime, $op:ty, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::BinaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::computation::{KnownType};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as KnownType<SymbolicSession>>::Type
                | {
                    match (x0, x1) {
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                            let h = sess.add_operation(&op, &[&h0.op, &h1.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                        (x0, x1) => Err(crate::error::Error::Unexpected(Some(format!("Binary runtime kernel flavor encountered Concrete arguments: {:?} and {:?}", x0, x1))))
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
                crate::execution::SyncSession,
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
                    dyn Fn(&crate::execution::SyncSession, &$plc, $t0, $t1, $t2) -> crate::error::Result<$u> + Send
                >> {
                    derive_runtime_kernel![ternary, $($kp)+, self]
                }
            }
        )+

        $(
            impl crate::kernels::TernaryKernel<
                crate::execution::AsyncSession,
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
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::AsyncSession, &$plc, $t0, $t1, $t2) -> crate::error::Result<$u> + Send>
                > {
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
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t2 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                derive_runtime_kernel![ternary, $($kp)+, self]
            }
        }
    };

    (__ternary hybrid, $op:ty, $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::TernaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t2 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};
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
                                let h = sess.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
                                Ok(Symbolic::Symbolic(h))
                            }
                            (x0, x1, x2) => Err(crate::error::Error::Unexpected(Some(format!("Ternary hybrid kernel flavor encountered Concrete arguments in Symbolic-only case: Arg0: {:?}, Arg1: {:?}, Arg2: {:?}.", x0, x1, x2))))
                        }
                    }
                }))
            }
        }
    };

    (__ternary concrete, $op:ty, $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::TernaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t2 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as KnownType<SymbolicSession>>::Type,
                    x2: <$t2 as KnownType<SymbolicSession>>::Type,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![ternary, $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)

                    match (x0, x1, x2) {
                        (Symbolic::Concrete(v0), Symbolic::Concrete(v1), Symbolic::Concrete(v2)) => {
                            let y = k(sess, plc, v0, v1, v2)?;
                            Ok(Symbolic::Concrete(y))
                        }
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
                            let h = sess.add_operation(op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                        (x0, x1, x2) => Err(crate::error::Error::Unexpected(Some(format!("Ternary concrete kernel flavor encountered mixed Symbolic/Concrete arguments during compilation: Arg0: {:?}, Arg1: {:?}, Arg2: {:?}.", x0, x1, x2))))
                    }
                }))
            }
        }
    };

    (__ternary runtime, $op:ty, $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::TernaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<
                <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>
            > {
                use crate::computation::{KnownType};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as KnownType<SymbolicSession>>::Type,
                    x2: <$t2 as KnownType<SymbolicSession>>::Type
                | {
                    match (x0, x1, x2) {
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
                            let h = sess.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                        (x0, x1, x2) => Err(crate::error::Error::Unexpected(Some(format!("Ternary runtime kernel flavor encountered Concrete arguments during compilation: Arg0: {:?}, Arg1: {:?}, Arg2: {:?}.", x0, x1, x2))))
                    }
                }))
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
                crate::execution::SyncSession,
                $plc,
                $ts,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<Box<
                    dyn Fn(&crate::execution::SyncSession, &$plc, Vec<$ts>) -> crate::error::Result<$u> + Send
                >> {
                    derive_runtime_kernel![variadic, $($kp)+, self]
                }
            }
        )+

        $(
            impl crate::kernels::VariadicKernel<
                crate::execution::AsyncSession,
                $plc,
                $ts,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::AsyncSession, &$plc, Vec<$ts>) -> crate::error::Result<$u> + Send>
                > {
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
            crate::execution::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                Vec<<$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type>
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                derive_runtime_kernel![variadic, $($kp)+, self]
            }
        }
    };

    (__variadic hybrid, $op:ty, $plc:ty, vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::VariadicKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                Vec<<$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type>
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};
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
                            let h = sess.add_operation(op, &handles, &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        } else {
                            Err(crate::error::Error::Unexpected(Some("Variadic hybrid kernel flavor found mixed symbolic and concrete values during compilation.".to_string())))
                        }
                    }
                }))
            }
        }
    };

    (__variadic concrete, $op:ty, $plc:ty, vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::VariadicKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                Vec<<$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type>
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};

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
                    let kernel_vals: Vec<_> = xs.iter().cloned().filter_map(|x| match x {
                        Symbolic::Concrete(v) => Some(v),
                        Symbolic::Symbolic(_) => None,
                    }).collect();
                    if kernel_vals.len() == xs.len() {
                        // success; we can apply kernel
                        let y = k(sess, plc, kernel_vals)?;
                        Ok(Symbolic::Concrete(y))
                    } else {
                        // operands did not match kernel so record in graph instead
                        let handles: Vec<_> = xs.iter().filter_map(Symbolic::symbolic_handle).map(|h| h.op.as_str()).collect();
                        if handles.len() == xs.len() {
                            // success; we can record in graph
                            let h = sess.add_operation(op, &handles, &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        } else {
                            Err(crate::error::Error::Unexpected(Some("Variadic concrete flavor found mixed symbolic and concrete value during compilation.".to_string())))
                        }
                    }
                }))
            }
        }
    };

    (__variadic runtime, $op:ty, $plc:ty, vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::VariadicKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                Vec<<$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type>
            ) -> crate::error::Result<
                <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>
            > {
                use crate::computation::{KnownType};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};

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

                    if res.len() == xs.len() {
                        let h = sess.add_operation(&op, &res, &plc.clone().into());
                        return Ok(Symbolic::Symbolic(h));
                    }

                    Err(crate::error::Error::Unexpected(Some(format!("Variadic runtime kernel found non-Symbolic arguments for {:?}", op))))
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
        impl crate::kernels::NullaryKernelCheck<crate::execution::SyncSession, $plc, $u> for $op {}

        impl $t<crate::execution::SyncSession, $u> for $plc {
            fn $f(&self, sess: &crate::execution::SyncSession, $($($attr_id:$attr_ty),*)?) -> $u {
                use crate::computation::{KnownType, NullarySignature};
                use crate::execution::{Session, SyncSession};
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

        impl crate::kernels::NullaryKernelCheck<crate::execution::AsyncSession, $plc, $u> for $op {}

        impl $t<
            crate::execution::AsyncSession,
            $u
        > for $plc {
            #[allow(unused_variables)]
            fn $f(
                &self,
                sess: &crate::execution::AsyncSession,
                $($($attr_id:$attr_ty),*,)?
            ) -> $u {
                unimplemented!("Async session should not be called via a trait call. Use AsyncSession::execute of a compiled computation instead")
            }
        }

        impl crate::kernels::NullaryKernelCheck<
            crate::execution::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
        > for $op {}

        impl $t<
            crate::execution::SymbolicSession,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $plc {
            fn $f(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*)?
            ) -> <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type {
                use crate::computation::{KnownType, NullarySignature};
                use crate::execution::{Session, SymbolicSession};
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
        impl crate::kernels::UnaryKernelCheck<crate::execution::SyncSession, $plc, $t0, $u> for $op {}

        impl $t<
            crate::execution::SyncSession,
            $t0,
            $u
        > for $plc {
            fn $f(
                &self,
                sess: &crate::execution::SyncSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &$t0
            ) -> $u {
                use crate::computation::{KnownType, UnarySignature};
                use crate::execution::{Session, SyncSession};
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

        impl crate::kernels::UnaryKernelCheck<crate::execution::AsyncSession, $plc, $t0, $u> for $op {}

        impl $t<
            crate::execution::AsyncSession,
            $t0,
            $u
        > for $plc {
            #[allow(unused_variables)]
            fn $f(
                &self,
                sess: &crate::execution::AsyncSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &$t0
            ) -> $u {
                unimplemented!("Async session should not be called via a trait call. Use AsyncSession::execute of a compiled computation instead")
            }
        }

        impl crate::kernels::UnaryKernelCheck<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
        > for $op {}

        impl $t<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $plc {
            fn $f(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type {
                use crate::computation::{KnownType, UnarySignature};
                use crate::execution::{Session, SymbolicSession};
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
        impl crate::kernels::BinaryKernelCheck<crate::execution::SyncSession, $plc, $t0, $t1, $u> for $op {}

        impl $t<crate::execution::SyncSession, $t0, $t1, $u> for $plc {
            fn $f(&self, sess: &crate::execution::SyncSession, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1) -> $u {
                use crate::computation::{KnownType, BinarySignature};
                use crate::execution::{Session, SyncSession};
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

        impl crate::kernels::BinaryKernelCheck<crate::execution::AsyncSession, $plc, $t0, $t1, $u> for $op {}

        impl $t<
            crate::execution::AsyncSession,
            $t0,
            $t1,
            $u
        > for $plc {
            #[allow(unused_variables)]
            fn $f(
                &self,
                sess: &crate::execution::AsyncSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &$t0,
                x1: &$t1,
            ) -> $u {
                unimplemented!("Async session should not be called via a trait call. Use AsyncSession::execute of a compiled computation instead")
            }
        }

        impl crate::kernels::BinaryKernelCheck<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$t1 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type,
        > for $op {}

        // impl $t<
        //     crate::execution::SymbolicSession,
        //     <$t0 as crate::computation::PartiallySymbolicType>::Type,
        //     <$t1 as crate::computation::PartiallySymbolicType>::Type,
        //     <$u as crate::computation::PartiallySymbolicType>::Type
        // > for $plc {
        //     fn $f(
        //         &self,
        //         sess: &crate::execution::SymbolicSession,
        //         $($($attr_id:$attr_ty),*,)?
        //         x0: &<$t0 as crate::computation::PartiallySymbolicType>::Type,
        //         x1: &<$t1 as crate::computation::PartiallySymbolicType>::Type
        //     ) -> <$u as crate::computation::PartiallySymbolicType>::Type {
        //         use crate::computation::{KnownType, BinarySignature};
        //         use crate::execution::{Session};
        //         use crate::execution::symbolic::{SymbolicSession, Symbolic};
        //         use std::convert::TryInto;
        //         let sig = BinarySignature {
        //             arg0: <$t0 as KnownType<SymbolicSession>>::TY,
        //             arg1: <$t1 as KnownType<SymbolicSession>>::TY,
        //             ret: <$u as KnownType<SymbolicSession>>::TY,
        //         };
        //         let op = $op {
        //             sig: sig.into(),
        //             $($($attr_id),*)?
        //         };
        //         let x0 = Symbolic::Concrete(x0.clone()).into();
        //         let x1 = Symbolic::Concrete(x1.clone()).into();
        //         sess.execute(op.into(), &self.into(), vec![x0, x1])
        //             .unwrap()
        //             .try_into()
        //             .unwrap()
        //     }
        // }

        impl $t<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$t1 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $f(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type,
                x1: &<$t1 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, BinarySignature};
                use crate::execution::{Session, SymbolicSession};
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
        impl crate::kernels::TernaryKernelCheck<crate::execution::SyncSession, $plc, $t0, $t1, $t2, $u> for $op {}

        impl $t<crate::execution::SyncSession, $t0, $t1, $t2, $u> for $plc {
            fn $f(&self, sess: &crate::execution::SyncSession, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1, x2: &$t2) -> $u {
                use crate::computation::{KnownType, TernarySignature};
                use crate::execution::{Session, SyncSession};
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

        impl crate::kernels::TernaryKernelCheck<crate::execution::AsyncSession, $plc, $t0, $t1, $t2, $u> for $op {}

        impl $t<
            crate::execution::AsyncSession,
            $t0,
            $t1,
            $t2,
            $u
        > for $plc {
            #[allow(unused_variables)]
            fn $f(
                &self,
                sess: &crate::execution::AsyncSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &$t0,
                x1: &$t1,
                x2: &$t2,
            ) -> $u {
                unimplemented!("Async session should not be called via a trait call. Use AsyncSession::execute of a compiled computation instead")
            }
        }

        impl crate::kernels::TernaryKernelCheck<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
        > for $op {}

        impl $t<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $plc {
            fn $f(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                x1: &<$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                x2: &<$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type {
                use crate::computation::{KnownType, TernarySignature};
                use crate::execution::{Session, SymbolicSession};
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
        impl crate::kernels::VariadicKernelCheck<crate::execution::SyncSession, $plc, $ts, $u> for $op {}

        impl $t<
            crate::execution::SyncSession,
            $ts,
            $u
        > for $plc {
            fn $f(
                &self,
                sess: &crate::execution::SyncSession,
                $($($attr_id:$attr_ty),*,)?
                xs: &[$ts]
            ) -> $u {
                use crate::computation::{KnownType, VariadicSignature};
                use crate::execution::{Session, SyncSession};
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

        impl crate::kernels::VariadicKernelCheck<crate::execution::AsyncSession, $plc, $ts, $u> for $op {}

        impl $t<
            crate::execution::AsyncSession,
            $ts,
            $u
        > for $plc {
            #[allow(unused_variables)]
            fn $f(
                &self,
                sess: &crate::execution::AsyncSession,
                $($($attr_id:$attr_ty),*,)?
                xs: &[$ts]
            ) -> $u {
                unimplemented!("Async session should not be called via a trait call. Use AsyncSession::execute of a compiled computation instead")
            }
        }

        impl crate::kernels::VariadicKernelCheck<
            crate::execution::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
        > for $op {}

        impl $t<
            crate::execution::SymbolicSession,
            <$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $plc {
            fn $f(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                xs: &[<$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type]
            ) -> <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type {
                use crate::computation::{KnownType, VariadicSignature};
                use crate::execution::{Session, SymbolicSession};
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

macro_rules! modelled_kernel {

    /*
    Nullary
    */

    ($trait:ident::$trait_fn:ident, $op:ident, [$( ($plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? () -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, () -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, () -> $u), )+]);

        // support for SyncSession
        $(
            impl crate::kernels::NullaryKernel<
                crate::execution::SyncSession,
                $plc,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::SyncSession, &$plc) -> crate::error::Result<$u> + Send>
                > {
                    derive_runtime_kernel![nullary, $(attributes[$($attr_id),+])? $($kp)+, self]
                }
            }

            impl $trait<crate::execution::SyncSession, $u> for $plc {
                fn $trait_fn(&self, sess: &crate::execution::SyncSession, $($($attr_id:$attr_ty,)*)?) -> $u {
                    use crate::computation::{KnownType, NullarySignature};
                    use crate::execution::{Session, SyncSession};
                    use std::convert::TryInto;

                    let sig = NullarySignature {
                        ret: <$u as KnownType<SyncSession>>::TY,
                    };
                    let op = $op {
                        sig: sig.into(),
                        $($($attr_id),*)?
                    };

                    let y = sess.execute(
                        op.into(),
                        &self.into(),
                        vec![],
                    ).unwrap();
                    y.try_into().unwrap()
                }
            }
        )+

        // support for AsyncSession
        $(
            impl crate::kernels::NullaryKernel<
                crate::execution::AsyncSession,
                $plc,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::AsyncSession, &$plc) -> crate::error::Result<$u> + Send>
                > {
                    derive_runtime_kernel![nullary, $(attributes[$($attr_id),+])? $($kp)+, self]
                }
            }

            impl $trait<
                crate::execution::AsyncSession,
                $u
            > for $plc {
                #[allow(unused_variables)]
                fn $trait_fn(
                    &self,
                    sess: &crate::execution::AsyncSession,
                    $($($attr_id:$attr_ty,)*)?
                ) -> $u {
                    unimplemented!("Async session should not be called via a trait call. Use AsyncSession::execute of a compiled computation instead")
                }
            }
        )+

        // support for SymbolicSession (based on flavour)
        $(
            modelled_kernel!(__nullary $flavour, $trait, $trait_fn, $op, $plc, $([$($attr_id:$attr_ty),*])? () -> $u => $($kp)+);
        )+
    };


    (__nullary hybrid, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? () -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::NullaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};
                use std::convert::TryInto;

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![nullary, $(attributes[$($attr_id),+])? $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)
                    let y = k(sess, plc)?;
                    Ok(y.into())
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty,)*)?
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, NullarySignature};
                use crate::execution::{Session, SymbolicSession};
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

    (__nullary concrete, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? () -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::NullaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                | {
                    #[allow(unused_variables)]
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![nullary, $(attributes[$($attr_id),+])? $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)
                    let y = k(sess, plc)?;
                    Ok(Symbolic::Concrete(y))
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$u as crate::computation::PartiallySymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*)?
            ) -> <$u as crate::computation::PartiallySymbolicType>::Type {
                use crate::computation::{KnownType, NullarySignature};
                use crate::execution::{Session};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};
                use std::convert::TryFrom;

                let sig = NullarySignature {
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };

                let y = sess.execute(op.into(), &self.into(), vec![]).unwrap();
                let y = Symbolic::try_from(y).unwrap();
                match y {
                    Symbolic::Concrete(y) => y,
                    Symbolic::Symbolic(_) => panic!(), // ok since this is concrete flavour
                }
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*)?
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, NullarySignature};
                use crate::execution::{Session};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};
                use std::convert::TryFrom;

                let sig = NullarySignature {
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };

                let y = sess.execute(op.into(), &self.into(), vec![]).unwrap();
                Symbolic::try_from(y).unwrap()
            }
        }
    };

    (__nullary transparent, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? () -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::NullaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                derive_runtime_kernel![nullary, $(attributes[$($attr_id),+])? $($kp)+, self]
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*)?
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, NullarySignature};
                use crate::execution::{Session, SymbolicSession};
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

    (__nullary runtime, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? () -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::NullaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{SymbolicSession, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc
                | {
                    let h = sess.add_operation(&op, &[], &plc.clone().into());
                    Ok(Symbolic::Symbolic(h))
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*)?
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, NullarySignature};
                use crate::execution::{Session, SymbolicSession};
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

    ($trait:ident::$trait_fn:ident, $op:ident, [$( ($plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty) -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0) -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, ($t0) -> $u), )+]);

        // support for SyncSession
        $(
            impl crate::kernels::UnaryKernel<
                crate::execution::SyncSession,
                $plc,
                $t0,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::SyncSession, &$plc, $t0) -> crate::error::Result<$u> + Send>
                > {
                    derive_runtime_kernel![unary, $(attributes[$($attr_id),+])? $($kp)+, self]
                }
            }

            impl $trait<crate::execution::SyncSession, $t0, $u> for $plc {
                fn $trait_fn(&self, sess: &crate::execution::SyncSession, $($($attr_id:$attr_ty,)*)? x0: &$t0) -> $u {
                    use crate::computation::{KnownType, UnarySignature};
                    use crate::execution::{Session, SyncSession};
                    use std::convert::TryInto;

                    let sig = UnarySignature {
                        arg0: <$t0 as KnownType<SyncSession>>::TY,
                        ret: <$u as KnownType<SyncSession>>::TY,
                    };
                    let op = $op {
                        sig: sig.into(),
                        $($($attr_id),*)?
                    };

                    let x0 = x0.clone().into();
                    let y = sess.execute(
                        op.into(),
                        &self.into(),
                        vec![x0],
                    ).unwrap();
                    y.try_into().unwrap()
                }
            }
        )+

        // support for AsyncSession
        $(
            impl crate::kernels::UnaryKernel<
                crate::execution::AsyncSession,
                $plc,
                $t0,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::AsyncSession, &$plc, $t0) -> crate::error::Result<$u> + Send>
                > {
                    derive_runtime_kernel![unary, $(attributes[$($attr_id),+])? $($kp)+, self]
                }
            }

            impl $trait<
                crate::execution::AsyncSession,
                $t0,
                $u
            > for $plc {
                #[allow(unused_variables)]
                fn $trait_fn(
                    &self,
                    sess: &crate::execution::AsyncSession,
                    $($($attr_id:$attr_ty,)*)?
                    x0: &$t0,
                ) -> $u {
                    unimplemented!("Async session should not be called via a trait call. Use AsyncSession::execute of a compiled computation instead")
                }
            }
        )+

        // support for SymbolicSession (based on flavour)
        $(
            modelled_kernel!(__unary $flavour, $trait, $trait_fn, $op, $plc, $([$($attr_id:$attr_ty),*])? ($t0) -> $u => $($kp)+);
        )+
    };

    (__unary hybrid, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::UnaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};
                use std::convert::TryInto;

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as crate::computation::KnownType<SymbolicSession>>::Type,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![unary, $(attributes[$($attr_id),+])? $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)

                    let v0 = x0.clone().try_into();

                    match v0 {
                        Ok(v0) => {
                            let y = k(sess, plc, v0)?;
                            Ok(y.into())
                        }
                        _ => match x0 {
                            Symbolic::Symbolic(h0) => {
                                let h = sess.add_operation(op, &[&h0.op], &plc.clone().into());
                                Ok(Symbolic::Symbolic(h))
                            }
                            _ => {
                                Err(crate::error::Error::Unexpected(Some("Expected symbolic value during compilation".to_string())))
                            }
                        }
                    }
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty,)*)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, UnarySignature};
                use crate::execution::{Session, SymbolicSession};
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

    (__unary concrete, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::UnaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as crate::computation::KnownType<SymbolicSession>>::Type,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![unary, $(attributes[$($attr_id),+])? $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)

                    match x0 {
                        Symbolic::Concrete(v0) => {
                            let y = k(sess, plc, v0)?;
                            Ok(Symbolic::Concrete(y))
                        }
                        Symbolic::Symbolic(h0) => {
                            let h = sess.add_operation(op, &[&h0.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                    }
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::PartiallySymbolicType>::Type,
            <$u as crate::computation::PartiallySymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::PartiallySymbolicType>::Type
            ) -> <$u as crate::computation::PartiallySymbolicType>::Type {
                use crate::computation::{KnownType, UnarySignature, SymbolicValue};
                use crate::execution::{Session};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};
                use std::convert::TryFrom;

                let sig = UnarySignature {
                    arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };

                let x0 = SymbolicValue::from(Symbolic::Concrete(x0.clone()));
                let y = sess.execute(op.into(), &self.into(), vec![x0]).unwrap();
                let y = Symbolic::try_from(y).unwrap();
                match y {
                    Symbolic::Concrete(y) => y,
                    Symbolic::Symbolic(_) => panic!(), // ok since this is concrete flavour
                }
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, UnarySignature, SymbolicValue};
                use crate::execution::{Session};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};
                use std::convert::TryFrom;

                let sig = UnarySignature {
                    arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };

                let x0 = SymbolicValue::from(x0.clone());
                let y = sess.execute(op.into(), &self.into(), vec![x0]).unwrap();
                Symbolic::try_from(y).unwrap()
            }
        }
    };

    (__unary transparent, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::UnaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                derive_runtime_kernel![unary, $(attributes[$($attr_id),+])? $($kp)+, self]
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, UnarySignature};
                use crate::execution::{Session, SymbolicSession};
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

    (__unary runtime, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::UnaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::computation::{KnownType};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type
                | {
                    match x0 {
                        Symbolic::Symbolic(h0) => {
                            let h = sess.add_operation(&op, &[&h0.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                        _ => Err(crate::error::Error::Unexpected(Some("Expected symbolic value during compilation".to_string())))
                    }
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, UnarySignature};
                use crate::execution::{Session, SymbolicSession};
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

    ($trait:ident::$trait_fn:ident, $op:ident, [$( ($plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty) -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0, $t1) -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, ($t0, $t1) -> $u), )+]);

        // support for SyncSession
        $(
            impl crate::kernels::BinaryKernel<
                crate::execution::SyncSession,
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
                    Box<dyn Fn(&crate::execution::SyncSession, &$plc, $t0, $t1) -> crate::error::Result<$u> + Send>
                > {
                    derive_runtime_kernel![binary, $(attributes[$($attr_id),+])? $($kp)+, self]
                }
            }

            impl $trait<crate::execution::SyncSession, $t0, $t1, $u> for $plc {
                fn $trait_fn(&self, sess: &crate::execution::SyncSession, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1) -> $u {
                    use crate::computation::{KnownType, BinarySignature};
                    use crate::execution::{Session, SyncSession};
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
        )+

        // support for AsyncSession
        $(
            impl crate::kernels::BinaryKernel<
                crate::execution::AsyncSession,
                $plc,
                $t0,
                $t1,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::AsyncSession, &$plc, $t0, $t1) -> crate::error::Result<$u> + Send>
                > {
                    derive_runtime_kernel![binary, $(attributes[$($attr_id),+])? $($kp)+, self]
                }
            }

            impl $trait<
                crate::execution::AsyncSession,
                $t0,
                $t1,
                $u
            > for $plc {
                #[allow(unused_variables)]
                fn $trait_fn(
                    &self,
                    sess: &crate::execution::AsyncSession,
                    $($($attr_id:$attr_ty),*,)?
                    x0: &$t0,
                    x1: &$t1,
                ) -> $u {
                    unimplemented!("Async session should not be called via a trait call. Use AsyncSession::execute of a compiled computation instead")
                }
            }
        )+

        // support for SymbolicSession (based on flavour)
        $(
            modelled_kernel!(__binary $flavour, $trait, $trait_fn, $op, $plc, $([$($attr_id:$attr_ty),*])? ($t0, $t1) -> $u => $($kp)+);
        )+


    };

    (__binary hybrid, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::BinaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};
                use std::convert::TryInto;

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as crate::computation::KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as crate::computation::KnownType<SymbolicSession>>::Type,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![binary, $(attributes[$($attr_id),+])? $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)

                    let v0 = x0.clone().try_into();
                    let v1 = x1.clone().try_into();

                    match (v0, v1) {
                        (Ok(v0), Ok(v1)) => {
                            let y = k(sess, plc, v0, v1)?;
                            Ok(y.into())
                        }
                        _ => match (x0, x1) {
                            (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                                let h = sess.add_operation(op, &[&h0.op, &h1.op], &plc.clone().into());
                                Ok(Symbolic::Symbolic(h))
                            }
                            _ => {
                                Err(crate::error::Error::Unexpected(Some("Mixed symbolic and concrete value during compilation".to_string())))
                            }
                        }
                    }
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$t1 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type,
                x1: &<$t1 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, BinarySignature};
                use crate::execution::{Session, SymbolicSession};
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

    (__binary concrete, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::BinaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as crate::computation::KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as crate::computation::KnownType<SymbolicSession>>::Type,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![binary, $(attributes[$($attr_id),+])? $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)

                    match (x0, x1) {
                        (Symbolic::Concrete(v0), Symbolic::Concrete(v1)) => {
                            let y = k(sess, plc, v0, v1)?;
                            Ok(Symbolic::Concrete(y))
                        }
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                            let h = sess.add_operation(op, &[&h0.op, &h1.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                        _ => Err(crate::error::Error::Unexpected(Some("Mixed symbolic and concrete value during compilation".to_string())))
                    }
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::PartiallySymbolicType>::Type,
            <$t1 as crate::computation::PartiallySymbolicType>::Type,
            <$u as crate::computation::PartiallySymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::PartiallySymbolicType>::Type,
                x1: &<$t1 as crate::computation::PartiallySymbolicType>::Type
            ) -> <$u as crate::computation::PartiallySymbolicType>::Type {
                use crate::computation::{KnownType, BinarySignature, SymbolicValue};
                use crate::execution::{Session};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};
                use std::convert::TryFrom;

                let sig = BinarySignature {
                    arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                    arg1: <$t1 as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };

                let x0 = SymbolicValue::from(Symbolic::Concrete(x0.clone()));
                let x1 = SymbolicValue::from(Symbolic::Concrete(x1.clone()));
                let y = sess.execute(op.into(), &self.into(), vec![x0, x1]).unwrap();
                let y = Symbolic::try_from(y).unwrap();
                match y {
                    Symbolic::Concrete(y) => y,
                    Symbolic::Symbolic(_) => panic!(), // ok since this is concrete flavour
                }
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$t1 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type,
                x1: &<$t1 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, BinarySignature, SymbolicValue};
                use crate::execution::{Session};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};
                use std::convert::TryFrom;

                let sig = BinarySignature {
                    arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                    arg1: <$t1 as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };

                let x0 = SymbolicValue::from(x0.clone());
                let x1 = SymbolicValue::from(x1.clone());
                let y = sess.execute(op.into(), &self.into(), vec![x0, x1]).unwrap();
                Symbolic::try_from(y).unwrap()
            }
        }
    };

    (__binary transparent, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::BinaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                derive_runtime_kernel![binary, $(attributes[$($attr_id),+])? $($kp)+, self]
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$t1 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type,
                x1: &<$t1 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, BinarySignature};
                use crate::execution::{Session, SymbolicSession};
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

    (__binary runtime, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::BinaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::computation::{KnownType};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as KnownType<SymbolicSession>>::Type
                | {
                    match (x0, x1) {
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                            let h = sess.add_operation(&op, &[&h0.op, &h1.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                        _ => Err(crate::error::Error::Unexpected(Some("Mixed symbolic and concrete value during compilation".to_string())))
                    }
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$t1 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type,
                x1: &<$t1 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, BinarySignature};
                use crate::execution::{Session, SymbolicSession};
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

    ($trait:ident::$trait_fn:ident, $op:ident, [$( ($plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u), )+]);

        // support for SyncSession
        $(
            impl crate::kernels::TernaryKernel<
                crate::execution::SyncSession,
                $plc,
                $t0,
                $t1,
                $t2,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::SyncSession, &$plc, $t0, $t1, $t2) -> crate::error::Result<$u> + Send>
                > {
                    derive_runtime_kernel![ternary, $(attributes[$($attr_id),+])? $($kp)+, self]
                }
            }

            impl $trait<crate::execution::SyncSession, $t0, $t1, $t2, $u> for $plc {
                fn $trait_fn(&self, sess: &crate::execution::SyncSession, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1, x2: &$t2) -> $u {
                    use crate::computation::{KnownType, TernarySignature};
                    use crate::execution::{Session, SyncSession};
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
        )+

        // support for AsyncSession
        $(
            impl crate::kernels::TernaryKernel<
                crate::execution::AsyncSession,
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
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::AsyncSession, &$plc, $t0, $t1, $t2) -> crate::error::Result<$u> + Send>
                > {
                    derive_runtime_kernel![ternary, $(attributes[$($attr_id),+])? $($kp)+, self]
                }
            }

            impl $trait<
                crate::execution::AsyncSession,
                $t0,
                $t1,
                $t2,
                $u
            > for $plc {
                #[allow(unused_variables)]
                fn $trait_fn(
                    &self,
                    sess: &crate::execution::AsyncSession,
                    $($($attr_id:$attr_ty),*,)?
                    x0: &$t0,
                    x1: &$t1,
                    x2: &$t2,
                ) -> $u {
                    unimplemented!("Async session should not be called via a trait call. Use AsyncSession::execute of a compiled computation instead")
                }
            }
        )+

        // support for SymbolicSession (based on flavour)
        $(
            modelled_kernel!(__ternary $flavour, $trait, $trait_fn, $op, $plc, $([$($attr_id:$attr_ty),*])? ($t0, $t1, $t2) -> $u => $($kp)+);
        )+


    };

    (__ternary hybrid, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::TernaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};
                use std::convert::TryInto;

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as crate::computation::KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as crate::computation::KnownType<SymbolicSession>>::Type,
                    x2: <$t2 as crate::computation::KnownType<SymbolicSession>>::Type,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![ternary, $(attributes[$($attr_id),+])? $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)

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
                                let h = sess.add_operation(op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
                                Ok(Symbolic::Symbolic(h))
                            }
                            _ => {
                                Err(crate::error::Error::Unexpected(Some("Mixed symbolic and concrete value during compilation".to_string())))
                            }
                        }
                    }
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$t1 as crate::computation::SymbolicType>::Type,
            <$t2 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type,
                x1: &<$t1 as crate::computation::SymbolicType>::Type,
                x2: &<$t2 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, TernarySignature};
                use crate::execution::{Session, SymbolicSession};
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

    (__ternary concrete, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::TernaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as crate::computation::KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as crate::computation::KnownType<SymbolicSession>>::Type,
                    x2: <$t2 as crate::computation::KnownType<SymbolicSession>>::Type,
                | {
                    // TODO derive k outside box (using self instead of op)
                    // Magic by Morten
                    let op = &op;

                    let k = derive_runtime_kernel![ternary, $(attributes[$($attr_id),+])? $($kp)+, op].unwrap();  // TODO: replace unwrap (easier with self)

                    match (x0, x1, x2) {
                        (Symbolic::Concrete(v0), Symbolic::Concrete(v1), Symbolic::Concrete(v2)) => {
                            let y = k(sess, plc, v0, v1, v2)?;
                            Ok(Symbolic::Concrete(y))
                        }
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
                            let h = sess.add_operation(op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                        _ => Err(crate::error::Error::Unexpected(Some("Mixed symbolic and concrete value during compilation".to_string())))
                    }
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::PartiallySymbolicType>::Type,
            <$t1 as crate::computation::PartiallySymbolicType>::Type,
            <$t2 as crate::computation::PartiallySymbolicType>::Type,
            <$u as crate::computation::PartiallySymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::PartiallySymbolicType>::Type,
                x1: &<$t1 as crate::computation::PartiallySymbolicType>::Type,
                x2: &<$t2 as crate::computation::PartiallySymbolicType>::Type
            ) -> <$u as crate::computation::PartiallySymbolicType>::Type {
                use crate::computation::{KnownType, TernarySignature, SymbolicValue};
                use crate::execution::{Session};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};
                use std::convert::TryFrom;

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

                let x0 = SymbolicValue::from(Symbolic::Concrete(x0.clone()));
                let x1 = SymbolicValue::from(Symbolic::Concrete(x1.clone()));
                let x2 = SymbolicValue::from(Symbolic::Concrete(x2.clone()));
                let y = sess.execute(op.into(), &self.into(), vec![x0, x1, x2]).unwrap();
                let y = Symbolic::try_from(y).unwrap();
                match y {
                    Symbolic::Concrete(y) => y,
                    Symbolic::Symbolic(_) => panic!(), // ok since this is concrete flavour
                }
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$t1 as crate::computation::SymbolicType>::Type,
            <$t2 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type,
                x1: &<$t1 as crate::computation::SymbolicType>::Type,
                x2: &<$t2 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, TernarySignature, SymbolicValue};
                use crate::execution::{Session};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};
                use std::convert::TryFrom;

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

                let x0 = SymbolicValue::from(x0.clone());
                let x1 = SymbolicValue::from(x1.clone());
                let x2 = SymbolicValue::from(x2.clone());
                let y = sess.execute(op.into(), &self.into(), vec![x0, x1, x2]).unwrap();
                Symbolic::try_from(y).unwrap()
            }
        }
    };

    (__ternary transparent, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::TernaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as KnownType<crate::execution::SymbolicSession>>::Type,
                <$t2 as KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                derive_runtime_kernel![ternary, $(attributes[$($attr_id),+])? $($kp)+, self]
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$t1 as crate::computation::SymbolicType>::Type,
            <$t2 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type,
                x1: &<$t1 as crate::computation::SymbolicType>::Type,
                x2: &<$t2 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, TernarySignature};
                use crate::execution::{Session, SymbolicSession};
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

    (__ternary runtime, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::TernaryKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
            ) -> crate::error::Result<<$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::computation::{KnownType};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};

                let op = self.clone();
                Ok(Box::new(move |
                    sess: &SymbolicSession,
                    plc: &$plc,
                    x0: <$t0 as KnownType<SymbolicSession>>::Type,
                    x1: <$t1 as KnownType<SymbolicSession>>::Type,
                    x2: <$t2 as KnownType<SymbolicSession>>::Type
                | {
                    match (x0, x1, x2) {
                        (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
                            let h = sess.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        }
                        _ => Err(crate::error::Error::Unexpected(Some("Mixed symbolic and concrete value during compilation".to_string())))
                    }
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$t0 as crate::computation::SymbolicType>::Type,
            <$t1 as crate::computation::SymbolicType>::Type,
            <$t2 as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as crate::computation::SymbolicType>::Type,
                x1: &<$t1 as crate::computation::SymbolicType>::Type,
                x2: &<$t2 as crate::computation::SymbolicType>::Type
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, TernarySignature};
                use crate::execution::{Session, SymbolicSession};
                use std::convert::TryInto;

                let sig = TernarySignature {
                    arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                    arg1: <$t1 as KnownType<SymbolicSession>>::TY,
                    arg2: <$t1 as KnownType<SymbolicSession>>::TY,
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

    // Variadic

    ($trait:ident::$trait_fn:ident, $op:ident, [$( ($plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? vec[$ts:ty] -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, vec[$ts] -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, vec[$ts] -> $u), )+]);

        // support for SyncSession
        $(
            impl crate::kernels::VariadicKernel<
                crate::execution::SyncSession,
                $plc,
                $ts,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<Box<
                    dyn Fn(&crate::execution::SyncSession, &$plc, Vec<$ts>) -> crate::error::Result<$u> + Send
                >> {
                    derive_runtime_kernel![variadic, $($kp)+, self]
                }
            }

            impl $trait<
                crate::execution::SyncSession,
                $ts,
                $u
            > for $plc {
                fn $trait_fn(
                    &self,
                    sess: &crate::execution::SyncSession,
                    $($($attr_id:$attr_ty),*,)?
                    xs: &[$ts]
                ) -> $u {
                    use crate::computation::{KnownType, VariadicSignature};
                    use crate::execution::{Session, SyncSession};
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
        )+
        // support for AsyncSession

        $(
            impl crate::kernels::VariadicKernel<
                crate::execution::AsyncSession,
                $plc,
                $ts,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _plc: &$plc,
                ) -> crate::error::Result<
                    Box<dyn Fn(&crate::execution::AsyncSession, &$plc, Vec<$ts>) -> crate::error::Result<$u> + Send>
                > {
                    derive_runtime_kernel![variadic, $($kp)+, self]
                }
            }

            impl $trait<
                crate::execution::AsyncSession,
                $ts,
                $u
            > for $plc {
                #[allow(unused_variables)]
                fn $trait_fn(
                    &self,
                    sess: &crate::execution::AsyncSession,
                    $($($attr_id:$attr_ty),*,)?
                    xs: &[$ts]
                ) -> $u {
                    unimplemented!("Async session should not be called via a trait call. Use AsyncSession::execute of a compiled computation instead")
                }
            }
        )+

        $(
            modelled_kernel!(__variadic $flavour, $trait, $trait_fn, $op, $plc, $([$($attr_id:$attr_ty),*])? vec[$ts] -> $u => $($kp)+);
        )+

    };

    (__variadic hybrid, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
            impl crate::kernels::VariadicKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                Vec<<$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type>
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};
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
                            let h = sess.add_operation(op, &handles, &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        } else {
                            Err(crate::error::Error::Unexpected(Some("Variadic hybrid kernel flavor found mixed symbolic and concrete values during compilation.".to_string())))
                        }
                    }
                }))
            }
        }


        impl $trait<
            crate::execution::SymbolicSession,
            <$ts as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                xs: &[$ts as crate::computation::SymbolicType>::Type]
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, VariadicSignature};
                use crate::execution::{Session, SymbolicSession};
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


    (__variadic concrete, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::VariadicKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                Vec<<$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type>
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                use crate::execution::symbolic::{Symbolic, SymbolicSession};

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
                    let kernel_vals: Vec<_> = xs.iter().cloned().filter_map(|x| match x {
                        Symbolic::Concrete(v) => Some(v),
                        Symbolic::Symbolic(_) => None,
                    }).collect();
                    if kernel_vals.len() == xs.len() {
                        // success; we can apply kernel
                        let y = k(sess, plc, kernel_vals)?;
                        Ok(Symbolic::Concrete(y))
                    } else {
                        // operands did not match kernel so record in graph instead
                        let handles: Vec<_> = xs.iter().filter_map(Symbolic::symbolic_handle).map(|h| h.op.as_str()).collect();
                        if handles.len() == xs.len() {
                            // success; we can record in graph
                            let h = sess.add_operation(op, &handles, &plc.clone().into());
                            Ok(Symbolic::Symbolic(h))
                        } else {
                            Err(crate::error::Error::Unexpected(Some("Variadic concrete flavor found mixed symbolic and concrete value during compilation.".to_string())))
                        }
                    }
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$ts as crate::computation::PartiallySymbolicType>::Type,
            <$u as crate::computation::PartiallySymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                xs: &[<$ts as crate::computation::PartiallySymbolicType>::Type]
            ) -> <$u as crate::computation::PartiallySymbolicType>::Type {
                use crate::computation::{KnownType, VariadicSignature, SymbolicValue};
                use crate::execution::{Session};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};
                use std::convert::TryFrom;

                let sig = VariadicSignature {
                    args: <$ts as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };

                let vs: Vec<SymbolicValue> = xs.iter().map(|x| Symbolic::Concrete(x.clone()).into()).collect();
                let y = sess.execute(op.into(), &self.into(), vs).unwrap();
                let y = Symbolic::try_from(y).unwrap();
                match y {
                    Symbolic::Concrete(y) => y,
                    Symbolic::Symbolic(_) => panic!(), // ok since this is concrete flavour
                }
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$ts as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                xs: &[<$ts as crate::computation::SymbolicType>::Type]
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, VariadicSignature, SymbolicValue};
                use crate::execution::{Session};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};
                use std::convert::TryFrom;

                let sig = VariadicSignature {
                    args: <$ts as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };

                let vs: Vec<SymbolicValue> = xs.iter().map(|x| x.clone().into()).collect();
                let y = sess.execute(op.into(), &self.into(), vs).unwrap();
                Symbolic::try_from(y).unwrap()
            }
        }
    };

    (__variadic transparent, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::VariadicKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                Vec<<$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type>
            ) -> crate::error::Result<<$u as KnownType<crate::execution::SymbolicSession>>::Type> + Send>>
            {
                derive_runtime_kernel![variadic, $($kp)+, self]
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$ts as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                xs: &[<$ts as crate::computation::SymbolicType>::Type]
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType};
                use crate::execution::{Session, SymbolicSession};
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

    (__variadic runtime, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        impl crate::kernels::VariadicKernel<
            crate::execution::SymbolicSession,
            $plc,
            <$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type
        > for $op
        {
            fn compile(&self, _plc: &$plc) -> crate::error::Result<Box<dyn Fn(
                &crate::execution::SymbolicSession,
                &$plc,
                Vec<<$ts as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type>
            ) -> crate::error::Result<
                <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type> + Send>
            > {
                use crate::computation::{KnownType};
                use crate::execution::symbolic::{SymbolicSession, Symbolic};

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

                    if res.len() == xs.len() {
                        let h = sess.add_operation(&op, &res, &plc.clone().into());
                        return Ok(Symbolic::Symbolic(h));
                    }

                    Err(crate::error::Error::Unexpected(Some(format!("Variadic runtime kernel found non-Symbolic arguments for {:?}", op))))
                }))
            }
        }

        impl $trait<
            crate::execution::SymbolicSession,
            <$ts as crate::computation::SymbolicType>::Type,
            <$u as crate::computation::SymbolicType>::Type
        > for $plc {
            fn $trait_fn(
                &self,
                sess: &crate::execution::SymbolicSession,
                $($($attr_id:$attr_ty),*,)?
                xs: &[<$ts as crate::computation::SymbolicType>::Type]
            ) -> <$u as crate::computation::SymbolicType>::Type {
                use crate::computation::{KnownType, UnarySignature};
                use crate::execution::{Session, SymbolicSession};
                use std::convert::TryInto;

                let sig = UnarySignature {
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




    // The rules rewriting attributes into each kernel line.
    // Can work for any arity and kind of kernel, but needs a rule per attribute count.

    // Any arity kernel, 1 attribute op
    ($trait:ident::$trait_fn:ident, $op:ident{$attr1_id:ident: $attr1_ty:ty}, [$( ($plc:ty, $($tail:tt)+), )+]) => {
        modelled_kernel! {
            $trait::$trait_fn, $op,
            [
                $(
                    ($plc, [$attr1_id: $attr1_ty] $($tail)+),
                )+
            ]
        }
    };

    // Any arity kernel, 2 attributes op
    ($trait:ident::$trait_fn:ident, $op:ident{$attr1_id:ident: $attr1_ty:ty, $attr2_id:ident: $attr2_ty:ty}, [$( ($plc:ty, $($tail:tt)+), )+]) => {
        modelled_kernel! {
            $trait::$trait_fn, $op,
            [
                $(
                    ($plc, [$attr1_id: $attr1_ty, $attr2_id: $attr2_ty] $($tail)+),
                )+
            ]
        }
    };

    // Any arity kernel, 3 attributes op
    ($trait:ident::$trait_fn:ident, $op:ident{$attr1_id:ident: $attr1_ty:ty, $attr2_id:ident: $attr2_ty:ty, $attr3_id:ident: $attr3_ty:ty}, [$( ($plc:ty, $($tail:tt)+), )+]) => {
        modelled_kernel! {
            $trait::$trait_fn, $op,
            [
                $(
                    ($plc, [$attr1_id: $attr1_ty, $attr2_id: $attr2_ty, $attr3_id: $attr3_ty] $($tail)+),
                )+
            ]
        }
    };

}

macro_rules! modelled_alias {
    /*
    Binary
    */
    ($src_t:ident::$src_f:ident, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $dst_t:ident::$dst_f:ident) => {
        impl $src_t<crate::execution::SyncSession, $t0, $t1, $u> for $plc {
            fn $src_f(&self, sess: &crate::execution::SyncSession, x0: &$t0, x1: &$t1) -> $u {
                $dst_t::$dst_f(self, sess, x0, x1)
            }
        }

        impl $src_t<crate::execution::AsyncSession, $t0, $t1, $u> for $plc {
            fn $src_f(&self, sess: &crate::execution::AsyncSession, x0: &$t0, x1: &$t1) -> $u {
                $dst_t::$dst_f(self, sess, x0, x1)
            }
        }

        impl
            $src_t<
                crate::execution::SymbolicSession,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            > for $plc
        {
            fn $src_f(
                &self,
                ctx: &crate::execution::SymbolicSession,
                x0: &<$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                x1: &<$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            ) -> <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type {
                $dst_t::$dst_f(self, ctx, x0, x1)
            }
        }
    };

    /*
    Ternary
    */
    ($src_t:ident::$src_f:ident, $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $dst_t:ident::$dst_f:ident) => {
        impl $src_t<crate::execution::SyncSession, $t0, $t1, $t2, $u> for $plc {
            fn $src_f(
                &self,
                sess: &crate::execution::SyncSession,
                x0: &$t0,
                x1: &$t1,
                x2: &$t2,
            ) -> $u {
                $dst_t::$dst_f(self, sess, x0, x1, x2)
            }
        }

        impl $src_t<crate::execution::AsyncSession, $t0, $t1, $t2, $u> for $plc {
            fn $src_f(
                &self,
                sess: &crate::execution::AsyncSession,
                x0: &$t0,
                x1: &$t1,
                x2: &$t2,
            ) -> $u {
                $dst_t::$dst_f(self, sess, x0, x1, x2)
            }
        }

        impl
            $src_t<
                crate::execution::SymbolicSession,
                <$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            > for $plc
        {
            fn $src_f(
                &self,
                ctx: &crate::execution::SymbolicSession,
                x0: &<$t0 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                x1: &<$t1 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
                x2: &<$t2 as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type,
            ) -> <$u as crate::computation::KnownType<crate::execution::SymbolicSession>>::Type {
                $dst_t::$dst_f(self, ctx, x0, x1, x2)
            }
        }
    };
}

macro_rules! moose_type {
    // Use this for unparameterised types that are already defined
    ($atomic:ident) => {
        impl crate::computation::PartiallySymbolicType for $atomic {
            type Type = $atomic;
        }

        impl crate::computation::CanonicalType for $atomic {
            type Type = $atomic;
        }

        impl crate::computation::CanonicalType for crate::execution::symbolic::Symbolic<$atomic> {
            type Type = $atomic;
        }

        impl From<$atomic> for <$atomic as crate::computation::SymbolicType>::Type {
            fn from(x: $atomic) -> Self {
                crate::execution::symbolic::Symbolic::Concrete(x)
            }
        }

        impl std::convert::TryFrom<<$atomic as crate::computation::SymbolicType>::Type>
            for $atomic
        {
            type Error = crate::error::Error;

            fn try_from(
                v: <$atomic as crate::computation::SymbolicType>::Type,
            ) -> crate::error::Result<Self> {
                match v {
                    crate::execution::symbolic::Symbolic::Concrete(x) => Ok(x),
                    _ => Err(crate::error::Error::Unexpected(None)), // TODO err message
                }
            }
        }
    };

    // Use this for undefined parameterised types that may be wrapping non-Moose types
    ($combined:ident = [atomic] $t:ty) => {
        pub type $combined = $t;

        impl crate::computation::PartiallySymbolicType for $combined {
            type Type = $combined;
        }

        impl crate::computation::CanonicalType for $combined {
            type Type = $combined;
        }

        impl crate::computation::CanonicalType for crate::execution::symbolic::Symbolic<$combined> {
            type Type = $combined;
        }

        impl From<$combined> for <$combined as crate::computation::SymbolicType>::Type {
            fn from(x: $combined) -> Self {
                crate::execution::symbolic::Symbolic::Concrete(x)
            }
        }

        impl std::convert::TryFrom<<$combined as crate::computation::SymbolicType>::Type>
            for $combined
        {
            type Error = crate::error::Error;

            fn try_from(
                v: <$combined as crate::computation::SymbolicType>::Type,
            ) -> crate::error::Result<Self> {
                match v {
                    crate::execution::symbolic::Symbolic::Concrete(x) => Ok(x),
                    _ => Err(crate::error::Error::Unexpected(None)), // TODO err message
                }
            }
        }
    };

    // Use this for undefined parameterised types that are wrapping a single Moose types
    ($combined:ident = $outer:ident<$inner:ident>) => {
        pub type $combined = $outer<$inner>;

        impl crate::computation::PartiallySymbolicType for $outer<$inner> {
            type Type = $outer<<$inner as crate::computation::SymbolicType>::Type>;
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
            for crate::execution::symbolic::Symbolic<
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
                crate::execution::symbolic::Symbolic::Concrete(x)
            }
        }

        // The kernel macros uses this to determine whether to invoke kernels, and
        // if so, to map symbolic values to (partially) concrete inputs
        impl std::convert::TryFrom<<$combined as crate::computation::SymbolicType>::Type>
            for $outer<<$inner as crate::computation::SymbolicType>::Type>
        {
            type Error = crate::error::Error;

            fn try_from(
                v: <$combined as crate::computation::SymbolicType>::Type,
            ) -> crate::error::Result<Self> {
                match v {
                    crate::execution::symbolic::Symbolic::Concrete(x) => Ok(x),
                    _ => Err(crate::error::Error::Unexpected(None)), // TODO err message
                }
            }
        }
    };

    // Use this for undefined parameterised types that are wrapping two Moose types
    ($combined:ident = $outer:ident<$inner1:ident, $inner2:ident>) => {
        pub type $combined = $outer<$inner1, $inner2>;

        impl crate::computation::PartiallySymbolicType for $outer<$inner1, $inner2> {
            type Type = $outer<
                <$inner1 as crate::computation::SymbolicType>::Type,
                <$inner2 as crate::computation::SymbolicType>::Type,
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
            for crate::execution::symbolic::Symbolic<
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
                crate::execution::symbolic::Symbolic::Concrete(x)
            }
        }

        // The kernel macros uses this to determine whether to invoke kernels, and
        // if so, to map symbolic values to (partially) concrete inputs
        impl std::convert::TryFrom<<$combined as crate::computation::SymbolicType>::Type>
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
                    crate::execution::symbolic::Symbolic::Concrete(x) => Ok(x),
                    _ => Err(crate::error::Error::Unexpected(None)), // TODO err message
                }
            }
        }
    };

    // Use this for undefined parameterised types that are wrapping three Moose types
    ($combined:ident = $outer:ident<$inner1:ident, $inner2:ident, $inner3:ident>) => {
        pub type $combined = $outer<$inner1, $inner2, $inner3>;

        impl crate::computation::PartiallySymbolicType for $outer<$inner1, $inner2, $inner3> {
            type Type = $outer<
                <$inner1 as crate::computation::SymbolicType>::Type,
                <$inner2 as crate::computation::SymbolicType>::Type,
                <$inner3 as crate::computation::SymbolicType>::Type,
            >;
        }

        impl crate::computation::CanonicalType for $outer<$inner1, $inner2, $inner3> {
            type Type = $outer<
                <$inner1 as crate::computation::CanonicalType>::Type,
                <$inner2 as crate::computation::CanonicalType>::Type,
                <$inner3 as crate::computation::CanonicalType>::Type,
            >;
        }

        impl crate::computation::CanonicalType
            for $outer<
                <$inner1 as crate::computation::SymbolicType>::Type,
                <$inner2 as crate::computation::SymbolicType>::Type,
                <$inner3 as crate::computation::SymbolicType>::Type,
            >
        {
            type Type = $outer<
                <$inner1 as crate::computation::CanonicalType>::Type,
                <$inner2 as crate::computation::CanonicalType>::Type,
                <$inner3 as crate::computation::CanonicalType>::Type,
            >;
        }

        impl crate::computation::CanonicalType
            for crate::execution::symbolic::Symbolic<
                $outer<
                    <$inner1 as crate::computation::SymbolicType>::Type,
                    <$inner2 as crate::computation::SymbolicType>::Type,
                    <$inner3 as crate::computation::SymbolicType>::Type,
                >,
            >
        {
            type Type = $outer<
                <$inner1 as crate::computation::CanonicalType>::Type,
                <$inner2 as crate::computation::CanonicalType>::Type,
                <$inner3 as crate::computation::CanonicalType>::Type,
            >;
        }

        // The kernel macro uses this to map (partially) concrete outputs to symbolic values
        impl
            From<
                $outer<
                    <$inner1 as crate::computation::SymbolicType>::Type,
                    <$inner2 as crate::computation::SymbolicType>::Type,
                    <$inner3 as crate::computation::SymbolicType>::Type,
                >,
            > for <$combined as crate::computation::SymbolicType>::Type
        {
            fn from(
                x: $outer<
                    <$inner1 as crate::computation::SymbolicType>::Type,
                    <$inner2 as crate::computation::SymbolicType>::Type,
                    <$inner3 as crate::computation::SymbolicType>::Type,
                >,
            ) -> Self {
                crate::execution::symbolic::Symbolic::Concrete(x)
            }
        }

        // The kernel macros uses this to determine whether to invoke kernels, and
        // if so, to map symbolic values to (partially) concrete inputs
        impl std::convert::TryFrom<<$combined as crate::computation::SymbolicType>::Type>
            for $outer<
                <$inner1 as crate::computation::SymbolicType>::Type,
                <$inner2 as crate::computation::SymbolicType>::Type,
                <$inner3 as crate::computation::SymbolicType>::Type,
            >
        {
            type Error = crate::error::Error;

            fn try_from(
                v: <$combined as crate::computation::SymbolicType>::Type,
            ) -> crate::error::Result<Self> {
                match v {
                    crate::execution::symbolic::Symbolic::Concrete(x) => Ok(x),
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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct N224;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct N256;

pub trait Const {
    const VALUE: usize;
}

impl Const for N64 {
    const VALUE: usize = 64;
}

impl Const for N128 {
    const VALUE: usize = 128;
}

impl Const for N224 {
    const VALUE: usize = 224;
}

impl Const for N256 {
    const VALUE: usize = 256;
}

pub trait Ring {
    type BitLength: Const;
}

pub trait BitArray {
    type Len: Const;
}

macro_rules! unmodelled {
    /*
    Nullary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? () -> $u:ty, $op:ident) => {
        impl crate::kernels::NullaryKernelCheck<crate::execution::SyncSession, $plc, $u> for $op {}
    };

    /*
    Unary
    */
    ($plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty) -> $u:ty, $op:ident) => {
        impl crate::kernels::UnaryKernelCheck<crate::execution::SyncSession, $plc, $t0, $u>
            for $op
        {
        }
    };

    /*
    Binary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty, $t1:ty) -> $u:ty, $op:ident) => {
        impl crate::kernels::BinaryKernelCheck<crate::execution::SyncSession, $plc, $t0, $t1, $u>
            for $op
        {
        }
    };

    /*
    Ternary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $op:ident) => {
        impl
            crate::kernels::TernaryKernelCheck<
                crate::execution::SyncSession,
                $plc,
                $t0,
                $t1,
                $t2,
                $u,
            > for $op
        {
        }
    };
}

pub mod additive;
pub mod boolean;
pub mod bristol_fashion;
pub mod compilation;
pub mod computation;
pub mod encrypted;
pub mod error; // TODO make non-pub
pub mod execution;
pub mod fixedpoint;
pub mod floatingpoint;
pub mod host;
pub mod kernels;
pub mod logical;
pub mod mirrored;
pub mod networking;
pub mod prelude;
pub mod prng;
pub mod replicated;
pub mod storage;
pub mod textual;
pub mod types;

#[doc(inline)]
pub use error::{Error, Result};
