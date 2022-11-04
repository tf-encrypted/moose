//! Core Moose framework.

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

macro_rules! derive_execute_kernel {

    /* Nullary */

    ($plc:ty, () -> $u:ty, $(attributes[$($_attrs:tt)*])? custom |$op_ke:ident| $ke:expr, $op:ident) => {{
        let kf: &dyn Fn(&Self) -> crate::error::Result<
            crate::kernels::TypedNullaryKernel<
                _,
                $plc,
                $u,
            >
        > = &|$op_ke| $ke;
        let k = kf(&$op)?;
        crate::execution::kernel_helpers::nullary::<
            _,
            $u,
            $plc,
            Box<_>,
        >(k)
    }};

    ($plc:ty, () -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {{
        $(
            let $attr = $op.$attr.clone();
        )+
        let k: crate::kernels::TypedNullaryKernel<
            _,
            _,
            _,
        > = Box::new(move |sess, plc| {
            $k(sess, &plc, $($attr.clone()),+)
        });
        crate::execution::kernel_helpers::nullary::<
            _,
            $u,
            $plc,
            Box<_>,
        >(k)
    }};

    ($plc:ty, () -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::nullary::<
            _,
            $u,
            $plc,
            fn(&_, &_) -> _,
        >($k)
    };

    /* Unary */

    ($plc:ty, ($t0:ty) -> $u:ty, $(attributes[$($_attrs:tt)*])? custom |$op_ke:ident| $ke:expr, $op:ident) => {{
        let kf: &dyn Fn(&Self) -> crate::error::Result<
            crate::kernels::TypedUnaryKernel<
                _,
                $plc,
                $t0,
                $u,
            >
        > = &|$op_ke| $ke;
        let k = kf(&$op)?;
        crate::execution::kernel_helpers::unary::<
            _,
            $t0,
            $u,
            $plc,
            Box<_>,
        >(k)
    }};

    ($plc:ty, ($t0:ty) -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {{
        $(
            let $attr = $op.$attr.clone();
        )+
        let k: crate::kernels::TypedUnaryKernel<
            _,
            $plc,
            $t0,
            $u,
        > = Box::new(move |sess, plc, x0| {
            $k(sess, &plc, $($attr.clone()),+, x0)
        });
        crate::execution::kernel_helpers::unary::<
            _,
            $t0,
            $u,
            $plc,
            Box<_>,
        >(k)
    }};

    ($plc:ty, ($t0:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::unary::<
            _,
            $t0,
            $u,
            $plc,
            fn(&_, &_, _) -> _,
        >($k)
    };

    /* Binary */

    ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty, custom |$op_ke:ident| $ke:expr, $op:ident) => {{
        let kf: &dyn Fn(&Self) -> crate::error::Result<
            crate::kernels::TypedBinaryKernel<
                _,
                $plc,
                $t0,
                $t1,
                $u,
            >
        > = &|$op_ke| $ke;
        let k = kf(&$op)?;
        crate::execution::kernel_helpers::binary::<
            _,
            $t0,
            $t1,
            $u,
            $plc,
            Box<_>,
        >(k)
    }};

    ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty, attributes[$($attr:ident$(: $prim_ty:ident)?),+] $k:path, $op:ident) => {{
        $(
            let $attr = $op.$attr.clone();
        )+
        let k: crate::kernels::TypedBinaryKernel<
            _,
            $plc,
            $t0,
            $t1,
            $u,
        > = Box::new(move |sess, plc, x0, x1| {
            $k(sess, &plc, $($attr.clone()),+, x0, x1)
        });
        crate::execution::kernel_helpers::binary::<
            _,
            $t0,
            $t1,
            $u,
            $plc,
            Box<_>,
        >(k)
    }};

    ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::binary::<
            _,
            $t0,
            $t1,
            $u,
            $plc,
            fn(&_, &_, _, _) -> _,
        >($k)
    };

    /* Ternary */

    ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::ternary::<
            _,
            $t0,
            $t1,
            $t2,
            $u,
            $plc,
            fn(&_, &_, _, _, _) -> _,
        >($k)
    };

    /* Variadic */

    ($plc:ty, vec[$ts:ty] -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {
        {
            $(
                let $attr = $op.$attr.clone();
            )+
            let k: crate::kernels::TypedVariadicKernel<
                _,
                $plc,
                $ts,
                $u,
            > = Box::new(move |sess, plc, ts| {
                $k(sess, &plc, $($attr.clone()),+, ts)
            });
            crate::execution::kernel_helpers::variadic::<
                _,
                $ts,
                $u,
                $plc,
                Box<_>,
            >(k)
        }
    };

    ($plc:ty, vec[$ts:ty] -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::variadic::<
            _,
            $ts,
            $u,
            $plc,
            fn(&_, &_, &[$ts]) -> _,
        >($k)
    };
}

#[cfg(feature = "compile")]
macro_rules! derive_symbolic_kernel {

    /* Nullary */

    (runtime $plc:ty, () -> $u:ty, $(attributes[$($_attrs:tt)*])? custom |$op_ke:ident| $ke:expr, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::nullary::<$u, $plc>(Operator::from($op.clone()))
    };

    (runtime $plc:ty, () -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::nullary::<$u, $plc>(Operator::from($op.clone()))
    };

    (runtime $plc:ty, () -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::nullary::<$u, $plc>(Operator::from($op.clone()))
    };

    (concrete $plc:ty, () -> $u:ty, $(attributes[$($_attrs:tt)*])? custom |$op_ke:ident| $ke:expr, $op:ident) => {{
        let kf: &dyn Fn(&Self) -> crate::error::Result<
            crate::kernels::TypedNullaryKernel<
                crate::execution::SymbolicSession,
                _,
                _,
            >
        > = &|$op_ke| $ke;
        let k = kf(&$op)?;
        crate::execution::kernel_helpers::symbolic::concrete::nullary::<
            $u,
            $plc,
            Box<_>,
        >(k)
    }};

    (concrete $plc:ty, () -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {{
        $(
            let $attr = $op.$attr.clone();
        )+
        let k: crate::kernels::TypedNullaryKernel<
            crate::execution::SymbolicSession,
            _,
            _,
        > = Box::new(move |sess, plc| {
            $k(sess, &plc, $($attr.clone()),+)
        });
        crate::execution::kernel_helpers::symbolic::concrete::nullary::<
            $u,
            $plc,
            Box<_>,
        >(k)
    }};

    (concrete $plc:ty, () -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::concrete::nullary::<
            $u,
            $plc,
            fn(&_, &_) -> _,
        >(Operator::from($op.clone()), k)
    };

    /* Unary */

    (runtime $plc:ty, ($t0:ty) -> $u:ty, $(attributes[$($_attrs:tt)*])? custom |$op_ke:ident| $ke:expr, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::unary::<$t0, $u, $plc>(Operator::from($op.clone()))
    };

    (runtime $plc:ty, ($t0:ty) -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::unary::<$t0, $u, $plc>(Operator::from($op.clone()))
    };

    (runtime $plc:ty, ($t0:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::unary::<$t0, $u, $plc>(Operator::from($op.clone()))
    };

    (concrete $plc:ty, ($t0:ty) -> $u:ty, $(attributes[$($_attrs:tt)*])? custom |$op_ke:ident| $ke:expr, $op:ident) => {{
        let kf: &dyn Fn(&Self) -> crate::error::Result<
            crate::kernels::TypedUnaryKernel<
                crate::execution::SymbolicSession,
                _,
                _,
                _,
            >
        > = &|$op_ke| $ke;
        let k = kf(&$op)?;
        crate::execution::kernel_helpers::symbolic::concrete::unary::<
            $t0,
            $u,
            $plc,
            Box<_>,
        >(Operator::from($op.clone()), k)
    }};

    (concrete $plc:ty, ($t0:ty) -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {{
        $(
            let $attr = $op.$attr.clone();
        )+
        let k: crate::kernels::TypedUnaryKernel<
            crate::execution::SymbolicSession,
            _,
            _,
            _,
        > = Box::new(move |sess, plc, x0| {
            $k(sess, &plc, $($attr.clone()),+, x0)
        });
        crate::execution::kernel_helpers::symbolic::concrete::unary::<
            $t0,
            $u,
            $plc,
            Box<_>,
        >(Operator::from($op.clone()), k)
    }};

    (concrete $plc:ty, ($t0:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::concrete::unary::<
            $t0,
            $u,
            $plc,
            fn(&_, &_, _) -> _,
        >(Operator::from($op.clone()), $k)
    };

    (transparent $plc:ty, ($t0:ty) -> $u:ty, $(attributes[$($_attrs:tt)*])? custom |$op_ke:ident| $ke:expr, $op:ident) => {{
        let kf: &dyn Fn(&Self) -> crate::error::Result<
            crate::kernels::TypedUnaryKernel<
                crate::execution::SymbolicSession,
                _,
                _,
                _,
            >
        > = &|$op_ke| $ke;
        let k = kf(&$op)?;
        crate::execution::kernel_helpers::symbolic::transparent::unary::<
            $t0,
            $u,
            $plc,
            Box<_>,
        >(k)
    }};

    (transparent $plc:ty, ($t0:ty) -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {{
        $(
            let $attr = $op.$attr.clone();
        )+
        let k: crate::kernels::TypedUnaryKernel<
            crate::execution::SymbolicSession,
            _,
            _,
            _,
        > = Box::new(move |sess, plc, x0| {
            $k(sess, &plc, $($attr.clone()),+, x0)
        });
        crate::execution::kernel_helpers::symbolic::transparent::unary::<
            $t0,
            $u,
            $plc,
            Box<_>,
        >(k)
    }};

    (transparent $plc:ty, ($t0:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::transparent::unary::<
            $t0,
            $u,
            $plc,
            fn(&_, &_, _) -> _,
        >($k)
    };

    (hybrid $plc:ty, ($t0:ty) -> $u:ty, $(attributes[$($_attrs:tt)*])? custom |$op_ke:ident| $ke:expr, $op:ident) => {{
        let kf: &dyn Fn(&Self) -> crate::error::Result<
            crate::kernels::TypedUnaryKernel<
                crate::execution::SymbolicSession,
                _,
                _,
                _,
            >
        > = &|$op_ke| $ke;
        let k = kf(&$op)?;
        crate::execution::kernel_helpers::symbolic::hybrid::unary::<
            $t0,
            $u,
            _,
            _,
            $plc,
            Box<_>,
        >(Operator::from($op.clone()), k)
    }};

    (hybrid $plc:ty, ($t0:ty) -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {{
        $(
            let $attr = $op.$attr.clone();
        )+
        let k: crate::kernels::TypedUnaryKernel<
            crate::execution::SymbolicSession,
            _,
            _,
            _,
        > = Box::new(move |sess, plc, x0| {
            $k(sess, &plc, $($attr.clone()),+, x0)
        });
        crate::execution::kernel_helpers::symbolic::hybrid::unary::<
            $t0,
            $u,
            _,
            _,
            $plc,
            Box<_>,
        >(Operator::from($op.clone()), k)
    }};

    (hybrid $plc:ty, ($t0:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::hybrid::unary::<
            $t0,
            $u,
            _,
            _,
            $plc,
            fn(&_, &_, _) -> _,
        >(Operator::from($op.clone()), $k)
    };

    /* Binary */

    (runtime $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, custom |$op_ke:ident| $ke:expr, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::binary::<$t0, $t1, $u, $plc>(Operator::from($op.clone()))
    };

    (runtime $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, attributes[$($attr:ident$(: $prim_ty:ident)?),+] $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::binary::<$t0, $t1, $u, $plc>(Operator::from($op.clone()))
    };

    (runtime $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::binary::<$t0, $t1, $u, $plc>(Operator::from($op.clone()))
    };

    (concrete $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, custom |$op_ke:ident| $ke:expr, $op:ident) => {{
        let kf: &dyn Fn(&Self) -> crate::error::Result<
            crate::kernels::TypedBinaryKernel<
                crate::execution::SymbolicSession,
                _,
                _,
                _,
                _,
            >
        > = &|$op_ke| $ke;
        let k = kf(&$op)?;
        crate::execution::kernel_helpers::symbolic::concrete::binary::<
            $t0,
            $u,
            $plc,
            Box<_>,
        >(Operator::from($op.clone()), k)
    }};

    (concrete $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, attributes[$($attr:ident$(: $prim_ty:ident)?),+] $k:path, $op:ident) => {{
        $(
            let $attr = $op.$attr.clone();
        )+
        let k: crate::kernels::TypedBinaryKernel<
            crate::execution::SymbolicSession,
            _,
            _,
            _,
            _,
        > = Box::new(move |sess, plc, x0, x1| {
            $k(sess, &plc, $($attr.clone()),+, x0, x1)
        });
        crate::execution::kernel_helpers::symbolic::concrete::binary::<
            $t0,
            $t1,
            $u,
            $plc,
            Box<_>,
        >(Operator::from($op.clone()), k)
    }};

    (concrete $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::concrete::binary::<
            $t0,
            $t1,
            $u,
            $plc,
            fn(&_, &_, _, _) -> _,
        >(Operator::from($op.clone()), $k)
    };

    (transparent $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, custom |$op_ke:ident| $ke:expr, $op:ident) => {{
        let kf: &dyn Fn(&Self) -> crate::error::Result<
            crate::kernels::TypedBinaryKernel<
                crate::execution::SymbolicSession,
                _,
                _,
                _,
                _,
            >
        > = &|$op_ke| $ke;
        let k = kf(&$op)?;
        crate::execution::kernel_helpers::symbolic::transparent::binary::<
            $t0,
            $t1,
            $u,
            $plc,
            Box<_>,
        >(Operator::from($op.clone()), k)
    }};

    (transparent $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {{
        $(
            let $attr = $op.$attr.clone();
        )+
        let k: crate::kernels::TypedBinaryKernel<
            crate::execution::SymbolicSession,
            _,
            _,
            _,
            _,
        > = Box::new(move |sess, plc, x0, x1| {
            $k(sess, &plc, $($attr.clone()),+, x0, x1)
        });
        crate::execution::kernel_helpers::symbolic::transparent::binary::<
            $t0,
            $t1,
            $u,
            $plc,
            Box<_>,
        >(Operator::from($op.clone()), k)
    }};

    (transparent $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::transparent::binary::<
            $t0,
            $t1,
            $u,
            $plc,
            fn(&_, &_, _, _) -> _,
        >($k)
    };

    (hybrid $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, custom |$op_ke:ident| $ke:expr, $op:ident) => {{
        let kf: &dyn Fn(&Self) -> crate::error::Result<
            crate::kernels::TypedBinaryKernel<
                crate::execution::SymbolicSession,
                _,
                _,
                _,
                _,
            >
        > = &|$op_ke| $ke;
        let k = kf(&$op)?;
        crate::execution::kernel_helpers::symbolic::hybrid::binary::<
            $t0,
            $t1,
            $u,
            _,
            _,
            _,
            $plc,
            Box<_>,
        >(Operator::from($op.clone()), k)
    }};

    (hybrid $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {{
        $(
            let $attr = $op.$attr.clone();
        )+
        let k: crate::kernels::TypedBinaryKernel<
            crate::execution::SymbolicSession,
            _,
            _,
            _,
            _,
        > = Box::new(move |sess, plc, x0, x1| {
            $k(sess, &plc, $($attr.clone()),+, x0, x1)
        });
        crate::execution::kernel_helpers::symbolic::hybrid::binary::<
            $t0,
            $t1,
            $u,
            _,
            _,
            _,
            $plc,
            Box<_>,
        >(Operator::from($op.clone()), k)
    }};

    (hybrid $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::hybrid::binary::<
            $t0,
            $t1,
            $u,
            _,
            _,
            _,
            $plc,
            fn(&_, &_, _, _) -> _,
        >(Operator::from($op.clone()), $k)
    };

    /* Ternary */

    (runtime $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::ternary::<
            $t0,
            $t1,
            $t2,
            $u,
            $plc,
        >(Operator::from($op.clone()))
    };

    (concrete $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::concrete::ternary::<
            $t0,
            $t1,
            $t2,
            $u,
            $plc,
            fn(&_, &_, _, _, _) -> _,
        >(
            Operator::from($op.clone()),
            $k,
        )
    };

    (transparent $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::transparent::ternary::<
            $t0,
            $t1,
            $t2,
            $u,
            $plc,
            fn(&_, &_, _, _, _) -> _,
        >(Operator::from($op.clone()), $k)
    };

    (hybrid $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::hybrid::ternary::<
            $t0,
            $t1,
            $t2,
            $u,
            _,
            _,
            _,
            _,
            $plc,
            fn(&_, &_, _, _, _) -> _,
        >(
            Operator::from($op.clone()),
            $k,
        )
    };

    /* Variadic */

    (runtime $plc:ty, vec[$ts:ty] -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::variadic::<$ts, $u, $plc>(Operator::from($op.clone()))
    };

    (runtime $plc:ty, vec[$ts:ty] -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::runtime::variadic::<$ts, $u, $plc>(Operator::from($op.clone()))
    };

    (transparent $plc:ty, vec[$ts:ty] -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::transparent::variadic::<
            $ts,
            $u,
            $plc,
            fn(&_, &_, &_) -> _,
        >($k)
    };

    (concrete $plc:ty, vec[$ts:ty] -> $u:ty, attributes[$($attr:ident),+] $k:path, $op:ident) => {
        {
            $(
                let $attr = $op.$attr.clone();
            )+
            let k: crate::kernels::TypedVariadicKernel<
                crate::execution::SymbolicSession,
                _,
                _,
                _,
            > = Box::new(move |sess, plc, ts| {
                $k(sess, &plc, $($attr.clone()),+, ts)
            });
            crate::execution::kernel_helpers::symbolic::concrete::variadic::<
                $ts,
                $u,
                $plc,
                Box<_>,
            >(Operator::from($op.clone()), k)
        }
    };

    (concrete $plc:ty, vec[$ts:ty] -> $u:ty, $k:path, $op:ident) => {
        crate::execution::kernel_helpers::symbolic::concrete::variadic::<
            $ts,
            $u,
            $plc,
            fn(&_, &_, &_) -> _,
        >(Operator::from($op.clone()), $k)
    };
}

#[allow(unused_macros)]
macro_rules! operands {
    ($($content:tt)*) => {
        vec![$($content)*]
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
macro_rules! modelled_kernel {

    /*
    Nullary
    */

    ($trait:ident::$trait_fn:ident, $op:ident, [$( ($plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? () -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        #[cfg(feature = "sync_execute")]
        impl crate::kernels::DispatchKernel<crate::execution::SyncSession, crate::computation::Value> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::SyncSession, crate::computation::Value>> {
                use crate::execution::SyncSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature {
                                ret: <$u as KnownType<SyncSession>>::TY,
                            })
                        ) => {
                            derive_execute_kernel![$plc, () -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        #[cfg(feature = "compile")]
        impl crate::kernels::DispatchKernel<crate::execution::SymbolicSession, crate::computation::SymbolicValue> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::SymbolicSession, crate::computation::SymbolicValue>> {
                use crate::execution::SymbolicSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature{
                                ret: <$u as KnownType<SymbolicSession>>::TY,
                            })
                        ) => {
                            derive_symbolic_kernel![$flavour $plc, () -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        #[cfg(feature = "async_execute")]
        impl crate::kernels::DispatchKernel<crate::execution::AsyncSession, crate::computation::Value> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::AsyncSession, crate::computation::Value>> {
                use crate::execution::AsyncSession;
                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature {
                                ret: <$u as KnownType<AsyncSession>>::TY,
                            })
                        ) => {
                            derive_execute_kernel![$plc, () -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        $(
            #[cfg(feature = "sync_execute")]
            impl $trait<
                crate::execution::SyncSession,
                $u
            > for $plc {
                fn $trait_fn(
                    &self,
                    sess: &crate::execution::SyncSession,
                    $($($attr_id:$attr_ty,)*)?
                ) -> $u {
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
                        &op.into(),
                        &self.into(),
                        operands![],
                    ).unwrap();
                    y.try_into().unwrap()
                }
            }

            #[cfg(feature = "async_execute")]
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

            // trait implementations of SymbolicSession (which are based on flavour)
            modelled_kernel!(__nullary $flavour, $trait, $trait_fn, $op, $plc, $([$($attr_id:$attr_ty),*])? () -> $u => $($kp)+);
        )+
    };

    (__nullary hybrid, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? () -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    (__nullary concrete, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? () -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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

                let y = sess.execute(&op.into(), &self.into(), operands![]).unwrap();
                let y = Symbolic::try_from(y).unwrap();
                match y {
                    Symbolic::Concrete(y) => y,
                    Symbolic::Symbolic(_) => panic!(), // ok since this is concrete flavour
                }
            }
        }

        #[cfg(feature = "compile")]
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

                let y = sess.execute(&op.into(), &self.into(), operands![]).unwrap();
                Symbolic::try_from(y).unwrap()
            }
        }
    };

    (__nullary transparent, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? () -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    (__nullary runtime, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? () -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![])
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
        #[cfg(feature = "sync_execute")]
        impl crate::kernels::DispatchKernel<crate::execution::SyncSession, crate::computation::Value> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::SyncSession, crate::computation::Value>> {
                use crate::execution::SyncSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Unary(UnarySignature{
                                arg0: <$t0 as KnownType<SyncSession>>::TY,
                                ret: <$u as KnownType<SyncSession>>::TY,
                            })
                        ) => {
                            derive_execute_kernel![$plc, ($t0) -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        #[cfg(feature = "compile")]
        impl crate::kernels::DispatchKernel<crate::execution::SymbolicSession, crate::computation::SymbolicValue> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::SymbolicSession, crate::computation::SymbolicValue>> {
                use crate::execution::SymbolicSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Unary(UnarySignature{
                                arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                                ret: <$u as KnownType<SymbolicSession>>::TY,
                            })
                        ) => {
                            derive_symbolic_kernel![$flavour $plc, ($t0) -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        #[cfg(feature = "async_execute")]
        impl crate::kernels::DispatchKernel<crate::execution::AsyncSession, crate::computation::Value> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::AsyncSession, crate::computation::Value>> {
                use crate::execution::AsyncSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Unary(UnarySignature{
                                arg0: <$t0 as KnownType<AsyncSession>>::TY,
                                ret: <$u as KnownType<AsyncSession>>::TY,
                            })
                        ) => {
                            derive_execute_kernel![$plc, ($t0) -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        $(
            #[cfg(feature = "sync_execute")]
            impl $trait<
                crate::execution::SyncSession,
                $t0,
                $u
            > for $plc {
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
                        &op.into(),
                        &self.into(),
                        operands![x0],
                    ).unwrap();
                    y.try_into().unwrap()
                }
            }

            #[cfg(feature = "async_execute")]
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

            // trait implementations of SymbolicSession (which are based on flavour)
            modelled_kernel!(__unary $flavour, $trait, $trait_fn, $op, $plc, $([$($attr_id:$attr_ty),*])? ($t0) -> $u => $($kp)+);
        )+
    };

    (__unary hybrid, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![x0.clone().into()])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    (__unary concrete, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                let y = sess.execute(&op.into(), &self.into(), operands![x0]).unwrap();
                let y = Symbolic::try_from(y).unwrap();
                match y {
                    Symbolic::Concrete(y) => y,
                    Symbolic::Symbolic(_) => panic!(), // ok since this is concrete flavour
                }
            }
        }

        #[cfg(feature = "compile")]
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
                let y = sess.execute(&op.into(), &self.into(), operands![x0]).unwrap();
                Symbolic::try_from(y).unwrap()
            }
        }
    };

    (__unary transparent, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![x0.clone().into()])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    (__unary runtime, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![x0.clone().into()])
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
        #[cfg(feature = "sync_execute")]
        impl crate::kernels::DispatchKernel<crate::execution::SyncSession, crate::computation::Value> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::SyncSession, crate::computation::Value>> {
                use crate::execution::SyncSession;

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
                            derive_execute_kernel![$plc, ($t0, $t1) -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        #[cfg(feature = "compile")]
        impl crate::kernels::DispatchKernel<crate::execution::SymbolicSession, crate::computation::SymbolicValue> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::SymbolicSession, crate::computation::SymbolicValue>> {
                use crate::execution::SymbolicSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Binary(BinarySignature{
                                arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                                arg1: <$t1 as KnownType<SymbolicSession>>::TY,
                                ret: <$u as KnownType<SymbolicSession>>::TY,
                            })
                        ) => {
                            derive_symbolic_kernel![$flavour $plc, ($t0, $t1) -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        #[cfg(feature = "async_execute")]
        impl crate::kernels::DispatchKernel<crate::execution::AsyncSession, crate::computation::Value> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::AsyncSession, crate::computation::Value>> {
                use crate::execution::AsyncSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Binary(BinarySignature{
                                arg0: <$t0 as KnownType<AsyncSession>>::TY,
                                arg1: <$t1 as KnownType<AsyncSession>>::TY,
                                ret: <$u as KnownType<AsyncSession>>::TY,
                            })
                        ) => {
                            derive_execute_kernel![$plc, ($t0, $t1) -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        $(
            #[cfg(feature = "sync_execute")]
            impl $trait<
                crate::execution::SyncSession,
                $t0,
                $t1,
                $u
            > for $plc {
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
                        &op.into(),
                        &self.into(),
                        operands![x0.clone().into(), x1.clone().into()],
                    )
                    .unwrap()
                    .try_into()
                    .unwrap()
                }
            }

            #[cfg(feature = "async_execute")]
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

            // trait implementations of SymbolicSession (which are based on flavour)
            modelled_kernel!(__binary $flavour, $trait, $trait_fn, $op, $plc, $([$($attr_id:$attr_ty),*])? ($t0, $t1) -> $u => $($kp)+);
        )+
    };

    (__binary hybrid, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![x0.clone().into(), x1.clone().into()])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    (__binary concrete, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                let y = sess.execute(&op.into(), &self.into(), operands![x0, x1]).unwrap();
                let y = Symbolic::try_from(y).unwrap();
                match y {
                    Symbolic::Concrete(y) => y,
                    Symbolic::Symbolic(_) => panic!(), // ok since this is concrete flavour
                }
            }
        }

        #[cfg(feature = "compile")]
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
                let y = sess.execute(&op.into(), &self.into(), operands![x0, x1]).unwrap();
                Symbolic::try_from(y).unwrap()
            }
        }
    };

    (__binary transparent, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![x0.clone().into(), x1.clone().into()])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    (__binary runtime, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![x0.clone().into(), x1.clone().into()])
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
        #[cfg(feature = "sync_execute")]
        impl crate::kernels::DispatchKernel<crate::execution::SyncSession, crate::computation::Value> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::SyncSession, crate::computation::Value>> {
                use crate::execution::SyncSession;

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
                            derive_execute_kernel![$plc, ($t0, $t1, $t2) -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        #[cfg(feature = "compile")]
        impl crate::kernels::DispatchKernel<crate::execution::SymbolicSession, crate::computation::SymbolicValue> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::SymbolicSession, crate::computation::SymbolicValue>> {
                use crate::execution::SymbolicSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Ternary(TernarySignature{
                                arg0: <$t0 as KnownType<SymbolicSession>>::TY,
                                arg1: <$t1 as KnownType<SymbolicSession>>::TY,
                                arg2: <$t2 as KnownType<SymbolicSession>>::TY,
                                ret: <$u as KnownType<SymbolicSession>>::TY,
                            })
                        ) => {
                            derive_symbolic_kernel![$flavour $plc, ($t0, $t1, $t2) -> $u, $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        #[cfg(feature = "async_execute")]
        impl crate::kernels::DispatchKernel<crate::execution::AsyncSession, crate::computation::Value> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::AsyncSession, crate::computation::Value>> {
                use crate::execution::AsyncSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Ternary(TernarySignature{
                                arg0: <$t0 as KnownType<AsyncSession>>::TY,
                                arg1: <$t1 as KnownType<AsyncSession>>::TY,
                                arg2: <$t2 as KnownType<AsyncSession>>::TY,
                                ret: <$u as KnownType<AsyncSession>>::TY,
                            })
                        ) => {
                            derive_execute_kernel![$plc, ($t0, $t1, $t2) -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        $(
            #[cfg(feature = "sync_execute")]
            impl $trait<
                crate::execution::SyncSession,
                $t0,
                $t1,
                $t2,
                $u
            > for $plc {
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
                        &op.into(),
                        &self.into(),
                        operands![x0.clone().into(), x1.clone().into(), x2.clone().into()],
                    )
                    .unwrap()
                    .try_into()
                    .unwrap()
                }
            }

            #[cfg(feature = "async_execute")]
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

            // trait implementations of SymbolicSession (which are based on flavour)
            modelled_kernel!(__ternary $flavour, $trait, $trait_fn, $op, $plc, $([$($attr_id:$attr_ty),*])? ($t0, $t1, $t2) -> $u => $($kp)+);
        )+
    };

    (__ternary hybrid, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![x0.clone().into(), x1.clone().into(), x2.clone().into()])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    (__ternary concrete, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                let y = sess.execute(&op.into(), &self.into(), operands![x0, x1, x2]).unwrap();
                let y = Symbolic::try_from(y).unwrap();
                match y {
                    Symbolic::Concrete(y) => y,
                    Symbolic::Symbolic(_) => panic!(), // ok since this is concrete flavour
                }
            }
        }

        #[cfg(feature = "compile")]
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
                let y = sess.execute(&op.into(), &self.into(), operands![x0, x1, x2]).unwrap();
                Symbolic::try_from(y).unwrap()
            }
        }

    };

    (__ternary transparent, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![x0.clone().into(), x1.clone().into(), x2.clone().into()])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    (__ternary runtime, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                sess.execute(&op.into(), &self.into(), operands![x0.clone().into(), x1.clone().into(), x2.clone().into()])
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }

    };

    // Variadic

    ($trait:ident::$trait_fn:ident, $op:ident, [$( ($plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? vec[$ts:ty] -> $u:ty => [$flavour:tt] $($kp:tt)+), )+]) => {
        #[cfg(feature = "sync_execute")]
        impl crate::kernels::DispatchKernel<crate::execution::SyncSession, crate::computation::Value> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::SyncSession, crate::computation::Value>> {
                use crate::execution::SyncSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Variadic(VariadicSignature{
                                args: <$ts as KnownType<SyncSession>>::TY,
                                ret: <$u as KnownType<SyncSession>>::TY,
                            })
                        ) => {
                            derive_execute_kernel![$plc, vec[$ts] -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        #[cfg(feature = "compile")]
        impl crate::kernels::DispatchKernel<crate::execution::SymbolicSession, crate::computation::SymbolicValue> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::SymbolicSession, crate::computation::SymbolicValue>> {
                use crate::execution::SymbolicSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Variadic(VariadicSignature{
                                args: <$ts as KnownType<SymbolicSession>>::TY,
                                ret: <$u as KnownType<SymbolicSession>>::TY,
                            })
                        ) => {
                            derive_symbolic_kernel![$flavour $plc, vec[$ts] -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        #[cfg(feature = "async_execute")]
        impl crate::kernels::DispatchKernel<crate::execution::AsyncSession, crate::computation::Value> for $op {
            fn compile(
                &self,
                plc: &crate::computation::Placement
            ) -> crate::error::Result<crate::kernels::Kernel<crate::execution::AsyncSession, crate::computation::Value>> {
                use crate::execution::AsyncSession;

                match (plc.ty(), self.sig.flatten()) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Variadic(VariadicSignature{
                                args: <$ts as KnownType<AsyncSession>>::TY,
                                ret: <$u as KnownType<AsyncSession>>::TY,
                            })
                        ) => {
                            derive_execute_kernel![$plc, vec[$ts] -> $u, $(attributes[$($attr_id),+])? $($kp)+, self]
                        }
                    )+
                    _ => Err(crate::error::Error::UnimplementedOperator(format!("{:?}", self)))
                }
            }
        }

        $(
            #[cfg(feature = "sync_execute")]
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
                    use crate::execution::{Session, SyncSession, Operands};
                    use std::convert::TryInto;

                    let sig = VariadicSignature {
                        args: <$ts as KnownType<SyncSession>>::TY,
                        ret: <$u as KnownType<SyncSession>>::TY,
                    };
                    let op = $op {
                        sig: sig.into(),
                        $($($attr_id),*)?
                    };
                    let vs: Operands<Value> = xs.iter().map(|x| x.clone().into()).collect();
                    sess.execute(&op.into(), &self.into(), vs)
                        .unwrap()
                        .try_into()
                        .unwrap()
                }
            }

            #[cfg(feature = "async_execute")]
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

            // trait implementations of SymbolicSession (which are based on flavour)
            modelled_kernel!(__variadic $flavour, $trait, $trait_fn, $op, $plc, $([$($attr_id:$attr_ty),*])? vec[$ts] -> $u => $($kp)+);
        )+
    };

    (__variadic hybrid, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                use crate::execution::{Session, SymbolicSession, Operands};
                use std::convert::TryInto;

                let sig = VariadicSignature {
                    args: <$ts as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                let vs: Operands<SymbolicValue> = xs.iter().map(|x| x.clone().into()).collect();
                sess.execute(&op.into(), &self.into(), vs)
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    (__variadic concrete, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                use crate::execution::{Session, Operands};
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

                let vs: Operands<SymbolicValue> = xs.iter().map(|x| Symbolic::Concrete(x.clone()).into()).collect();
                let y = sess.execute(&op.into(), &self.into(), vs).unwrap();
                let y = Symbolic::try_from(y).unwrap();
                match y {
                    Symbolic::Concrete(y) => y,
                    Symbolic::Symbolic(_) => panic!(), // ok since this is concrete flavour
                }
            }
        }

        #[cfg(feature = "compile")]
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
                use crate::execution::{Session, Operands};
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

                let vs: Operands<SymbolicValue> = xs.iter().map(|x| x.clone().into()).collect();
                let y = sess.execute(&op.into(), &self.into(), vs).unwrap();
                Symbolic::try_from(y).unwrap()
            }
        }
    };

    (__variadic transparent, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                use crate::execution::{Session, SymbolicSession, Operands};
                use std::convert::TryInto;

                let sig = VariadicSignature {
                    args: <$ts as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                let vs: Operands<SymbolicValue> = xs.iter().map(|x| x.clone().into()).collect();
                sess.execute(&op.into(), &self.into(), vs)
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        }
    };

    (__variadic runtime, $trait:ident, $trait_fn:ident, $op:ident, $plc:ty, $([$($attr_id:ident: $attr_ty:ty),+])? vec[$ts:ty] -> $u:ty => $($kp:tt)+) => {
        #[cfg(feature = "compile")]
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
                use crate::computation::{KnownType, VariadicSignature};
                use crate::execution::{Session, SymbolicSession, Operands};
                use std::convert::TryInto;

                let sig = VariadicSignature {
                    args: <$ts as KnownType<SymbolicSession>>::TY,
                    ret: <$u as KnownType<SymbolicSession>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };

                let vs: Operands<SymbolicValue> = xs.iter().map(|x| x.clone().into()).collect();
                sess.execute(&op.into(), &self.into(), vs)
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

macro_rules! moose_type {
    // Use this for unparameterised types that are already defined
    ($atomic:ident) => {
        #[cfg(feature = "compile")]
        impl crate::computation::PartiallySymbolicType for $atomic {
            type Type = $atomic;
        }

        impl crate::computation::CanonicalType for $atomic {
            type Type = $atomic;
        }

        #[cfg(feature = "compile")]
        impl crate::computation::CanonicalType for crate::execution::symbolic::Symbolic<$atomic> {
            type Type = $atomic;
        }

        #[cfg(feature = "compile")]
        impl From<$atomic> for <$atomic as crate::computation::SymbolicType>::Type {
            fn from(x: $atomic) -> Self {
                crate::execution::symbolic::Symbolic::Concrete(x)
            }
        }

        #[cfg(feature = "compile")]
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

        #[cfg(feature = "compile")]
        impl crate::computation::PartiallySymbolicType for $combined {
            type Type = $combined;
        }

        impl crate::computation::CanonicalType for $combined {
            type Type = $combined;
        }

        #[cfg(feature = "compile")]
        impl crate::computation::CanonicalType for crate::execution::symbolic::Symbolic<$combined> {
            type Type = $combined;
        }

        #[cfg(feature = "compile")]
        impl From<$combined> for <$combined as crate::computation::SymbolicType>::Type {
            fn from(x: $combined) -> Self {
                crate::execution::symbolic::Symbolic::Concrete(x)
            }
        }

        #[cfg(feature = "compile")]
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

        #[cfg(feature = "compile")]
        impl crate::computation::PartiallySymbolicType for $outer<$inner> {
            type Type = $outer<<$inner as crate::computation::SymbolicType>::Type>;
        }

        impl crate::computation::CanonicalType for $outer<$inner> {
            type Type = $outer<<$inner as crate::computation::CanonicalType>::Type>;
        }

        #[cfg(feature = "compile")]
        impl crate::computation::CanonicalType
            for $outer<<$inner as crate::computation::SymbolicType>::Type>
        {
            type Type = $outer<<$inner as crate::computation::CanonicalType>::Type>;
        }

        #[cfg(feature = "compile")]
        impl crate::computation::CanonicalType
            for crate::execution::symbolic::Symbolic<
                $outer<<$inner as crate::computation::SymbolicType>::Type>,
            >
        {
            type Type = $outer<<$inner as crate::computation::CanonicalType>::Type>;
        }

        // The kernel macro uses this to map (partially) concrete outputs to symbolic values
        #[cfg(feature = "compile")]
        impl From<$outer<<$inner as crate::computation::SymbolicType>::Type>>
            for <$combined as crate::computation::SymbolicType>::Type
        {
            fn from(x: $outer<<$inner as crate::computation::SymbolicType>::Type>) -> Self {
                crate::execution::symbolic::Symbolic::Concrete(x)
            }
        }

        // The kernel macros uses this to determine whether to invoke kernels, and
        // if so, to map symbolic values to (partially) concrete inputs
        #[cfg(feature = "compile")]
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

        #[cfg(feature = "compile")]
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

        #[cfg(feature = "compile")]
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

        #[cfg(feature = "compile")]
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
        #[cfg(feature = "compile")]
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
        #[cfg(feature = "compile")]
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

        #[cfg(feature = "compile")]
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

        #[cfg(feature = "compile")]
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

        #[cfg(feature = "compile")]
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
        #[cfg(feature = "compile")]
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
        #[cfg(feature = "compile")]
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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct N64;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct N128;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct N224;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
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

pub trait TensorLike {
    type Scalar;
}

pub(crate) trait BitArray {
    type Len: Const;
}

pub(crate) trait Underlying {
    type TensorType;
}

pub(crate) trait MirroredCounterpart {
    type MirroredType;
}

pub mod additive;
pub mod boolean;
pub mod bristol_fashion;
pub mod choreography;
#[cfg(feature = "compile")]
pub mod compilation;
pub mod computation;
pub mod encrypted;
pub mod error; // TODO make non-pub
pub mod execution;
pub mod fixedpoint;
pub mod floatingpoint;
mod grpc;
pub mod host;
pub mod integer;
pub mod kernels;
pub mod logical;
pub mod mirrored;
pub mod networking;
pub mod prelude;
pub mod reindeer;
pub mod replicated;
pub mod storage;
pub mod textual;
pub mod types;

pub use error::{Error, Result};
pub use tokio;
