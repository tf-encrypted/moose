// Returns the context-specific type for the given basic type.
macro_rules! cs {
    ($t:ty) => {
        <$t as KnownType<C>>::Type
    };
}

macro_rules! derive_runtime_kernel {
    (nullary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> Box<dyn Fn(&_, &_,) -> _> = &|$op| $kf;
            kf($self)
        }
    };
    (unary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> Box<dyn Fn(&_, &_, _) -> _> = &|$op| $kf;
            kf($self)
        }
    };
    (binary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> Box<dyn Fn(&_, &_, _, _) -> _> = &|$op| $kf;
            kf($self)
        }
    };
    (ternary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> Box<dyn Fn(&_, &_, _, _, _) -> _> = &|$op| $kf;
            kf($self)
        }
    };

    (nullary, attributes[$($attr:ident)+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
            )+
            Box::new(move |ctx, plc| {
                $k(ctx, plc, $($attr.clone()),+)
            })
        }
    };
    (unary, attributes[$($attr:ident)+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
            )+
            Box::new(move |ctx, plc, x0| {
                $k(ctx, plc, $($attr.clone()),+, x0)
            })
        }
    };
    (binary, attributes[$($attr:ident)+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
            )+
            Box::new(move |ctx, plc, x0, x1| {
                $k(ctx, plc, $($attr.clone()),+, x0, x1)
            })
        }
    };
    (ternary, attributes[$($attr:ident)+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
            )+
            Box::new(move |ctx, plc, x0, x1, x2| {
                $k(ctx, plc, $($attr.clone()),+), x0, x1, x2
            })
        }
    };

    (nullary, $k:expr, $self:ident) => {
        Box::new($k)
    };
    (unary, $k:expr, $self:ident) => {
        Box::new($k)
    };
    (binary, $k:expr, $self:ident) => {
        Box::new($k)
    };
    (ternary, $k:expr, $self:ident) => {
        Box::new($k)
    };
}

macro_rules! concrete_dispatch_kernel {

    /*
    Nullaray
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::kernels::ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c crate::kernels::ConcreteContext, plc: &crate::computation::Placement) -> Box<dyn Fn(Vec<crate::computation::Value>) -> crate::computation::Value + 'c> {
                use crate::computation::{KnownPlacement, KnownType, Signature, NullarySignature};
                use crate::kernels::{ConcreteContext, NullaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature{
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();

                            let k = <$op as NullaryKernel<ConcreteContext, $plc, $u>>::compile(self, &ctx, &plc);

                            Box::new(move |operands: Vec<crate::computation::Value>| {
                                assert_eq!(operands.len(), 0);

                                let y: $u = k(&ctx, &plc);
                                y.into()
                            })
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }
        }
    };

    /*
    Unary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::kernels::ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c crate::kernels::ConcreteContext, plc: &crate::computation::Placement) -> Box<dyn Fn(Vec<crate::computation::Value>) -> crate::computation::Value + 'c> {
                use crate::computation::{KnownPlacement, KnownType, Signature, UnarySignature, Value};
                use crate::kernels::{ConcreteContext, UnaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Unary(UnarySignature {
                                arg0: <$t0>::TY,
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();

                            let k = <$op as UnaryKernel<ConcreteContext, $plc, $t0, $u>>::compile(self, &ctx, &plc);

                            Box::new(move |operands: Vec<Value>| {
                                assert_eq!(operands.len(), 1);

                                let x0: $t0 = operands.get(0).unwrap().clone().try_into().unwrap();

                                let y: $u = k(&ctx, &plc, x0);
                                y.into()
                            })
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }
        }
    };

    /*
    Binary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::kernels::ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c crate::kernels::ConcreteContext, plc: &crate::computation::Placement) -> Box<dyn Fn(Vec<crate::computation::Value>) -> crate::computation::Value + 'c> {
                use crate::computation::{KnownPlacement, KnownType, Signature, BinarySignature, Value};
                use crate::kernels::{ConcreteContext, BinaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Binary(BinarySignature{
                                arg0: <$t0>::TY,
                                arg1: <$t1>::TY,
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();

                            let k = <$op as BinaryKernel<
                                ConcreteContext,
                                $plc,
                                $t0,
                                $t1,
                                $u
                            >>::compile(self, &ctx, &plc);

                            Box::new(move |operands| -> Value {
                                assert_eq!(operands.len(), 2);

                                let x0: $t0 = operands.get(0).unwrap().clone().try_into().unwrap();
                                let x1: $t1 = operands.get(1).unwrap().clone().try_into().unwrap();

                                let y: $u = k(&ctx, &plc, x0, x1);
                                y.into()
                            })
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }
        }
    };

    /*
    Ternary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty), )+]) => {
        impl crate::kernels::DispatchKernel<crate::kernels::ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c crate::kernels::ConcreteContext, plc: &crate::computation::Placement) -> Box<dyn Fn(Vec<crate::computation::Value>) -> crate::computation::Value + 'c> {
                use crate::computation::{KnownPlacement, KnownType, Signature, TernarySignature, Value};
                use crate::kernels::{ConcreteContext, TernaryKernel};
                use std::convert::TryInto;

                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Ternary(TernarySignature{
                                arg0: <$t0>::TY,
                                arg1: <$t1>::TY,
                                arg2: <$t2>::TY,
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();

                            let k = <$op as TernaryKernel<ConcreteContext, $plc, $t0, $t1, $t2, $u>>::compile(self, &ctx, &plc);

                            Box::new(move |operands: Vec<Value>| -> Value {
                                assert_eq!(operands.len(), 3);

                                let x0: $t0 = operands.get(0).unwrap().clone().try_into().unwrap();
                                let x1: $t1 = operands.get(1).unwrap().clone().try_into().unwrap();
                                let x2: $t2 = operands.get(2).unwrap().clone().try_into().unwrap();

                                let y: $u = k(&ctx, &plc, x0, x1, x2);
                                y.into()
                            })
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }
        }
    };
}

/// Kernel function is never used in symbolic contexts
macro_rules! kernel {

    /*
    Nullary
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, () -> $u), )+]);
        // symbolic_dispatch_kernel!($op, [$( ($plc, () -> $u), )+]);

        $(
            impl NullaryKernel<
                ConcreteContext,
                $plc,
                $u
            > for $op
            {
                fn compile(&self, _ctx: &ConcreteContext, _plc: &$plc) -> Box<dyn Fn(&ConcreteContext, &$plc) -> $u> {
                    derive_runtime_kernel![nullary, $($kp)+, self]
                }
            }
        )+

        // $(
        //     impl NullaryKernel<
        //         SymbolicContext,
        //         $plc,
        //         <$u as KnownType>::Symbolic
        //     > for $op
        //     {
        //         fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
        //             &SymbolicContext,
        //             &$plc)
        //             -> <$u as KnownType>::Symbolic>
        //         {
        //             let op = self.clone();
        //             Box::new(move |
        //                 ctx: &SymbolicContext,
        //                 plc: &$plc,
        //             | {
        //                 let op_name = ctx.add_operation(&op, &[], &plc.clone().into());
        //                 Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
        //             })
        //         }
        //     }
        // )+
    };

    /*
    Unary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0) -> $u), )+]);
        // symbolic_dispatch_kernel!($op, [$( ($plc, ($t0) -> $u), )+]);

        $(
            impl crate::kernels::UnaryKernel<
                crate::kernels::ConcreteContext,
                $plc,
                $t0,
                $u
            > for $op
            {
                fn compile(&self, _ctx: &crate::kernels::ConcreteContext, _plc: &$plc) -> Box<dyn Fn(&crate::kernels::ConcreteContext, &$plc, $t0) -> $u> {
                    derive_runtime_kernel![unary, $($kp)+, self]
                }
            }
        )+

        // $(
        //     impl UnaryKernel<
        //         SymbolicContext,
        //         $plc,
        //         <$t0 as KnownType>::Symbolic,
        //         <$u as KnownType>::Symbolic
        //     > for $op
        //     {
        //         fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
        //             &SymbolicContext,
        //             &$plc,
        //             <$t0 as KnownType>::Symbolic)
        //             -> <$u as KnownType>::Symbolic>
        //         {
        //             let op = self.clone();
        //             Box::new(move |
        //                 ctx: &SymbolicContext,
        //                 plc: &$plc,
        //                 x0: <$t0 as KnownType>::Symbolic,
        //             | {
        //                 match x0 {
        //                     Symbolic::Symbolic(h0) => {
        //                         let op_name = ctx.add_operation(&op, &[&h0.op], &plc.clone().into());
        //                         Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
        //                     }
        //                     _ => unimplemented!()
        //                 }
        //             })
        //         }
        //     }
        // )+
    };

    /*
    Binary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0, $t1) -> $u), )+]);
        // symbolic_dispatch_kernel!($op, [$( ($plc, ($t0, $t1) -> $u), )+]);

        $(
            impl crate::kernels::BinaryKernel<
                ConcreteContext,
                $plc,
                $t0,
                $t1,
                $u
            > for $op
            {
                fn compile(&self, _ctx: &ConcreteContext, _plc: &$plc) -> Box<dyn Fn(&ConcreteContext, &$plc, $t0, $t1) -> $u> {
                    derive_runtime_kernel![binary, $($kp)+, self]
                }
            }
        )+

        // $(
        //     impl BinaryKernel<
        //         SymbolicContext,
        //         $plc,
        //         <$t0 as KnownType>::Symbolic,
        //         <$t1 as KnownType>::Symbolic,
        //         <$u as KnownType>::Symbolic
        //     > for $op
        //     {
        //         fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
        //             &SymbolicContext,
        //             &$plc,
        //             <$t0 as KnownType>::Symbolic,
        //             <$t1 as KnownType>::Symbolic)
        //             -> <$u as KnownType>::Symbolic>
        //         {
        //             let op = self.clone();
        //             Box::new(move |
        //                 ctx: &SymbolicContext,
        //                 plc: &$plc,
        //                 x0: <$t0 as KnownType>::Symbolic,
        //                 x1: <$t1 as KnownType>::Symbolic,
        //             | {
        //                 match (x0, x1) {
        //                     (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
        //                         let op_name = ctx.add_operation(&op, &[&h0.op, &h1.op], &plc.clone().into());
        //                         Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
        //                     }
        //                     _ => unimplemented!()
        //                 }
        //             })
        //         }
        //     }
        // )+
    };

    /*
    Ternary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u), )+]);
        // symbolic_dispatch_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u), )+]);

        $(
            impl TernaryKernel<
                ConcreteContext,
                $plc,
                $t0,
                $t1,
                $t2,
                $u
            > for $op
            {
                fn compile(&self, _ctx: &ConcreteContext, _plc: &$plc) -> Box<dyn Fn(&ConcreteContext, &$plc, $t0, $t1, $t2) -> $u> {
                    derive_runtime_kernel![ternary, $($kp)+, self]
                }
            }
        )+

        // $(
        //     impl TernaryKernel<
        //         SymbolicContext,
        //         $plc,
        //         <$t0 as KnownType>::Symbolic,
        //         <$t1 as KnownType>::Symbolic,
        //         <$t2 as KnownType>::Symbolic,
        //         <$u as KnownType>::Symbolic
        //     > for $op
        //     {
        //         fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
        //             &SymbolicContext,
        //             &$plc,
        //             <$t0 as KnownType>::Symbolic,
        //             <$t1 as KnownType>::Symbolic,
        //             <$t2 as KnownType>::Symbolic)
        //             -> <$u as KnownType>::Symbolic>
        //         {
        //             let op = self.clone();
        //             Box::new(move |
        //                 ctx: &SymbolicContext,
        //                 plc: &$plc,
        //                 x0: <$t0 as KnownType>::Symbolic,
        //                 x1: <$t1 as KnownType>::Symbolic,
        //                 x2: <$t2 as KnownType>::Symbolic,
        //             | {
        //                 match (x0, x1, x2) {
        //                     (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
        //                         let op_name = ctx.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
        //                         Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
        //                     }
        //                     _ => unimplemented!()
        //                 }
        //             })
        //         }
        //     }
        // )+
    };
}

/// Kernel function maybe be evaluated in symbolic contexts
macro_rules! hybrid_kernel {

    /*
    Nullary
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, () -> $u), )+]);
        // symbolic_dispatch_kernel!($op, [$( ($plc, () -> $u), )+]);

        $(
            impl crate::kernels::NullaryKernel<
                crate::kernels::ConcreteContext,
                $plc,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _ctx: &crate::kernels::ConcreteContext,
                    _plc: &$plc,
                ) -> Box<dyn Fn(
                    &crate::kernels::ConcreteContext,
                    &$plc)
                    -> $u>
                {
                    derive_runtime_kernel![nullary, $($kp)+, self]
                }
            }
        )+

        // $(
        //     impl NullaryKernel<
        //         SymbolicContext,
        //         $plc,
        //         <$u as KnownType>::Symbolic
        //     > for $op
        //     {
        //         fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
        //             &SymbolicContext,
        //             &$plc)
        //             -> <$u as KnownType>::Symbolic>
        //         {
        //             let k = derive_runtime_kernel![nullary, $($kp)+, self];

        //             Box::new(move |
        //                 ctx: &SymbolicContext,
        //                 plc: &$plc,
        //             | {
        //                 let y = k(ctx, &plc);
        //                 y.into()
        //             })
        //         }
        //     }
        // )+
    };

    /*
    Unary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0) -> $u), )+]);
        // symbolic_dispatch_kernel!($op, [$( ($plc, ($t0) -> $u), )+]);

        $(
            impl crate::kernels::UnaryKernel<
                crate::kernels::ConcreteContext,
                $plc,
                $t0,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _ctx: &crate::kernels::ConcreteContext,
                    _plc: &$plc,
                ) -> Box<dyn Fn(&crate::kernels::ConcreteContext, &$plc, $t0) -> $u> {
                    derive_runtime_kernel![unary, $($kp)+, self]
                }
            }
        )+

        // $(
        //     impl UnaryKernel<
        //         SymbolicContext,
        //         $plc,
        //         <$t0 as KnownType>::Symbolic,
        //         <$u as KnownType>::Symbolic
        //     > for $op
        //     {
        //         fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
        //             &SymbolicContext,
        //             &$plc,
        //             <$t0 as KnownType>::Symbolic)
        //             -> <$u as KnownType>::Symbolic>
        //         {
        //             let k = derive_runtime_kernel![unary, $($kp)+, self];

        //             let op = self.clone();
        //             Box::new(move |
        //                 ctx: &SymbolicContext,
        //                 plc: &$plc,
        //                 x0: <$t0 as KnownType>::Symbolic,
        //             | {
        //                 let v0 = x0.clone().try_into();

        //                 match v0 {
        //                     Ok(v0) => {
        //                         let y = k(ctx, &plc, v0);
        //                         y.into()
        //                     }
        //                     _ => match x0 {
        //                         Symbolic::Symbolic(h0) => {
        //                             let op_name = ctx.add_operation(&op, &[&h0.op], &plc.clone().into());
        //                             Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
        //                         }
        //                         _ => unimplemented!() // ok
        //                     }
        //                 }
        //             })
        //         }
        //     }
        // )+
    };

    /*
    Binary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0, $t1) -> $u), )+]);
        // symbolic_dispatch_kernel!($op, [$( ($plc, ($t0, $t1) -> $u), )+]);

        $(
            impl crate::kernels::BinaryKernel<
                crate::kernels::ConcreteContext,
                $plc,
                $t0,
                $t1,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _ctx: &crate::kernels::ConcreteContext,
                    _plc: &$plc
                ) -> Box<dyn Fn(&crate::kernels::ConcreteContext, &$plc, $t0, $t1) -> $u> {
                    derive_runtime_kernel![binary, $($kp)+, self]
                }
            }
        )+

        // $(
        //     impl BinaryKernel<
        //         SymbolicContext,
        //         $plc,
        //         <$t0 as KnownType>::Symbolic,
        //         <$t1 as KnownType>::Symbolic,
        //         <$u as KnownType>::Symbolic
        //     > for $op
        //     {
        //         fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
        //             &SymbolicContext,
        //             &$plc,
        //             <$t0 as KnownType>::Symbolic,
        //             <$t1 as KnownType>::Symbolic)
        //             -> <$u as KnownType>::Symbolic>
        //         {
        //             let k = derive_runtime_kernel![binary, $($kp)+, self];

        //             let op = self.clone();
        //             Box::new(move |
        //                 ctx: &SymbolicContext,
        //                 plc: &$plc,
        //                 x0: <$t0 as KnownType>::Symbolic,
        //                 x1: <$t1 as KnownType>::Symbolic,
        //             | {
        //                 let v0 = x0.clone().try_into();
        //                 let v1 = x1.clone().try_into();

        //                 match (v0, v1) {
        //                     (Ok(v0), Ok(v1)) => {
        //                         let y = k(ctx, &plc, v0, v1);
        //                         y.into()
        //                     }
        //                     _ => match (x0, x1) {
        //                         (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
        //                             let op_name = ctx.add_operation(&op, &[&h0.op, &h1.op], &plc.clone().into());
        //                             Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
        //                         }
        //                         _ => unimplemented!() // ok
        //                     }
        //                 }
        //             })
        //         }
        //     }
        // )+
    };

    /*
    Ternary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u), )+]);
        // symbolic_dispatch_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u), )+]);

        $(
            impl crate::kernels::TernaryKernel<
                crate::kernels::ConcreteContext,
                $plc,
                $t0,
                $t1,
                $t2,
                $u
            > for $op
            {
                fn compile(
                    &self,
                    _ctx: &crate::kernels::ConcreteContext,
                    _plc: &$plc,
                ) -> Box<dyn Fn(&crate::kernels::ConcreteContext, &$plc, $t0, $t1, $t2) -> $u> {
                    derive_runtime_kernel![ternary, $($kp)+, self]
                }
            }
        )+

        // $(
        //     impl TernaryKernel<
        //         SymbolicContext,
        //         $plc,
        //         <$t0 as KnownType>::Symbolic,
        //         <$t1 as KnownType>::Symbolic,
        //         <$t2 as KnownType>::Symbolic,
        //         <$u as KnownType>::Symbolic
        //     > for $op
        //     {
        //         fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
        //             &SymbolicContext,
        //             &$plc,
        //             <$t0 as KnownType>::Symbolic,
        //             <$t1 as KnownType>::Symbolic,
        //             <$t2 as KnownType>::Symbolic)
        //             -> <$u as KnownType>::Symbolic>
        //         {
        //             let k = derive_runtime_kernel![ternary, $($kp)+, self];

        //             let op = self.clone();
        //             Box::new(move |
        //                 ctx: &SymbolicContext,
        //                 plc: &$plc,
        //                 x0: <$t0 as KnownType>::Symbolic,
        //                 x1: <$t1 as KnownType>::Symbolic,
        //                 x2: <$t2 as KnownType>::Symbolic,
        //             | {
        //                 let v0 = x0.clone().try_into();
        //                 let v1 = x1.clone().try_into();
        //                 let v2 = x2.clone().try_into();

        //                 match (v0, v1, v2) {
        //                     (Ok(v0), Ok(v1), Ok(v2)) => {
        //                         let y = k(ctx, &plc, v0, v1, v2);
        //                         y.into()
        //                     }
        //                     _ => match (x0, x1, x2) {
        //                         (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
        //                             let op_name = ctx.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
        //                             Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
        //                         }
        //                         _ => unimplemented!() // ok
        //                     }
        //                 }
        //             })
        //         }
        //     }
        // )+
    };
}

macro_rules! modelled {
    /*
    Nullary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? () -> $u:ty, $op:ident) => {
        impl crate::kernels::NullaryKernelCheck<ConcreteContext, $plc, $u> for $op {}

        impl $t<ConcreteContext, $u> for $plc {
            fn $f(&self, ctx: &ConcreteContext, $($($attr_id:$attr_ty),*)?) -> $u {
                use crate::computation::{KnownType, NullarySignature};
                use crate::kernels::Context;
                use std::convert::TryInto;

                let sig = NullarySignature {
                    ret: <$u as KnownType<ConcreteContext>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(op.into(), &self.into(), vec![])
                    .try_into()
                    .unwrap()
            }
        }

        // impl $t<SymbolicContext, <$u as KnownType>::Symbolic> for $plc {
        //     fn $f(&self, ctx: &SymbolicContext, $($($attr_id:$attr_ty),*)?) -> <$u as KnownType>::Symbolic {
        //         let sig = NullarySignature {
        //             ret: <$u as KnownType>::TY,
        //         };
        //         let op = $op {
        //             sig: sig.into(),
        //             $($($attr_id),*)?
        //         };
        //         ctx.execute(op.into(), &self.into(), vec![])
        //             .try_into()
        //             .unwrap()
        //     }
        // }
    };

    /*
    Unary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty) -> $u:ty, $op:ident) => {
        impl crate::kernels::UnaryKernelCheck<crate::kernels::ConcreteContext, $plc, $t0, $u> for $op {}

        impl $t<crate::kernels::ConcreteContext, $t0, $u> for $plc {
            fn $f(&self, ctx: &crate::kernels::ConcreteContext, $($($attr_id:$attr_ty),*,)? x0: &$t0) -> $u {
                use crate::computation::{KnownType, UnarySignature};
                use crate::kernels::{Context, ConcreteContext};
                use std::convert::TryInto;

                let sig = UnarySignature {
                    arg0: <$t0 as KnownType<ConcreteContext>>::TY,
                    ret: <$u as KnownType<ConcreteContext>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(op.into(), &self.into(), vec![x0.clone().into()])
                    .try_into()
                    .unwrap()
            }
        }

        // impl $t<SymbolicContext, <$t0 as KnownType>::Symbolic> for $plc {
        //     type Output = <$u as KnownType>::Symbolic;

        //     fn $f(&self, ctx: &SymbolicContext, $($($attr_id:$attr_ty),*,)? x0: &<$t0 as KnownType>::Symbolic) -> Self::Output {
        //         let sig = UnarySignature {
        //             arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
        //             ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
        //         };
        //         let op = $op {
        //             sig: sig.into(),
        //             $($($attr_id),*)?
        //         };
        //         ctx.execute(op.into(), &self.into(), vec![x0.clone().into()])
        //             .try_into()
        //             .unwrap()
        //     }
        // }
    };

    /*
    Binary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty, $t1:ty) -> $u:ty, $op:ident) => {
        impl crate::kernels::BinaryKernelCheck<crate::kernels::ConcreteContext, $plc, $t0, $t1, $u> for $op {}

        impl $t<crate::kernels::ConcreteContext, $t0, $t1, $u> for $plc {
            fn $f(&self, ctx: &crate::kernels::ConcreteContext, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1) -> $u {
                use crate::computation::{KnownType, BinarySignature};
                use crate::kernels::{Context, ConcreteContext};
                use std::convert::TryInto;

                let sig = BinarySignature {
                    arg0: <$t0 as KnownType<ConcreteContext>>::TY,
                    arg1: <$t1 as KnownType<ConcreteContext>>::TY,
                    ret: <$u as KnownType<ConcreteContext>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(
                    op.into(),
                    &self.into(),
                    vec![x0.clone().into(), x1.clone().into()],
                )
                .try_into()
                .unwrap()
            }
        }

        // impl $t<SymbolicContext, <$t0 as KnownType>::Symbolic, <$t1 as KnownType>::Symbolic>
        //     for $plc
        // {
        //     type Output = <$u as KnownType>::Symbolic;

        //     fn $f(
        //         &self,
        //         ctx: &SymbolicContext,
        //         $($($attr_id:$attr_ty),*,)?
        //         x0: &<$t0 as KnownType>::Symbolic,
        //         x1: &<$t1 as KnownType>::Symbolic,
        //     ) -> Self::Output {
        //         let sig = BinarySignature {
        //             arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
        //             arg1: <<$t1 as KnownType>::Symbolic as KnownType>::TY,
        //             ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
        //         };
        //         let op = $op {
        //             sig: sig.into(),
        //             $($($attr_id),*)?
        //         };
        //         ctx.execute(
        //             op.into(),
        //             &self.into(),
        //             vec![x0.clone().into(), x1.clone().into()],
        //         )
        //         .try_into()
        //         .unwrap()
        //     }
        // }
    };

    /*
    Ternary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $op:ident) => {
        impl crate::kernels::TernaryKernelCheck<crate::kernels::ConcreteContext, $plc, $t0, $t1, $t2, $u> for $op {}

        impl $t<crate::kernels::ConcreteContext, $t0, $t1, $t2, $u> for $plc {
            fn $f(&self, ctx: &crate::kernels::ConcreteContext, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1, x2: &$t2) -> $u {
                use crate::computation::{KnownType, TernarySignature};
                use crate::kernels::{Context, ConcreteContext};
                use std::convert::TryInto;

                let sig = TernarySignature {
                    arg0: <$t0 as KnownType<ConcreteContext>>::TY,
                    arg1: <$t1 as KnownType<ConcreteContext>>::TY,
                    arg2: <$t2 as KnownType<ConcreteContext>>::TY,
                    ret: <$u as KnownType<ConcreteContext>>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(
                    op.into(),
                    &self.into(),
                    vec![x0.clone().into(), x1.clone().into(), x2.clone().into()],
                )
                .try_into()
                .unwrap()
            }
        }

        // impl
        //     $t<
        //         SymbolicContext,
        //         <$t0 as KnownType>::Symbolic,
        //         <$t1 as KnownType>::Symbolic,
        //         <$t2 as KnownType>::Symbolic,
        //     > for $plc
        // {
        //     type Output = <$u as KnownType>::Symbolic;

        //     fn $f(
        //         &self,
        //         ctx: &SymbolicContext,
        //         $($($attr_id:$attr_ty),*,)?
        //         x0: &<$t0 as KnownType>::Symbolic,
        //         x1: &<$t1 as KnownType>::Symbolic,
        //         x2: &<$t2 as KnownType>::Symbolic,
        //     ) -> Self::Output {
        //         let sig = TernarySignature {
        //             arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
        //             arg1: <<$t1 as KnownType>::Symbolic as KnownType>::TY,
        //             arg2: <<$t2 as KnownType>::Symbolic as KnownType>::TY,
        //             ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
        //         };
        //         let op = $op {
        //             sig: sig.into(),
        //             $($($attr_id),*)?
        //         };
        //         ctx.execute(
        //             op.into(),
        //             &self.into(),
        //             vec![x0.clone().into(), x1.clone().into(), x2.clone().into()],
        //         )
        //         .try_into()
        //         .unwrap()
        //     }
        // }
    };
}

macro_rules! modelled_alias {
    /*
    Binary
    */
    ($src_t:ident::$src_f:ident, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $dst_t:ident::$dst_f:ident) => {
        impl $src_t<ConcreteContext, $t0, $t1, $u> for $plc {
            fn $src_f(&self, ctx: &ConcreteContext, x0: &$t0, x1: &$t1) -> $u {
                $dst_t::$dst_f(self, ctx, x0, x1)
            }
        }

        // impl $src_t<SymbolicContext, <$t0 as KnownType>::Symbolic, <$t1 as KnownType>::Symbolic>
        //     for $plc
        // {
        //     type Output = <$u as KnownType>::Symbolic;

        //     fn $src_f(
        //         &self,
        //         ctx: &SymbolicContext,
        //         x0: &<$t0 as KnownType>::Symbolic,
        //         x1: &<$t1 as KnownType>::Symbolic,
        //     ) -> Self::Output {
        //         $dst_t::$dst_f(self, ctx, x0, x1)
        //     }
        // }
    };
}

pub mod additive;
pub mod bit;
pub mod compilation;
pub mod computation;
pub mod error;
pub mod execution;
pub mod fixedpoint;
pub mod kernels;
pub mod networking;
pub mod prim;
pub mod prng;
pub mod python_computation;
pub mod replicated;
pub mod ring;
pub mod standard;
pub mod storage;
pub mod text_computation;
pub mod utils;
