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
                $k(ctx, plc, $($attr),+)
            })
        }
    };
    (unary, attributes[$($attr:ident)+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
            )+
            Box::new(move |ctx, plc, x0| {
                $k(ctx, plc, $($attr),+, x0)
            })
        }
    };
    (binary, attributes[$($attr:ident)+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
            )+
            Box::new(move |ctx, plc, x0, x1| {
                $k(ctx, plc, $($attr),+, x0, x1)
            })
        }
    };
    (ternary, attributes[$($attr:ident)+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
            )+
            Box::new(move |ctx, plc, x0, x1, x2| {
                $k(ctx, plc, $($attr),+), x0, x1, x2
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
        impl crate::kernels::DispatchKernel<ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c ConcreteContext, plc: &crate::computation::Placement) -> Box<dyn Fn(Vec<crate::computation::Value>) -> crate::computation::Value + 'c> {
                use crate::computation::{KnownPlacement, KnownType, Signature, NullarySignature};
                use crate::kernels::NullaryKernel;
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
        impl crate::kernels::DispatchKernel<ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c ConcreteContext, plc: &crate::computation::Placement) -> Box<dyn Fn(Vec<crate::computation::Value>) -> crate::computation::Value + 'c> {
                use crate::computation::{KnownPlacement, KnownType, Signature, UnarySignature};
                use crate::kernels::UnaryKernel;
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

                            Box::new(move |operands: Vec<crate::computation::Value>| {
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
        impl crate::kernels::DispatchKernel<ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c ConcreteContext, plc: &crate::computation::Placement) -> Box<dyn Fn(Vec<crate::computation::Value>) -> crate::computation::Value + 'c> {
                use crate::computation::{KnownPlacement, KnownType, Signature, BinarySignature};
                use crate::kernels::BinaryKernel;
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

                            Box::new(move |operands| -> crate::computation::Value {
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
        impl DispatchKernel<ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c ConcreteContext, plc: &Placement) -> Box<dyn Fn(Vec<Value>) -> Value + 'c> {
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
                fn compile(&self, ctx: &ConcreteContext, plc: &$plc) -> Box<dyn Fn(&ConcreteContext, &$plc) -> $u> {
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
                ConcreteContext,
                $plc,
                $t0,
                $u
            > for $op
            {
                fn compile(&self, ctx: &ConcreteContext, plc: &$plc) -> Box<dyn Fn(&ConcreteContext, &$plc, $t0) -> $u> {
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
                fn compile(&self, ctx: &ConcreteContext, plc: &$plc) -> Box<dyn Fn(&ConcreteContext, &$plc, $t0, $t1) -> $u> {
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
                fn compile(&self, ctx: &ConcreteContext, plc: &$plc) -> Box<dyn Fn(&ConcreteContext, &$plc, $t0, $t1, $t2) -> $u> {
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
