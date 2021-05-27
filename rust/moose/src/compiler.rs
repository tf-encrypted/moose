#![allow(dead_code)]
#![allow(unused_variables)]

use std::convert::{TryFrom, TryInto};

#[derive(Debug, Clone)]
enum Placement {
    Host(HostPlacement),
    Replicated(ReplicatedPlacement),
}

enum PlacementInstantiation {
    Host,
    Rep,
}

#[derive(Debug, Clone)]
enum Ty {
    HostFixedTy,
    Ring64TensorTy,
    Ring128TensorTy,
    Replicated64TensorTy,
    Replicated128TensorTy,
}

enum Value {
    HostFixed(HostFixed),
    RepFixed(RepFixed),
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    Replicated64Tensor(ReplicatedTensor<Ring64Tensor>),
    Replicated128Tensor(ReplicatedTensor<Ring128Tensor>),
}

#[derive(Clone, Debug)]
struct RingTensor<T>(T);

#[derive(Clone, Debug)]
enum Ring64Tensor {
    Symbolic(Operator),
    Concrete(RingTensor<u64>),
}

#[derive(Clone, Debug)]
enum Ring128Tensor {
    Symbolic(Operator),
    Concrete(RingTensor<u128>),
}

impl From<Ring64Tensor> for Value {
    fn from(x: Ring64Tensor) -> Value {
        Value::Ring64Tensor(x)
    }
}

impl From<Ring128Tensor> for Value {
    fn from(x: Ring128Tensor) -> Value {
        Value::Ring128Tensor(x)
    }
}

impl From<ReplicatedTensor<Ring64Tensor>> for Value {
    fn from(x: ReplicatedTensor<Ring64Tensor>) -> Value {
        Value::Replicated64Tensor(x)
    }
}

impl From<ReplicatedTensor<Ring128Tensor>> for Value {
    fn from(x: ReplicatedTensor<Ring128Tensor>) -> Value {
        Value::Replicated128Tensor(x)
    }
}

impl TryFrom<Value> for Ring64Tensor {
    type Error = ();

    fn try_from(x: Value) -> Result<Ring64Tensor, Self::Error> {
        match x {
            Value::Ring64Tensor(x) => Ok(x),
            _ => Err(())
        }
    }
}

impl TryFrom<Value> for Ring128Tensor {
    type Error = ();

    fn try_from(x: Value) -> Result<Ring128Tensor, Self::Error> {
        match x {
            Value::Ring128Tensor(x) => Ok(x),
            _ => Err(())
        }
    }
}

impl TryFrom<Value> for ReplicatedTensor<Ring64Tensor> {
    type Error = ();

    fn try_from(x: Value) -> Result<ReplicatedTensor<Ring64Tensor>, Self::Error> {
        match x {
            Value::Replicated64Tensor(x) => Ok(x),
            _ => Err(())
        }
    }
}

impl TryFrom<Value> for ReplicatedTensor<Ring128Tensor> {
    type Error = ();

    fn try_from(x: Value) -> Result<ReplicatedTensor<Ring128Tensor>, Self::Error> {
        match x {
            Value::Replicated128Tensor(x) => Ok(x),
            _ => Err(())
        }
    }
}

impl From<RingTensor<u64>> for Ring64Tensor {
    fn from(x: RingTensor<u64>) -> Ring64Tensor {
        Ring64Tensor::Concrete(x)
    }
}

impl From<RingTensor<u128>> for Ring128Tensor {
    fn from(x: RingTensor<u128>) -> Ring128Tensor {
        Ring128Tensor::Concrete(x)
    }
}

// impl HostFixed {
//     fn placement(&self) -> Placement {
//         Placement::Host
//     }
// }

impl From<HostFixed> for Value {
    fn from(x: HostFixed) -> Value {
        Value::HostFixed(x)
    }
}

impl From<RepFixed> for Value {
    fn from(x: RepFixed) -> Value {
        Value::RepFixed(x)
    }
}

struct HostFixed {}

struct RepFixed {}

// struct Context {}

// impl Context {
//     fn placement_instantiation(&self, plc: Placement) -> PlacementInstantiation {
//         unimplemented!()
//     }
// }

use std::ops::{Add, Sub};

impl<T: Add<T, Output=T>> Add<RingTensor<T>> for RingTensor<T> {
    type Output = RingTensor<T>;

    fn add(self, other: RingTensor<T>) -> Self::Output {
        RingTensor(self.0 + other.0)
    }
}

impl<T: Sub<T, Output=T>> Sub<RingTensor<T>> for RingTensor<T> {
    type Output = RingTensor<T>;

    fn sub(self, other: RingTensor<T>) -> Self::Output {
        RingTensor(self.0 - other.0)
    }
}

// enum Symbolic<T> {
//     Concrete(T),
//     Symbolic,
// }

#[derive(Clone, Debug)]
struct ReplicatedTensor<R> {
    shares: [[R; 2]; 3],
}

#[derive(Clone, Debug)]
struct ReplicatedPlacement {
    players: [String; 3]
}

#[derive(Clone, Debug)]
struct HostPlacement{ player: String }

trait PlacementAdd<T, U> {
    type Output;

    fn add(&self, ctx: &ConcreteContext, x: T, y: U) -> Self::Output;
}

impl PlacementAdd<&Ring64Tensor, &Ring64Tensor> for HostPlacement {
    type Output = Ring64Tensor;

    fn add(&self, ctx: &ConcreteContext, x: &Ring64Tensor, y: &Ring64Tensor) -> Self::Output {
        ctx.execute_binary(
            RingAddOp {
                lhs: Ty::Ring64TensorTy,
                rhs: Ty::Ring64TensorTy,
                plc: Placement::Host(self.clone()),
            }.into(),
            x.clone().into(),
            y.clone().into()
        ).try_into().unwrap()
    }
}

impl PlacementAdd<&Ring128Tensor, &Ring128Tensor> for HostPlacement {
    type Output = Ring128Tensor;

    fn add(&self, ctx: &ConcreteContext, x: &Ring128Tensor, y: &Ring128Tensor) -> Self::Output {
        ctx.execute_binary(
            RingAddOp {
                lhs: Ty::Ring128TensorTy,
                rhs: Ty::Ring128TensorTy,
                plc: Placement::Host(self.clone()),
            }.into(),
            x.clone().into(),
            y.clone().into()
        ).try_into().unwrap()
    }
}

// impl PlacementAdd<&Symbolic<Ring64Tensor>, &Symbolic<Ring64Tensor>> for HostPlacement {
//     type Output = Symbolic<Ring64Tensor>;

//     fn add(&self, x: &Symbolic<Ring64Tensor>, y: &Symbolic<Ring64Tensor>) -> Self::Output {
//         RingAddOp {
//             lhs: Ty::Ring64TensorTy,
//             rhs: Ty::Ring64TensorTy,
//             plc: Placement::Host(self.clone()),
//         }.compile()(x.clone().into(), y.clone().into()).try_into().unwrap()
//     }
// }

impl PlacementAdd<&ReplicatedTensor<Ring64Tensor>, &ReplicatedTensor<Ring64Tensor>> for ReplicatedPlacement {
    type Output = ReplicatedTensor<Ring64Tensor>;

    fn add(&self, ctx: &ConcreteContext, x: &ReplicatedTensor<Ring64Tensor>, y: &ReplicatedTensor<Ring64Tensor>) -> Self::Output {
        ctx.execute_binary(
            RepAddOp {
                lhs: Ty::Replicated64TensorTy,
                rhs: Ty::Replicated64TensorTy,
                plc: Placement::Replicated(self.clone()),
            }.into(),
            x.clone().into(),
            y.clone().into()
        ).try_into().unwrap()
    }
}

impl PlacementAdd<&ReplicatedTensor<Ring128Tensor>, &ReplicatedTensor<Ring128Tensor>> for ReplicatedPlacement {
    type Output = ReplicatedTensor<Ring128Tensor>;

    fn add(&self, ctx: &ConcreteContext, x: &ReplicatedTensor<Ring128Tensor>, y: &ReplicatedTensor<Ring128Tensor>) -> Self::Output {
        ctx.execute_binary(
            RepAddOp {
                lhs: Ty::Replicated128TensorTy,
                rhs: Ty::Replicated128TensorTy,
                plc: Placement::Replicated(self.clone()),
            }.into(),
            x.clone().into(),
            y.clone().into()
        ).try_into().unwrap()
    }
}

trait PlacementSub<T, U> {
    type Output;

    fn sub(&self, ctx: &ConcreteContext, x: T, y: U) -> Self::Output;
}

impl PlacementSub<&Ring64Tensor, &Ring64Tensor> for HostPlacement {
    type Output = Ring64Tensor;

    fn sub(&self, ctx: &ConcreteContext, x: &Ring64Tensor, y: &Ring64Tensor) -> Self::Output {
        ctx.execute_binary(
            RingSubOp {
                lhs: Ty::Ring64TensorTy,
                rhs: Ty::Ring64TensorTy,
                plc: Placement::Host(self.clone()),
            }.into(),
            x.clone().into(),
            y.clone().into()
        ).try_into().unwrap()
    }
}

impl PlacementSub<&Ring128Tensor, &Ring128Tensor> for HostPlacement {
    type Output = Ring128Tensor;

    fn sub(&self, ctx: &ConcreteContext, x: &Ring128Tensor, y: &Ring128Tensor) -> Self::Output {
        ctx.execute_binary(
            RingSubOp {
                lhs: Ty::Ring128TensorTy,
                rhs: Ty::Ring128TensorTy,
                plc: Placement::Host(self.clone()),
            }.into(),
            x.clone().into(),
            y.clone().into()
        ).try_into().unwrap()
    }
}

trait Context<T> {
    fn execute_nullary(&self, op: Operator) -> T;
    fn execute_binary(&self, op: Operator, x: T, y: T) -> T;
}

#[derive(Clone, Debug)]
struct ConcreteContext {}

impl Context<Value> for ConcreteContext {
    fn execute_nullary(&self, op: Operator) -> Value {
        match op {
            Operator::RingSample(op) => op.compile()().try_into().unwrap(),
            _ => unimplemented!()
        }
    }

    fn execute_binary(&self, op: Operator, x: Value, y: Value) -> Value {
        match op {
            Operator::RepAdd(op) => op.compile(self)(x, y).try_into().unwrap(),
            Operator::RingAdd(op) => op.compile(self)(x, y).try_into().unwrap(),
            Operator::RingSub(op) => op.compile(self)(x, y).try_into().unwrap(),
            _ => unimplemented!()
        }
    }
}

#[derive(Clone, Debug)]
struct RepAddOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

impl RepAddOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Value, Value) -> Value> {
        match (&self.plc, &self.lhs, &self.rhs) {

            (Placement::Replicated(plc), Ty::Replicated64TensorTy, Ty::Replicated64TensorTy) => {
                let ctx = ctx.clone();
                let plc = plc.clone();

                Box::new(move |x: Value, y: Value| {
                    match (x, y) {
                        (Value::Replicated64Tensor(x), Value::Replicated64Tensor(y)) => {
                            Self::abstract_kernel(&ctx, &plc, x, y).into()
                        }
                        _ => unimplemented!()
                    }
                })
            }

            (Placement::Replicated(plc), Ty::Replicated128TensorTy, Ty::Replicated128TensorTy) => {
                let ctx = ctx.clone();
                let plc = plc.clone();

                Box::new(move |x: Value, y: Value| {
                    match (x, y) {
                        (Value::Replicated128Tensor(x), Value::Replicated128Tensor(y)) => {
                            Self::abstract_kernel(&ctx, &plc, x, y).into()
                        }
                        _ => unimplemented!()
                    }
                })
            }


            // (Placement::Rep, Ty::HostFixedTy, Ty::HostFixedTy) => {
            //     |x: Value, y: Value| {
            //         // let x_owner = ctx.placement_instantiation(x.placement());
            //         // let y_owner = ctx.placement_instantiation(y.placement());

            //         // let xe = x_owner.share(x);
            //         // let ye = y_owner.share(y);
            //         // add(xe, ye)
            //         unimplemented!()
            //     }
            // }
            // (Placement::Rep, Ty::RepFixedTy, Ty::RepFixedTy) => {
            //     |x: Value, y: Value| {
            //         unimplemented!()
            //     }
            // }
            _ => unimplemented!()
        }
    }

    // pulling this out as a function to cover abstract replicated tensors (based on ring64, ring128, etc)
    fn abstract_kernel<R>(ctx: &ConcreteContext, rep: &ReplicatedPlacement, x: ReplicatedTensor<R>, y: ReplicatedTensor<R>) -> ReplicatedTensor<R>
    where
        for<'x, 'y> HostPlacement: PlacementAdd<&'x R, &'y R, Output=R>,
    {
        let player0 = HostPlacement{ player: rep.players[0].clone() };
        let player1 = HostPlacement{ player: rep.players[1].clone() };
        let player2 = HostPlacement{ player: rep.players[2].clone() };

        let ReplicatedTensor {
            shares: [
                [x00, x10],
                [x11, x21],
                [x22, x02],
            ]
        } = &x;

        let ReplicatedTensor { 
            shares: [
                [y00, y10],
                [y11, y21],
                [y22, y02],
            ]
        } = &y;

        // this could be turned into something like `let z00 = player0.with(|x00, y00| { x00 + y00 }, x00, y00)`
        let z00 = player0.add(ctx, x00, y00);
        let z10 = player0.add(ctx, x10, y10);

        let z11 = player1.add(ctx, x11, y11);
        let z21 = player1.add(ctx, x21, y21);

        let z22 = player2.add(ctx, x22, y22);
        let z02 = player2.add(ctx, x02, y02);

        ReplicatedTensor {
            shares: [
                [z00, z10],
                [z11, z21],
                [z22, z02],
            ]
        }
    }
}

#[derive(Clone, Debug)]
struct RepShareOp {
    lhs: Ty,
    plc: Placement,
}

// impl RepShareOp {
//     pub fn compile(&self) -> Box<dyn Fn(Value) -> Value> {
//         match (&self.plc, &self.lhs) {

//             (Placement::Replicated(rep_plc), Ty::Ring64TensorTy) => {
//                 let rep_plc = rep_plc.clone();

//                 Box::new(move |x: Value| {
//                     match x {
//                         Value::Ring64Tensor(x) => {
//                             match x {
//                                 Ring64Tensor::Concrete(x) => {
//                                     Self::abstract_kernel(&rep_plc, x).into()
//                                 }
//                             }
//                         }
//                         _ => unimplemented!()
//                     }
//                 })
//             }

//             (Placement::Replicated(rep_plc), Ty::Ring128TensorTy) => {
//                 let rep_plc = rep_plc.clone();

//                 Box::new(move |x: Value| {
//                     match x {
//                         Value::Ring128Tensor(x) => {
//                             match x {
//                                 Ring128Tensor::Concrete(x) => {
//                                     Self::abstract_kernel(&rep_plc, x).into()
//                                 }
//                             }    
//                         }
//                         _ => unimplemented!()
//                     }
//                 })
//             }


//             // (Placement::Rep, Ty::HostFixedTy, Ty::HostFixedTy) => {
//             //     |x: Value, y: Value| {
//             //         // let x_owner = ctx.placement_instantiation(x.placement());
//             //         // let y_owner = ctx.placement_instantiation(y.placement());

//             //         // let xe = x_owner.share(x);
//             //         // let ye = y_owner.share(y);
//             //         // add(xe, ye)
//             //         unimplemented!()
//             //     }
//             // }
//             // (Placement::Rep, Ty::RepFixedTy, Ty::RepFixedTy) => {
//             //     |x: Value, y: Value| {
//             //         unimplemented!()
//             //     }
//             // }
//             _ => unimplemented!()
//         }
//     }

//     // pulling this out as a function to cover abstract replicated tensors (based on ring64, ring128, etc)
//     fn abstract_kernel<R>(rep: &ReplicatedPlacement, x: R) -> ReplicatedTensor<R>
//     where
//         for<'x, 'y> HostPlacement: PlacementAdd<&'x R, &'y R, Output=R>,
//         for<'x, 'y> HostPlacement: PlacementSub<&'x R, &'y R, Output=R>,
//         HostPlacement: PlacementSample<R>,
//     {
//         let player0 = HostPlacement{ player: rep.players[0].clone() };
//         let player1 = HostPlacement{ player: rep.players[1].clone() };
//         let player2 = HostPlacement{ player: rep.players[2].clone() };

//         // TODO we should not use player0 here, but rather the owner of `x`
//         let x0 = player0.sample();
//         let x1 = player0.sample();
//         let x2 = player0.sub(&x, &player0.add(&x0, &x1));

//         ReplicatedTensor {
//             shares: [
//                 [x0, x1],
//                 [x1, x2],
//                 [x2, x0],
//             ]
//         }
//     }
// }

#[derive(Clone, Debug)]
struct RingAddOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

impl RingAddOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Value, Value) -> Value> {
        match (&self.plc, &self.lhs, &self.rhs) {

            (Placement::Host(_), Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                let op = self.clone();
                Box::new(move |x: Value, y: Value| -> Value {
                    match (x, y) {
                        (Value::Ring64Tensor(x), Value::Ring64Tensor(y)) => {
                            match (x, y) {
                                (Ring64Tensor::Symbolic(x_op), Ring64Tensor::Symbolic(y_op)) => {
                                    Ring64Tensor::Symbolic(Operator::RingAdd(op.clone()))
                                }
                                (Ring64Tensor::Concrete(x), Ring64Tensor::Concrete(y)) => {
                                    Self::abstract_kernel(x, y).into()
                                }
                                _ => unimplemented!()
                            }.into()
                        }
                        _ => unimplemented!()
                    }
                })
            }

            (Placement::Host(_), Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                let op = self.clone();
                Box::new(move |x: Value, y: Value| -> Value {
                    match (x, y) {
                        (Value::Ring128Tensor(x), Value::Ring128Tensor(y)) => {
                            match (x, y) {
                                (Ring128Tensor::Symbolic(x_op), Ring128Tensor::Symbolic(y_op)) => {
                                    Ring128Tensor::Symbolic(Operator::RingAdd(op.clone()))
                                }
                                (Ring128Tensor::Concrete(x), Ring128Tensor::Concrete(y)) => {
                                    Self::abstract_kernel(x, y).into()
                                }
                                _ => unimplemented!()
                            }.into()
                        }
                        _ => unimplemented!()
                    }
                })
            }

            _ => unimplemented!()
        }
    }

    fn abstract_kernel<T>(x: RingTensor<T>, y: RingTensor<T>) -> RingTensor<T>
    where
        RingTensor<T>: Add<RingTensor<T>, Output=RingTensor<T>>,
    {
        x + y
    }
}

#[derive(Clone, Debug)]
struct RingSubOp {
    lhs: Ty,
    rhs: Ty,
    plc: Placement,
}

impl RingSubOp {
    pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Value, Value) -> Value> {
        match (&self.plc, &self.lhs, &self.rhs) {

            (Placement::Host(_), Ty::Ring64TensorTy, Ty::Ring64TensorTy) => {
                let op = self.clone();
                Box::new(move |x: Value, y: Value| -> Value {
                    match (x, y) {
                        (Value::Ring64Tensor(x), Value::Ring64Tensor(y)) => {
                            match (x, y) {
                                (Ring64Tensor::Symbolic(x_op), Ring64Tensor::Symbolic(y_op)) => {
                                    Ring64Tensor::Symbolic(Operator::RingSub(op.clone()))
                                }
                                (Ring64Tensor::Concrete(x), Ring64Tensor::Concrete(y)) => {
                                    Self::abstract_kernel(x, y).into()
                                }
                                _ => unimplemented!()
                            }.into()
                        }
                        _ => unimplemented!()
                    }
                })
            }

            (Placement::Host(_), Ty::Ring128TensorTy, Ty::Ring128TensorTy) => {
                let op = self.clone();
                Box::new(move |x: Value, y: Value| -> Value {
                    match (x, y) {
                        (Value::Ring128Tensor(x), Value::Ring128Tensor(y)) => {
                            match (x, y) {
                                (Ring128Tensor::Symbolic(x_op), Ring128Tensor::Symbolic(y_op)) => {
                                    Ring128Tensor::Symbolic(Operator::RingSub(op.clone()))
                                }
                                (Ring128Tensor::Concrete(x), Ring128Tensor::Concrete(y)) => {
                                    Self::abstract_kernel(x, y).into()
                                }
                                _ => unimplemented!()
                            }.into()
                        }
                        _ => unimplemented!()
                    }
                })
            }

            _ => unimplemented!()
        }
    }

    fn abstract_kernel<T>(x: RingTensor<T>, y: RingTensor<T>) -> RingTensor<T>
    where
        RingTensor<T>: Sub<RingTensor<T>, Output=RingTensor<T>>,
    {
        x - y
    }
}

trait PlacementSample<O> {
    fn sample(&self) -> O;
}

// impl PlacementSample for HostPlacement {
//     type Output
// }

#[derive(Clone, Debug)]
struct RingSampleOp {
    ty: Ty,
    plc: Placement,
}

impl RingSampleOp {
    pub fn compile(&self) -> Box<dyn Fn() -> Value> {
        match (&self.plc, &self.ty) {

            (Placement::Host(_), Ty::Ring64TensorTy) => {
                Box::new(move || -> Value {
                    Ring64Tensor::Concrete(RingTensor(5)).into()
                })
            }

            (Placement::Host(_), Ty::Ring128TensorTy) => {
                Box::new(move || -> Value {
                    Ring128Tensor::Concrete(RingTensor(5)).into()
                })
            }

            _ => unimplemented!()
        }
    }
}

#[derive(Clone, Debug)]
struct ConstantOp {
    // val: Value,
    plc: Placement,
}

impl ConstantOp {
    //     pub fn compile(&self) -> Box<dyn Fn() -> Value> {
    //         let val = self.val.clone();

    //         match (&self.plc) {

    //             (Placement::Host(_)) => {
    //                 Box::new(move || -> Value {
    //                     val.clone()
    //                 })
    //             }

    //             _ => unimplemented!()
    //         }
    //     }

    //     fn abstract_kernel<T>(x: RingTensor<T>, y: RingTensor<T>) -> RingTensor<T>
    //     where
    //         RingTensor<T>: Add<RingTensor<T>, Output=RingTensor<T>>,
    //     {
    //         x + y
    //     }
}

#[derive(Clone, Debug)]
enum Operator {
    RingAdd(RingAddOp),
    RingSub(RingSubOp),
    RingSample(RingSampleOp),
    RepAdd(RepAddOp),
    RepShare(RepShareOp),
    Constant(ConstantOp),
}

impl From<RingAddOp> for Operator {
    fn from(op: RingAddOp) -> Self {
        Operator::RingAdd(op)
    }
}

impl From<RingSubOp> for Operator {
    fn from(op: RingSubOp) -> Self {
        Operator::RingSub(op)
    }
}

impl From<RingSampleOp> for Operator {
    fn from(op: RingSampleOp) -> Self {
        Operator::RingSample(op)
    }
}

impl From<RepAddOp> for Operator {
    fn from(op: RepAddOp) -> Self {
        Operator::RepAdd(op)
    }
}

enum HostPlacementInst {
    Symbolic,
    Concrete,
}

// impl HostPlacementInst {
//     fn placement(&self) -> Placement {
//         Placement::Host
//     }

//     fn sample_uniform(&self) -> RingTensor {
//         let op = RingSampleOp {
//             plc: self.placement(),
//         };



//         match self {
//             HostPlacementInst::Symbolic => {
//                 RingTensor::Symbolic(RingSampleOp{
//                     plc: self.placement(),
//                 })
//             }
//             HostPlacementInst::Concrete => {
//                 RingTensor::Concrete(ConcreteRingTensor(5))
//             }
//         }
//     }
// }


// fn rep_share(plc: HostPlacement, x: RingTensor) -> ReplicatedTensor {

// }

#[test]
fn test_rep_add() {
    // let x = ReplicatedTensor{
    //     shares: [
    //         [Ring64Tensor::Concrete(RingTensor(1)), Ring64Tensor::Concrete(RingTensor(2))],
    //         [Ring64Tensor::Concrete(RingTensor(2)), Ring64Tensor::Concrete(RingTensor(3))],
    //         [Ring64Tensor::Concrete(RingTensor(3)), Ring64Tensor::Concrete(RingTensor(1))],
    //     ]
    // };

    // let y = ReplicatedTensor{
    //     shares: [
    //         [Ring64Tensor::Concrete(RingTensor(1)), Ring64Tensor::Concrete(RingTensor(2))],
    //         [Ring64Tensor::Concrete(RingTensor(2)), Ring64Tensor::Concrete(RingTensor(3))],
    //         [Ring64Tensor::Concrete(RingTensor(3)), Ring64Tensor::Concrete(RingTensor(1))],
    //     ]
    // };

    // let x = ReplicatedTensor{
    //     shares: [
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //     ]
    // };

    // let y = ReplicatedTensor{
    //     shares: [
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring64Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //     ]
    // };

    let x = ReplicatedTensor{
        shares: [
            [Ring128Tensor::Concrete(RingTensor(1)), Ring128Tensor::Concrete(RingTensor(2))],
            [Ring128Tensor::Concrete(RingTensor(2)), Ring128Tensor::Concrete(RingTensor(3))],
            [Ring128Tensor::Concrete(RingTensor(3)), Ring128Tensor::Concrete(RingTensor(1))],
        ]
    };

    let y = ReplicatedTensor{
        shares: [
            [Ring128Tensor::Concrete(RingTensor(1)), Ring128Tensor::Concrete(RingTensor(2))],
            [Ring128Tensor::Concrete(RingTensor(2)), Ring128Tensor::Concrete(RingTensor(3))],
            [Ring128Tensor::Concrete(RingTensor(3)), Ring128Tensor::Concrete(RingTensor(1))],
        ]
    };

    // let x = ReplicatedTensor{
    //     shares: [
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //     ]
    // };

    // let y = ReplicatedTensor{
    //     shares: [
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //         [Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{})), Ring128Tensor::Symbolic(Operator::Constant(ConstantOp{}))],
    //     ]
    // };

    let ctx = ConcreteContext{};

    let rep_plc = ReplicatedPlacement{ players: ["alice".into(), "bob".into(), "carole".into()] };

    let z: ReplicatedTensor<_> = rep_plc.add(&ctx, &x, &y);
    println!("{:?}", z);

    assert!(false);
} 
