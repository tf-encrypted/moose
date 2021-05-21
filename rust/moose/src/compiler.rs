#![allow(dead_code)]
#![allow(unused_variables)]

enum Placement {
    Host,
    Rep,
}

enum PlacementInstantiation {
    Host,
    Rep,
}

enum Value {
    HostFixed(HostFixed),
    RepFixed(RepFixed),
}

impl HostFixed {
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

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

struct Context {}

impl Context {
    fn placement_instantiation(&self, plc: Placement) -> PlacementInstantiation {
        unimplemented!()
    }
}

struct StdAddOp {}

impl StdAddOp {
    fn execute(&self, ctx: &Context, plc: &Placement, x: Value, y: Value) -> Value {
        match (plc, x, y) {
            (Placement::Rep, Value::HostFixed(x), Value::HostFixed(y)) => {
                let x_owner = ctx.placement_instantiation(x.placement());
                let y_owner = ctx.placement_instantiation(y.placement());

                let xe = x_owner.share(x);
                let ye = y_owner.share(y);
                add(xe, ye)
            }
            (Placement::Rep, Value::RepFixed(x), Value::RepFixed(y)) => {
                unimplemented!()
            }
            _ => unimplemented!()
        }.into()
    }
}

struct StdMulOp {

}

struct RepAddOp {}

struct RepMulOp {}

fn share(x: HostFixed) -> RepFixed {
    unimplemented!()
}

fn add(x: RepFixed, y: RepFixed) -> RepFixed {
    unimplemented!()
}
