use crate::computation::*;
use std::convert::TryFrom;
use nom::character::complete::{space0, space1, alphanumeric1};

impl TryFrom<&str> for Computation {
    type Error = anyhow::Error;

    fn try_from(source: &str) -> anyhow::Result<Computation> {
        match parse_assignment(source) {
            Err(e) => Err(anyhow::anyhow!("Failed to parse {} due to {}", source, e)),

            // TODO: Temporary line, only supporting one operation
            Ok((_, operation)) => Ok(Computation { operations: vec![operation] })
        }
        
    }
}

named!(parse_assignment<&str,Operation>,
    do_parse!(
        space0 >>
        tag!("let") >>
        space1 >>
        identifier: alphanumeric1 >>
        space1 >>
        tag!("=") >>
        space1 >>
        operator: parse_operator >>
        space1 >>
        placement: parse_placement >>
        (Operation {
            name: identifier.into(),
            kind: operator,
            inputs: vec![],
            placement: placement,
        })
    )
);

named!(parse_placement<&str,Placement>,
    do_parse!(
        tag!("@") >>
        name: alphanumeric1 >>
        (Placement::Host(HostPlacement {
            owner: Role::from(name),
        }))
    )
);

named!(parse_operator<&str,Operator>,
    alt!(
        preceded!(tag!("Identity"), identity) |
        preceded!(tag!("Constant"), constant)
        // TODO: rest of the definitions
    )
);

named!(identity<&str,Operator>,
    do_parse!(
        tag!("TODO: Fill this in") >>
        (Operator::Identity(IdentityOp{ty: Ty::Float32TensorTy}))
    )
);

named!(constant<&str,Operator>,
    do_parse!(
        space0 >>
        tag!("(") >>
        x: take_until!(")") >>
        tag!(")") >>
        (Operator::Constant(ConstantOp{value: Value::String(x.to_string())}))
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_assignment() {
        let parsed = parse_assignment("let x = Constant([1.0]) @alice");
        println!("{:?}", parsed);
        if let Ok((_, op)) = parsed {
            assert_eq!(op.name, "x");
            // TODO: a very temporary assert
            assert_eq!(format!("{:?}", op), "Operation { name: \"x\", kind: Constant(ConstantOp { value: String(\"[1.0]\") }), inputs: [], placement: Host(HostPlacement { owner: Role(\"alice\") }) }");
        }
    }
}