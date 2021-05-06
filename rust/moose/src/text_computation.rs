use crate::computation::*;
use std::convert::TryFrom;
use nom::{
    character::complete::{space0, space1, alphanumeric1, line_ending},
    sequence::{delimited}
};

impl TryFrom<&str> for Computation {
    type Error = anyhow::Error;

    fn try_from(source: &str) -> anyhow::Result<Computation> {
        match parse_computation(source) {
            Err(e) => Err(anyhow::anyhow!("Failed to parse {} due to {}", source, e)),

            Ok((_, computation)) => Ok(computation)
        }
        
    }
}

// Adapted from nom::recepies
fn ws<'a, F: 'a, O, E: nom::error::ParseError<&'a str>>(inner: F) -> impl FnMut(&'a str) -> nom::IResult<&'a str, O, E>
  where
  F: Fn(&'a str) -> nom::IResult<&'a str, O, E>,
{
  delimited(
    space0,
    inner,
    space0
  )
}
  

named!(parse_computation<&str,Computation>,
    do_parse!(
        operations: separated_list0!(many1!(line_ending), parse_assignment) >>
        (Computation { operations} )
    )
);

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
        space0 >>
        (Operation {
            name: identifier.into(),
            kind: operator.0,
            inputs: operator.1,
            placement,
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

named!(parse_operator<&str,(Operator, Vec<String>)>,
    alt!(
        preceded!(tag!("Constant"), constant) |
        preceded!(tag!("StdAdd"), stdadd)
        // TODO: rest of the definitions
    )
);

named!(constant<&str,(Operator, Vec<String>)>,
    do_parse!(
        space0 >>
        tag!("(") >>
        x: take_until!(")") >>
        tag!(")") >>
        (Operator::Constant(ConstantOp{value: Value::String(x.to_string())}), vec![])
    )
);

named!(stdadd<&str,(Operator, Vec<String>)>,
    do_parse!(
        space0 >>
        args: delimited!(
            tag!("("),
            separated_list0!(tag!(","), map!(ws(alphanumeric1), |s| s.to_string())),
            tag!(")")) >>
        (Operator::StdAdd(StdAddOp{
            lhs: Ty::Float32TensorTy,
            rhs: Ty::Float32TensorTy,
            }), args)
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

    #[test]
    fn test_sample_computation() {
        let parsed = parse_computation("let x = Constant([1.0]) @alice
            let y = Constant([1.0]) @bob
            let z = StdAdd(x, y) @carole");
        println!("{:?}", parsed);
    }

}