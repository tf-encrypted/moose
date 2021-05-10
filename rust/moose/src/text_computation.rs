use crate::computation::*;
use std::convert::TryFrom;
use nom::{
    character::complete::{space0, space1, multispace0, alphanumeric1, line_ending},
    bytes::complete::tag,
    multi::separated_list0,
    error::{make_error, ErrorKind},
    Err::Error,
    IResult,
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



// From nom::recepies
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

// Logical error definition

fn emit_logical_error(input: &str) -> IResult<&str, &str> {
    Err(Error(make_error(input, ErrorKind::Eof)))
}









named!(parse_computation<&str,Computation>,
    do_parse!(
        operations: separated_list0!(many1!(line_ending), parse_assignment) >>
        (Computation { operations} )
    )
);

named!(parse_assignment<&str,Operation>,
    do_parse!(
        multispace0 >>
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
        opt!(call!(parse_type_definition, 0)) >>
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
        types: call!(parse_type_definition, 2) >>
        (Operator::StdAdd(StdAddOp{
            lhs: types.0[0],
            rhs: types.0[1],
            }), args)
    )
);

// : (Float32Tensor, Float32Tensor) -> Float32Tensor
named_args!(parse_type_definition(arg_count: usize)<&str,(Vec<Ty>, Ty)>,
    do_parse!(
        space0 >>
        tag!(":") >>
        space0 >>
        args_types: call!(delimited(
            tag("("),
            separated_list0(tag(","), ws(parse_type)),
            tag(")"))) >>
        cond!(args_types.len() < arg_count, emit_logical_error) >>
        space0 >>
        tag!("->") >>
        space0 >>
        result_type: parse_type >>
        (args_types, result_type)
    )
);

named!(parse_type<&str,Ty>,
    alt!(
        value!(Ty::UnitTy, tag!("UnitTy")) |
        value!(Ty::StringTy, tag!("StringTy")) |
        value!(Ty::Float32Ty, tag!("Float32Ty")) |
        value!(Ty::Float64Ty, tag!("Float64Ty")) |
        value!(Ty::Ring64TensorTy, tag!("Ring64TensorTy")) |
        value!(Ty::Ring128TensorTy, tag!("Ring128TensorTy")) |
        value!(Ty::ShapeTy, tag!("ShapeTy")) |
        value!(Ty::SeedTy, tag!("SeedTy")) |
        value!(Ty::PrfKeyTy, tag!("PrfKeyTy")) |
        value!(Ty::NonceTy, tag!("NonceTy")) |
        value!(Ty::Float32TensorTy, tag!("Float32Tensor")) |
        value!(Ty::Float64TensorTy, tag!("Float64TensorTy")) |
        value!(Ty::Int8TensorTy, tag!("Int8TensorTy")) |
        value!(Ty::Int16TensorTy, tag!("Int16TensorTy")) |
        value!(Ty::Int32TensorTy, tag!("Int32TensorTy")) |
        value!(Ty::Int64TensorTy, tag!("Int64TensorTy")) |
        value!(Ty::Uint8TensorTy, tag!("Uint8TensorTy")) |
        value!(Ty::Uint16TensorTy, tag!("Uint16TensorTy")) |
        value!(Ty::Uint32TensorTy, tag!("Uint32TensorTy")) |
        value!(Ty::Uint64TensorTy, tag!("Uint64TensorTy"))
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_parsing() {
        let parsed_type = parse_type("UnitTy");
        let parsed = parse_type_definition(": (Float32Tensor, Float32Tensor) -> Float32Tensor", 0);
        println!("{:?} # {:?}", parsed_type, parsed);
    }

    #[test]
    fn test_constant() {
        let parsed = parse_assignment("x = Constant([1.0]): () -> Float32Tensor @alice");
        println!("{:?}", parsed);
        if let Ok((_, op)) = parsed {
            assert_eq!(op.name, "x");
            // TODO: a very temporary assert
            assert_eq!(format!("{:?}", op), "Operation { name: \"x\", kind: Constant(ConstantOp { value: String(\"[1.0]\") }), inputs: [], placement: Host(HostPlacement { owner: Role(\"alice\") }) }");
        }
    }

    #[test]
    fn test_stdadd() {
        let parsed = parse_assignment("z = StdAdd(x, y): (Float32Tensor, Float32Tensor) -> Float32Tensor @carole");
        println!("{:?}", parsed);
        if let Ok((_, op)) = parsed {
            assert_eq!(op.name, "z");
        }
    }

    #[test]
    fn test_stdadd_err() {
        let parsed = parse_assignment("z = StdAdd(x, y): (Float32Tensor) -> Float32Tensor @carole");
        println!("{:?}", parsed);
        if let Ok((_, op)) = parsed {
            assert_eq!(op.name, "z");
        }
    }


    #[test]
    fn test_sample_computation() {
        let parsed = parse_computation("x = Constant([1.0]) @alice
            y = Constant([1.0]): () -> Float32Tensor @bob
            z = StdAdd(x, y): (Float32Tensor, Float32Tensor) -> Float32Tensor @carole");
        println!("{:?}", parsed);
        if let Ok((_, comp)) = parsed {
            println!("Computation {:?}", comp);
        }
    }

}