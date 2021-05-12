use crate::computation::*;
use std::convert::TryFrom;
use nom::{
    character::complete::{space0, alphanumeric1, line_ending},
    bytes::complete::{tag, take_until},
    multi::{many1, separated_list0},
    error::{make_error, ErrorKind, ParseError},
    branch::{alt},
    combinator::{all_consuming, opt, map},
    Err::Error,
    IResult,
    sequence::{delimited, preceded, tuple}
};

impl TryFrom<&str> for Computation {
    type Error = anyhow::Error;

    fn try_from(source: &str) -> anyhow::Result<Computation> {
        match parse_computation::<(&str, ErrorKind)>(source) {
            Err(e) => Err(anyhow::anyhow!("Failed to parse {} due to {}", source, e)),

            Ok((_, computation)) => Ok(computation)
        }
    }
}



// From nom::recepies
fn ws<'a, F: 'a, O, E: ParseError<&'a str>>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
  where
  F: Fn(&'a str) -> IResult<&'a str, O, E>,
{
  delimited(
    space0,
    inner,
    space0
  )
}

// Identical to nom::separated_list0, but errors if a child parser errors (isntead of terminating)
fn separated_list0_err<I, O, O2, E, F, G>(
    mut sep: G,
    mut f: F,
  ) -> impl FnMut(I) -> IResult<I, Vec<O>, E>
  where
    I: Clone + PartialEq,
    F: nom::Parser<I, O, E>,
    G: nom::Parser<I, O2, E>,
    E: ParseError<I>,
  {
    move |mut i: I| {
      let mut res = Vec::new();
  
      match f.parse(i.clone()) {
        Err(nom::Err::Error(_)) => return Ok((i, res)),
        Err(e) => return Err(e),
        Ok((i1, o)) => {
          res.push(o);
          i = i1;
        }
      }
  
      loop {
        match sep.parse(i.clone()) {
          Err(nom::Err::Error(_)) => return Ok((i, res)),
          Err(e) => return Err(e),
          Ok((i1, _)) => {
            if i1 == i {
              return Err(nom::Err::Error(E::from_error_kind(i1, ErrorKind::SeparatedList)));
            }
  
            match f.parse(i1.clone()) {
              // This is the line I need to comment out: 
              // Err(nom::Err::Error(_)) => return Ok((i, res)),
              Err(e) => return Err(e),
              Ok((i2, o)) => {
                res.push(o);
                i = i2;
              }
            }
          }
        }
      }
    }
  }

fn parse_computation<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Computation, E> {
    all_consuming(map(separated_list0_err(many1(line_ending), parse_assignment),
        |operations| Computation { operations}
    ))(input)
}

fn parse_assignment<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Operation, E> {
    let (input, identifier) = ws(alphanumeric1)(input)?;
    let (input, _) = tag("=")(input)?;
    let (input, operator) = ws(parse_operator)(input)?;
    let (input, placement) = ws(parse_placement)(input)?;
    Ok((input,
        Operation {
            name: identifier.into(),
            kind: operator.0,
            inputs: operator.1,
            placement,
        }
    ))
}

fn parse_placement<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Placement, E> {
    map(
        tuple((tag("@"), alphanumeric1)),
        |(_, name)| 
            Placement::Host(HostPlacement {
                owner: Role::from(name),
            })
    )(input)
}

fn parse_operator<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (Operator, Vec<String>), E> {
    alt((
        preceded(tag("Constant"), constant),
        preceded(tag("StdAdd"), stdadd),
        // TODO: rest of the definitions
    ))(input)
}

fn argument_list<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Vec<String>, E> {
    delimited(
        tag("("),
        separated_list0(tag(","), map(ws(alphanumeric1), |s| s.to_string())),
        tag(")"))(input)
}

fn constant<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (Operator, Vec<String>), E> {
    let (input, x) = delimited(
        tag("("),
        take_until(")"),
        tag(")"))(input)?;
    let (input, _optional_types) = opt(parse_type_definition0)(input)?;

    Ok((input,
        (Operator::Constant(ConstantOp{value: Value::String(x.to_string())}), vec![])
    ))
}

fn stdadd<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (Operator, Vec<String>), E> {
    let (input, args) = argument_list(input)?;
    let (input, types) = parse_type_definition(input, 2)?;
    Ok((input,
        (Operator::StdAdd(StdAddOp{
            lhs: types.0[0],
            rhs: types.0[1],
        }),
        args)
    ))
}

// : (Float32Tensor, Float32Tensor) -> Float32Tensor
fn parse_type_definition<'a, E: 'a + ParseError<&'a str>>(input: &'a str, arg_count: usize) -> IResult<&'a str, (Vec<Ty>, Ty), E> {
    let (input, _) = ws(tag(":"))(input)?;
    let (input, args_types) = delimited(
        tag("("),
        separated_list0(tag(","), ws(parse_type)),
        tag(")"))(input)?;
    let (input, _) = ws(tag("->"))(input)?;
    let (input, result_type) = ws(parse_type)(input)?;

    if args_types.len() < arg_count {
        Err(Error(make_error(input, ErrorKind::Tag))) // TODO: Custom error message
    } else {
        Ok((input, (args_types, result_type)))
    }
}

fn parse_type_definition0<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (Vec<Ty>, Ty), E> {
    parse_type_definition(input, 0)
}

fn parse_type<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Ty, E> {
    let (i, type_name) = alphanumeric1(input)?;
    match type_name {
        "Unit" => Ok((i, Ty::UnitTy)),
        "String" => Ok((i, Ty::StringTy)),
        "Float32" => Ok((i, Ty::Float32Ty)),
        "Float64" => Ok((i, Ty::Float64Ty)),
        "Ring64Tensor" => Ok((i, Ty::Ring64TensorTy)),
        "Ring128Tensor" => Ok((i, Ty::Ring128TensorTy)),
        "Shape" => Ok((i, Ty::ShapeTy)),
        "Seed" => Ok((i, Ty::SeedTy)),
        "PrfKey" => Ok((i, Ty::PrfKeyTy)),
        "Nonce" => Ok((i, Ty::NonceTy)),
        "Float32Tensor" => Ok((i, Ty::Float32TensorTy)),
        "Float64Tensor" => Ok((i, Ty::Float64TensorTy)),
        "Int8Tensor" => Ok((i, Ty::Int8TensorTy)),
        "Int16Tensor" => Ok((i, Ty::Int16TensorTy)),
        "Int32Tensor" => Ok((i, Ty::Int32TensorTy)),
        "Int64Tensor" => Ok((i, Ty::Int64TensorTy)),
        "Uint8Tensor" => Ok((i, Ty::Uint8TensorTy)),
        "Uint16Tensor" => Ok((i, Ty::Uint16TensorTy)),
        "Uint32Tensor" => Ok((i, Ty::Uint32TensorTy)),
        "Uint64Tensor" => Ok((i, Ty::Uint64TensorTy)),
        _ => Err(Error(make_error(input, ErrorKind::Tag)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nom::error::{convert_error, VerboseError};

    #[test]
    fn test_type_parsing() {
        let parsed_type = parse_type::<(&str, ErrorKind)>("Unit");
        let parsed = parse_type_definition::<(&str, ErrorKind)>(": (Float32Tensor, Float32Tensor) -> Float32Tensor", 0);
        println!("{:?} # {:?}", parsed_type, parsed);

        let parsed: IResult<_, _, VerboseError<&str>> = parse_type("blah");
        if let Err(Error(e)) = parsed {
            println!("Error! {}", convert_error("blah", e));
        }
    }

    #[test]
    fn test_constant() {
        let parsed = parse_assignment::<(&str, ErrorKind)>("x = Constant([1.0]): () -> Float32Tensor @alice");
        println!("{:?}", parsed);
        if let Ok((_, op)) = parsed {
            assert_eq!(op.name, "x");
            // TODO: a very temporary assert
            assert_eq!(format!("{:?}", op), "Operation { name: \"x\", kind: Constant(ConstantOp { value: String(\"[1.0]\") }), inputs: [], placement: Host(HostPlacement { owner: Role(\"alice\") }) }");
        }
    }

    #[test]
    fn test_stdadd() {
        let parsed = parse_assignment::<(&str, ErrorKind)>("z = StdAdd(x, y): (Float32Tensor, Float32Tensor) -> Float32Tensor @carole");
        println!("{:?}", parsed);
        if let Ok((_, op)) = parsed {
            assert_eq!(op.name, "z");
        }
    }

    #[test]
    fn test_stdadd_err() {
        let data = "z = StdAdd(x, y): (Float32Tensor) -> Float32Tensor @carole";
        let parsed: IResult<_, _, VerboseError<&str>> = parse_assignment(data);
        if let Err(Error(e)) = parsed {
            println!("Error! {}", convert_error(data, e));
        }
    }


    #[test]
    fn test_sample_computation() {
        let parsed = parse_computation::<(&str, ErrorKind)>("x = Constant([1.0]) @alice
            y = Constant([1.0]): () -> Float32Tensor @bob
            z = StdAdd(x, y): (Float32Tensor, Float32Tensor) -> Float32Tensor @carole");
        if let Ok((_, comp)) = parsed {
            println!("Computation {:#?}", comp);
        }
    }

    #[test]
    fn test_sample_computation_err() {
        let data = "a = Constant('a') @alice
            err = StdAdd(x, y): (Float32Tensor) -> Float32Tensor @carole
            b = Constant('b') @alice";
        let parsed: IResult<_, _, VerboseError<&str>> = parse_computation(data);
        println!("{:?}", parsed);
        if let Err(Error(e)) = parsed {
            println!("Error! {}", convert_error(data, e));
        }
    }

}