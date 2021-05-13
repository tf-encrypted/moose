use crate::computation::*;
use std::convert::TryFrom;
use nom::{
    character::complete::{space0, alphanumeric0, alphanumeric1, line_ending, multispace1, char, one_of, digit1},
    bytes::complete::{escaped, tag, take_until, take_while_m_n, is_not},
    multi::{fold_many0, fill, many0, many1, separated_list0},
    number::complete::{double, float},
    error::{make_error, ErrorKind, ParseError, FromExternalError},
    branch::{alt},
    combinator::{all_consuming, cut, opt, map, not, verify, map_opt, map_res, value},
    Err::{Error, Failure},
    IResult,
    sequence::{delimited, preceded, terminated, tuple}
};
use crate::standard::{Shape,};
use crate::prim::{Nonce, PrfKey, Seed};
use std::convert::TryInto;


impl TryFrom<&str> for Computation {
    type Error = anyhow::Error;

    fn try_from(source: &str) -> anyhow::Result<Computation> {
        match parse_computation::<(&str, ErrorKind)>(source) {
            Err(e) => Err(anyhow::anyhow!("Failed to parse {} due to {}", source, e)),

            Ok((_, computation)) => Ok(computation)
        }
    }
}



fn parse_computation<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Computation, E> {
    all_consuming(map(separated_list0(many1(line_ending), parse_assignment),
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
        preceded(tag("Constant"), cut(constant)),
        preceded(tag("StdAdd"), cut(stdadd)),
        // TODO: rest of the definitions
    ))(input)
}

fn constant<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (Operator, Vec<String>), E> {
    let (input, x) = delimited(
        tag("("),
        ws(value_literal),
        tag(")"))(input)?;
    let (input, _optional_types) = opt(type_definition0)(input)?;

    Ok((input,
        (Operator::Constant(ConstantOp{value: x}), vec![])
    ))
}

fn stdadd<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (Operator, Vec<String>), E> {
    let (input, args) = argument_list(input)?;
    let (input, (args_types, _result_type)) = type_definition(input, 2)?;
    Ok((input,
        (Operator::StdAdd(StdAddOp{
            lhs: args_types[0],
            rhs: args_types[1],
        }),
        args)
    ))
}

fn argument_list<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Vec<String>, E> {
    delimited(
        tag("("),
        separated_list0(tag(","), map(ws(alphanumeric1), |s| s.to_string())),
        tag(")"))(input)
}

// : (Float32Tensor, Float32Tensor) -> Float32Tensor
fn type_definition<'a, E: 'a + ParseError<&'a str>>(input: &'a str, arg_count: usize) -> IResult<&'a str, (Vec<Ty>, Ty), E> {
    let (input, _) = ws(tag(":"))(input)?;
    let (input, args_types) = verify(delimited(
        tag("("),
        separated_list0(tag(","), ws(parse_type)),
        tag(")")), |v: &Vec<Ty>| v.len() >= arg_count)(input)?;
    let (input, _) = ws(tag("->"))(input)?;
    let (input, result_type) = ws(parse_type)(input)?;

    Ok((input, (args_types, result_type)))
}

fn type_definition0<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (Vec<Ty>, Ty), E> {
    type_definition(input, 0)
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

fn value_literal<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Value, E> {
    alt((
        map(tuple((parse_hex, type_literal("Seed"))), |(v, _)| Value::Seed(Seed(v))),
        map(tuple((parse_hex, type_literal("PrfKey"))), |(v, _)| Value::PrfKey(PrfKey(v))),
        map(tuple((float, type_literal("Float32"))), |(x, _)| Value::Float32(x)),
        map(tuple((double, opt(type_literal("Float64")))), |(x, _)| Value::Float64(x)),
        map(tuple((string, opt(type_literal("String")))), |(s, _)| Value::String(s)),
        map(tuple((vector(parse_int), type_literal("Ring64Tensor"))), |(v, _)| Value::Ring64Tensor(v.into())),
        map(tuple((vector(parse_int), type_literal("Ring128Tensor"))), |(v, _)| Value::Ring128Tensor(v.into())),
        map(tuple((vector(parse_int), type_literal("Shape"))), |(v, _): (Vec<usize>, &str)| Value::Shape(Shape(v))),
        map(tuple((vector(parse_int), type_literal("Nonce"))), |(v, _)| Value::Nonce(Nonce(v))),
        map(tuple((vector(parse_int), type_literal("Int8Tensor"))), |(v, _)| Value::Int8Tensor(v.into())),
        map(tuple((vector(parse_int), type_literal("Int16Tensor"))), |(v, _)| Value::Int16Tensor(v.into())),
        map(tuple((vector(parse_int), type_literal("Int32Tensor"))), |(v, _)| Value::Int32Tensor(v.into())),
        map(tuple((vector(parse_int), type_literal("Int64Tensor"))), |(v, _)| Value::Int64Tensor(v.into())),
        map(tuple((vector(parse_int), type_literal("Uint8Tensor"))), |(v, _)| Value::Uint8Tensor(v.into())),
        map(tuple((vector(parse_int), type_literal("Uint16Tensor"))), |(v, _)| Value::Uint16Tensor(v.into())),
        map(tuple((vector(parse_int), type_literal("Uint32Tensor"))), |(v, _)| Value::Uint32Tensor(v.into())),
        map(tuple((vector(parse_int), type_literal("Uint64Tensor"))), |(v, _)| Value::Uint64Tensor(v.into())),
        map(tuple((vector(float), type_literal("Float32Tensor"))), |(v, _)| Value::Float32Tensor(v.into())),
        map(tuple((vector(double), type_literal("Float64Tensor"))), |(v, _)| Value::Float64Tensor(v.into())),
    ))(input)
}

fn type_literal<'a, E: 'a + ParseError<&'a str>>(expected_type: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str, E> {
    move |input: &'a str| {
        let (input, _) = ws(tag(":"))(input)?;
        ws(tag(expected_type))(input)
    }
}

fn vector<'a, F: 'a, O, E: 'a + ParseError<&'a str>>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, Vec<O>, E>
  where
  F: Fn(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(
        tag("["),
        separated_list0(ws(tag(",")), inner),
        tag("]")
    )
}

fn parse_int<'a, O: std::str::FromStr, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, O, E> {
    map_res(digit1, |s: &str| s.parse::<O>())(input).map_err(|_: nom::Err<nom::error::Error<&str>>| Error(make_error(input, ErrorKind::MapRes)))
}

fn parse_hex_u8<'a, E>(input: &'a str) -> IResult<&'a str, u8, E>
where
  E: ParseError<&'a str>,
{
    let parse_hex = take_while_m_n(2, 2, |c: char| c.is_ascii_hexdigit());
    map_res(parse_hex, move |hex| u8::from_str_radix(hex, 16))(input)
        .map_err(|_: nom::Err<nom::error::Error<&str>>| Error(make_error(input, ErrorKind::MapRes)))
}

fn parse_hex<'a, E, const N: usize>(input: &'a str) -> IResult<&'a str, [u8; N], E>
where
  E: ParseError<&'a str>,
{
    let mut buf: [u8; N] = [0; N];
    let (rest, ()) = fill(parse_hex_u8, &mut buf)(input)?;
    Ok((rest, buf))
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

// From nom examples (MIT licesnse, so it is ok)
/// Parse an escaped character: \n, \t, \r, etc.
/// Does not support unicode escaping as in \u{00AC} 

fn parse_hex_u32<'a, E>(input: &'a str) -> IResult<&'a str, u32, E>
where
  E: ParseError<&'a str>,
{
    let parse_hex = take_while_m_n(1, 6, |c: char| c.is_ascii_hexdigit());
    let parse_delimited_hex = preceded(
      char('u'),
      delimited(char('{'), parse_hex, char('}')),
    );
    map_res(parse_delimited_hex, move |hex| u32::from_str_radix(hex, 16))(input)
        .map_err(|_: nom::Err<nom::error::Error<&str>>| Error(make_error(input, ErrorKind::MapRes)))
}

fn parse_unicode<'a, E>(input: &'a str) -> IResult<&'a str, char, E>
where
  E: ParseError<&'a str>,
{
  map_opt(parse_hex_u32, |value| std::char::from_u32(value))(input)
}


fn parse_escaped_char<'a, E>(input: &'a str) -> IResult<&'a str, char, E>
where
  E: ParseError<&'a str>,
{
  preceded(
    char('\\'),
    alt((
      parse_unicode,
      value('\n', char('n')),
      value('\r', char('r')),
      value('\t', char('t')),
      value('\u{08}', char('b')),
      value('\u{0C}', char('f')),
      value('\\', char('\\')),
      value('/', char('/')),
      value('"', char('"')),
    )),
  )(input)
}

/// Parse a backslash, followed by any amount of whitespace. This is used later
/// to discard any escaped whitespace.
fn parse_escaped_whitespace<'a, E: ParseError<&'a str>>(
  input: &'a str,
) -> IResult<&'a str, &'a str, E> {
  preceded(char('\\'), multispace1)(input)
}

/// Parse a non-empty block of text that doesn't include \ or "
fn parse_literal<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, &'a str, E> {
  let not_quote_slash = is_not("\"\\");
  verify(not_quote_slash, |s: &str| !s.is_empty())(input)
}

/// A string fragment contains a fragment of a string being parsed: either
/// a non-empty Literal (a series of non-escaped characters), a single
/// parsed escaped character, or a block of escaped whitespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StringFragment<'a> {
  Literal(&'a str),
  EscapedChar(char),
  EscapedWS,
}

/// Combine parse_literal, parse_escaped_whitespace, and parse_escaped_char
/// into a StringFragment.
fn parse_fragment<'a, E>(input: &'a str) -> IResult<&'a str, StringFragment<'a>, E>
where
  E: ParseError<&'a str>,
{
  alt((
    map(parse_literal, StringFragment::Literal),
    map(parse_escaped_char, StringFragment::EscapedChar),
    value(StringFragment::EscapedWS, parse_escaped_whitespace),
  ))(input)
}

/// Parse a string. Use a loop of parse_fragment and push all of the fragments
/// into an output string.
fn string<'a, E>(input: &'a str) -> IResult<&'a str, String, E>
where
  E: ParseError<&'a str>,
{
  let build_string = fold_many0(
    parse_fragment,
    String::new(),
    |mut string, fragment| {
      match fragment {
        StringFragment::Literal(s) => string.push_str(s),
        StringFragment::EscapedChar(c) => string.push(c),
        StringFragment::EscapedWS => {}
      }
      string
    },
  );
  delimited(char('"'), build_string, char('"'))(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nom::error::{convert_error, VerboseError};

    #[test]
    fn test_value_literal() -> Result<(), anyhow::Error> {
        let (_, parsed_f64) = value_literal::<(&str, ErrorKind)>("1.23")?;
        assert_eq!(parsed_f64, Value::Float64(1.23));
        let (_, parsed_f32) = value_literal::<(&str, ErrorKind)>("1.23 : Float32")?;
        assert_eq!(parsed_f32, Value::Float32(1.23));
        let (_, parsed_f64) = value_literal::<(&str, ErrorKind)>("1.23 : Float64")?;
        assert_eq!(parsed_f64, Value::Float64(1.23));
        let (_, parsed_str) = value_literal::<(&str, ErrorKind)>("\"abc\"")?;
        assert_eq!(parsed_str, Value::String("abc".into()));
        let (_, parsed_str) = value_literal::<(&str, ErrorKind)>("\"abc\" : String")?;
        assert_eq!(parsed_str, Value::String("abc".into()));
        let (_, parsed_str) = value_literal::<(&str, ErrorKind)>("\"1.23\"")?;
        assert_eq!(parsed_str, Value::String("1.23".into()));
        let (_, parsed_str) = value_literal::<(&str, ErrorKind)>("\"1. 2\\\"3\"")?;
        assert_eq!(parsed_str, Value::String("1. 2\"3".into()));
        let (_, parsed_ring64_tensor) = value_literal::<(&str, ErrorKind)>("[1,2,3] : Ring64Tensor")?;
        assert_eq!(parsed_ring64_tensor, Value::Ring64Tensor(vec![1, 2, 3].into()));
        let (_, parsed_ring128_tensor) = value_literal::<(&str, ErrorKind)>("[1,2,3] : Ring128Tensor")?;
        assert_eq!(parsed_ring128_tensor, Value::Ring128Tensor(vec![1, 2, 3].into()));
        let (_, parsed_shape) = value_literal::<(&str, ErrorKind)>("[1,2,3] : Shape")?;
        assert_eq!(parsed_shape, Value::Shape(Shape(vec![1, 2, 3])));
        let (_, parsed_u8_tensor) = value_literal::<(&str, ErrorKind)>("[1,2,3] : Uint8Tensor")?;
        assert_eq!(parsed_u8_tensor, Value::Uint8Tensor(vec![1, 2, 3].into()));
        let (_, parsed_seed) = value_literal::<(&str, ErrorKind)>("529c2fc9bf573d077f45f42b19cfb8d4 : Seed")?;
        assert_eq!(parsed_seed, Value::Seed(Seed([0x52, 0x9c, 0x2f, 0xc9, 0xbf, 0x57, 0x3d, 0x07, 0x7f, 0x45, 0xf4, 0x2b, 0x19, 0xcf, 0xb8, 0xd4])));
        Ok(())
    }

    #[test]
    fn test_type_parsing() {
        let parsed_type = parse_type::<(&str, ErrorKind)>("Unit");
        let parsed = type_definition::<(&str, ErrorKind)>(": (Float32Tensor, Float32Tensor) -> Float32Tensor", 0);
        println!("{:?} # {:?}", parsed_type, parsed);

        let parsed: IResult<_, _, VerboseError<&str>> = parse_type("blah");
        if let Err(Error(e)) = parsed {
            println!("Error! {}", convert_error("blah", e));
        }
    }

    #[test]
    fn test_constant() {
        let parsed = parse_assignment::<(&str, ErrorKind)>("x = Constant(\"[1.0]\"): () -> Float32Tensor @alice");
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
        let data = "a = Constant(\"a\") @alice
            err = StdAdd(x, y): (Float32Tensor) -> Float32Tensor @carole
            b = Constant(\"b\") @alice";
        let parsed: IResult<_, _, VerboseError<&str>> = parse_computation(data);
        println!("{:?}", parsed);
        if let Err(Failure(e)) = parsed {
            println!("Error!\n{}", convert_error(data, e));
        }
    }

}