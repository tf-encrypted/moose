use crate::computation::*;
use crate::prim::{Nonce, PrfKey, Seed};
use crate::standard::Shape;
use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take_while_m_n},
    character::complete::{alphanumeric1, char, digit1, line_ending, multispace1, space0},
    combinator::{all_consuming, cut, map, map_opt, map_res, opt, value, verify},
    error::{convert_error, make_error, ErrorKind, ParseError, VerboseError},
    multi::{fill, fold_many0, many1, separated_list0},
    number::complete::{double, float},
    sequence::{delimited, preceded, tuple},
    Err::{Error, Failure},
    IResult,
};
use std::convert::TryFrom;

impl TryFrom<&str> for Computation {
    type Error = anyhow::Error;

    fn try_from(source: &str) -> anyhow::Result<Computation> {
        match parse_computation::<VerboseError<&str>>(source) {
            Err(Failure(e)) => Err(anyhow::anyhow!(
                "Failed to parse computation\n{}",
                convert_error(source, e)
            )),
            Err(e) => Err(anyhow::anyhow!("Failed to parse {} due to {}", source, e)),
            Ok((_, computation)) => Ok(computation),
        }
    }
}

/// Parse the computation line by line
fn parse_computation<'a, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Computation, E> {
    all_consuming(map(
        separated_list0(many1(line_ending), parse_assignment),
        |operations| Computation { operations },
    ))(input)
}

/// Parse an individual assignment in the form of
///
/// `Identifier = Operation : TypeDefinition @Placement`
fn parse_assignment<'a, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operation, E> {
    let (input, identifier) = ws(alphanumeric1)(input)?;
    let (input, _) = tag("=")(input)?;
    let (input, operator) = ws(parse_operator)(input)?;
    let (input, placement) = ws(parse_placement)(input)?;
    Ok((
        input,
        Operation {
            name: identifier.into(),
            kind: operator.0,
            inputs: operator.1,
            placement,
        },
    ))
}

/// Parse placement
fn parse_placement<'a, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Placement, E> {
    alt((
        preceded(
            tag("@Host"),
            cut(map(
                delimited(ws(tag("(")), alphanumeric1, ws(tag(")"))),
                |name| {
                    Placement::Host(HostPlacement {
                        owner: Role::from(name),
                    })
                },
            )),
        ),
        preceded(
            tag("@Replicated"),
            cut(map(
                delimited(
                    ws(tag("(")),
                    verify(
                        separated_list0(tag(","), ws(alphanumeric1)),
                        |v: &Vec<&str>| v.len() == 3,
                    ),
                    ws(tag(")")),
                ),
                |names| {
                    Placement::Replicated(ReplicatedPlacement {
                        owners: [
                            Role::from(names[0]),
                            Role::from(names[1]),
                            Role::from(names[2]),
                        ],
                    })
                },
            )),
        ),
    ))(input)
}

/// Constructs a parser for a simple unary operation.
macro_rules! std_unary {
    ($typ:expr, $sub:ident) => {
        |input: &'a str| {
            let (input, args) = argument_list(input)?;
            let (input, (args_types, _result_type)) = type_definition(1)(input)?;
            Ok((input, ($typ($sub { ty: args_types[0] }), args)))
        }
    };
}

/// Constructs a parser for a simple binary operation.
macro_rules! std_binary {
    ($typ:expr, $sub:ident) => {
        |input: &'a str| {
            let (input, args) = argument_list(input)?;
            let (input, (args_types, _result_type)) = type_definition(2)(input)?;
            Ok((
                input,
                (
                    $typ($sub {
                        lhs: args_types[0],
                        rhs: args_types[1],
                    }),
                    args,
                ),
            ))
        }
    };
}

/// Parse operator - maps names to structs.
fn parse_operator<'a, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, (Operator, Vec<String>), E> {
    let part1 = alt((
        preceded(
            tag("Identity"),
            cut(std_unary!(Operator::Identity, IdentityOp)),
        ),
        preceded(tag("Load"), cut(std_unary!(Operator::Load, LoadOp))),
        preceded(tag("Save"), cut(std_unary!(Operator::Save, SaveOp))),
        preceded(tag("Send"), cut(send_operator)),
        preceded(tag("Output"), cut(std_unary!(Operator::Output, OutputOp))),
        preceded(tag("Constant"), cut(constant)),
        preceded(tag("StdAdd"), cut(std_binary!(Operator::StdAdd, StdAddOp))),
        preceded(tag("StdSub"), cut(std_binary!(Operator::StdSub, StdSubOp))),
        preceded(tag("StdMul"), cut(std_binary!(Operator::StdMul, StdMulOp))),
        preceded(tag("StdDiv"), cut(std_binary!(Operator::StdDiv, StdDivOp))),
        preceded(tag("StdDot"), cut(std_binary!(Operator::StdDot, StdDotOp))),
        preceded(
            tag("StdOnes"),
            cut(std_unary!(Operator::StdOnes, StdOnesOp)),
        ),
        preceded(
            tag("StdReshape"),
            cut(std_unary!(Operator::StdReshape, StdReshapeOp)),
        ),
        preceded(
            tag("StdShape"),
            cut(std_unary!(Operator::StdShape, StdShapeOp)),
        ),
        preceded(
            tag("StdTranspose"),
            cut(std_unary!(Operator::StdTranspose, StdTransposeOp)),
        ),
        preceded(
            tag("StdInverse"),
            cut(std_unary!(Operator::StdInverse, StdInverseOp)),
        ),
    ));
    let part2 = alt((
        preceded(
            tag("RingAdd"),
            cut(std_binary!(Operator::RingAdd, RingAddOp)),
        ),
        preceded(
            tag("RingSub"),
            cut(std_binary!(Operator::RingSub, RingSubOp)),
        ),
        preceded(
            tag("RingMul"),
            cut(std_binary!(Operator::RingMul, RingMulOp)),
        ),
        preceded(
            tag("RingDot"),
            cut(std_binary!(Operator::RingDot, RingDotOp)),
        ),
        preceded(
            tag("RingShape"),
            cut(std_unary!(Operator::RingShape, RingShapeOp)),
        ),
        preceded(tag("StdMean"), cut(stdmean)),
        preceded(tag("RingSample"), cut(ring_sample)),
        preceded(tag("PrimDeriveSeed"), cut(prim_derive_seed)),
        preceded(tag("PrimGenPrfKey"), cut(prim_gen_prf_key)),
    ));
    alt((
        part1, part2,
        // TODO: rest of the definitions
    ))(input)
}

/// Parses the Constant
fn constant<'a, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, (Operator, Vec<String>), E> {
    let (input, x) = delimited(tag("("), ws(value_literal), tag(")"))(input)?;
    let (input, _optional_types) = opt(type_definition(0))(input)?;

    Ok((input, (Operator::Constant(ConstantOp { value: x }), vec![])))
}

/// Parses StdMean
fn stdmean<'a, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, (Operator, Vec<String>), E> {
    let (input, args) = argument_list(input)?;
    let (input, opt_axis) = opt(attributes_single("axis", parse_int))(input)?;
    let (input, (args_types, _result_type)) = type_definition(1)(input)?;
    Ok((
        input,
        (
            Operator::StdMean(StdMeanOp {
                ty: args_types[0],
                axis: opt_axis,
            }),
            args,
        ),
    ))
}

/// Parses Send operator
fn send_operator<'a, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, (Operator, Vec<String>), E> {
    let (input, args) = argument_list(input)?;
    let (input, (rendezvous_key, receiver)) =
        attributes_pair("rendezvous_key", string, "receiver", string)(input)?;
    let (input, _opt_type) = opt(type_definition(0))(input)?;
    Ok((
        input,
        (
            Operator::Send(SendOp {
                rendezvous_key,
                receiver: Role::from(receiver),
            }),
            args,
        ),
    ))
}

/// Parses RingSample
fn ring_sample<'a, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, (Operator, Vec<String>), E> {
    let (input, args) = argument_list(input)?;
    let (input, opt_max_value) = opt(attributes_single("max_value", parse_int))(input)?;
    let (input, (_args_types, result_type)) = type_definition(2)(input)?;
    Ok((
        input,
        (
            Operator::RingSample(RingSampleOp {
                output: result_type,
                max_value: opt_max_value,
            }),
            args,
        ),
    ))
}

/// Parses PrimGenPrfKey
fn prim_gen_prf_key<'a, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, (Operator, Vec<String>), E> {
    let (input, args) = argument_list(input)?;
    Ok((input, (Operator::PrimGenPrfKey(PrimGenPrfKeyOp), args)))
}

/// Parses PrimDeriveSeed
fn prim_derive_seed<'a, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, (Operator, Vec<String>), E> {
    let (input, args) = ws(argument_list)(input)?;
    let (input, nonce) = attributes_single("nonce", map(vector(parse_int), Nonce))(input)?;
    Ok((
        input,
        (Operator::PrimDeriveSeed(PrimDeriveSeedOp { nonce }), args),
    ))
}

/// Parses list of arguments in the form of
///
/// `(name, name, name)`
fn argument_list<'a, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Vec<String>, E> {
    delimited(
        tag("("),
        separated_list0(tag(","), map(ws(alphanumeric1), |s| s.to_string())),
        tag(")"),
    )(input)
}

/// Parses list of attributes. Currently only supporting one attribute
fn attributes_single<'a, O, F: 'a, E: 'a + ParseError<&'a str>>(
    name: &'a str,
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(
        tuple((ws(tag("{")), ws(tag(name)), ws(tag("=")))),
        inner,
        ws(tag("}")),
    )
}

/// Parses list of attributes. Parses two attributes.
fn attributes_pair<'a, O1, O2, F1: 'a, F2: 'a, E: 'a + ParseError<&'a str>>(
    name1: &'a str,
    inner1: F1,
    name2: &'a str,
    inner2: F2,
) -> impl FnMut(&'a str) -> IResult<&'a str, (O1, O2), E>
where
    F1: FnMut(&'a str) -> IResult<&'a str, O1, E>,
    F2: FnMut(&'a str) -> IResult<&'a str, O2, E>,
{
    delimited(
        ws(tag("{")),
        nom::branch::permutation((
            map(tuple((ws(tag(name1)), ws(tag("=")), inner1)), |(_, _, v)| v),
            map(tuple((ws(tag(name2)), ws(tag("=")), inner2)), |(_, _, v)| v),
        )),
        ws(tag("}")),
    )
}

/// Parses operator's type definition in the form of
///
/// `: (Float32Tensor, Float32Tensor) -> Float32Tensor`
///
/// * `arg_count` - the number of required arguments
fn type_definition<'a, E: 'a + ParseError<&'a str>>(
    arg_count: usize,
) -> impl FnMut(&'a str) -> IResult<&'a str, (Vec<Ty>, Ty), E> {
    move |input: &'a str| {
        let (input, _) = ws(tag(":"))(input)?;
        let (input, args_types) = verify(
            delimited(
                tag("("),
                separated_list0(tag(","), ws(parse_type)),
                tag(")"),
            ),
            |v: &Vec<Ty>| v.len() >= arg_count,
        )(input)?;
        let (input, _) = ws(tag("->"))(input)?;
        let (input, result_type) = ws(parse_type)(input)?;

        Ok((input, (args_types, result_type)))
    }
}

/// Parse individual type's literal
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
        _ => Err(Error(make_error(input, ErrorKind::Tag))),
    }
}

/// Parse literal value.
fn value_literal<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Value, E> {
    alt((
        map(tuple((parse_hex, type_literal("Seed"))), |(v, _)| {
            Value::Seed(Seed(v))
        }),
        map(tuple((parse_hex, type_literal("PrfKey"))), |(v, _)| {
            Value::PrfKey(PrfKey(v))
        }),
        map(tuple((float, type_literal("Float32"))), |(x, _)| {
            Value::Float32(x)
        }),
        map(tuple((double, opt(type_literal("Float64")))), |(x, _)| {
            Value::Float64(x)
        }),
        map(tuple((string, opt(type_literal("String")))), |(s, _)| {
            Value::String(s)
        }),
        map(
            tuple((vector(parse_int), type_literal("Ring64Tensor"))),
            |(v, _)| Value::Ring64Tensor(v.into()),
        ),
        map(
            tuple((vector(parse_int), type_literal("Ring128Tensor"))),
            |(v, _)| Value::Ring128Tensor(v.into()),
        ),
        map(
            tuple((vector(parse_int), type_literal("Shape"))),
            |(v, _): (Vec<usize>, &str)| Value::Shape(Shape(v)),
        ),
        map(
            tuple((vector(parse_int), type_literal("Nonce"))),
            |(v, _)| Value::Nonce(Nonce(v)),
        ),
        // 1D arrars
        alt((
            map(
                tuple((vector(parse_int), type_literal("Int8Tensor"))),
                |(v, _)| Value::Int8Tensor(v.into()),
            ),
            map(
                tuple((vector(parse_int), type_literal("Int16Tensor"))),
                |(v, _)| Value::Int16Tensor(v.into()),
            ),
            map(
                tuple((vector(parse_int), type_literal("Int32Tensor"))),
                |(v, _)| Value::Int32Tensor(v.into()),
            ),
            map(
                tuple((vector(parse_int), type_literal("Int64Tensor"))),
                |(v, _)| Value::Int64Tensor(v.into()),
            ),
            map(
                tuple((vector(parse_int), type_literal("Uint8Tensor"))),
                |(v, _)| Value::Uint8Tensor(v.into()),
            ),
            map(
                tuple((vector(parse_int), type_literal("Uint16Tensor"))),
                |(v, _)| Value::Uint16Tensor(v.into()),
            ),
            map(
                tuple((vector(parse_int), type_literal("Uint32Tensor"))),
                |(v, _)| Value::Uint32Tensor(v.into()),
            ),
            map(
                tuple((vector(parse_int), type_literal("Uint64Tensor"))),
                |(v, _)| Value::Uint64Tensor(v.into()),
            ),
            map(
                tuple((vector(float), type_literal("Float32Tensor"))),
                |(v, _)| Value::Float32Tensor(v.into()),
            ),
            map(
                tuple((vector(double), type_literal("Float64Tensor"))),
                |(v, _)| Value::Float64Tensor(v.into()),
            ),
        )),
        // 2D arrars
        alt((
            map(
                tuple((vector2(parse_int), type_literal("Int8Tensor"))),
                |(v, _)| Value::Int8Tensor(v.into()),
            ),
            map(
                tuple((vector2(parse_int), type_literal("Int16Tensor"))),
                |(v, _)| Value::Int16Tensor(v.into()),
            ),
            map(
                tuple((vector2(parse_int), type_literal("Int32Tensor"))),
                |(v, _)| Value::Int32Tensor(v.into()),
            ),
            map(
                tuple((vector2(parse_int), type_literal("Int64Tensor"))),
                |(v, _)| Value::Int64Tensor(v.into()),
            ),
            map(
                tuple((vector2(parse_int), type_literal("Uint8Tensor"))),
                |(v, _)| Value::Uint8Tensor(v.into()),
            ),
            map(
                tuple((vector2(parse_int), type_literal("Uint16Tensor"))),
                |(v, _)| Value::Uint16Tensor(v.into()),
            ),
            map(
                tuple((vector2(parse_int), type_literal("Uint32Tensor"))),
                |(v, _)| Value::Uint32Tensor(v.into()),
            ),
            map(
                tuple((vector2(parse_int), type_literal("Uint64Tensor"))),
                |(v, _)| Value::Uint64Tensor(v.into()),
            ),
            map(
                tuple((vector2(float), type_literal("Float32Tensor"))),
                |(v, _)| Value::Float32Tensor(v.into()),
            ),
            map(
                tuple((vector2(double), type_literal("Float64Tensor"))),
                |(v, _)| Value::Float64Tensor(v.into()),
            ),
        )),
    ))(input)
}

/// Expects the specified type literal to be present.
fn type_literal<'a, E: 'a + ParseError<&'a str>>(
    expected_type: &'a str,
) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str, E> {
    move |input: &'a str| {
        let (input, _) = ws(tag(":"))(input)?;
        ws(tag(expected_type))(input)
    }
}

/// Parses the vector of items, using the supplied innter parser.
fn vector<'a, F: 'a, O, E: 'a + ParseError<&'a str>>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, Vec<O>, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(tag("["), separated_list0(ws(tag(",")), inner), tag("]"))
}

fn vector2<'a, F: 'a, O: 'a, E: 'a>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, ndarray::Array2<O>, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E> + Copy,
    O: Clone,
    E: ParseError<&'a str>,
{
    move |input: &'a str| {
        let (input, vec2) = vector(vector(inner))(input)?;
        let mut data = Vec::new();

        let ncols = vec2.first().map_or(0, |row| row.len());
        let mut nrows = 0;

        for row in &vec2 {
            data.extend_from_slice(&row);
            nrows += 1;
        }

        ndarray::Array2::from_shape_vec((nrows, ncols), data)
            .map(|a| (input, a))
            .map_err(|_: ndarray::ShapeError| Error(make_error(input, ErrorKind::MapRes)))
    }
}

/// Parse integer (or anything implementing FromStr from decimal digits)
fn parse_int<'a, O: std::str::FromStr, E: 'a + ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, O, E> {
    map_res(digit1, |s: &str| s.parse::<O>())(input)
        .map_err(|_: nom::Err<nom::error::Error<&str>>| Error(make_error(input, ErrorKind::MapRes)))
}

/// Parse a single byte, writte as two hex character.
///
/// Leading '0' is mandatory for bytes 0x00 - 0x0F.
fn parse_hex_u8<'a, E>(input: &'a str) -> IResult<&'a str, u8, E>
where
    E: ParseError<&'a str>,
{
    let parse_hex = take_while_m_n(2, 2, |c: char| c.is_ascii_hexdigit());
    map_res(parse_hex, move |hex| u8::from_str_radix(hex, 16))(input)
        .map_err(|_: nom::Err<nom::error::Error<&str>>| Error(make_error(input, ErrorKind::MapRes)))
}

/// Parse a hux dump, without any separators. Errors out if there is not enough data to fill an array of length N.
fn parse_hex<'a, E, const N: usize>(input: &'a str) -> IResult<&'a str, [u8; N], E>
where
    E: ParseError<&'a str>,
{
    let mut buf: [u8; N] = [0; N];
    let (rest, ()) = fill(parse_hex_u8, &mut buf)(input)?;
    Ok((rest, buf))
}

/// From nom::recepies
/// Wraps the innner parser in optional spaces.
fn ws<'a, F: 'a, O, E: ParseError<&'a str>>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: Fn(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(space0, inner, space0)
}

/// Parse an escaped character: \n, \t, \r, \u{00AC}, etc.
/// From nom examples (MIT licesnse, so it is ok)
fn parse_hex_u32<'a, E>(input: &'a str) -> IResult<&'a str, u32, E>
where
    E: ParseError<&'a str>,
{
    let parse_hex = take_while_m_n(1, 6, |c: char| c.is_ascii_hexdigit());
    let parse_delimited_hex = preceded(char('u'), delimited(char('{'), parse_hex, char('}')));
    map_res(parse_delimited_hex, move |hex| u32::from_str_radix(hex, 16))(input)
        .map_err(|_: nom::Err<nom::error::Error<&str>>| Error(make_error(input, ErrorKind::MapRes)))
}

/// Parses a single unicode character.
/// From nom examples (MIT licesnse, so it is ok)
fn parse_unicode<'a, E>(input: &'a str) -> IResult<&'a str, char, E>
where
    E: ParseError<&'a str>,
{
    map_opt(parse_hex_u32, std::char::from_u32)(input)
}

/// Parses any supported escaped character
/// From nom examples (MIT licesnse, so it is ok)
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
/// From nom examples (MIT licesnse, so it is ok)
fn parse_escaped_whitespace<'a, E: ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, &'a str, E> {
    preceded(char('\\'), multispace1)(input)
}

/// Parse a non-empty block of text that doesn't include \ or "
/// From nom examples (MIT licesnse, so it is ok)
fn parse_literal<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, &'a str, E> {
    let not_quote_slash = is_not("\"\\");
    verify(not_quote_slash, |s: &str| !s.is_empty())(input)
}

/// A string fragment contains a fragment of a string being parsed: either
/// a non-empty Literal (a series of non-escaped characters), a single
/// parsed escaped character, or a block of escaped whitespace.
/// From nom examples (MIT licesnse, so it is ok)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StringFragment<'a> {
    Literal(&'a str),
    EscapedChar(char),
    EscapedWS,
}

/// Combine parse_literal, parse_escaped_whitespace, and parse_escaped_char
/// into a StringFragment.
/// From nom examples (MIT licesnse, so it is ok)
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
/// From nom examples (MIT licesnse, so it is ok)
fn string<'a, E>(input: &'a str) -> IResult<&'a str, String, E>
where
    E: ParseError<&'a str>,
{
    let build_string = fold_many0(parse_fragment, String::new(), |mut string, fragment| {
        match fragment {
            StringFragment::Literal(s) => string.push_str(s),
            StringFragment::EscapedChar(c) => string.push(c),
            StringFragment::EscapedWS => {}
        }
        string
    });
    delimited(char('"'), build_string, char('"'))(input)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (_, parsed_ring64_tensor) =
            value_literal::<(&str, ErrorKind)>("[1,2,3] : Ring64Tensor")?;
        assert_eq!(
            parsed_ring64_tensor,
            Value::Ring64Tensor(vec![1, 2, 3].into())
        );
        let (_, parsed_ring128_tensor) =
            value_literal::<(&str, ErrorKind)>("[1,2,3] : Ring128Tensor")?;
        assert_eq!(
            parsed_ring128_tensor,
            Value::Ring128Tensor(vec![1, 2, 3].into())
        );
        let (_, parsed_shape) = value_literal::<(&str, ErrorKind)>("[1,2,3] : Shape")?;
        assert_eq!(parsed_shape, Value::Shape(Shape(vec![1, 2, 3])));
        let (_, parsed_u8_tensor) = value_literal::<(&str, ErrorKind)>("[1,2,3] : Uint8Tensor")?;
        assert_eq!(parsed_u8_tensor, Value::Uint8Tensor(vec![1, 2, 3].into()));
        let (_, parsed_seed) =
            value_literal::<(&str, ErrorKind)>("529c2fc9bf573d077f45f42b19cfb8d4 : Seed")?;
        assert_eq!(
            parsed_seed,
            Value::Seed(Seed([
                0x52, 0x9c, 0x2f, 0xc9, 0xbf, 0x57, 0x3d, 0x07, 0x7f, 0x45, 0xf4, 0x2b, 0x19, 0xcf,
                0xb8, 0xd4
            ]))
        );
        Ok(())
    }

    #[test]
    fn test_type_parsing() -> Result<(), anyhow::Error> {
        let (_, parsed_type) = parse_type::<(&str, ErrorKind)>("Unit")?;
        assert_eq!(parsed_type, Ty::UnitTy);
        let (_, parsed) = type_definition::<(&str, ErrorKind)>(0)(
            ": (Float32Tensor, Float64Tensor) -> Uint16Tensor",
        )?;
        assert_eq!(
            parsed,
            (
                vec!(Ty::Float32TensorTy, Ty::Float64TensorTy),
                Ty::Uint16TensorTy
            )
        );

        let parsed: IResult<_, _, VerboseError<&str>> = parse_type("blah");
        if let Err(Error(e)) = parsed {
            println!("Error! {}", convert_error("blah", e));
        }
        Ok(())
    }

    #[test]
    fn test_constant() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "x = Constant([1.0] : Float32Tensor): () -> Float32Tensor @Host(alice)",
        )?;
        assert_eq!(op.name, "x");
        assert_eq!(format!("{:?}", op), "Operation { name: \"x\", kind: Constant(ConstantOp { value: Float32Tensor(StandardTensor([1.0], shape=[1], strides=[1], layout=CFcf (0xf), dynamic ndim=1)) }), inputs: [], placement: Host(HostPlacement { owner: Role(\"alice\") }) }");
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "x = Constant([[1.0, 2.0], [3.0, 4.0]] : Float32Tensor): () -> Float32Tensor @Replicated(alice, bob, charlie)",
        )?;
        println!("{:#?}", op);
        Ok(())
    }

    #[test]
    fn test_stdbinary() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z = StdAdd(x, y): (Float32Tensor, Float32Tensor) -> Float32Tensor @Host(carole)",
        )?;
        assert_eq!(op.name, "z");
        println!("{:#?}", op);
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z = StdMul(x, y): (Float32Tensor, Float32Tensor) -> Float32Tensor @Host(carole)",
        )?;
        assert_eq!(op.name, "z");
        println!("{:#?}", op);
        Ok(())
    }

    #[test]
    fn test_stdadd_err() {
        let data = "z = StdAdd(x, y): (Float32Tensor) -> Float32Tensor @Host(carole)";
        let parsed: IResult<_, _, VerboseError<&str>> = parse_assignment(data);
        if let Err(Failure(e)) = parsed {
            println!("Error! {}", convert_error(data, e));
        }
    }

    #[test]
    fn test_primgenprfkey() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>("key = PrimGenPrfKey() @Host(alice)")?;
        assert_eq!(op.name, "key");
        Ok(())
    }

    #[test]
    fn test_seed() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "seed = PrimDeriveSeed(key) {nonce = [1, 2, 3]} @Host(alice)",
        )?;
        assert_eq!(op.name, "seed");
        Ok(())
    }

    #[test]
    fn test_send() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "send = Send() {rendezvous_key = \"abc\" receiver = \"bob\"} @Host(alice)",
        )?;
        assert_eq!(op.name, "send");
        println!("{:#?}", op);
        Ok(())
    }

    #[test]
    fn test_output() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z = Output(x10): (Ring64Tensor) -> Unit @Host(alice)",
        )?;
        assert_eq!(op.name, "z");
        Ok(())
    }

    #[test]
    fn test_ring_sample() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "x10 = RingSample(shape, seed){max_value = 1}: (Shape, Seed) -> Ring64Tensor @Host(alice)",
        )?;
        assert_eq!(op.name, "x10");
        Ok(())
    }

    #[test]
    fn test_sample_computation() {
        let parsed = parse_computation::<(&str, ErrorKind)>(
            "x = Constant([1.0]: Float32Tensor) @Host(alice)
            y = Constant([1.0]: Float32Tensor): () -> Float32Tensor @bob
            z = StdAdd(x, y): (Float32Tensor, Float32Tensor) -> Float32Tensor @Host(carole)",
        );
        if let Ok((_, comp)) = parsed {
            println!("Computation {:#?}", comp);
        }
        // TODO: Asserts
    }

    #[test]
    fn test_sample_computation_err() {
        let data = "a = Constant(\"a\") @Host(alice)
            err = StdAdd(x, y): (Float32Tensor) -> Float32Tensor @Host(carole)
            b = Constant(\"b\") @Host(alice)";
        let parsed: IResult<_, _, VerboseError<&str>> = parse_computation(data);
        println!("{:?}", parsed);
        if let Err(Failure(e)) = parsed {
            println!("Error!\n{}", convert_error(data, e));
        }
        // TODO: Asserts
    }
}
