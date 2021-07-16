use crate::computation::*;
use crate::prim::{RawNonce, RawPrfKey, RawSeed};
use crate::standard::{RawShape, Shape};
use nom::{
    branch::{alt, permutation},
    bytes::complete::{is_not, tag, take_while_m_n},
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace1, space0},
    combinator::{all_consuming, cut, eof, map, map_opt, map_res, opt, recognize, value, verify},
    error::{
        context, convert_error, make_error, ContextError, ErrorKind, ParseError, VerboseError,
    },
    multi::{fill, fold_many0, many0, many1, separated_list0},
    number::complete::{double, float},
    sequence::{delimited, pair, preceded, tuple},
    Err::{Error, Failure},
    IResult,
};
use std::convert::TryFrom;
use std::str::FromStr;

impl TryFrom<&str> for Computation {
    type Error = anyhow::Error;

    fn try_from(source: &str) -> anyhow::Result<Computation> {
        verbose_parse_computation(source)
    }
}

impl TryFrom<String> for Computation {
    type Error = anyhow::Error;

    fn try_from(source: String) -> anyhow::Result<Computation> {
        verbose_parse_computation(&source)
    }
}

impl TryFrom<&str> for Constant {
    type Error = anyhow::Error;

    fn try_from(source: &str) -> anyhow::Result<Constant> {
        constant_literal::<(&str, ErrorKind)>(source)
            .map(|(_, v)| v)
            .map_err(|_| anyhow::anyhow!("Failed to parse constant literal {}", source))
    }
}

impl FromStr for Constant {
    type Err = anyhow::Error;
    fn from_str(source: &str) -> Result<Self, Self::Err> {
        constant_literal::<(&str, ErrorKind)>(source)
            .map(|(_, v)| v)
            .map_err(|_| anyhow::anyhow!("Failed to parse constant literal {}", source))
    }
}

impl TryFrom<&str> for Value {
    type Error = anyhow::Error;

    fn try_from(source: &str) -> anyhow::Result<Value> {
        value_literal::<(&str, ErrorKind)>(source)
            .map(|(_, v)| v)
            .map_err(|_| anyhow::anyhow!("Failed to parse value literal {}", source))
    }
}

impl FromStr for Value {
    type Err = anyhow::Error;
    fn from_str(source: &str) -> Result<Self, Self::Err> {
        value_literal::<(&str, ErrorKind)>(source)
            .map(|(_, v)| v)
            .map_err(|_| anyhow::anyhow!("Failed to parse value literal {}", source))
    }
}

/// Parses the computation and returns a verbose error description if it fails.
pub fn verbose_parse_computation(source: &str) -> anyhow::Result<Computation> {
    match parse_computation::<VerboseError<&str>>(source) {
        Err(Failure(e)) => Err(anyhow::anyhow!(
            "Failed to parse computation\n{}",
            convert_error(source, e)
        )),
        Err(e) => Err(anyhow::anyhow!("Failed to parse {} due to {}", source, e)),
        Ok((_, computation)) => Ok(computation),
    }
}

/// Parses the computation line by line.
fn parse_computation<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Computation, E> {
    let body = all_consuming(map(
        separated_list0(many1(multispace1), cut(parse_line)),
        |operations| Computation {
            operations: operations.into_iter().flatten().collect(),
        },
    ));
    // Allow any number of empty lines around the body
    delimited(many0(multispace1), body, many0(multispace1))(input)
}

/// Parses a single logical line of the textual IR
fn parse_line<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Option<Operation>, E> {
    alt((
        recognize_comment,
        map(parse_assignment, Some),
        value(None, eof),
    ))(input)
}

/// Recognizes and consumes away a comment
fn recognize_comment<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Option<Operation>, E> {
    value(
        None, // Output is thrown away.
        pair(ws(tag("//")), is_not("\n\r")),
    )(input)
}

/// Parses an individual assignment.
///
/// Accepts an assignment in the form of
///
/// `Identifier = Operation : TypeDefinition @Placement`
fn parse_assignment<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operation, E> {
    let (input, identifier) = ws(identifier)(input)?;
    let (input, _) = tag("=")(input)?;
    let (input, operator) = ws(parse_operator)(input)?;
    let (input, args) = opt(argument_list)(input)?;
    let (input, placement) = ws(parse_placement)(input)?;
    Ok((
        input,
        Operation {
            name: identifier.into(),
            kind: operator,
            inputs: args.unwrap_or_default(),
            placement,
        },
    ))
}

/// Parses placement.
fn parse_placement<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Placement, E> {
    alt((
        preceded(
            tag("@Host"),
            cut(context(
                "Expecting alphanumeric host name as in @Host(alice)",
                map(
                    delimited(ws(tag("(")), alphanumeric1, ws(tag(")"))),
                    |name| {
                        Placement::Host(HostPlacement {
                            owner: Role::from(name),
                        })
                    },
                ),
            )),
        ),
        preceded(
            tag("@Replicated"),
            cut(context(
                "Expecting host names triplet as in @Replicated(alice, bob, charlie)",
                map(
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
                ),
            )),
        ),
    ))(input)
}

/// Parses list of attributes.
macro_rules! attributes {
    ($inner:expr) => {
        |input: &'a str| delimited(ws(tag("{")), permutation($inner), ws(tag("}")))(input)
    };
}

/// Constructs a parser for a simple unary operation.
macro_rules! std_unary {
    ($sub:ident) => {
        |input: &'a str| {
            let (input, sig) = type_definition(1)(input)?;
            Ok((input, $sub { sig }.into()))
        }
    };
}

/// Constructs a parser for a simple binary operation.
macro_rules! std_binary {
    ($sub:ident) => {
        |input: &'a str| {
            let (input, sig) = type_definition(2)(input)?;
            Ok((input, $sub { sig }.into()))
        }
    };
}

/// Constructs a parser for a simple binary operation.
macro_rules! operation_on_axis {
    ($sub:ident) => {
        |input: &'a str| {
            let (input, opt_axis) = opt(attributes_single("axis", parse_int))(input)?;
            let (input, sig) = type_definition(1)(input)?;
            Ok((
                input,
                $sub {
                    sig,
                    axis: opt_axis,
                }
                .into(),
            ))
        }
    };
}

/// Parses operator - maps names to structs.
fn parse_operator<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let part1 = alt((
        preceded(tag(IdentityOp::SHORT_NAME), cut(std_unary!(IdentityOp))),
        preceded(tag(LoadOp::SHORT_NAME), cut(std_unary!(LoadOp))),
        preceded(tag(SendOp::SHORT_NAME), cut(send_operator)),
        preceded(tag(ReceiveOp::SHORT_NAME), cut(receive_operator)),
        preceded(tag(InputOp::SHORT_NAME), cut(input_operator)),
        preceded(tag(OutputOp::SHORT_NAME), cut(std_unary!(OutputOp))),
        preceded(tag(ConstantOp::SHORT_NAME), cut(constant)),
        preceded(tag(ShapeOp::SHORT_NAME), cut(std_unary!(ShapeOp))),
        preceded(tag(BitFillOp::SHORT_NAME), cut(bit_fill)),
        preceded(tag(RingFillOp::SHORT_NAME), cut(ring_fill)),
        preceded(tag(SaveOp::SHORT_NAME), cut(save_operator)),
        preceded(tag(StdAddOp::SHORT_NAME), cut(std_binary!(StdAddOp))),
        preceded(tag(StdSubOp::SHORT_NAME), cut(std_binary!(StdSubOp))),
        preceded(tag(StdMulOp::SHORT_NAME), cut(std_binary!(StdMulOp))),
        preceded(tag(StdDivOp::SHORT_NAME), cut(std_binary!(StdDivOp))),
        preceded(tag(StdDotOp::SHORT_NAME), cut(std_binary!(StdDotOp))),
        preceded(
            tag(StdMeanOp::SHORT_NAME),
            cut(operation_on_axis!(StdMeanOp)),
        ),
        preceded(tag(StdExpandDimsOp::SHORT_NAME), cut(stdexpanddims)),
        preceded(tag(StdReshapeOp::SHORT_NAME), cut(std_unary!(StdReshapeOp))),
        preceded(tag(StdAtLeast2DOp::SHORT_NAME), cut(stdatleast2d)),
        preceded(tag(StdSliceOp::SHORT_NAME), cut(stdslice)),
    ));
    let part2 = alt((
        preceded(tag(StdSumOp::SHORT_NAME), cut(operation_on_axis!(StdSumOp))),
        preceded(tag(StdOnesOp::SHORT_NAME), cut(std_unary!(StdOnesOp))),
        preceded(tag(StdConcatenateOp::SHORT_NAME), cut(stdconcatenate)),
        preceded(
            tag(StdTransposeOp::SHORT_NAME),
            cut(std_unary!(StdTransposeOp)),
        ),
        preceded(tag(StdInverseOp::SHORT_NAME), cut(std_unary!(StdInverseOp))),
        preceded(tag(RingAddOp::SHORT_NAME), cut(std_binary!(RingAddOp))),
        preceded(tag(RingSubOp::SHORT_NAME), cut(std_binary!(RingSubOp))),
        preceded(tag(RingMulOp::SHORT_NAME), cut(std_binary!(RingMulOp))),
        preceded(tag(RingDotOp::SHORT_NAME), cut(std_binary!(RingDotOp))),
        preceded(
            tag(RingSumOp::SHORT_NAME),
            cut(operation_on_axis!(RingSumOp)),
        ),
        preceded(tag(RingSampleOp::SHORT_NAME), cut(ring_sample)),
        preceded(tag(RingShlOp::SHORT_NAME), cut(ring_shl)),
        preceded(tag(RingShrOp::SHORT_NAME), cut(ring_shr)),
        preceded(tag(PrimDeriveSeedOp::SHORT_NAME), cut(prim_derive_seed)),
        preceded(tag(PrimPrfKeyGenOp::SHORT_NAME), cut(prim_gen_prf_key)),
        preceded(
            tag(FixedpointRingEncodeOp::SHORT_NAME),
            cut(fixed_point_ring_encode),
        ),
        preceded(
            tag(FixedpointRingDecodeOp::SHORT_NAME),
            cut(fixed_point_ring_decode),
        ),
        preceded(
            tag(FixedpointRingMeanOp::SHORT_NAME),
            cut(fixed_point_ring_mean),
        ),
    ));
    let part3 = alt((
        preceded(tag(RingInjectOp::SHORT_NAME), cut(ring_inject)),
        preceded(tag(BitExtractOp::SHORT_NAME), cut(bit_extract)),
        preceded(tag(BitSampleOp::SHORT_NAME), cut(bit_sample)),
        preceded(tag(BitXorOp::SHORT_NAME), cut(bit_xor)),
    ));
    alt((part1, part2, part3))(input)
}

/// Parses a Constant
fn constant<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, value) = attributes_single("value", constant_literal)(input)?;
    let (input, optional_type) = opt(type_definition(0))(input)?;
    let sig = optional_type.unwrap_or_else(|| Signature::nullary(value.ty()));

    Ok((input, ConstantOp { sig, value }.into()))
}

/// Parses a Send operator
fn send_operator<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, (rendezvous_key, receiver)) = attributes!((
        attributes_member("rendezvous_key", string),
        attributes_member("receiver", string)
    ))(input)?;
    let (input, optional_type) = opt(type_definition(0))(input)?;
    let sig = optional_type.unwrap_or_else(|| Signature::unary(Ty::Unknown, Ty::Unknown));
    Ok((
        input,
        SendOp {
            sig,
            rendezvous_key,
            receiver: Role::from(receiver),
        }
        .into(),
    ))
}

/// Parses a Receive operator
fn receive_operator<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, (rendezvous_key, sender)) = attributes!((
        attributes_member("rendezvous_key", string),
        attributes_member("sender", string)
    ))(input)?;
    let (input, sig) = type_definition(0)(input)?;
    Ok((
        input,
        ReceiveOp {
            sig,
            rendezvous_key,
            sender: Role::from(sender),
        }
        .into(),
    ))
}

/// Parses an Input operator
fn input_operator<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, arg_name) = attributes_single("arg_name", string)(input)?;
    let (input, sig) = type_definition(0)(input)?;
    Ok((input, InputOp { sig, arg_name }.into()))
}

/// Parses a StdExpandDims operator
fn stdexpanddims<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, axis) = attributes_single("axis", parse_int)(input)?;
    let (input, sig) = type_definition(1)(input)?;
    Ok((input, StdExpandDimsOp { sig, axis }.into()))
}

/// Parses a StdAtLeast2D operator.
fn stdatleast2d<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, to_column_vector) = attributes_single("to_column_vector", parse_bool)(input)?;
    let (input, sig) = type_definition(1)(input)?;
    Ok((
        input,
        StdAtLeast2DOp {
            sig,
            to_column_vector,
        }
        .into(),
    ))
}

/// Parses a StdSlice operator.
fn stdslice<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, (start, end)) = attributes!((
        attributes_member("start", parse_int),
        attributes_member("end", parse_int)
    ))(input)?;
    let (input, sig) = type_definition(1)(input)?;
    Ok((input, StdSliceOp { sig, start, end }.into()))
}

/// Parses a StdConcatenate operator.
fn stdconcatenate<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, axis) = attributes_single("axis", parse_int)(input)?;
    let (input, sig) = type_definition(1)(input)?;
    Ok((input, StdConcatenateOp { sig, axis }.into()))
}

/// Parses a RingSample operator.
fn ring_sample<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, opt_max_value) = opt(attributes_single("max_value", parse_int))(input)?;
    let (input, sig) = type_definition(0)(input)?;
    Ok((
        input,
        RingSampleOp {
            sig,
            max_value: opt_max_value,
        }
        .into(),
    ))
}

/// Parses a BitFill operator.
fn bit_fill<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, value) = attributes_single("value", constant_literal)(input)?;
    let (input, sig) = type_definition(1)(input)?;
    Ok((input, BitFillOp { sig, value }.into()))
}

/// Parses a RingFill operator.
fn ring_fill<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, value) = attributes_single("value", constant_literal)(input)?;
    let (input, sig) = type_definition(1)(input)?;
    Ok((input, RingFillOp { sig, value }.into()))
}

/// Parses a RingShl operator.
fn ring_shl<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, amount) = attributes_single("amount", parse_int)(input)?;
    let (input, sig) = type_definition(0)(input)?;
    Ok((input, RingShlOp { sig, amount }.into()))
}

/// Parses a RingShr operator.
fn ring_shr<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, amount) = attributes_single("amount", parse_int)(input)?;
    let (input, sig) = type_definition(0)(input)?;
    Ok((input, RingShrOp { sig, amount }.into()))
}

/// Parses a PrimPrfKeyGen operator.
fn prim_gen_prf_key<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    Ok((
        input,
        PrimPrfKeyGenOp {
            sig: Signature::nullary(Ty::PrfKey),
        }
        .into(),
    ))
}

/// Parses a PrimDeriveSeed operator.
fn prim_derive_seed<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, sync_key) = attributes_single("sync_key", map(vector(parse_int), RawNonce))(input)?;
    let (input, opt_sig) = opt(type_definition(0))(input)?;
    let sig = opt_sig.unwrap_or_else(|| Signature::nullary(Ty::Seed));
    Ok((input, PrimDeriveSeedOp { sig, sync_key }.into()))
}

/// Parses a FixedpointRingEncode operator.
fn fixed_point_ring_encode<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, (scaling_base, scaling_exp)) = attributes!((
        attributes_member("scaling_base", parse_int),
        attributes_member("scaling_exp", parse_int)
    ))(input)?;
    let (input, sig) = type_definition(0)(input)?;
    Ok((
        input,
        FixedpointRingEncodeOp {
            sig,
            scaling_base,
            scaling_exp,
        }
        .into(),
    ))
}

/// Parses a FixedpointRingDecode operator.
fn fixed_point_ring_decode<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, (scaling_base, scaling_exp)) = attributes!((
        attributes_member("scaling_base", parse_int),
        attributes_member("scaling_exp", parse_int)
    ))(input)?;
    let (input, sig) = type_definition(1)(input)?;
    Ok((
        input,
        FixedpointRingDecodeOp {
            sig,
            scaling_base,
            scaling_exp,
        }
        .into(),
    ))
}

/// Parses a Save operator.
fn save_operator<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, sig) = type_definition(2)(input)?;
    Ok((input, SaveOp { sig }.into()))
}

/// Parses a FixedpointRingMean operator.
fn fixed_point_ring_mean<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, (scaling_base, scaling_exp, axis)) = attributes!((
        attributes_member("scaling_base", parse_int),
        attributes_member("scaling_exp", parse_int),
        opt(attributes_member("axis", parse_int))
    ))(input)?;

    let (input, sig) = type_definition(0)(input)?;
    Ok((
        input,
        FixedpointRingMeanOp {
            sig,
            axis,
            scaling_base,
            scaling_exp,
        }
        .into(),
    ))
}

/// Parses a RingInject operator.
fn ring_inject<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, bit_idx) = attributes_single("bit_idx", parse_int)(input)?;
    let (input, sig) = type_definition(1)(input)?;
    Ok((input, RingInjectOp { sig, bit_idx }.into()))
}

/// Parses a BitExtract operator.
fn bit_extract<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, bit_idx) = attributes_single("bit_idx", parse_int)(input)?;
    let (input, sig) = type_definition(1)(input)?;
    Ok((input, BitExtractOp { sig, bit_idx }.into()))
}

/// Parses a BitSample operator.
fn bit_sample<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, opt_args) = opt(type_definition(0))(input)?;
    let sig = opt_args.unwrap_or_else(|| Signature::nullary(Ty::BitTensor));
    Ok((input, BitSampleOp { sig }.into()))
}

/// Parses a BitXor operator.
fn bit_xor<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, opt_sig) = opt(type_definition(0))(input)?;
    let sig = opt_sig.unwrap_or_else(|| Signature::nullary(Ty::BitTensor));
    Ok((input, BitXorOp { sig }.into()))
}

/// Parses list of arguments.
///
/// Accepts input in the form of
///
/// `(name, name, name)`
fn argument_list<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Vec<String>, E> {
    context(
        "Expecting comma separated list of identifiers",
        delimited(
            tag("("),
            separated_list0(tag(","), map(ws(identifier), |s| s.to_string())),
            tag(")"),
        ),
    )(input)
}

/// Parses list of attributes when there is only one attribute.
///
/// This is an optimization to avoid permutation cast for the simple case.
fn attributes_single<'a, O, F: 'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    name: &'a str,
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(ws(tag("{")), attributes_member(name, inner), ws(tag("}")))
}

/// Parses a single attribute with an optional comma at the end.
fn attributes_member<'a, O, F: 'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    name1: &'a str,
    inner1: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    map(
        tuple((ws(tag(name1)), ws(tag("=")), inner1, opt(ws(tag(","))))),
        |(_, _, v, _)| v,
    )
}

/// Parses operator's type definition
///
/// Accepts input in the form of
///
/// `: (Float32Tensor, Float32Tensor) -> Float32Tensor`
///
/// * `arg_count` - the number of required arguments
fn type_definition<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    arg_count: usize,
) -> impl FnMut(&'a str) -> IResult<&'a str, Signature, E> {
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

        match args_types.len() {
            0 => Ok((input, Signature::nullary(result_type))),
            1 => Ok((input, Signature::unary(args_types[0], result_type))),
            2 => Ok((
                input,
                Signature::binary(args_types[0], args_types[1], result_type),
            )),
            3 => Ok((
                input,
                Signature::ternary(args_types[0], args_types[1], args_types[2], result_type),
            )),
            _ => Err(Error(make_error(input, ErrorKind::Tag))),
        }
    }
}

/// Parses an individual type's literal
fn parse_type<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Ty, E> {
    let (i, type_name) = alphanumeric1(input)?;
    match type_name {
        "Unknown" => Ok((i, Ty::Unknown)),
        "Shape" => Ok((i, Ty::Shape)),
        "Seed" => Ok((i, Ty::Seed)),
        "PrfKey" => Ok((i, Ty::PrfKey)),
        "Nonce" => Ok((i, Ty::Nonce)),
        "String" => Ok((i, Ty::String)),
        "BitTensor" => Ok((i, Ty::BitTensor)),
        "Ring64Tensor" => Ok((i, Ty::Ring64Tensor)),
        "Ring128Tensor" => Ok((i, Ty::Ring128Tensor)),
        "Float32Tensor" => Ok((i, Ty::Float32Tensor)),
        "Float64Tensor" => Ok((i, Ty::Float64Tensor)),
        "Int8Tensor" => Ok((i, Ty::Int8Tensor)),
        "Int16Tensor" => Ok((i, Ty::Int16Tensor)),
        "Int32Tensor" => Ok((i, Ty::Int32Tensor)),
        "Int64Tensor" => Ok((i, Ty::Int64Tensor)),
        "Uint8Tensor" => Ok((i, Ty::Uint8Tensor)),
        "Uint16Tensor" => Ok((i, Ty::Uint16Tensor)),
        "Uint32Tensor" => Ok((i, Ty::Uint32Tensor)),
        "Uint64Tensor" => Ok((i, Ty::Uint64Tensor)),
        "Fixed64Tensor" => Ok((i, Ty::Fixed64Tensor)),
        "Fixed128Tensor" => Ok((i, Ty::Fixed128Tensor)),
        "Replicated64Tensor" => Ok((i, Ty::Replicated64Tensor)),
        "Replicated128Tensor" => Ok((i, Ty::Replicated128Tensor)),
        "ReplicatedBitTensor" => Ok((i, Ty::ReplicatedBitTensor)),
        "ReplicatedSetup" => Ok((i, Ty::ReplicatedSetup)),
        "Additive64Tensor" => Ok((i, Ty::Additive64Tensor)),
        "Additive128Tensor" => Ok((i, Ty::Additive128Tensor)),
        "Unit" => Ok((i, Ty::Unit)),
        "Float32" => Ok((i, Ty::Float32)),
        "Float64" => Ok((i, Ty::Float64)),
        "Ring64" => Ok((i, Ty::Ring64)),
        "Ring128" => Ok((i, Ty::Ring128)),
        _ => Err(Error(make_error(input, ErrorKind::Tag))),
    }
}

fn constant_literal_helper<'a, O1, F1, F2, E>(
    expected_type: &'a str,
    parser: F1,
    mapper: F2,
) -> impl FnMut(&'a str) -> IResult<&'a str, Constant, E>
where
    F1: FnMut(&'a str) -> IResult<&'a str, O1, E>,
    F2: FnMut(O1) -> Constant,
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    map(
        preceded(
            tag(expected_type),
            delimited(ws(tag("(")), parser, ws(tag(")"))),
        ),
        mapper,
    )
}

/// Parses a literal for a constant (not a placed value).
fn constant_literal<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Constant, E> {
    alt((
        constant_literal_helper("Seed", parse_hex, |v| Constant::RawSeed(RawSeed(v))),
        constant_literal_helper("PrfKey", parse_hex, |v| Constant::RawPrfKey(RawPrfKey(v))),
        constant_literal_helper("Float32", float, Constant::Float32),
        constant_literal_helper("Float64", double, Constant::Float64),
        constant_literal_helper("String", string, Constant::String),
        map(ws(string), Constant::String), // Alternative syntax for strings - no type
        constant_literal_helper("Ring64", parse_int, Constant::Ring64),
        constant_literal_helper("Ring128", parse_int, Constant::Ring128),
        constant_literal_helper("Shape", vector(parse_int), |v| {
            Constant::RawShape(RawShape(v))
        }),
        // constant_literal_helper("Nonce", vector(parse_int), |v| Value::Nonce(Nonce(v))), // TODO
        // 1D arrars
        alt((
            constant_literal_helper("Int8Tensor", vector(parse_int), |v| {
                Constant::Int8Tensor(v.into())
            }),
            constant_literal_helper("Int16Tensor", vector(parse_int), |v| {
                Constant::Int16Tensor(v.into())
            }),
            constant_literal_helper("Int32Tensor", vector(parse_int), |v| {
                Constant::Int32Tensor(v.into())
            }),
            constant_literal_helper("Int64Tensor", vector(parse_int), |v| {
                Constant::Int64Tensor(v.into())
            }),
            constant_literal_helper("Uint8Tensor", vector(parse_int), |v| {
                Constant::Uint8Tensor(v.into())
            }),
            constant_literal_helper("Uint16Tensor", vector(parse_int), |v| {
                Constant::Uint16Tensor(v.into())
            }),
            constant_literal_helper("Uint32Tensor", vector(parse_int), |v| {
                Constant::Uint32Tensor(v.into())
            }),
            constant_literal_helper("Uint64Tensor", vector(parse_int), |v| {
                Constant::Uint64Tensor(v.into())
            }),
            constant_literal_helper("Float32Tensor", vector(float), |v| {
                Constant::Float32Tensor(v.into())
            }),
            constant_literal_helper("Float64Tensor", vector(double), |v| {
                Constant::Float64Tensor(v.into())
            }),
            constant_literal_helper("Ring64Tensor", vector(parse_int), |v| {
                Constant::Ring64Tensor(v.into())
            }),
            constant_literal_helper("Ring128Tensor", vector(parse_int), |v| {
                Constant::Ring128Tensor(v.into())
            }),
        )),
        // 2D arrars
        alt((
            constant_literal_helper("Int8Tensor", vector2(parse_int), |v| {
                Constant::Int8Tensor(v.into())
            }),
            constant_literal_helper("Int16Tensor", vector2(parse_int), |v| {
                Constant::Int16Tensor(v.into())
            }),
            constant_literal_helper("Int32Tensor", vector2(parse_int), |v| {
                Constant::Int32Tensor(v.into())
            }),
            constant_literal_helper("Int64Tensor", vector2(parse_int), |v| {
                Constant::Int64Tensor(v.into())
            }),
            constant_literal_helper("Uint8Tensor", vector2(parse_int), |v| {
                Constant::Uint8Tensor(v.into())
            }),
            constant_literal_helper("Uint16Tensor", vector2(parse_int), |v| {
                Constant::Uint16Tensor(v.into())
            }),
            constant_literal_helper("Uint32Tensor", vector2(parse_int), |v| {
                Constant::Uint32Tensor(v.into())
            }),
            constant_literal_helper("Uint64Tensor", vector2(parse_int), |v| {
                Constant::Uint64Tensor(v.into())
            }),
            constant_literal_helper("Float32Tensor", vector2(float), |v| {
                Constant::Float32Tensor(v.into())
            }),
            constant_literal_helper("Float64Tensor", vector2(double), |v| {
                Constant::Float64Tensor(v.into())
            }),
            constant_literal_helper(
                "Ring64Tensor",
                vector2(parse_int),
                |v: ndarray::ArrayD<u64>| Constant::Ring64Tensor(v.into()),
            ),
            constant_literal_helper(
                "Ring128Tensor",
                vector2(parse_int),
                |v: ndarray::ArrayD<u128>| Constant::Ring128Tensor(v.into()),
            ),
        )),
    ))(input)
}

/// Parses a literal for a constant (not a placed value).
fn value_literal<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Value, E> {
    let (input, (v, p)) = tuple((constant_literal, ws(parse_placement)))(input)?;
    match p {
        Placement::Host(h) => Ok((input, v.place(&h))),
        _ => unimplemented!(), // TODO (lvorona) return parsing error that we do not support other placements in the textual form
    }
}

/// Parses a vector of items, using the supplied innter parser.
fn vector<'a, F: 'a, O, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, Vec<O>, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(tag("["), separated_list0(ws(tag(",")), inner), tag("]"))
}

/// Parses a 2D vector of items, using the supplied innter parser.
fn vector2<'a, F: 'a, O: 'a, E: 'a>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, ndarray::ArrayD<O>, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E> + Copy,
    O: Clone,
    E: ParseError<&'a str> + ContextError<&'a str>,
{
    move |input: &'a str| {
        let (input, vec2) = vector(vector(inner))(input)?;
        let mut data = Vec::new();

        let ncols = vec2.first().map_or(0, |row| row.len());
        let mut nrows = 0;

        for row in &vec2 {
            data.extend_from_slice(row);
            nrows += 1;
        }

        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[nrows, ncols]), data)
            .map(|a| (input, a))
            .map_err(|_: ndarray::ShapeError| Error(make_error(input, ErrorKind::MapRes)))
    }
}

/// Parses integer (or anything implementing FromStr from decimal digits)
fn parse_int<'a, O: std::str::FromStr, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, O, E> {
    map_res(digit1, |s: &str| s.parse::<O>())(input)
        .map_err(|_: nom::Err<nom::error::Error<&str>>| Error(make_error(input, ErrorKind::MapRes)))
}

/// Parses a single byte, writte as two hex character.
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

/// Parse sa hux dump, without any separators.
///
/// Errors out if there is not enough data to fill an array of length N.
fn parse_hex<'a, E, const N: usize>(input: &'a str) -> IResult<&'a str, [u8; N], E>
where
    E: ParseError<&'a str>,
{
    let mut buf: [u8; N] = [0; N];
    let (rest, ()) = fill(parse_hex_u8, &mut buf)(input)?;
    Ok((rest, buf))
}

/// Wraps the innner parser in optional spaces.
///
/// From nom::recepies
fn ws<'a, F: 'a, O, E: ParseError<&'a str>>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: Fn(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(space0, inner, space0)
}

/// Parses an identifier
///
/// From nom::recepies
pub fn identifier<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, &'a str, E> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

/// Parses an escaped character: \n, \t, \r, \u{00AC}, etc.
///
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
///
/// From nom examples (MIT licesnse, so it is ok)
fn parse_unicode<'a, E>(input: &'a str) -> IResult<&'a str, char, E>
where
    E: ParseError<&'a str>,
{
    map_opt(parse_hex_u32, std::char::from_u32)(input)
}

/// Parses any supported escaped character
///
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
///
/// From nom examples (MIT licesnse, so it is ok)
fn parse_escaped_whitespace<'a, E: ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, &'a str, E> {
    preceded(char('\\'), multispace1)(input)
}

/// Parse a non-empty block of text that doesn't include \ or "
///
/// From nom examples (MIT licesnse, so it is ok)
fn parse_literal<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, &'a str, E> {
    let not_quote_slash = is_not("\"\\");
    verify(not_quote_slash, |s: &str| !s.is_empty())(input)
}

/// A string fragment contains a fragment of a string being parsed: either
/// a non-empty Literal (a series of non-escaped characters), a single
/// parsed escaped character, or a block of escaped whitespace.
///
/// From nom examples (MIT licesnse, so it is ok)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StringFragment<'a> {
    Literal(&'a str),
    EscapedChar(char),
    EscapedWS,
}

/// Combine parse_literal, parse_escaped_whitespace, and parse_escaped_char
/// into a StringFragment.
///
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
///
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

/// A very simple boolean parser.
///
/// Only accepts literals `true` and `false`.
fn parse_bool<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, bool, E> {
    alt((value(true, tag("true")), value(false, tag("false"))))(input)
}

/// A serializer to produce the same textual format from a computation
pub trait ToTextual {
    fn to_textual(&self) -> String;
}

impl ToTextual for Computation {
    fn to_textual(&self) -> String {
        itertools::join(self.operations.iter().map(|op| op.to_textual()), "\n")
    }
}

impl ToTextual for Operation {
    fn to_textual(&self) -> String {
        format!(
            "{} = {} ({}) {}",
            self.name,
            self.kind.to_textual(),
            self.inputs.join(", "),
            self.placement.to_textual(),
        )
    }
}

impl ToTextual for Placement {
    fn to_textual(&self) -> String {
        match self {
            Placement::Host(p) => p.to_textual(),
            Placement::Replicated(p) => p.to_textual(),
            Placement::Additive(p) => p.to_textual(),
        }
    }
}

impl ToTextual for HostPlacement {
    fn to_textual(&self) -> String {
        format!("@Host({})", self.owner)
    }
}

impl ToTextual for ReplicatedPlacement {
    fn to_textual(&self) -> String {
        format!(
            "@Replicated({}, {}, {})",
            self.owners[0], self.owners[1], self.owners[2]
        )
    }
}

impl ToTextual for AdditivePlacement {
    fn to_textual(&self) -> String {
        format!("@Additive({}, {})", self.owners[0], self.owners[1])
    }
}

impl ToTextual for Operator {
    fn to_textual(&self) -> String {
        use Operator::*;
        match self {
            Identity(op) => op.to_textual(),
            Load(op) => op.to_textual(),
            Save(op) => op.to_textual(),
            Send(op) => op.to_textual(),
            Receive(op) => op.to_textual(),
            Input(op) => op.to_textual(),
            Output(op) => op.to_textual(),
            Constant(op) => op.to_textual(),
            Shape(op) => op.to_textual(),
            BitFill(op) => op.to_textual(),
            RingFill(op) => op.to_textual(),
            StdAdd(op) => op.to_textual(),
            StdSub(op) => op.to_textual(),
            StdMul(op) => op.to_textual(),
            StdDiv(op) => op.to_textual(),
            StdDot(op) => op.to_textual(),
            StdMean(op) => op.to_textual(),
            StdOnes(op) => op.to_textual(),
            StdConcatenate(op) => op.to_textual(),
            StdExpandDims(op) => op.to_textual(),
            StdReshape(op) => op.to_textual(),
            StdAtLeast2D(op) => op.to_textual(),
            StdSlice(op) => op.to_textual(),
            StdSum(op) => op.to_textual(),
            StdTranspose(op) => op.to_textual(),
            StdInverse(op) => op.to_textual(),
            RingNeg(op) => op.to_textual(),
            RingAdd(op) => op.to_textual(),
            RingSub(op) => op.to_textual(),
            RingMul(op) => op.to_textual(),
            RingDot(op) => op.to_textual(),
            RingMean(op) => op.to_textual(),
            RingSum(op) => op.to_textual(),
            RingSample(op) => op.to_textual(),
            RingShl(op) => op.to_textual(),
            RingShr(op) => op.to_textual(),
            RingInject(op) => op.to_textual(),
            BitExtract(op) => op.to_textual(),
            BitSample(op) => op.to_textual(),
            // BitXor(op) => op.to_textual(),
            // BitAnd(op) => op.to_textual(),
            PrimDeriveSeed(op) => op.to_textual(),
            PrimPrfKeyGen(op) => op.to_textual(),
            FixedpointRingEncode(op) => op.to_textual(),
            FixedpointRingDecode(op) => op.to_textual(),
            FixedpointRingMean(op) => op.to_textual(),
            RepSetup(op) => op.to_textual(),
            RepShare(op) => op.to_textual(),
            RepReveal(op) => op.to_textual(),
            RepDot(op) => op.to_textual(),
            RepMean(op) => op.to_textual(),
            RepSum(op) => op.to_textual(),
            RepAdd(op) => op.to_textual(),
            RepSub(op) => op.to_textual(),
            RepMul(op) => op.to_textual(),
            RepTruncPr(op) => op.to_textual(),
            _ => unimplemented!("{:?}", self),
        }
    }
}

macro_rules! standard_op_to_textual {
    ($op:ty, $format:expr, $($member:tt),* ) => {
        impl ToTextual for $op {
            fn to_textual(&self) -> String {
                format!(
                    $format,
                    $(self.$member.to_textual(),)*
                    op = self.short_name(),
                )
            }
        }
    };
}

standard_op_to_textual!(ConstantOp, "{op}{{value = {}}}", value);
standard_op_to_textual!(IdentityOp, "{op}: {}", sig);
standard_op_to_textual!(LoadOp, "{op}: {}", sig);
standard_op_to_textual!(SaveOp, "{op}: {}", sig);
standard_op_to_textual!(
    SendOp,
    "{op} {{rendezvous_key={}, receiver={}}}: {}",
    rendezvous_key,
    receiver,
    sig
);
standard_op_to_textual!(
    ReceiveOp,
    "{op} {{rendezvous_key={}, sender={}}} : {}",
    rendezvous_key,
    sender,
    sig
);
standard_op_to_textual!(InputOp, "{op} {{arg_name={}}}: {}", arg_name, sig);
standard_op_to_textual!(OutputOp, "{op}: {}", sig);
standard_op_to_textual!(StdAddOp, "{op}: {}", sig);
standard_op_to_textual!(StdSubOp, "{op}: {}", sig);
standard_op_to_textual!(StdMulOp, "{op}: {}", sig);
standard_op_to_textual!(StdDivOp, "{op}: {}", sig);
standard_op_to_textual!(StdDotOp, "{op}: {}", sig);
standard_op_to_textual!(StdOnesOp, "{op}: {}", sig);
standard_op_to_textual!(StdConcatenateOp, "{op}{{axis={}}}: {}", axis, sig);
standard_op_to_textual!(StdExpandDimsOp, "{op}{{axis={}}}: {}", axis, sig);
standard_op_to_textual!(StdReshapeOp, "{op}: {}", sig);
standard_op_to_textual!(BitFillOp, "{op}{{value={}}}: {}", value, sig);
standard_op_to_textual!(RingFillOp, "{op}{{value={}}}: {}", value, sig);
standard_op_to_textual!(
    StdAtLeast2DOp,
    "{op}{{to_column_vector={}}}: {}",
    to_column_vector,
    sig
);
standard_op_to_textual!(StdSliceOp, "{op}{{start={}, end={}}}: {}", start, end, sig);
standard_op_to_textual!(StdTransposeOp, "{op}: {}", sig);
standard_op_to_textual!(StdInverseOp, "{op}: {}", sig);
standard_op_to_textual!(ShapeOp, "{op}: {}", sig);
standard_op_to_textual!(RingNegOp, "{op}: {}", sig);
standard_op_to_textual!(RingAddOp, "{op}: {}", sig);
standard_op_to_textual!(RingSubOp, "{op}: {}", sig);
standard_op_to_textual!(RingMulOp, "{op}: {}", sig);
standard_op_to_textual!(RingDotOp, "{op}: {}", sig);
standard_op_to_textual!(RingShlOp, "{op}{{amount={}}}: {}", amount, sig);
standard_op_to_textual!(RingShrOp, "{op}{{amount={}}}: {}", amount, sig);
standard_op_to_textual!(RingInjectOp, "{op}{{bit_idx={}}}: {}", bit_idx, sig);
standard_op_to_textual!(BitExtractOp, "{op}{{bit_idx={}}}: {}", bit_idx, sig);
standard_op_to_textual!(BitSampleOp, "{op}: {}", sig);
standard_op_to_textual!(PrimDeriveSeedOp, "{op}{{sync_key={}}}: {}", sync_key, sig);
standard_op_to_textual!(PrimPrfKeyGenOp, "{op}: {}", sig);
standard_op_to_textual!(
    FixedpointRingEncodeOp,
    "{op}{{scaling_base={}, scaling_exp={}}}: {}",
    scaling_base,
    scaling_exp,
    sig
);
standard_op_to_textual!(
    FixedpointRingDecodeOp,
    "{op}{{scaling_base={}, scaling_exp={}}}: {}",
    scaling_base,
    scaling_exp,
    sig
);
standard_op_to_textual!(RepSetupOp, "{op}: {}", sig);
standard_op_to_textual!(RepShareOp, "{op}: {}", sig);
standard_op_to_textual!(RepRevealOp, "{op}: {}", sig);
standard_op_to_textual!(RepDotOp, "{op}: {}", sig);
standard_op_to_textual!(RepAddOp, "{op}: {}", sig);
standard_op_to_textual!(RepSubOp, "{op}: {}", sig);
standard_op_to_textual!(RepMulOp, "{op}: {}", sig);
standard_op_to_textual!(RepTruncPrOp, "{op}{{amount={}}}: {}", amount, sig);

macro_rules! op_with_axis_to_textual {
    ($op:tt) => {
        impl ToTextual for $op {
            fn to_textual(&self) -> String {
                match self {
                    $op { sig, axis: Some(a) } => {
                        format!(
                            "{}{{axis = {}}}: {}",
                            self.short_name(),
                            a,
                            sig.to_textual()
                        )
                    }
                    $op { sig, axis: None } => {
                        format!("{}: {}", self.short_name(), sig.to_textual())
                    }
                }
            }
        }
    };
}

op_with_axis_to_textual!(StdMeanOp);
op_with_axis_to_textual!(StdSumOp);
op_with_axis_to_textual!(RingMeanOp);
op_with_axis_to_textual!(RingSumOp);
op_with_axis_to_textual!(RepMeanOp);
op_with_axis_to_textual!(RepSumOp);

impl ToTextual for FixedpointRingMeanOp {
    fn to_textual(&self) -> String {
        match self {
            FixedpointRingMeanOp {
                sig,
                axis: Some(a),
                scaling_base,
                scaling_exp,
            } => {
                format!(
                    "FixedpointRingMean{{axis = {}, scaling_base={}, scaling_exp={}}}: {}",
                    a,
                    scaling_base,
                    scaling_exp,
                    sig.to_textual()
                )
            }
            FixedpointRingMeanOp {
                sig,
                axis: None,
                scaling_base,
                scaling_exp,
            } => format!(
                "FixedpointRingMean{{scaling_base={}, scaling_exp={}}}: {}",
                scaling_base,
                scaling_exp,
                sig.to_textual()
            ),
        }
    }
}

impl ToTextual for RingSampleOp {
    fn to_textual(&self) -> String {
        match self {
            RingSampleOp {
                sig,
                max_value: Some(a),
            } => format!("RingSample{{max_value = {}}}: {}", a, sig.to_textual()),
            RingSampleOp {
                sig,
                max_value: None,
            } => format!("RingSample: {}", sig.to_textual()),
        }
    }
}

impl ToTextual for Ty {
    fn to_textual(&self) -> String {
        match self {
            Ty::Unit => "Unit",
            Ty::String => "String",
            Ty::Float32 => "Float32",
            Ty::Float64 => "Float64",
            Ty::Ring64 => "Ring64",
            Ty::Ring128 => "Ring128",
            Ty::Ring64Tensor => "Ring64Tensor",
            Ty::Ring128Tensor => "Ring128Tensor",
            Ty::Bit => "Bit",
            Ty::BitTensor => "BitTensor",
            Ty::Shape => "Shape",
            Ty::Seed => "Seed",
            Ty::PrfKey => "PrfKey",
            Ty::Nonce => "Nonce",
            Ty::Float32Tensor => "Float32Tensor",
            Ty::Float64Tensor => "Float64Tensor",
            Ty::Int8Tensor => "Int8Tensor",
            Ty::Int16Tensor => "Int16Tensor",
            Ty::Int32Tensor => "Int32Tensor",
            Ty::Int64Tensor => "Int64Tensor",
            Ty::Uint8Tensor => "Uint8Tensor",
            Ty::Uint16Tensor => "Uint16Tensor",
            Ty::Uint32Tensor => "Uint32Tensor",
            Ty::Uint64Tensor => "Uint64Tensor",
            Ty::Unknown => "Unknown",
            Ty::Fixed64Tensor => "Fixed64Tensor",
            Ty::Fixed128Tensor => "Fixed128Tensor",
            Ty::Replicated64Tensor => "Replicated64Tensor",
            Ty::Replicated128Tensor => "Replicated128Tensor",
            Ty::ReplicatedBitTensor => "ReplicatedBitTensor",
            Ty::ReplicatedSetup => "ReplicatedSetup",
            Ty::Additive64Tensor => "Additive64Tensor",
            Ty::Additive128Tensor => "Additive128Tensor",
        }
        .to_string()
    }
}

// TODO: This will need to be either removed or output the owner as well (lvorona)
impl ToTextual for Value {
    fn to_textual(&self) -> String {
        match self {
            Value::Int8Tensor(x) => format!("Int8Tensor({})", x.0.to_textual()),
            Value::Int16Tensor(x) => format!("Int16Tensor({})", x.0.to_textual()),
            Value::Int32Tensor(x) => format!("Int32Tensor({})", x.0.to_textual()),
            Value::Int64Tensor(x) => format!("Int64Tensor({})", x.0.to_textual()),
            Value::Uint8Tensor(x) => format!("Uint8Tensor({})", x.0.to_textual()),
            Value::Uint16Tensor(x) => format!("Uint16Tensor({})", x.0.to_textual()),
            Value::Uint32Tensor(x) => format!("Uint32Tensor({})", x.0.to_textual()),
            Value::Uint64Tensor(x) => format!("Uint64Tensor({})", x.0.to_textual()),
            Value::Float32Tensor(x) => format!("Float32Tensor({})", x.0.to_textual()),
            Value::Float64Tensor(x) => format!("Float64Tensor({})", x.0.to_textual()),
            Value::Ring64Tensor(x) => format!("Ring64Tensor({})", x.0.to_textual()),
            Value::Ring128Tensor(x) => format!("Ring128Tensor({})", x.0.to_textual()),
            Value::Float32(x) => format!("Float32({})", x),
            Value::Float64(x) => format!("Float64({})", x),
            Value::String(x) => format!("String({})", x.to_textual()),
            Value::Ring64(x) => format!("Ring64({})", x),
            Value::Ring128(x) => format!("Ring128({})", x),
            Value::Shape(Shape(x, _)) => format!("Shape({:?})", x),
            // Value::Nonce(Nonce(x)) => format!("Nonce({:?})", x),
            // Value::Seed(Seed(x)) => format!("Seed({})", x.to_textual()),
            // Value::PrfKey(PrfKey(x)) => format!("PrfKey({})", x.to_textual()),
            _ => unimplemented!(), // TODO
        }
    }
}

impl ToTextual for Constant {
    fn to_textual(&self) -> String {
        match self {
            Constant::Int8Tensor(x) => format!("Int8Tensor({})", x.0.to_textual()),
            Constant::Int16Tensor(x) => format!("Int16Tensor({})", x.0.to_textual()),
            Constant::Int32Tensor(x) => format!("Int32Tensor({})", x.0.to_textual()),
            Constant::Int64Tensor(x) => format!("Int64Tensor({})", x.0.to_textual()),
            Constant::Uint8Tensor(x) => format!("Uint8Tensor({})", x.0.to_textual()),
            Constant::Uint16Tensor(x) => format!("Uint16Tensor({})", x.0.to_textual()),
            Constant::Uint32Tensor(x) => format!("Uint32Tensor({})", x.0.to_textual()),
            Constant::Uint64Tensor(x) => format!("Uint64Tensor({})", x.0.to_textual()),
            Constant::Float32Tensor(x) => format!("Float32Tensor({})", x.0.to_textual()),
            Constant::Float64Tensor(x) => format!("Float64Tensor({})", x.0.to_textual()),
            Constant::Ring64Tensor(x) => format!("Ring64Tensor({})", x.0.to_textual()),
            Constant::Ring128Tensor(x) => format!("Ring128Tensor({})", x.0.to_textual()),
            Constant::Float32(x) => format!("Float32({})", x),
            Constant::Float64(x) => format!("Float64({})", x),
            Constant::String(x) => format!("String({})", x.to_textual()),
            Constant::Ring64(x) => format!("Ring64({})", x),
            Constant::Ring128(x) => format!("Ring128({})", x),
            Constant::RawShape(RawShape(x)) => format!("Shape({:?})", x),
            Constant::RawNonce(RawNonce(x)) => format!("Nonce({:?})", x),
            Constant::RawSeed(RawSeed(x)) => format!("Seed({})", x.to_textual()),
            Constant::RawPrfKey(RawPrfKey(x)) => format!("PrfKey({})", x.to_textual()),
            _ => unimplemented!(), // TODO
        }
    }
}

impl<T: std::fmt::Debug> ToTextual for ndarray::ArrayD<T> {
    fn to_textual(&self) -> String {
        match self.shape() {
            [_len] => format!("{:?}", self.as_slice().unwrap()),
            [cols, rows] => {
                let mut buffer = String::from("[");
                let mut first_row = true;
                for r in 0..*rows {
                    if !first_row {
                        buffer.push_str(", ");
                    }
                    let mut first_col = true;
                    buffer.push('[');
                    for c in 0..*cols {
                        if !first_col {
                            buffer.push_str(", ");
                        }
                        buffer += &format!("{:?}", self[[r, c]]);
                        first_col = false;
                    }
                    buffer.push(']');
                    first_row = false;
                }
                buffer.push(']');
                buffer
            }
            _ => unimplemented!(),
        }
    }
}

impl ToTextual for Role {
    fn to_textual(&self) -> String {
        format!("{:?}", self.0)
    }
}

// TODO: lvorona revisit this - this should not require a special ToTextual
impl ToTextual for RawNonce {
    fn to_textual(&self) -> String {
        format!("{:?}", self.0)
    }
}

impl ToTextual for Signature {
    fn to_textual(&self) -> String {
        match self {
            Signature::Nullary(NullarySignature { ret }) => format!("() -> {}", ret.to_textual()),
            Signature::Unary(UnarySignature { arg0, ret }) => {
                format!("({}) -> {}", arg0.to_textual(), ret.to_textual())
            }
            Signature::Binary(BinarySignature { arg0, arg1, ret }) => format!(
                "({}, {}) -> {}",
                arg0.to_textual(),
                arg1.to_textual(),
                ret.to_textual()
            ),
            Signature::Ternary(TernarySignature {
                arg0,
                arg1,
                arg2,
                ret,
            }) => format!(
                "({}, {}, {}) -> {}",
                arg0.to_textual(),
                arg1.to_textual(),
                arg2.to_textual(),
                ret.to_textual()
            ),
        }
    }
}

macro_rules! use_debug_to_textual {
    ($op:ty) => {
        impl ToTextual for $op {
            fn to_textual(&self) -> String {
                format!("{:?}", self)
            }
        }
    };
}

use_debug_to_textual!(String);
use_debug_to_textual!(usize);
use_debug_to_textual!(u32);
use_debug_to_textual!(u64);
use_debug_to_textual!(bool);

impl ToTextual for [u8] {
    fn to_textual(&self) -> String {
        let mut s = String::new();
        for &byte in self {
            s.push_str(&format!("{:02x}", byte));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_literal() -> Result<(), anyhow::Error> {
        let (_, parsed_f32) = constant_literal::<(&str, ErrorKind)>("Float32(1.23)")?;
        assert_eq!(parsed_f32, Constant::Float32(1.23));
        let (_, parsed_f64) = constant_literal::<(&str, ErrorKind)>("Float64(1.23)")?;
        assert_eq!(parsed_f64, Constant::Float64(1.23));
        let (_, parsed_str) = constant_literal::<(&str, ErrorKind)>("\"abc\"")?;
        assert_eq!(parsed_str, Constant::String("abc".into()));
        let (_, parsed_str) = constant_literal::<(&str, ErrorKind)>("String(\"abc\")")?;
        assert_eq!(parsed_str, Constant::String("abc".into()));
        let (_, parsed_str) = constant_literal::<(&str, ErrorKind)>("\"1.23\"")?;
        assert_eq!(parsed_str, Constant::String("1.23".into()));
        let (_, parsed_str) = constant_literal::<(&str, ErrorKind)>("\"1. 2\\\"3\"")?;
        assert_eq!(parsed_str, Constant::String("1. 2\"3".into()));
        let (_, parsed_ring64_tensor) =
            constant_literal::<(&str, ErrorKind)>("Ring64Tensor([1,2,3])")?;
        assert_eq!(
            parsed_ring64_tensor,
            Constant::Ring64Tensor(vec![1, 2, 3].into())
        );
        let (_, parsed_ring128_tensor) =
            constant_literal::<(&str, ErrorKind)>("Ring128Tensor([1,2,3])")?;
        assert_eq!(
            parsed_ring128_tensor,
            Constant::Ring128Tensor(vec![1, 2, 3].into())
        );
        let (_, parsed_shape) = constant_literal::<(&str, ErrorKind)>("Shape([1,2,3])")?;
        assert_eq!(parsed_shape, Constant::RawShape(RawShape(vec![1, 2, 3])));
        let (_, parsed_u8_tensor) = constant_literal::<(&str, ErrorKind)>("Uint8Tensor([1,2,3])")?;
        assert_eq!(
            parsed_u8_tensor,
            Constant::Uint8Tensor(vec![1, 2, 3].into())
        );
        let (_, parsed_seed) =
            constant_literal::<(&str, ErrorKind)>("Seed(529c2fc9bf573d077f45f42b19cfb8d4)")?;
        assert_eq!(
            parsed_seed,
            Constant::RawSeed(RawSeed([
                0x52, 0x9c, 0x2f, 0xc9, 0xbf, 0x57, 0x3d, 0x07, 0x7f, 0x45, 0xf4, 0x2b, 0x19, 0xcf,
                0xb8, 0xd4
            ]))
        );
        let (_, parsed_ring64) = constant_literal::<(&str, ErrorKind)>("Ring64(42)")?;
        assert_eq!(parsed_ring64, Constant::Ring64(42));

        Ok(())
    }

    #[test]
    fn test_array_literal() -> Result<(), anyhow::Error> {
        use ndarray::prelude::*;
        use std::convert::TryInto;
        let parsed_f32: Constant = "Float32Tensor([[1.0, 2.0], [3.0, 4.0]])".try_into()?;

        let x = crate::standard::Float32Tensor::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );

        assert_eq!(parsed_f32, Constant::Float32Tensor(x));

        let parsed_ring64: Constant = "Ring64Tensor([[1, 2], [3, 4]])".try_into()?;

        let x_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = crate::ring::Ring64Tensor::from(x_backing);

        assert_eq!(parsed_ring64, Constant::Ring64Tensor(x));

        Ok(())
    }

    #[test]
    fn test_type_parsing() -> Result<(), anyhow::Error> {
        let (_, parsed_type) = parse_type::<(&str, ErrorKind)>("Unit")?;
        assert_eq!(parsed_type, Ty::Unit);
        let (_, parsed) = type_definition::<(&str, ErrorKind)>(0)(
            ": (Float32Tensor, Float64Tensor) -> Uint16Tensor",
        )?;
        assert_eq!(
            parsed,
            Signature::binary(Ty::Float32Tensor, Ty::Float64Tensor, Ty::Uint16Tensor),
        );

        let parsed: IResult<_, _, VerboseError<&str>> = parse_type("blah");
        if let Err(Error(e)) = parsed {
            assert_eq!(
                convert_error("blah", e),
                "0: at line 1, in Tag:\nblah\n^\n\n"
            );
        } else {
            panic!("Type parsing should have given an error on an invalid type, but did not");
        }
        Ok(())
    }

    #[test]
    fn test_constant() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "x = Constant{value = Float32Tensor([1.0])}: () -> Float32Tensor () @Host(alice)",
        )?;
        assert_eq!(op.name, "x");
        assert_eq!(
            op.kind,
            Operator::Constant(ConstantOp {
                sig: Signature::nullary(Ty::Float32Tensor),
                value: Constant::Float32Tensor(vec![1.0].into())
            })
        );

        // 2D tensor
        use ndarray::prelude::*;
        let x = crate::standard::Float32Tensor::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "x = Constant{value = Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor () @Replicated(alice, bob, charlie)",
        )?;
        assert_eq!(
            op.kind,
            Operator::Constant(ConstantOp {
                sig: Signature::nullary(Ty::Float32Tensor),
                value: Constant::Float32Tensor(x)
            })
        );
        Ok(())
    }

    #[test]
    fn test_stdbinary() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z = StdAdd: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(carole)",
        )?;
        assert_eq!(op.name, "z");
        assert_eq!(
            op.kind,
            Operator::StdAdd(StdAddOp {
                sig: Signature::binary(Ty::Float32Tensor, Ty::Float32Tensor, Ty::Float32Tensor),
            })
        );
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z = StdMul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(carole)",
        )?;
        assert_eq!(op.name, "z");
        assert_eq!(
            op.kind,
            Operator::StdMul(StdMulOp {
                sig: Signature::binary(Ty::Float32Tensor, Ty::Float32Tensor, Ty::Float32Tensor),
            })
        );
        Ok(())
    }

    #[test]
    fn test_stdadd_err() {
        let data = "z = StdAdd: (Float32Tensor) -> Float32Tensor (x, y) @Host(carole)";
        let parsed: IResult<_, _, VerboseError<&str>> = parse_assignment(data);
        if let Err(Failure(e)) = parsed {
            assert_eq!(convert_error(data, e), "0: at line 1, in Verify:\nz = StdAdd: (Float32Tensor) -> Float32Tensor (x, y) @Host(carole)\n            ^\n\n");
        } else {
            panic!("Type parsing should have given an error on an invalid type, but did not");
        }
    }

    #[test]
    fn test_primprfkeygen() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>("key = PrimPrfKeyGen() @Host(alice)")?;
        assert_eq!(op.name, "key");
        Ok(())
    }

    #[test]
    fn test_seed() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "seed = PrimDeriveSeed{sync_key = [1, 2, 3]}(key)@Host(alice)",
        )?;
        assert_eq!(op.name, "seed");
        assert_eq!(
            op.kind,
            Operator::PrimDeriveSeed(PrimDeriveSeedOp {
                sig: Signature::nullary(Ty::Seed),
                sync_key: RawNonce(vec![1, 2, 3])
            })
        );
        Ok(())
    }

    #[test]
    fn test_send() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            r#"send = Send{rendezvous_key = "abc" receiver = "bob"}() @Host(alice)"#,
        )?;
        assert_eq!(op.name, "send");
        assert_eq!(
            op.kind,
            Operator::Send(SendOp {
                sig: Signature::unary(Ty::Unknown, Ty::Unknown),
                rendezvous_key: "abc".into(),
                receiver: Role::from("bob")
            })
        );
        Ok(())
    }

    #[test]
    fn test_receive() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            r#"receive = Receive{rendezvous_key = "abc", sender = "bob"} : () -> Float32Tensor () @Host(alice)"#,
        )?;
        assert_eq!(op.name, "receive");
        assert_eq!(
            op.kind,
            Operator::Receive(ReceiveOp {
                sig: Signature::nullary(Ty::Float32Tensor),
                rendezvous_key: "abc".into(),
                sender: Role::from("bob"),
            })
        );
        Ok(())
    }

    #[test]
    fn test_output() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z = Output: (Ring64Tensor) -> Ring64Tensor (x10) @Host(alice)",
        )?;
        assert_eq!(op.name, "z");
        Ok(())
    }

    #[test]
    fn test_ring_sample() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "x10 = RingSample{max_value = 1}: (Shape, Seed) -> Ring64Tensor (shape, seed) @Host(alice)",
        )?;
        assert_eq!(op.name, "x10");
        Ok(())
    }

    #[test]
    fn test_fixedpoint_ring_mean() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "op = FixedpointRingMean{scaling_base = 3, scaling_exp = 1, axis = 0} : () -> Float32Tensor () @Host(alice)",
        )?;
        assert_eq!(
            op.kind,
            Operator::FixedpointRingMean(FixedpointRingMeanOp {
                sig: Signature::nullary(Ty::Float32Tensor),
                axis: Some(0),
                scaling_base: 3,
                scaling_exp: 1,
            })
        );

        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "op = FixedpointRingMean{scaling_base = 3, scaling_exp = 1} : () -> Float32Tensor () @Host(alice)",
        )?;
        assert_eq!(
            op.kind,
            Operator::FixedpointRingMean(FixedpointRingMeanOp {
                sig: Signature::nullary(Ty::Float32Tensor),
                axis: None,
                scaling_base: 3,
                scaling_exp: 1,
            })
        );

        Ok(())
    }

    #[test]
    fn test_underscore() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "x_shape = Constant{value = Shape([2, 2])} () @Host(alice)",
        )?;
        assert_eq!(op.name, "x_shape");
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z_result = StdAdd: (Float32Tensor, Float32Tensor) -> Float32Tensor (x_shape, y_shape) @Host(carole)",
        )?;
        assert_eq!(op.name, "z_result");
        assert_eq!(op.inputs, vec!["x_shape", "y_shape"]);
        Ok(())
    }

    #[test]
    fn test_various() -> Result<(), anyhow::Error> {
        // The following tests are verifying that each valid line is parsed successfuly.
        // It does not assert on the result.
        parse_assignment::<(&str, ErrorKind)>(
            r#"z = Input{arg_name = "prompt"}: () -> Float32Tensor () @Host(alice)"#,
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = StdExpandDims {axis = 0}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = StdAtLeast2D {to_column_vector = false}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = StdSlice {start = 1, end = 2}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = RingSum {axis = 0}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = RingFill {value = Ring64(42)}: (Shape) -> Ring64Tensor (s) @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = RingShl {amount = 2}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = RingShr {amount = 2}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = FixedpointRingDecode {scaling_base = 3, scaling_exp = 2}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = FixedpointRingEncode {scaling_base = 3, scaling_exp = 2}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = RingInject {bit_idx = 2} : (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = BitExtract {bit_idx = 2} : (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>("z = BitSample() @Host(alice)")?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = BitFill {value = Ring64(0)}: (Shape) -> BitTensor (s) @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>("z = BitXor() @Host(alice)")?;

        parse_assignment::<(&str, ErrorKind)>(
            "load = Load: (String, String) -> Float64Tensor (xuri, xconstant) @Host(alice)",
        )?;

        Ok(())
    }

    #[test]
    fn test_sample_computation() -> Result<(), anyhow::Error> {
        let (_, comp) = parse_computation::<(&str, ErrorKind)>(
            "x = Constant{value = Float32Tensor([1.0])}() @Host(alice)
            y = Constant{value = Float32Tensor([2.0])}: () -> Float32Tensor () @Host(bob)
            // ignore = Constant([1.0]: Float32Tensor) @Host(alice)
            z = StdAdd: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(carole)
            ",
        )?;
        assert_eq!(comp.operations.len(), 3);
        assert_eq!(
            comp.operations[0].kind,
            Operator::Constant(ConstantOp {
                sig: Signature::nullary(Ty::Float32Tensor),
                value: Constant::Float32Tensor(vec![1.0].into())
            })
        );
        assert_eq!(
            comp.operations[1].kind,
            Operator::Constant(ConstantOp {
                sig: Signature::nullary(Ty::Float32Tensor),
                value: Constant::Float32Tensor(vec![2.0].into())
            })
        );
        assert_eq!(comp.operations[2].name, "z");
        assert_eq!(
            comp.operations[2].kind,
            Operator::StdAdd(StdAddOp {
                sig: Signature::binary(Ty::Float32Tensor, Ty::Float32Tensor, Ty::Float32Tensor),
            })
        );
        assert_eq!(comp.operations[2].inputs, vec!("x", "y"));
        assert_eq!(
            comp.operations[2].placement,
            Placement::Host(HostPlacement {
                owner: Role::from("carole"),
            })
        );
        Ok(())
    }

    #[test]
    fn test_sample_computation_err() {
        let data = r#"a = Constant{value = "a"} () @Host(alice)
            err = StdAdd: (Float32Tensor) -> Float32Tensor (x, y) @Host(carole)
            b = Constant{value = "b"} () @Host(alice)"#;
        let parsed: IResult<_, _, VerboseError<&str>> = parse_computation(data);
        if let Err(Failure(e)) = parsed {
            assert_eq!(convert_error(data, e), "0: at line 2, in Verify:\n            err = StdAdd: (Float32Tensor) -> Float32Tensor (x, y) @Host(carole)\n                          ^\n\n");
        }
    }

    #[test]
    fn test_computation_try_into() -> Result<(), anyhow::Error> {
        use std::convert::TryInto;
        let comp: Computation = "x = Constant{value = Float32Tensor([1.0])} @Host(alice)
            y = Constant{value = Float32Tensor([2.0])}: () -> Float32Tensor () @Host(bob)
            z = StdAdd: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(carole)"
            .try_into()?;
        assert_eq!(comp.operations.len(), 3);
        Ok(())
    }

    #[test]
    fn test_value_try_into() -> Result<(), anyhow::Error> {
        use std::convert::TryInto;
        let v: Constant = "Float32Tensor([1.0, 2.0, 3.0])".try_into()?;
        assert_eq!(v, Constant::Float32Tensor(vec![1.0, 2.0, 3.0].into()));
        Ok(())
    }

    #[test]
    fn test_whitespace() -> Result<(), anyhow::Error> {
        use std::convert::TryInto;
        let source = r#"
        x = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])} @Host(alice)

        y = Constant {value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])} @Host(bob)

        "#;
        let comp: Computation = source.try_into()?;
        assert_eq!(comp.operations.len(), 2);
        Ok(())
    }

    #[test]
    fn test_computation_into_text() -> Result<(), anyhow::Error> {
        use std::convert::TryInto;
        let comp: Computation = "x = Constant{value = Float32Tensor([1.0])} @Host(alice)
            y = Constant{value = Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(bob)
            z = StdAdd: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Replicated(alice, bob, carole)
            seed = PrimDeriveSeed{sync_key = [1, 2, 3]} (key) @Host(alice)
            seed2 = Constant{value = Seed(529c2fc9bf573d077f45f42b19cfb8d4)} @Host(alice)
            o = Output: (Float32Tensor) -> Float32Tensor (z) @Host(alice)"
            .try_into()?;
        let textual = comp.to_textual();
        // After serializing it into the textual IR we need to make sure it parses back the same
        let comp2: Computation = textual.try_into()?;
        assert_eq!(comp.operations[0], comp2.operations[0]);
        Ok(())
    }
}
