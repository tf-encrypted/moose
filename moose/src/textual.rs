//! Textual representation of computations

use crate::additive::AdditivePlacement;
use crate::computation::*;
use crate::host::{RawPrfKey, RawSeed, RawShape, SliceInfo, SliceInfoElem, SyncKey};
use crate::logical::TensorDType;
use crate::mirrored::Mirrored3Placement;
use crate::replicated::ReplicatedPlacement;
use crate::types::*;
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
            .map_err(|e| {
                anyhow::anyhow!("Failed to parse constant literal {} due to {}", source, e)
            })
    }
}

impl FromStr for Constant {
    type Err = anyhow::Error;
    fn from_str(source: &str) -> Result<Self, Self::Err> {
        constant_literal(source)
            .map(|(_, v)| v)
            .map_err(|e| friendly_error("Failed to parse constant literal", source, e))
    }
}

impl TryFrom<&str> for Value {
    type Error = anyhow::Error;

    fn try_from(source: &str) -> anyhow::Result<Value> {
        value_literal(source)
            .map(|(_, v)| v)
            .map_err(|e| friendly_error("Failed to parse value literal", source, e))
    }
}

impl FromStr for Value {
    type Err = anyhow::Error;
    fn from_str(source: &str) -> Result<Self, Self::Err> {
        value_literal(source)
            .map(|(_, v)| v)
            .map_err(|e| friendly_error("Failed to parse value literal", source, e))
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

pub trait FromTextual<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>> {
    fn from_textual(input: &'a str) -> IResult<&'a str, Operator, E>;
}

/// Parses operator - maps names to structs.
fn parse_operator<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    // NOTE: Ideally, we would group all of these parser declarations into a single `alt`
    // combinator. However, `alt` expects a tuple of parsers with cardinality at most 21.
    // We get around this by nesting calls of `alt`, as recommended by the function docs:
    // https://docs.rs/nom/7.0.0/nom/branch/fn.alt.html
    let part1 = alt((
        IdentityOp::from_textual,
        LoadOp::from_textual,
        SendOp::from_textual,
        ReceiveOp::from_textual,
        InputOp::from_textual,
        OutputOp::from_textual,
        ConstantOp::from_textual,
        ShapeOp::from_textual,
        RingFillOp::from_textual,
        SaveOp::from_textual,
        HostAddOp::from_textual,
        HostSubOp::from_textual,
        HostMulOp::from_textual,
        HostDivOp::from_textual,
        HostDotOp::from_textual,
        HostMeanOp::from_textual,
        preceded(tag(HostExpandDimsOp::SHORT_NAME), cut(hostexpanddims)),
        HostReshapeOp::from_textual,
        HostAtLeast2DOp::from_textual,
        HostSliceOp::from_textual,
    ));
    let part2 = alt((
        HostSumOp::from_textual,
        HostOnesOp::from_textual,
        ConcatOp::from_textual,
        HostTransposeOp::from_textual,
        HostInverseOp::from_textual,
        RingAddOp::from_textual,
        RingSubOp::from_textual,
        RingMulOp::from_textual,
        RingDotOp::from_textual,
        RingSampleSeededOp::from_textual,
        RingSampleOp::from_textual,
        RingShlOp::from_textual,
        RingShrOp::from_textual,
        preceded(tag(PrimDeriveSeedOp::SHORT_NAME), cut(prim_derive_seed)),
        PrimPrfKeyGenOp::from_textual,
        RingFixedpointEncodeOp::from_textual,
        RingFixedpointDecodeOp::from_textual,
        RingFixedpointMeanOp::from_textual,
        FixedpointEncodeOp::from_textual,
        FixedpointDecodeOp::from_textual,
    ));
    let part3 = alt((
        RingInjectOp::from_textual,
        BitExtractOp::from_textual,
        BitSampleSeededOp::from_textual,
        BitSampleOp::from_textual,
        BitXorOp::from_textual,
        BitAndOp::from_textual,
        HostSqrtOp::from_textual,
        HostDiagOp::from_textual,
        HostSqueezeOp::from_textual,
        AddNOp::from_textual,
        AddOp::from_textual,
        SubOp::from_textual,
        MulOp::from_textual,
        DivOp::from_textual,
        DotOp::from_textual,
        MeanOp::from_textual,
        RingNegOp::from_textual,
        HostShlDimOp::from_textual,
        HostBitDecOp::from_textual,
        FillOp::from_textual,
        IndexAxisOp::from_textual,
    ));
    let part4 = alt((
        DemirrorOp::from_textual,
        MirrorOp::from_textual,
        MaximumOp::from_textual,
        SoftmaxOp::from_textual,
        BroadcastOp::from_textual,
    ));
    alt((part1, part2, part3, part4))(input)
}

/// Parses a HostExpandDims operator
fn hostexpanddims<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, axis) = attributes_single("axis", vector(parse_int))(input)?;
    let (input, sig) = operator_signature(1)(input)?;
    Ok((input, HostExpandDimsOp { sig, axis }.into()))
}

/// Parses a PrimDeriveSeed operator.
fn prim_derive_seed<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, sync_key) =
        attributes_single("sync_key", map_res(vector(parse_int), SyncKey::try_from))(input)
            .map_err(|_: nom::Err<nom::error::Error<&str>>| {
                Error(make_error(input, ErrorKind::MapRes))
            })?;
    let (input, opt_sig) = opt(operator_signature(0))(input)?;
    let sig = opt_sig.unwrap_or_else(|| Signature::nullary(Ty::Seed));
    Ok((input, PrimDeriveSeedOp { sig, sync_key }.into()))
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
pub fn attributes_single<'a, O, F: 'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    name: &'a str,
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(ws(tag("{")), attributes_member(name, inner), ws(tag("}")))
}

/// Parses a single attribute with an optional comma at the end.
pub fn attributes_member<'a, O, F: 'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
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

/// Parses operator's type signature
///
/// Accepts input in the form of
///
/// `: (Float32Tensor, Float32Tensor) -> Float32Tensor`
/// `: ([Float32Tensor]) -> Float32Tensor`
///
/// * `arg_count` - the number of required arguments
pub fn operator_signature<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    arg_count: usize,
) -> impl FnMut(&'a str) -> IResult<&'a str, Signature, E> {
    preceded(
        ws(tag(":")),
        alt((fixed_arrity_signature(arg_count), variadic_signature())),
    )
}

/// Parses operator's type signature - fixed arity form
///
/// Accepts input in the form of
///
/// `(Float32Tensor, Float32Tensor) -> Float32Tensor`
///
/// * `arg_count` - the number of required arguments
fn fixed_arrity_signature<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    arg_count: usize,
) -> impl FnMut(&'a str) -> IResult<&'a str, Signature, E> {
    move |input: &'a str| {
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

/// Parses operator's type signature - variadic form
///
/// Accepts input in the form of
///
/// `[Float32Tensor] -> Float32Tensor`
fn variadic_signature<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
) -> impl FnMut(&'a str) -> IResult<&'a str, Signature, E> {
    move |input: &'a str| {
        let (input, (_, args_type, _)) =
            tuple((ws(tag("[")), ws(parse_type), ws(tag("]"))))(input)?;

        let (input, _) = ws(tag("->"))(input)?;
        let (input, result_type) = ws(parse_type)(input)?;

        Ok((input, Signature::variadic(args_type, result_type)))
    }
}

/// Parses an individual type's literal
fn parse_type<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Ty, E> {
    let (i, type_name) = alphanumeric1(input)?;
    match type_name {
        "Unknown" => Ok((i, Ty::Unknown)),
        "Shape" => Ok((i, Ty::HostShape)),
        "Seed" => Ok((i, Ty::Seed)),
        "PrfKey" => Ok((i, Ty::PrfKey)),
        "String" => Ok((i, Ty::HostString)),
        "BitTensor" => Ok((i, Ty::HostBitTensor)),
        "BitArray64" => Ok((i, Ty::HostBitArray64)),
        "BitArray128" => Ok((i, Ty::HostBitArray128)),
        "BitArray256" => Ok((i, Ty::HostBitArray256)),
        "Ring64Tensor" => Ok((i, Ty::HostRing64Tensor)),
        "Ring128Tensor" => Ok((i, Ty::HostRing128Tensor)),
        "Float32Tensor" => Ok((i, Ty::HostFloat32Tensor)),
        "Float64Tensor" => Ok((i, Ty::HostFloat64Tensor)),
        "Int8Tensor" => Ok((i, Ty::HostInt8Tensor)),
        "Int16Tensor" => Ok((i, Ty::HostInt16Tensor)),
        "Int32Tensor" => Ok((i, Ty::HostInt32Tensor)),
        "Int64Tensor" => Ok((i, Ty::HostInt64Tensor)),
        "Uint8Tensor" => Ok((i, Ty::HostUint8Tensor)),
        "Uint16Tensor" => Ok((i, Ty::HostUint16Tensor)),
        "Uint32Tensor" => Ok((i, Ty::HostUint32Tensor)),
        "Uint64Tensor" => Ok((i, Ty::HostUint64Tensor)),
        "HostFixed64Tensor" => Ok((i, Ty::HostFixed64Tensor)),
        "HostFixed128Tensor" => Ok((i, Ty::HostFixed128Tensor)),
        "Replicated64Tensor" => Ok((i, Ty::ReplicatedRing64Tensor)),
        "Replicated128Tensor" => Ok((i, Ty::ReplicatedRing128Tensor)),
        "ReplicatedBitTensor" => Ok((i, Ty::ReplicatedBitTensor)),
        "ReplicatedSetup" => Ok((i, Ty::ReplicatedSetup)),
        "Additive64Tensor" => Ok((i, Ty::AdditiveRing64Tensor)),
        "Additive128Tensor" => Ok((i, Ty::AdditiveRing128Tensor)),
        "ReplicatedShape" => Ok((i, Ty::ReplicatedShape)),
        "AdditiveBitTensor" => Ok((i, Ty::AdditiveBitTensor)),
        "AdditiveShape" => Ok((i, Ty::AdditiveShape)),
        "Fixed64Tensor" => Ok((i, Ty::Fixed64Tensor)),
        "Fixed128Tensor" => Ok((i, Ty::Fixed128Tensor)),
        "BooleanTensor" => Ok((i, Ty::BooleanTensor)),
        "Unit" => Ok((i, Ty::Unit)),
        "Float32" => Ok((i, Ty::Float32)),
        "Float64" => Ok((i, Ty::Float64)),
        "Ring64" => Ok((i, Ty::Ring64)),
        "Ring128" => Ok((i, Ty::Ring128)),
        "Tensor" => Ok((i, Ty::Tensor(TensorDType::Float64))), // TODO: Find the way to represent inner in the textual
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
pub fn constant_literal<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
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
        constant_literal_helper("Bit", parse_int, Constant::Bit),
        // 1D arrays
        alt((
            constant_literal_helper("Int8Tensor", vector(parse_int), |v| {
                Constant::HostInt8Tensor(v.into())
            }),
            constant_literal_helper("Int16Tensor", vector(parse_int), |v| {
                Constant::HostInt16Tensor(v.into())
            }),
            constant_literal_helper("Int32Tensor", vector(parse_int), |v| {
                Constant::HostInt32Tensor(v.into())
            }),
            constant_literal_helper("Int64Tensor", vector(parse_int), |v| {
                Constant::HostInt64Tensor(v.into())
            }),
            constant_literal_helper("Uint8Tensor", vector(parse_int), |v| {
                Constant::HostUint8Tensor(v.into())
            }),
            constant_literal_helper("Uint16Tensor", vector(parse_int), |v| {
                Constant::HostUint16Tensor(v.into())
            }),
            constant_literal_helper("Uint32Tensor", vector(parse_int), |v| {
                Constant::HostUint32Tensor(v.into())
            }),
            constant_literal_helper("Uint64Tensor", vector(parse_int), |v| {
                Constant::HostUint64Tensor(v.into())
            }),
            constant_literal_helper("Float32Tensor", vector(float), |v| {
                Constant::HostFloat32Tensor(v.into())
            }),
            constant_literal_helper("Float64Tensor", vector(double), |v| {
                Constant::HostFloat64Tensor(v.into())
            }),
            constant_literal_helper("Ring64Tensor", vector(parse_int), |v| {
                Constant::HostRing64Tensor(v.into())
            }),
            constant_literal_helper("Ring128Tensor", vector(parse_int), |v| {
                Constant::HostRing128Tensor(v.into())
            }),
            constant_literal_helper("HostBitTensor", vector(parse_int), |v| {
                Constant::HostBitTensor(v.into())
            }),
        )),
        // 2D arrays
        alt((
            constant_literal_helper("Int8Tensor", vector2(parse_int), |v| {
                Constant::HostInt8Tensor(v.into())
            }),
            constant_literal_helper("Int16Tensor", vector2(parse_int), |v| {
                Constant::HostInt16Tensor(v.into())
            }),
            constant_literal_helper("Int32Tensor", vector2(parse_int), |v| {
                Constant::HostInt32Tensor(v.into())
            }),
            constant_literal_helper("Int64Tensor", vector2(parse_int), |v| {
                Constant::HostInt64Tensor(v.into())
            }),
            constant_literal_helper("Uint8Tensor", vector2(parse_int), |v| {
                Constant::HostUint8Tensor(v.into())
            }),
            constant_literal_helper("Uint16Tensor", vector2(parse_int), |v| {
                Constant::HostUint16Tensor(v.into())
            }),
            constant_literal_helper("Uint32Tensor", vector2(parse_int), |v| {
                Constant::HostUint32Tensor(v.into())
            }),
            constant_literal_helper("Uint64Tensor", vector2(parse_int), |v| {
                Constant::HostUint64Tensor(v.into())
            }),
            constant_literal_helper("Float32Tensor", vector2(float), |v| {
                Constant::HostFloat32Tensor(v.into())
            }),
            constant_literal_helper("Float64Tensor", vector2(double), |v| {
                Constant::HostFloat64Tensor(v.into())
            }),
            constant_literal_helper(
                "Ring64Tensor",
                vector2(parse_int),
                |v: ndarray::ArrayD<u64>| Constant::HostRing64Tensor(v.into()),
            ),
            constant_literal_helper(
                "Ring128Tensor",
                vector2(parse_int),
                |v: ndarray::ArrayD<u128>| Constant::HostRing128Tensor(v.into()),
            ),
            constant_literal_helper("HostBitTensor", vector2(parse_int), |v| {
                Constant::HostBitTensor(v.into())
            }),
        )),
    ))(input)
}

/// Parses a literal for a value (a placed value).
fn value_literal<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Value, E> {
    context(
        "Expecting a value literal followed by a HostPlacement",
        alt((
            host_value_literal,
            host_fixed64_tensor,
            host_fixed128_tensor,
        )),
    )(input)
}

fn host_value_literal<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Value, E> {
    let (input, (v, p)) = tuple((constant_literal, ws(parse_placement)))(input)?;
    match p {
        Placement::Host(h) => Ok((input, v.place(&h))),
        _ => Err(Error(make_error(input, ErrorKind::MapRes))),
    }
}

fn host_fixed64_tensor<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Value, E> {
    let (input, (_, integral_precision, _, fractional_precision, _, tensor, placement)) =
        preceded(
            tag("HostFixed64Tensor"),
            tuple((
                ws(tag("[")),
                parse_int,
                ws(tag("/")),
                parse_int,
                ws(tag("]")),
                delimited(ws(tag("(")), vector(parse_int), ws(tag(")"))),
                ws(parse_placement),
            )),
        )(input)?;
    let placement = match placement {
        Placement::Host(h) => h,
        _ => return Err(Error(make_error(input, ErrorKind::MapRes))),
    };
    // This is a lot of internals. Will probably have a helper in the host.rs to go from Vec<u64> to HostFixed64Tensor.
    let tensor: Vec<std::num::Wrapping<u64>> = tensor.into_iter().map(std::num::Wrapping).collect();
    Ok((
        input,
        Value::HostFixed64Tensor(Box::new(HostFixed64Tensor {
            tensor: crate::host::HostRingTensor::<u64>(
                ndarray::Array::from(tensor).into_dyn(),
                placement,
            ),
            integral_precision,
            fractional_precision,
        })),
    ))
}

fn host_fixed128_tensor<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Value, E> {
    let (input, (_, integral_precision, _, fractional_precision, _, tensor, placement)) =
        preceded(
            tag("HostFixed128Tensor"),
            tuple((
                ws(tag("[")),
                parse_int,
                ws(tag("/")),
                parse_int,
                ws(tag("]")),
                delimited(ws(tag("(")), vector(parse_int), ws(tag(")"))),
                ws(parse_placement),
            )),
        )(input)?;
    let placement = match placement {
        Placement::Host(h) => h,
        _ => return Err(Error(make_error(input, ErrorKind::MapRes))),
    };
    let tensor: Vec<u128> = tensor;
    Ok((
        input,
        Value::HostFixed128Tensor(Box::new(HostFixed128Tensor {
            tensor: HostRing128Tensor::from_raw_plc(ndarray::Array::from(tensor), placement),
            integral_precision,
            fractional_precision,
        })),
    ))
}
/// Parses a vector of items, using the supplied inner parser.
fn vector<'a, F: 'a, O, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, Vec<O>, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(tag("["), separated_list0(ws(tag(",")), inner), tag("]"))
}

/// Parses a 2D vector of items, using the supplied inner parser.
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

/// Parses a literal for a Slice info (start, end, step)
pub fn slice_info_literal<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, SliceInfo, E> {
    let (input, (start, end, step)) = attributes!((
        attributes_member("start", parse_int),
        opt(attributes_member("end", parse_int)),
        opt(attributes_member("step", parse_int)),
    ))(input)?;

    Ok((input, SliceInfo(vec![SliceInfoElem { start, end, step }])))
}

/// Parses integer (or anything implementing FromStr from decimal digits)
pub fn parse_int<'a, O: std::str::FromStr, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, O, E> {
    map_res(
        recognize(tuple((opt(alt((tag("-"), tag("+")))), digit1))),
        |s: &str| s.parse::<O>(),
    )(input)
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
pub fn parse_hex<'a, E, const N: usize>(input: &'a str) -> IResult<&'a str, [u8; N], E>
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
pub fn ws<'a, F: 'a, O, E: ParseError<&'a str>>(
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
pub fn string<'a, E>(input: &'a str) -> IResult<&'a str, String, E>
where
    E: ParseError<&'a str>,
{
    let build_string = fold_many0(parse_fragment, String::new, |mut string, fragment| {
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
pub fn parse_bool<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, bool, E> {
    alt((value(true, tag("true")), value(false, tag("false"))))(input)
}

/// A helper convertor from a nom error to a generic error
///
/// Sample usage:
/// ```rust
/// use moose::textual::{parse_bool, friendly_error};
/// let source = "blah";
/// parse_bool(source).map_err(|e| friendly_error("Failed to parse a boolean", source, e));
/// ```
/// Note that it binds the E in the parser to be a `VerboseError`.
pub fn friendly_error(
    message: &str,
    source: &str,
    e: nom::Err<VerboseError<&str>>,
) -> anyhow::Error {
    match e {
        Failure(e) => anyhow::anyhow!("{} {}", message, convert_error(source, e)),
        Error(e) => anyhow::anyhow!("{} {}", message, convert_error(source, e)),
        _ => anyhow::anyhow!("{} {} due to {}", message, source, e),
    }
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
            Placement::Mirrored3(p) => p.to_textual(),
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

impl ToTextual for Mirrored3Placement {
    fn to_textual(&self) -> String {
        format!(
            "@Mirrored3({}, {}, {})",
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
            Cast(op) => op.to_textual(),
            Load(op) => op.to_textual(),
            Save(op) => op.to_textual(),
            Send(op) => op.to_textual(),
            Receive(op) => op.to_textual(),
            Input(op) => op.to_textual(),
            Output(op) => op.to_textual(),
            Constant(op) => op.to_textual(),
            Shape(op) => op.to_textual(),
            Broadcast(op) => op.to_textual(),
            Softmax(op) => op.to_textual(),
            AtLeast2D(op) => op.to_textual(),
            IndexAxis(op) => op.to_textual(),
            Slice(op) => op.to_textual(),
            Ones(op) => op.to_textual(),
            ExpandDims(op) => op.to_textual(),
            Concat(op) => op.to_textual(),
            Transpose(op) => op.to_textual(),
            Dot(op) => op.to_textual(),
            Inverse(op) => op.to_textual(),
            Add(op) => op.to_textual(),
            Sub(op) => op.to_textual(),
            Mul(op) => op.to_textual(),
            Mean(op) => op.to_textual(),
            Sum(op) => op.to_textual(),
            Div(op) => op.to_textual(),
            BitXor(op) => op.to_textual(),
            BitAnd(op) => op.to_textual(),
            BitNeg(op) => op.to_textual(),
            BitOr(op) => op.to_textual(),
            RingFill(op) => op.to_textual(),
            HostAdd(op) => op.to_textual(),
            HostSub(op) => op.to_textual(),
            HostMul(op) => op.to_textual(),
            HostDiv(op) => op.to_textual(),
            HostDot(op) => op.to_textual(),
            HostMean(op) => op.to_textual(),
            HostSqrt(op) => op.to_textual(),
            HostOnes(op) => op.to_textual(),
            HostExpandDims(op) => op.to_textual(),
            HostSqueeze(op) => op.to_textual(),
            HostReshape(op) => op.to_textual(),
            HostAtLeast2D(op) => op.to_textual(),
            HostSlice(op) => op.to_textual(),
            HostDiag(op) => op.to_textual(),
            HostShlDim(op) => op.to_textual(),
            HostBitDec(op) => op.to_textual(),
            HostSum(op) => op.to_textual(),
            HostTranspose(op) => op.to_textual(),
            HostInverse(op) => op.to_textual(),
            Sign(op) => op.to_textual(),
            RingNeg(op) => op.to_textual(),
            RingAdd(op) => op.to_textual(),
            RingSub(op) => op.to_textual(),
            RingMul(op) => op.to_textual(),
            RingDot(op) => op.to_textual(),
            RingFixedpointEncode(op) => op.to_textual(),
            RingFixedpointDecode(op) => op.to_textual(),
            RingFixedpointMean(op) => op.to_textual(),
            RingSample(op) => op.to_textual(),
            RingSampleSeeded(op) => op.to_textual(),
            RingShl(op) => op.to_textual(),
            RingShr(op) => op.to_textual(),
            RingInject(op) => op.to_textual(),
            BitExtract(op) => op.to_textual(),
            BitSample(op) => op.to_textual(),
            BitSampleSeeded(op) => op.to_textual(),
            PrimDeriveSeed(op) => op.to_textual(),
            PrimPrfKeyGen(op) => op.to_textual(),
            AesDecrypt(op) => op.to_textual(),
            FixedpointEncode(op) => op.to_textual(),
            FixedpointDecode(op) => op.to_textual(),
            FixedpointAdd(op) => op.to_textual(),
            FixedpointSub(op) => op.to_textual(),
            FixedpointMul(op) => op.to_textual(),
            FixedpointDiv(op) => op.to_textual(),
            FixedpointDot(op) => op.to_textual(),
            FixedpointTruncPr(op) => op.to_textual(),
            FixedpointMean(op) => op.to_textual(),
            FixedpointSum(op) => op.to_textual(),
            FloatingpointAdd(op) => op.to_textual(),
            FloatingpointSub(op) => op.to_textual(),
            FloatingpointMul(op) => op.to_textual(),
            FloatingpointDiv(op) => op.to_textual(),
            FloatingpointDot(op) => op.to_textual(),
            FloatingpointAtLeast2D(op) => op.to_textual(),
            FloatingpointOnes(op) => op.to_textual(),
            FloatingpointConcat(op) => op.to_textual(),
            FloatingpointExpandDims(op) => op.to_textual(),
            FloatingpointTranspose(op) => op.to_textual(),
            FloatingpointInverse(op) => op.to_textual(),
            FloatingpointMean(op) => op.to_textual(),
            FloatingpointSum(op) => op.to_textual(),
            RepSetup(op) => op.to_textual(),
            RepShare(op) => op.to_textual(),
            RepReveal(op) => op.to_textual(),
            RepDot(op) => op.to_textual(),
            RepFixedpointMean(op) => op.to_textual(),
            RepSum(op) => op.to_textual(),
            AddN(op) => op.to_textual(),
            RepAdd(op) => op.to_textual(),
            RepSub(op) => op.to_textual(),
            RepMul(op) => op.to_textual(),
            RepAnd(op) => op.to_textual(),
            RepXor(op) => op.to_textual(),
            RepNeg(op) => op.to_textual(),
            RepTruncPr(op) => op.to_textual(),
            AdtReveal(op) => op.to_textual(),
            AdtFill(op) => op.to_textual(),
            AdtAdd(op) => op.to_textual(),
            AdtSub(op) => op.to_textual(),
            AdtMul(op) => op.to_textual(),
            AdtShl(op) => op.to_textual(),
            AdtToRep(op) => op.to_textual(),
            RepAbs(op) => op.to_textual(),
            Fill(op) => op.to_textual(),
            RepMsb(op) => op.to_textual(),
            RepShl(op) => op.to_textual(),
            RepToAdt(op) => op.to_textual(),
            Index(op) => op.to_textual(),
            RepDiag(op) => op.to_textual(),
            RepBitDec(op) => op.to_textual(),
            RepBitCompose(op) => op.to_textual(),
            RepSlice(op) => op.to_textual(),
            RepShlDim(op) => op.to_textual(),
            RepEqual(op) => op.to_textual(),
            Mux(op) => op.to_textual(),
            Neg(op) => op.to_textual(),
            Pow2(op) => op.to_textual(),
            Exp(op) => op.to_textual(),
            Sigmoid(op) => op.to_textual(),
            Less(op) => op.to_textual(),
            GreaterThan(op) => op.to_textual(),
            Demirror(op) => op.to_textual(),
            Mirror(op) => op.to_textual(),
            Maximum(op) => op.to_textual(),
        }
    }
}

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

op_with_axis_to_textual!(MeanOp);
op_with_axis_to_textual!(SumOp);
op_with_axis_to_textual!(HostMeanOp);
op_with_axis_to_textual!(HostSumOp);
op_with_axis_to_textual!(RepSumOp);
op_with_axis_to_textual!(FixedpointSumOp);
op_with_axis_to_textual!(FloatingpointMeanOp);
op_with_axis_to_textual!(FloatingpointSumOp);
op_with_axis_to_textual!(HostSqueezeOp);

impl ToTextual for FixedpointMeanOp {
    fn to_textual(&self) -> String {
        match self {
            FixedpointMeanOp { sig, axis: Some(a) } => {
                format!("FixedpointMean{{axis = {}}}: {}", a, sig.to_textual())
            }
            FixedpointMeanOp { sig, axis: None } => {
                format!("FixedpointMean{{}}: {}", sig.to_textual())
            }
        }
    }
}

impl ToTextual for RingFixedpointMeanOp {
    fn to_textual(&self) -> String {
        match self {
            RingFixedpointMeanOp {
                sig,
                axis: Some(a),
                scaling_base,
                scaling_exp,
            } => {
                format!(
                    "RingFixedpointMean{{axis = {}, scaling_base={}, scaling_exp={}}}: {}",
                    a,
                    scaling_base,
                    scaling_exp,
                    sig.to_textual()
                )
            }
            RingFixedpointMeanOp {
                sig,
                axis: None,
                scaling_base,
                scaling_exp,
            } => format!(
                "RingFixedpointMean{{scaling_base={}, scaling_exp={}}}: {}",
                scaling_base,
                scaling_exp,
                sig.to_textual()
            ),
        }
    }
}

impl ToTextual for RepFixedpointMeanOp {
    fn to_textual(&self) -> String {
        match self {
            RepFixedpointMeanOp {
                sig,
                axis: Some(a),
                scaling_base,
                scaling_exp,
            } => {
                format!(
                    "RepFixedpointMean{{axis = {}, scaling_base={}, scaling_exp={}}}: {}",
                    a,
                    scaling_base,
                    scaling_exp,
                    sig.to_textual()
                )
            }
            RepFixedpointMeanOp {
                sig,
                axis: None,
                scaling_base,
                scaling_exp,
            } => format!(
                "RepFixedpointMean{{scaling_base={}, scaling_exp={}}}: {}",
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
            } => format!("RingSample{{}}: {}", sig.to_textual()),
        }
    }
}

impl ToTextual for RingSampleSeededOp {
    fn to_textual(&self) -> String {
        match self {
            RingSampleSeededOp {
                sig,
                max_value: Some(a),
            } => format!(
                "RingSampleSeeded{{max_value = {}}}: {}",
                a,
                sig.to_textual()
            ),
            RingSampleSeededOp {
                sig,
                max_value: None,
            } => format!("RingSampleSeeded{{}}: {}", sig.to_textual()),
        }
    }
}

impl ToTextual for SoftmaxOp {
    fn to_textual(&self) -> String {
        match self {
            SoftmaxOp {
                sig,
                axis: Some(a),
                upmost_index,
            } => format!(
                "Softmax{{max_value = {}}}: {} {}",
                a,
                sig.to_textual(),
                upmost_index
            ),
            SoftmaxOp {
                sig,
                axis: None,
                upmost_index,
            } => format!("Softmax{{}}: {} {}", sig.to_textual(), upmost_index),
        }
    }
}

impl ToTextual for Ty {
    fn to_textual(&self) -> String {
        match self {
            Ty::Unit => "Unit".to_string(),
            Ty::HostString => "String".to_string(),
            Ty::Float32 => "Float32".to_string(),
            Ty::Float64 => "Float64".to_string(),
            Ty::Ring64 => "Ring64".to_string(),
            Ty::Ring128 => "Ring128".to_string(),
            Ty::Fixed => "Fixed".to_string(),
            Ty::Tensor(i) => format!("Tensor({})", i), // TODO (lvorona) Come up with a textual format here
            Ty::HostRing64Tensor => "Ring64Tensor".to_string(),
            Ty::HostRing128Tensor => "Ring128Tensor".to_string(),
            Ty::Bit => "Bit".to_string(),
            Ty::HostBitTensor => "BitTensor".to_string(),
            Ty::HostBitArray64 => "BitArray64".to_string(),
            Ty::HostBitArray128 => "BitArray128".to_string(),
            Ty::HostBitArray224 => "BitArray224".to_string(),
            Ty::HostBitArray256 => "BitArray256".to_string(),
            Ty::HostShape => "Shape".to_string(),
            Ty::Seed => "Seed".to_string(),
            Ty::PrfKey => "PrfKey".to_string(),
            Ty::HostFloat32Tensor => "Float32Tensor".to_string(),
            Ty::HostFloat64Tensor => "Float64Tensor".to_string(),
            Ty::HostInt8Tensor => "Int8Tensor".to_string(),
            Ty::HostInt16Tensor => "Int16Tensor".to_string(),
            Ty::HostInt32Tensor => "Int32Tensor".to_string(),
            Ty::HostInt64Tensor => "Int64Tensor".to_string(),
            Ty::HostUint8Tensor => "Uint8Tensor".to_string(),
            Ty::HostUint16Tensor => "Uint16Tensor".to_string(),
            Ty::HostUint32Tensor => "Uint32Tensor".to_string(),
            Ty::HostUint64Tensor => "Uint64Tensor".to_string(),
            Ty::Unknown => "Unknown".to_string(),
            Ty::HostFixed64Tensor => "HostFixed64Tensor".to_string(),
            Ty::HostFixed128Tensor => "HostFixed128Tensor".to_string(),
            Ty::ReplicatedRing64Tensor => "ReplicatedRing64Tensor".to_string(),
            Ty::ReplicatedRing128Tensor => "ReplicatedRing128Tensor".to_string(),
            Ty::ReplicatedFixed64Tensor => "ReplicatedFixed64Tensor".to_string(),
            Ty::ReplicatedFixed128Tensor => "ReplicatedFixed128Tensor".to_string(),
            Ty::ReplicatedBitTensor => "ReplicatedBitTensor".to_string(),
            Ty::ReplicatedBitArray64 => "ReplicatedBitArray64".to_string(),
            Ty::ReplicatedBitArray128 => "ReplicatedBitArray128".to_string(),
            Ty::ReplicatedBitArray224 => "ReplicatedBitArray224".to_string(),
            Ty::ReplicatedSetup => "ReplicatedSetup".to_string(),
            Ty::ReplicatedShape => "ReplicatedShape".to_string(),
            Ty::AdditiveBitTensor => "AdditiveBitTensor".to_string(),
            Ty::AdditiveRing64Tensor => "Additive64Tensor".to_string(),
            Ty::AdditiveRing128Tensor => "Additive128Tensor".to_string(),
            Ty::AdditiveShape => "AdditiveShape".to_string(),
            Ty::BooleanTensor => "BooleanTensor".to_string(),
            Ty::Fixed64Tensor => "Fixed64Tensor".to_string(),
            Ty::Fixed128Tensor => "Fixed128Tensor".to_string(),
            Ty::Float32Tensor => "Float32Tensor".to_string(),
            Ty::Float64Tensor => "Float64Tensor".to_string(),
            Ty::Mirrored3Ring64Tensor => "Mirrored3Ring64Tensor".to_string(),
            Ty::Mirrored3Ring128Tensor => "Mirrored3Ring128Tensor".to_string(),
            Ty::Mirrored3BitTensor => "Mirrored3BitTensor".to_string(),
            Ty::Mirrored3Float32 => "Mirrored3Float32".to_string(),
            Ty::Mirrored3Float64 => "Mirrored3Float64".to_string(),
            Ty::Mirrored3Fixed64Tensor => "Mirrored3Fixed64Tensor".to_string(),
            Ty::Mirrored3Fixed128Tensor => "Mirrored3Fixed128Tensor".to_string(),
            Ty::HostFixed128AesTensor => "HostFixed128AesTensor".to_string(),
            Ty::HostAesKey => "HostAesKey".to_string(),
            Ty::ReplicatedAesKey => "ReplicatedAesKey".to_string(),
            Ty::Fixed128AesTensor => "Fixed128AesTensor".to_string(),
            Ty::AesTensor => "AesTensor".to_string(),
            Ty::AesKey => "AesKey".to_string(),
        }
    }
}

macro_rules! format_to_textual {
    ($format:expr, $($member:expr),*) => {
        format!($format, $($member.to_textual(),)*)
    };
}

impl ToTextual for Value {
    fn to_textual(&self) -> String {
        match self {
            Value::HostInt8Tensor(x) => format_to_textual!("Int8Tensor({}) {}", x.0, x.1),
            Value::HostInt16Tensor(x) => format_to_textual!("Int16Tensor({}) {}", x.0, x.1),
            Value::HostInt32Tensor(x) => format_to_textual!("Int32Tensor({}) {}", x.0, x.1),
            Value::HostInt64Tensor(x) => format_to_textual!("Int64Tensor({}) {}", x.0, x.1),
            Value::HostUint8Tensor(x) => format_to_textual!("Uint8Tensor({}) {}", x.0, x.1),
            Value::HostUint16Tensor(x) => format_to_textual!("Uint16Tensor({}) {}", x.0, x.1),
            Value::HostUint32Tensor(x) => format_to_textual!("Uint32Tensor({}) {}", x.0, x.1),
            Value::HostUint64Tensor(x) => format_to_textual!("Uint64Tensor({}) {}", x.0, x.1),
            Value::HostFloat32Tensor(x) => format_to_textual!("Float32Tensor({}) {}", x.0, x.1),
            Value::HostFloat64Tensor(x) => format_to_textual!("Float64Tensor({}) {}", x.0, x.1),
            Value::HostRing64Tensor(x) => format_to_textual!("Ring64Tensor({}) {}", x.0, x.1),
            Value::HostRing128Tensor(x) => format_to_textual!("Ring128Tensor({}) {}", x.0, x.1),
            // TODO: Hosted floats for values
            Value::Float32(x) => format!("Float32({}) @Host(TODO)", x),
            Value::Float64(x) => format!("Float64({}) @Host(TODO)", x),
            Value::Fixed(x) => format!("Fixed[{}]({})", x.precision, x.value),
            Value::HostString(x) => format_to_textual!("String({}) {}", x.0, x.1),
            Value::Ring64(x) => format!("Ring64({})", x),
            Value::Ring128(x) => format!("Ring128({})", x),
            Value::HostShape(x) => format!("Shape({:?}) {}", x.0 .0, x.1.to_textual()),
            Value::Seed(x) => format_to_textual!("Seed({}) {}", x.0 .0, x.1),
            Value::PrfKey(x) => format_to_textual!("PrfKey({}) {}", x.0 .0, x.1),
            Value::Bit(x) => format!("Bit({})", x),
            Value::Unit(_) => "Unit".to_string(),
            Value::HostBitTensor(x) => format_to_textual!("HostBitTensor({}) {}", x.0, x.1),
            Value::HostFixed64Tensor(x) => format_to_textual!(
                "HostFixed64Tensor[{}/{}]({}) {}",
                x.integral_precision,
                x.fractional_precision,
                x.tensor.0,
                x.tensor.1
            ),
            Value::HostFixed128Tensor(x) => format_to_textual!(
                "HostFixed128Tensor[{}/{}]({}) {}",
                x.integral_precision,
                x.fractional_precision,
                x.tensor.0,
                x.tensor.1
            ),
            Value::HostBitArray64(_) | Value::Tensor(_) | Value::HostBitArray128(_) => {
                unimplemented!()
            }
            Value::HostBitArray224(_) => unimplemented!(),
            Value::HostBitArray256(_) => unimplemented!(),
            // The following value variants live in the replicated form and can not be represented in the textual computation graph.
            Value::Fixed64Tensor(_)
            | Value::Fixed128Tensor(_)
            | Value::BooleanTensor(_)
            | Value::Float32Tensor(_)
            | Value::Float64Tensor(_)
            | Value::ReplicatedShape(_)
            | Value::ReplicatedSetup(_)
            | Value::ReplicatedBitTensor(_)
            | Value::ReplicatedBitArray64(_)
            | Value::ReplicatedBitArray128(_)
            | Value::ReplicatedBitArray224(_)
            | Value::ReplicatedRing64Tensor(_)
            | Value::ReplicatedRing128Tensor(_)
            | Value::ReplicatedFixed64Tensor(_)
            | Value::ReplicatedFixed128Tensor(_)
            | Value::Mirrored3Ring64Tensor(_)
            | Value::Mirrored3Ring128Tensor(_)
            | Value::Mirrored3BitTensor(_)
            | Value::Mirrored3Float32(_)
            | Value::Mirrored3Float64(_)
            | Value::Mirrored3Fixed64Tensor(_)
            | Value::Mirrored3Fixed128Tensor(_)
            | Value::AdditiveShape(_)
            | Value::AdditiveBitTensor(_)
            | Value::AdditiveRing64Tensor(_)
            | Value::AdditiveRing128Tensor(_) => {
                unimplemented!("Unsupported Value variant: {:?}", self)
            }
            Value::HostFixed128AesTensor(_) => {
                unimplemented!()
            }
            Value::HostAesKey(_) => unimplemented!(),
            Value::ReplicatedAesKey(_) => unimplemented!(),
            Value::Fixed128AesTensor(_) => unimplemented!(),
            Value::AesTensor(_) => unimplemented!(),
            Value::AesKey(_) => unimplemented!(),
        }
    }
}

impl ToTextual for Constant {
    fn to_textual(&self) -> String {
        match self {
            Constant::HostInt8Tensor(x) => format!("Int8Tensor({})", x.0.to_textual()),
            Constant::HostInt16Tensor(x) => format!("Int16Tensor({})", x.0.to_textual()),
            Constant::HostInt32Tensor(x) => format!("Int32Tensor({})", x.0.to_textual()),
            Constant::HostInt64Tensor(x) => format!("Int64Tensor({})", x.0.to_textual()),
            Constant::HostUint8Tensor(x) => format!("Uint8Tensor({})", x.0.to_textual()),
            Constant::HostUint16Tensor(x) => format!("Uint16Tensor({})", x.0.to_textual()),
            Constant::HostUint32Tensor(x) => format!("Uint32Tensor({})", x.0.to_textual()),
            Constant::HostUint64Tensor(x) => format!("Uint64Tensor({})", x.0.to_textual()),
            Constant::HostFloat32Tensor(x) => format!("Float32Tensor({})", x.0.to_textual()),
            Constant::HostFloat64Tensor(x) => format!("Float64Tensor({})", x.0.to_textual()),
            Constant::HostRing64Tensor(x) => format!("Ring64Tensor({})", x.0.to_textual()),
            Constant::HostRing128Tensor(x) => format!("Ring128Tensor({})", x.0.to_textual()),
            Constant::Float32(x) => format!("Float32({})", x),
            Constant::Float64(x) => format!("Float64({})", x),
            Constant::String(x) => format!("String({})", x.to_textual()),
            Constant::Ring64(x) => format!("Ring64({})", x),
            Constant::Ring128(x) => format!("Ring128({})", x),
            Constant::Fixed(FixedpointConstant { value, precision }) => {
                format!("Fixed({}, {})", value, precision)
            }
            Constant::RawShape(RawShape(x)) => format!("Shape({:?})", x),
            Constant::RawSeed(RawSeed(x)) => format!("Seed({})", x.to_textual()),
            Constant::RawPrfKey(RawPrfKey(x)) => format!("PrfKey({})", x.to_textual()),
            Constant::Bit(x) => format!("Bit({})", x),
            Constant::HostBitTensor(x) => format!("HostBitTensor({})", x.0.to_textual()),
        }
    }
}

impl<T: std::fmt::Debug> ToTextual for ndarray::ArrayD<T> {
    fn to_textual(&self) -> String {
        match self.shape() {
            [_len] => format!("{:?}", self.as_slice().unwrap()),
            [rows, cols] => {
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
            _ => unimplemented!("ArrayD.to_textual() unimplemented for tensors of rank > 3"),
        }
    }
}

impl ToTextual for Role {
    fn to_textual(&self) -> String {
        format!("{:?}", self.0)
    }
}

// Required to serialize PrimDeriveSeedOp
impl ToTextual for SyncKey {
    fn to_textual(&self) -> String {
        format!("{:?}", self.as_bytes())
    }
}

// Required to serialize Send/Receive
impl ToTextual for RendezvousKey {
    fn to_textual(&self) -> String {
        self.as_bytes().to_textual()
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
            Signature::Variadic(VariadicSignature { args, ret }) => {
                format!("[{}] -> {}", args.to_textual(), ret.to_textual())
            }
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
use_debug_to_textual!(Vec<u32>);
use_debug_to_textual!(u64);
use_debug_to_textual!(bool);
use_debug_to_textual!(RawShape);

impl ToTextual for SliceInfo {
    fn to_textual(&self) -> String {
        match self.0.first() {
            Some(e) => {
                let end_string = e.end.map(|v| format!(", end = {}", v)).unwrap_or_default();
                let step_string = e
                    .step
                    .map(|v| format!(", step = {}", v))
                    .unwrap_or_default();
                format!("{{start = {}{}{}}}", e.start, end_string, step_string)
            }
            _ => format!("{:?}", self.0), // Fallback to debug print
        }
    }
}

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
    use rstest::rstest;
    use std::convert::TryInto;

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
            Constant::HostRing64Tensor(vec![1, 2, 3].into())
        );
        let (_, parsed_ring128_tensor) =
            constant_literal::<(&str, ErrorKind)>("Ring128Tensor([1,2,3])")?;
        assert_eq!(
            parsed_ring128_tensor,
            Constant::HostRing128Tensor(vec![1, 2, 3].into())
        );
        let (_, parsed_shape) = constant_literal::<(&str, ErrorKind)>("Shape([1,2,3])")?;
        assert_eq!(parsed_shape, Constant::RawShape(RawShape(vec![1, 2, 3])));
        let (_, parsed_u8_tensor) = constant_literal::<(&str, ErrorKind)>("Uint8Tensor([1,2,3])")?;
        assert_eq!(
            parsed_u8_tensor,
            Constant::HostUint8Tensor(vec![1, 2, 3].into())
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
    fn test_array_literal_non_square() -> Result<(), anyhow::Error> {
        let parsed_f32: Constant =
            "Float32Tensor([[1.0, 11, 12, 13], [3.0, 21, 22, 23]])".try_into()?;
        assert_eq!(
            parsed_f32.to_textual(),
            "Float32Tensor([[1.0, 11.0, 12.0, 13.0], [3.0, 21.0, 22.0, 23.0]])"
        );
        Ok(())
    }

    #[test]
    fn test_array_literal() -> Result<(), anyhow::Error> {
        use ndarray::prelude::*;
        use std::convert::TryInto;
        let parsed_f32: Constant = "Float32Tensor([[1.0, 2.0], [3.0, 4.0]])".try_into()?;

        let x = HostFloat32Tensor::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );

        assert_eq!(parsed_f32, Constant::HostFloat32Tensor(x));

        let parsed_ring64: Constant = "Ring64Tensor([[1, 2], [3, 4]])".try_into()?;

        let x_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = HostRing64Tensor::from(x_backing);

        assert_eq!(parsed_ring64, Constant::HostRing64Tensor(x));

        Ok(())
    }

    #[test]
    fn test_type_parsing() -> Result<(), anyhow::Error> {
        let (_, parsed_type) = parse_type::<(&str, ErrorKind)>("Unit")?;
        assert_eq!(parsed_type, Ty::Unit);
        let (_, parsed) = operator_signature::<(&str, ErrorKind)>(0)(
            ": (Float32Tensor, Float64Tensor) -> Uint16Tensor",
        )?;
        assert_eq!(
            parsed,
            Signature::binary(
                Ty::HostFloat32Tensor,
                Ty::HostFloat64Tensor,
                Ty::HostUint16Tensor
            ),
        );

        let (_, parsed) =
            operator_signature::<(&str, ErrorKind)>(0)(": [Float32Tensor] -> Float32Tensor")?;
        assert_eq!(
            parsed,
            Signature::variadic(Ty::HostFloat32Tensor, Ty::HostFloat32Tensor),
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
                sig: Signature::nullary(Ty::HostFloat32Tensor),
                value: Constant::HostFloat32Tensor(vec![1.0].into())
            })
        );

        // 2D tensor
        use ndarray::prelude::*;
        let x = HostFloat32Tensor::from(
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
                sig: Signature::nullary(Ty::HostFloat32Tensor),
                value: Constant::HostFloat32Tensor(x)
            })
        );
        Ok(())
    }

    #[test]
    fn test_stdbinary() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z = HostAdd: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(carole)",
        )?;
        assert_eq!(op.name, "z");
        assert_eq!(
            op.kind,
            Operator::HostAdd(HostAddOp {
                sig: Signature::binary(
                    Ty::HostFloat32Tensor,
                    Ty::HostFloat32Tensor,
                    Ty::HostFloat32Tensor
                ),
            })
        );
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z = HostMul: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(carole)",
        )?;
        assert_eq!(op.name, "z");
        assert_eq!(
            op.kind,
            Operator::HostMul(HostMulOp {
                sig: Signature::binary(
                    Ty::HostFloat32Tensor,
                    Ty::HostFloat32Tensor,
                    Ty::HostFloat32Tensor
                ),
            })
        );
        Ok(())
    }

    #[test]
    fn test_primprfkeygen() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "key = PrimPrfKeyGen: () -> PrfKey () @Host(alice)",
        )?;
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
                sync_key: SyncKey::try_from(vec![1, 2, 3])?
            })
        );
        Ok(())
    }

    #[test]
    fn test_send() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            r#"send = Send{rendezvous_key = 30313233343536373839616263646566, receiver = "bob"}: (Float32Tensor) -> Unit() @Host(alice)"#,
        )?;
        assert_eq!(op.name, "send");
        assert_eq!(
            op.kind,
            Operator::Send(SendOp {
                sig: Signature::unary(Ty::HostFloat32Tensor, Ty::Unit),
                rendezvous_key: "0123456789abcdef".try_into()?,
                receiver: Role::from("bob")
            })
        );
        Ok(())
    }

    #[test]
    fn test_receive() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            r#"receive = Receive{rendezvous_key = 30313233343536373839616263646566, sender = "bob"} : () -> Float32Tensor () @Host(alice)"#,
        )?;
        assert_eq!(op.name, "receive");
        assert_eq!(
            op.kind,
            Operator::Receive(ReceiveOp {
                sig: Signature::nullary(Ty::HostFloat32Tensor),
                rendezvous_key: "0123456789abcdef".try_into()?,
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
            "x10 = RingSampleSeeded{max_value = 1}: (Shape, Seed) -> Ring64Tensor (shape, seed) @Host(alice)",
        )?;
        assert_eq!(op.name, "x10");
        Ok(())
    }

    #[test]
    fn test_slice_option() -> Result<(), anyhow::Error> {
        let input = "x10 = HostSlice{slice = {start = 1, end = 10, step = -1}}: (Ring64Tensor) -> Ring64Tensor (x) @Host(alice)";
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(input)?;
        assert_eq!(op.name, "x10");
        assert_eq!(
            op.kind,
            Operator::HostSlice(HostSliceOp {
                sig: Signature::unary(Ty::HostRing64Tensor, Ty::HostRing64Tensor),
                slice: SliceInfo(vec![SliceInfoElem {
                    start: 1,
                    end: Some(10),
                    step: Some(-1),
                }])
            })
        );
        assert_eq!(op.to_textual(), input);
        Ok(())
    }

    #[test]
    fn test_slice() -> Result<(), anyhow::Error> {
        let input = "x10 = HostSlice{slice = {start = 1, end = 10, step = 1}}: (Ring64Tensor) -> Ring64Tensor (x) @Host(alice)";
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(input)?;
        assert_eq!(op.name, "x10");
        assert_eq!(
            op.kind,
            Operator::HostSlice(HostSliceOp {
                sig: Signature::unary(Ty::HostRing64Tensor, Ty::HostRing64Tensor),
                slice: SliceInfo(vec![SliceInfoElem {
                    start: 1,
                    end: Some(10),
                    step: Some(1),
                }])
            })
        );
        assert_eq!(op.to_textual(), input);
        Ok(())
    }

    #[test]
    fn test_fixedpoint_ring_mean() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "op = RingFixedpointMean{axis = 0, scaling_base = 3, scaling_exp = 1} : () -> Float32Tensor () @Host(alice)",
        )?;
        assert_eq!(
            op.kind,
            Operator::RingFixedpointMean(RingFixedpointMeanOp {
                sig: Signature::nullary(Ty::HostFloat32Tensor),
                axis: Some(0),
                scaling_base: 3,
                scaling_exp: 1,
            })
        );

        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "op = RingFixedpointMean{scaling_base = 3, scaling_exp = 1} : () -> Float32Tensor () @Host(alice)",
        )?;
        assert_eq!(
            op.kind,
            Operator::RingFixedpointMean(RingFixedpointMeanOp {
                sig: Signature::nullary(Ty::HostFloat32Tensor),
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
            "x_shape = Constant{value = Shape([2, 2])}: () -> Shape () @Host(alice)",
        )?;
        assert_eq!(op.name, "x_shape");
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z_result = HostAdd: (Float32Tensor, Float32Tensor) -> Float32Tensor (x_shape, y_shape) @Host(carole)",
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
            "z = HostExpandDims {axis = [0]}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = HostAtLeast2D {to_column_vector = false}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = HostSlice {slice = {start = 1, end = 2}}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = HostDiag: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = HostSqrt: (Float32Tensor) -> Float32Tensor () @Host(alice)",
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
            "z = RingFixedpointDecode {scaling_base = 3, scaling_exp = 2}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = RingFixedpointEncode {scaling_base = 3, scaling_exp = 2}: (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = RingInject {bit_idx = 2} : (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = BitExtract {bit_idx = 2} : (Float32Tensor) -> Float32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = BitSampleSeeded: (Shape, Seed) -> BitTensor (shape, seed) @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = BitXor: (BitTensor, BitTensor) -> BitTensor (x, y) @Host(alice)",
        )?;

        parse_assignment::<(&str, ErrorKind)>(
            "load = Load: (String, String) -> Float64Tensor (xuri, xconstant) @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "addN = AddN: [String] -> String (xuri, xconstant) @Host(alice)",
        )?;

        Ok(())
    }

    #[test]
    fn test_sample_computation() -> Result<(), anyhow::Error> {
        let (_, comp) = parse_computation::<(&str, ErrorKind)>(
            "x = Constant{value = Float32Tensor([1.0])}: () -> Float32Tensor() @Host(alice)
            y = Constant{value = Float32Tensor([2.0])}: () -> Float32Tensor () @Host(bob)
            // ignore = Constant([1.0]: Float32Tensor) @Host(alice)
            z = HostAdd: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(carole)
            ",
        )?;
        assert_eq!(comp.operations.len(), 3);
        assert_eq!(
            comp.operations[0].kind,
            Operator::Constant(ConstantOp {
                sig: Signature::nullary(Ty::HostFloat32Tensor),
                value: Constant::HostFloat32Tensor(vec![1.0].into())
            })
        );
        assert_eq!(
            comp.operations[1].kind,
            Operator::Constant(ConstantOp {
                sig: Signature::nullary(Ty::HostFloat32Tensor),
                value: Constant::HostFloat32Tensor(vec![2.0].into())
            })
        );
        assert_eq!(comp.operations[2].name, "z");
        assert_eq!(
            comp.operations[2].kind,
            Operator::HostAdd(HostAddOp {
                sig: Signature::binary(
                    Ty::HostFloat32Tensor,
                    Ty::HostFloat32Tensor,
                    Ty::HostFloat32Tensor
                ),
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
        let data = r#"a = Constant{value = "a"}: () -> Float32Tensor () @Host(alice)
            err = HostAdd: (Float32Tensor) -> Float32Tensor (x, y) @Host(carole)
            b = Constant{value = "b"}: () -> Float32Tensor () @Host(alice)"#;
        let emsg = r#"0: at line 2, in Tag:
            err = HostAdd: (Float32Tensor) -> Float32Tensor (x, y) @Host(carole)
                            ^

1: at line 2, in Alt:
            err = HostAdd: (Float32Tensor) -> Float32Tensor (x, y) @Host(carole)
                           ^

"#;
        let parsed: IResult<_, _, VerboseError<&str>> = parse_computation(data);
        if let Err(Failure(e)) = parsed {
            assert_eq!(convert_error(data, e), emsg);
        }
    }

    #[test]
    fn test_computation_try_into() -> Result<(), anyhow::Error> {
        use std::convert::TryInto;
        let comp: Computation =
            "x = Constant{value = Float32Tensor([1.0])}: () -> Float32Tensor @Host(alice)
            y = Constant{value = Float32Tensor([2.0])}: () -> Float32Tensor () @Host(bob)
            z = HostAdd: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Host(carole)"
                .try_into()?;
        assert_eq!(comp.operations.len(), 3);
        Ok(())
    }

    #[test]
    fn test_constant_try_into() -> Result<(), anyhow::Error> {
        let v: Constant = "Float32Tensor([1.0, 2.0, 3.0])".try_into()?;
        assert_eq!(v, Constant::HostFloat32Tensor(vec![1.0, 2.0, 3.0].into()));
        Ok(())
    }

    #[rstest]
    #[case("Int8Tensor([2, 3]) @Host(alice)")]
    #[case("Int16Tensor([2, 3]) @Host(alice)")]
    #[case("Int32Tensor([2, 3]) @Host(alice)")]
    #[case("Int64Tensor([2, 3]) @Host(alice)")]
    #[case("Uint8Tensor([2, 3]) @Host(alice)")]
    #[case("Uint16Tensor([2, 3]) @Host(alice)")]
    #[case("Uint32Tensor([2, 3]) @Host(alice)")]
    #[case("Uint64Tensor([2, 3]) @Host(alice)")]
    #[case("Float32Tensor([2.1, 3.2]) @Host(alice)")]
    #[case("Float64Tensor([2.1, 3.2]) @Host(alice)")]
    #[case("Ring64Tensor([2, 3]) @Host(alice)")]
    #[case("Ring128Tensor([2, 3]) @Host(alice)")]
    #[case("Float32(2.1) @Host(TODO)")]
    #[case("Float64(2.1) @Host(TODO)")]
    #[case("String(\"hi\") @Host(alice)")]
    #[case("HostBitTensor([0, 1, 1, 0]) @Host(alice)")]
    #[case("Shape([3, 2]) @Host(alice)")]
    #[case("Seed(529c2fc9bf573d077f45f42b19cfb8d4) @Host(alice)")]
    #[case("PrfKey(00000000000000000000000000000000) @Host(alice)")]
    #[case("HostFixed64Tensor[7/12]([2, 42, 12]) @Host(alice)")]
    #[case("HostFixed128Tensor[7/12]([2, 42, 12]) @Host(alice)")]
    fn test_value_round_trip(#[case] input: String) -> Result<(), anyhow::Error> {
        let value: Value = input.parse()?;
        let textual = value.to_textual();
        assert_eq!(textual, input);
        Ok(())
    }

    #[test]
    fn test_whitespace() -> Result<(), anyhow::Error> {
        use std::convert::TryInto;
        let source = r#"
        x = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(alice)

        y = Constant{value=Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(bob)

        "#;
        let comp: Computation = source.try_into()?;
        assert_eq!(comp.operations.len(), 2);
        Ok(())
    }

    #[test]
    fn test_computation_into_text() -> Result<(), anyhow::Error> {
        use std::convert::TryInto;
        let comp: Computation = "x = Constant{value = Float32Tensor([1.0])}: () -> Float32Tensor @Host(alice)
            y = Constant{value = Float32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> Float32Tensor @Host(bob)
            z = HostAdd: (Float32Tensor, Float32Tensor) -> Float32Tensor (x, y) @Replicated(alice, bob, carole)
            seed = PrimDeriveSeed{sync_key = [1, 2, 3]} (key) @Host(alice)
            seed2 = Constant{value = Seed(529c2fc9bf573d077f45f42b19cfb8d4)}: () -> Seed @Host(alice)
            o = Output: (Float32Tensor) -> Float32Tensor (z) @Host(alice)"
            .try_into()?;
        let textual = comp.to_textual();
        // After serializing it into the textual IR we need to make sure it parses back the same
        let comp2: Computation = textual.try_into()?;
        assert_eq!(comp.operations[0], comp2.operations[0]);
        Ok(())
    }
}
