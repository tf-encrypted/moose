use super::*;

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
        Err(e) => Err(friendly_error("Failed to parse computation", source, e)),
        Ok((_, computation)) => Ok(computation),
    }
}

/// Parses the computation and returns a simple error if it fails. It is only about 10% faster than the verbose version.
pub fn fast_parse_computation(source: &str) -> anyhow::Result<Computation> {
    match parse_computation::<(&str, ErrorKind)>(source) {
        Ok((_, computation)) => Ok(computation),
        e => Err(anyhow::anyhow!(
            "Failed to parse computation due to {:?}",
            e
        )),
    }
}

pub fn parallel_parse_computation(source: &str, chunks: usize) -> anyhow::Result<Computation> {
    // Split the source into `chunks` parts at line breaks.
    let mut parts = Vec::<&str>::with_capacity(chunks);
    let mut left: usize = 0;
    let step = source.len() / chunks;
    for _ in 0..chunks {
        let right = left + step;
        let right = if right > source.len() {
            source.len()
        } else {
            // Find the next line break or use the end of string, if there is none.
            source[right..]
                .find('\n')
                .map(|i| i + right + 1)
                .unwrap_or(source.len())
        };
        if left != right {
            parts.push(&source[left..right]);
        }
        left = right;
    }
    let portions: Vec<_> = parts
        .par_iter()
        .map(|s| {
            parse_operations::<VerboseError<&str>>(s)
                .map(|t| t.1) // Dropping the remainder
                .map_err(|e| friendly_error("Failed to parse computation", source, e))
        })
        .collect();
    let mut operations = Vec::new();
    for p in portions {
        operations.append(&mut p?);
    }

    Ok(Computation { operations })
}

fn parse_operations<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Vec<Operation>, E> {
    // A decent guess for the initial capacity of the vector.
    let capacity = 4 + input.len() / 100;
    let (input, operations) = all_consuming(fold_many0(
        parse_line,
        || Vec::with_capacity(capacity),
        |mut acc: Vec<_>, item| {
            acc.push(item);
            acc
        },
    ))(input)?;
    Ok((input, operations))
}

/// Parses the computation line by line.
fn parse_computation<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Computation, E> {
    let (input, operations) = parse_operations(input)?;
    Ok((input, Computation { operations }))
}

/// Parses a single logical line of the textual IR
fn parse_line<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operation, E> {
    let (input, _) = many0(recognize_comment)(input)?;
    let (input, _) = many0(multispace1)(input)?;
    let (input, op) = parse_assignment(input)?;
    let (input, _) = many0(multispace1)(input)?;
    Ok((input, op))
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
        preceded(
            tag("@Mirrored3"),
            cut(context(
                "Expecting host names triplet as in @Mirrored3(alice, bob, charlie)",
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
                        Placement::Mirrored3(Mirrored3Placement {
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

/// A specific helper function to be called from the computation when failing to parse an operator.
///
/// Defined here instead of a lambda to avoid leaking too much textual internals into the computation.
pub(crate) fn parse_operator_error<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    Err(Error(make_error(input, ErrorKind::Tag)))
}

/// Parses operator - maps names to structs.
fn parse_operator<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, Operator, E> {
    let (input, op_name) = ws(alphanumeric1)(input)?;
    Operator::get_from_textual(op_name)(input)
}

impl<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>> FromTextual<'a, E> for SendOp {
    fn from_textual(input: &'a str) -> IResult<&'a str, Operator, E> {
        let (input, _) = ws(tag("{"))(input)?;
        let (input, (_, _, rendezvous_key)) =
            tuple((tag("rendezvous_key"), ws(tag("=")), parse_hex_zero_padded))(input)?;
        let (input, _) = ws(tag(","))(input)?;
        let (input, (_, _, receiver)) = tuple((tag("receiver"), ws(tag("=")), string))(input)?;
        let (input, _) = ws(tag("}"))(input)?;
        let (input, sig) = operator_signature(1)(input)?;
        Ok((
            input,
            SendOp {
                sig,
                rendezvous_key: RendezvousKey::from_bytes(rendezvous_key),
                receiver: Role::from(receiver),
            }
            .into(),
        ))
    }
}

impl<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>> FromTextual<'a, E> for ReceiveOp {
    fn from_textual(input: &'a str) -> IResult<&'a str, Operator, E> {
        let (input, _) = ws(tag("{"))(input)?;
        let (input, (_, _, rendezvous_key)) =
            tuple((tag("rendezvous_key"), ws(tag("=")), parse_hex_zero_padded))(input)?;
        let (input, _) = ws(tag(","))(input)?;
        let (input, (_, _, sender)) = tuple((tag("sender"), ws(tag("=")), string))(input)?;
        let (input, _) = ws(tag("}"))(input)?;
        let (input, sig) = operator_signature(0)(input)?;
        Ok((
            input,
            ReceiveOp {
                sig,
                rendezvous_key: RendezvousKey::from_bytes(rendezvous_key),
                sender: Role::from(sender),
            }
            .into(),
        ))
    }
}

impl<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>> FromTextual<'a, E> for ExpandDimsOp {
    fn from_textual(input: &'a str) -> IResult<&'a str, Operator, E> {
        let (input, axis) = attributes_single("axis", vector(parse_int))(input)?;
        let (input, sig) = operator_signature(1)(input)?;
        Ok((input, ExpandDimsOp { sig, axis }.into()))
    }
}

impl<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>> FromTextual<'a, E> for DeriveSeedOp {
    fn from_textual(input: &'a str) -> IResult<&'a str, Operator, E> {
        let (input, sync_key) = attributes_single(
            "sync_key",
            alt((
                // Deprecated representation using vector of ints. Pre v0.1.5
                map_res(vector(parse_int), SyncKey::try_from),
                // Expected format
                map(parse_hex_zero_padded, SyncKey::from_bytes),
            )),
        )(input)
        .map_err(|_: nom::Err<nom::error::Error<&str>>| {
            Error(make_error(input, ErrorKind::MapRes))
        })?;
        let (input, opt_sig) = opt(operator_signature(0))(input)?;
        let sig = opt_sig.unwrap_or_else(|| Signature::nullary(Ty::HostSeed));
        Ok((input, DeriveSeedOp { sig, sync_key }.into()))
    }
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
pub(crate) fn attributes_single<'a, O, F: 'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
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
/// `: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor`
/// `: ([HostFloat32Tensor]) -> HostFloat32Tensor`
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
/// `(HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor`
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
/// `[HostFloat32Tensor] -> HostFloat32Tensor`
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
    let (i, inner) = opt(tuple((tag("<"), parse_tensor_dtype, tag(">"))))(i)?;
    let inner = inner.map(|t| t.1);
    let result = Ty::from_name(type_name, inner);
    match result {
        Some(ty) => Ok((i, ty)),
        _ => Err(Error(make_error(input, ErrorKind::Tag))),
    }
}

fn parse_tensor_dtype<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, TensorDType, E> {
    alt((
        value(TensorDType::Unknown, tag(TensorDType::Unknown.short_name())),
        value(TensorDType::Float32, tag(TensorDType::Float32.short_name())),
        value(TensorDType::Float64, tag(TensorDType::Float64.short_name())),
        value(TensorDType::Bool, tag(TensorDType::Bool.short_name())),
        preceded(
            tag(TensorDType::Fixed64 {
                integral_precision: 0,
                fractional_precision: 0,
            }
            .short_name()),
            map(
                tuple((tag("("), parse_int, ws(tag(",")), parse_int, tag(")"))),
                |t| TensorDType::Fixed64 {
                    integral_precision: t.1,
                    fractional_precision: t.3,
                },
            ),
        ),
        preceded(
            tag(TensorDType::Fixed128 {
                integral_precision: 0,
                fractional_precision: 0,
            }
            .short_name()),
            map(
                tuple((tag("("), parse_int, ws(tag(",")), parse_int, tag(")"))),
                |t| TensorDType::Fixed128 {
                    integral_precision: t.1,
                    fractional_precision: t.3,
                },
            ),
        ),
    ))(input)
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
        constant_literal_helper(Ty::HostSeed.short_name(), parse_hex, |v| {
            Constant::RawSeed(RawSeed(v))
        }),
        constant_literal_helper(Ty::HostPrfKey.short_name(), parse_hex, |v| {
            Constant::RawPrfKey(RawPrfKey(v))
        }),
        constant_literal_helper(Ty::Float32.short_name(), float, Constant::Float32),
        constant_literal_helper(Ty::Float64.short_name(), double, Constant::Float64),
        constant_literal_helper(Ty::HostString.short_name(), string, Constant::String),
        map(ws(string), Constant::String), // Alternative syntax for strings - no type
        constant_literal_helper(Ty::Ring64.short_name(), parse_int, Constant::Ring64),
        constant_literal_helper(Ty::Ring128.short_name(), parse_int, Constant::Ring128),
        constant_literal_helper(Ty::HostShape.short_name(), vector(parse_int), |v| {
            Constant::RawShape(RawShape(v))
        }),
        constant_literal_helper(Ty::Bit.short_name(), parse_int, Constant::Bit),
        // 1D arrays
        alt((
            constant_literal_helper(Ty::HostInt8Tensor.short_name(), vector(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostInt8Tensor(t)
            }),
            constant_literal_helper(Ty::HostInt16Tensor.short_name(), vector(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostInt16Tensor(t)
            }),
            constant_literal_helper(Ty::HostInt32Tensor.short_name(), vector(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostInt32Tensor(t)
            }),
            constant_literal_helper(Ty::HostInt64Tensor.short_name(), vector(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostInt64Tensor(t)
            }),
            constant_literal_helper(Ty::HostUint8Tensor.short_name(), vector(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostUint8Tensor(t)
            }),
            constant_literal_helper(Ty::HostUint16Tensor.short_name(), vector(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostUint16Tensor(t)
            }),
            constant_literal_helper(Ty::HostUint32Tensor.short_name(), vector(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostUint32Tensor(t)
            }),
            constant_literal_helper(Ty::HostUint64Tensor.short_name(), vector(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostUint64Tensor(t)
            }),
            constant_literal_helper(Ty::HostFloat32Tensor.short_name(), vector(float), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostFloat32Tensor(t)
            }),
            constant_literal_helper(Ty::HostFloat64Tensor.short_name(), vector(double), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostFloat64Tensor(t)
            }),
            constant_literal_helper(Ty::HostRing64Tensor.short_name(), vector(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostRing64Tensor(t)
            }),
            constant_literal_helper(Ty::HostRing128Tensor.short_name(), vector(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostRing128Tensor(t)
            }),
            constant_literal_helper(Ty::HostBitTensor.short_name(), vector(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostBitTensor(t)
            }),
        )),
        // 2D arrays
        alt((
            constant_literal_helper(Ty::HostInt8Tensor.short_name(), vector2(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostInt8Tensor(t)
            }),
            constant_literal_helper(Ty::HostInt16Tensor.short_name(), vector2(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostInt16Tensor(t)
            }),
            constant_literal_helper(Ty::HostInt32Tensor.short_name(), vector2(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostInt32Tensor(t)
            }),
            constant_literal_helper(Ty::HostInt64Tensor.short_name(), vector2(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostInt64Tensor(t)
            }),
            constant_literal_helper(Ty::HostUint8Tensor.short_name(), vector2(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostUint8Tensor(t)
            }),
            constant_literal_helper(Ty::HostUint16Tensor.short_name(), vector2(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostUint16Tensor(t)
            }),
            constant_literal_helper(Ty::HostUint32Tensor.short_name(), vector2(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostUint32Tensor(t)
            }),
            constant_literal_helper(Ty::HostUint64Tensor.short_name(), vector2(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostUint64Tensor(t)
            }),
            constant_literal_helper(Ty::HostFloat32Tensor.short_name(), vector2(float), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostFloat32Tensor(t)
            }),
            constant_literal_helper(Ty::HostFloat64Tensor.short_name(), vector2(double), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostFloat64Tensor(t)
            }),
            constant_literal_helper(
                Ty::HostRing64Tensor.short_name(),
                vector2(parse_int),
                |v: ndarray::ArrayD<u64>| {
                    let plc = HostPlacement::from("TODO");
                    let t = plc.from_raw(v);
                    Constant::HostRing64Tensor(t)
                },
            ),
            constant_literal_helper(
                Ty::HostRing128Tensor.short_name(),
                vector2(parse_int),
                |v: ndarray::ArrayD<u128>| {
                    let plc = HostPlacement::from("TODO");
                    let t = plc.from_raw(v);
                    Constant::HostRing128Tensor(t)
                },
            ),
            constant_literal_helper(Ty::HostBitTensor.short_name(), vector2(parse_int), |v| {
                let plc = HostPlacement::from("TODO");
                let t = plc.from_raw(v);
                Constant::HostBitTensor(t)
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
            tag(Ty::HostFixed64Tensor.short_name()),
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
                ndarray::Array::from(tensor).into_dyn().into_shared(),
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
            tag(Ty::HostFixed128Tensor.short_name()),
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
            tensor: placement.from_raw(tensor),
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

/// Parses a single byte, written as two hex character.
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

/// Parse sa hux dump, without any separators.
///
/// Leaves the tail zeroed if there is not enough data to fill an array of length N.
pub fn parse_hex_zero_padded<'a, E, const N: usize>(input: &'a str) -> IResult<&'a str, [u8; N], E>
where
    E: ParseError<&'a str>,
{
    let mut buf: [u8; N] = [0; N];
    let mut input = <&str>::clone(&input); // Explicitly cloning just the reference.
    for elem in buf.iter_mut() {
        match parse_hex_u8(input) {
            Ok((i, o)) => {
                *elem = o;
                input = i;
            }
            Err(Error(_)) => {
                return Ok((input, buf)); // Unlike `fill`, return the buffer if the child parser errored.
            }
            Err(e) => {
                return Err(e);
            }
        }
    }
    Ok((input, buf))
}

/// Wraps the inner parser in optional spaces.
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
fn identifier<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, &'a str, E> {
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
/// Note that it binds the E in the parser to be a `VerboseError`.
fn friendly_error(message: &str, source: &str, e: nom::Err<VerboseError<&str>>) -> anyhow::Error {
    if !source.contains('\n') {
        return anyhow::anyhow!(
            "{}. The input contains no line breaks, which usually indicates invalid format. {}",
            message,
            e
        );
    }
    match e {
        Failure(e) => anyhow::anyhow!("{}\n{}", message, convert_error(source, e)),
        Error(e) => anyhow::anyhow!("{}\n{}", message, convert_error(source, e)),
        _ => anyhow::anyhow!("{} due to {}", message, e),
    }
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
            Reshape(op) => op.to_textual(),
            Squeeze(op) => op.to_textual(),
            Transpose(op) => op.to_textual(),
            Dot(op) => op.to_textual(),
            Inverse(op) => op.to_textual(),
            Add(op) => op.to_textual(),
            Sub(op) => op.to_textual(),
            Mul(op) => op.to_textual(),
            Mean(op) => op.to_textual(),
            Sum(op) => op.to_textual(),
            Div(op) => op.to_textual(),
            Xor(op) => op.to_textual(),
            And(op) => op.to_textual(),
            Or(op) => op.to_textual(),
            Sqrt(op) => op.to_textual(),
            Diag(op) => op.to_textual(),
            ShlDim(op) => op.to_textual(),
            Sign(op) => op.to_textual(),
            RingFixedpointArgmax(op) => op.to_textual(),
            RingFixedpointEncode(op) => op.to_textual(),
            RingFixedpointDecode(op) => op.to_textual(),
            RingFixedpointMean(op) => op.to_textual(),
            Sample(op) => op.to_textual(),
            SampleSeeded(op) => op.to_textual(),
            Shl(op) => op.to_textual(),
            Shr(op) => op.to_textual(),
            RingInject(op) => op.to_textual(),
            BitExtract(op) => op.to_textual(),
            DeriveSeed(op) => op.to_textual(),
            PrfKeyGen(op) => op.to_textual(),
            Decrypt(op) => op.to_textual(),
            FixedpointEncode(op) => op.to_textual(),
            FixedpointDecode(op) => op.to_textual(),
            Share(op) => op.to_textual(),
            Reveal(op) => op.to_textual(),
            AddN(op) => op.to_textual(),
            TruncPr(op) => op.to_textual(),
            AdtToRep(op) => op.to_textual(),
            Abs(op) => op.to_textual(),
            Fill(op) => op.to_textual(),
            Msb(op) => op.to_textual(),
            RepToAdt(op) => op.to_textual(),
            Index(op) => op.to_textual(),
            BitDecompose(op) => op.to_textual(),
            BitCompose(op) => op.to_textual(),
            Mux(op) => op.to_textual(),
            Neg(op) => op.to_textual(),
            Pow2(op) => op.to_textual(),
            Exp(op) => op.to_textual(),
            Sigmoid(op) => op.to_textual(),
            Log2(op) => op.to_textual(),
            Log(op) => op.to_textual(),
            Equal(op) => op.to_textual(),
            EqualZero(op) => op.to_textual(),
            LessThan(op) => op.to_textual(),
            GreaterThan(op) => op.to_textual(),
            Demirror(op) => op.to_textual(),
            Mirror(op) => op.to_textual(),
            Maximum(op) => op.to_textual(),
            Argmax(op) => op.to_textual(),
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
op_with_axis_to_textual!(SqueezeOp);

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

impl ToTextual for SampleOp {
    fn to_textual(&self) -> String {
        match self {
            SampleOp {
                sig,
                max_value: Some(a),
            } => format!("Sample{{max_value = {}}}: {}", a, sig.to_textual()),
            SampleOp {
                sig,
                max_value: None,
            } => format!("Sample{{}}: {}", sig.to_textual()),
        }
    }
}

impl ToTextual for SampleSeededOp {
    fn to_textual(&self) -> String {
        match self {
            SampleSeededOp {
                sig,
                max_value: Some(a),
            } => format!("SampleSeeded{{max_value = {}}}: {}", a, sig.to_textual()),
            SampleSeededOp {
                sig,
                max_value: None,
            } => format!("SampleSeeded{{}}: {}", sig.to_textual()),
        }
    }
}

impl ToTextual for Ty {
    fn to_textual(&self) -> String {
        match self {
            Ty::Tensor(inner) => format!("{}<{}>", self.short_name(), inner.to_textual()),
            _ => self.short_name().to_string(),
        }
    }
}

impl ToTextual for TensorDType {
    fn to_textual(&self) -> String {
        match self {
            TensorDType::Fixed64 {
                integral_precision,
                fractional_precision,
            } => format!(
                "{}({}, {})",
                self.short_name(),
                integral_precision,
                fractional_precision
            ),
            TensorDType::Fixed128 {
                integral_precision,
                fractional_precision,
            } => format!(
                "{}({}, {})",
                self.short_name(),
                integral_precision,
                fractional_precision
            ),
            _ => self.short_name().to_string(),
        }
    }
}

macro_rules! format_to_textual {
    ($format:expr, $ty:expr, $($member:expr),*) => {
        format!($format, $ty.short_name(), $($member.to_textual(),)*)
    };
}

impl ToTextual for Value {
    fn to_textual(&self) -> String {
        match self {
            Value::HostInt8Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostInt16Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostInt32Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostInt64Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostUint8Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostUint16Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostUint32Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostUint64Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostFloat32Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostFloat64Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostRing64Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostRing128Tensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            // TODO: Hosted floats for values
            Value::Float32(x) => format!("{}({}) @Host(TODO)", self.ty().short_name(), x),
            Value::Float64(x) => format!("{}({}) @Host(TODO)", self.ty().short_name(), x),
            Value::Fixed(x) => format!("{}[{}]({})", self.ty().short_name(), x.precision, x.value),
            Value::HostString(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::Ring64(x) => format!("{}({})", self.ty().short_name(), x),
            Value::Ring128(x) => format!("{}({})", self.ty().short_name(), x),
            Value::HostShape(x) => format!(
                "{}({:?}) {}",
                self.ty().short_name(),
                x.0 .0,
                x.1.to_textual()
            ),
            Value::HostSeed(x) => format_to_textual!("{}({}) {}", self.ty(), x.0 .0, x.1),
            Value::HostPrfKey(x) => format_to_textual!("{}({}) {}", self.ty(), x.0 .0, x.1),
            Value::Bit(x) => format!("{}({})", self.ty().short_name(), x),
            Value::HostUnit(_) => self.ty().short_name().to_string(),
            Value::HostBitTensor(x) => format_to_textual!("{}({}) {}", self.ty(), x.0, x.1),
            Value::HostFixed64Tensor(x) => format_to_textual!(
                "{}[{}/{}]({}) {}",
                self.ty(),
                x.integral_precision,
                x.fractional_precision,
                x.tensor.0,
                x.tensor.1
            ),
            Value::HostFixed128Tensor(x) => format_to_textual!(
                "{}[{}/{}]({}) {}",
                self.ty(),
                x.integral_precision,
                x.fractional_precision,
                x.tensor.0,
                x.tensor.1
            ),
            Value::Tensor(_)
            | Value::TensorShape(_)
            | Value::HostBitArray64(_)
            | Value::HostBitArray128(_)
            | Value::HostBitArray224(_)
            | Value::HostBitArray256(_) => {
                unimplemented!()
            }
            // The following value variants live in the replicated form and can not be represented in the textual computation graph.
            Value::Fixed64Tensor(_)
            | Value::Fixed128Tensor(_)
            | Value::BooleanTensor(_)
            | Value::Float32Tensor(_)
            | Value::Float64Tensor(_)
            | Value::Uint64Tensor(_)
            | Value::ReplicatedShape(_)
            | Value::ReplicatedBitTensor(_)
            | Value::ReplicatedBitArray64(_)
            | Value::ReplicatedBitArray128(_)
            | Value::ReplicatedBitArray224(_)
            | Value::ReplicatedRing64Tensor(_)
            | Value::ReplicatedRing128Tensor(_)
            | Value::ReplicatedFixed64Tensor(_)
            | Value::ReplicatedFixed128Tensor(_)
            | Value::ReplicatedUint64Tensor(_)
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
            Constant::HostInt8Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::HostInt16Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::HostInt32Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::HostInt64Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::HostUint8Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::HostUint16Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::HostUint32Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::HostUint64Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::HostFloat32Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::HostFloat64Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::HostRing64Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::HostRing128Tensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
            Constant::Float32(x) => format!("{}({})", self.ty().short_name(), x),
            Constant::Float64(x) => format!("{}({})", self.ty().short_name(), x),
            Constant::String(x) => format!("{}({})", self.ty().short_name(), x.to_textual()),
            Constant::Ring64(x) => format!("{}({})", self.ty().short_name(), x),
            Constant::Ring128(x) => format!("{}({})", self.ty().short_name(), x),
            Constant::Fixed(FixedpointConstant { value, precision }) => {
                format!("{}({}, {})", self.ty().short_name(), value, precision)
            }
            Constant::RawShape(RawShape(x)) => format!("Shape({:?})", x),
            Constant::RawSeed(RawSeed(x)) => format!("HostSeed({})", x.to_textual()),
            Constant::RawPrfKey(RawPrfKey(x)) => format!("HostPrfKey({})", x.to_textual()),
            Constant::Bit(x) => format!("{}({})", self.ty().short_name(), x),
            Constant::HostBitTensor(x) => {
                format!("{}({})", self.ty().short_name(), x.0.to_textual())
            }
        }
    }
}

impl<T: std::fmt::Debug> ToTextual for ArcArrayD<T> {
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
            _ => unimplemented!("ArcArrayD.to_textual() unimplemented for tensors of rank > 3"),
        }
    }
}

impl ToTextual for Role {
    fn to_textual(&self) -> String {
        format!("{:?}", self.0)
    }
}

// Required to serialize DeriveSeedOp
impl ToTextual for SyncKey {
    fn to_textual(&self) -> String {
        self.as_bytes().to_textual()
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

impl ToTextual for crate::host::BitArrayRepr {
    fn to_textual(&self) -> String {
        self.into_array::<u8>().unwrap().into_shared().to_textual()
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
use_debug_to_textual!(Vec<usize>);
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
    use crate::host::FromRaw;
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
        let (_, parsed_str) = constant_literal::<(&str, ErrorKind)>("HostString(\"abc\")")?;
        assert_eq!(parsed_str, Constant::String("abc".into()));
        let (_, parsed_str) = constant_literal::<(&str, ErrorKind)>("\"1.23\"")?;
        assert_eq!(parsed_str, Constant::String("1.23".into()));
        let (_, parsed_str) = constant_literal::<(&str, ErrorKind)>("\"1. 2\\\"3\"")?;
        assert_eq!(parsed_str, Constant::String("1. 2\"3".into()));
        let (_, parsed_ring64_tensor) =
            constant_literal::<(&str, ErrorKind)>("HostRing64Tensor([1,2,3])")?;
        let plc = HostPlacement::from("TODO");
        assert_eq!(
            parsed_ring64_tensor,
            Constant::HostRing64Tensor(plc.from_raw(vec![1, 2, 3]))
        );
        let (_, parsed_ring128_tensor) =
            constant_literal::<(&str, ErrorKind)>("HostRing128Tensor([1,2,3])")?;
        assert_eq!(
            parsed_ring128_tensor,
            Constant::HostRing128Tensor(plc.from_raw(vec![1, 2, 3]))
        );
        let (_, parsed_shape) = constant_literal::<(&str, ErrorKind)>("HostShape([1,2,3])")?;
        assert_eq!(parsed_shape, Constant::RawShape(RawShape(vec![1, 2, 3])));
        let (_, parsed_u8_tensor) =
            constant_literal::<(&str, ErrorKind)>("HostUint8Tensor([1,2,3])")?;
        assert_eq!(
            parsed_u8_tensor,
            Constant::HostUint8Tensor(plc.from_raw(vec![1, 2, 3]))
        );
        let (_, parsed_seed) =
            constant_literal::<(&str, ErrorKind)>("HostSeed(529c2fc9bf573d077f45f42b19cfb8d4)")?;
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
            "HostFloat32Tensor([[1.0, 11, 12, 13], [3.0, 21, 22, 23]])".try_into()?;
        assert_eq!(
            parsed_f32.to_textual(),
            "HostFloat32Tensor([[1.0, 11.0, 12.0, 13.0], [3.0, 21.0, 22.0, 23.0]])"
        );
        Ok(())
    }

    #[test]
    fn test_array_literal() -> Result<(), anyhow::Error> {
        use ndarray::prelude::*;
        use std::convert::TryInto;

        let plc = HostPlacement::from("TODO");

        let parsed_f32: Constant = "HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])".try_into()?;
        let x = plc.from_raw(array![[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(parsed_f32, Constant::HostFloat32Tensor(x));

        let parsed_ring64: Constant = "HostRing64Tensor([[1, 2], [3, 4]])".try_into()?;
        let x = plc.from_raw(array![[1, 2], [3, 4]]);
        assert_eq!(parsed_ring64, Constant::HostRing64Tensor(x));

        Ok(())
    }

    #[test]
    fn test_type_parsing() -> Result<(), anyhow::Error> {
        let (_, parsed_type) = parse_type::<(&str, ErrorKind)>("HostUnit")?;
        assert_eq!(parsed_type, Ty::HostUnit);
        let (_, parsed) = operator_signature::<(&str, ErrorKind)>(0)(
            ": (HostFloat32Tensor, HostFloat64Tensor) -> HostUint16Tensor",
        )?;
        assert_eq!(
            parsed,
            Signature::binary(
                Ty::HostFloat32Tensor,
                Ty::HostFloat64Tensor,
                Ty::HostUint16Tensor
            ),
        );

        let (_, parsed) = operator_signature::<(&str, ErrorKind)>(0)(
            ": [HostFloat32Tensor] -> HostFloat32Tensor",
        )?;
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
        let (_, parsed_type) = parse_type::<(&str, ErrorKind)>("Tensor<Float64>")?;
        assert_eq!(parsed_type, Ty::Tensor(TensorDType::Float64));
        Ok(())
    }

    #[test]
    fn test_constant() -> Result<(), anyhow::Error> {
        let host = HostPlacement::from("TODO");

        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "x = Constant{value = HostFloat32Tensor([1.0])}: () -> HostFloat32Tensor () @Host(alice)",
        )?;
        assert_eq!(op.name, "x");
        assert_eq!(
            op.kind,
            Operator::Constant(ConstantOp {
                sig: Signature::nullary(Ty::HostFloat32Tensor),
                value: Constant::HostFloat32Tensor(host.from_raw(vec![1.0]))
            })
        );

        // 2D tensor
        use ndarray::prelude::*;
        let x = host.from_raw(array![[1.0, 2.0], [3.0, 4.0]]);
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "x = Constant{value = HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor () @Replicated(alice, bob, charlie)",
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
            "z = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(carole)",
        )?;
        assert_eq!(op.name, "z");
        assert_eq!(
            op.kind,
            Operator::Add(AddOp {
                sig: Signature::binary(
                    Ty::HostFloat32Tensor,
                    Ty::HostFloat32Tensor,
                    Ty::HostFloat32Tensor
                ),
            })
        );
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(carole)",
        )?;
        assert_eq!(op.name, "z");
        assert_eq!(
            op.kind,
            Operator::Mul(MulOp {
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
            "key = PrfKeyGen: () -> HostPrfKey () @Host(alice)",
        )?;
        assert_eq!(op.name, "key");
        Ok(())
    }

    #[test]
    fn test_seed() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "seed = DeriveSeed{sync_key = [1, 2, 3]}(key)@Host(alice)",
        )?;
        assert_eq!(op.name, "seed");
        assert_eq!(
            op.kind,
            Operator::DeriveSeed(DeriveSeedOp {
                sig: Signature::nullary(Ty::HostSeed),
                sync_key: SyncKey::try_from(vec![1, 2, 3])?
            })
        );
        Ok(())
    }

    #[test]
    fn test_seed_hex() -> Result<(), anyhow::Error> {
        let source =
            "seed = DeriveSeed{sync_key = 01020300000000000000000000000000}: () -> HostSeed (key) @Host(alice)";
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(source)?;
        assert_eq!(op.name, "seed");
        assert_eq!(
            op.kind,
            Operator::DeriveSeed(DeriveSeedOp {
                sig: Signature::nullary(Ty::HostSeed),
                sync_key: SyncKey::try_from(vec![1, 2, 3])?
            })
        );
        // Verify that it serializes back to the same format
        assert_eq!(source, op.to_textual());
        // Verify the shorthand format
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "seed = DeriveSeed{sync_key = 010203}(key)@Host(alice)",
        )?;
        assert_eq!(source, op.to_textual());
        Ok(())
    }

    #[test]
    fn test_send() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            r#"send = Send{rendezvous_key = 30313233343536373839616263646566, receiver = "bob"}: (HostFloat32Tensor) -> HostUnit() @Host(alice)"#,
        )?;
        assert_eq!(op.name, "send");
        assert_eq!(
            op.kind,
            Operator::Send(SendOp {
                sig: Signature::unary(Ty::HostFloat32Tensor, Ty::HostUnit),
                rendezvous_key: "0123456789abcdef".try_into()?,
                receiver: Role::from("bob")
            })
        );
        Ok(())
    }

    #[test]
    fn test_send_shortened() -> Result<(), anyhow::Error> {
        // Test that rendezvous_key can be specified without the tailing zeroes.
        let (_, op1) = parse_assignment::<(&str, ErrorKind)>(
            r#"send = Send{rendezvous_key = 179704, receiver = "bob"}: (HostFloat32Tensor) -> Unit() @Host(alice)"#,
        )?;
        let (_, op2) = parse_assignment::<(&str, ErrorKind)>(
            r#"send = Send{rendezvous_key = 17970400000000000000000000000000, receiver = "bob"}: (HostFloat32Tensor) -> Unit() @Host(alice)"#,
        )?;
        let rendezvous_key1 = match op1.kind {
            Operator::Send(SendOp { rendezvous_key, .. }) => rendezvous_key,
            _ => panic!("Incorrect op type parsed"),
        };
        let rendezvous_key2 = match op2.kind {
            Operator::Send(SendOp { rendezvous_key, .. }) => rendezvous_key,
            _ => panic!("Incorrect op type parsed"),
        };
        assert_eq!(rendezvous_key1, rendezvous_key2);
        Ok(())
    }

    #[test]
    fn test_receive() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            r#"receive = Receive{rendezvous_key = 30313233343536373839616263646566, sender = "bob"} : () -> HostFloat32Tensor () @Host(alice)"#,
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
            "z = Output: (HostRing64Tensor) -> HostRing64Tensor (x10) @Host(alice)",
        )?;
        assert_eq!(op.name, "z");
        Ok(())
    }

    #[test]
    fn test_ring_sample() -> Result<(), anyhow::Error> {
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "x10 = SampleSeeded{max_value = 1}: (HostShape, HostSeed) -> HostRing64Tensor (shape, seed) @Host(alice)",
        )?;
        assert_eq!(op.name, "x10");
        Ok(())
    }

    #[test]
    fn test_slice_option() -> Result<(), anyhow::Error> {
        let input = "x10 = Slice{slice = {start = 1, end = 10, step = -1}}: (HostRing64Tensor) -> HostRing64Tensor (x) @Host(alice)";
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(input)?;
        assert_eq!(op.name, "x10");
        assert_eq!(
            op.kind,
            Operator::Slice(SliceOp {
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
        let input = "x10 = Slice{slice = {start = 1, end = 10, step = 1}}: (HostRing64Tensor) -> HostRing64Tensor (x) @Host(alice)";
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(input)?;
        assert_eq!(op.name, "x10");
        assert_eq!(
            op.kind,
            Operator::Slice(SliceOp {
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
            "op = RingFixedpointMean{axis = 0, scaling_base = 3, scaling_exp = 1} : () -> HostFloat32Tensor () @Host(alice)",
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
            "op = RingFixedpointMean{scaling_base = 3, scaling_exp = 1} : () -> HostFloat32Tensor () @Host(alice)",
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
            "x_shape = Constant{value = HostShape([2, 2])}: () -> HostShape () @Host(alice)",
        )?;
        assert_eq!(op.name, "x_shape");
        let (_, op) = parse_assignment::<(&str, ErrorKind)>(
            "z_result = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x_shape, y_shape) @Host(carole)",
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
            r#"z = Input{arg_name = "prompt"}: () -> HostFloat32Tensor () @Host(alice)"#,
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = ExpandDims {axis = [0]}: (HostFloat32Tensor) -> HostFloat32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = AtLeast2D {to_column_vector = false}: (HostFloat32Tensor) -> HostFloat32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = Slice {slice = {start = 1, end = 2}}: (HostFloat32Tensor) -> HostFloat32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = Diag: (HostFloat32Tensor) -> HostFloat32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = Sqrt: (HostFloat32Tensor) -> HostFloat32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = Fill {value = Ring64(42)}: (HostShape) -> HostRing64Tensor (s) @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = Shl {amount = 2}: (HostFloat32Tensor) -> HostFloat32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = Shr {amount = 2}: (HostFloat32Tensor) -> HostFloat32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = RingFixedpointDecode {scaling_base = 3, scaling_exp = 2}: (HostFloat32Tensor) -> HostFloat32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = RingFixedpointEncode {scaling_base = 3, scaling_exp = 2}: (HostFloat32Tensor) -> HostFloat32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = RingInject {bit_idx = 2} : (HostFloat32Tensor) -> HostFloat32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = BitExtract {bit_idx = 2} : (HostFloat32Tensor) -> HostFloat32Tensor () @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = SampleSeeded {}: (HostShape, HostSeed) -> HostBitTensor (shape, seed) @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "z = Xor: (HostBitTensor, HostBitTensor) -> HostBitTensor (x, y) @Host(alice)",
        )?;

        parse_assignment::<(&str, ErrorKind)>(
            "load = Load: (HostString, HostString) -> HostFloat64Tensor (xuri, xconstant) @Host(alice)",
        )?;
        parse_assignment::<(&str, ErrorKind)>(
            "addN = AddN: [HostString] -> HostString (xuri, xconstant) @Host(alice)",
        )?;

        Ok(())
    }

    #[test]
    fn test_sample_computation() -> Result<(), anyhow::Error> {
        let host = HostPlacement::from("TODO");

        let (_, comp) = parse_computation::<(&str, ErrorKind)>(
            "x = Constant{value = HostFloat32Tensor([1.0])}: () -> HostFloat32Tensor() @Host(alice)
            y = Constant{value = HostFloat32Tensor([2.0])}: () -> HostFloat32Tensor () @Host(bob)
            // ignore = Constant([1.0]: HostFloat32Tensor) @Host(alice)
            z = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(carole)
            ",
        )?;
        assert_eq!(comp.operations.len(), 3);
        assert_eq!(
            comp.operations[0].kind,
            Operator::Constant(ConstantOp {
                sig: Signature::nullary(Ty::HostFloat32Tensor),
                value: Constant::HostFloat32Tensor(host.from_raw(vec![1.0]))
            })
        );
        assert_eq!(
            comp.operations[1].kind,
            Operator::Constant(ConstantOp {
                sig: Signature::nullary(Ty::HostFloat32Tensor),
                value: Constant::HostFloat32Tensor(host.from_raw(vec![2.0]))
            })
        );
        assert_eq!(comp.operations[2].name, "z");
        assert_eq!(
            comp.operations[2].kind,
            Operator::Add(AddOp {
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
        let data = r#"a = Constant{value = "a"}: () -> HostFloat32Tensor () @Host(alice)
            err = Add: (HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(carole)
            b = Constant{value = "b"}: () -> HostFloat32Tensor () @Host(alice)"#;
        let emsg = r#"0: at line 2, in Tag:
            err = Add: (HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(carole)
                            ^

1: at line 2, in Alt:
            err = Add: (HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(carole)
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
            "x = Constant{value = HostFloat32Tensor([1.0])}: () -> HostFloat32Tensor @Host(alice)
            y = Constant{value = HostFloat32Tensor([2.0])}: () -> HostFloat32Tensor () @Host(bob)
            z = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(carole)"
                .try_into()?;
        assert_eq!(comp.operations.len(), 3);
        Ok(())
    }

    #[test]
    fn test_verbose_parse_computation() -> Result<(), anyhow::Error> {
        let comp: Computation = verbose_parse_computation("x = Constant{value = HostFloat32Tensor([1.0])}: () -> HostFloat32Tensor @Host(alice)
            y = Constant{value = HostFloat32Tensor([2.0])}: () -> HostFloat32Tensor () @Host(bob)
            z = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(carole)"
                )?;
        assert_eq!(comp.operations.len(), 3);
        Ok(())
    }

    #[test]
    fn test_fast_parse_computation() -> Result<(), anyhow::Error> {
        let comp: Computation = fast_parse_computation("x = Constant{value = HostFloat32Tensor([1.0])}: () -> HostFloat32Tensor @Host(alice)
            y = Constant{value = HostFloat32Tensor([2.0])}: () -> HostFloat32Tensor () @Host(bob)
            z = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(carole)"
                )?;
        assert_eq!(comp.operations.len(), 3);
        Ok(())
    }

    #[test]
    fn test_parallel_parse_computation() -> Result<(), anyhow::Error> {
        let comp: Computation = parallel_parse_computation("x = Constant{value = HostFloat32Tensor([1.0])}: () -> HostFloat32Tensor @Host(alice)
            y = Constant{value = HostFloat32Tensor([2.0])}: () -> HostFloat32Tensor () @Host(bob)
            z = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(carole)", 3)?;
        assert_eq!(comp.operations.len(), 3);
        Ok(())
    }

    #[test]
    fn test_constant_try_into() -> Result<(), anyhow::Error> {
        let host = HostPlacement::from("TODO");
        let v: Constant = "HostFloat32Tensor([1.0, 2.0, 3.0])".try_into()?;
        assert_eq!(
            v,
            Constant::HostFloat32Tensor(host.from_raw(vec![1.0, 2.0, 3.0]))
        );
        Ok(())
    }

    #[rstest]
    #[case("HostInt8Tensor([2, 3]) @Host(alice)")]
    #[case("HostInt16Tensor([2, 3]) @Host(alice)")]
    #[case("HostInt32Tensor([2, 3]) @Host(alice)")]
    #[case("HostInt64Tensor([2, 3]) @Host(alice)")]
    #[case("HostUint8Tensor([2, 3]) @Host(alice)")]
    #[case("HostUint16Tensor([2, 3]) @Host(alice)")]
    #[case("HostUint32Tensor([2, 3]) @Host(alice)")]
    #[case("HostUint64Tensor([2, 3]) @Host(alice)")]
    #[case("HostFloat32Tensor([2.1, 3.2]) @Host(alice)")]
    #[case("HostFloat64Tensor([2.1, 3.2]) @Host(alice)")]
    #[case("HostRing64Tensor([2, 3]) @Host(alice)")]
    #[case("HostRing128Tensor([2, 3]) @Host(alice)")]
    #[case("Float32(2.1) @Host(TODO)")]
    #[case("Float64(2.1) @Host(TODO)")]
    #[case("HostString(\"hi\") @Host(alice)")]
    #[case("HostBitTensor([0, 1, 1, 0]) @Host(alice)")]
    #[case("HostShape([3, 2]) @Host(alice)")]
    #[case("HostSeed(529c2fc9bf573d077f45f42b19cfb8d4) @Host(alice)")]
    #[case("HostPrfKey(00000000000000000000000000000000) @Host(alice)")]
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
        x = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)

        y = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(bob)

        "#;
        let comp: Computation = source.try_into()?;
        assert_eq!(comp.operations.len(), 2);
        Ok(())
    }

    #[test]
    fn test_computation_into_text() -> Result<(), anyhow::Error> {
        use std::convert::TryInto;
        let comp: Computation = "x = Constant{value = HostFloat32Tensor([1.0])}: () -> HostFloat32Tensor @Host(alice)
            y = Constant{value = HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(bob)
            z = Add: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Replicated(alice, bob, carole)
            seed = DeriveSeed{sync_key = 010203} (key) @Host(alice)
            seed2 = Constant{value = HostSeed(529c2fc9bf573d077f45f42b19cfb8d4)}: () -> HostSeed @Host(alice)
            o = Output: (HostFloat32Tensor) -> HostFloat32Tensor (z) @Host(alice)"
            .try_into()?;
        let textual = comp.to_textual();
        // After serializing it into the textual IR we need to make sure it parses back the same
        let comp2: Computation = textual.try_into()?;
        assert_eq!(comp.operations, comp2.operations);
        Ok(())
    }

    #[test]
    fn test_high_level_computation_into_text() -> Result<(), anyhow::Error> {
        use std::convert::TryInto;
        let comp: Computation = r#"constant_0 = Constant{value = HostFloat64Tensor([[0.12131529]])}: () -> Tensor<Float64> () @Host(player2)
        cast_0 = Cast: (Tensor<Float64>) -> Tensor<Fixed128(24, 40)> (constant_0) @Host(player2)
        x = Input{arg_name = "x"}: () -> AesTensor () @Host(player0)
        key = Input{arg_name = "key"}: () -> AesKey () @Replicated(player0, player1, player2)
        decrypt_0 = Decrypt: (AesKey, AesTensor) -> Tensor<Fixed128(24, 40)> (key, x) @Replicated(player0, player1, player2)
        dot_0 = Dot: (Tensor<Fixed128(24, 40)>, Tensor<Fixed128(24, 40)>) -> Tensor<Fixed128(24, 40)> (decrypt_0, cast_0) @Replicated(player0, player1, player2)
        cast_1 = Cast: (Tensor<Fixed128(24, 40)>) -> Tensor<Float64> (dot_0) @Host(player1)
        output_0 = Output: (Tensor<Float64>) -> Tensor<Float64> (cast_1) @Host(player1)"#.try_into()?;
        let textual = comp.to_textual();
        // After serializing it into the textual IR we need to make sure it parses back the same
        let comp2: Computation = textual.try_into()?;
        assert_eq!(comp.operations, comp2.operations);
        Ok(())
    }
}
