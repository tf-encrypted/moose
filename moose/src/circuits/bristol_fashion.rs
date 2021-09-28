//! This module contains code for loading Bistrol Fashion circuits as (partial) computations.

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{newline, space0, u64};
use nom::combinator::value;
use nom::error::Error;
use nom::multi::{length_count, many0, many_m_n, separated_list0};
use nom::sequence::{delimited, terminated, tuple};
use std::io::{self, prelude::*, BufReader};

#[derive(Debug)]
pub struct Circuit {
    number_of_gates: usize,
    number_of_wires: usize,
    gates: Vec<Gate>,
}

#[derive(Debug)]
pub struct Gate {
    kind: GateKind,
    input_wires: Vec<usize>,  // TODO could use small_vec here
    output_wires: Vec<usize>, // TODO could use small_vec here
}

#[derive(Clone, Debug)]
pub enum GateKind {
    Xor,
    And,
    Inv,
}

type Res<T, U> = nom::IResult<T, U, Error<T>>;

fn parse_circuit(bytes: &[u8]) -> Res<&[u8], Circuit> {
    // First line is just two usize values
    let (bytes, (number_of_gates, number_of_wires)) =
        terminated(tuple((parse_usize, parse_usize)), newline)(bytes)?;
    // Next two lines contain a count of inputs/outputs followed by its indexes
    let (bytes, _inputs) = terminated(length_count(parse_usize, parse_usize), newline)(bytes)?;
    let (bytes, _outputs) = terminated(length_count(parse_usize, parse_usize), newline)(bytes)?;
    // Some empty line in the input. I wonder if we should allow more empty lines or make it optional (lvorona)
    let (bytes, _) = terminated(many0(parse_usize), newline)(bytes)?;
    // The rest of the file is gate definitions
    let (bytes, gates) = separated_list0(newline, parse_gate)(bytes)?;

    Ok((
        bytes,
        Circuit {
            number_of_gates,
            number_of_wires,
            gates,
        },
    ))
}

fn parse_usize(line: &[u8]) -> Res<&[u8], usize> {
    let (line, res) = delimited(space0, u64, space0)(line)?;
    Ok((line, res as usize))
}

fn parse_gate(line: &[u8]) -> Res<&[u8], Gate> {
    let (line, inputs_count) = parse_usize(line)?;
    let (line, outputs_count) = parse_usize(line)?;
    let (line, input_wires) = many_m_n(inputs_count, inputs_count, parse_usize)(line)?;
    let (line, output_wires) = many_m_n(outputs_count, outputs_count, parse_usize)(line)?;
    let (line, kind) = delimited(space0, parse_kind, space0)(line)?;
    Ok((
        line,
        Gate {
            kind,
            input_wires,
            output_wires,
        },
    ))
}

fn parse_kind(line: &[u8]) -> Res<&[u8], GateKind> {
    alt((
        value(GateKind::Xor, tag("XOR")),
        value(GateKind::And, tag("AND")),
        value(GateKind::Inv, tag("INV")),
    ))(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_aes() {
        let circuit = parse_circuit(crate::circuits::AES_128);
        println!("Parsed circuit: {:?}", circuit);
    }

    #[test]
    fn test_parse_gate() {
        let gate = parse_gate("2 1 33280 33282 3691 XOR".as_bytes());
        println!("Parsed gate: {:?}", gate);
    }
}
