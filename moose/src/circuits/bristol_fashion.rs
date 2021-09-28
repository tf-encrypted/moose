//! This module contains code for loading Bistrol Fashion circuits as (partial) computations.

use std::io::{self, prelude::*, BufReader};
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::error::Error;
use nom::multi::many_m_n;
use nom::character::complete::{u64};
use nom::combinator::value;

use crate::text_computation::ws;

pub struct Circuit {
    number_of_gates: usize,
    number_of_wires: usize,
    gates: Vec<Gate>,
}


#[derive(Debug)]
pub struct Gate {
    kind: GateKind,
    input_wires: Vec<usize>, // TODO could use small_vec here
    output_wires: Vec<usize>, // TODO could use small_vec here
}

#[derive(Clone, Debug)]
pub enum GateKind {
    Xor,
    And,
    Inv,
}

fn parse_circuit(bytes: &[u8]) -> io::Result<()> {
    let mut reader = BufReader::new(bytes);
    // let mut buffer = String::new();
    // reader.read_line(&mut buffer)?;
    // let ngates_nwires = buffer.clone();
    // buffer.clear();
    // reader.read_line(&mut buffer)?;
    // let inputs = buffer.clone();
    // buffer.clear();
    // reader.read_line(&mut buffer)?;
    // let outputs = buffer;

    let lines = reader.lines().collect::<io::Result<Vec<_>>>()?;
    let ngates_nwires = lines.get(0).unwrap();
    let inputs = lines.get(1).unwrap();
    let outputs = lines.get(2).unwrap();

    let gates: Vec<Gate> = lines.iter().skip(3).map(|s| {
        parse_gate(s).unwrap().1
    }).collect();
    // let txt = std::str::from_utf8(bytes).unwrap();
    // let parser = separated_list0(newline, parse_gate);
    // let _ = parser(txt);

    Ok(())
}

type Res<T, U> = nom::IResult<T, U, Error<T>>;

fn parse_usize(line: &str) -> Res<&str, usize> {
    let (line, res) = ws(u64)(line)?;
    Ok((line, res as usize))
}

fn parse_gate(line: &str) -> Res<&str, Gate> {
    let (line, inputs_count) = parse_usize(line)?;
    let (line, outputs_count) = parse_usize(line)?;
    let (line, input_wires) = many_m_n(inputs_count, inputs_count, parse_usize)(line)?;
    let (line, output_wires) = many_m_n(outputs_count, outputs_count, parse_usize)(line)?;
    let (line, kind) = ws(parse_kind)(line)?;
    Ok((line, Gate {
        kind,
        input_wires,
        output_wires
    }))
}

fn parse_kind(line: &str) -> Res<&str, GateKind> {
    alt((
        value(GateKind::Xor,  tag("XOR")),
        value(GateKind::And,  tag("AND")),
        value(GateKind::Inv,  tag("INV")),
    ))(line)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_aes() {
        parse_circuit(crate::circuits::AES_128);
    }

    #[test]
    fn test_parse_gate() {
        let gate = parse_gate("2 1 33280 33282 3691 XOR");
        println!("Parsed gate: {:?}", gate);
    }
}
