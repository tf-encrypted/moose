//! This module contains code for loading Bistrol Fashion circuits as (partial) computations.

use std::io::{self, prelude::*, BufReader};
use nom::*;
use nom::error::{ContextError, ParseError, Error, VerboseError};
use nom::multi::separated_list0;
use nom::character::complete::{newline, space0, digit0};
use nom::number::complete::{be_u32};
use nom::sequence::pair;
use nom::combinator::{map, map_res};

pub struct Circuit {
    number_of_gates: usize,
    number_of_wires: usize,
    gates: Vec<Gate>,
}

pub struct Gate {
    kind: GateKind,
    input_wires: Vec<usize>, // TODO could use small_vec here
    output_wires: Vec<usize>, // TODO could use small_vec here
}

pub enum GateKind {
    Xor,
    And,
    Inv,
}

fn parse_circuit(bytes: &[u8]) -> io::Result<()> {
    // let reader = BufReader::new(bytes);

    // let lines = reader.lines().collect::<io::Result<Vec<_>>>()?;
    // let ngates_nwires = lines.get(0).unwrap();
    // let inputs = lines.get(1).unwrap();
    // let outputs = lines.get(2).unwrap();

    // Ok(())

    // let txt = std::str::from_utf8(bytes).unwrap();
    // let parser = separated_list0(newline, parse_gate);
    // let _ = parser(txt);

    Ok(())
}

type Res<T, U> = IResult<T, U, Error<T>>;

fn parse_gate(line: &str) -> Res<&str, Option<Gate>> {
    let first_parser = separated_list0(space0, usize_parser);
    let second_parser = parse_kind;
    let combined_parser = map_res(pair(first_parser, second_parser),
        |(x, y)| {
            Ok(None)
        });
    combined_parser(line)
    // let parser = map(
    //     combined_parser,
    //     |(values, kind)| {
    //         let num_inputs = *values.get(0).unwrap();
    //         let num_outputs = *values.get(1).unwrap();
    //         assert_eq!(num_inputs + num_outputs, values.len());
    //         values
    //     });
    // parser(line)
    // Ok((line, None))
}

fn parse_kind(line: &str) -> Res<&str, Option<GateKind>> {
    unimplemented!()
}

use std::str::FromStr;

named!(usize_parser<&str, usize>,
    map_res!(
        recognize!(tuple!(opt!(char!('-')), digit0)),
        FromStr::from_str)
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_aes() {
        parse_circuit(crate::circuits::AES_128);
    }

    #[test]
    fn test_parse_gate() {
        let _ = parse_gate("2 1 33280 33282 3691 XOR");
    }
}