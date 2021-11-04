//! This module contains code for loading Bistrol Fashion circuits as (partial) computations.

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{newline, space0, u64};
use nom::combinator::{all_consuming, value};
use nom::multi::{length_count, many0, many_m_n, separated_list0};
use nom::sequence::{delimited, terminated, tuple};
use std::convert::TryFrom;

use crate::kernels::{PlacementAnd, PlacementNeg, PlacementXor, Session};

const AES_128: &[u8] = include_bytes!("aes_128.txt");

pub fn aes128<S: Session, P, BitT>(sess: &S, plc: &P, key: Vec<BitT>, block: Vec<BitT>) -> Vec<BitT>
where
    BitT: Clone,
    P: PlacementXor<S, BitT, BitT, BitT>,
    P: PlacementAnd<S, BitT, BitT, BitT>,
    P: PlacementNeg<S, BitT, BitT>,
{
    // Note that per comments on the Bristol website, inputs and outputs
    // for this particular circuit are given in reverse order!

    let circuit = Circuit::try_from(AES_128).unwrap();

    // TODO(Morten)
    // everything below is essentially circuit independent and should
    // be moved into eg an `eval` function on `Circuit`

    let mut wires: Vec<Option<BitT>> = vec![None; circuit.num_wires];

    assert_eq!(key.len(), 128);
    for (i, val) in key.into_iter().rev().enumerate() {
        *wires.get_mut(i).unwrap() = Some(val);
    }

    assert_eq!(block.len(), 128);
    for (i, val) in block.into_iter().rev().enumerate() {
        *wires.get_mut(i + 128).unwrap() = Some(val);
    }

    for gate in circuit.gates {
        use GateKind::*;
        match gate.kind {
            Xor => {
                let x_wire = *gate.input_wires.get(0).unwrap();
                let y_wire = *gate.input_wires.get(1).unwrap();
                let x = wires.get(x_wire).unwrap().clone().unwrap();
                let y = wires.get(y_wire).unwrap().clone().unwrap();

                let z = plc.xor(sess, &x, &y);
                let z_wire = *gate.output_wires.get(0).unwrap();
                *wires.get_mut(z_wire).unwrap() = Some(z);
            }
            And => {
                let x_wire = *gate.input_wires.get(0).unwrap();
                let y_wire = *gate.input_wires.get(1).unwrap();
                let x = wires.get(x_wire).unwrap().clone().unwrap();
                let y = wires.get(y_wire).unwrap().clone().unwrap();

                let z = plc.and(sess, &x, &y);
                let z_wire = *gate.output_wires.get(0).unwrap();
                *wires.get_mut(z_wire).unwrap() = Some(z);
            }
            Inv => {
                let x_wire = *gate.input_wires.get(0).unwrap();
                let x = wires.get(x_wire).unwrap().clone().unwrap();

                let y = plc.neg(sess, &x);
                let y_wire = *gate.output_wires.get(0).unwrap();
                *wires.get_mut(y_wire).unwrap() = Some(y);
            }
        }
    }

    wires
        .into_iter()
        .rev()
        .take(128)
        .map(|val| val.unwrap())
        .collect()
}

#[derive(Debug)]
pub struct Circuit {
    num_gates: usize,
    num_wires: usize,
    num_inputs: usize,
    input_wires: Vec<usize>,
    num_outputs: usize,
    output_wires: Vec<usize>,
    gates: Vec<Gate>,
}

impl TryFrom<&[u8]> for Circuit {
    type Error = crate::error::Error;
    fn try_from(bytes: &[u8]) -> Result<Circuit, Self::Error> {
        parse_circuit(bytes)
            .map_err(|e| {
                println!("{:?}", e);
                crate::error::Error::Unexpected(None)
            })
            .map(|res| res.1)
    }
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

type Res<T, U> = nom::IResult<T, U, nom::error::Error<T>>;

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
    // Optional blank lines at end
    let (bytes, _) = all_consuming(many0(newline))(bytes)?;
    assert_eq!(number_of_gates, gates.len()); // TODO return Err or move to TryFrom
    Ok((
        bytes,
        Circuit {
            num_gates: number_of_gates,
            num_wires: number_of_wires,
            num_inputs: 2, // TODO
            input_wires: vec![128, 128],
            num_outputs: 1, // TODO
            output_wires: vec![128],
            gates,
        },
    ))
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

fn parse_usize(line: &[u8]) -> Res<&[u8], usize> {
    let (line, res) = delimited(space0, u64, space0)(line)?;
    Ok((line, res as usize))
}

pub fn byte_to_bits_le(byte: &u8) -> Vec<u8> {
    (0..8).map(|i| (byte >> i) & 1).collect::<Vec<_>>()
}

pub fn byte_to_bits_be(byte: &u8) -> Vec<u8> {
    (0..8).map(|i| (byte >> (7 - i)) & 1).collect::<Vec<_>>()
}

pub fn bits_to_byte_le(bits: &[u8]) -> u8 {
    (0..8)
        .map(|i| bits[i] << i)
        .reduce(std::ops::Add::add)
        .unwrap()
}

pub fn bits_to_byte_be(bits: &[u8]) -> u8 {
    (0..8)
        .map(|i| bits[i] << (7 - i))
        .reduce(std::ops::Add::add)
        .unwrap()
}

pub fn byte_vec_to_bit_vec_be(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(byte_to_bits_be).collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::computation::{HostPlacement, ReplicatedPlacement, Role};
    use crate::host::HostBitTensor;
    use crate::kernels::SyncSession;
    use crate::replicated::ReplicatedBitTensor;

    #[test]
    fn test_parse_aes() {
        let _circuit = Circuit::try_from(AES_128).unwrap();
    }

    // test vectors from https://csrc.nist.gov/csrc/media/publications/fips/197/final/documents/fips-197.pdf
    const K: [u8; 16] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f,
    ];
    const M: [u8; 16] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ];
    const C: [u8; 16] = [
        0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5,
        0x5a,
    ];

    #[test]
    fn test_aes_reference() {
        let expected_c = {
            use aes::cipher::{generic_array::GenericArray, BlockEncrypt};
            use aes::{Aes128, Block, NewBlockCipher};

            let mut block = Block::clone_from_slice(&M);
            let key = GenericArray::from_slice(&K);
            let cipher = Aes128::new(key);
            cipher.encrypt_block(&mut block);
            block
        };

        assert_eq!(expected_c.as_slice(), &C);
    }

    #[test]
    fn test_aes_host() {
        let actual_c = {
            let host = HostPlacement {
                owner: Role::from("host"),
            };

            let k: Vec<HostBitTensor> = K
                .iter()
                .flat_map(byte_to_bits_be)
                .map(|b| HostBitTensor::from_slice_plc(&[b], host.clone()))
                .collect();

            let m: Vec<HostBitTensor> = M
                .iter()
                .flat_map(byte_to_bits_be)
                .map(|b| HostBitTensor::from_slice_plc(&[b], host.clone()))
                .collect();

            let sess = SyncSession::default();
            let c_bits: Vec<u8> = aes128(&sess, &host, k, m)
                .iter()
                .map(|t| t.0[0] & 1)
                .collect();
            let c: Vec<u8> = c_bits.chunks(8).map(bits_to_byte_be).collect();
            c
        };

        assert_eq!(actual_c.as_slice(), &C);
    }

    #[test]
    fn test_aes_replicated() {
        use crate::kernels::{PlacementReveal, PlacementShareSetup};

        let actual_c = {
            let host = HostPlacement {
                owner: Role::from("host"),
            };
            let rep = ReplicatedPlacement {
                owners: [Role::from("alice"), Role::from("bob"), Role::from("carole")],
            };

            let sess = SyncSession::default();
            let setup = sess.replicated_setup(&rep);

            let k: Vec<ReplicatedBitTensor> = K
                .iter()
                .flat_map(byte_to_bits_be)
                .map(|b| {
                    rep.share(
                        &sess,
                        setup.as_ref(),
                        &HostBitTensor::from_slice_plc(&[b], host.clone()),
                    )
                })
                .collect();

            let m: Vec<ReplicatedBitTensor> = M
                .iter()
                .flat_map(byte_to_bits_be)
                .map(|b| {
                    rep.share(
                        &sess,
                        setup.as_ref(),
                        &HostBitTensor::from_slice_plc(&[b], host.clone()),
                    )
                })
                .collect();

            let c_bits: Vec<u8> =
                aes128::<SyncSession, ReplicatedPlacement, ReplicatedBitTensor>(&sess, &rep, k, m)
                    .iter()
                    .map(|te| {
                        let t = host.reveal(&sess, te);
                        t.0[0] & 1
                    })
                    .collect();
            let c: Vec<u8> = c_bits.chunks(8).map(bits_to_byte_be).collect();
            c
        };

        assert_eq!(actual_c.as_slice(), &C);
    }
}
