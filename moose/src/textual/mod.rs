//! Textual representation of computations.

use crate::additive::AdditivePlacement;
use crate::computation::*;
use crate::host::{
    ArcArrayD, FromRaw, HostPlacement, RawPrfKey, RawSeed, RawShape, SliceInfo, SliceInfoElem,
    SyncKey,
};
use crate::logical::{TensorDType, TensorShape};
use crate::mirrored::Mirrored3Placement;
use crate::replicated::ReplicatedPlacement;
use crate::types::*;
use nom::{
    branch::{alt, permutation},
    bytes::complete::{is_not, tag, take_while_m_n},
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace1, space0},
    combinator::{all_consuming, cut, map, map_opt, map_res, opt, recognize, value, verify},
    error::{
        context, convert_error, make_error, ContextError, ErrorKind, ParseError, VerboseError,
    },
    multi::{fill, fold_many0, many0, separated_list0},
    number::complete::{double, float},
    sequence::{delimited, pair, preceded, tuple},
    Err::{Error, Failure},
    IResult,
};
use rayon::prelude::*;
use std::convert::TryFrom;
use std::str::FromStr;

mod parsing;
pub use parsing::*;

pub trait FromTextual<'a, E: 'a + ParseError<&'a str> + ContextError<&'a str>> {
    fn from_textual(input: &'a str) -> IResult<&'a str, Operator, E>;
}

/// A serializer to produce the same textual format from a computation
pub trait ToTextual {
    fn to_textual(&self) -> String;
}
