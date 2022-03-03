use crate::host::RawShape;
use bitvec::prelude::*;
use ndarray::{prelude::*, RemoveAxis};
use std::sync::Arc;

#[derive(Clone, PartialEq)]
pub struct BitArrayRepr {
    pub data: Arc<BitVec<u8, Lsb0>>,
    pub dim: IxDyn,
}

impl BitArrayRepr {
    pub fn from_raw(data: BitVec<u8, Lsb0>, dim: IxDyn) -> Self {
        BitArrayRepr {
            data: Arc::new(data),
            dim,
        }
    }

    pub fn from_vec(vec: Vec<u8>, shape: &RawShape) -> Self {
        let data = vec.iter().map(|&ai| ai != 0).collect();
        let dim = IxDyn(&shape.0);
        BitArrayRepr {
            data: Arc::new(data),
            dim,
        }
    }

    pub fn from_elem(shape: &RawShape, elem: u8) -> Self {
        let dim = IxDyn(&shape.0);
        let data = BitVec::repeat(elem != 0, dim.size());
        BitArrayRepr {
            data: Arc::new(data),
            dim,
        }
    }

    /// Return the number of dimensions (axes) in the array
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    /// Return the shape of the array as a slice.
    pub fn shape(&self) -> &[usize] {
        self.dim.slice()
    }

    pub fn index_axis(&self, axis: usize, index: usize) -> BitArrayRepr {
        let dim = self.dim.remove_axis(Axis(axis));
        if dim.slice() == &[1] {
            // Just a get element call
            let pos = IxDyn::stride_offset(&IxDyn(&[0, index]), &self.dim) as usize;
            return BitArrayRepr {
                data: Arc::new(BitVec::from_slice(&[self.data[pos] as u8])),
                dim,
            };
        }
        println!(
            "{:?}\naxis: {} index: {}\ndim: {:?}\n",
            self.dim,
            axis,
            index,
            dim.slice()
        );
        todo!("bit index_axis")
    }

    pub fn into_diag(&self) -> BitArrayRepr {
        let len = self.dim.slice().iter().cloned().min().unwrap_or(1);
        let mut data: BitVec<u8, Lsb0> = BitVec::repeat(false, len);
        match len {
            1 => data.set(0, self.data[0]),
            2 => {
                data.set(0, self.data[0]);
                let pos = IxDyn::stride_offset(&IxDyn(&[1, 1]), &self.dim) as usize;
                data.set(1, self.data[pos])
            }
            // Should probably find a way to write it for any dimensions using IxDyn
            _ => todo!(),
        };
        BitArrayRepr {
            data: Arc::new(data),
            dim: IxDyn(&[len]),
        }
    }
}

impl std::ops::BitXor for &BitArrayRepr {
    type Output = BitArrayRepr;
    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut data = (*self.data).clone();
        data ^= Arc::as_ref(&rhs.data);
        BitArrayRepr {
            data: Arc::new(data),
            dim: self.dim.clone(),
        }
    }
}

impl std::ops::Not for &BitArrayRepr {
    type Output = BitArrayRepr;
    fn not(self) -> Self::Output {
        let data = !(*self.data).clone();
        BitArrayRepr {
            data: Arc::new(data),
            dim: self.dim.clone(),
        }
    }
}

impl std::ops::BitAnd for &BitArrayRepr {
    type Output = BitArrayRepr;
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut data = (*self.data).clone();
        data &= Arc::as_ref(&rhs.data);
        BitArrayRepr {
            data: Arc::new(data),
            dim: self.dim.clone(),
        }
    }
}

impl std::ops::BitOr for &BitArrayRepr {
    type Output = BitArrayRepr;
    fn bitor(self, rhs: Self) -> Self::Output {
        let mut data = (*self.data).clone();
        data |= Arc::as_ref(&rhs.data);
        BitArrayRepr {
            data: Arc::new(data),
            dim: self.dim.clone(),
        }
    }
}
