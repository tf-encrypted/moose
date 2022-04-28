use crate::host::RawShape;
use crate::host::SliceInfo;
use anyhow::anyhow;
use bitvec::prelude::*;
use ndarray::{prelude::*, RemoveAxis};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Hash, Clone, PartialEq)]
pub struct BitArrayRepr {
    pub data: Arc<BitVec<u8, Lsb0>>,
    pub dim: Arc<IxDyn>,
}

fn do_collapse_axis<D: Dimension>(dims: &mut D, strides: &D, axis: usize, index: usize) -> isize {
    let dim = dims.slice()[axis];
    let stride = strides.slice()[axis];
    assert!(
        index < dim,
        "collapse_axis: Index {} must be less than axis length {} for \
         array with shape {:?}",
        index,
        dim,
        *dims
    );
    dims.slice_mut()[axis] = 1;
    (index as isize * (stride as isize)) as isize
}

impl BitArrayRepr {
    pub fn new_with_shape(dim: Arc<IxDyn>) -> Self {
        let data = BitVec::repeat(false, dim.size());
        BitArrayRepr {
            data: Arc::new(data),
            dim,
        }
    }

    pub fn from_raw(data: BitVec<u8, Lsb0>, dim: IxDyn) -> Self {
        BitArrayRepr {
            data: Arc::new(data),
            dim: Arc::new(dim),
        }
    }

    pub fn from_vec(vec: Vec<u8>, shape: &RawShape) -> Self {
        let data: BitVec<u8, Lsb0> = vec.iter().map(|&ai| ai != 0).collect();
        let dim = IxDyn(&shape.0);
        BitArrayRepr {
            data: Arc::new(data),
            dim: Arc::new(dim),
        }
    }

    pub fn from_elem(shape: &RawShape, elem: u8) -> Self {
        let dim = IxDyn(&shape.0);
        let data = BitVec::repeat(elem != 0, dim.size());
        BitArrayRepr {
            data: Arc::new(data),
            dim: Arc::new(dim),
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

    /// Converts into ArrayD for ndarray interop
    pub fn into_array<T: From<u8>>(&self) -> anyhow::Result<ArrayD<T>> {
        Array::from_iter(
            self.data
                .iter()
                .map(|item| if *item { T::from(1) } else { T::from(0) }),
        )
        .into_shape(IxDyn(self.shape()))
        .map_err(|e| anyhow!("Invalid shape {}", e))
    }

    pub fn index_axis(&self, axis: usize, index: usize) -> BitArrayRepr {
        // collapse axis
        let mut dim = IxDyn(self.dim.slice());
        let strides = dim.default_strides();

        let offset = do_collapse_axis(&mut dim, &strides, axis, index) as usize;
        let new_dim = self.dim.remove_axis(Axis(axis));
        let new_ptr = self.data.as_bitslice();

        // TODO(Dragos) can we make this init more efficient? ie use the ptr construction that BitVec has
        let data = {
            if new_dim.size() > 0 {
                Arc::new(BitVec::from_bitslice(
                    &new_ptr[offset..offset + new_dim.size()],
                ))
            } else {
                Arc::new(BitVec::from_bitslice(&new_ptr[offset..offset + 1]))
            }
        };
        BitArrayRepr {
            data,
            dim: Arc::new(new_dim),
        }
    }

    pub fn into_diag(&self) -> BitArrayRepr {
        let len = self.dim.slice().iter().cloned().min().unwrap_or(1);
        let mut data: BitVec<u8, Lsb0> = BitVec::EMPTY;
        match len {
            1 => data.push(self.data[0]),
            2 => {
                data.push(self.data[0]);
                let pos =
                    IxDyn::stride_offset(&IxDyn(&[1, 1]), &self.dim.default_strides()) as usize;
                data.push(self.data[pos])
            }
            // Should probably find a way to write it for any dimensions using IxDyn
            _ => todo!(),
        };
        BitArrayRepr {
            data: Arc::new(data),
            dim: Arc::new(IxDyn(&[len])),
        }
    }

    pub(crate) fn slice(&self, _info: SliceInfo) -> anyhow::Result<BitArrayRepr> {
        // TODO(Dragos) Implement this in future
        Err(anyhow::anyhow!("slicing not implemented for BitArray yet"))
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
