use super::*;

/// Mux
pub trait PlacementMux<S: Session, T, U, V, O> {
    fn mux(&self, sess: &S, s: &T, x: &U, y: &V) -> O;
}

modelled!(PlacementMux::mux, ReplicatedPlacement, (BooleanTensor, Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (BooleanTensor, Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (Tensor, Tensor, Tensor) -> Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor, MuxOp);
modelled!(PlacementMux::mux, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor, MuxOp);

kernel! {
    MuxOp,
    [
        (ReplicatedPlacement, (Tensor, Tensor, Tensor) -> Tensor => [concrete] Self::logical_rep_kernel),
        (ReplicatedPlacement, (BooleanTensor, Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (BooleanTensor, Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => [concrete] Self::fixed_rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor  => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing128Tensor, ReplicatedRing128Tensor) -> ReplicatedRing128Tensor  => [transparent] Self::rep_bit_selector_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedRing64Tensor, ReplicatedRing64Tensor) -> ReplicatedRing64Tensor => [transparent] Self::rep_bit_selector_kernel),
        (ReplicatedPlacement, (ReplicatedRing128Tensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedRing64Tensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::rep_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed64Tensor, ReplicatedFixed64Tensor) -> ReplicatedFixed64Tensor => [hybrid] Self::rep_bit_selector_fixed_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedFixed128Tensor, ReplicatedFixed128Tensor) -> ReplicatedFixed128Tensor => [hybrid] Self::rep_bit_selector_fixed_kernel),
    ]
}
