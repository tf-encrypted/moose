//! Support for bit (de-)composition

use super::*;
use std::convert::TryInto;

impl BitDecomposeOp {
    pub(crate) fn rep_ring_kernel<S: Session, RepRingT, RepBitT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepRingT,
    ) -> Result<RepBitArray<RepBitT, N>>
    where
        RepRingT: Ring<BitLength = N>,
        ReplicatedPlacement: PlacementSplit<S, RepRingT, RepBitT>,
        ReplicatedPlacement: BinaryAdder<S, RepBitT>,
    {
        let (x0, x1) = rep.split(sess, &x);
        let res = rep.binary_adder(sess, &x0, &x1, RepRingT::BitLength::VALUE);
        Ok(RepBitArray(res, PhantomData))
    }
}

impl BitComposeOp {
    pub(crate) fn rep_kernel<S: Session, ShapeT, RepRingT, RepBitArrayT, RepBitT, N: Const>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepBitArrayT,
    ) -> Result<RepRingT>
    where
        RepRingT: Ring<BitLength = N>,
        ReplicatedPlacement: PlacementIndex<S, RepBitArrayT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementShape<S, RepBitT, ShapeT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepRingT>,
    {
        let v: Vec<_> = (0..RepRingT::BitLength::VALUE)
            .map(|i| rep.index(sess, i, &x))
            .collect();

        let zeros = rep.fill(sess, 0u64.into(), &rep.shape(sess, &v[0]));
        Ok(v.iter().enumerate().fold(zeros, |x, (i, y)| {
            rep.add(sess, &x, &rep.ring_inject(sess, i, y))
        }))
    }
}

/// Split a replicated secret `x` into two replicated bits `(x1, x2)`
///
/// This is done such that when interpreted as ring tensors, they reconstruct to `x`
/// i.e. `(x1 + x2) = x` over the ring. This is useful for some protocols that don't
/// necessarily need the full bit-decomposition such as exponentiation.
pub(crate) trait PlacementSplit<S: Session, T, O> {
    fn split(&self, sess: &S, x: &T) -> (O, O);
}

#[cfg(feature = "compile")]
impl<S: Session, HostRingT, HostBitT>
    PlacementSplit<S, Symbolic<RepTensor<HostRingT>>, Symbolic<RepTensor<HostBitT>>>
    for ReplicatedPlacement
where
    HostRingT: Placed<Placement = HostPlacement>,
    HostBitT: Placed<Placement = HostPlacement>,
    ReplicatedPlacement: PlacementSplit<S, RepTensor<HostRingT>, RepTensor<HostBitT>>,
    RepTensor<HostRingT>: Into<Symbolic<RepTensor<HostRingT>>>,
    RepTensor<HostBitT>: Into<Symbolic<RepTensor<HostBitT>>>,
{
    fn split(
        &self,
        sess: &S,
        x: &Symbolic<RepTensor<HostRingT>>,
    ) -> (Symbolic<RepTensor<HostBitT>>, Symbolic<RepTensor<HostBitT>>) {
        let concrete_x = match x {
            Symbolic::Concrete(x) => x,
            Symbolic::Symbolic(_) => {
                unimplemented!()
            }
        };
        let (a, b) = Self::split(self, sess, concrete_x);
        (a.into(), b.into())
    }
}

impl<S: Session, HostRingT, HostBitT> PlacementSplit<S, RepTensor<HostRingT>, RepTensor<HostBitT>>
    for ReplicatedPlacement
where
    HostShape: KnownType<S>,
    HostBitT: Clone,

    HostPlacement: PlacementFill<S, m!(HostShape), HostBitT>,
    HostPlacement: PlacementShape<S, HostBitT, m!(HostShape)>,
    HostPlacement: PlacementAdd<S, HostRingT, HostRingT, HostRingT>,

    RepTensor<HostBitT>: CanonicalType,
    <RepTensor<HostBitT> as CanonicalType>::Type: KnownType<S>,
    m!(c!(RepTensor<HostBitT>)): TryInto<RepTensor<HostBitT>>,

    ReplicatedPlacement: PlacementShare<S, HostBitT, m!(c!(RepTensor<HostBitT>))>,
    HostPlacement: PlacementBitDecompose<S, HostRingT, HostBitT>,
{
    fn split(
        &self,
        sess: &S,
        x: &RepTensor<HostRingT>,
    ) -> (RepTensor<HostBitT>, RepTensor<HostBitT>) {
        let (player0, player1, player2) = self.host_placements();

        let RepTensor {
            shares: [[x00, x10], [_x11, x21], [x22, _x02]],
        } = &x;

        let left = with_context!(player0, sess, x00 + x10);
        let bsl = player0.bit_decompose(sess, &left);

        // transform x2 into boolean sharing
        let x2_on_1 = player1.bit_decompose(sess, x21);
        let x2_on_2 = player2.bit_decompose(sess, x22);

        let p0_zero = player0.fill(sess, 0_u8.into(), &player0.shape(sess, &bsl));
        let p1_zero = player1.fill(sess, 0_u8.into(), &player1.shape(sess, &x2_on_1));
        let p2_zero = player2.fill(sess, 0_u8.into(), &player2.shape(sess, &x2_on_2));

        let rep_bsl = self.share(sess, &bsl);
        let rep_bsr = RepTensor {
            shares: [
                [p0_zero.clone(), p0_zero],
                [p1_zero, x2_on_1],
                [x2_on_2, p2_zero],
            ],
        };

        (rep_bsl.try_into().ok().unwrap(), rep_bsr)
    }
}
