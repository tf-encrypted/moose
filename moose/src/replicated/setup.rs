//! Support for generating replicated setup

use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepSetup<PrfKeyT> {
    pub keys: [[PrfKeyT; 2]; 3],
}

impl<PrfKeyT> Placed for RepSetup<PrfKeyT>
where
    PrfKeyT: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        let RepSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = self;

        let owner0 = k00.placement()?.owner;
        let owner1 = k11.placement()?.owner;
        let owner2 = k22.placement()?.owner;

        if k10.placement()?.owner == owner0
            && k21.placement()?.owner == owner1
            && k02.placement()?.owner == owner2
        {
            let owners = [owner0, owner1, owner2];
            Ok(ReplicatedPlacement { owners })
        } else {
            Err(Error::MalformedPlacement)
        }
    }
}

impl RepSetupOp {
    pub(crate) fn kernel<S: Session, PrfKeyT: Clone>(
        sess: &S,
        rep: &ReplicatedPlacement,
    ) -> Result<RepSetup<PrfKeyT>>
    where
        HostPlacement: PlacementKeyGen<S, PrfKeyT>,
        HostPlacement: PlacementPlace<S, PrfKeyT>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let k0 = player0.gen_key(sess);
        let k1 = player1.gen_key(sess);
        let k2 = player2.gen_key(sess);

        Ok(RepSetup {
            keys: [
                [
                    player0.place(sess, k0.clone()),
                    player0.place(sess, k1.clone()),
                ],
                [player1.place(sess, k1), player1.place(sess, k2.clone())],
                [player2.place(sess, k2), player2.place(sess, k0)],
            ],
        })
    }
}
