//! Support for generating replicated setup

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

impl ReplicatedPlacement {
    #[cfg(any(feature = "compile", feature = "sync_execute"))]
    pub(crate) fn gen_setup<S: Session, PrfKeyT>(&self, sess: &S) -> Result<RepSetup<PrfKeyT>>
    where
        PrfKeyT: Clone,
        HostPlacement: PlacementKeyGen<S, PrfKeyT>,
        HostPlacement: PlacementPlace<S, PrfKeyT>,
    {
        let (player0, player1, player2) = self.host_placements();

        let k00 = player0.gen_key(sess);
        let k11 = player1.gen_key(sess);
        let k22 = player2.gen_key(sess);

        let k10 = player0.place(sess, k11.clone());
        let k21 = player1.place(sess, k22.clone());
        let k02 = player2.place(sess, k00.clone());

        Ok(RepSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        })
    }
}
