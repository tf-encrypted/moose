use moose_macros::with_context;

struct Player;

impl Player {
    pub fn add(&self, ctx: &str, x: &i64, y: &i64) -> i64 {
        assert_eq!(ctx, "c");
        x + y
    }

    pub fn mul(&self, ctx: &str, x: &i64, y: &i64) -> i64 {
        assert_eq!(ctx, "c");
        x * y
    }
}

fn main() {
    let player = Player{};
    let ctx = "c";
    let x = 0;
    let y = 1;
    let z = 2;
    let res = with_context!(player, ctx, x + y * z);
    assert_eq!(res, 2);
}
