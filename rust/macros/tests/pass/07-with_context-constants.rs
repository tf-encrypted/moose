use macros::with_context;

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
    let x = 45;
    let y = 646;
    let z = -465465;
    let res = with_context!(player, ctx, x + 2 * y * z);
    assert_eq!(res, x + 2 * y * z);
}
