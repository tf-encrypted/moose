extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::parse::{Parse, ParseStream, Result};
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::{parse_macro_input, parse_quote, BinOp, Expr, ExprBinary, Ident, Token};

/// Macros to convert expression into player/context invocations.
///
/// For example, converts
/// eval_with_context!(a, b, x + y * z)
/// into
/// a.add(b, &x, a.mul(b, &y, &z))
#[proc_macro]
pub fn eval_with_context(input: TokenStream) -> TokenStream {
    let EvalWithContext {
        player,
        context,
        mut expr,
    } = parse_macro_input!(input as EvalWithContext);
    unsugar(player, context, &mut expr);
    TokenStream::from(quote!(#expr))
}

/// Input members of the macros signature
struct EvalWithContext {
    player: Ident,
    context: Ident,
    expr: Expr,
}

impl Parse for EvalWithContext {
    fn parse(input: ParseStream) -> Result<Self> {
        let player: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let context: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let expr: Expr = input.parse()?;
        Ok(EvalWithContext {
            player,
            context,
            expr,
        })
    }
}

/// The main function for the eval_with_context macros.
///
/// Parses the expression and replaced binary operations with calls to the player/context methods.
fn unsugar(player: Ident, context: Ident, expr: &'_ mut Expr) {
    struct Visitor {
        player: Ident,
        context: Ident,
    }
    impl VisitMut for Visitor {
        fn visit_expr_mut(self: &mut Visitor, expr: &mut Expr) {
            // 1: subrecruse
            syn::visit_mut::visit_expr_mut(self, expr);
            // 2: process the outermost layer
            let player = &self.player;
            let context = &self.context;

            let (left, op, right) = match *expr {
                Expr::Binary(ExprBinary {
                    ref mut left,
                    op,
                    ref mut right,
                    ..
                }) => (left, op, right),
                _ => {
                    return;
                }
            };
            let span = op.span();
            let bin_fun = match op {
                BinOp::Add(_) => quote_spanned!(span=>#player.add),
                BinOp::Mul(_) => quote_spanned!(span=>#player.mul),
                _ => return,
            };

            *expr = parse_quote!( #bin_fun(#context,  &#left, &#right) );
        }
    }

    let mut visitor = Visitor { player, context };
    visitor.visit_expr_mut(expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trim_margin::MarginTrimmable;

    #[test]
    fn test_normal() {
        let t = trybuild::TestCases::new();
        t.pass("tests/pass/*.rs");
    }

    /// Utility function to return a pretty-printed token stream, if it is valid enough Rust code.
    fn format_tokenstream(code: proc_macro2::TokenStream) -> String {
        use std::io::{Read, Write};
        use std::process::{Command, Stdio};
        fn format(input: &str) -> Option<String> {
            let mut rustfmt = Command::new("rustfmt")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .ok()?;
            match rustfmt.stdin.take().unwrap() {
                ref mut stdin => {
                    Write::write_all(stdin, input.as_bytes()).ok()?;
                }
            }
            let mut stdout = String::new();
            Read::read_to_string(&mut rustfmt.stdout.take().unwrap(), &mut stdout).ok()?;
            if rustfmt.wait().ok()?.success() {
                Some(stdout)
            } else {
                None
            }
        }
        let string_code = code.to_string();
        format(&string_code).unwrap_or(string_code)
    }

    #[test]
    fn test_direct() {
        let player: Ident = parse_quote!(p);
        let context: Ident = parse_quote!(q);
        let mut e: Expr = parse_quote!(a + b * c);
        unsugar(player, context.into(), &mut e);
        let result = format_tokenstream(quote!(fn main() {let z = #e;}));
        
        // Make sure the produced code matches the expectation
        let expected = r#"
        |fn main() {
        |    let z = p.add(q, &a, &p.mul(q, &b, &c));
        |}
        |"#
        .trim_margin()
        .unwrap();
        assert_eq!(expected, result);
    }
}
