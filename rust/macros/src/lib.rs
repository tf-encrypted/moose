extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::parse::{Parse, ParseStream, Result};
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::{
    parse_macro_input, parse_quote, BinOp, Expr, ExprBinary, ExprCall, ExprPath, Ident, Token,
};

/// Macros to convert expression into player/context invocations.
///
/// For example, converts
/// with_context!(a, b, x + y * z)
/// into
/// a.add(b, &x, a.mul(b, &y, &z))
#[proc_macro]
pub fn with_context(input: TokenStream) -> TokenStream {
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
    player: Expr,
    context: Ident,
    expr: Expr,
}

impl Parse for EvalWithContext {
    fn parse(input: ParseStream) -> Result<Self> {
        let player: Expr = input.parse()?;
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

/// The main function for the with_context macros.
///
/// Parses the expression and replaced binary operations with calls to the player/context methods.
fn unsugar(player: Expr, context: Ident, expr: &'_ mut Expr) {
    struct Visitor {
        player: Expr,
        context: Ident,
    }
    impl VisitMut for Visitor {
        fn visit_expr_mut(self: &mut Visitor, expr: &mut Expr) {
            // 1: subrecruse
            syn::visit_mut::visit_expr_mut(self, expr);
            // 2: process the outermost layer
            let player = &self.player;
            let context = &self.context;

            match *expr {
                Expr::Binary(ExprBinary {
                    ref mut left,
                    op,
                    ref mut right,
                    ..
                }) => {
                    let span = op.span();
                    let bin_fun = match op {
                        BinOp::Add(_) => quote_spanned!(span=>#player.add),
                        BinOp::Sub(_) => quote_spanned!(span=>#player.sub),
                        BinOp::Mul(_) => quote_spanned!(span=>#player.mul),
                        BinOp::Div(_) => quote_spanned!(span=>#player.div),
                        _ => return,
                    };

                    *expr = parse_quote!( #bin_fun(#context,  &#left, &#right) );
                }
                Expr::Call(ExprCall {
                    ref func, ref args, ..
                }) => {
                    match func.as_ref() {
                        // It is only safe to work with individual idents (paths of len 1)
                        Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
                            *expr = parse_quote!( #player.#func(#context,  #args) );
                        }
                        _ => {
                            // Ignore anything else
                        }
                    }
                }
                _ => {
                    // Ignore, do not make any changes
                }
            }
        }
    }

    let mut visitor = Visitor { player, context };
    visitor.visit_expr_mut(expr)
}

/// Derive macro to produce simple names of the enum structs
#[proc_macro_derive(ShortName)]
pub fn short_name_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();
    let name = &ast.ident;
    // Note, we only need to truncate the name by the charaters to get rid of the `Op` suffix.
    // If we refactor to not have that suffix anymore we can just use `stringify!(#name)` inside `quote!` below.
    let mut ident_string = name.to_string();
    if ident_string.ends_with("Op") {
        ident_string.truncate(ident_string.len() - 2);
    }
    let gen = quote! {
        impl HasShortName for #name {
            fn short_name(&self) -> &str {
                #ident_string
            }
        }
        impl #name {
            pub const SHORT_NAME: &'static str = #ident_string;
        }
    };
    gen.into()
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
        use std::io::Write;
        use std::process::{Command, Stdio};
        fn format(input: &str) -> Option<String> {
            let mut rustfmt = Command::new("rustfmt")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .ok()?;
            let child_stdin = rustfmt.stdin.as_mut()?;
            child_stdin.write_all(input.as_bytes()).ok()?;
            let output = rustfmt.wait_with_output().ok()?;
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        }
        let string_code = code.to_string();
        format(&string_code).unwrap_or(string_code)
    }

    #[test]
    fn test_direct() {
        let player: Expr = parse_quote!(p);
        let context: Ident = parse_quote!(q);
        let mut e: Expr = parse_quote!(a + b * c);
        unsugar(player, context, &mut e);
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

    #[test]
    /// Making sure the expression can have anything at all and not mess up our macros
    fn test_sub_expr() {
        let player: Expr = parse_quote!(p);
        let context: Ident = parse_quote!(q);
        let mut e: Expr = parse_quote!(a::new(d) + b.member * func(c));
        unsugar(player, context, &mut e);
        let result = format_tokenstream(quote!(fn main() {let z = #e;}));

        // Make sure the produced code matches the expectation
        let expected = r#"
        |fn main() {
        |    let z = p.add(q, &a::new(d), &p.mul(q, &b.member, &p.func(q, c)));
        |}
        |"#
        .trim_margin()
        .unwrap();
        assert_eq!(expected, result);
    }

    #[test]
    /// Making sure the expression can have anything at all and not mess up our macros
    fn test_sub_expr_inside() {
        let player: Expr = parse_quote!(p);
        let context: Ident = parse_quote!(q);
        let mut e: Expr = parse_quote!(a + func(b + c));
        unsugar(player, context, &mut e);
        let result = format_tokenstream(quote!(fn main() {let z = #e;}));

        // Make sure the produced code matches the expectation
        let expected = r#"
        |fn main() {
        |    let z = p.add(q, &a, &p.func(q, p.add(q, &b, &c)));
        |}
        |"#
        .trim_margin()
        .unwrap();
        assert_eq!(expected, result);
    }

    #[test]
    /// Trying to do the custom functions
    fn test_sub_expr_inside_expanded() {
        let player: Expr = parse_quote!(p);
        let context: Ident = parse_quote!(q);
        let mut e: Expr = parse_quote!(a + func(&b, &c));
        unsugar(player, context, &mut e);
        let result = format_tokenstream(quote!(fn main() {let z = #e;}));

        // Make sure the produced code matches the expectation
        let expected = r#"
        |fn main() {
        |    let z = p.add(q, &a, &p.func(q, &b, &c));
        |}
        |"#
        .trim_margin()
        .unwrap();
        assert_eq!(expected, result);
    }
}
