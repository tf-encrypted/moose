extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::parse::{Parse, ParseStream, Result};
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::PathArguments::AngleBracketed;
use syn::{
    parse_macro_input, parse_quote, BinOp, Data, Expr, ExprBinary, ExprCall, ExprPath, Fields,
    GenericArgument, Ident, Token, Type,
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

/// Derive macro to support textual format
#[proc_macro_derive(ToTextual)]
pub fn to_textual_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();
    let name = &ast.ident;
    // Note, we only need to truncate the name by the charaters to get rid of the `Op` suffix.
    // If we refactor to not have that suffix anymore we can just use `stringify!(#name)` inside `quote!` below.
    let mut ident_string = name.to_string();
    if ident_string.ends_with("Op") {
        ident_string.truncate(ident_string.len() - 2);
    }
    let formatter = match ast.data {
        Data::Struct(ref data) => match data.fields {
            Fields::Named(ref fields) => {
                // Building the format string in an old fashion way
                let mut format_string = String::from("{op}");
                let mut has_attributes = false;
                for item in fields.named.iter() {
                    match item.ident.as_ref().map(|i| i.to_string()) {
                        // Member "sig" is special, since it is the op's signature, not an attribute.
                        Some(ref name) if name != "sig" => {
                            if !has_attributes {
                                has_attributes = true;
                                format_string.push_str("{{");
                            } else {
                                format_string.push_str(", ");
                            }
                            format_string.push_str(name);
                            format_string.push_str(" = {");
                            format_string.push_str(name);
                            format_string.push('}');
                        }
                        _ => {}
                    }
                }
                if has_attributes {
                    format_string.push_str("}}");
                }
                format_string.push_str(": {sig}");

                // Simply iterate over all the members converting each into "value = self.value.to_textual()" call.
                let recurse = fields.named.iter().filter_map(|f| {
                    let id = &f.ident;
                    match id.as_ref().map(|i| i.to_string()) {
                        Some(name) if name != "sig" => Some(quote_spanned! {f.span()=>
                            #id = self.#id.to_textual()
                        }),
                        _ => None,
                    }
                });
                quote! {
                    format!(#format_string, op = #ident_string, #(#recurse ,)* sig = self.sig.to_textual())
                }
            }
            _ => quote!(),
        },
        _ => quote!(),
    };
    let gen = quote! {
        impl crate::textual::ToTextual for #name {
            fn to_textual(&self) -> String {
                #formatter
            }
        }
    };
    gen.into()
}

fn parser_for_type(name: &Option<String>, ty: &Type) -> Option<proc_macro2::TokenStream> {
    match ty {
        Type::Path(tp) if tp.path.is_ident("String") => {
            Some(quote!(crate::textual::attributes_member(#name, crate::textual::string)))
        }
        Type::Path(tp) if tp.path.is_ident("bool") => {
            Some(quote!(crate::textual::attributes_member(#name, crate::textual::parse_bool)))
        }
        Type::Path(tp) if tp.path.is_ident("u32") => {
            Some(quote!(crate::textual::attributes_member(#name, crate::textual::parse_int)))
        }
        Type::Path(tp) if tp.path.is_ident("u64") => {
            Some(quote!(crate::textual::attributes_member(#name, crate::textual::parse_int)))
        }
        Type::Path(tp) if tp.path.is_ident("usize") => {
            Some(quote!(crate::textual::attributes_member(#name, crate::textual::parse_int)))
        }
        Type::Path(tp) if tp.path.is_ident("SliceInfo") => Some(
            quote!(crate::textual::attributes_member(#name, crate::textual::slice_info_literal)),
        ),
        Type::Path(tp) if tp.path.is_ident("Constant") => {
            Some(quote!(crate::textual::attributes_member(#name, crate::textual::constant_literal)))
        }
        Type::Path(tp) if tp.path.is_ident("RendezvousKey") => {
            Some(quote!(crate::textual::attributes_member(#name, map(
                crate::textual::parse_hex,
                RendezvousKey::from_bytes
            ))))
        }
        Type::Path(tp) if tp.path.is_ident("Role") => Some(
            quote!(crate::textual::attributes_member(#name, map(crate::textual::string, Role::from))),
        ),
        Type::Path(tp) if tp.path.is_ident("Signature") => None, // One more clause to ignore the signature field.
        // Recursively recognizing the Option is a bit tough.
        Type::Path(tp) if tp.path.segments.len() == 1 && tp.path.segments[0].ident == "Option" => {
            match &tp.path.segments[0].arguments {
                AngleBracketed(arg) => match arg.args.first().unwrap() {
                    GenericArgument::Type(inner) => {
                        let inner_parser = parser_for_type(name, inner);
                        Some(quote!(opt(#inner_parser)))
                    }
                    _ => panic!("Expected a single type inside the Option"),
                },
                _ => panic!("Expected angled brackets after the Option"),
            }
        }
        _ => panic!(
            "The from textual macro could not derive a parser for an attribute with the type {:?}",
            ty,
        ),
    }
}

#[proc_macro_derive(FromTextual)]
pub fn from_textual_derive(input: TokenStream) -> TokenStream {
    let item_struct = syn::parse::<syn::ItemStruct>(input).unwrap();

    let name = &item_struct.ident;
    // Note, we only need to truncate the name by the charaters to get rid of the `Op` suffix.
    // If we refactor to not have that suffix anymore we can just use `stringify!(#name)` inside `quote!` below.
    let mut ident_string = name.to_string();
    if ident_string.ends_with("Op") {
        ident_string.truncate(ident_string.len() - 2);
    }

    // Grab all the field names (except `sig`) as a comma-separated list
    let mut attr_count = 0;
    let attributes =
        match item_struct.fields {
            Fields::Named(ref fields) => {
                let names = fields.named.iter().filter_map(|f| {
                    match f.ident.as_ref().map(|i| i.to_string()) {
                        Some(name) if name != "sig" => {
                            attr_count += 1;
                            Some(f.ident.as_ref())
                        }
                        _ => None,
                    }
                });

                quote! { #(#names ,)* }
            }
            _ => quote!(),
        };

    // Generate a parser for each attribute except `sig`.
    let attr_parsers = match item_struct.fields {
        Fields::Named(ref fields) => {
            let members = fields.named.iter().filter_map(|f| {
                let name = f.ident.as_ref().map(|i| i.to_string());
                parser_for_type(&name, &f.ty)
            });
            quote! { #(#members ),* }
        }
        _ => quote!(),
    };

    // We actually have 3 distinct cases here
    let attr_parser = match attr_count {
        // With 0 extra attributes we should just skip this block
        0 => None,
        // With one extra attribute we must call its parser directly
        1 => Some(quote! {
            let (input, #attributes) = delimited(ws(tag("{")), #attr_parsers, ws(tag("}")))(input)?;
        }),
        // With more than one extra attribute we have to wrap the parsers in a `permutation` call to allow attributes in any order.
        _ => Some(quote! {
            let (input, (#attributes)) = delimited(ws(tag("{")), permutation((#attr_parsers)), ws(tag("}")))(input)?;
        }),
    };

    let gen = quote! {
        impl<'a, E: 'a + nom::error::ParseError<&'a str> + nom::error::ContextError<&'a str>> crate::textual::FromTextual<'a, E> for #name {
            fn from_textual(input: &'a str) -> nom::IResult<&'a str, Operator, E> {
                use nom::branch::permutation;
                use nom::bytes::complete::tag;
                use nom::combinator::{cut, opt, map};
                use nom::sequence::{delimited, preceded};
                use crate::textual::ws;

                let parser = |input: &'a str| {
                    #attr_parser
                    let (input, sig) = crate::textual::operator_signature(0)(input)?;
                    Ok((input, #name {
                        sig,
                        #attributes
                    }.into()))
                };
                preceded(tag(#ident_string), cut(parser))(input)
            }
        }
    };
    // Uncomment the following line to see the generated code unrolled.
    // println!("==========================\n{}\n", format_tokenstream(gen.clone()));
    gen.into()
}

/// Utility function to return a pretty-printed token stream, if it is valid enough Rust code.
#[allow(dead_code)] // It is used in the tests, but is exceptionally helpful while debugging a macros
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

#[cfg(test)]
mod tests {
    use super::*;
    use trim_margin::MarginTrimmable;

    #[test]
    fn test_normal() {
        let t = trybuild::TestCases::new();
        t.pass("tests/pass/*.rs");
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
