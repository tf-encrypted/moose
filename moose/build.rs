fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rustc-link-search=native={}", "/opt/homebrew/opt/openblas/lib");
    tonic_build::compile_protos("protos/choreography.proto")?;
    tonic_build::compile_protos("protos/networking.proto")?;
    Ok(())
}
