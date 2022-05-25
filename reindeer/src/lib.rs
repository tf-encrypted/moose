use tonic::transport::{Certificate, ClientTlsConfig, Identity, ServerTlsConfig};

pub fn setup_tracing(identity: &str) -> Result<(), Box<dyn std::error::Error>> {
    use opentelemetry::sdk::trace::Config;
    use opentelemetry::sdk::Resource;
    use opentelemetry::KeyValue;
    use tracing_subscriber::{prelude::*, EnvFilter};

    let tracer = opentelemetry_jaeger::new_pipeline()
        .with_service_name("rudolph")
        .with_trace_config(
            Config::default().with_resource(Resource::new(vec![KeyValue::new(
                "identity",
                identity.to_string(),
            )])),
        )
        .install_simple()?;
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(telemetry)
        .try_init()?;

    Ok(())
}

pub fn setup_tls_client(
    my_cert_name: &str,
    certs_dir: &str,
) -> Result<ClientTlsConfig, Box<dyn std::error::Error>> {
    let (client_identity, ca_cert) = load_identity_and_ca(my_cert_name, certs_dir)?;
    let client_tls = ClientTlsConfig::new()
        .identity(client_identity)
        .ca_certificate(ca_cert);
    Ok(client_tls)
}

pub fn setup_tls_server(
    my_cert_name: &str,
    certs_dir: &str,
) -> Result<ServerTlsConfig, Box<dyn std::error::Error>> {
    let (identity, ca_cert) = load_identity_and_ca(my_cert_name, certs_dir)?;
    let server_tls = ServerTlsConfig::new()
        .identity(identity)
        .client_ca_root(ca_cert);
    Ok(server_tls)
}

const CA_NAME: &str = "ca";

pub fn load_identity_and_ca(
    my_cert_name: &str,
    certs_dir: &str,
) -> Result<(Identity, Certificate), Box<dyn std::error::Error>> {
    let my_cert_raw = std::fs::read(format!("{}/{}.crt", certs_dir, my_cert_name))?;
    let my_key_raw = std::fs::read(format!("{}/{}.key", certs_dir, my_cert_name))?;
    let identity = Identity::from_pem(my_cert_raw, my_key_raw);

    let ca_cert_raw = std::fs::read(format!("{}/{}.crt", certs_dir, CA_NAME))?;
    let ca_cert = Certificate::from_pem(ca_cert_raw);

    Ok((identity, ca_cert))
}
