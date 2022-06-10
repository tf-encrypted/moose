FROM rust:1.61

RUN apt update && \
    apt install -y libopenblas-dev

RUN rustup component add rustfmt

RUN cargo install moose
