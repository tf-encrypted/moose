FROM rust:1.61

RUN apt update && \
    apt install -y libopenblas-dev

RUN cargo install moose
