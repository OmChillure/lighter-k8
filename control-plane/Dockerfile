FROM rust:1-slim-bookworm as builder

WORKDIR /app

RUN apt-get update && apt-get install -y pkg-config libssl-dev libpq-dev && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml .
COPY src ./src
COPY migrations ./migrations

RUN cargo build --release

FROM debian:bookworm-slim   

RUN apt-get update && apt-get install -y libssl3 ca-certificates libpq5 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/control-plane ./control-plane

ENV PORT=8080
EXPOSE 8080

CMD ["./control-plane"]
