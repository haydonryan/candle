test:
    cargo test --workspace

check:
    cargo fmt --all -- --check
    cargo clippy --workspace --tests --examples --benches -- -Dwarnings
