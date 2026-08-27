test:
    cargo test

check:
    cargo fmt --check
    cargo clippy --tests --examples -- -Dwarnings
