//! DeepSeek V4 transformers-parity harness wiring (story #4271).
//!
//! This test wires the candle-vs-transformers parity harness into the cargo
//! test suite. The harness itself is `candle-examples/examples/deepseekv4/
//! parity_harness.py`, which compares candle (eager + flash) logits against the
//! HuggingFace `transformers` reference on the tiny sample model within a bf16
//! tolerance.
//!
//! It is *skippable by design*:
//!   - It only runs when the environment explicitly opts in with
//!     `DEEPSEEK_V4_TRANSFORMERS_PARITY=1` and points at a prebuilt candle
//!     example binary via `DEEPSEEK_V4_CANDLE_BIN` (so the normal test suite
//!     never triggers a nested `cargo build` of the example).
//!   - The python `torch`/`transformers` packages AND the tiny checkpoint
//!     (weights + tokenizer) must be present, otherwise it skips.
//!
//! The candle-internal eager-vs-flash parity (which needs no python) is covered
//! by the `flash_eager_parity*` tests in `models/deepseek_v4/mod.rs`.

use std::path::PathBuf;
use std::process::Command;

/// Location of the harness script relative to this crate's manifest dir.
fn harness_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../candle-examples/examples/deepseekv4/parity_harness.py")
}

/// Whether the python-side transformers comparison can run: python3 present,
/// `transformers`/`torch` importable, and the tiny checkpoint has weights +
/// tokenizer.
fn can_run_python_comparison() -> bool {
    if !harness_script().is_file() {
        return false;
    }
    let probe = Command::new("python3")
        .args(["-c", "import torch, transformers"])
        .output();
    if !probe.map(|o| o.status.success()).unwrap_or(false) {
        return false;
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sample_models/deepseek-v4-tiny");
    let has_weights = dir.join("model.safetensors").is_file()
        || dir.join("model.safetensors.index.json").is_file();
    has_weights && dir.join("tokenizer.json").is_file()
}

#[test]
fn deepseek_v4_transformers_parity_harness() {
    // Opt-in gate: the harness must be driven with a prebuilt candle example so
    // `cargo test` never shells out to a nested `cargo build`.
    let Ok(bin) = std::env::var("DEEPSEEK_V4_CANDLE_BIN") else {
        eprintln!(
            "SKIP deepseek_v4_transformers_parity_harness: set \
             DEEPSEEK_V4_CANDLE_BIN=<path to built deepseekv4 example> and \
             DEEPSEEK_V4_TRANSFORMERS_PARITY=1 to run the candle-vs-transformers \
             comparison."
        );
        return;
    };
    if std::env::var("DEEPSEEK_V4_TRANSFORMERS_PARITY").as_deref() != Ok("1") {
        eprintln!(
            "SKIP deepseek_v4_transformers_parity_harness: set \
             DEEPSEEK_V4_TRANSFORMERS_PARITY=1 to run the candle-vs-transformers \
             comparison."
        );
        return;
    }
    if !can_run_python_comparison() {
        eprintln!(
            "SKIP deepseek_v4_transformers_parity_harness: python torch/transformers \
             and/or the tiny checkpoint (weights + tokenizer) are not available. \
             To enable, install 'transformers[torch]' and populate \
             sample_models/deepseek-v4-tiny/ with the HF DeepSeek-V4 weights + \
             tokenizer, then run the harness directly."
        );
        return;
    }
    let status = Command::new("python3")
        .arg(harness_script())
        .arg("--model-dir")
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sample_models/deepseek-v4-tiny"))
        .arg("--candle-bin")
        .arg(&bin)
        .arg("--mode")
        .arg("both")
        .status()
        .expect("failed to spawn parity_harness.py");
    assert!(
        status.success(),
        "transformers-parity harness exited with {:?}",
        status.code()
    );
}
