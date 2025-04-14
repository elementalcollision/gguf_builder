# GGUF Builder

This Rust project provides a tool to convert model weights from the SafeTensors format into the GGUF (GGML Universal Format).

## Features

*   Loads `.safetensors` model files.
*   Builds GGUF v3 files containing model metadata and tensor data.
*   Uses a workspace structure with dedicated crates for:
    *   `safetensors-loader`: Loading and accessing SafeTensors data.
    *   `gguf-builder`: Constructing and writing GGUF files.
    *   `core-logic`: (Placeholder for shared logic if needed).

## Usage

```bash
# Build the project
cargo build --release

# Run the conversion tool (example)
./target/release/ggml-builder --input path/to/model.safetensors --output path/to/model.gguf
```

## Development

*   Build: `cargo build`
*   Check: `cargo check`
*   Test: `cargo test` (Note: Some tests in `safetensors-loader` requiring the `serialize` feature are currently disabled due to build conflicts). 