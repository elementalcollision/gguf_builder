use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::fs::File;
use std::io::BufWriter;

// Import our library crates
use safetensors_loader::{load_safetensors, SafeTensorDtype, SafeTensorsModel};
use gguf_builder::{GGUFWriter, GGUFValue, TensorInfo as GGUFTensorInfo, GGMLTypeRenamed};
// use core_logic; // TODO: Import if shared logic is needed directly

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input SafeTensors file path.
    #[arg(short, long)]
    input: PathBuf,

    /// Output file path.
    #[arg(short, long)]
    output: PathBuf,

    // TODO: Add arguments for specific GGUF metadata (architecture, etc.)
    // TODO: Add arguments for quantization options if needed later
}

fn main() -> Result<()> {
    // Initialize logging (e.g., using RUST_LOG env var)
    env_logger::init();

    let args = Args::parse();

    log::info!("Starting conversion process...");
    log::info!("Input: {:?}", args.input);
    log::info!("Output: {:?}", args.output);
    // log::info!("Format: {:?}", args.format); // Format is implicitly GGUF now

    // --- Load Input SafeTensors File ---
    let loaded_model = load_safetensors(&args.input)
        .with_context(|| format!("Failed to load input safetensors file: {:?}", args.input))?;
    // Handle Result before counting tensors
    let tensor_count = loaded_model.tensors()?.count();
    log::info!("Successfully loaded safetensors with {} tensors.", tensor_count);

    // --- Build GGUF Output ---
    // Output format is implicitly GGUF now
    log::info!("Building GGUF output.");
    build_gguf(&loaded_model, &args.output)?;

    log::info!("Conversion process completed successfully.");
    Ok(())
}

// --- GGUF Building Logic ---
fn build_gguf(model: &SafeTensorsModel, output_path: &PathBuf) -> Result<()> {
    log::info!("Starting GGUF build process for: {:?}", output_path);
    let mut gguf_writer = GGUFWriter::new();

    // --- 1. Add Metadata (Example - needs proper values) ---
    // TODO: Get these from model config or command line args
    gguf_writer.add_metadata(
        "general.architecture".to_string(),
        GGUFValue::String("unknown".to_string()), // Placeholder!
    );
    gguf_writer.add_metadata(
        "general.name".to_string(),
        GGUFValue::String("converted_model".to_string()), // Placeholder!
    );
    // Add other relevant metadata from safetensors if available
    // e.g., model.metadata.metadata() if that map exists

    // --- 2. Prepare Tensor Info and Data ---
    let mut tensor_data_slices: Vec<&[u8]> = Vec::new();
    let mut processed_tensor_count = 0; // Renamed to avoid confusion with total count

    // Handle Result before iterating
    for item in model.tensors()? {
        let (name, view) = item; // Destructure the tuple from the iterator
        log::debug!(
            "Processing tensor: {} | Type: {:?} | Shape: {:?}",
            name, view.dtype(), view.shape()
        );
        
        let ggml_type = match view.dtype() {
            SafeTensorDtype::F64 => GGMLTypeRenamed::F64,
            SafeTensorDtype::F32 => GGMLTypeRenamed::F32,
            SafeTensorDtype::F16 => GGMLTypeRenamed::F16,
            SafeTensorDtype::BF16 => GGMLTypeRenamed::BF16,
            SafeTensorDtype::I64 => GGMLTypeRenamed::I64,
            SafeTensorDtype::I32 => GGMLTypeRenamed::I32,
            SafeTensorDtype::I16 => GGMLTypeRenamed::I16,
            SafeTensorDtype::I8 => GGMLTypeRenamed::I8,
            // TODO: Handle U8, U16, U32, U64 if needed and GGUF supports them
            // TODO: Map quantized types if input can be quantized
            other => {
                log::warn!("Unsupported tensor type {:?} for tensor '{}'. Skipping.", other, name);
                continue; // Skip unsupported tensors for now
                // Alternatively, could try to convert (e.g., F64 -> F32) but that requires allocation
                // bail!("Unsupported tensor type {:?} for GGUF conversion", other);
            }
        };

        let info = GGUFTensorInfo {
            name: name.clone(),
            // Convert usize shape dimensions to u64
            dimensions: view.shape().iter().map(|&d| d as u64).collect(),
            ggml_type,
            offset: 0, // Writer calculates this
        };

        gguf_writer.add_tensor_info(info);
        tensor_data_slices.push(view.data());
        processed_tensor_count += 1;
    }
    log::info!("Prepared {} tensors for GGUF writing.", processed_tensor_count);

    // --- 3. Write GGUF File ---
    let file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {:?}", output_path))?;
    // Use BufWriter for potentially better performance
    let mut writer = BufWriter::new(file);

    gguf_writer.write(&mut writer, &tensor_data_slices)
        .with_context(|| format!("Failed to write GGUF data to: {:?}", output_path))?;

    log::info!("Successfully wrote GGUF file to: {:?}", output_path);
    Ok(())
}

// --- MLX Building Logic (Placeholder - remove as logic is in mlx_builder crate) ---
// fn build_mlx(model: &SafeTensorsModel, output_path: &PathBuf) -> Result<()> {
//     unimplemented!("MLX building not yet implemented");
// }
