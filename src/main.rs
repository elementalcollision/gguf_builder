use anyhow::{Context, Result, bail};
use clap::Parser;
use std::path::PathBuf;
use std::fs::{self, File};
use std::io::BufWriter;
use serde_json::Value as JsonValue;

// Import our library crates
use safetensors_loader::{load_safetensors, SafeTensorDtype, SafeTensorsModel};
use gguf_builder::{GGUFWriter, GGUFValue, TensorInfo as GGUFTensorInfo, GGMLTypeRenamed};
// use core_logic; // TODO: Import if shared logic is needed directly

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory containing the model files (shards, config.json).
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Output GGUF file path.
    #[arg(short, long)]
    output: PathBuf,

    // config path is now inferred relative to model_dir
    // /// Optional path to the model's config.json file.
    // #[arg(long)]
    // config: Option<PathBuf>,

    // TODO: Add arguments for specific GGUF metadata (architecture, etc.)
    // TODO: Add arguments for quantization options if needed later
}

// Helper function to get the GGUF architecture prefix from config names
fn get_arch_prefix(config: &JsonValue) -> Option<&str> {
    // Prioritize 'architectures' array, fallback to 'model_type'
    if let Some(arch) = config.get("architectures").and_then(|v| v.as_array()?.get(0)?.as_str()) {
        match arch {
            // Add more mappings as needed
            _ if arch.contains("Llama") => Some("llama"),
            _ if arch.contains("Mistral") => Some("mistral"),
            _ if arch.contains("Mixtral") => Some("mistral"), // Mixtral often uses mistral prefix
            // ... add other architectures like Falcon, MPT, etc.
            _ => None,
        }
    } else if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
         match model_type {
            "llama" => Some("llama"),
            "mistral" => Some("mistral"),
            "mixtral" => Some("mistral"),
            // ... add other model_types
            _ => None,
        }
    } else {
        None
    }
}

// Helper to extract a String value from JSON
fn get_string(config: &JsonValue, key: &str) -> Result<Option<String>> {
    match config.get(key) {
        Some(val) => val.as_str()
            .ok_or_else(|| anyhow::anyhow!("Metadata key '{}' is not a valid string", key))
            .map(|s| Some(s.to_string())),
        None => Ok(None), // Key not present
    }
}

// Standalone helper function to add a metadata field
fn add_metadata_field(
    writer: &mut GGUFWriter,
    config: &JsonValue,
    gguf_key: String, 
    config_key: &str, 
    required: bool, 
    default_val: Option<GGUFValue>
) -> Result<()> {
    if let Some(json_val) = config.get(config_key) {
        // Value found in config, try parsing
        if let Some(val_f64) = json_val.as_f64() {
            // It's a float
            log::trace!("Setting {} = {}", gguf_key, val_f64 as f32);
            writer.add_metadata(gguf_key, GGUFValue::Float32(val_f64 as f32));
            Ok(())
        } else if let Some(val_u64) = json_val.as_u64() {
             // It's an unsigned integer
            log::trace!("Setting {} = {}", gguf_key, val_u64 as u32);
            writer.add_metadata(gguf_key, GGUFValue::Uint32(val_u64 as u32));
            Ok(())
        } else if let Some(val_i64) = json_val.as_i64() {
            // It's a signed integer - GGUF usually uses unsigned for these counts, but handle just in case
            log::warn!("Metadata key '{}' is a signed integer ({}), attempting to cast to u32 for GGUF key '{}'", config_key, val_i64, gguf_key);
            if val_i64 >= 0 {
                writer.add_metadata(gguf_key, GGUFValue::Uint32(val_i64 as u32));
                 Ok(())
            } else {
                 Err(anyhow::anyhow!("Metadata key '{}' has negative value ({}) unsuitable for GGUF key '{}'", config_key, val_i64, gguf_key))
            }
        } else {
            // Type mismatch (e.g., string, bool, array found when number expected)
            Err(anyhow::anyhow!("Metadata key '{}' has unexpected type '{:?}' for GGUF key '{}'", config_key, json_val, gguf_key))
        }
    } else {
        // Value not found in config
        if let Some(default) = default_val {
            log::trace!("Using default for {}: {:?}", gguf_key, default);
            writer.add_metadata(gguf_key, default);
            Ok(())
        } else if required {
            log::error!("Required metadata key '{}' (maps to GGUF '{}') not found in config!", config_key, gguf_key);
            Err(anyhow::anyhow!("Missing required metadata key: {}", config_key))
        } else {
            log::trace!("Optional metadata key '{}' not found, skipping.", config_key);
            Ok(())
        }
    }
}

fn main() -> Result<()> {
    // Initialize logging (e.g., using RUST_LOG env var)
    env_logger::init();

    let args = Args::parse();

    // Validate model directory
    if !args.model_dir.is_dir() {
        bail!("Provided model path is not a directory: {:?}", args.model_dir);
    }

    log::info!("Starting conversion process...");
    log::info!("Model Directory: {:?}", args.model_dir);
    log::info!("Output: {:?}", args.output);
    
    // Infer config path
    let config_path = args.model_dir.join("config.json");
    log::info!("Attempting to load config from: {:?}", config_path);

    // --- Load Config File --- 
    let config_data: Option<JsonValue> = if config_path.exists() {
        log::info!("Loading configuration from {:?}...", config_path);
        let config_str = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;
        let data: JsonValue = serde_json::from_str(&config_str)
            .with_context(|| format!("Failed to parse JSON from config file: {:?}", config_path))?;
        log::debug!("Successfully parsed config file:
{:#?}", data);
        Some(data)
    } else {
        log::warn!("Config file not found at {:?}. Proceeding without config.", config_path);
        None
    };

    // --- Load Input SafeTensors Model (passing directory now) ---
    // The loader will need refactoring to handle the directory and sharding
    let loaded_model = load_safetensors(&args.model_dir)
        .with_context(|| format!("Failed to load model from directory: {:?}", args.model_dir))?;
    
    // Temporarily comment out tensor count log until loader is fully sharded
    // let tensor_count = loaded_model.tensors()?.count(); 
    // log::info!("Successfully loaded (first shard of) safetensors model.", tensor_count); 
    log::info!("Successfully loaded (first shard of) safetensors model.");

    // --- Build GGUF Output ---
    log::info!("Building GGUF output.");
    build_gguf(&loaded_model, &args.output, config_data.as_ref())?;

    log::info!("Conversion process completed successfully.");
    Ok(())
}

// --- GGUF Building Logic ---
fn build_gguf(model: &SafeTensorsModel, output_path: &PathBuf, config_data: Option<&JsonValue>) -> Result<()> {
    log::info!("Starting GGUF build process for: {:?}", output_path);
    let mut gguf_writer = GGUFWriter::new();

    // --- 1. Add Metadata ---
    if let Some(config) = config_data {
        log::info!("Extracting metadata from provided config...");
        
        // Determine the primary config object (check for text_config nesting)
        let text_config = config.get("text_config").unwrap_or(config);
        log::debug!("Using config object: {:#?}", text_config);

        let arch_prefix = get_arch_prefix(config); // Arch prefix still comes from top-level

        // 1.1 General Metadata (from top-level config)
        let arch_name = if let Some(arch) = config.get("architectures").and_then(|v| v.as_array()?.get(0)?.as_str()) {
            arch.to_string()
        } else if let Some(model_type) = get_string(config, "model_type")? {
             model_type
        } else {
            "unknown".to_string()
        };
        log::info!("Architecture: {}", arch_name);
        gguf_writer.add_metadata("general.architecture".to_string(), GGUFValue::String(arch_name));
        let model_name = output_path.file_stem().and_then(|s| s.to_str()).unwrap_or("converted_model").to_string();
        gguf_writer.add_metadata("general.name".to_string(), GGUFValue::String(model_name));

        // 1.2 Architecture-Specific Metadata (use text_config)
        if let Some(prefix) = arch_prefix {
            log::info!("Using architecture prefix: {}", prefix);
            
            // Call helper function using text_config
            add_metadata_field(&mut gguf_writer, text_config, format!("{}.context_length", prefix), "max_position_embeddings", true, None)?;
            add_metadata_field(&mut gguf_writer, text_config, format!("{}.embedding_length", prefix), "hidden_size", true, None)?;
            add_metadata_field(&mut gguf_writer, text_config, format!("{}.block_count", prefix), "num_hidden_layers", true, None)?;
            add_metadata_field(&mut gguf_writer, text_config, format!("{}.feed_forward_length", prefix), "intermediate_size", true, None)?;
            add_metadata_field(&mut gguf_writer, text_config, format!("{}.attention.head_count", prefix), "num_attention_heads", true, None)?;
            add_metadata_field(&mut gguf_writer, text_config, format!("{}.attention.layer_norm_rms_epsilon", prefix), "rms_norm_eps", true, None)?;
            add_metadata_field(&mut gguf_writer, text_config, format!("{}.rope.freq_base", prefix), "rope_theta", false, Some(GGUFValue::Float32(10000.0)))?;

            // Handle KV head count separately (using text_config)
            let kv_heads_opt = text_config.get("num_key_value_heads").and_then(|v| v.as_u64()).map(|v| v as u32);
            let attn_heads_opt = text_config.get("num_attention_heads").and_then(|v| v.as_u64()).map(|v| v as u32);
            let kv_head_count = kv_heads_opt.or(attn_heads_opt);
            if let Some(kv_count) = kv_head_count {
                 log::trace!("Setting {}.attention.head_count_kv = {}", prefix, kv_count);
                 gguf_writer.add_metadata(format!("{}.attention.head_count_kv", prefix), GGUFValue::Uint32(kv_count)); 
            } else {
                log::error!("Could not determine attention.head_count_kv even from num_attention_heads"); 
            }

            // Handle RoPE Dimension Count separately (using text_config)
            let rope_dim_opt = text_config.get("rope_dim").and_then(|v| v.as_u64()).map(|v| v as u32);
            if let Some(rope_dim) = rope_dim_opt {
                 log::trace!("Setting {}.rope.dimension_count = {}", prefix, rope_dim);
                 gguf_writer.add_metadata(format!("{}.rope.dimension_count", prefix), GGUFValue::Uint32(rope_dim)); 
            } else {
                 let hidden_opt = text_config.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as u32);
                 let heads_opt = text_config.get("num_attention_heads").and_then(|v| v.as_u64()).map(|v| v as u32);
                 if let (Some(hidden), Some(heads)) = (hidden_opt, heads_opt) {
                     if heads > 0 {
                        let calculated_rope_dim = hidden / heads;
                        log::info!("Calculating rope.dimension_count = hidden_size / head_count = {}", calculated_rope_dim);
                        gguf_writer.add_metadata(format!("{}.rope.dimension_count", prefix), GGUFValue::Uint32(calculated_rope_dim)); 
                     } else {
                         log::warn!("Cannot calculate rope.dimension_count (num_attention_heads is 0)");
                     }
                 } else {
                      log::warn!("Could not determine rope.dimension_count from hidden_size/num_attention_heads");
                 }
            }
            
            // MoE parameters (optional, using text_config)
            add_metadata_field(&mut gguf_writer, text_config, format!("{}.expert_count", prefix), "num_local_experts", false, None)?;
            add_metadata_field(&mut gguf_writer, text_config, format!("{}.expert_used_count", prefix), "num_experts_per_tok", false, None)?;

        } else {
            log::warn!("Could not determine GGUF architecture prefix. Skipping architecture-specific metadata.");
        }

        // 1.3 Tokenizer Metadata (using text_config or top-level config)
        gguf_writer.add_metadata("tokenizer.ggml.model".to_string(), GGUFValue::String(arch_prefix.unwrap_or("unknown").to_string()));
        let vocab_size_opt = text_config.get("vocab_size").and_then(|v| v.as_u64()).map(|v| v as u32);
        if let Some(vocab_size) = vocab_size_opt {
             gguf_writer.add_metadata("tokenizer.ggml.vocab_size".to_string(), GGUFValue::Uint32(vocab_size));
        } else {
            log::warn!("Could not find vocab_size in config");
        }
         let bos_id_opt = text_config.get("bos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32)
                            .or_else(|| config.get("bos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32));
         if let Some(id) = bos_id_opt {
             gguf_writer.add_metadata("tokenizer.ggml.bos_token_id".to_string(), GGUFValue::Uint32(id));
        } else {
            log::warn!("Could not find bos_token_id in config");
        }
        let eos_id_opt = text_config.get("eos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32)
                            .or_else(|| config.get("eos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32));
        if let Some(id) = eos_id_opt {
             gguf_writer.add_metadata("tokenizer.ggml.eos_token_id".to_string(), GGUFValue::Uint32(id));
        } else {
            log::warn!("Could not find eos_token_id in config");
        }
        // Check both nested and top-level for pad_token_id
        let pad_token_id_opt = text_config.get("pad_token_id").and_then(|v| v.as_u64()).map(|v| v as u32)
                                .or_else(|| config.get("pad_token_id").and_then(|v| v.as_u64()).map(|v| v as u32));
        if let Some(id) = pad_token_id_opt {
             let pad_val = text_config.get("pad_token_id").or_else(|| config.get("pad_token_id"));
             if let Some(val) = pad_val {
                 if !val.is_null() {
                     gguf_writer.add_metadata("tokenizer.ggml.padding_token_id".to_string(), GGUFValue::Uint32(id));
                 }
             }
        } else {
            log::trace!("pad_token_id not found or not an integer, skipping padding_token_id");
        }
        let unk_id_opt = text_config.get("unk_token_id").and_then(|v| v.as_u64()).map(|v| v as u32)
                            .or_else(|| config.get("unk_token_id").and_then(|v| v.as_u64()).map(|v| v as u32));
        if let Some(id) = unk_id_opt {
             gguf_writer.add_metadata("tokenizer.ggml.unknown_token_id".to_string(), GGUFValue::Uint32(id));
        } else {
            log::trace!("unk_token_id not found or not an integer, skipping unknown_token_id");
        }

    } else {
        log::warn!("No config.json provided. GGUF file will be missing essential metadata!");
        // Add minimal required placeholders if possible, though likely insufficient
        gguf_writer.add_metadata(
            "general.architecture".to_string(),
            GGUFValue::String("unknown".to_string()), 
        );
    }

    // --- 2. Prepare Tensor Info and Data ---
    let mut tensor_data_slices: Vec<Vec<u8>> = Vec::new(); // Store owned Vec<u8> temporarily
    let mut processed_tensor_count = 0;

    // Get the tensor iterator (handle Result before loop)
    let tensor_iterator = model.tensors()?;

    // Iterate over tensors (yields (String, String) - name, shard_filename)
    for item in tensor_iterator {
        // item is now (String, String)
        let (name, shard_filename): (String, String) = item; // name is owned String
        
        let mut tensor_info: Option<GGUFTensorInfo> = None;
        let mut tensor_data: Option<Vec<u8>> = None;

        // Use get_tensor_view, passing the shard_filename
        model.get_tensor_view(&name, &shard_filename, |view| {
            log::debug!(
                "Processing tensor: {} (from {}) | Type: {:?} | Shape: {:?}",
                name, shard_filename, view.dtype(), view.shape()
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
                other => {
                    // Use bail! within the closure? Need to check error handling.
                    // For now, log and skip, setting Option to None
                    log::warn!("Unsupported tensor type {:?} for tensor '{}'. Skipping.", other, name);
                    return Ok(()); // Indicate skipping this tensor within the closure
                }
            };

            // Create tensor info within the closure
            tensor_info = Some(GGUFTensorInfo {
                name: name.clone(),
                dimensions: view.shape().iter().map(|&d| d as u64).collect(),
                ggml_type,
                offset: 0, // Writer calculates this
            });

            // Copy tensor data into an owned Vec<u8> within the closure
            tensor_data = Some(view.data().to_vec());

            Ok(()) // Return Ok from the closure
        })?;

        // Add info and data if they were successfully extracted
        if let (Some(info), Some(data)) = (tensor_info, tensor_data) {
            gguf_writer.add_tensor_info(info);
            tensor_data_slices.push(data); // Add the owned Vec<u8>
            processed_tensor_count += 1;
        } else {
             log::warn!("Skipping tensor '{}' due to processing error or unsupported type.", name);
        }
    }
    log::info!("Prepared {} tensors for GGUF writing.", processed_tensor_count);

    // --- 3. Write GGUF File ---
    let file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {:?}", output_path))?;
    let mut writer = BufWriter::new(file);

    // Convert Vec<Vec<u8>> to Vec<&[u8]> for the writer
    let tensor_data_ref_slices: Vec<&[u8]> = tensor_data_slices.iter().map(|v| v.as_slice()).collect();

    gguf_writer.write(&mut writer, &tensor_data_ref_slices)
        .with_context(|| format!("Failed to write GGUF data to: {:?}", output_path))?;

    log::info!("Successfully wrote GGUF file to: {:?}", output_path);
    Ok(())
}

// --- MLX Building Logic (Placeholder - remove as logic is in mlx_builder crate) ---
// fn build_mlx(model: &SafeTensorsModel, output_path: &PathBuf) -> Result<()> {
//     unimplemented!("MLX building not yet implemented");
// }
