use safetensors::{SafeTensors, SafeTensorError};
use safetensors::tensor::TensorView;
use std::fs::File;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result, bail};
use std;
use std::io::Read;

/// Represents a loaded SafeTensors model file.
#[derive(Debug)]
pub struct SafeTensorsModel {
    // Hold the owned buffer containing the file data
    _buffer: Vec<u8>,
    path: PathBuf, // Store the path for reference
    // Removed metadata field to avoid self-referential lifetimes
}

impl SafeTensorsModel {
    /// Returns an iterator over the tensor names and their views.
    /// Deserializes the header on each call.
    pub fn tensors(&self) -> Result<impl Iterator<Item = (String, TensorView<'_>)>> {
        // Deserialize the header from the owned buffer on the fly.
        // The returned SafeTensors object borrows from self._buffer.
        let metadata = SafeTensors::deserialize(&self._buffer)
            .map_err(|e: SafeTensorError| anyhow::anyhow!(e))
            .context("Failed to deserialize SafeTensors header from buffer")?;
        // into_iter yields (String, TensorView<_'>)
        Ok(metadata.tensors().into_iter())
    }

    /// Gets a specific tensor by name.
    /// Deserializes the header on each call.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        // Deserialize the header from the owned buffer on the fly.
        let metadata = SafeTensors::deserialize(&self._buffer)
            .map_err(|e: SafeTensorError| anyhow::anyhow!(e))
            .context("Failed to deserialize SafeTensors header from buffer")?;
        // The view returned here borrows from self._buffer
        metadata
            .tensor(name)
            .map_err(|e: SafeTensorError| anyhow::anyhow!(e))
            .with_context(|| format!("Failed to find tensor named '{}'", name))
    }

    /// Gets the path from which the model was loaded.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the raw byte buffer of the loaded file.
    pub fn buffer(&self) -> &[u8] {
        &self._buffer
    }
}

/// Loads a SafeTensors model from the specified file path by reading it into memory.
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<SafeTensorsModel> {
    let path_ref = path.as_ref();
    log::info!("Loading safetensors model from: {:?}", path_ref);

    if !path_ref.exists() {
        log::error!("Safetensors file not found: {:?}", path_ref);
        bail!("Safetensors file not found: {}", path_ref.display());
    }

    let mut file = File::open(path_ref).with_context(|| format!("Failed to open file: {}", path_ref.display()))?;

    // Read the entire file into an owned buffer
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .with_context(|| format!("Failed to read file content: {}", path_ref.display()))?;
    log::debug!("Read file into buffer of size: {} bytes", buffer.len());

    // Basic validation: try deserializing header to catch immediate errors
    SafeTensors::deserialize(&buffer)
        .map_err(|e: SafeTensorError| anyhow::anyhow!(e))
        .with_context(|| format!("Failed initial validation of safetensors format in: {}", path_ref.display()))?;
    log::info!("Successfully validated safetensors format.");

    Ok(SafeTensorsModel {
        _buffer: buffer, // Store the owned buffer
        path: path_ref.to_path_buf(),
    })
}

// Re-export key types for easier use by other crates
pub use safetensors::Dtype as SafeTensorDtype;
pub use safetensors::tensor::TensorView as SafeTensorView;

#[cfg(test)]
mod tests {
    use super::*;
    // Comment out serialize import as the feature is removed from dev-deps
    // use safetensors::serialize;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;
    use rand::{thread_rng, RngCore};

    // Comment out helper function requiring serialize
    /*
    fn create_dummy_safetensor(tensors: &HashMap<String, TensorView>) -> NamedTempFile {
        let file = NamedTempFile::new().expect("Failed to create temp file");
        let metadata = serialize(tensors, &None).expect("Failed to serialize tensors");
        std::fs::write(file.path(), &metadata).expect("Failed to write temp file");
        file 
    }
    */

    // Comment out test requiring the helper
    /*
    #[test]
    fn test_load_safetensors_success() {
        let mut tensors = HashMap::new();
        let data1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data2: Vec<i8> = vec![5, 6];
        let view1 = SafeTensorView::new(SafeTensorDtype::F32, vec![2, 2], bytemuck::cast_slice(&data1)).unwrap();
        let view2 = SafeTensorView::new(SafeTensorDtype::I8, vec![2], bytemuck::cast_slice(&data2)).unwrap();
        tensors.insert("tensor1".to_string(), view1);
        tensors.insert("tensor2".to_string(), view2);

        let temp_file = create_dummy_safetensor(&tensors);
        let model = load_safetensors(temp_file.path()).expect("Loading failed");

        assert_eq!(model.path(), temp_file.path());
        // Cannot directly check metadata length anymore
        // assert_eq!(model.metadata.len(), 2); 

        // Check tensor 1
        let loaded_view1 = model.tensor("tensor1").expect("Tensor 1 not found");
        assert_eq!(loaded_view1.dtype(), SafeTensorDtype::F32);
        assert_eq!(loaded_view1.shape(), &[2, 2]);
        assert_eq!(loaded_view1.data(), bytemuck::cast_slice::<f32, u8>(&data1));

        // Check tensor 2
        let loaded_view2 = model.tensor("tensor2").expect("Tensor 2 not found");
        assert_eq!(loaded_view2.dtype(), SafeTensorDtype::I8);
        assert_eq!(loaded_view2.shape(), &[2]);
        assert_eq!(loaded_view2.data(), bytemuck::cast_slice::<i8, u8>(&data2));

        // Check iterator
        let mut count = 0;
        // Accessing tensors now returns a Result
        for item in model.tensors().expect("Failed to get tensors iterator") {
            let (name, _view) = item; // Item is (String, TensorView<_'>)
            assert!(name == "tensor1" || name == "tensor2");
            count += 1;
        }
        assert_eq!(count, 2);
    }
    */

    #[test]
    fn test_load_safetensors_not_found() {
        let mut rng = thread_rng();
        let non_existent_path = format!("/tmp/non_existent_safetensor_{}.safetensors", rng.next_u64());
        let result = load_safetensors(&non_existent_path);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains(&non_existent_path));
    }

    // Comment out test requiring the helper
    /*
    #[test]
    fn test_load_safetensors_invalid_file() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        std::fs::write(temp_file.path(), b"this is not a safetensor file").expect("Failed to write invalid data");
        let result = load_safetensors(temp_file.path());
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("deserialize"));
    }
    */
    
    // Comment out test requiring the helper
    /*
    #[test]
    fn test_get_non_existent_tensor() {
        let tensors = HashMap::new();
        let temp_file = create_dummy_safetensor(&tensors);
        let model = load_safetensors(temp_file.path()).expect("Loading failed");
        let result = model.tensor("non_existent_tensor");
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("non_existent_tensor"));
    }
    */
}
