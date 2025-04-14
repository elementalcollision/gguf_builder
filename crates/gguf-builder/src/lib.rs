// use half::f16;
use std::collections::HashMap;
use thiserror::Error;
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{Seek, SeekFrom, Write};
use std::time::Instant;

// Constants based on GGUF spec (likely V3)
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const GGUF_VERSION: u32 = 3; // Target GGUF version 3
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGUFValueType {
    // IMPORTANT: The order and u32 representation MUST match gguf spec
    // Check https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

// Function to safely get the u32 representation
fn gguf_value_type_to_u32(val_type: GGUFValueType) -> u32 {
    // This relies on the explicit assignments in the enum definition matching the spec
    val_type as u32
}

#[derive(Debug, Clone, PartialEq)]
pub enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    //Potentially add complex types like f16 later if needed in metadata
    //F16(f16),
}

// Helper to get the GGUFValueType from a GGUFValue instance
impl GGUFValue {
    fn get_type(&self) -> GGUFValueType {
         match self {
            GGUFValue::Uint8(_) => GGUFValueType::Uint8,
            GGUFValue::Int8(_) => GGUFValueType::Int8,
            GGUFValue::Uint16(_) => GGUFValueType::Uint16,
            GGUFValue::Int16(_) => GGUFValueType::Int16,
            GGUFValue::Uint32(_) => GGUFValueType::Uint32,
            GGUFValue::Int32(_) => GGUFValueType::Int32,
            GGUFValue::Float32(_) => GGUFValueType::Float32,
            GGUFValue::Bool(_) => GGUFValueType::Bool,
            GGUFValue::String(_) => GGUFValueType::String,
            GGUFValue::Array(_) => GGUFValueType::Array,
            GGUFValue::Uint64(_) => GGUFValueType::Uint64,
            GGUFValue::Int64(_) => GGUFValueType::Int64,
            GGUFValue::Float64(_) => GGUFValueType::Float64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl Default for GGUFHeader {
    fn default() -> Self {
        GGUFHeader {
            magic: GGUF_MAGIC,
            version: GGUF_VERSION,
            tensor_count: 0, // Will be updated later
            metadata_kv_count: 0, // Will be updated later
        }
    }
}

// Basic structure for the writer
#[derive(Default)]
pub struct GGUFWriter {
    header: GGUFHeader,
    metadata: HashMap<String, GGUFValue>,
    tensor_infos: Vec<TensorInfo>,
    // Association of TensorInfo with its data needs to be handled.
    // We can modify add_tensor_info or require data in write().
    // For now, assuming `write` receives corresponding data.
}

impl GGUFWriter {
    /// Creates a new, empty GGUFWriter.
    pub fn new() -> Self {
        log::debug!("Creating new GGUFWriter (v{})", GGUF_VERSION);
        GGUFWriter::default()
    }

    /// Adds a metadata key-value pair.
    /// Replaces existing value if key already exists.
    pub fn add_metadata(&mut self, key: String, value: GGUFValue) {
        log::trace!("Adding metadata: key='{}', value_type='{:?}'", key, value.get_type());
        self.metadata.insert(key, value);
    }

    /// Adds tensor information (metadata only).
    /// The actual tensor data must be provided later during the final write step,
    /// in the same order the info was added.
    /// The offset field in the provided TensorInfo will be ignored and recalculated.
    pub fn add_tensor_info(&mut self, mut info: TensorInfo) {
        log::trace!("Adding tensor info: name='{}', type='{:?}', dims={:?}", info.name, info.ggml_type, info.dimensions);
        // Clear offset as it will be calculated during write
        info.offset = 0;
        self.tensor_infos.push(info);
    }

    /// Writes the complete GGUF file to the provided writer.
    ///
    /// The `tensor_data` slice must contain byte slices corresponding to each
    /// tensor added via `add_tensor_info`, in the same order.
    pub fn write<W: Write + Seek>(
        &mut self,
        writer: &mut W,
        tensor_data: &[&[u8]],
    ) -> Result<(), GGUFWriteError> {
        let total_start_time = Instant::now();
        log::info!("Starting GGUF file write...");

        if self.tensor_infos.len() != tensor_data.len() {
             log::error!(
                "Tensor info count ({}) does not match tensor data count ({}).",
                self.tensor_infos.len(), tensor_data.len()
            );
            return Err(GGUFWriteError::TensorCountMismatch {
                info_count: self.tensor_infos.len(),
                data_count: tensor_data.len(),
            });
        }

        // 1. Update header counts
        self.header.metadata_kv_count = self.metadata.len() as u64;
        self.header.tensor_count = self.tensor_infos.len() as u64;
        log::debug!("Header prepared: tensor_count={}, metadata_kv_count={}", self.header.tensor_count, self.header.metadata_kv_count);


        // --- Write Header (Initial) ---
        let header_start_time = Instant::now();
        write_header(writer, &self.header)?;
        let initial_header_offset = writer.stream_position()?;
        log::debug!("Initial header written ({} bytes) in {:?}", initial_header_offset, header_start_time.elapsed());

        // --- Write Metadata ---
        let metadata_start_time = Instant::now();
        log::debug!("Writing {} metadata pairs...", self.metadata.len());
        for (key, value) in &self.metadata {
            log::trace!("Writing metadata kv: {}", key);
            write_metadata_kv(writer, key, value)?;
        }
        let metadata_bytes_written = writer.stream_position()? - initial_header_offset;
        log::debug!("Metadata section written ({} bytes) in {:?}", metadata_bytes_written, metadata_start_time.elapsed());


        // --- Calculate Tensor Info Section Size & Data Start Offset ---
        let calc_start_time = Instant::now();
        let mut tensor_info_size_calculator = std::io::Cursor::new(Vec::new());
        for info in &self.tensor_infos {
            // Simulate writing tensor info to calculate size
            write_tensor_info(&mut tensor_info_size_calculator, info)?;
        }
        let tensor_info_total_size = tensor_info_size_calculator.position();

        let current_pos_after_metadata = initial_header_offset + metadata_bytes_written;
        let tensor_data_start_pos = current_pos_after_metadata + tensor_info_total_size;
        let alignment = GGUF_DEFAULT_ALIGNMENT;
        let padding_before_data = (alignment - (tensor_data_start_pos % alignment)) % alignment;
        let final_tensor_data_start_offset = tensor_data_start_pos + padding_before_data;
        log::debug!(
            "Tensor info section size calculated: {} bytes. Tensor data section will start at offset: {} (includes {} padding bytes). Calculation took {:?}",
            tensor_info_total_size, final_tensor_data_start_offset, padding_before_data, calc_start_time.elapsed()
        );

        // --- Write Tensor Infos ---
        let tensor_info_start_time = Instant::now();
        log::debug!("Writing {} tensor info entries...", self.tensor_infos.len());
        let mut next_tensor_offset = final_tensor_data_start_offset;
        for (i, info) in self.tensor_infos.iter_mut().enumerate() {
             // Assign calculated offset
             info.offset = next_tensor_offset;
             log::trace!("Writing tensor info for '{}', calculated offset: {}", info.name, info.offset);
             write_tensor_info(writer, info)?;

             // Calculate offset for the *next* tensor
             let data_len = tensor_data[i].len() as u64;
             let next_pos_unaligned = next_tensor_offset + data_len;
             let padding_for_next = (alignment - (next_pos_unaligned % alignment)) % alignment;
             next_tensor_offset = next_pos_unaligned + padding_for_next;
        }
         let tensor_info_bytes_written = writer.stream_position()? - current_pos_after_metadata;
         // Sanity check calculation vs actual size
         if tensor_info_bytes_written != tensor_info_total_size {
            log::warn!(
                "Calculated tensor info size ({}) != actual written size ({}). This might indicate a bug.",
                tensor_info_total_size, tensor_info_bytes_written
            );
            // Potentially return an error here if strict checking is needed
         }
        log::debug!("Tensor info section written ({} bytes) in {:?}", tensor_info_bytes_written, tensor_info_start_time.elapsed());

        // --- Write Padding Before Data Section ---
        let padding1_start_time = Instant::now();
        let padding_bytes1 = write_padding(writer)?;
        log::debug!("Initial padding written ({} bytes) in {:?}", padding_bytes1, padding1_start_time.elapsed());


        // --- Write Tensor Data ---
        let tensor_data_start_time = Instant::now();
        log::debug!("Writing tensor data...");
        let mut total_tensor_bytes: u64 = 0;
        let mut total_padding_bytes: u64 = padding_bytes1; // Start with initial padding

        for (i, data) in tensor_data.iter().enumerate() {
            let tensor_write_start = Instant::now();
            let current_pos = writer.stream_position()?;
            // Optional sanity check against calculated offset
            if current_pos != self.tensor_infos[i].offset {
                 log::warn!(
                    "Offset mismatch for tensor '{}': expected {}, actual {}. Alignment issue?",
                    self.tensor_infos[i].name, self.tensor_infos[i].offset, current_pos
                 );
                 // Could seek here if needed: writer.seek(SeekFrom::Start(self.tensor_infos[i].offset))?;
            }

            log::trace!(
                "Writing tensor {} ({}) data ({} bytes) at offset {}",
                i, self.tensor_infos[i].name, data.len(), current_pos
            );
            write_tensor_data(writer, data)?;
            total_tensor_bytes += data.len() as u64;

            // Pad after each tensor except the last
            let padding_bytes_after = if i < tensor_data.len() - 1 {
                write_padding(writer)?
            } else {
                0
            };
            total_padding_bytes += padding_bytes_after;
            log::trace!(
                "Tensor '{}' written in {:?}, padded with {} bytes after.",
                self.tensor_infos[i].name, tensor_write_start.elapsed(), padding_bytes_after
            );

        }
        log::debug!(
            "Tensor data section written ({} tensors, {} data bytes, {} padding bytes) in {:?}",
            self.tensor_infos.len(), total_tensor_bytes, total_padding_bytes, tensor_data_start_time.elapsed()
        );

        // --- Final Header Update ---
        let final_header_start_time = Instant::now();
        let final_pos = writer.stream_position()?;
        log::debug!("Seeking back to file start to finalize header...");
        writer.seek(SeekFrom::Start(0))?;
        write_header(writer, &self.header)?;
        log::debug!("Final header written in {:?}", final_header_start_time.elapsed());
        writer.seek(SeekFrom::Start(final_pos))?; // Seek back to the end
        log::debug!("Writer position reset to end of file: {}", final_pos);


        log::info!(
            "GGUF file write completed successfully in {:?}. Total size: {} bytes.",
            total_start_time.elapsed(), final_pos
        );
        Ok(())
    }
}

// Placeholder for TensorInfo
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GGMLTypeRenamed,
    pub offset: u64,
}

// GGMLType enum definition (ensure #[repr(u32)])
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // Allow original names for now, will rename below
pub enum GGMLType {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
    Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9, Q2_K = 10, // Renamed below
    Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15, // Renamed below
    I8 = 16, I16 = 17, I32 = 18, I64 = 19, F64 = 20, BF16 = 21,
}

// Rename variants to satisfy linter - this requires updating all usage sites
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGMLTypeRenamed {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2, // Keeping these as is, convention seems mixed
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    BF16 = 21,
}

// Error types for the writing process
#[derive(Error, Debug)]
pub enum GGUFWriteError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid metadata value type used in array or nested array detected")]
    InvalidMetadataValueType,
    #[error("Tensor info count ({info_count}) does not match tensor data count ({data_count})")]
    TensorCountMismatch { info_count: usize, data_count: usize },
    #[error("Tensor '{0}' has too many dimensions ({1} > max {2})")]
    TooManyDimensions(String, usize, usize),
    #[error("Unsupported GGUFValue for direct writing: {0:?}")] // Example for future use
    UnsupportedValue(GGUFValueType),
}

// Helper function to write GGUF strings
fn write_string<W: Write>(writer: &mut W, s: &str) -> Result<(), GGUFWriteError> {
    let bytes = s.as_bytes();
    writer.write_u64::<LittleEndian>(bytes.len() as u64)?;
    writer.write_all(bytes)?;
    Ok(())
}

// Helper function to write GGUF metadata values
fn write_metadata_value<W: Write>(writer: &mut W, value: &GGUFValue) -> Result<(), GGUFWriteError> {
     match value {
        GGUFValue::Uint8(v) => writer.write_u8(*v)?,
        GGUFValue::Int8(v) => writer.write_i8(*v)?,
        GGUFValue::Uint16(v) => writer.write_u16::<LittleEndian>(*v)?,
        GGUFValue::Int16(v) => writer.write_i16::<LittleEndian>(*v)?,
        GGUFValue::Uint32(v) => writer.write_u32::<LittleEndian>(*v)?,
        GGUFValue::Int32(v) => writer.write_i32::<LittleEndian>(*v)?,
        GGUFValue::Float32(v) => writer.write_f32::<LittleEndian>(*v)?,
        GGUFValue::Uint64(v) => writer.write_u64::<LittleEndian>(*v)?,
        GGUFValue::Int64(v) => writer.write_i64::<LittleEndian>(*v)?,
        GGUFValue::Float64(v) => writer.write_f64::<LittleEndian>(*v)?,
        GGUFValue::Bool(v) => writer.write_u8(if *v { 1 } else { 0 })?,
        GGUFValue::String(s) => write_string(writer, s)?,
        GGUFValue::Array(arr) => {
            // Array writing: type, count, then each element
            if arr.is_empty() {
                writer.write_u32::<LittleEndian>(gguf_value_type_to_u32(GGUFValueType::Uint8))?; // Use type 0 for empty array
                writer.write_u64::<LittleEndian>(0)?;
            } else {
                let element_type = arr[0].get_type();
                writer.write_u32::<LittleEndian>(gguf_value_type_to_u32(element_type))?;
                writer.write_u64::<LittleEndian>(arr.len() as u64)?;
                for element in arr {
                    // Optional: Check if element.get_type() == element_type
                    if element.get_type() != element_type {
                         log::error!("Inconsistent types found in metadata array. Expected {:?}, found {:?}.", element_type, element.get_type());
                         return Err(GGUFWriteError::InvalidMetadataValueType); 
                    }
                    write_metadata_value(writer, element)?;
                }
            }
        }
    }
    Ok(())
}

/// Writes a single metadata key-value pair.
fn write_metadata_kv<W: Write>(writer: &mut W, key: &str, value: &GGUFValue) -> Result<(), GGUFWriteError> {
    write_string(writer, key)?;
    let value_type = value.get_type();
    writer.write_u32::<LittleEndian>(gguf_value_type_to_u32(value_type))?;
    write_metadata_value(writer, value)?;
    Ok(())
}

/// Writes the GGUF header to the provided writer.
fn write_header<W: Write>(writer: &mut W, header: &GGUFHeader) -> Result<(), GGUFWriteError> {
    writer.write_u32::<LittleEndian>(header.magic)?;
    writer.write_u32::<LittleEndian>(header.version)?;
    writer.write_u64::<LittleEndian>(header.tensor_count)?;
    writer.write_u64::<LittleEndian>(header.metadata_kv_count)?;
    Ok(())
}

/// Writes a single tensor info entry.
fn write_tensor_info<W: Write>(writer: &mut W, info: &TensorInfo) -> Result<(), GGUFWriteError> {
    write_string(writer, &info.name)?;
    writer.write_u32::<LittleEndian>(info.dimensions.len() as u32)?;
    for &dim in info.dimensions.iter().rev() { // Write dimensions in reverse order
        writer.write_u64::<LittleEndian>(dim)?;
    }
    writer.write_u32::<LittleEndian>(info.ggml_type as u32)?;
    writer.write_u64::<LittleEndian>(info.offset)?;
    Ok(())
}

/// Writes padding bytes to align the writer to GGUF_DEFAULT_ALIGNMENT.
fn write_padding<W: Write + Seek>(writer: &mut W) -> Result<u64, GGUFWriteError> {
    let current_pos = writer.stream_position()?;
    let alignment = GGUF_DEFAULT_ALIGNMENT;
    let padding = (alignment - (current_pos % alignment)) % alignment;
    if padding > 0 {
        let zero_buf = [0u8; 32]; // Max padding needed is alignment - 1
        let mut remaining = padding;
        while remaining > 0 {
             let to_write = std::cmp::min(remaining, zero_buf.len() as u64);
             writer.write_all(&zero_buf[..to_write as usize])?;
             remaining -= to_write;
        }
    }
    Ok(padding)
}

/// Writes the raw tensor data bytes.
fn write_tensor_data<W: Write>(writer: &mut W, data: &[u8]) -> Result<(), GGUFWriteError> {
    writer.write_all(data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*; // Import items from parent module
    use std::io::Cursor;
    use byteorder::ReadBytesExt; // For reading back values in tests

    #[test]
    fn test_write_string() {
        let mut buf = Cursor::new(Vec::new());
        let test_str = "hello";
        write_string(&mut buf, test_str).unwrap();

        let written_bytes = buf.into_inner();
        let mut reader = Cursor::new(written_bytes);

        let len = reader.read_u64::<LittleEndian>().unwrap();
        assert_eq!(len, test_str.len() as u64);

        let mut str_bytes = vec![0u8; len as usize];
        reader.read_exact(&mut str_bytes).unwrap();
        assert_eq!(str_bytes, test_str.as_bytes());
    }

    #[test]
    fn test_write_header() {
        let mut buf = Cursor::new(Vec::new());
        let header = GGUFHeader {
            magic: GGUF_MAGIC,
            version: GGUF_VERSION,
            tensor_count: 5,
            metadata_kv_count: 10,
        };
        write_header(&mut buf, &header).unwrap();

        let written_bytes = buf.into_inner();
        assert_eq!(written_bytes.len(), 4 + 4 + 8 + 8); // Sizes of u32, u32, u64, u64
        let mut reader = Cursor::new(written_bytes);

        assert_eq!(reader.read_u32::<LittleEndian>().unwrap(), GGUF_MAGIC);
        assert_eq!(reader.read_u32::<LittleEndian>().unwrap(), GGUF_VERSION);
        assert_eq!(reader.read_u64::<LittleEndian>().unwrap(), 5);
        assert_eq!(reader.read_u64::<LittleEndian>().unwrap(), 10);
    }

    #[test]
    fn test_write_metadata_kv_simple() {
        let mut buf = Cursor::new(Vec::new());
        let key = "my_key";
        let value = GGUFValue::Uint32(12345);
        let value_type = value.get_type();

        write_metadata_kv(&mut buf, key, &value).unwrap();

        let written_bytes = buf.into_inner();
        let mut reader = Cursor::new(written_bytes);

        // Read key back
        let key_len = reader.read_u64::<LittleEndian>().unwrap();
        assert_eq!(key_len, key.len() as u64);
        let mut key_bytes = vec![0u8; key_len as usize];
        reader.read_exact(&mut key_bytes).unwrap();
        assert_eq!(key_bytes, key.as_bytes());

        // Read value type and value
        let type_id = reader.read_u32::<LittleEndian>().unwrap();
        assert_eq!(type_id, gguf_value_type_to_u32(value_type));
        let val_read = reader.read_u32::<LittleEndian>().unwrap();
        assert_eq!(val_read, 12345);
    }

     #[test]
    fn test_write_metadata_kv_array() {
        let mut buf = Cursor::new(Vec::new());
        let key = "my_array";
        let value = GGUFValue::Array(vec![
            GGUFValue::Int16(10), 
            GGUFValue::Int16(-20),
        ]);
        let value_type = value.get_type(); // Should be Array
        let element_type = GGUFValueType::Int16;

        write_metadata_kv(&mut buf, key, &value).unwrap();

        let written_bytes = buf.into_inner();
        let mut reader = Cursor::new(written_bytes);

        // Read key
        let key_len = reader.read_u64::<LittleEndian>().unwrap();
        reader.seek(SeekFrom::Current(key_len as i64)).unwrap(); // Skip key bytes
        
        // Read value type (should be Array)
        let type_id = reader.read_u32::<LittleEndian>().unwrap();
        assert_eq!(type_id, gguf_value_type_to_u32(value_type));

        // Read array header (element type, count)
        let arr_element_type_id = reader.read_u32::<LittleEndian>().unwrap();
        let arr_count = reader.read_u64::<LittleEndian>().unwrap();
        assert_eq!(arr_element_type_id, gguf_value_type_to_u32(element_type));
        assert_eq!(arr_count, 2);

        // Read array elements
        assert_eq!(reader.read_i16::<LittleEndian>().unwrap(), 10);
        assert_eq!(reader.read_i16::<LittleEndian>().unwrap(), -20);
    }


    #[test]
    fn test_write_tensor_info() {
        let info = TensorInfo {
            name: "test.tensor".to_string(),
            dimensions: vec![10, 20, 30], 
            ggml_type: GGMLTypeRenamed::F32, // Use renamed enum
            offset: 123456789,
        };

        let mut buffer = std::io::Cursor::new(Vec::new());
        write_tensor_info(&mut buffer, &info).unwrap();
        buffer.seek(SeekFrom::Start(0)).unwrap();

        // Read back and verify
        let name_len = buffer.read_u64::<LittleEndian>().unwrap();
        assert_eq!(name_len, info.name.len() as u64);
        let mut name_bytes = vec![0u8; name_len as usize];
        buffer.read_exact(&mut name_bytes).unwrap();
        assert_eq!(String::from_utf8(name_bytes).unwrap(), info.name);

        let n_dims = buffer.read_u32::<LittleEndian>().unwrap();
        assert_eq!(n_dims, info.dimensions.len() as u32);
        let mut read_dims = vec![0u64; n_dims as usize];
        for i in 0..(n_dims as usize) {
            read_dims[i] = buffer.read_u64::<LittleEndian>().unwrap();
        }
        // Remember they are written in reverse
        assert_eq!(read_dims, vec![30, 20, 10]);

        let ggml_type_u32 = buffer.read_u32::<LittleEndian>().unwrap();
        assert_eq!(ggml_type_u32, info.ggml_type as u32);

        let offset = buffer.read_u64::<LittleEndian>().unwrap();
        assert_eq!(offset, info.offset);
    }

    #[test]
    fn test_write_padding() {
        let mut buf = Cursor::new(Vec::new());
        buf.write_all(&[1, 2, 3, 4, 5]).unwrap(); // Current pos = 5
        assert_eq!(buf.position(), 5);
        
        let padding_written = write_padding(&mut buf).unwrap();
        let expected_padding = (GGUF_DEFAULT_ALIGNMENT - (5 % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT;
        assert_eq!(padding_written, expected_padding);
        assert_eq!(buf.position(), 5 + expected_padding);

        // Check if already aligned
        let padding_written_2 = write_padding(&mut buf).unwrap();
        assert_eq!(padding_written_2, 0);
        assert_eq!(buf.position(), 5 + expected_padding); // Position shouldn't change
    }
    
    #[test]
    fn test_gguf_writer_integration() {
        let mut writer = GGUFWriter::new();

        writer.add_metadata("test.int".to_string(), GGUFValue::Int32(123));
        writer.add_metadata("test.bool".to_string(), GGUFValue::Bool(true));
        writer.add_metadata("test.string".to_string(), GGUFValue::String("hello gguf".to_string()));

        let tensor1_info = TensorInfo {
            name: "tensor1".to_string(),
            dimensions: vec![100, 200],
            ggml_type: GGMLTypeRenamed::F32, // Use renamed enum
            offset: 0, // Will be calculated
        };
        let tensor2_info = TensorInfo {
            name: "tensor2".to_string(),
            dimensions: vec![50],
            ggml_type: GGMLTypeRenamed::Q8_0, // Use renamed enum
            offset: 0, // Will be calculated
        };

        writer.add_tensor_info(tensor1_info.clone());
        writer.add_tensor_info(tensor2_info.clone());

        let tensor1_data: Vec<f32> = vec![1.0f32; 100 * 200];
        let tensor2_data: Vec<i8> = vec![5i8; 50];
        let tensor_data_slices: Vec<&[u8]> = vec![
            bytemuck::cast_slice(&tensor1_data),
            bytemuck::cast_slice(&tensor2_data),
        ];

        let mut buffer = std::io::Cursor::new(Vec::new());
        writer.write(&mut buffer, &tensor_data_slices).unwrap();

        // TODO: Add more robust validation here by reading back the GGUF file
        // For now, just check if it wrote something without erroring
        assert!(buffer.position() > 0);
        println!("Integration test wrote {} bytes.", buffer.position());

        // Basic header check
        buffer.seek(SeekFrom::Start(0)).unwrap();
        assert_eq!(buffer.read_u32::<LittleEndian>().unwrap(), GGUF_MAGIC);
        assert_eq!(buffer.read_u32::<LittleEndian>().unwrap(), GGUF_VERSION);
        assert_eq!(buffer.read_u64::<LittleEndian>().unwrap(), 2); // tensor_count
        assert_eq!(buffer.read_u64::<LittleEndian>().unwrap(), 3); // metadata_kv_count

    }
}
