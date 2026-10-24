//! A from-scratch ONNX runtime specialized for the Paradee-8M v1.1 int8 TTS
//! model. No dependencies: the protobuf reader, graph loader, tensor type and
//! every op kernel are implemented here.
//!
//! The numeric kernels index slices directly rather than using iterator chains.
//! That is deliberate: the index form is what the optimizer vectorizes, and it
//! keeps the tap loops readable against the ONNX spec. The lints below are
//! allowed crate-wide for that reason.

// Index loops in the kernels are clearer and vectorize better than the
// iterator equivalents clippy suggests.
#![allow(clippy::needless_range_loop)]
// `chunks_exact(4)` / `chunks_exact(8)` is the idiomatic way to reinterpret a
// byte buffer as f32/f64; the suggested alternative is not clearer.
#![allow(clippy::chunks_exact_to_as_chunks)]
// Kernels take their operands as separate slices; grouping them into a struct
// would obscure the ONNX operand order.
#![allow(clippy::too_many_arguments)]
// `T::len` is the element count, not a container length, so `is_empty` would be
// misleading.
#![allow(clippy::len_without_is_empty)]
// The erf coefficients and the radix-5 twiddles are written to full f32
// precision on purpose; rounding them to the shortest representation that
// round-trips would change the numerics.
#![allow(clippy::excessive_precision)]

pub mod onnx;
pub mod pb;
pub mod quant;
pub mod tensor;
