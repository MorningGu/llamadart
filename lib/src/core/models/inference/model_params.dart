import '../config/gpu_backend.dart';

import '../config/lora_config.dart';

/// Configuration parameters for loading a Llama model.
///
/// These parameters affect the initial model loading and context allocation.
/// Most of these cannot be changed once the model is loaded.
///
/// Context batching fields in this class mirror llama.cpp semantics:
/// `n_batch` is the logical max decode batch, and `n_ubatch` is the
/// physical micro-batch size.
///
/// Example:
/// ```dart
/// final params = ModelParams(
///   contextSize: 4096,
///   gpuLayers: 33, // Offload 33 layers to GPU
///   mainGpu: 1, // Use the second GPU device as the primary device
/// );
/// await engine.loadModel('path/to/model.gguf', modelParams: params);
/// ```
class ModelParams {
  /// Context size (n_ctx) in tokens.
  final int contextSize;

  /// Number of model layers to offload to the GPU (n_gpu_layers).
  final int gpuLayers;

  /// Preferred GPU backend for inference.
  final GpuBackend preferredBackend;

  /// Primary GPU device index for model loading.
  ///
  /// This is passed through to llama.cpp `llama_model_params.main_gpu`.
  /// Backend-specific device ordering is defined by llama.cpp and the active
  /// backend. Defaults to 0 to preserve llama.cpp's default behavior.
  final int mainGpu;

  /// Initial LoRA adapters to load along with the model.
  final List<LoraAdapterConfig> loras;

  /// Optional chat template to override the model's default template.
  final String? chatTemplate;

  /// Number of threads to use for generation (n_threads).
  ///
  /// Set to 0 for automatic detection.
  final int numberOfThreads;

  /// Number of threads to use for batch processing (n_threads_batch).
  ///
  /// Set to 0 for automatic detection.
  final int numberOfThreadsBatch;

  /// Maximum prompt/eval tokens per decode call (n_batch).
  ///
  /// Mirrors llama.cpp `llama_context_params.n_batch` (logical max batch).
  /// See also upstream CLI flag `--batch-size`.
  ///
  /// Set to 0 (or negative) to default to [contextSize].
  final int batchSize;

  /// Micro-batch size used by backend schedulers (n_ubatch).
  ///
  /// Mirrors llama.cpp `llama_context_params.n_ubatch` (physical max batch).
  /// See also upstream CLI flag `--ubatch-size`.
  ///
  /// Set to 0 (or negative) to default to [batchSize].
  final int microBatchSize;

  /// Maximum parallel sequence slots in context memory (n_seq_max).
  ///
  /// Values greater than 1 allow true multi-sequence batching (for example,
  /// embedding batches with independent sequence IDs).
  ///
  /// Set to 1 to preserve single-sequence behavior.
  final int maxParallelSequences;

  /// Maximum number of GPU layers to safely offload all layers.
  static const int maxGpuLayers = 999;

  /// Creates configuration for the model.
  const ModelParams({
    this.contextSize = 4096,
    this.gpuLayers = maxGpuLayers,
    this.preferredBackend = GpuBackend.auto,
    this.mainGpu = 0,
    this.loras = const [],
    this.chatTemplate,
    this.numberOfThreads = 0,
    this.numberOfThreadsBatch = 0,
    this.batchSize = 0,
    this.microBatchSize = 0,
    this.maxParallelSequences = 1,
  });

  /// Creates a copy of this [ModelParams] with updated fields.
  ModelParams copyWith({
    int? contextSize,
    int? gpuLayers,
    GpuBackend? preferredBackend,
    int? mainGpu,
    List<LoraAdapterConfig>? loras,
    String? chatTemplate,
    int? numberOfThreads,
    int? numberOfThreadsBatch,
    int? batchSize,
    int? microBatchSize,
    int? maxParallelSequences,
  }) {
    return ModelParams(
      contextSize: contextSize ?? this.contextSize,
      gpuLayers: gpuLayers ?? this.gpuLayers,
      preferredBackend: preferredBackend ?? this.preferredBackend,
      mainGpu: mainGpu ?? this.mainGpu,
      loras: loras ?? this.loras,
      chatTemplate: chatTemplate ?? this.chatTemplate,
      numberOfThreads: numberOfThreads ?? this.numberOfThreads,
      numberOfThreadsBatch: numberOfThreadsBatch ?? this.numberOfThreadsBatch,
      batchSize: batchSize ?? this.batchSize,
      microBatchSize: microBatchSize ?? this.microBatchSize,
      maxParallelSequences: maxParallelSequences ?? this.maxParallelSequences,
    );
  }
}
