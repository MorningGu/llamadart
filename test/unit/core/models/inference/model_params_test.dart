import 'package:llamadart/src/core/models/config/gpu_backend.dart';
import 'package:llamadart/src/core/models/inference/model_params.dart';
import 'package:test/test.dart';

void main() {
  test('ModelParams defaults preserve legacy context batching behavior', () {
    const params = ModelParams();

    expect(params.contextSize, 4096);
    expect(params.gpuLayers, ModelParams.maxGpuLayers);
    expect(params.preferredBackend, GpuBackend.auto);
    expect(params.splitMode, ModelSplitMode.layer);
    expect(params.mainGpu, 0);
    expect(params.chatTemplate, isNull);
    expect(params.numberOfThreads, 0);
    expect(params.numberOfThreadsBatch, 0);
    expect(params.batchSize, 0);
    expect(params.microBatchSize, 0);
    expect(params.maxParallelSequences, 1);
    expect(ModelParams.maxGpuLayers, 999);
  });

  test('ModelParams copyWith updates selected fields', () {
    const params = ModelParams(contextSize: 1024);
    final updated = params.copyWith(
      gpuLayers: 2,
      preferredBackend: GpuBackend.metal,
      splitMode: ModelSplitMode.none,
      mainGpu: 1,
      batchSize: 256,
      microBatchSize: 64,
      maxParallelSequences: 8,
    );

    expect(updated.contextSize, 1024);
    expect(updated.gpuLayers, 2);
    expect(updated.preferredBackend, GpuBackend.metal);
    expect(updated.splitMode, ModelSplitMode.none);
    expect(updated.mainGpu, 1);
    expect(updated.batchSize, 256);
    expect(updated.microBatchSize, 64);
    expect(updated.maxParallelSequences, 8);
  });

  test('ModelParams copyWith preserves unspecified fields', () {
    const original = ModelParams(
      contextSize: 3072,
      gpuLayers: 8,
      preferredBackend: GpuBackend.cuda,
      splitMode: ModelSplitMode.row,
      mainGpu: 2,
      chatTemplate: 'custom-template',
      numberOfThreads: 6,
      numberOfThreadsBatch: 4,
      batchSize: 512,
      microBatchSize: 128,
      maxParallelSequences: 4,
    );

    final updated = original.copyWith(gpuLayers: 12);

    expect(updated.contextSize, 3072);
    expect(updated.gpuLayers, 12);
    expect(updated.preferredBackend, GpuBackend.cuda);
    expect(updated.splitMode, ModelSplitMode.row);
    expect(updated.mainGpu, 2);
    expect(updated.chatTemplate, 'custom-template');
    expect(updated.numberOfThreads, 6);
    expect(updated.numberOfThreadsBatch, 4);
    expect(updated.batchSize, 512);
    expect(updated.microBatchSize, 128);
    expect(updated.maxParallelSequences, 4);
  });
}
