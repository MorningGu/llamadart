import 'package:llamadart/src/backends/llama_cpp/bindings.dart';
import 'package:llamadart/src/backends/llama_cpp/load_param_helpers.dart';
import 'package:llamadart/src/core/models/config/flash_attention.dart';
import 'package:llamadart/src/core/models/config/kv_cache_type.dart';
import 'package:test/test.dart';

void main() {
  group('ggmlTypeFor', () {
    test('f16 → GGML_TYPE_F16', () {
      expect(ggmlTypeFor(KvCacheType.f16), ggml_type.GGML_TYPE_F16);
    });

    test('q8_0 → GGML_TYPE_Q8_0', () {
      expect(ggmlTypeFor(KvCacheType.q8_0), ggml_type.GGML_TYPE_Q8_0);
    });

    test('q4_0 → GGML_TYPE_Q4_0', () {
      expect(ggmlTypeFor(KvCacheType.q4_0), ggml_type.GGML_TYPE_Q4_0);
    });
  });

  group('resolveFlashAttention', () {
    test('auto + F16/F16 → auto (no promotion needed)', () {
      expect(
        resolveFlashAttention(
          requested: FlashAttention.auto,
          cacheTypeK: KvCacheType.f16,
          cacheTypeV: KvCacheType.f16,
        ),
        FlashAttention.auto,
      );
    });

    test('auto + Q8_0 K → enabled (auto-promote)', () {
      expect(
        resolveFlashAttention(
          requested: FlashAttention.auto,
          cacheTypeK: KvCacheType.q8_0,
          cacheTypeV: KvCacheType.f16,
        ),
        FlashAttention.enabled,
      );
    });

    test('auto + Q4_0 V → enabled (auto-promote)', () {
      expect(
        resolveFlashAttention(
          requested: FlashAttention.auto,
          cacheTypeK: KvCacheType.f16,
          cacheTypeV: KvCacheType.q4_0,
        ),
        FlashAttention.enabled,
      );
    });

    test('auto + Q8_0 K/V → enabled (auto-promote)', () {
      expect(
        resolveFlashAttention(
          requested: FlashAttention.auto,
          cacheTypeK: KvCacheType.q8_0,
          cacheTypeV: KvCacheType.q8_0,
        ),
        FlashAttention.enabled,
      );
    });

    test('explicit enabled passes through unchanged regardless of KV', () {
      for (final k in KvCacheType.values) {
        for (final v in KvCacheType.values) {
          expect(
            resolveFlashAttention(
              requested: FlashAttention.enabled,
              cacheTypeK: k,
              cacheTypeV: v,
            ),
            FlashAttention.enabled,
            reason: 'enabled should stay enabled for k=$k v=$v',
          );
        }
      }
    });

    test('explicit disabled passes through unchanged for F16 (no promotion)',
        () {
      // The disabled+non-F16 combination is rejected by ModelParams's
      // constructor; this helper isn't responsible for that validation.
      // For F16/F16, disabled is legal and should pass through.
      expect(
        resolveFlashAttention(
          requested: FlashAttention.disabled,
          cacheTypeK: KvCacheType.f16,
          cacheTypeV: KvCacheType.f16,
        ),
        FlashAttention.disabled,
      );
    });
  });
}
