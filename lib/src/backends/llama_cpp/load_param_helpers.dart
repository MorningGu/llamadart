// Pure helpers for the load path. Kept here (vs inlined in the service) so
// they can be unit-tested without going through `LlamaEngine.loadModel`,
// which is integration-level and needs a real model file.

import '../../core/models/config/flash_attention.dart';
import '../../core/models/config/kv_cache_type.dart';
import 'bindings.dart';

/// Maps llamadart's [KvCacheType] enum to llama.cpp's `ggml_type`. Pure
/// switch, no side effects.
ggml_type ggmlTypeFor(KvCacheType type) {
  switch (type) {
    case KvCacheType.f16:
      return ggml_type.GGML_TYPE_F16;
    case KvCacheType.q8_0:
      return ggml_type.GGML_TYPE_Q8_0;
    case KvCacheType.q4_0:
      return ggml_type.GGML_TYPE_Q4_0;
  }
}

/// Resolves the user-requested [FlashAttention] given the requested KV
/// cache types. llama.cpp refuses non-F16 KV without flash attention, so
/// `auto` is auto-promoted to `enabled` when either KV type isn't F16.
/// Explicit `enabled` / `disabled` are passed through unchanged.
///
/// Pairing this with [ModelParams]'s constructor-side ArgumentError on
/// `(non-F16 KV, FA disabled)` ensures the only ambiguous case (`auto`)
/// gets resolved deterministically here.
FlashAttention resolveFlashAttention({
  required FlashAttention requested,
  required KvCacheType cacheTypeK,
  required KvCacheType cacheTypeV,
}) {
  final wantsKvQuantization =
      cacheTypeK != KvCacheType.f16 || cacheTypeV != KvCacheType.f16;
  if (requested == FlashAttention.auto && wantsKvQuantization) {
    return FlashAttention.enabled;
  }
  return requested;
}
