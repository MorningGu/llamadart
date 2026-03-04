#!/usr/bin/env python3
import json
import sys
from typing import Any

from playwright.sync_api import sync_playwright


DEFAULT_APP_URL = "http://127.0.0.1:7357"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/"
    "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf?download=true"
)
DEFAULT_MMPROJ_URL = (
    "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/"
    "mmproj-Qwen_Qwen3.5-0.8B-f16.gguf?download=true"
)


def main() -> int:
    app_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_APP_URL
    model_url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_URL
    mmproj_url = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MMPROJ_URL

    console_logs: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=False,
            args=[
                "--enable-unsafe-webgpu",
                "--disable-vulkan-surface",
                "--enable-features=Vulkan",
            ],
        )
        page = browser.new_page()
        page.set_default_timeout(0)

        def on_console(message: Any) -> None:
            try:
                text = message.text
            except Exception:  # pragma: no cover
                text = str(message)
            console_logs.append({"type": message.type, "text": text})

        page.on("console", on_console)

        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_function("() => typeof window.LlamaWebGpuBridge === 'function'")

        result = page.evaluate(
            """
            async ({ modelUrl, mmprojUrl }) => {
              const withTimeout = (promise, ms, label) =>
                Promise.race([
                  promise,
                  new Promise((_, reject) =>
                    setTimeout(() => reject(new Error(`${label} timeout (${ms}ms)`)), ms),
                  ),
                ]);

              const cfg = {
                workerUrl:
                  typeof window.__llamadartBridgeWorkerUrl === 'string'
                    ? window.__llamadartBridgeWorkerUrl
                    : undefined,
                coreModuleUrl:
                  typeof window.__llamadartBridgeCoreModuleUrl === 'string'
                    ? window.__llamadartBridgeCoreModuleUrl
                    : undefined,
                coreModuleUrlMem64:
                  typeof window.__llamadartBridgeCoreModuleUrlMem64 === 'string'
                    ? window.__llamadartBridgeCoreModuleUrlMem64
                    : undefined,
                wasmUrl:
                  typeof window.__llamadartBridgeWasmUrl === 'string'
                    ? window.__llamadartBridgeWasmUrl
                    : undefined,
                wasmUrlMem64:
                  typeof window.__llamadartBridgeWasmUrlMem64 === 'string'
                    ? window.__llamadartBridgeWasmUrlMem64
                    : undefined,
                preferMemory64: window.__llamadartBridgePreferMemory64 === true,
                threadPoolSize:
                  Number.isFinite(Number(window.__llamadartBridgeThreadPoolSize))
                    ? Number(window.__llamadartBridgeThreadPoolSize)
                    : undefined,
                logLevel: 3,
              };

              const bridge = new window.LlamaWebGpuBridge(cfg);
              const loadProgress = [];
              const timings = {};
              const startedAt = performance.now();
              const diagnostics = {
                workerUrl: cfg.workerUrl || null,
                coreModuleUrl: cfg.coreModuleUrl || null,
                bridgeAssetSource: window.__llamadartBridgeAssetSource || null,
                bridgeModuleUrl: window.__llamadartBridgeModuleUrl || null,
                configuredWorkerStallTimeoutMs: cfg.workerGenerationStallTimeoutMs || null,
                effectiveWorkerStallTimeoutMs:
                  typeof bridge._workerCompletionStallTimeoutMs === 'function'
                    ? bridge._workerCompletionStallTimeoutMs({
                        parts: [{ type: 'image', bytes: new Uint8Array([1]) }],
                      })
                    : null,
              };

              const probeFetch = async (url, label) => {
                const controller = new AbortController();
                const timer = setTimeout(() => controller.abort(), 45000);
                try {
                  const response = await fetch(url, {
                    method: 'GET',
                    cache: 'no-store',
                    mode: 'cors',
                    signal: controller.signal,
                  });
                  const contentLength = response.headers.get('content-length');
                  await response.body?.cancel?.();
                  diagnostics[`${label}Probe`] = {
                    ok: response.ok,
                    status: response.status,
                    contentLength,
                    redirected: response.redirected,
                    type: response.type,
                  };
                } catch (error) {
                  diagnostics[`${label}Probe`] = {
                    ok: false,
                    error: String(error),
                  };
                } finally {
                  clearTimeout(timer);
                }
              };

              await probeFetch(modelUrl, 'model');
              await probeFetch(mmprojUrl, 'mmproj');

              try {
                const modelLoadStart = performance.now();
                await withTimeout(
                  bridge.loadModelFromUrl(modelUrl, {
                    nCtx: 4096,
                    nGpuLayers: 99,
                    nThreads: 4,
                    useCache: false,
                    remoteFetchThresholdBytes: 9000000000000,
                    progressCallback: (progress) => {
                      const loaded = Number(progress?.loaded || 0);
                      const total = Number(progress?.total || 0);
                      loadProgress.push({
                        loaded,
                        total,
                        atMs: performance.now() - startedAt,
                      });
                    },
                  }),
                  8 * 60 * 1000,
                  'loadModelFromUrl',
                );
                timings.modelLoadMs = Math.round(performance.now() - modelLoadStart);

                const mmprojStart = performance.now();
                await withTimeout(
                  bridge.loadMultimodalProjector(mmprojUrl),
                  5 * 60 * 1000,
                  'loadMultimodalProjector',
                );
                timings.mmprojLoadMs = Math.round(performance.now() - mmprojStart);

                diagnostics.preInferenceBackend =
                  typeof bridge.getBackendName === 'function'
                    ? bridge.getBackendName()
                    : null;
                diagnostics.preInferenceGpuActive =
                  typeof bridge.isGpuActive === 'function'
                    ? bridge.isGpuActive()
                    : null;
                diagnostics.preInferenceMetadata =
                  typeof bridge.getModelMetadata === 'function'
                    ? bridge.getModelMetadata()
                    : {};

                const buildSyntheticImageBytes = async () => {
                  const width = 3072;
                  const height = 1792;
                  let blob = null;

                  const drawPattern = (ctx) => {
                    ctx.fillStyle = '#0f172a';
                    ctx.fillRect(0, 0, width, height);
                    const grad = ctx.createLinearGradient(0, 0, width, height);
                    grad.addColorStop(0, '#60a5fa');
                    grad.addColorStop(1, '#f97316');
                    ctx.fillStyle = grad;
                    ctx.fillRect(180, 180, width - 360, height - 360);
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 180px sans-serif';
                    ctx.fillText('llamadart multimodal test', 240, 420);
                    ctx.fillStyle = '#111827';
                    ctx.font = '120px sans-serif';
                    ctx.fillText('large synthetic image payload', 240, 620);
                  };

                  if (typeof OffscreenCanvas === 'function') {
                    const canvas = new OffscreenCanvas(width, height);
                    const ctx = canvas.getContext('2d');
                    if (ctx && typeof canvas.convertToBlob === 'function') {
                      drawPattern(ctx);
                      blob = await canvas.convertToBlob({ type: 'image/png' });
                    }
                  }

                  if (!blob && typeof document !== 'undefined' && typeof document.createElement === 'function') {
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                      drawPattern(ctx);
                      blob = await new Promise((resolve, reject) => {
                        canvas.toBlob((value) => {
                          if (value) {
                            resolve(value);
                            return;
                          }
                          reject(new Error('Failed to create PNG blob from canvas'));
                        }, 'image/png');
                      });
                    }
                  }

                  if (!blob) {
                    throw new Error('Failed to synthesize test image payload');
                  }

                  return new Uint8Array(await blob.arrayBuffer());
                };

                const imageBytes = await buildSyntheticImageBytes();
                diagnostics.syntheticImageBytes = imageBytes.length;

                const inferStart = performance.now();
                const output = await withTimeout(
                  bridge.createCompletion('what do you see?', {
                    nPredict: 64,
                    temp: 0.2,
                    topK: 40,
                    topP: 0.95,
                    penalty: 1.1,
                    parts: [
                      {
                        type: 'image',
                        bytes: imageBytes,
                      },
                    ],
                  }),
                  4 * 60 * 1000,
                  'createCompletion',
                );
                timings.inferenceMs = Math.round(performance.now() - inferStart);

                const metadata = bridge.getModelMetadata?.() || {};
                return {
                  ok: true,
                  output: String(output || ''),
                  timings,
                  metadata,
                  diagnostics,
                  progressSamples: loadProgress.slice(-8),
                };
              } catch (error) {
                const runtimeNotes = Array.isArray(bridge?._runtime?._runtimeNotes)
                  ? bridge._runtime._runtimeNotes.slice(-30)
                  : [];
                return {
                  ok: false,
                  error: String(error),
                  errorStack:
                    error && typeof error === 'object' && typeof error.stack === 'string'
                      ? error.stack
                      : null,
                  timings,
                  diagnostics: {
                    ...diagnostics,
                    workerFallbackReason: window.__llamadartBridgeWorkerFallbackReason || null,
                    bridgeLoadError: window.__llamadartBridgeLoadError || null,
                    workerPendingCalls:
                      Number(bridge?._workerProxy?._pending?.size) || 0,
                    runtimeNotes,
                    runtimeModelBytes: Number(bridge?._runtime?._modelBytes || 0),
                    runtimeModelSource: bridge?._runtime?._modelSource || null,
                    runtimeCoreVariant: bridge?._runtime?._coreVariant || null,
                    runtimeLastCoreError: bridge?._runtime?._lastCoreErrorText || null,
                    runtimeLastCoreHint: bridge?._runtime?._lastCoreErrorHint || null,
                  },
                  progressSamples: loadProgress.slice(-12),
                };
              } finally {
                try {
                  await bridge.dispose();
                } catch (_) {
                  // best-effort disposal only
                }
              }
            }
            """,
            {"modelUrl": model_url, "mmprojUrl": mmproj_url},
        )

        browser.close()

    print(
        json.dumps(
            {
                "result": result,
                "consoleTail": console_logs[-40:],
            },
            indent=2,
        )
    )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
