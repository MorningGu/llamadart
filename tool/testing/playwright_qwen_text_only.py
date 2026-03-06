#!/usr/bin/env python3
import argparse

from playwright_qwen_harness import (
    DEFAULT_APP_URL,
    DEFAULT_MODEL_URL,
    print_json_result,
    run_bridge_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("app_url", nargs="?", default=DEFAULT_APP_URL)
    parser.add_argument("model_url", nargs="?", default=DEFAULT_MODEL_URL)
    parser.add_argument("--channel", type=str, default="chromium")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    payload = run_bridge_evaluation(
        app_url=args.app_url,
        channel=args.channel,
        headed=args.headed,
        default_timeout_ms=0,
        console_tail_count=8,
        evaluate_script=
        """
            async ({ modelUrl }) => {
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
              const timings = {};

              try {
                const loadStart = performance.now();
                await withTimeout(
                  bridge.loadModelFromUrl(modelUrl, {
                    nCtx: 4096,
                    nGpuLayers: 99,
                    nThreads: 4,
                    useCache: false,
                    remoteFetchThresholdBytes: 9000000000000,
                  }),
                  8 * 60 * 1000,
                  'loadModelFromUrl',
                );
                timings.modelLoadMs = Math.round(performance.now() - loadStart);

                let tokenCount = 0;
                const inferStart = performance.now();
                const output = await withTimeout(
                  bridge.createCompletion(
                    'Write five concise bullet points about WebGPU performance tuning in browsers.',
                    {
                      nPredict: 160,
                      temp: 0.2,
                      topK: 40,
                      topP: 0.95,
                      penalty: 1.1,
                      onToken: () => {
                        tokenCount += 1;
                      },
                    },
                  ),
                  3 * 60 * 1000,
                  'createCompletion',
                );
                timings.inferenceMs = Math.round(performance.now() - inferStart);

                const metadata =
                  typeof bridge.getModelMetadata === 'function'
                    ? bridge.getModelMetadata() || {}
                    : {};
                const tokensPerSecond =
                  timings.inferenceMs > 0 ? (tokenCount * 1000.0) / timings.inferenceMs : null;

                return {
                  ok: true,
                  tokenCount,
                  tokensPerSecond,
                  outputPreview: String(output || '').slice(0, 220),
                  timings,
                  metadata,
                };
              } catch (error) {
                return {
                  ok: false,
                  error: String(error),
                  timings,
                };
              } finally {
                try {
                  await bridge.dispose();
                } catch (_) {
                  // best-effort cleanup only
                }
              }
            }
            """,
        payload={"modelUrl": args.model_url},
    )

    print_json_result(payload)
    result = payload.get("result", {})
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
