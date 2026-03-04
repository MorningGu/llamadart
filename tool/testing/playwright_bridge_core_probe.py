#!/usr/bin/env python3
import json
import sys

from playwright.sync_api import sync_playwright


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7357"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--enable-unsafe-webgpu",
                "--disable-vulkan-surface",
                "--enable-features=Vulkan",
            ],
        )
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")
        page.wait_for_function("() => typeof window.LlamaWebGpuBridge === 'function'")

        result = page.evaluate(
            """
            async () => {
              const withTimeout = (promise, ms, label) =>
                Promise.race([
                  promise,
                  new Promise((_, reject) =>
                    setTimeout(() => reject(new Error(`${label} timeout`)), ms),
                  ),
                ]);

              const importDiagnostics = {};
              const coreModuleUrl =
                typeof window.__llamadartBridgeCoreModuleUrl === 'string'
                  ? window.__llamadartBridgeCoreModuleUrl
                  : null;
              if (coreModuleUrl) {
                try {
                  await withTimeout(import(coreModuleUrl), 30000, 'core module import');
                  importDiagnostics.coreImportOk = true;
                } catch (error) {
                  importDiagnostics.coreImportOk = false;
                  importDiagnostics.coreImportError = String(error);
                }
              }

              const bridge = new window.LlamaWebGpuBridge({
                disableWorker: true,
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
                logLevel: 3,
              });

              const runtime = bridge._runtime;
              try {
                await Promise.race([
                  runtime._ensureCore(),
                  new Promise((_, reject) =>
                    setTimeout(() => reject(new Error('ensureCore timeout')), 130000),
                  ),
                ]);
                return {
                  ok: true,
                  importDiagnostics,
                  coreVariant: runtime._coreVariant || null,
                  runtimeNotes: Array.isArray(runtime._runtimeNotes)
                    ? runtime._runtimeNotes.slice(-20)
                    : [],
                };
              } catch (error) {
                return {
                  ok: false,
                  error: String(error),
                  importDiagnostics,
                  coreVariant: runtime._coreVariant || null,
                  runtimeNotes: Array.isArray(runtime._runtimeNotes)
                    ? runtime._runtimeNotes.slice(-20)
                    : [],
                };
              } finally {
                try {
                  await bridge.dispose();
                } catch (_) {
                  // best-effort cleanup only
                }
              }
            }
            """
        )

        browser.close()

    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
