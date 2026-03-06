#!/usr/bin/env python3
import argparse
import http.server
import socketserver


class CoiStaticHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7357)
    parser.add_argument("--directory", type=str, required=True)
    args = parser.parse_args()

    handler = lambda *h_args, **h_kwargs: CoiStaticHandler(
        *h_args,
        directory=args.directory,
        **h_kwargs,
    )

    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
