# ytverbatim

Turn YouTube URLs into verified, timestamped transcripts for LLM context.

Language models need transcripts they can cite. `ytverbatim` aims to produce text where every line carries a timestamp, so a quote can be traced back to the exact moment in the video. It works with your own logged-in browser session, so Premium and members-only videos you can already watch are covered. Speech recognition runs locally on Apple Silicon (MLX), with a focus on Mandarin and English.

**Status: pre-alpha — not usable yet.**

## Planned for v1

- Auth status and a `doctor` command that explain problems with clear fixes
- Local ASR by default, with YouTube captions used as cross-checking evidence
- Timestamps on every segment, word-level where the engine provides them
- Guarded LLM correction that never alters numbers silently
- A versioned JSON output plus a markdown rendering

## Requirements

- macOS on Apple Silicon
- `ffmpeg`
- A JavaScript runtime for yt-dlp: Deno or Node 22 or newer

## Responsible use

Use `ytverbatim` only on content you can already access with your own account. It does not circumvent DRM. You are responsible for complying with YouTube's Terms of Service and with copyright law. Your cookies never leave your machine.

## License

[MIT](LICENSE)
