# ytverbatim

Turn YouTube URLs into verified, timestamped transcripts for LLM context.

Language models need transcripts they can cite. `ytverbatim` aims to produce text where every line carries a timestamp, so a quote can be traced back to the exact moment in the video. It works with your own logged-in browser session, so Premium and members-only videos you can already watch are covered. Speech recognition runs locally on Apple Silicon (MLX), with a focus on Mandarin and English.

**Status: pre-alpha.** The first usable slice works: `ytverbatim run` turns a video into a transcript with local Whisper and no LLM. Correction, numeric verification and the engine selection are still to come.

## Usage

```
ytverbatim run URL... [--out DIR] [--auth SPEC] [--engine ID] [--captions/--no-captions] [--force] [--ignore-cooldown]
```

Each video (playlists and channels are expanded) gets a folder `<out>/<video id>/` holding `context.json` (the versioned contract), `context.md` (the same transcript as markdown with a timestamped link per paragraph and YouTube chapters as headings) and the downloaded audio. A video that already has a `context.json` from the selected engine is skipped unless `--force` is given; one from another engine is transcribed again. Exit codes: 0 ok, 1 some videos failed, 2 auth error, 3 batch stopped. `ytverbatim fetch` downloads without transcribing; `ytverbatim auth` sets up your browser login (see [docs/auth.md](docs/auth.md)).

When YouTube answers with a bot check or a rate limit, the batch stops and a cooldown starts (1 hour, doubling on repeats up to 24 hours). Until it ends, `run` and `fetch` refuse to start; only a `run` whose targets are all single videos it has already finished goes ahead, because it requests nothing. `--ignore-cooldown` starts anyway, at the risk of your account being flagged; use it only when you know the block has lifted.

The default engine, `w-prod`, is mlx-whisper large-v3 and needs the `whisper` extra (`uv sync --extra whisper`); the other engines need the `asr` extra. `run` checks for the engine's extra before it downloads anything.

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
