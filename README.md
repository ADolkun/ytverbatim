# ytverbatim

Turn YouTube URLs into verified, timestamped transcripts for LLM context.

Language models need transcripts they can cite. `ytverbatim` aims to produce text where every line carries a timestamp, so a quote can be traced back to the exact moment in the video. It works with your own logged-in browser session, so Premium and members-only videos you can already watch are covered. Speech recognition runs locally on Apple Silicon (MLX), with a focus on Mandarin and English.

**Status: pre-alpha.** `ytverbatim run` turns a video into a transcript with local Whisper, restores missing punctuation and checks numbers against independent readings where there are any (the video's own captions, or an optional second engine). The engine selection is still to come.

## Usage

```
ytverbatim run URL... [--out DIR] [--auth SPEC] [--engine ID] [--source asr|captions] [--correct/--no-correct]
                      [--chapters auto|none] [--llm SPEC] [--captions/--no-captions] [--force] [--ignore-cooldown]
ytverbatim render CONTEXT.json [--stdout]
```

Each video (playlists and channels are expanded) gets a folder `<out>/<video id>/` holding `context.json` (the versioned contract), `context.md` (the same transcript as markdown with a timestamped link per paragraph and YouTube chapters as headings) and the downloaded audio. Exit codes: 0 ok, 1 some videos failed, 2 auth error, 3 batch stopped. `ytverbatim fetch` downloads without transcribing; `ytverbatim auth` sets up your browser login (see [docs/auth.md](docs/auth.md)); `ytverbatim render` rewrites the markdown from a `context.json`.

The stages run in this order: metadata, captions and audio; ASR (or, with `--source captions`, the manual caption track, falling back to ASR with a warning when there is none); normalize (Simplified Chinese by default, `[text] script = "keep"` to leave the script alone; fillers are kept unless `[text] fillers = "remove"`); punctuation, only where the transcript has too few sentence marks; correction, off unless `--correct`; numeric verification against captions in the transcript's language (without such captions or a second opinion the numbers rest on the transcript alone, and a warning says so); then sentences, paragraphs, chapters and the documents. Every stage is stored under `<video>/.stages/`, keyed by what it read: a rerun with other options recomputes only the stages those options reach, an interrupted run resumes, and a video whose `context.json` is current is skipped. `--force` discards the stored stages and recomputes every one, including the paid LLM calls. The 16 kHz WAV is deleted once the transcript is stored; the compressed audio stays.

`--llm PROVIDER:MODEL` (for example `claude-cli:claude-sonnet-5-5`, or `openai-compat:MODEL` with `[llm] base_url` set) gives the text stages an LLM; `--llm none` turns it off. A provider other than the configured one starts from its own defaults, so a configured `base_url` or `effort` does not get in the way. The LLM's effort and endpoint host are part of its identity, so changing them redoes the LLM stages. Without one, punctuation is left as transcribed and disputed numbers stay flagged with every reading; both are recorded in `quality.warnings` and printed as a notice. `--correct` needs an LLM. `llm_usage` is what the document cost in LLM tokens, per stage: batches reused from earlier runs count too, so a rerun that reuses everything reports the same totals. It is not what one run spent. As a library, `ytverbatim.process(url, options, llm=client)` takes any object that satisfies `ytverbatim.LLMClient`.

When YouTube answers with a bot check or a rate limit, the batch stops and a cooldown starts (1 hour, doubling on repeats up to 24 hours). Until it ends, `run` and `fetch` refuse to start; only a `run` whose targets are all single videos it has already finished goes ahead, because it requests nothing. `--ignore-cooldown` starts anyway, at the risk of your account being flagged; use it only when you know the block has lifted.

The default engine, `w-prod`, is mlx-whisper large-v3 and needs the `whisper` extra (`uv sync --extra whisper`); the other engines need the `asr` extra. `run` checks for the engine's extra before it downloads anything.

ASR runs in a separate worker process under a memory watch, so a runaway model cannot take down the Mac. The worker is killed if its physical footprint passes `memory_cap_gib` (default: half of physical memory) or if the system's available memory falls below 10%. The batch then stops with a fix for whichever one fired, and the next run resumes from the chunks already finished. On an 8 GiB Mac the default cap of 4 GiB may be too tight for `w-prod`; raise `asr.memory_cap_gib` (at most 90% of physical memory). To change the cap, set it in `~/.config/ytverbatim/config.toml`:

```toml
[asr]
memory_cap_gib = 8
```

## Planned for v1

- Auth status and a `doctor` command that explain problems with clear fixes
- Local ASR by default, with YouTube captions used as cross-checking evidence
- Timestamps on every segment, word-level where the engine provides them
- A versioned JSON output plus a markdown rendering

## Requirements

- macOS on Apple Silicon
- `ffmpeg`
- A JavaScript runtime for yt-dlp: Deno or Node 22 or newer

## Responsible use

Use `ytverbatim` only on content you can already access with your own account. It does not circumvent DRM. You are responsible for complying with YouTube's Terms of Service and with copyright law. Your cookies never leave your machine.

## License

[MIT](LICENSE)
