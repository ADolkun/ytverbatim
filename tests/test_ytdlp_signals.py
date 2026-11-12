from pathlib import Path

import pytest

from ytverbatim.ytdlp.signals import DecryptCounts, LineKind, classify, js_challenge_failed, line_kind
from ytverbatim.ytdlp.types import Signal

FIXTURES = Path(__file__).parent / "fixtures" / "ytdlp"
VID = "BaW_jenozKc"
HINT = (
    "Use --cookies-from-browser or --cookies for the authentication. "
    "See  https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp  for how to manually pass "
    "cookies. Also see  https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies  for tips on "
    "effectively exporting YouTube cookies"
)
EJS_HELP = (
    "Ensure you have a supported JavaScript runtime and challenge solver script distribution installed. "
    "Review any warnings presented before this message. For more details, refer to  "
    "https://github.com/yt-dlp/yt-dlp/wiki/EJS"
)
N_CHALLENGE_WARNING = f"WARNING: [youtube] {VID}: n challenge solving failed: Some formats may be missing. {EJS_HELP}"
JS_MISSING_WARNING = (
    "WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use "
    "another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS "
    "runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  "
    "for details on installing one"
)
ROTATED_WARNING = (
    "WARNING: [youtube] The provided YouTube account cookies are no longer valid. They have likely been rotated in "
    "the browser as a security measure. For tips on how to effectively export YouTube cookies, refer to  "
    "https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies ."
)

# One exact message per primary A2 row, as yt-dlp prints it.
PRIMARY_ROWS = [
    (Signal.BOT_CHECK, f"ERROR: [youtube] {VID}: Sign in to confirm you’re not a bot. {HINT}"),
    (Signal.BOT_CHECK, f"ERROR: [youtube] {VID}: Sign in to confirm you're not a bot. {HINT}"),
    (
        Signal.IP_BLOCKED,
        f"ERROR: [youtube] {VID}: All player responses are invalid. Your IP is likely being blocked by Youtube",
    ),
    (
        Signal.CAPTCHA,
        f"ERROR: [youtube] {VID}: Video unavailable. YouTube is requiring a captcha challenge before playback",
    ),
    (
        Signal.RATE_LIMITED,
        f"ERROR: [youtube] {VID}: This content isn’t available, try again later. The current session has been "
        "rate-limited by YouTube for up to an hour. It is recommended to use `-t sleep` to add a delay between "
        "video requests to avoid exceeding the rate limit. For more information, refer to  "
        "https://github.com/yt-dlp/yt-dlp/wiki/Extractors#this-content-isnt-available-try-again-later",
    ),
    (Signal.RATE_LIMITED, f"ERROR: [youtube] {VID}: This content isn't available, try again later."),
    (
        Signal.HTTP_429,
        f"ERROR: [youtube] {VID}: Unable to download webpage: HTTP Error 429: Too Many Requests "
        "(caused by <HTTPError 429: Too Many Requests>)",
    ),
    (Signal.HTTP_429, "ERROR: Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests"),
    (Signal.AGE_GATE, f"ERROR: [youtube] {VID}: Sign in to confirm your age. This video may be inappropriate. {HINT}"),
    (
        Signal.PRIVATE,
        f"ERROR: [youtube] {VID}: Private video. Sign in if you've been granted access to this video. {HINT}",
    ),
    (
        Signal.MEMBERS_ONLY,
        f"ERROR: [youtube] {VID}: Join this channel to get access to members-only content like this video, "
        "and other exclusive perks.",
    ),
    (Signal.MEMBERS_ONLY, f"ERROR: [youtube] {VID}: This video is available to this channel’s members on level: X"),
    (Signal.PREMIUM_ONLY, f"ERROR: [youtube] {VID}: This video is only available to YouTube Premium members"),
    (
        Signal.LOGIN_REQUIRED,
        f"ERROR: [youtube:favorites] ytfav: Login details are needed to download this content. {HINT}",
    ),
    (Signal.LOGIN_REQUIRED, f"ERROR: [youtube] {VID}: This video is only available for registered users. {HINT}"),
    (
        Signal.GEO,
        f"ERROR: [youtube] {VID}: The uploader has not made this video available in your country"
        "\nThis video is available in Japan.",
    ),
    (Signal.GEO, f"ERROR: [youtube] {VID}: This video is not available from your location due to geo restriction"),
    (Signal.UNAVAILABLE, f"ERROR: [youtube] {VID}: Video unavailable"),
    (Signal.UNAVAILABLE, f"ERROR: [youtube] {VID}: Video unavailable. This video has been removed by the uploader"),
    (
        Signal.UNAVAILABLE,
        f"ERROR: [youtube] {VID}: Video unavailable. This video is no longer available because the YouTube "
        "account associated with this video has been terminated.",
    ),
    (Signal.DRM, f"ERROR: [youtube] {VID}: This video is DRM protected"),
    (
        Signal.PO_TOKEN_GVS,
        f"ERROR: [youtube] {VID}: ios client https formats require a GVS PO Token which was not provided.",
    ),
    (
        Signal.PO_TOKEN_SUBS,
        f"ERROR: [youtube] {VID}: Some web client subtitles require a PO Token which was not provided.",
    ),
    (
        Signal.NO_FORMATS,
        f"ERROR: [youtube] {VID}: Requested format is not available. Use --list-formats for a list of available "
        "formats",
    ),
    (Signal.NO_FORMATS, f"ERROR: [youtube] {VID}: No video formats found!"),
    (Signal.HTTP_403, "ERROR: unable to download video data: HTTP Error 403: Forbidden"),
    (Signal.TRANSIENT, f"ERROR: [youtube] {VID}: Unable to download webpage: HTTP Error 503: Service Unavailable"),
    (Signal.TRANSIENT, "ERROR: HTTP Error 500: Internal Server Error"),
    (Signal.TRANSIENT, "ERROR: HTTP Error 502: Bad Gateway"),
    (Signal.TRANSIENT, "ERROR: HTTP Error 504: Gateway Timeout"),
    (Signal.TRANSIENT, f"ERROR: [youtube] {VID}: Unable to download webpage: The read operation timed out"),
    (
        Signal.TRANSIENT,
        f"ERROR: [youtube] {VID}: Unable to download webpage: <urlopen error [Errno 8] "
        "Temporary failure in name resolution>",
    ),
    (Signal.TRANSIENT, "ERROR: [Errno 54] Connection reset by peer"),
    (Signal.TRANSIENT, "ERROR: ('Connection aborted.', RemoteDisconnected('closed'))"),
    (Signal.TRANSIENT, "ERROR: IncompleteRead(1024 bytes read, 4096 more expected)"),
]

SECONDARY_ROWS = [
    (Signal.ROTATED, ROTATED_WARNING),
    (Signal.PREMIUM, "[debug] [youtube] Detected YouTube Premium subscription"),
    (Signal.AUTH_COOKIES_FOUND, "[debug] [youtube] Found YouTube account cookies"),
    (
        Signal.AGE_VERIFICATION,
        f"[youtube] {VID}: This video is age-restricted and YouTube is requiring account age-verification; "
        "some formats may be missing",
    ),
    (Signal.COOKIE_DB_MISSING, 'ERROR: could not find chrome cookies database in "/tmp/example/pin"'),
    (Signal.KEYCHAIN, "WARNING: find-generic-password failed"),
    (Signal.KEYCHAIN, "WARNING: exception running find-generic-password: boom"),
    (Signal.KEYCHAIN, "WARNING: cannot decrypt v10 cookies: no key found"),
    (Signal.KEYCHAIN, "WARNING: cannot decrypt v11 cookies: no key found"),
    (Signal.DECRYPT_PARTIAL, "Extracted 940 cookies from chrome (12 could not be decrypted)"),
    (Signal.COOKIE_LOAD, "ERROR: failed to load cookies"),
    (Signal.SKIP_CLIENT_NO_COOKIES, 'WARNING: [youtube] Skipping client "visionos" since it does not support cookies'),
    (Signal.JS_MISSING, JS_MISSING_WARNING),
    (Signal.N_CHALLENGE, N_CHALLENGE_WARNING),
    (
        Signal.PO_TOKEN_GVS,
        f"[debug] [youtube] {VID}: web client https formats require a GVS PO Token which was not provided. "
        "They will be skipped as they may yield HTTP Error 403.",
    ),
]


@pytest.mark.parametrize(("signal", "stderr"), PRIMARY_ROWS)
def test_primary_rows(signal: Signal, stderr: str) -> None:
    assert classify(stderr).primary is signal


@pytest.mark.parametrize(("signal", "stderr"), SECONDARY_ROWS)
def test_secondary_rows(signal: Signal, stderr: str) -> None:
    found = classify(stderr)
    assert signal in found.secondary
    assert found.primary is None


def test_every_signal_has_a_row() -> None:
    covered = {s for s, _ in PRIMARY_ROWS} | {s for s, _ in SECONDARY_ROWS}
    assert covered == set(Signal)


def test_bot_check_with_cookie_hint_is_not_login_required() -> None:
    found = classify(f"ERROR: [youtube] {VID}: Sign in to confirm you’re not a bot. {HINT}")
    assert found.primary is Signal.BOT_CHECK
    assert Signal.LOGIN_REQUIRED not in found.secondary


def test_members_error_with_n_challenge_warning_is_members_only() -> None:
    stderr = "\n".join(
        [
            N_CHALLENGE_WARNING,
            f"ERROR: [youtube] {VID}: Join this channel to get access to members-only content like this video.",
        ]
    )
    found = classify(stderr)
    assert found.primary is Signal.MEMBERS_ONLY
    assert Signal.N_CHALLENGE in found.secondary
    assert not found.js_challenge_failed


def test_retriable_text_in_warnings_does_not_become_the_cause() -> None:
    stderr = "\n".join(
        [
            "WARNING: [youtube] Unable to download webpage: HTTP Error 503: Service Unavailable. Retrying (1/3)...",
            f"ERROR: [youtube] {VID}: Video unavailable",
        ]
    )
    found = classify(stderr)
    assert found.primary is Signal.UNAVAILABLE
    assert Signal.TRANSIENT in found.secondary


def test_highest_precedence_error_line_wins() -> None:
    stderr = "\n".join(["ERROR: HTTP Error 503: Service Unavailable", f"ERROR: [youtube] {VID}: Private video. {HINT}"])
    assert classify(stderr).primary is Signal.PRIVATE


@pytest.mark.parametrize("warning", [N_CHALLENGE_WARNING, JS_MISSING_WARNING])
def test_no_formats_after_js_failure_is_distinguishable(warning: str) -> None:
    found = classify("\n".join([warning, f"ERROR: [youtube] {VID}: Requested format is not available."]))
    assert found.primary is Signal.NO_FORMATS
    assert found.js_challenge_failed
    assert js_challenge_failed(found.primary, found.secondary)


def test_no_formats_alone_is_not_a_js_failure() -> None:
    assert not classify(f"ERROR: [youtube] {VID}: No video formats found!").js_challenge_failed


def test_n_challenge_never_becomes_primary() -> None:
    assert classify(f"ERROR: [youtube] {VID}: n challenge solving failed: Some formats may be missing").primary is None


def test_http_429_and_playability_rate_limit_are_distinct() -> None:
    assert classify("ERROR: HTTP Error 429: Too Many Requests").primary is Signal.HTTP_429
    assert classify(f"ERROR: [youtube] {VID}: This content isn’t available, try again later.").primary is (
        Signal.RATE_LIMITED
    )


def test_rotated_on_success_lands_in_secondary_with_warning_text() -> None:
    found = classify(ROTATED_WARNING)
    assert found.primary is None
    assert found.secondary == (Signal.ROTATED,)
    assert found.warnings == (ROTATED_WARNING.removeprefix("WARNING: "),)


def test_decrypt_partial_counts() -> None:
    found = classify("Extracting cookies from chrome\nExtracted 940 cookies from chrome (12 could not be decrypted)")
    assert found.decrypt_partial == DecryptCounts(extracted=940, browser="chrome", failed=12)
    assert classify("Extracted 940 cookies from chrome").decrypt_partial is None


def test_ansi_is_stripped_and_matching_ignores_case() -> None:
    found = classify(f"\x1b[0;31mERROR:\x1b[0m [youtube] {VID}: VIDEO UNAVAILABLE\r\n")
    assert found.primary is Signal.UNAVAILABLE


def test_unknown_error_has_no_primary() -> None:
    found = classify("ERROR: something new went wrong\nyt-dlp: error: no such option: --bogus")
    assert (found.primary, found.secondary) == (None, ())


def test_line_kinds() -> None:
    assert line_kind("ERROR: x") is LineKind.ERROR
    assert line_kind("WARNING: x") is LineKind.WARNING
    assert line_kind("[debug] x") is LineKind.DEBUG
    assert line_kind("[youtube] x") is LineKind.OTHER


def test_live_members_only_fixture() -> None:
    found = classify((FIXTURES / "members_only_anonymous.txt").read_text(encoding="utf-8"))
    assert found.primary is Signal.MEMBERS_ONLY
    assert found.secondary == ()


def test_live_public_fixture_is_clean() -> None:
    found = classify((FIXTURES / "public_anonymous.txt").read_text(encoding="utf-8"))
    assert (found.primary, found.secondary, found.warnings) == (None, (), ())
