# Signing in

Public videos need no login. Members-only videos, and Premium formats, need your own YouTube session. ytverbatim borrows it from your browser; it never asks for a password.

## How it works

- For each job, ytverbatim reads the cookies of **one Chrome profile you chose**, live from the profile on disk.
- It keeps only `youtube.com` and `google.com` cookies and writes them to a private snapshot file (mode `0600`, in a `0700` directory under `~/.local/state/ytverbatim/auth/`).
- yt-dlp uses that snapshot for the job, and the snapshot is deleted afterwards. Your browser's own cookie database is never written.
- Cookies never leave your machine except in the requests yt-dlp makes to YouTube itself. They are never logged, printed, or sent anywhere else.
- Videos are always tried anonymously first. Cookies are used only when YouTube asks for a login.

Brave, Chromium, Edge and Vivaldi work the same way: replace `chrome` with the browser name.

## Setup

1. List your profiles. Emails are masked unless you pass `--show-emails`:

   ```console
   $ ytverbatim auth profiles
   Default  'Person 1'  (cookies, last used, a***@example.com)
   Work     'Work'      (cookies, b***@example.com)
   Use one with: ytverbatim auth use chrome:<DIR>
   ```

2. Pick the profile that is signed in to YouTube with the account you want, by its directory name:

   ```console
   $ ytverbatim auth use chrome:Default
   ```

   This only checks, offline, that the profile has a cookie database, then saves the choice to `~/.local/state/ytverbatim/auth/source.json`. `ytverbatim auth use none` clears it. The `YTVERBATIM_AUTH` environment variable, or a command's `--auth` option, overrides the saved choice.

3. Check that YouTube accepts the login:

   ```console
   $ ytverbatim auth status --refresh
   ✓ Signed in · Premium: unknown · verified 2026-01-01 12:00
   ```

   `auth status` takes a snapshot and makes one request (`:ytfav`, your Liked videos list, of which nothing is shown or kept). A good result is cached for an hour; `--refresh` skips the cache and `--json` prints a machine-readable report. Premium shows as `unknown` until a fetch has seen it.

`ytverbatim auth doctor` checks everything at once: ffmpeg and ffprobe, a JavaScript runtime, the yt-dlp version, the source, the snapshot, the login, and any active cooldown.

When several profiles have cookies, a bare `chrome` source is refused rather than guessed: guessing could silently pick another Google account.

## Keychain

Chrome encrypts its cookies with a key stored in the macOS Keychain item "Chrome Safe Storage". The first snapshot makes macOS ask whether `security` may read it. Run `ytverbatim auth status --refresh` in Terminal, enter your login password, and click **Always Allow**, so later runs do not prompt again. If you click Deny, or nobody answers within two minutes, the state is `keychain_denied`.

## Cookie file

If you would rather not let ytverbatim read your browser, export a cookie file instead:

1. Open a **new private (incognito) window** and sign in to YouTube there.
2. Export the `youtube.com` cookies in Netscape `cookies.txt` format with a cookie-export extension you trust.
3. Close the private window **without signing out**. Signing out would invalidate the exported session.
4. Run `ytverbatim auth use cookie-file:PATH`.

A file expires much sooner than a live profile. YouTube rotates session cookies regularly, and an exported copy cannot follow the rotation, so expect to re-export when `auth status` reports `rotated` or `cookie_file_logged_out`. The private window keeps the exported session apart from your everyday browser session, which would otherwise rotate it away. ytverbatim snapshots the file like a browser profile and never writes to it.

## Why live reading is preferred

Chrome is rolling out Device Bound Session Credentials (DBSC), which tie Google session cookies to a key held by the device. Exported cookies lose that binding, so they are likely to stop working sooner, or at all, as DBSC spreads. Reading the profile live gives yt-dlp the same cookies Chrome currently holds.

## Account safety

ytverbatim keeps your account's footprint small:

- Videos are fetched anonymously first; your login is used only when YouTube requires it.
- Batches sleep between requests, and each run has a video cap.
- A bot check, captcha, IP block, or rate limit stops the batch and starts a cooldown (one hour, doubling on repeats, at most a day). Later runs refuse to start until it ends.
- **Never add cookies to get past a bot check.** That ties the flagged traffic to your account. Wait out the cooldown, and disconnect any VPN or proxy.

## PO token

YouTube sometimes asks for a Proof of Origin (PO) token before serving some formats or subtitles. yt-dlp can get one from an optional provider plugin, such as `bgutil-ytdlp-pot-provider`, that you install yourself into the same environment and run per its own instructions. ytverbatim never installs one or depends on one; the plugin is GPL-licensed. Without it, most videos still work.

## States and fixes

Every auth state has exactly one fix. `auth status` and `auth doctor` print it next to the state.

### no_source

No auth source is set, so members-only videos will fail. Run `ytverbatim auth profiles`, then `ytverbatim auth use chrome:<DIR>`.

### browser_not_found

The browser's data directory does not exist, or the source names an unsupported browser. Open the browser once and sign in to YouTube, or use a [cookie file](#cookie-file).

### profile_not_found

The named profile has no cookie database, or several profiles have cookies and none was named. The message lists the profiles that do. Run `ytverbatim auth use chrome:<DIR>` with one of them.

### cookie_file_invalid

The cookie file is missing, unreadable, or not in Netscape `cookies.txt` format (it must start with the cookie-file header and use LF line endings). Re-export it; see [Cookie file](#cookie-file).

### keychain_denied

macOS refused, or never answered, the Keychain request for the cookie key. See [Keychain](#keychain).

### logged_out

The profile has no signed-in YouTube session. Open youtube.com in that profile, sign in, and re-run.

### cookie_file_logged_out

The exported file holds no signed-in session, or it expired. Re-export it from a fresh private window; see [Cookie file](#cookie-file).

### rotated

YouTube replaced the session's cookies, so the snapshot is no longer valid. With a browser profile, keep a youtube.com tab open in that profile and re-run; if it repeats, sign out and back in there. With a cookie file, re-export it.

### bot_check

YouTube flagged the connection as automated. Wait until the cooldown ends, disconnect any VPN or proxy, and do not add cookies to get past it. See [Account safety](#account-safety).

### captcha

YouTube asked for a captcha. Treat it like [bot_check](#bot_check).

### ip_blocked

YouTube is likely blocking your IP address. Treat it like [bot_check](#bot_check).

### rate_limited

YouTube rate-limited the session, for up to an hour. Wait until the cooldown ends, then use a smaller `--max-videos`.

### not_member

You are signed in, but the account is not a member of the channel, or its membership tier is too low for the video.

### auth_uncertain

The login could not be confirmed, for example because the check request failed for an unrelated reason. Run `ytverbatim auth status --refresh`; if it fails again, run `ytverbatim auth doctor`.

### auth_required

The video needs a signed-in account and no source is set. Follow [Setup](#setup).

### js_runtime_missing

yt-dlp needs a JavaScript runtime to solve YouTube's challenges. Install Deno (`brew install deno`) or Node 22 or newer. Deno is preferred because its sandbox is stricter.

### ytdlp_stale

The installed yt-dlp is more than 45 days old, and YouTube changes often. Upgrade yt-dlp in the environment ytverbatim runs from.

### po_token_required

YouTube wants a PO token for some formats or subtitles. See [PO token](#po-token).

### ok

Signed in and accepted by YouTube. The line also shows Premium (yes, no, or unknown) and when the login was verified.
