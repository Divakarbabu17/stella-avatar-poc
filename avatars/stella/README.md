# Stella avatar

The folder name (`stella`) is the avatar ID (`MUSETALK_AVATAR_ID` in `.env`;
the same value is passed to the musetalk service). IDs: lowercase letters,
digits, `-` and `_`.

## Files

| File | Used by | Required |
|---|---|---|
| `reference.mp4` | musetalk service | One of these two. A video is recommended. |
| `reference.png` / `reference.jpg` | musetalk service | Used only if there is no video. |
| `idle.png` / `idle.jpg` / `idle.webp` | Web page, shown while idle/processing | Optional |

MuseTalk looks for `reference.mp4`, `.mov`, `.mkv`, `.avi`, then `.png`, `.jpg`,
`.jpeg`, in that order.

## What the official MuseTalk code supports

- **Video** (`.mp4/.mov/.mkv/.avi`): the real-time pipeline extracts every frame
  and loops them forward and backward while Stella speaks, so subtle natural
  head motion and blinking carry over. The service re-encodes it to **25 fps**,
  the frame rate MuseTalk was trained on.
- **Image** (`.png/.jpg`): supported by MuseTalk (as a one-frame sequence), but
  only the mouth region moves, so the result looks static.

## Recommended reference video

- 5-15 seconds, one person, face clearly visible and roughly frontal.
- Neutral expression, mouth closed or relaxed, little head movement.
- Even lighting; no hands, microphones or hair covering the mouth or jaw.
- 720p or 1080p. MuseTalk generates the face region at 256 x 256 and blends it
  back into the frame.
- Use footage of a person who has consented to being used as Stella.

## Replacing the avatar

1. Put the new `reference.mp4` (or `.png`) in this folder.
2. Restart the service: `docker compose restart musetalk`.
   The service fingerprints the reference file and re-prepares the avatar
   automatically when it changes (face detection, masks and latents, typically
   tens of seconds on a GPU).
3. Optionally put a matching `idle.png` here. If you don't, the page shows a
   neutral placeholder.

To add another avatar, create `avatars/<new-id>/` with its own `reference.mp4`,
set `MUSETALK_AVATAR_ID=<new-id>` in `.env` and run `docker compose up -d`.
