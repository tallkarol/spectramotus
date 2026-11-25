# Spectramotus · Pi starter kit

This is an offline content/player kit, **not a bootable microSD image**. Hardware performance has not yet been tested. It includes the current Plitvice portrait, wave, thumbs-up, blink idle, and the same gesture player used on the computer. No account, API key, generation credit or Wi-Fi is needed during playback.

## Tomorrow: first setup

1. Flash **Raspberry Pi OS Desktop (64-bit)** using Raspberry Pi Imager. Use Desktop, not Lite. Configure your user and Wi-Fi for initial setup. A keyboard/mouse will make this first run easier.
2. Boot the Pi. Confirm Chromium and Python 3 are installed; if missing, install them from Raspberry Pi OS packages while online: `sudo apt update && sudo apt install chromium python3`.
3. Set the display to portrait orientation using the desktop display settings. Disable screen blanking. Connect a USB webcam that appears in Chromium. A CSI camera may need a separate browser-compatible bridge; it is not assumed to work out of the box.
4. Transfer this entire unzipped kit to a permanent folder such as `~/Spectramotus` using a USB drive or Wi-Fi. Copying it to a blank microSD alone will not boot a Pi.
5. In that folder, run `python3 launch.py --verify-only`, then `python3 launch.py`.
6. Click **Start camera** and allow the localhost camera permission. Test wave → thumbs-up during playback → idle. The dedicated Chromium profile remembers permission; no permission-bypass flags are used.
7. Close the preview and run `python3 launch.py --kiosk`. The portrait fills the display. **Setup** (bottom-right) or the **S key** reveals controls. The camera starts automatically in kiosk mode, using the permission granted during setup. Alt+F4 exits Chromium. Ctrl+C stops the launcher.
8. When the test passes, run `python3 launch.py --install-autostart`. This appends one launch command to your user's `~/.config/labwc/autostart`, backing up an existing file. Enable **desktop auto-login** in Pi settings. Reboot and test with Wi-Fi off.

Official OS kiosk reference: https://www.raspberrypi.com/tutorials/how-to-use-a-raspberry-pi-in-kiosk-mode/

## Computer preview

`python3 launch.py` opens the same player at localhost. It uses only Python's standard library; there is nothing to install with pip. Camera permission is requested normally. Use `--no-browser` if you prefer to open the printed URL yourself. Stop with Ctrl+C.

## Updating the portrait

Stop the player. Keep a copy of the previous `portrait.zip`, then replace it with an approved pack downloaded from the studio. The launcher validates paths, checksums, and expected contents before serving anything. It uses the trusted player code shipped in this kit. To update recognition/player code, replace the kit with a newer release. Nothing is uploaded from the Pi.

## Hardware acceptance check

- Boot twice and confirm the portrait and camera start.
- Disconnect Wi-Fi and confirm all clips and recognition still work.
- Try five waves and five thumbs-ups, including alternating gestures during playback.
- Leave it running for 30 minutes: note accidental triggers, tracking rate, heat, frame drops, and black flashes between clips.
- Unplug/reconnect the camera; confirm setup reports a useful error and Start camera recovers.
- Confirm portrait fit, brightness, screen rotation and screen blanking settings on the actual display.

Do not assume that desktop recognition speed predicts Pi performance. We will tune capture resolution and frame rate after measuring the chosen hardware.

## Removing autostart

Remove the `# Spectramotus player` comment and its following launch line from `~/.config/labwc/autostart`. The dedicated browser profile is in `~/.local/share/spectramotus/chromium`.
