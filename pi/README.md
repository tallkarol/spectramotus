# Spectramotus · Pi 5 test-drive kit

An offline player and content kit, **not a bootable microSD image**. Contains Karol's existing wave, thumbs-up and blink idle, with locally trimmed reaction openings. No generation, accounts, API keys or internet needed for playback. Uses the same player and recognizer as the studio. Pi hardware performance is not yet verified.

## First run on Dave's Pi 5 (4 GB)

1. Use **Raspberry Pi OS Desktop, 64-bit**. A fresh OS card is preferable to overwriting Dave's working Pi 3 card. Keep the old card as a fallback. Have a keyboard/mouse available for setup.
2. Power off before connecting the ribbon camera. Pi 5 uses a **22-pin mini connector**; the Pi 3 camera cable does not fit directly. Camera and display cables are not interchangeable. Configure the screen's orientation using desktop display settings, and disable screen blanking.
3. Copy and unzip the entire kit into a permanent folder such as `~/Spectramotus`. Open a terminal in that folder.
4. While online, run `sh setup.sh`. This installs Chromium, system Python, Picamera2 and rpicam-apps, checks the portrait pack, and writes `hardware-report.json`. No Python virtual environment is needed on the Pi.
5. Confirm `rpicam-hello --list-cameras` lists the ribbon camera. If it does not, stop here and check the cable, connection and OS; the player cannot fix hardware enumeration. Close any other camera application before starting.
6. Run **`python3 launch.py --camera csi`**. Click **Start camera**. You should see your hand in the mirrored preview and tracking overlay. CSI uses a local Picamera2 adapter instead of a browser webcam permission prompt. Frames stay in memory on the Pi and are never saved or sent off-device. Stop camera releases it; closing the player also releases it, with a 15-second inactivity fallback.
7. Follow **TEST-DRIVE.md** before enabling startup.
8. When the windowed test works, run `python3 launch.py --camera csi --kiosk`. **Setup** or the **S key** reveals controls. Alt+F4 closes Chromium; Ctrl+C stops the launcher.
9. After a successful kiosk test: `python3 launch.py --camera csi --install-autostart`. Enable desktop auto-login in Pi settings, reboot twice, then test without Wi-Fi.

## Tune by comparing two runs

The default `--camera-profile balanced` uses 960×720 capture, matching the desktop acquisition setup. If inference is slow or the Pi is overloaded, stop the launcher and try:

```sh
python3 launch.py --camera csi --camera-profile light
```

Light uses 640×480. Both cap capture/recognition at approximately 15 samples/second, allow only one inference in flight, use the same confidence/motion filters, and keep recognition active during reactions. Lower resolution is an experiment, not a guaranteed accuracy improvement. Keep the profile that gives the best observed recognition and response. The report records source, profile, frame age, inference timing and periodic sample rate. Playback events include cumulative decoded/dropped frame counts per video element.

## USB camera or computer

For a USB webcam: `python3 launch.py --camera usb`. Allow Chromium camera permission normally. USB detail can be changed while the camera is stopped in Detection details.

For computer playback: `python3 launch.py` (Python standard library only; Picamera2 is imported only for CSI). Use `--no-browser` to choose a browser yourself. Use the printed **127.0.0.1** URL; a LAN IP does not provide the same camera/security behavior. The player is intentionally not exposed on the LAN.

## Reports and recovery

- `python3 doctor.py` writes `hardware-report.json`: OS/architecture, Chromium, camera enumeration, memory, temperature, throttling and free space. It contains no camera images, IPs, account credentials or Pi serial number.
- **Detection details → Download timing report** saves a gesture JSON file in Chromium's Downloads folder. Download it after each test before closing the player. No frames or raw landmarks are exported.
- Blank camera: check `rpicam-hello --list-cameras`, close competing camera apps, press Stop then Start camera. Use system `python3`, not a pip-installed Picamera2 environment.
- Image rotated: adjust camera mounting first. Display rotation and camera orientation are separate; report any remaining orientation issue before changing recognition thresholds.
- Bad playback: run the manual buttons with camera off, then on. This separates decoding issues from recognition load. Preserve both reports.
- A kit update must be unzipped to a new folder first. Verify it, then switch the launch path. Keep the previous kit until the new one passes.

## Updating content

Stop the launcher, back up `portrait.zip`, then replace it with an approved studio export. Paths, checksums and contents are verified before serving; the kit's bundled player replaces imported executable runtime. For player/recognition updates, replace the complete kit. Private Karol media is not committed to the public repository.

## Autostart

This kit supports Pi OS's labwc desktop autostart. It backs up an existing `~/.config/labwc/autostart` and appends one launch command. If already installed, edit that command when moving the kit or changing camera/profile options. To remove, delete `# Spectramotus player` and its following command. Chromium's dedicated profile is in `~/.local/share/spectramotus/chromium`.

References: [Pi camera connections](https://www.raspberrypi.com/documentation/accessories/camera.html), [camera software](https://www.raspberrypi.com/documentation/accessories/camera-software.html), [official kiosk setup](https://www.raspberrypi.com/tutorials/how-to-use-a-raspberry-pi-in-kiosk-mode/).
