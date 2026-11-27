# Face Guard: on-device face recognition with liveness and anti-spoofing (Flutter, Android)

Real-time face recognition for up to 1000 registered people, with:

* **Live camera detection.** ML Kit face detection runs in fast mode with tracking, landmarks and classification.
* **Liveness checks.** Randomised active challenges are combined with a passive anti-spoof CNN, so a photo, a printed picture or a phone/tablet screen showing the person is rejected.
* **Registration gated by liveness.** A face can only be enrolled after it passes the same live-face check.
* **Fully offline.** Nothing leaves the phone.

Target device: **Motorola Edge 50 Fusion** (Android 14). Any Android 8+ phone with a front camera should work.

---

## 1. Build and run

Requirements: the latest stable Flutter (3.29 or newer), Android SDK, and USB debugging enabled on the phone.

```bash
# Windows: double-click setup.bat, or run it in a terminal
./setup.sh          # runs `flutter create` for the Android scaffolding (keeps existing files) + pub get
flutter test        # unit tests for the core logic (no device needed)
flutter run --release   # always test performance in release mode
# or build an installable APK:
flutter build apk --release   # -> build/app/outputs/flutter-apk/app-release.apk
```

> Debug builds run Dart unoptimised and can be several times slower. Judge
> speed and smoothness **only** with `--release` (or `--profile`).

---

## 2. How it works

```
Camera stream (NV21, 720x480) ──► ML Kit face detector (fast, tracking, landmarks, eye/smile probs)
        │  frames arriving while busy are DROPPED (never queued → no lag build-up)
        ▼
  Liveness engine (pure Dart)                 ML worker isolate (never blocks the UI)
  ─ positioning (size / centre / pose)        ─ crops straight from NV21 (no full-frame conversion)
  ─ random challenges: first is always        ─ 5-point ArcFace alignment → MobileFaceNet (192-d)
    blink or smile, then turn / nod / …          with flip-averaging
  ─ passive CNN score must stay low           ─ Deep-Tree anti-spoof CNN (print + replay attacks)
  ─ identity must stay consistent             ─ sharpness (Laplacian) gate
        ▼
  Recognition: brute-force cosine search over 1000×192 matrix (<1 ms) → name + similarity
```

**Anti-spoofing layers**

| Layer | Stops |
|---|---|
| First challenge is always an expression change (blink/smile) | Printed photos, still images on a screen |
| Random challenge order + 7 s timeout | Pre-recorded videos |
| Passive anti-spoof CNN on every ~120 ms | Photos/videos on phone & tablet screens, prints |
| Identity consistency (embeddings during the check must match) | Swapping faces/photos mid-check |
| Passive check keeps running after verification | Holding up a photo after a real person passes |
| Track change ⇒ restart | A second person stepping in |

**Speed**

* At most one ML job is in flight, and stale frames are dropped, so latency never accumulates.
* All cropping and inference run in a background isolate. Detection runs natively in ML Kit.
* Measured on a desktop CPU: MobileFaceNet takes about 8 ms and the anti-spoof CNN about 47 ms. Expect roughly 10–20 ms and 50–90 ms on the Edge 50 Fusion. Detection typically takes 15–30 ms per frame.
* With the default single challenge, a typical recognition takes about 1–2 s, including the blink or smile.

---

## 3. Calibrate on your phone

The stats line at the bottom of the scan screen shows FPS, timings, the **spoof score**, the **sharp** value and the **best similarity**.

1. **Anti-spoof threshold** (default 0.20, from the model's reference implementation):
   * Look at the camera live and note the typical `spoof score`.
   * Then show a photo of yourself on another phone and note its score.
   * Set *Settings → Anti-spoof threshold* between the two values.
2. **Blur gate** (default 80): set it below the `sharp` value a live face normally shows in your lighting. If it is too high, verification ends with "Image too blurry".
3. **Match threshold** (default 0.60): register 2–3 people and watch `best sim`:
   * Note the value for the right person and for unregistered people.
   * Set the threshold between the two. Raise it for fewer false accepts; lower it if real users are rejected.
4. **Challenges:** 1 is fastest. Use 2–3 for higher-security doors.

---

## 4. On-device test checklist

| # | Test | Expected |
|---|---|---|
| 1 | Register 3 people | Each passes the challenges, then the "Register?" dialog shows a thumbnail |
| 2 | Register the same person twice | Warning: "already matches X" |
| 3 | Live recognition of each person | Green box + name within about 2 s |
| 4 | An unregistered person | Orange "Unknown person" |
| 5 | Show a registered person's **printed photo** | Times out on blink/smile, or "Spoof detected" (red) |
| 6 | Show their photo/video on **another phone screen** | "Spoof detected", or fails the random challenge |
| 7 | Real person passes, then a photo is held up | The track changes or the spoof score rises, and it resets |
| 8 | Two people in frame during registration | "Only one person in the frame" |
| 9 | Leave the app and return (home button) | Camera resumes, no crash |
| 10 | Switch to the back camera | Works, and the box is not mirrored |
| 11 | Delete or rename in *Manage Faces* | Recognition reflects the change immediately |
| 12 | Register a face when 1000 are already enrolled | Blocked with a limit message |

---

## 5. Project layout

```
lib/
  core/                  pure Dart, unit-tested
    face_index.dart      1000-face in-memory matrix + cosine search
    face_align.dart      5-point ArcFace similarity alignment
    yuv_sampler.dart     rotated/scaled crop sampled straight from NV21
    liveness.dart        challenge state machine + passive score fusion
    bmp.dart, settings.dart
  services/
    ml_worker.dart       background isolate running both TFLite models
    face_scan_controller.dart   camera → detector → liveness → worker pipeline
    face_repository.dart SQLite storage (embeddings, thumbnails, settings)
  ui/                    screens and camera overlay
assets/models/           MobileFaceNet.tflite, FaceAntiSpoofing.tflite (MIT, syaringan357)
test/                    unit and widget tests
tool/verify/             Python checks of the crop geometry and model I/O
```

## 6. Limitations

* Software liveness on a single RGB camera raises the bar a lot, but it is not bank-grade (iBeta / ISO 30107-3 level). The passive model was trained for print and replay attacks. Realistic 3D masks need depth/IR hardware or a commercial SDK. For high-security deployments, combine this with a PIN or card.
* Recognition accuracy depends on lighting. Register people in the same lighting where recognition will happen.
* Face data is stored unencrypted in the app's private storage. Add encryption, for example with SQLCipher, if your policy requires it.
