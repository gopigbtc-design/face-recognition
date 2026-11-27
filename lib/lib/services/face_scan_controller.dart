import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Rect, Size;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../core/face_align.dart';
import '../core/face_index.dart';
import '../core/liveness.dart';
import '../core/settings.dart';
import '../core/yuv_sampler.dart';
import 'app_services.dart';
import 'ml_worker.dart';

enum ScanMode { recognize, register }

enum ScanStatus { starting, noFace, adjusting, challenge, verifying, recognized, unknown, captured, failed, error }

class RegistrationCapture {
  RegistrationCapture({required this.embedding, this.thumbBmp, this.duplicate});
  final Float32List embedding;
  final Uint8List? thumbBmp;

  /// Set when the face already matches a registered person.
  final FaceMatch? duplicate;
}

/// Drives the whole pipeline for one screen:
///
///   camera stream ──► ML Kit (fast mode, tracking) ──► liveness engine
///                         │
///                         └─► ML worker isolate (crop + MobileFaceNet +
///                             anti-spoof CNN), at most one job in flight.
///
/// Frames that arrive while the detector is busy are dropped rather than
/// queued, so latency never builds up and the UI never stalls.
class FaceScanController extends ChangeNotifier {
  FaceScanController({required this.mode}) {
    _engine = LivenessEngine(_settings.livenessConfig(register: mode == ScanMode.register));
  }

  static const int registerSamples = 5;

  final ScanMode mode;
  final FaceDetector _detector = FaceDetector(
    options: FaceDetectorOptions(
      performanceMode: FaceDetectorMode.fast,
      enableClassification: true, // eye-open + smile probabilities
      enableLandmarks: true, // eye positions for roll alignment
      enableTracking: true,
      minFaceSize: 0.15,
    ),
  );
  late LivenessEngine _engine;

  // ---- camera ----
  CameraController? camera;
  List<CameraDescription> _cameras = const [];
  int _cameraIdx = 0;
  bool get isFrontCamera =>
      _cameras.isNotEmpty && _cameras[_cameraIdx].lensDirection == CameraLensDirection.front;
  bool get canSwitchCamera => _cameras.length > 1;

  // ---- state read by the UI ----
  ScanStatus status = ScanStatus.starting;
  String message = 'Starting camera…';
  String? error;
  Rect? faceBox;
  /// Upright size of the analysed frames (ML Kit coordinate space).
  Size imageSize = const Size(480, 720);

  /// Upright size of the preview texture.
  Size previewSize = const Size(480, 720);
  int faceCount = 0;
  double progress = 0;
  FaceMatch? match;
  double? bestSimilarity;
  bool identifyDone = false;
  RegistrationCapture? capture;
  LivenessPhase get phase => _engine.phase;

  // ---- perf stats ----
  double fps = 0;
  int detectMs = 0, embedMs = 0, spoofMs = 0;
  double? lastSpoofScore;
  int? lastSharpness;
  int _fpsFrames = 0, _fpsStart = 0;

  // ---- internals ----
  bool _detBusy = false, _mlBusy = false, _disposed = false, _paused = false, _switching = false, _starting = false;
  int? _trackId;
  int _lastFaceMs = 0;
  int _generation = 0;
  Float32List? _anchor;
  final List<Float32List> _recent = [];
  final List<Float32List> _captures = [];
  Uint8List? _thumb;
  int _lastEmbedMs = 0, _lastSpoofReqMs = 0;

  AppSettings get _settings => AppServices.settings.value;
  int get _now => DateTime.now().millisecondsSinceEpoch;

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  Future<void> init() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        _setError('No camera found on this device');
        return;
      }
      final front = _cameras.indexWhere((c) => c.lensDirection == CameraLensDirection.front);
      _cameraIdx = front >= 0 ? front : 0;
      await _startCamera();
    } on CameraException catch (e) {
      _setError(_cameraErrorText(e));
    } catch (e) {
      _setError('Could not start camera: $e');
    }
  }

  String _cameraErrorText(CameraException e) {
    if (e.code.contains('AccessDenied') || e.code.contains('Permission')) {
      return 'Camera permission denied.\nEnable it in Android Settings → Apps → Face Guard.';
    }
    return 'Camera error: ${e.description ?? e.code}';
  }

  Future<void> _startCamera() async {
    if (_starting) return;
    _starting = true;
    try {
      await _startCameraInner();
    } finally {
      _starting = false;
    }
  }

  Future<void> _startCameraInner() async {
    final c = CameraController(
      _cameras[_cameraIdx],
      // 720x480: plenty for faces at arm's length and keeps detection fast.
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888,
    );
    try {
      await c.initialize();
    } catch (_) {
      await c.dispose();
      rethrow;
    }
    if (_disposed || _paused) {
      await c.dispose();
      return;
    }
    camera = c;
    final ps = c.value.previewSize;
    if (ps != null) {
      previewSize = Size(math.min(ps.width, ps.height), math.max(ps.width, ps.height));
      imageSize = previewSize;
    }
    _resetSession();
    _trackId = null;
    error = null;
    status = ScanStatus.noFace;
    message = 'Look at the camera';
    notifyListeners();
    if (capture == null) await c.startImageStream(_onImage);
  }

  Future<void> _stopCamera() async {
    final c = camera;
    camera = null;
    if (!_disposed) notifyListeners();
    if (c == null) return;
    try {
      if (c.value.isStreamingImages) await c.stopImageStream();
    } catch (_) {}
    await c.dispose();
  }

  /// Release the camera (app went to background).
  Future<void> pause() async {
    _paused = true;
    await _stopCamera();
  }

  /// Re-acquire the camera (app back in foreground).
  Future<void> resume() async {
    if (_disposed || camera != null) return;
    _paused = false;
    // Still initialising (e.g. the permission prompt caused inactive/resumed).
    if (_starting) return;
    try {
      await _startCamera();
    } on CameraException catch (e) {
      _setError(_cameraErrorText(e));
    }
  }

  Future<void> switchCamera() async {
    if (_switching || _cameras.length < 2) return;
    _switching = true;
    try {
      await _stopCamera();
      _cameraIdx = (_cameraIdx + 1) % _cameras.length;
      await _startCamera();
    } on CameraException catch (e) {
      _setError(_cameraErrorText(e));
    } finally {
      _switching = false;
    }
  }

  /// Registration: discard the capture and scan again.
  Future<void> retake() async {
    capture = null;
    _resetSession();
    status = ScanStatus.noFace;
    message = 'Look at the camera';
    notifyListeners();
    final c = camera;
    if (c != null && !c.value.isStreamingImages) await c.startImageStream(_onImage);
  }

  @override
  void dispose() {
    _disposed = true;
    final c = camera;
    camera = null;
    if (c != null) {
      () async {
        try {
          if (c.value.isStreamingImages) await c.stopImageStream();
        } catch (_) {}
        await c.dispose();
      }();
    }
    _detector.close();
    super.dispose();
  }

  void _setError(String text) {
    error = text;
    status = ScanStatus.error;
    message = text;
    if (!_disposed) notifyListeners();
  }

  void _resetSession() {
    _generation++;
    _engine = LivenessEngine(_settings.livenessConfig(register: mode == ScanMode.register));
    _anchor = null;
    _recent.clear();
    _captures.clear();
    _thumb = null;
    match = null;
    bestSimilarity = null;
    identifyDone = false;
    progress = 0;
  }

  // ===========================================================================
  // Frame pipeline
  // ===========================================================================

  void _onImage(CameraImage image) {
    if (_detBusy || _disposed || _paused || capture != null) return; // drop frame
    _detBusy = true;
    _processImage(image).whenComplete(() => _detBusy = false);
  }

  Future<void> _processImage(CameraImage image) async {
    try {
      // App is locked to portrait, so the needed rotation is the sensor's.
      final rotation = _cameras[_cameraIdx].sensorOrientation;
      final frame = _toNv21(image, rotation);
      if (frame == null) {
        _setError('Unsupported camera image format (${image.format.group})');
        return;
      }
      final input = InputImage.fromBytes(
        bytes: frame.bytes,
        metadata: InputImageMetadata(
          size: Size(frame.width.toDouble(), frame.height.toDouble()),
          rotation: InputImageRotationValue.fromRawValue(rotation) ?? InputImageRotation.rotation0deg,
          format: InputImageFormat.nv21,
          bytesPerRow: frame.rowStride,
        ),
      );
      final sw = Stopwatch()..start();
      final faces = await _detector.processImage(input);
      detectMs = sw.elapsedMilliseconds;
      if (_disposed || capture != null) return;
      _tickFps();
      _handleFaces(frame, faces);
      notifyListeners();
    } catch (e, st) {
      debugPrint('Frame processing error: $e\n$st');
    }
  }

  Nv21Frame? _toNv21(CameraImage img, int rotation) {
    final planes = img.planes;
    if (planes.isEmpty) return null;
    final p0 = planes.first;
    final minNv21 = img.width * img.height * 3 ~/ 2;
    // Normal Android path: a single NV21 plane.
    if (planes.length == 1 || (img.format.group == ImageFormatGroup.nv21 && p0.bytes.length >= minNv21)) {
      if (p0.bytes.length < p0.bytesPerRow * (img.height - 1) + img.width) return null;
      return Nv21Frame(
        bytes: p0.bytes,
        width: img.width,
        height: img.height,
        rowStride: p0.bytesPerRow,
        rotation: rotation,
      );
    }
    // Fallback: 3-plane YUV_420_888.
    if (planes.length == 3) {
      final bytes = yuv420ToNv21(
        width: img.width,
        height: img.height,
        y: planes[0].bytes,
        yRowStride: planes[0].bytesPerRow,
        u: planes[1].bytes,
        v: planes[2].bytes,
        uvRowStride: planes[1].bytesPerRow,
        uvPixelStride: planes[1].bytesPerPixel ?? 1,
      );
      return Nv21Frame(bytes: bytes, width: img.width, height: img.height, rowStride: img.width, rotation: rotation);
    }
    return null;
  }

  void _tickFps() {
    final now = _now;
    if (_fpsStart == 0) _fpsStart = now;
    _fpsFrames++;
    final dt = now - _fpsStart;
    if (dt >= 1000) {
      fps = _fpsFrames * 1000 / dt;
      _fpsFrames = 0;
      _fpsStart = now;
    }
  }

  static double _area(Face f) => f.boundingBox.width * f.boundingBox.height;

  void _handleFaces(Nv21Frame frame, List<Face> faces) {
    final now = _now;
    imageSize = Size(frame.uprightWidth.toDouble(), frame.uprightHeight.toDouble());
    faceCount = faces.length;

    // Keep a failure message on screen briefly, then start over.
    if (_engine.phase == LivenessPhase.failed) {
      status = ScanStatus.failed;
      message = _engine.instruction;
      progress = 0;
      faceBox = faces.isEmpty ? null : faces.first.boundingBox;
      if (now - _engine.failedAtMs > 2000) _resetSession();
      return;
    }

    if (faces.isEmpty) {
      faceBox = null;
      if (now - _lastFaceMs > 800) {
        if (_engine.phase != LivenessPhase.idle) _resetSession();
        _trackId = null;
        status = ScanStatus.noFace;
        message = AppServices.repo.count == 0 && mode == ScanMode.recognize
            ? 'No faces registered yet – register someone first'
            : 'Look at the camera';
        progress = 0;
      }
      return;
    }

    faces.sort((a, b) => _area(b).compareTo(_area(a)));
    final face = faces.first;
    _lastFaceMs = now;
    faceBox = face.boundingBox;

    // A different tracked face means a different person: start over.
    final tid = face.trackingId;
    if (tid != null && tid != _trackId) {
      if (_trackId != null) _resetSession();
      _trackId = tid;
    }

    if (_engine.phase == LivenessPhase.idle) _engine.start();

    final sig = FaceSignals(
      tMs: now,
      leftEyeOpen: face.leftEyeOpenProbability,
      rightEyeOpen: face.rightEyeOpenProbability,
      smile: face.smilingProbability,
      yaw: face.headEulerAngleY ?? 0,
      pitch: face.headEulerAngleX ?? 0,
      roll: face.headEulerAngleZ ?? 0,
    );
    final hint = _positionHint(face, faces);
    _engine.onFrame(sig, wellPositioned: hint == null);

    if (mode == ScanMode.recognize &&
        _engine.phase == LivenessPhase.passed &&
        !identifyDone &&
        _recent.isNotEmpty) {
      _identify();
    }

    _maybeDispatch(frame, face, sig, hint, now);
    _updateStatus(hint);
  }

  String? _positionHint(Face face, List<Face> faces) {
    final b = face.boundingBox;
    final w = imageSize.width, h = imageSize.height;
    if (mode == ScanMode.register && faces.length > 1 && _area(faces[1]) > 0.4 * _area(face)) {
      return 'Only one person in the frame, please';
    }
    if (b.width < _settings.minFaceRatio * w) return 'Move closer';
    if (b.width > 0.85 * w) return 'Move back a little';
    final m = 0.03 * w;
    if (b.left < -m || b.top < -m || b.right > w + m || b.bottom > h + m) {
      return 'Center your face in the frame';
    }
    if ((face.headEulerAngleY ?? 0).abs() > 18 || (face.headEulerAngleX ?? 0).abs() > 18) {
      return 'Look straight at the camera';
    }
    return null;
  }

  /// 5-point alignment (eyes, nose base, mouth corners) for MobileFaceNet.
  /// Returns null when ML Kit did not report all five landmarks.
  CropSpec? _alignedCrop(Face face) {
    math.Point<double>? pt(FaceLandmarkType t) {
      final p = face.landmarks[t]?.position;
      return p == null ? null : math.Point<double>(p.x.toDouble(), p.y.toDouble());
    }

    final e1 = pt(FaceLandmarkType.leftEye), e2 = pt(FaceLandmarkType.rightEye);
    final nose = pt(FaceLandmarkType.noseBase);
    final m1 = pt(FaceLandmarkType.leftMouth), m2 = pt(FaceLandmarkType.rightMouth);
    if (e1 == null || e2 == null || nose == null || m1 == null || m2 == null) return null;
    // Template order is by image position (left to right), not anatomical side.
    final eyes = e1.x <= e2.x ? [e1, e2] : [e2, e1];
    final mouth = m1.x <= m2.x ? [m1, m2] : [m2, m1];
    return alignToTemplate([eyes[0], eyes[1], nose, mouth[0], mouth[1]]);
  }

  CropSpec _boxCrop(Face face) {
    final b = face.boundingBox;
    double angle = 0;
    final l = face.landmarks[FaceLandmarkType.leftEye]?.position;
    final r = face.landmarks[FaceLandmarkType.rightEye]?.position;
    if (l != null && r != null) {
      final first = l.x <= r.x ? l : r;
      final second = l.x <= r.x ? r : l;
      angle = math.atan2((second.y - first.y).toDouble(), (second.x - first.x).toDouble());
      angle = angle.clamp(-0.7, 0.7).toDouble();
    }
    return CropSpec(
      cx: b.center.dx,
      cy: b.center.dy,
      side: math.max(b.width, b.height),
      angle: angle,
    );
  }

  void _maybeDispatch(Nv21Frame frame, Face face, FaceSignals sig, String? hint, int now) {
    if (_mlBusy || !AppServices.worker.isReady) return;
    final phase = _engine.phase;
    final frontal = sig.yaw.abs() < 15 && sig.pitch.abs() < 15;

    bool wantSpoof = false;
    if (_settings.antiSpoofEnabled) {
      switch (phase) {
        case LivenessPhase.positioning:
        case LivenessPhase.challenge:
        case LivenessPhase.verifyingPassive:
          // As fast as the worker allows: this is what gates recognition.
          wantSpoof = true;
          break;
        case LivenessPhase.passed:
          // Keep watching after verification, at a lower rate.
          wantSpoof = now - _lastSpoofReqMs >= 400;
          break;
        default:
          break;
      }
    }

    bool wantEmbed = false, wantThumb = false;
    if (frontal) {
      switch (phase) {
        case LivenessPhase.positioning:
          // Compute identity early so the name appears the moment liveness passes.
          wantEmbed = (_anchor == null || now - _lastEmbedMs >= 300) && hint == null;
          break;
        case LivenessPhase.challenge:
        case LivenessPhase.verifyingPassive:
          wantEmbed = _anchor == null || now - _lastEmbedMs >= 300;
          break;
        case LivenessPhase.passed:
          if (mode == ScanMode.recognize) {
            wantEmbed = now - _lastEmbedMs >= (match == null ? 250 : 1000);
          } else {
            wantEmbed = hint == null && _captures.length < registerSamples && now - _lastEmbedMs >= 150;
            wantThumb = wantEmbed && _thumb == null;
          }
          break;
        default:
          break;
      }
    }
    final faceCrop = wantEmbed ? _alignedCrop(face) : null;
    if (faceCrop == null) {
      wantEmbed = false;
      wantThumb = false;
    }
    if (!wantSpoof && !wantEmbed) return;

    _mlBusy = true;
    if (wantSpoof) _lastSpoofReqMs = now;
    if (wantEmbed) _lastEmbedMs = now;
    final gen = _generation;
    AppServices.worker
        .process(
          frame: frame,
          boxCrop: _boxCrop(face),
          faceCrop: faceCrop,
          embed: wantEmbed,
          spoof: wantSpoof,
          thumb: wantThumb,
        )
        .then((r) {
      if (!_disposed && gen == _generation && capture == null) _onMlResult(r);
    }).catchError((Object e) {
      debugPrint('ML worker error: $e');
    }).whenComplete(() => _mlBusy = false);
  }

  void _onMlResult(MlResult r) {
    if (r.error != null) {
      debugPrint('ML worker error: ${r.error}');
      return;
    }
    final now = _now;
    if (r.embedMs > 0) embedMs = r.embedMs;
    if (r.spoofMs > 0) spoofMs = r.spoofMs;
    if (r.spoofScore != null) {
      lastSpoofScore = r.spoofScore;
      lastSharpness = r.laplacian;
      _engine.onPassive(r.spoofScore!, r.laplacian ?? 0, now);
    }

    final e = r.embedding;
    if (e != null && _engine.phase != LivenessPhase.failed) {
      if (_anchor == null) {
        _anchor = e;
      } else if (dot(_anchor!, e) < math.min(0.5, _settings.matchThreshold - 0.1)) {
        // Someone swapped the face/photo while the track was kept alive.
        _engine.fail('Face changed during verification', now);
      }
      if (_engine.phase != LivenessPhase.failed) {
        if (r.thumbBmp != null) _thumb ??= r.thumbBmp;
        _recent.add(e);
        if (_recent.length > 3) _recent.removeAt(0);
        if (_engine.phase == LivenessPhase.passed) {
          if (mode == ScanMode.recognize) {
            _identify();
          } else {
            _captures.add(e);
            if (_captures.length >= registerSamples) _finishRegistration();
          }
        }
      }
    }
    _updateStatus(null);
    notifyListeners();
  }

  void _identify() {
    final thr = _settings.matchThreshold;
    final m = AppServices.repo.index.best(averageEmbeddings(_recent));
    identifyDone = true;
    bestSimilarity = m?.similarity;
    if (m == null) {
      match = null;
    } else if (m.similarity >= thr) {
      match = m;
    } else if (match != null && match!.personId == m.personId && m.similarity >= thr - 0.05) {
      match = m; // hysteresis: avoid flicker around the threshold
    } else {
      match = null;
    }
  }

  void _finishRegistration() {
    final emb = averageEmbeddings(_captures);
    final best = AppServices.repo.index.best(emb);
    capture = RegistrationCapture(
      embedding: emb,
      thumbBmp: _thumb,
      duplicate: (best != null && best.similarity >= _settings.matchThreshold) ? best : null,
    );
    status = ScanStatus.captured;
    message = 'Face captured';
    progress = 1;
    final c = camera;
    if (c != null && c.value.isStreamingImages) {
      c.stopImageStream().catchError((_) {});
    }
  }

  void _updateStatus(String? hint) {
    if (capture != null) return;
    switch (_engine.phase) {
      case LivenessPhase.idle:
        status = ScanStatus.noFace;
        message = 'Look at the camera';
        break;
      case LivenessPhase.positioning:
        status = ScanStatus.adjusting;
        message = hint ?? 'Hold still…';
        break;
      case LivenessPhase.challenge:
        status = ScanStatus.challenge;
        message = _engine.instruction;
        break;
      case LivenessPhase.verifyingPassive:
        status = ScanStatus.verifying;
        message = _engine.instruction;
        break;
      case LivenessPhase.passed:
        if (mode == ScanMode.recognize) {
          if (match != null) {
            status = ScanStatus.recognized;
            message = match!.name;
          } else if (identifyDone) {
            status = ScanStatus.unknown;
            message = AppServices.repo.count == 0 ? 'Unknown – no faces registered' : 'Unknown person';
          } else {
            status = ScanStatus.verifying;
            message = 'Identifying…';
          }
        } else {
          status = ScanStatus.verifying;
          message = hint ?? 'Capturing – keep looking at the camera';
        }
        break;
      case LivenessPhase.failed:
        status = ScanStatus.failed;
        message = _engine.instruction;
        break;
    }
    progress = _engine.phase == LivenessPhase.passed && mode == ScanMode.register
        ? 0.9 + 0.1 * _captures.length / registerSamples
        : _engine.progress;
  }
}
