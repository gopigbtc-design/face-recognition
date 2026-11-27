import 'dart:math' as math;

/// Per-frame face signals (from ML Kit), decoupled from the plugin so the
/// liveness logic is pure Dart and unit-testable.
class FaceSignals {
  const FaceSignals({
    required this.tMs,
    this.leftEyeOpen,
    this.rightEyeOpen,
    this.smile,
    this.yaw = 0,
    this.pitch = 0,
    this.roll = 0,
  });

  final int tMs;
  final double? leftEyeOpen;
  final double? rightEyeOpen;
  final double? smile;
  final double yaw;
  final double pitch;
  final double roll;
}

enum ChallengeType { blink, smile, turnHead, nod }

extension ChallengeText on ChallengeType {
  String get instruction {
    switch (this) {
      case ChallengeType.blink:
        return 'Blink your eyes';
      case ChallengeType.smile:
        return 'Give a big smile';
      case ChallengeType.turnHead:
        return 'Turn your head to one side, then back';
      case ChallengeType.nod:
        return 'Nod your head (up/down), then back';
    }
  }
}

/// Tracks a single active challenge. Each challenge requires a *change* in
/// state (e.g. open → closed → open), which a printed photo or a still image
/// on a screen cannot produce.
class ChallengeTracker {
  ChallengeTracker(this.type, this.startMs);

  final ChallengeType type;
  final int startMs;
  int stage = 0;
  int _stageMs = 0;

  int get _stages => type == ChallengeType.smile ? 2 : 3;
  double get progress => stage / _stages;

  /// Returns true once the challenge has been completed.
  bool update(FaceSignals s) {
    switch (type) {
      case ChallengeType.blink:
        final l = s.leftEyeOpen, r = s.rightEyeOpen;
        if (l == null || r == null) return false;
        final open = l > 0.6 && r > 0.6;
        final closed = l < 0.3 && r < 0.3;
        if (stage == 0 && open) {
          stage = 1;
        } else if (stage == 1 && closed) {
          stage = 2;
          _stageMs = s.tMs;
        } else if (stage == 2) {
          if (open) {
            stage = 3;
            return true;
          }
          // A real blink is short; eyes "closed" for long is suspicious.
          if (s.tMs - _stageMs > 2000) stage = 1;
        }
        return false;
      case ChallengeType.smile:
        final sm = s.smile;
        if (sm == null) return false;
        if (stage == 0 && sm < 0.35) {
          stage = 1;
        } else if (stage == 1 && sm > 0.8) {
          stage = 2;
          return true;
        }
        return false;
      case ChallengeType.turnHead:
        final a = s.yaw.abs();
        if (stage == 0 && a < 12) {
          stage = 1;
        } else if (stage == 1 && a > 25) {
          stage = 2;
        } else if (stage == 2 && a < 12) {
          stage = 3;
          return true;
        }
        return false;
      case ChallengeType.nod:
        final a = s.pitch.abs();
        if (stage == 0 && a < 10) {
          stage = 1;
        } else if (stage == 1 && a > 14) {
          stage = 2;
        } else if (stage == 2 && a < 10) {
          stage = 3;
          return true;
        }
        return false;
    }
  }
}

class LivenessConfig {
  const LivenessConfig({
    this.challengeCount = 0,
    this.challengeTimeoutMs = 7000,
    this.positionHoldMs = 150,
    this.adaptiveBlink = true,
    this.passiveEnabled = true,
    this.spoofThreshold = 0.2,
    this.minPassiveSamples = 3,
    this.passiveWindow = 10,
    this.minLaplacian = 80,
    this.passiveTimeoutMs = 4000,
  });

  /// 0 = passive only (no user action needed).
  final int challengeCount;
  final int challengeTimeoutMs;

  /// In passive mode, ask for a single blink only when the anti-spoof score
  /// is borderline (between half the threshold and the threshold).
  final bool adaptiveBlink;
  final int positionHoldMs;
  final bool passiveEnabled;

  /// Anti-spoof CNN score above which a frame is considered an attack.
  final double spoofThreshold;
  final int minPassiveSamples;
  final int passiveWindow;

  /// Crops with fewer strong Laplacian edges are too blurry to judge.
  final int minLaplacian;
  final int passiveTimeoutMs;
}

enum LivenessPhase { idle, positioning, challenge, verifyingPassive, passed, failed }

/// Combines active challenges with passive (CNN + texture) anti-spoofing.
class LivenessEngine {
  LivenessEngine(this.config, {math.Random? random}) : _rng = random ?? math.Random();

  LivenessConfig config;
  final math.Random _rng;

  LivenessPhase phase = LivenessPhase.idle;
  List<ChallengeType> challenges = const [];
  int challengeIndex = 0;
  ChallengeTracker? tracker;
  String? failureReason;
  int failedAtMs = 0;
  int? _positionedSince;
  int _passiveStartMs = 0;
  int blurStreak = 0;
  bool escalated = false;
  final List<double> _scores = <double>[];

  List<double> get passiveScores => List.unmodifiable(_scores);
  bool get tooBlurry => blurStreak >= 15;

  void reset() {
    phase = LivenessPhase.idle;
    challenges = const [];
    challengeIndex = 0;
    tracker = null;
    failureReason = null;
    _positionedSince = null;
    blurStreak = 0;
    escalated = false;
    _scores.clear();
  }

  void start() {
    reset();
    phase = LivenessPhase.positioning;
    // Passive mode (count 0): no challenges at all. Otherwise the first
    // challenge is a blink (a still photo can never blink) and the rest are
    // drawn randomly so a pre-recorded video cannot anticipate them.
    final rest = <ChallengeType>[ChallengeType.turnHead, ChallengeType.nod]..shuffle(_rng);
    final int n = math.max(0, math.min(config.challengeCount, 1 + rest.length));
    challenges = [ChallengeType.blink, ...rest].take(n).toList();
  }

  void fail(String reason, int tMs) {
    phase = LivenessPhase.failed;
    failureReason = reason;
    failedAtMs = tMs;
    tracker = null;
  }

  bool get _active =>
      phase == LivenessPhase.positioning ||
      phase == LivenessPhase.challenge ||
      phase == LivenessPhase.verifyingPassive ||
      phase == LivenessPhase.passed;

  /// Feed one detection result. [wellPositioned] = face size/pose/centering OK.
  void onFrame(FaceSignals s, {required bool wellPositioned}) {
    switch (phase) {
      case LivenessPhase.positioning:
        if (wellPositioned) {
          _positionedSince ??= s.tMs;
          if (s.tMs - _positionedSince! >= config.positionHoldMs) {
            if (challenges.isEmpty) {
              phase = LivenessPhase.verifyingPassive;
              _passiveStartMs = s.tMs;
              _checkPassive(s.tMs);
            } else {
              phase = LivenessPhase.challenge;
              challengeIndex = 0;
              tracker = ChallengeTracker(challenges[0], s.tMs);
            }
          }
        } else {
          _positionedSince = null;
        }
        break;
      case LivenessPhase.challenge:
        final t = tracker!;
        if (t.update(s)) {
          challengeIndex++;
          if (challengeIndex >= challenges.length) {
            tracker = null;
            phase = LivenessPhase.verifyingPassive;
            _passiveStartMs = s.tMs;
            _checkPassive(s.tMs);
          } else {
            tracker = ChallengeTracker(challenges[challengeIndex], s.tMs);
          }
        } else if (s.tMs - t.startMs > config.challengeTimeoutMs) {
          fail('Timed out: "${t.type.instruction}"', s.tMs);
        }
        break;
      case LivenessPhase.verifyingPassive:
        _checkPassive(s.tMs);
        break;
      case LivenessPhase.idle:
      case LivenessPhase.passed:
      case LivenessPhase.failed:
        break;
    }
  }

  /// Feed one passive anti-spoof measurement.
  void onPassive(double spoofScore, int laplacian, int tMs) {
    if (!_active) return;
    if (laplacian < config.minLaplacian) {
      blurStreak++;
      return;
    }
    blurStreak = 0;
    _scores.add(spoofScore);
    while (_scores.length > config.passiveWindow) {
      _scores.removeAt(0);
    }
    if (config.passiveEnabled && passiveSpoof) {
      fail('Spoof detected (photo / screen / mask)', tMs);
    } else if (phase == LivenessPhase.verifyingPassive) {
      _checkPassive(tMs);
    }
  }

  double get _median {
    final s = List<double>.from(_scores)..sort();
    final m = s.length ~/ 2;
    return s.length.isOdd ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  double get _fracAbove =>
      _scores.where((v) => v >= config.spoofThreshold).length / _scores.length;

  bool get passiveOk =>
      _scores.length >= config.minPassiveSamples &&
      _median < config.spoofThreshold &&
      _fracAbove <= 0.25;

  bool get passiveSpoof =>
      _scores.length >= config.minPassiveSamples &&
      (_median >= config.spoofThreshold || _fracAbove >= 0.5);

  void _checkPassive(int tMs) {
    if (!config.passiveEnabled) {
      phase = LivenessPhase.passed;
    } else if (passiveOk) {
      if (config.adaptiveBlink &&
          challenges.isEmpty &&
          !escalated &&
          _median >= config.spoofThreshold * 0.5) {
        // Borderline score: confirm with one quick blink.
        escalated = true;
        challenges = const [ChallengeType.blink];
        challengeIndex = 0;
        tracker = ChallengeTracker(ChallengeType.blink, tMs);
        phase = LivenessPhase.challenge;
        return;
      }
      phase = LivenessPhase.passed;
    } else if (tMs - _passiveStartMs > config.passiveTimeoutMs) {
      fail(tooBlurry
          ? 'Image too blurry – hold still, improve lighting'
          : 'Could not confirm a live face', tMs);
    }
  }

  String get instruction {
    switch (phase) {
      case LivenessPhase.idle:
        return 'Look at the camera';
      case LivenessPhase.positioning:
        return 'Look at the camera';
      case LivenessPhase.challenge:
        return tracker?.type.instruction ?? '';
      case LivenessPhase.verifyingPassive:
        return tooBlurry ? 'Hold still – image is blurry' : 'Checking…';
      case LivenessPhase.passed:
        return 'Live face verified';
      case LivenessPhase.failed:
        return failureReason ?? 'Verification failed';
    }
  }

  double get progress {
    switch (phase) {
      case LivenessPhase.challenge:
        final n = challenges.length;
        return 0.9 * (challengeIndex + (tracker?.progress ?? 0)) / n;
      case LivenessPhase.verifyingPassive:
        return 0.95;
      case LivenessPhase.passed:
        return 1;
      default:
        return 0;
    }
  }
}
