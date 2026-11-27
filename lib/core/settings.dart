import 'liveness.dart';

class AppSettings {
  const AppSettings({
    this.matchThreshold = 0.60,
    this.spoofThreshold = 0.20,
    this.recognizeChallenges = 1,
    this.registerChallenges = 2,
    this.minFaceRatio = 0.25,
    this.minSharpness = 80,
    this.antiSpoofEnabled = true,
    this.showStats = true,
  });

  /// Cosine similarity needed to accept an identity (0.60 ≈ the reference
  /// MobileFaceNet threshold of squared-L2 distance 0.8).
  final double matchThreshold;
  final double spoofThreshold;
  final int recognizeChallenges;
  final int registerChallenges;

  /// Minimum face width relative to the frame width.
  final double minFaceRatio;

  /// Anti-spoof frames whose crop has fewer strong edges than this are
  /// treated as too blurry to judge and skipped.
  final int minSharpness;
  final bool antiSpoofEnabled;
  final bool showStats;

  LivenessConfig livenessConfig({required bool register}) => LivenessConfig(
        challengeCount: register ? registerChallenges : recognizeChallenges,
        passiveEnabled: antiSpoofEnabled,
        spoofThreshold: spoofThreshold,
        minLaplacian: minSharpness,
      );

  AppSettings copyWith({
    double? matchThreshold,
    double? spoofThreshold,
    int? recognizeChallenges,
    int? registerChallenges,
    double? minFaceRatio,
    int? minSharpness,
    bool? antiSpoofEnabled,
    bool? showStats,
  }) =>
      AppSettings(
        matchThreshold: matchThreshold ?? this.matchThreshold,
        spoofThreshold: spoofThreshold ?? this.spoofThreshold,
        recognizeChallenges: recognizeChallenges ?? this.recognizeChallenges,
        registerChallenges: registerChallenges ?? this.registerChallenges,
        minFaceRatio: minFaceRatio ?? this.minFaceRatio,
        minSharpness: minSharpness ?? this.minSharpness,
        antiSpoofEnabled: antiSpoofEnabled ?? this.antiSpoofEnabled,
        showStats: showStats ?? this.showStats,
      );

  Map<String, String> toMap() => {
        'matchThreshold': '$matchThreshold',
        'spoofThreshold': '$spoofThreshold',
        'recognizeChallenges': '$recognizeChallenges',
        'registerChallenges': '$registerChallenges',
        'minFaceRatio': '$minFaceRatio',
        'minSharpness': '$minSharpness',
        'antiSpoofEnabled': '$antiSpoofEnabled',
        'showStats': '$showStats',
      };

  factory AppSettings.fromMap(Map<String, String> m) {
    const d = AppSettings();
    double dbl(String k, double def) => double.tryParse(m[k] ?? '') ?? def;
    int i(String k, int def) => int.tryParse(m[k] ?? '') ?? def;
    bool b(String k, bool def) => m[k] == null ? def : m[k] == 'true';
    return AppSettings(
      matchThreshold: dbl('matchThreshold', d.matchThreshold),
      spoofThreshold: dbl('spoofThreshold', d.spoofThreshold),
      recognizeChallenges: i('recognizeChallenges', d.recognizeChallenges),
      registerChallenges: i('registerChallenges', d.registerChallenges),
      minFaceRatio: dbl('minFaceRatio', d.minFaceRatio),
      minSharpness: i('minSharpness', d.minSharpness),
      antiSpoofEnabled: b('antiSpoofEnabled', d.antiSpoofEnabled),
      showStats: b('showStats', d.showStats),
    );
  }
}
