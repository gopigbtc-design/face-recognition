import 'dart:math' as math;

import 'yuv_sampler.dart';

/// Standard InsightFace/ArcFace 5-point template for a 112x112 face:
/// image-left eye, image-right eye, nose, image-left mouth corner,
/// image-right mouth corner.
const List<List<double>> kArcFaceTemplate = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

/// Computes the least-squares similarity transform (scale, rotation,
/// translation) that maps [kArcFaceTemplate] onto the detected landmarks and
/// expresses it as a [CropSpec] for a 112x112 sample.
///
/// MobileFaceNet was trained on faces aligned this way; with a plain box crop
/// its embeddings drift badly with small framing changes, while aligned crops
/// stay stable (verified: cos-sim ≥ 0.92 under 3 px landmark noise).
///
/// [points] must be in template order (see above) in upright image space.
CropSpec alignToTemplate(List<math.Point<double>> points) {
  if (points.length != kArcFaceTemplate.length) {
    throw ArgumentError('Need ${kArcFaceTemplate.length} landmarks');
  }
  final n = points.length;
  double psx = 0, psy = 0, qsx = 0, qsy = 0;
  for (int i = 0; i < n; i++) {
    psx += kArcFaceTemplate[i][0];
    psy += kArcFaceTemplate[i][1];
    qsx += points[i].x;
    qsy += points[i].y;
  }
  psx /= n;
  psy /= n;
  qsx /= n;
  qsy /= n;
  double a = 0, b = 0, norm = 0;
  for (int i = 0; i < n; i++) {
    final px = kArcFaceTemplate[i][0] - psx, py = kArcFaceTemplate[i][1] - psy;
    final qx = points[i].x - qsx, qy = points[i].y - qsy;
    a += px * qx + py * qy;
    b += px * qy - py * qx;
    norm += px * px + py * py;
  }
  final sc = a / norm; // s·cosθ
  final ss = b / norm; // s·sinθ
  final s = math.sqrt(sc * sc + ss * ss);
  final angle = math.atan2(ss, sc);
  final tx = qsx - (sc * psx - ss * psy);
  final ty = qsy - (ss * psx + sc * psy);
  const c = 56.0; // template centre
  return CropSpec(
    cx: sc * c - ss * c + tx,
    cy: ss * c + sc * c + ty,
    side: 112 * s,
    angle: angle,
  );
}
