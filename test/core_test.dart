import 'dart:math' as math;
import 'dart:typed_data';

import 'package:face_guard/core/bmp.dart';
import 'package:face_guard/core/face_align.dart';
import 'package:face_guard/core/face_index.dart';
import 'package:face_guard/core/liveness.dart';
import 'package:face_guard/core/settings.dart';
import 'package:face_guard/core/yuv_sampler.dart';
import 'package:flutter_test/flutter_test.dart';

Float32List randomEmbedding(math.Random r) =>
    l2Normalize(Float32List.fromList(List.generate(kEmbeddingSize, (_) => r.nextDouble() * 2 - 1)));

Float32List noisy(Float32List e, math.Random r, double amount) => l2Normalize(
    Float32List.fromList(List.generate(e.length, (i) => e[i] + (r.nextDouble() * 2 - 1) * amount)));

/// Builds a grey NV21 sensor frame whose upright image has luma f(x, y).
Nv21Frame frameFromUpright(int uw, int uh, int rotation, double Function(double x, double y) f) {
  final w = rotation % 180 == 0 ? uw : uh;
  final h = rotation % 180 == 0 ? uh : uw;
  final bytes = Uint8List(w * h * 3 ~/ 2)..fillRange(w * h, w * h * 3 ~/ 2, 128);
  for (int ry = 0; ry < h; ry++) {
    for (int rx = 0; rx < w; rx++) {
      // raw pixel centre -> upright continuous coords (forward of the sampler's inverse map)
      final cx = rx + 0.5, cy = ry + 0.5;
      double ux, uy;
      switch (rotation) {
        case 90:
          ux = h - cy;
          uy = cx;
          break;
        case 180:
          ux = w - cx;
          uy = h - cy;
          break;
        case 270:
          ux = cy;
          uy = w - cx;
          break;
        default:
          ux = cx;
          uy = cy;
      }
      bytes[ry * w + rx] = f(ux, uy).round();
    }
  }
  return Nv21Frame(bytes: bytes, width: w, height: h, rowStride: w, rotation: rotation);
}

void main() {
  group('FaceIndex', () {
    test('finds the right person among 1000 and is fast', () {
      final r = math.Random(1);
      final index = FaceIndex();
      final people = <Float32List>[];
      for (int i = 0; i < kMaxFaces; i++) {
        final e = randomEmbedding(r);
        people.add(e);
        index.add(i, 'p$i', e);
      }
      expect(index.isFull, isTrue);
      expect(() => index.add(5000, 'x', randomEmbedding(r)), throwsStateError);

      final sw = Stopwatch()..start();
      for (int k = 0; k < 50; k++) {
        final target = r.nextInt(kMaxFaces);
        final m = index.best(noisy(people[target], r, 0.03))!;
        expect(m.personId, target);
        expect(m.similarity, greaterThan(0.9));
      }
      sw.stop();
      // 50 full searches over 1000 faces; on a phone (AOT) each is < 1 ms.
      expect(sw.elapsedMilliseconds, lessThan(1000));
    });

    test('remove and rename keep the matrix consistent', () {
      final r = math.Random(2);
      final index = FaceIndex(capacity: 3);
      final a = randomEmbedding(r), b = randomEmbedding(r), c = randomEmbedding(r);
      index
        ..add(1, 'a', a)
        ..add(2, 'b', b)
        ..add(3, 'c', c);
      expect(index.remove(1), isTrue);
      expect(index.length, 2);
      expect(index.best(c)!.personId, 3);
      expect(index.best(b)!.personId, 2);
      index.rename(3, 'charlie');
      expect(index.best(c)!.name, 'charlie');
      index.add(4, 'd', a); // freed slot reusable
      expect(index.best(a)!.personId, 4);
      expect(index.remove(99), isFalse);
    });

    test('averageEmbeddings is normalised', () {
      final r = math.Random(3);
      final avg = averageEmbeddings([randomEmbedding(r), randomEmbedding(r)]);
      expect(dot(avg, avg), closeTo(1, 1e-5));
    });
  });

  group('yuv sampler', () {
    // Linear luma so bilinear interpolation is exact away from borders.
    double f(double x, double y) => 30 + 0.5 * x + 0.3 * y;

    for (final rot in [0, 90, 180, 270]) {
      test('upright mapping is correct for rotation $rot', () {
        final frame = frameFromUpright(160, 240, rot, f);
        expect(frame.uprightWidth, 160);
        expect(frame.uprightHeight, 240);
        for (final angle in [0.0, 0.3, -0.4]) {
          const crop = CropSpec(cx: 80, cy: 120, side: 60);
          final spec = CropSpec(cx: crop.cx, cy: crop.cy, side: crop.side, angle: angle);
          const size = 20;
          final rgb = sampleRgb(frame, spec, size);
          final step = spec.side / size;
          for (int v = 0; v < size; v += 3) {
            for (int u = 0; u < size; u += 3) {
              final dx = (u + 0.5) * step - spec.side / 2;
              final dy = (v + 0.5) * step - spec.side / 2;
              final ux = spec.cx + dx * math.cos(angle) - dy * math.sin(angle);
              final uy = spec.cy + dx * math.sin(angle) + dy * math.cos(angle);
              final got = rgb[(v * size + u) * 3];
              expect(got, closeTo(f(ux, uy), 1.5), reason: 'rot=$rot angle=$angle u=$u v=$v');
            }
          }
        }
      });
    }

    test('row padding gives identical output', () {
      final compact = frameFromUpright(64, 96, 270, f);
      const pad = 16;
      final w = compact.width, h = compact.height, stride = w + pad;
      final padded = Uint8List(stride * h + stride * (h ~/ 2));
      for (int y = 0; y < h; y++) {
        padded.setRange(y * stride, y * stride + w, compact.bytes, y * w);
      }
      for (int y = 0; y < h ~/ 2; y++) {
        padded.setRange(stride * h + y * stride, stride * h + y * stride + w, compact.bytes, w * h + y * w);
      }
      final pf = Nv21Frame(bytes: padded, width: w, height: h, rowStride: stride, rotation: 270);
      const crop = CropSpec(cx: 32, cy: 48, side: 40, angle: 0.2);
      expect(sampleRgb(pf, crop, 24), sampleRgb(compact, crop, 24));
    });

    test('yuv420ToNv21 interleaves V then U', () {
      final nv = yuv420ToNv21(
        width: 4,
        height: 2,
        y: Uint8List.fromList(List.generate(8, (i) => i)),
        yRowStride: 4,
        u: Uint8List.fromList([10, 0, 11]),
        v: Uint8List.fromList([20, 0, 21]),
        uvRowStride: 4,
        uvPixelStride: 2,
      );
      expect(nv.sublist(0, 8), List.generate(8, (i) => i));
      expect(nv.sublist(8), [20, 10, 21, 11]);
    });

    test('laplacian is 0 on flat image and high on a checkerboard', () {
      final flat = Uint8List(32 * 32 * 3)..fillRange(0, 32 * 32 * 3, 100);
      expect(laplacianScore(flat, 32), 0);
      final cb = Uint8List(32 * 32 * 3);
      for (int i = 0; i < 32 * 32; i++) {
        final v = ((i % 32) + (i ~/ 32)).isEven ? 255 : 0;
        cb[i * 3] = cb[i * 3 + 1] = cb[i * 3 + 2] = v;
      }
      expect(laplacianScore(cb, 32), greaterThan(400));
    });
  });

  group('face alignment', () {
    test('template points map onto themselves', () {
      final pts = [for (final p in kArcFaceTemplate) math.Point<double>(p[0], p[1])];
      final c = alignToTemplate(pts);
      expect(c.cx, closeTo(56, 1e-6));
      expect(c.cy, closeTo(56, 1e-6));
      expect(c.side, closeTo(112, 1e-6));
      expect(c.angle, closeTo(0, 1e-9));
    });

    test('recovers scale, rotation and translation', () {
      const s = 2.5, th = 0.3, tx = 140.0, ty = 210.0;
      final pts = [
        for (final p in kArcFaceTemplate)
          math.Point<double>(
            s * (p[0] * math.cos(th) - p[1] * math.sin(th)) + tx,
            s * (p[0] * math.sin(th) + p[1] * math.cos(th)) + ty,
          )
      ];
      final c = alignToTemplate(pts);
      expect(c.side, closeTo(112 * s, 1e-6));
      expect(c.angle, closeTo(th, 1e-9));
      expect(c.cx, closeTo(s * (56 * math.cos(th) - 56 * math.sin(th)) + tx, 1e-6));
      expect(c.cy, closeTo(s * (56 * math.sin(th) + 56 * math.cos(th)) + ty, 1e-6));
    });
  });

  group('liveness', () {
    const open = 0.95, closed = 0.05;

    /// Drives whatever challenge is active with a genuine response.
    List<FaceSignals> respond(ChallengeType t, int t0) {
      FaceSignals s(int dt, {double eyes = open, double smile = 0.1, double yaw = 0, double pitch = 0}) =>
          FaceSignals(tMs: t0 + dt, leftEyeOpen: eyes, rightEyeOpen: eyes, smile: smile, yaw: yaw, pitch: pitch);
      switch (t) {
        case ChallengeType.blink:
          return [s(0), s(60, eyes: closed), s(120, eyes: closed), s(200)];
        case ChallengeType.smile:
          return [s(0), s(150, smile: 0.95)];
        case ChallengeType.turnHead:
          return [s(0), s(200, yaw: 30), s(400, yaw: 5)];
        case ChallengeType.nod:
          return [s(0), s(200, pitch: -18), s(400, pitch: 3)];
      }
    }

    LivenessEngine engine(int challenges, {int seed = 7}) =>
        LivenessEngine(LivenessConfig(challengeCount: challenges, minLaplacian: 50), random: math.Random(seed))..start();

    void feedLive(LivenessEngine e, int t) {
      for (int i = 0; i < 6; i++) {
        e.onPassive(0.02, 500, t);
      }
    }

    test('first challenge is always blink or smile; count respected', () {
      for (int seed = 0; seed < 30; seed++) {
        final e = engine(3, seed: seed);
        expect([ChallengeType.blink, ChallengeType.smile], contains(e.challenges.first));
        expect(e.challenges.length, 3);
        expect(e.challenges.toSet().length, 3);
      }
    });

    test('live user completing random challenges passes', () {
      for (int seed = 0; seed < 20; seed++) {
        final e = engine(3, seed: seed);
        int t = 0;
        e.onFrame(FaceSignals(tMs: t, leftEyeOpen: open, rightEyeOpen: open), wellPositioned: true);
        t += 400;
        e.onFrame(FaceSignals(tMs: t, leftEyeOpen: open, rightEyeOpen: open), wellPositioned: true);
        expect(e.phase, LivenessPhase.challenge);
        feedLive(e, t);
        while (e.phase == LivenessPhase.challenge) {
          for (final s in respond(e.tracker!.type, t)) {
            e.onFrame(s, wellPositioned: true);
          }
          t += 600;
        }
        expect(e.phase, LivenessPhase.passed, reason: 'seed $seed: ${e.failureReason}');
      }
    });

    test('static photo (no expression change) times out', () {
      final e = engine(1);
      for (int t = 0; t <= 9000; t += 33) {
        e.onFrame(FaceSignals(tMs: t, leftEyeOpen: open, rightEyeOpen: open, smile: 0.9), wellPositioned: true);
        if (e.phase == LivenessPhase.failed) break;
      }
      expect(e.phase, LivenessPhase.failed);
      expect(e.failureReason, contains('Timed out'));
    });

    test('passive spoof scores fail immediately', () {
      final e = engine(1);
      for (int i = 0; i < 5; i++) {
        e.onPassive(0.9, 500, i * 100);
      }
      expect(e.phase, LivenessPhase.failed);
      expect(e.failureReason, contains('Spoof'));
    });

    test('challenges alone are not enough without passive evidence', () {
      final e = engine(1);
      int t = 0;
      e.onFrame(FaceSignals(tMs: t, leftEyeOpen: open, rightEyeOpen: open, smile: 0.1), wellPositioned: true);
      t = 400;
      e.onFrame(FaceSignals(tMs: t, leftEyeOpen: open, rightEyeOpen: open, smile: 0.1), wellPositioned: true);
      for (final s in respond(e.tracker!.type, t)) {
        e.onFrame(s, wellPositioned: true);
      }
      expect(e.phase, LivenessPhase.verifyingPassive);
      feedLive(e, t + 700);
      expect(e.phase, LivenessPhase.passed);
    });

    test('blurry frames are ignored, not counted', () {
      final e = engine(1);
      for (int i = 0; i < 20; i++) {
        e.onPassive(0.9, 10, i * 50);
      }
      expect(e.passiveScores, isEmpty);
      expect(e.tooBlurry, isTrue);
      expect(e.phase, isNot(LivenessPhase.failed));
    });

    test('eyes closed for too long does not count as a blink', () {
      final tr = ChallengeTracker(ChallengeType.blink, 0);
      expect(tr.update(const FaceSignals(tMs: 0, leftEyeOpen: open, rightEyeOpen: open)), isFalse);
      expect(tr.update(const FaceSignals(tMs: 50, leftEyeOpen: closed, rightEyeOpen: closed)), isFalse);
      expect(tr.update(const FaceSignals(tMs: 2600, leftEyeOpen: closed, rightEyeOpen: closed)), isFalse);
      expect(tr.update(const FaceSignals(tMs: 2700, leftEyeOpen: open, rightEyeOpen: open)), isFalse);
    });
  });

  test('BMP header and pixel order', () {
    final rgb = Uint8List.fromList([255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30]); // 2x2
    final bmp = encodeBmp(rgb, 2, 2);
    expect(String.fromCharCodes(bmp.sublist(0, 2)), 'BM');
    final bd = ByteData.sublistView(bmp);
    expect(bd.getUint32(2, Endian.little), bmp.length);
    expect(bd.getInt32(18, Endian.little), 2);
    expect(bd.getUint16(28, Endian.little), 24);
    // bottom-up: first stored row is image row 1 => pixels (0,0,255) and (10,20,30) as BGR
    expect(bmp.sublist(54, 60), [255, 0, 0, 30, 20, 10]);
  });

  test('settings round-trip', () {
    const s = AppSettings(matchThreshold: 0.7, recognizeChallenges: 2, antiSpoofEnabled: false, minSharpness: 120);
    final back = AppSettings.fromMap(s.toMap());
    expect(back.matchThreshold, 0.7);
    expect(back.recognizeChallenges, 2);
    expect(back.antiSpoofEnabled, isFalse);
    expect(back.minSharpness, 120);
    expect(AppSettings.fromMap({}).matchThreshold, const AppSettings().matchThreshold);
  });
}
