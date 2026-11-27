import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../services/app_services.dart';
import '../services/face_scan_controller.dart';

Color statusColor(ScanStatus s) {
  switch (s) {
    case ScanStatus.recognized:
    case ScanStatus.captured:
      return const Color(0xFF2ECC71);
    case ScanStatus.unknown:
      return const Color(0xFFFF9800);
    case ScanStatus.failed:
    case ScanStatus.error:
      return const Color(0xFFE53935);
    case ScanStatus.challenge:
    case ScanStatus.verifying:
      return const Color(0xFF29B6F6);
    case ScanStatus.adjusting:
      return const Color(0xFFFFC107);
    case ScanStatus.starting:
    case ScanStatus.noFace:
      return Colors.white70;
  }
}

IconData statusIcon(ScanStatus s) {
  switch (s) {
    case ScanStatus.recognized:
      return Icons.verified_user;
    case ScanStatus.captured:
      return Icons.check_circle;
    case ScanStatus.unknown:
      return Icons.help_outline;
    case ScanStatus.failed:
      return Icons.gpp_bad;
    case ScanStatus.error:
      return Icons.error_outline;
    case ScanStatus.challenge:
      return Icons.touch_app;
    case ScanStatus.verifying:
      return Icons.shield_outlined;
    case ScanStatus.adjusting:
      return Icons.center_focus_strong;
    case ScanStatus.starting:
    case ScanStatus.noFace:
      return Icons.face_retouching_natural;
  }
}

/// Full-screen camera preview with face box, status banner and progress.
class FaceScanView extends StatelessWidget {
  const FaceScanView({super.key, required this.controller});

  final FaceScanController controller;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (context, _) {
        final c = controller;
        if (c.error != null) {
          return Center(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Text(c.error!, textAlign: TextAlign.center, style: const TextStyle(fontSize: 16)),
            ),
          );
        }
        final cam = c.camera;
        if (cam == null || !cam.value.isInitialized) {
          return const Center(child: CircularProgressIndicator());
        }
        final color = statusColor(c.status);
        return Stack(
          fit: StackFit.expand,
          children: [
            // Preview and overlay share the upright image coordinate space,
            // then are scaled together to cover the screen.
            ClipRect(
              child: FittedBox(
                fit: BoxFit.cover,
                child: SizedBox(
                  width: c.previewSize.width,
                  height: c.previewSize.height,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      CameraPreview(cam),
                      CustomPaint(
                        painter: FaceOverlayPainter(
                          box: c.faceBox,
                          imageSize: c.imageSize,
                          mirror: c.isFrontCamera,
                          color: color,
                          label: c.status == ScanStatus.recognized ? c.match?.name : null,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            Positioned(left: 12, right: 12, top: 12, child: _StatusBanner(controller: c, color: color)),
            if (AppServices.settings.value.showStats)
              Positioned(left: 12, bottom: 12, child: _Stats(controller: c)),
          ],
        );
      },
    );
  }
}

class _StatusBanner extends StatelessWidget {
  const _StatusBanner({required this.controller, required this.color});

  final FaceScanController controller;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final c = controller;
    final recognized = c.status == ScanStatus.recognized;
    return Material(
      color: Colors.black.withValues(alpha: 0.65),
      borderRadius: BorderRadius.circular(14),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Icon(statusIcon(c.status), color: color, size: recognized ? 34 : 26),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        c.message,
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: recognized ? 22 : 17,
                          fontWeight: recognized ? FontWeight.bold : FontWeight.w500,
                        ),
                      ),
                      if (recognized && c.match != null)
                        Text(
                          'Live ✓  •  match ${(c.match!.similarity * 100).toStringAsFixed(0)}%',
                          style: TextStyle(color: color, fontSize: 13),
                        ),
                    ],
                  ),
                ),
              ],
            ),
            if (c.progress > 0 && c.progress < 1) ...[
              const SizedBox(height: 10),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: c.progress,
                  minHeight: 6,
                  color: color,
                  backgroundColor: Colors.white12,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _Stats extends StatelessWidget {
  const _Stats({required this.controller});

  final FaceScanController controller;

  @override
  Widget build(BuildContext context) {
    final c = controller;
    final spoof = c.lastSpoofScore;
    final sim = c.bestSimilarity;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(6)),
      child: Text(
        '${c.fps.toStringAsFixed(0)} fps • detect ${c.detectMs}ms • embed ${c.embedMs}ms • '
        'spoof ${c.spoofMs}ms\nfaces ${c.faceCount} • spoof score ${spoof?.toStringAsFixed(3) ?? '-'}'
        ' • sharp ${c.lastSharpness ?? '-'}'
        ' • best sim ${sim?.toStringAsFixed(2) ?? '-'} • ${c.phase.name}',
        style: const TextStyle(color: Colors.white70, fontSize: 11, fontFamily: 'monospace'),
      ),
    );
  }
}

class FaceOverlayPainter extends CustomPainter {
  FaceOverlayPainter({
    required this.box,
    required this.imageSize,
    required this.mirror,
    required this.color,
    this.label,
  });

  final Rect? box;
  final Size imageSize;
  final bool mirror;
  final Color color;
  final String? label;

  @override
  void paint(Canvas canvas, Size size) {
    final b = box;
    if (b == null) return;
    // Analysis frames and preview share the optical centre but may differ in
    // aspect ratio, so map with a centred "cover" transform.
    final scale = math.max(size.width / imageSize.width, size.height / imageSize.height);
    final dx = (size.width - imageSize.width * scale) / 2;
    final dy = (size.height - imageSize.height * scale) / 2;
    // The front-camera preview is mirrored; ML Kit coordinates are not.
    final left = mirror ? imageSize.width - b.right : b.left;
    final rect = Rect.fromLTWH(dx + left * scale, dy + b.top * scale, b.width * scale, b.height * scale);
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4;
    canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(18)), paint);

    final text = label;
    if (text != null && text.isNotEmpty) {
      final tp = TextPainter(
        text: TextSpan(
          text: ' $text ',
          style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold),
        ),
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: size.width);
      final bg = Rect.fromLTWH(rect.left, rect.bottom + 6, tp.width, tp.height);
      canvas.drawRRect(RRect.fromRectAndRadius(bg, const Radius.circular(6)), Paint()..color = color);
      tp.paint(canvas, bg.topLeft);
    }
  }

  @override
  bool shouldRepaint(FaceOverlayPainter old) =>
      old.box != box || old.color != color || old.label != label || old.mirror != mirror || old.imageSize != imageSize;
}
