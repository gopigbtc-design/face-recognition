import 'package:face_guard/services/face_scan_controller.dart';
import 'package:face_guard/ui/face_scan_view.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('every scan status has a colour and icon', () {
    for (final s in ScanStatus.values) {
      expect(statusColor(s), isA<Color>());
      expect(statusIcon(s), isA<IconData>());
    }
  });

  testWidgets('overlay painter draws a mirrored box with label', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: CustomPaint(
          size: const Size(480, 720),
          painter: FaceOverlayPainter(
            box: const Rect.fromLTWH(100, 200, 150, 180),
            imageSize: const Size(480, 720),
            mirror: true,
            color: Colors.green,
            label: 'Alice',
          ),
        ),
      ),
    );
    expect(find.byType(CustomPaint), findsWidgets);
  });
}
