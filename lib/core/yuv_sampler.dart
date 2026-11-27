import 'dart:math' as math;
import 'dart:typed_data';

/// A camera frame in NV21 layout (Y plane followed by interleaved V/U).
///
/// [width]/[height] are the raw sensor dimensions. [rotation] (0/90/180/270)
/// is the clockwise rotation that makes the image upright, i.e. the same
/// value given to ML Kit, so ML Kit bounding boxes live in "upright" space.
class Nv21Frame {
  Nv21Frame({
    required this.bytes,
    required this.width,
    required this.height,
    required this.rowStride,
    required this.rotation,
  }) : assert(rotation % 90 == 0);

  final Uint8List bytes;
  final int width;
  final int height;
  final int rowStride;
  final int rotation;

  int get uprightWidth => rotation % 180 == 0 ? width : height;
  int get uprightHeight => rotation % 180 == 0 ? height : width;
  int get uvOffset => rowStride * height;
}

/// An (optionally rotated) square crop in upright image coordinates.
class CropSpec {
  const CropSpec({
    required this.cx,
    required this.cy,
    required this.side,
    this.angle = 0,
  });

  final double cx;
  final double cy;
  final double side;

  /// Radians. The crop's x axis points along (cos(angle), sin(angle)) in the
  /// upright image (y pointing down), so passing the eye-line angle produces
  /// a roll-aligned face.
  final double angle;
}

/// Samples [crop] from [frame] into a [size]x[size] RGB888 buffer, using
/// bilinear interpolation on luma. Only the output pixels are computed, so the
/// full frame is never converted.
Uint8List sampleRgb(Nv21Frame frame, CropSpec crop, int size) {
  final out = Uint8List(size * size * 3);
  final w = frame.width;
  final h = frame.height;
  final stride = frame.rowStride;
  final uvOff = frame.uvOffset;
  final src = frame.bytes;
  // The last chroma row may be unpadded, so only require `w` bytes for it.
  final hasUv = src.length >= uvOff + stride * (h ~/ 2 - 1) + w;
  final rot = frame.rotation % 360;

  final cosA = math.cos(crop.angle);
  final sinA = math.sin(crop.angle);
  final step = crop.side / size;
  final half = crop.side / 2;

  int o = 0;
  for (int v = 0; v < size; v++) {
    final dy = (v + 0.5) * step - half;
    for (int u = 0; u < size; u++) {
      final dx = (u + 0.5) * step - half;
      final ux = crop.cx + dx * cosA - dy * sinA;
      final uy = crop.cy + dx * sinA + dy * cosA;

      // Upright (continuous) -> raw (continuous).
      double rx, ry;
      switch (rot) {
        case 90:
          rx = uy;
          ry = h - ux;
          break;
        case 180:
          rx = w - ux;
          ry = h - uy;
          break;
        case 270:
          rx = w - uy;
          ry = ux;
          break;
        default:
          rx = ux;
          ry = uy;
      }

      // Pixel centres are at integer + 0.5.
      double fx = rx - 0.5;
      double fy = ry - 0.5;
      if (fx < 0) fx = 0;
      if (fy < 0) fy = 0;
      if (fx > w - 1) fx = (w - 1).toDouble();
      if (fy > h - 1) fy = (h - 1).toDouble();
      final x0 = fx.floor();
      final y0 = fy.floor();
      final x1 = x0 + 1 < w ? x0 + 1 : x0;
      final y1 = y0 + 1 < h ? y0 + 1 : y0;
      final ax = fx - x0;
      final ay = fy - y0;
      final r0 = y0 * stride;
      final r1 = y1 * stride;
      final top = src[r0 + x0] + (src[r0 + x1] - src[r0 + x0]) * ax;
      final bot = src[r1 + x0] + (src[r1 + x1] - src[r1 + x0]) * ax;
      final yy = top + (bot - top) * ay;

      double r, g, b;
      if (hasUv) {
        int xi = (fx + 0.5).toInt();
        int yi = (fy + 0.5).toInt();
        if (xi > w - 1) xi = w - 1;
        if (yi > h - 1) yi = h - 1;
        final uvIdx = uvOff + (yi >> 1) * stride + (xi & ~1);
        final vv = src[uvIdx] - 128.0;
        final uu = src[uvIdx + 1] - 128.0;
        r = yy + 1.402 * vv;
        g = yy - 0.344136 * uu - 0.714136 * vv;
        b = yy + 1.772 * uu;
      } else {
        r = g = b = yy;
      }
      out[o++] = r < 0 ? 0 : (r > 255 ? 255 : r.round());
      out[o++] = g < 0 ? 0 : (g > 255 ? 255 : g.round());
      out[o++] = b < 0 ? 0 : (b > 255 ? 255 : b.round());
    }
  }
  return out;
}

/// Converts a 3-plane YUV_420_888 image into a compact NV21 buffer.
Uint8List yuv420ToNv21({
  required int width,
  required int height,
  required Uint8List y,
  required int yRowStride,
  required Uint8List u,
  required Uint8List v,
  required int uvRowStride,
  required int uvPixelStride,
}) {
  final out = Uint8List(width * height + 2 * (width ~/ 2) * (height ~/ 2));
  int o = 0;
  for (int r = 0; r < height; r++) {
    out.setRange(o, o + width, y, r * yRowStride);
    o += width;
  }
  for (int r = 0; r < height ~/ 2; r++) {
    for (int c = 0; c < width ~/ 2; c++) {
      final i = r * uvRowStride + c * uvPixelStride;
      out[o++] = v[i];
      out[o++] = u[i];
    }
  }
  return out;
}

/// Counts strong Laplacian responses on a greyscale version of an RGB image.
/// Low counts mean a blurry / low-detail crop. Matches the reference
/// FaceAntiSpoofing implementation (kernel [[0,1,0],[1,-4,1],[0,1,0]], >50).
int laplacianScore(Uint8List rgb, int size, {int threshold = 50}) {
  final grey = Int32List(size * size);
  for (int i = 0, p = 0; i < grey.length; i++, p += 3) {
    grey[i] = (rgb[p] * 0.3 + rgb[p + 1] * 0.59 + rgb[p + 2] * 0.11).toInt();
  }
  int score = 0;
  for (int y = 1; y < size - 1; y++) {
    for (int x = 1; x < size - 1; x++) {
      final c = y * size + x;
      final v = grey[c - size] + grey[c + size] + grey[c - 1] + grey[c + 1] - 4 * grey[c];
      if (v > threshold) score++;
    }
  }
  return score;
}
