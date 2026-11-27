import 'dart:typed_data';

/// Encodes an RGB888 buffer as a 24-bit BMP (displayable with Image.memory),
/// so thumbnails need no extra image-codec dependency.
Uint8List encodeBmp(Uint8List rgb, int width, int height) {
  final rowSize = ((width * 3 + 3) ~/ 4) * 4;
  final pixelBytes = rowSize * height;
  const headerSize = 54;
  final out = Uint8List(headerSize + pixelBytes);
  final bd = ByteData.sublistView(out);
  out[0] = 0x42; // 'B'
  out[1] = 0x4D; // 'M'
  bd.setUint32(2, out.length, Endian.little);
  bd.setUint32(10, headerSize, Endian.little);
  bd.setUint32(14, 40, Endian.little); // BITMAPINFOHEADER
  bd.setInt32(18, width, Endian.little);
  bd.setInt32(22, height, Endian.little); // positive = bottom-up rows
  bd.setUint16(26, 1, Endian.little); // planes
  bd.setUint16(28, 24, Endian.little); // bpp
  bd.setUint32(34, pixelBytes, Endian.little);
  for (int y = 0; y < height; y++) {
    final dstRow = headerSize + (height - 1 - y) * rowSize;
    for (int x = 0; x < width; x++) {
      final s = (y * width + x) * 3;
      final d = dstRow + x * 3;
      out[d] = rgb[s + 2];
      out[d + 1] = rgb[s + 1];
      out[d + 2] = rgb[s];
    }
  }
  return out;
}
