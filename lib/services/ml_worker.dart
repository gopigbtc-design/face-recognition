import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../core/bmp.dart';
import '../core/face_index.dart';
import '../core/yuv_sampler.dart';

/// One unit of work for the background isolate.
class MlRequest {
  MlRequest({
    required this.id,
    required this.nv21,
    required this.width,
    required this.height,
    required this.rowStride,
    required this.rotation,
    required this.boxCrop,
    this.faceCrop,
    required this.embed,
    required this.spoof,
    required this.thumb,
  });

  final int id;
  final TransferableTypedData nv21;
  final int width, height, rowStride, rotation;
  /// Square crop around the detection box (anti-spoof model, thumbnail).
  final CropSpec boxCrop;

  /// 5-point aligned crop for MobileFaceNet (required when [embed]).
  final CropSpec? faceCrop;
  final bool embed, spoof, thumb;
}

class MlResult {
  MlResult({
    required this.id,
    this.embedding,
    this.spoofScore,
    this.laplacian,
    this.thumbBmp,
    this.embedMs = 0,
    this.spoofMs = 0,
    this.error,
  });

  final int id;
  final Float32List? embedding;
  final double? spoofScore;
  final int? laplacian;
  final Uint8List? thumbBmp;
  final int embedMs, spoofMs;
  final String? error;
}

class _InitMsg {
  _InitMsg(this.port, this.faceNet, this.antiSpoof);
  final SendPort port;
  final Uint8List faceNet;
  final Uint8List antiSpoof;
}

/// Owns both TFLite models inside a long-lived background isolate so that
/// cropping, preprocessing and inference never block the UI thread.
class MlWorker {
  SendPort? _send;
  ReceivePort? _recv;
  Isolate? _isolate;
  final Map<int, Completer<MlResult>> _pending = {};
  int _nextId = 0;

  bool get isReady => _send != null;

  Future<void> start() async {
    final faceNet = (await rootBundle.load('assets/models/MobileFaceNet.tflite')).buffer.asUint8List();
    final antiSpoof = (await rootBundle.load('assets/models/FaceAntiSpoofing.tflite')).buffer.asUint8List();
    final recv = ReceivePort();
    _recv = recv;
    final ready = Completer<void>();
    recv.listen((msg) {
      if (msg is SendPort) {
        _send = msg;
        ready.complete();
      } else if (msg is String && !ready.isCompleted) {
        ready.completeError(StateError(msg));
      } else if (msg is MlResult) {
        _pending.remove(msg.id)?.complete(msg);
      }
    });
    _isolate = await Isolate.spawn(_workerMain, _InitMsg(recv.sendPort, faceNet, antiSpoof),
        debugName: 'ml_worker');
    await ready.future;
  }

  Future<MlResult> process({
    required Nv21Frame frame,
    required CropSpec boxCrop,
    CropSpec? faceCrop,
    bool embed = false,
    bool spoof = false,
    bool thumb = false,
  }) {
    final id = _nextId++;
    final c = Completer<MlResult>();
    _pending[id] = c;
    _send!.send(MlRequest(
      id: id,
      nv21: TransferableTypedData.fromList([frame.bytes]),
      width: frame.width,
      height: frame.height,
      rowStride: frame.rowStride,
      rotation: frame.rotation,
      boxCrop: boxCrop,
      faceCrop: faceCrop,
      embed: embed && faceCrop != null,
      spoof: spoof,
      thumb: thumb,
    ));
    return c.future;
  }

  void dispose() {
    _isolate?.kill(priority: Isolate.immediate);
    _recv?.close();
    for (final c in _pending.values) {
      c.completeError(StateError('worker disposed'));
    }
    _pending.clear();
    _send = null;
  }
}

// ---------------------------------------------------------------------------
// Isolate side
// ---------------------------------------------------------------------------

const int _faceSize = 112;
const int _spoofSize = 256;
const int _thumbSize = 96;

void _workerMain(_InitMsg init) {
  final _Models models;
  try {
    models = _Models(init.faceNet, init.antiSpoof);
  } catch (e) {
    init.port.send('Failed to load models: $e');
    return;
  }
  final port = ReceivePort();
  init.port.send(port.sendPort);
  port.listen((msg) {
    if (msg is MlRequest) {
      MlResult res;
      try {
        res = models.run(msg);
      } catch (e) {
        res = MlResult(id: msg.id, error: '$e');
      }
      init.port.send(res);
    }
  });
}

class _Models {
  _Models(Uint8List faceNetBytes, Uint8List antiSpoofBytes) {
    final opts = InterpreterOptions()..threads = 4;
    faceNet = Interpreter.fromBuffer(faceNetBytes, options: opts);
    faceNet.allocateTensors();
    final opts2 = InterpreterOptions()..threads = 4;
    antiSpoof = Interpreter.fromBuffer(antiSpoofBytes, options: opts2);
    antiSpoof.allocateTensors();

    final inShape = faceNet.getInputTensors().first.shape; // [2,112,112,3]
    faceBatch = inShape.first;
    final outs = antiSpoof.getOutputTensors();
    clsIdx = 0;
    maskIdx = outs.length > 1 ? 1 : 0;
    for (int i = 0; i < outs.length; i++) {
      if (outs[i].name == 'Identity') clsIdx = i;
      if (outs[i].name == 'Identity_1') maskIdx = i;
    }
  }

  late final Interpreter faceNet;
  late final Interpreter antiSpoof;
  late final int faceBatch;
  late final int clsIdx;
  late final int maskIdx;

  MlResult run(MlRequest r) {
    final frame = Nv21Frame(
      bytes: r.nv21.materialize().asUint8List(),
      width: r.width,
      height: r.height,
      rowStride: r.rowStride,
      rotation: r.rotation,
    );
    final sw = Stopwatch()..start();

    Float32List? embedding;
    Uint8List? thumb;
    int embedMs = 0;
    if (r.embed) {
      final rgb = sampleRgb(frame, r.faceCrop!, _faceSize);
      embedding = _embed(rgb);
      embedMs = sw.elapsedMilliseconds;
    }
    if (r.thumb) {
      thumb = encodeBmp(sampleRgb(frame, r.boxCrop, _thumbSize), _thumbSize, _thumbSize);
    }

    double? spoof;
    int? lap;
    int spoofMs = 0;
    if (r.spoof) {
      sw.reset();
      final rgb = sampleRgb(frame, r.boxCrop, _spoofSize);
      lap = laplacianScore(rgb, _spoofSize);
      spoof = _spoofScore(rgb);
      spoofMs = sw.elapsedMilliseconds;
    }
    return MlResult(
      id: r.id,
      embedding: embedding,
      spoofScore: spoof,
      laplacian: lap,
      thumbBmp: thumb,
      embedMs: embedMs,
      spoofMs: spoofMs,
    );
  }

  /// MobileFaceNet expects (x - 127.5) / 128. The model has a fixed batch of
  /// 2, so slot 0 holds the face and slot 1 its mirror image; averaging both
  /// embeddings (flip test-time augmentation) improves robustness for free.
  Float32List _embed(Uint8List rgb) {
    const px = _faceSize * _faceSize;
    final input = Float32List(faceBatch * px * 3);
    for (int y = 0; y < _faceSize; y++) {
      for (int x = 0; x < _faceSize; x++) {
        final s = (y * _faceSize + x) * 3;
        final d0 = s;
        for (int c = 0; c < 3; c++) {
          input[d0 + c] = (rgb[s + c] - 127.5) / 128.0;
        }
        if (faceBatch > 1) {
          final d1 = px * 3 + (y * _faceSize + (_faceSize - 1 - x)) * 3;
          for (int c = 0; c < 3; c++) {
            input[d1 + c] = (rgb[s + c] - 127.5) / 128.0;
          }
        }
      }
    }
    faceNet.getInputTensors().first.data = input.buffer.asUint8List();
    faceNet.invoke();
    final out = _readFloats(faceNet.getOutputTensors().first.data);
    final dim = out.length ~/ faceBatch;
    final e0 = l2Normalize(Float32List.sublistView(out, 0, dim));
    if (faceBatch == 1) return e0;
    final e1 = l2Normalize(Float32List.sublistView(out, dim, 2 * dim));
    final sum = Float32List(dim);
    for (int i = 0; i < dim; i++) {
      sum[i] = e0[i] + e1[i];
    }
    return l2Normalize(sum);
  }

  /// FaceAntiSpoofing expects RGB / 255; score = Σ |cls_i| * leafMask_i.
  /// Higher = more likely an attack (reference threshold 0.2).
  double _spoofScore(Uint8List rgb) {
    final input = Float32List(rgb.length);
    for (int i = 0; i < rgb.length; i++) {
      input[i] = rgb[i] / 255.0;
    }
    antiSpoof.getInputTensors().first.data = input.buffer.asUint8List();
    antiSpoof.invoke();
    final outs = antiSpoof.getOutputTensors();
    final cls = _readFloats(outs[clsIdx].data);
    final mask = _readFloats(outs[maskIdx].data);
    double score = 0;
    for (int i = 0; i < cls.length && i < mask.length; i++) {
      score += cls[i].abs() * mask[i];
    }
    return score;
  }

  static Float32List _readFloats(Uint8List bytes) =>
      Uint8List.fromList(bytes).buffer.asFloat32List();
}
