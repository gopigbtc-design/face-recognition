import 'dart:math' as math;
import 'dart:typed_data';

/// MobileFaceNet output size.
const int kEmbeddingSize = 192;

/// Maximum number of registered people.
const int kMaxFaces = 1000;

/// Returns a new L2-normalised copy of [v].
Float32List l2Normalize(Float32List v) {
  double sum = 0;
  for (int i = 0; i < v.length; i++) {
    sum += v[i] * v[i];
  }
  final norm = math.sqrt(sum);
  final out = Float32List(v.length);
  if (norm < 1e-10) return out;
  for (int i = 0; i < v.length; i++) {
    out[i] = v[i] / norm;
  }
  return out;
}

/// Dot product. For L2-normalised vectors this is the cosine similarity.
double dot(Float32List a, Float32List b) {
  double s = 0;
  for (int i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

/// Normalised mean of several (normalised) embeddings.
Float32List averageEmbeddings(List<Float32List> list) {
  if (list.isEmpty) throw ArgumentError('empty list');
  final acc = Float32List(list.first.length);
  for (final e in list) {
    for (int i = 0; i < acc.length; i++) {
      acc[i] += e[i];
    }
  }
  return l2Normalize(acc);
}

class FaceMatch {
  const FaceMatch(this.personId, this.name, this.similarity);
  final int personId;
  final String name;

  /// Cosine similarity in [-1, 1].
  final double similarity;

  @override
  String toString() => 'FaceMatch($personId, $name, ${similarity.toStringAsFixed(3)})';
}

/// In-memory brute-force index stored as one contiguous matrix.
///
/// 1000 x 192 floats = 768 KB; a full search is ~192k multiply-adds,
/// well under 1 ms on a phone, so no approximate index is needed.
class FaceIndex {
  FaceIndex({this.dim = kEmbeddingSize, this.capacity = kMaxFaces})
      : _data = Float32List(dim * capacity);

  final int dim;
  final int capacity;
  final Float32List _data;
  final List<int> _ids = <int>[];
  final List<String> _names = <String>[];

  int get length => _ids.length;
  bool get isFull => _ids.length >= capacity;
  List<int> get ids => List.unmodifiable(_ids);

  void clear() {
    _ids.clear();
    _names.clear();
  }

  void add(int id, String name, Float32List embedding) {
    if (embedding.length != dim) {
      throw ArgumentError('embedding must have $dim values');
    }
    if (_ids.contains(id)) {
      throw StateError('id $id already indexed');
    }
    if (isFull) throw StateError('Face limit of $capacity reached');
    final n = l2Normalize(embedding);
    _data.setRange(length * dim, length * dim + dim, n);
    _ids.add(id);
    _names.add(name);
  }

  bool remove(int id) {
    final idx = _ids.indexOf(id);
    if (idx < 0) return false;
    final last = length - 1;
    if (idx != last) {
      // Swap-remove keeps the matrix dense.
      _data.setRange(idx * dim, idx * dim + dim, _data, last * dim);
      _ids[idx] = _ids[last];
      _names[idx] = _names[last];
    }
    _ids.removeLast();
    _names.removeLast();
    return true;
  }

  void rename(int id, String name) {
    final idx = _ids.indexOf(id);
    if (idx >= 0) _names[idx] = name;
  }

  /// Best match for a normalised [query], or null if the index is empty.
  FaceMatch? best(Float32List query) {
    if (_ids.isEmpty) return null;
    int bestIdx = -1;
    double bestSim = -2;
    final d = _data;
    for (int r = 0; r < _ids.length; r++) {
      final base = r * dim;
      double s = 0;
      for (int i = 0; i < dim; i++) {
        s += d[base + i] * query[i];
      }
      if (s > bestSim) {
        bestSim = s;
        bestIdx = r;
      }
    }
    return FaceMatch(_ids[bestIdx], _names[bestIdx], bestSim);
  }
}
