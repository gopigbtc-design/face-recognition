import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

import '../core/face_index.dart';
import '../core/settings.dart';

class Person {
  const Person({required this.id, required this.name, required this.createdAt});
  final int id;
  final String name;
  final int createdAt;
}

/// SQLite persistence + the in-memory [FaceIndex] used for matching.
class FaceRepository {
  late Database _db;
  final FaceIndex index = FaceIndex();
  final Map<int, Person> _people = {};

  List<Person> get people =>
      _people.values.toList()..sort((a, b) => a.name.toLowerCase().compareTo(b.name.toLowerCase()));
  int get count => _people.length;
  bool get isFull => _people.length >= kMaxFaces;

  Future<void> open() async {
    final dir = await getDatabasesPath();
    _db = await openDatabase(
      p.join(dir, 'face_guard.db'),
      version: 1,
      onCreate: (db, _) async {
        await db.execute('CREATE TABLE persons('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, '
            'name TEXT NOT NULL, '
            'created_at INTEGER NOT NULL, '
            'embedding BLOB NOT NULL, '
            'thumb BLOB)');
        await db.execute('CREATE TABLE settings(k TEXT PRIMARY KEY, v TEXT)');
      },
    );
    final rows = await _db.query('persons', columns: ['id', 'name', 'created_at', 'embedding']);
    index.clear();
    _people.clear();
    for (final r in rows) {
      final person = Person(
        id: r['id'] as int,
        name: r['name'] as String,
        createdAt: r['created_at'] as int,
      );
      final emb = _blobToEmbedding(r['embedding'] as Uint8List);
      if (emb.length != kEmbeddingSize || index.isFull) continue;
      _people[person.id] = person;
      index.add(person.id, person.name, emb);
    }
  }

  Future<Person> add(String name, Float32List embedding, Uint8List? thumbBmp) async {
    if (isFull) throw StateError('Face limit of $kMaxFaces reached');
    final now = DateTime.now().millisecondsSinceEpoch;
    final id = await _db.insert('persons', {
      'name': name,
      'created_at': now,
      'embedding': Uint8List.fromList(embedding.buffer.asUint8List(embedding.offsetInBytes, embedding.lengthInBytes)),
      'thumb': thumbBmp,
    });
    final person = Person(id: id, name: name, createdAt: now);
    _people[id] = person;
    index.add(id, name, embedding);
    return person;
  }

  Future<void> delete(int id) async {
    await _db.delete('persons', where: 'id = ?', whereArgs: [id]);
    _people.remove(id);
    index.remove(id);
  }

  Future<void> rename(int id, String name) async {
    await _db.update('persons', {'name': name}, where: 'id = ?', whereArgs: [id]);
    final old = _people[id];
    if (old != null) _people[id] = Person(id: id, name: name, createdAt: old.createdAt);
    index.rename(id, name);
  }

  final Map<int, Uint8List?> _thumbCache = {};

  Future<Uint8List?> thumb(int id) async {
    if (_thumbCache.containsKey(id)) return _thumbCache[id];
    final rows = await _db.query('persons', columns: ['thumb'], where: 'id = ?', whereArgs: [id]);
    final t = rows.isEmpty ? null : rows.first['thumb'] as Uint8List?;
    if (_thumbCache.length > 200) _thumbCache.clear();
    _thumbCache[id] = t;
    return t;
  }

  Future<AppSettings> loadSettings() async {
    final rows = await _db.query('settings');
    return AppSettings.fromMap({for (final r in rows) r['k'] as String: r['v'] as String});
  }

  Future<void> saveSettings(AppSettings s) async {
    final batch = _db.batch();
    s.toMap().forEach((k, v) {
      batch.insert('settings', {'k': k, 'v': v}, conflictAlgorithm: ConflictAlgorithm.replace);
    });
    await batch.commit(noResult: true);
  }

  static Float32List _blobToEmbedding(Uint8List blob) =>
      Uint8List.fromList(blob).buffer.asFloat32List();
}
