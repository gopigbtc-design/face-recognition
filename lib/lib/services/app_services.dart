import 'package:flutter/foundation.dart';

import '../core/settings.dart';
import 'face_repository.dart';
import 'ml_worker.dart';

/// Tiny service locator: one repository, one ML worker, live settings.
class AppServices {
  AppServices._();

  static final FaceRepository repo = FaceRepository();
  static final MlWorker worker = MlWorker();
  static final ValueNotifier<AppSettings> settings = ValueNotifier(const AppSettings());

  static Future<void> init() async {
    await repo.open();
    settings.value = await repo.loadSettings();
    await worker.start();
  }

  static Future<void> updateSettings(AppSettings s) async {
    settings.value = s;
    await repo.saveSettings(s);
  }
}
