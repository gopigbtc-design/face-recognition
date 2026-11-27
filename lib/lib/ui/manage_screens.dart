import 'package:flutter/material.dart';

import '../core/face_index.dart';
import '../core/settings.dart';
import '../services/app_services.dart';
import '../services/face_repository.dart';
import 'scan_screens.dart';

// ---------------------------------------------------------------------------
// Registered faces
// ---------------------------------------------------------------------------

class FacesScreen extends StatefulWidget {
  const FacesScreen({super.key});

  @override
  State<FacesScreen> createState() => _FacesScreenState();
}

class _FacesScreenState extends State<FacesScreen> {
  String _query = '';

  List<Person> get _filtered {
    final q = _query.trim().toLowerCase();
    final all = AppServices.repo.people;
    return q.isEmpty ? all : all.where((p) => p.name.toLowerCase().contains(q)).toList();
  }

  Future<void> _rename(Person p) async {
    final ctrl = TextEditingController(text: p.name);
    final name = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Rename'),
        content: TextField(controller: ctrl, autofocus: true, maxLength: 40),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, ctrl.text.trim()), child: const Text('Save')),
        ],
      ),
    );
    ctrl.dispose();
    if (name == null || name.isEmpty) return;
    await AppServices.repo.rename(p.id, name);
    if (mounted) setState(() {});
  }

  Future<void> _delete(Person p) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Delete "${p.name}"?'),
        content: const Text('Their face data will be removed from this device.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: Colors.red),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (ok != true) return;
    await AppServices.repo.delete(p.id);
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final list = _filtered;
    return Scaffold(
      appBar: AppBar(title: Text('Registered faces (${AppServices.repo.count}/$kMaxFaces)')),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () async {
          await startRegistration(context);
          if (mounted) setState(() {});
        },
        icon: const Icon(Icons.person_add),
        label: const Text('Register'),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 12, 4),
            child: TextField(
              decoration: const InputDecoration(
                prefixIcon: Icon(Icons.search),
                hintText: 'Search by name',
                border: OutlineInputBorder(),
                isDense: true,
              ),
              onChanged: (v) => setState(() => _query = v),
            ),
          ),
          Expanded(
            child: list.isEmpty
                ? const Center(child: Text('No faces registered'))
                : ListView.builder(
                    itemCount: list.length,
                    itemBuilder: (context, i) {
                      final p = list[i];
                      return ListTile(
                        key: ValueKey(p.id),
                        leading: _Thumb(id: p.id),
                        title: Text(p.name),
                        subtitle: Text(
                          'Registered ${DateTime.fromMillisecondsSinceEpoch(p.createdAt).toLocal().toString().substring(0, 16)}',
                        ),
                        trailing: PopupMenuButton<String>(
                          onSelected: (v) => v == 'rename' ? _rename(p) : _delete(p),
                          itemBuilder: (_) => const [
                            PopupMenuItem(value: 'rename', child: Text('Rename')),
                            PopupMenuItem(value: 'delete', child: Text('Delete')),
                          ],
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

class _Thumb extends StatelessWidget {
  const _Thumb({required this.id});
  final int id;

  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: AppServices.repo.thumb(id),
      builder: (context, snap) {
        final bytes = snap.data;
        return CircleAvatar(
          radius: 24,
          backgroundImage: bytes != null ? MemoryImage(bytes) : null,
          child: bytes == null ? const Icon(Icons.person) : null,
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ValueListenableBuilder<AppSettings>(
        valueListenable: AppServices.settings,
        builder: (context, s, _) {
          void save(AppSettings n) => AppServices.updateSettings(n);
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              _SliderTile(
                title: 'Match threshold',
                help: 'Higher = stricter identity match (fewer false accepts).',
                value: s.matchThreshold,
                min: 0.40,
                max: 0.85,
                divisions: 45,
                format: (v) => v.toStringAsFixed(2),
                onChanged: (v) => save(s.copyWith(matchThreshold: v)),
              ),
              _SliderTile(
                title: 'Anti-spoof threshold',
                help: 'Lower = stricter (rejects more photos/screens, may reject some real faces in bad light).',
                value: s.spoofThreshold,
                min: 0.05,
                max: 0.50,
                divisions: 45,
                format: (v) => v.toStringAsFixed(2),
                onChanged: (v) => save(s.copyWith(spoofThreshold: v)),
              ),
              _SliderTile(
                title: 'Liveness challenges – recognition',
                help: '0 = passive (just look at the camera, fastest). '
                    '1 = blink, 2–3 = blink + head movements (harder to fool with videos).',
                value: s.recognizeChallenges.toDouble(),
                min: 0,
                max: 3,
                divisions: 3,
                format: (v) => v.toStringAsFixed(0),
                onChanged: (v) => save(s.copyWith(recognizeChallenges: v.round())),
              ),
              _SliderTile(
                title: 'Liveness challenges – registration',
                help: '0 = passive (just look at the camera). Higher = extra checks before enrolling.',
                value: s.registerChallenges.toDouble(),
                min: 0,
                max: 3,
                divisions: 3,
                format: (v) => v.toStringAsFixed(0),
                onChanged: (v) => save(s.copyWith(registerChallenges: v.round())),
              ),
              _SliderTile(
                title: 'Minimum face size',
                help: 'Face width as a fraction of the frame. Smaller allows standing further away.',
                value: s.minFaceRatio,
                min: 0.15,
                max: 0.50,
                divisions: 35,
                format: (v) => '${(v * 100).toStringAsFixed(0)}%',
                onChanged: (v) => save(s.copyWith(minFaceRatio: v)),
              ),
              _SliderTile(
                title: 'Blur gate (sharpness)',
                help: 'Anti-spoof frames below this edge count are skipped as blurry. '
                    'Watch "sharp" in the stats line; set this below what a live face shows.',
                value: s.minSharpness.toDouble(),
                min: 0,
                max: 1000,
                divisions: 50,
                format: (v) => v.toStringAsFixed(0),
                onChanged: (v) => save(s.copyWith(minSharpness: v.round())),
              ),
              SwitchListTile(
                title: const Text('Ask for a blink only when unsure'),
                subtitle: const Text('In passive mode, a borderline anti-spoof score triggers one quick blink.'),
                value: s.adaptiveBlink,
                onChanged: (v) => save(s.copyWith(adaptiveBlink: v)),
              ),
              SwitchListTile(
                title: const Text('Passive anti-spoof model'),
                subtitle: const Text('Keep ON. Turning off leaves only the active challenges.'),
                value: s.antiSpoofEnabled,
                onChanged: (v) => save(s.copyWith(antiSpoofEnabled: v)),
              ),
              SwitchListTile(
                title: const Text('Show performance stats'),
                value: s.showStats,
                onChanged: (v) => save(s.copyWith(showStats: v)),
              ),
              const SizedBox(height: 8),
              TextButton(
                onPressed: () => save(const AppSettings()),
                child: const Text('Reset to defaults'),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _SliderTile extends StatelessWidget {
  const _SliderTile({
    required this.title,
    required this.help,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.format,
    required this.onChanged,
  });

  final String title, help;
  final double value, min, max;
  final int divisions;
  final String Function(double) format;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(child: Text(title, style: Theme.of(context).textTheme.titleMedium)),
              Text(format(value), style: const TextStyle(fontWeight: FontWeight.bold)),
            ],
          ),
          Text(help, style: Theme.of(context).textTheme.bodySmall),
          Slider(
            value: value.clamp(min, max).toDouble(),
            min: min,
            max: max,
            divisions: divisions,
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }
}
