import 'package:flutter/material.dart';

import '../core/face_index.dart';
import '../services/app_services.dart';
import '../services/face_scan_controller.dart';
import 'face_scan_view.dart';

/// Mixin that owns a [FaceScanController] and releases the camera when the
/// app goes to the background.
mixin _ScanLifecycle<T extends StatefulWidget> on State<T>, WidgetsBindingObserver {
  late final FaceScanController controller;

  ScanMode get scanMode;

  void initScan() {
    controller = FaceScanController(mode: scanMode);
    WidgetsBinding.instance.addObserver(this);
    controller.init();
  }

  void disposeScan() {
    WidgetsBinding.instance.removeObserver(this);
    controller.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      controller.pause();
    } else if (state == AppLifecycleState.resumed) {
      controller.resume();
    }
  }

  Widget switchCameraButton() => AnimatedBuilder(
        animation: controller,
        builder: (_, __) => controller.canSwitchCamera
            ? IconButton(
                tooltip: 'Switch camera',
                icon: const Icon(Icons.cameraswitch),
                onPressed: controller.switchCamera,
              )
            : const SizedBox.shrink(),
      );
}

// ---------------------------------------------------------------------------
// Live recognition
// ---------------------------------------------------------------------------

class RecognitionScreen extends StatefulWidget {
  const RecognitionScreen({super.key});

  @override
  State<RecognitionScreen> createState() => _RecognitionScreenState();
}

class _RecognitionScreenState extends State<RecognitionScreen>
    with WidgetsBindingObserver, _ScanLifecycle<RecognitionScreen> {
  @override
  ScanMode get scanMode => ScanMode.recognize;

  @override
  void initState() {
    super.initState();
    initScan();
  }

  @override
  void dispose() {
    disposeScan();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Live Recognition'),
        actions: [switchCameraButton()],
      ),
      body: FaceScanView(controller: controller),
    );
  }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key, required this.name});

  final String name;

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen>
    with WidgetsBindingObserver, _ScanLifecycle<RegisterScreen> {
  bool _dialogOpen = false;
  bool _saving = false;

  @override
  ScanMode get scanMode => ScanMode.register;

  @override
  void initState() {
    super.initState();
    initScan();
    controller.addListener(_onChange);
  }

  @override
  void dispose() {
    controller.removeListener(_onChange);
    disposeScan();
    super.dispose();
  }

  void _onChange() {
    final cap = controller.capture;
    if (cap != null && !_dialogOpen && mounted) {
      _dialogOpen = true;
      WidgetsBinding.instance.addPostFrameCallback((_) => _confirm(cap));
    }
  }

  Future<void> _confirm(RegistrationCapture cap) async {
    final dup = cap.duplicate;
    final save = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        title: Text('Register "${widget.name}"?'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (cap.thumbBmp != null)
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.memory(cap.thumbBmp!, width: 140, height: 140, fit: BoxFit.cover, gaplessPlayback: true),
              ),
            const SizedBox(height: 12),
            const Text('Liveness verified ✓'),
            if (dup != null) ...[
              const SizedBox(height: 12),
              Text(
                'This face already matches "${dup.name}" '
                '(${(dup.similarity * 100).toStringAsFixed(0)}% similar). '
                'Registering again creates a duplicate.',
                style: const TextStyle(color: Colors.orange),
              ),
            ],
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Retake')),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(dup != null ? 'Register anyway' : 'Save'),
          ),
        ],
      ),
    );
    if (!mounted) return;
    if (save == true) {
      setState(() => _saving = true);
      try {
        await AppServices.repo.add(widget.name, cap.embedding, cap.thumbBmp);
        if (!mounted) return;
        Navigator.pop(context, true);
      } catch (e) {
        if (!mounted) return;
        setState(() => _saving = false);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Could not save: $e')));
        _dialogOpen = false;
        await controller.retake();
      }
    } else {
      _dialogOpen = false;
      await controller.retake();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text('Register: ${widget.name}'),
        actions: [switchCameraButton()],
      ),
      body: _saving ? const Center(child: CircularProgressIndicator()) : FaceScanView(controller: controller),
    );
  }
}

/// Asks for the person's name, then opens [RegisterScreen].
Future<bool> startRegistration(BuildContext context) async {
  if (AppServices.repo.isFull) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Limit of $kMaxFaces faces reached. Delete someone first.')),
    );
    return false;
  }
  final name = await showDialog<String>(context: context, builder: (_) => const _NameDialog());
  if (name == null || !context.mounted) return false;
  final ok = await Navigator.push<bool>(
    context,
    MaterialPageRoute(builder: (_) => RegisterScreen(name: name)),
  );
  if (ok == true && context.mounted) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('"$name" registered')));
  }
  return ok == true;
}

class _NameDialog extends StatefulWidget {
  const _NameDialog();

  @override
  State<_NameDialog> createState() => _NameDialogState();
}

class _NameDialogState extends State<_NameDialog> {
  final _form = GlobalKey<FormState>();
  final _ctrl = TextEditingController();

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  void _submit() {
    if (_form.currentState!.validate()) Navigator.pop(context, _ctrl.text.trim());
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('New person'),
      content: Form(
        key: _form,
        child: TextFormField(
          controller: _ctrl,
          autofocus: true,
          maxLength: 40,
          textCapitalization: TextCapitalization.words,
          decoration: const InputDecoration(labelText: 'Full name / ID'),
          validator: (v) => (v == null || v.trim().isEmpty) ? 'Enter a name' : null,
          onFieldSubmitted: (_) => _submit(),
        ),
      ),
      actions: [
        TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
        FilledButton(onPressed: _submit, child: const Text('Continue')),
      ],
    );
  }
}
