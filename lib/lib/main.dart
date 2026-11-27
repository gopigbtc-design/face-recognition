import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'core/face_index.dart';
import 'services/app_services.dart';
import 'ui/manage_screens.dart';
import 'ui/scan_screens.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Portrait only: keeps camera rotation handling simple and deterministic.
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const FaceGuardApp());
}

class FaceGuardApp extends StatelessWidget {
  const FaceGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Guard',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1E88E5), brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: const BootScreen(),
    );
  }
}

/// Opens the database and loads both ML models once, before anything else.
class BootScreen extends StatefulWidget {
  const BootScreen({super.key});

  @override
  State<BootScreen> createState() => _BootScreenState();
}

class _BootScreenState extends State<BootScreen> {
  late Future<void> _boot = AppServices.init();

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<void>(
      future: _boot,
      builder: (context, snap) {
        if (snap.connectionState != ConnectionState.done) {
          return const Scaffold(
            body: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [CircularProgressIndicator(), SizedBox(height: 16), Text('Loading face models…')],
              ),
            ),
          );
        }
        if (snap.hasError) {
          return Scaffold(
            body: Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text('Startup failed:\n${snap.error}', textAlign: TextAlign.center),
                    const SizedBox(height: 16),
                    FilledButton(
                      onPressed: () => setState(() => _boot = AppServices.init()),
                      child: const Text('Retry'),
                    ),
                  ],
                ),
              ),
            ),
          );
        }
        return const HomeScreen();
      },
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Future<void> _open(Widget page) async {
    await Navigator.push(context, MaterialPageRoute(builder: (_) => page));
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final count = AppServices.repo.count;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Guard'),
        actions: [
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.settings),
            onPressed: () => _open(const SettingsScreen()),
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Row(
                    children: [
                      const Icon(Icons.groups, size: 40),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('$count / $kMaxFaces', style: Theme.of(context).textTheme.headlineMedium),
                            const Text('faces registered on this device'),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),
              _BigButton(
                icon: Icons.face_retouching_natural,
                label: 'Live Recognition',
                subtitle: 'Liveness check + identify',
                onTap: () => _open(const RecognitionScreen()),
              ),
              _BigButton(
                icon: Icons.person_add_alt_1,
                label: 'Register New Face',
                subtitle: 'Requires a live-face check',
                onTap: () async {
                  await startRegistration(context);
                  if (mounted) setState(() {});
                },
              ),
              _BigButton(
                icon: Icons.manage_accounts,
                label: 'Manage Faces',
                subtitle: 'Search, rename, delete',
                onTap: () => _open(const FacesScreen()),
              ),
              const Spacer(),
              const Text(
                'All processing happens on this device. No images leave the phone.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white54, fontSize: 12),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _BigButton extends StatelessWidget {
  const _BigButton({required this.icon, required this.label, required this.subtitle, required this.onTap});

  final IconData icon;
  final String label, subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Card(
        clipBehavior: Clip.antiAlias,
        child: ListTile(
          contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
          leading: Icon(icon, size: 36),
          title: Text(label, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
          subtitle: Text(subtitle),
          trailing: const Icon(Icons.chevron_right),
          onTap: onTap,
        ),
      ),
    );
  }
}
