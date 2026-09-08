# The mobile app

React Native 0.81 + Expo 54 (managed), TypeScript strict. Path alias `@/*` → `src/*`.

- **Screens:** `HomeScreen` is the only one registered in `App.tsx` — it carries both the language/mode picker and the live feed. Connection state is handled in-place, not by a separate loading screen.
- **`src/context/WebSocketContext.tsx`** owns the whole WS session: socket lifecycle, message parsing, and the hardcoded `RUNPOD_SERVER_URL` (`getServerUrl()` swaps in `<host>/asr` when `EXPO_PUBLIC_USE_LOCAL_ASR=1` on web). Update it there when changing environments.
- **Audio:** `react-native-live-audio-stream` → binary PCM frames → server → JSON `final` messages back. `useAudioRecording.web.ts` / `tts.web.ts` are web variants.
- **Languages** (`src/constants/languages.ts`): ko, ja, zh, es, en. Conversation modes `mode-1` (speaker), `mode-2` (one earphone), `mode-3` (both) — unrelated to the eval mode2/3/4 in `evaluation/`.
- **Colors:** `HomeScreen` defines `LANG_COLORS` inline; there is no shared theme module.

