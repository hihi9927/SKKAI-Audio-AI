# WhisperLiveKit WebSocket Server & App Usage Guide

This guide explains how to use WhisperLiveKit with the new WebSocket server and web client app (similar to SimulStreaming).

## Overview

WhisperLiveKit now includes a WebSocket-based server that works with a simple web client for real-time speech-to-text with translation. This is similar to the SimulStreaming architecture but powered by WhisperLiveKit.

## Installation

1. Install WhisperLiveKit with the translation dependency:

```bash
pip install -e .
pip install deep-translator
```

## Running the Server

### Option 1: Using the command-line script

```bash
# Basic usage (default: ws://0.0.0.0:8001)
wlk-ws

# With custom host and port
wlk-ws --host 0.0.0.0 --port 8001

# With language specification
wlk-ws --lan ko  # Force Korean
wlk-ws --lan en  # Force English
wlk-ws --lan auto  # Auto-detect (default)

# With specific backend
wlk-ws --backend faster-whisper
wlk-ws --backend mlx-whisper
```

### Option 2: Running the Python module directly

```bash
python -m whisperlivekit.websocket_server --host 0.0.0.0 --port 8001
```

## Using the Web Client

1. Open the client in your browser:

```bash
# Serve the app folder (you can use any HTTP server)
cd app
python -m http.server 8080
```

2. Open your browser and navigate to:
```
http://localhost:8080
```

3. The client will automatically:
   - Connect to the WebSocket server at `ws://localhost:8001`
   - Request microphone permission
   - Start streaming audio to the server
   - Display real-time transcription and translation

## Client Configuration

You can configure the client by modifying localStorage in your browser console:

```javascript
// Change server URL
localStorage.setItem('serverUrl', 'ws://your-server:8001');

// Change display mode
localStorage.setItem('displayMode', 'both');  // 'both', 'translateOnly', 'transcriptOnly'

// Change language hint
localStorage.setItem('languageHint', 'auto');  // 'auto', 'ko', 'en'
```

## Display Modes

The client supports three display modes:

1. **both** (default): Shows both original transcription and translation
2. **translateOnly**: Shows only the translated text
3. **transcriptOnly**: Shows only the original transcription (no translation)

## Keyboard Shortcuts

- **Space**: Toggle recording on/off

## WebSocket Protocol

The client and server communicate using JSON messages:

### Client → Server

**Start message:**
```json
{
  "type": "start",
  "lang": "auto",  // "auto", "ko", or "en"
  "displayMode": "both"  // "both", "translateOnly", or "transcriptOnly"
}
```

**Audio data:**
- Binary data: Int16 PCM audio at 16kHz, mono channel

**Finish message:**
```json
{
  "type": "finish"
}
```

**Stop message:**
```json
{
  "type": "stop"
}
```

### Server → Client

**Hello message:**
```json
{
  "type": "hello",
  "message": "Connected to WhisperLiveKit Streaming Server"
}
```

**Ready message:**
```json
{
  "type": "ready",
  "message": "Server ready to receive audio"
}
```

**Partial result:**
```json
{
  "type": "partial",
  "start": 1000,  // milliseconds
  "end": 2000,
  "original": "현재 인식 중인 텍스트",
  "last_translation": "Last completed translation",
  "language": "ko"
}
```

**Final result:**
```json
{
  "type": "final",
  "start": 1000,
  "end": 3000,
  "original": "완성된 문장",
  "polished": "Completed sentence",
  "language": "ko",
  "ko": "완성된 문장",
  "en": "Completed sentence"
}
```

## Translation Features

- **Automatic language detection**: Server detects Korean or English
- **Bidirectional translation**:
  - Korean → English
  - English → Korean
- **Sentence-based translation**: Waits for complete sentences before translating
- **Partial results**: Shows ongoing transcription while waiting for sentence completion

## Architecture Comparison

### SimulStreaming vs WhisperLiveKit WebSocket

Both share a similar architecture:

1. **WebSocket Server**
   - SimulStreaming: Custom WebSocket handler with SimulWhisper ASR
   - WhisperLiveKit: Custom WebSocket handler with WhisperLiveKit transcription engine

2. **Client App**
   - Both use the same web client architecture
   - HTML + JavaScript + CSS
   - AudioContext for audio capture
   - WebSocket for communication

3. **Translation**
   - Both use Google Translator (deep-translator)
   - Both support Korean ↔ English translation

## Troubleshooting

### Server won't start
- Check if port 8001 is already in use
- Try a different port: `wlk-ws --port 8002`

### Client can't connect
- Verify server is running: check console logs
- Check server URL in browser console: `console.log(localStorage.getItem('serverUrl'))`
- Update if needed: `localStorage.setItem('serverUrl', 'ws://localhost:8001')`

### No transcription appearing
- Check microphone permission in browser
- Open browser console (F12) to see detailed logs
- Verify audio is being sent: look for "📤 Audio chunk sent" messages

### Translation not working
- Install deep-translator: `pip install deep-translator`
- Check internet connection (Google Translate API requires internet)

## Advanced Usage

### Using with ngrok for remote access

```bash
# Start the server
wlk-ws --host 0.0.0.0 --port 8001

# In another terminal, start ngrok
ngrok http 8001

# Update client to use ngrok URL
localStorage.setItem('serverUrl', 'wss://your-subdomain.ngrok-free.app');
```

### Integration with Electron

The client code is designed to work with Electron apps (see `setupElectronIntegration()` in app.js). This allows:
- Click-through transparent window
- Draggable interface
- Settings window integration

## Credits

- Based on architecture from [SimulStreaming](https://github.com/backspacetg/simul_whisper)
- Powered by [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit)
- Translation via [deep-translator](https://github.com/nidhaloff/deep-translator)
