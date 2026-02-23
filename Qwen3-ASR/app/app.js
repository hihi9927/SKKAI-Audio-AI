const DOM = {
  currTextOriginal: document.getElementById('currTextOriginal'),
  currTextTranslated: document.getElementById('currTextTranslated'),
  current: document.getElementById('current'),
  mainPanel: document.getElementById('mainPanel')
};

const state = {
  ws: null,
  mediaRecorder: null,
  mediaStream: null,
  audioContext: null,
  audioWorkletNode: null,
  audioBuffer: [],
  isRecording: false,
  isServerConnected: false,
  translationHistory: [],
  recordingEnabled: false,
  // Default to latest mobile app endpoint (overridable via settings/localStorage)
  SERVER_URL: 'wss://supervigorously-unforded-lavone.ngrok-free.dev',
  currentOriginal: '',
  currentTranslated: '',
  isConnecting: false,
  isBlackFullscreen: false,
  displayMode: 'both', // 'translateOnly', 'transcriptOnly', 'both'
  backgroundBrightness: undefined,
  autoAdjustOpacity: false,
  isFirstMessage: true,
  loadingInterval: null,
  lastLoggedText: '' // Track last logged text to avoid duplicates
};

// 그라데이션 바 색상 업데이트 (서버 연결 + 녹음 상태 모두 확인)
function updateGradientBar() {
  const ruleElement = document.querySelector('.rule');
  if (!ruleElement) return;

  // 서버 연결되어 있고 녹음 중일 때만 파란색, 그 외는 모두 빨간색
  const isFullyOperational = state.isServerConnected && state.isRecording;

  if (isFullyOperational) {
    ruleElement.classList.remove('paused');
    console.log('✅ 정상 작동: 파란 그라데이션');
  } else {
    ruleElement.classList.add('paused');
    console.log('⚠️ 비정상 상태: 빨간 그라데이션 (서버:', state.isServerConnected, '녹음:', state.isRecording, ')');
  }
}

function updateServerStatus(connected) {
  state.isServerConnected = connected;
  console.log('🔌 서버 상태:', connected ? '연결됨' : '연결 끊김');

  // 그라데이션 바 업데이트
  updateGradientBar();

  // 설정 창으로 상태 전달
  if (window.require) {
    try {
      const { ipcRenderer } = window.require('electron');
      if (ipcRenderer) {
        ipcRenderer.send('status-update', 'server', connected);
      }
    } catch(e) {}
  }
}

function updateRecordingStatus(recording) {
  state.isRecording = recording;
  console.log('🎤 녹음 상태:', recording ? '활성' : '비활성');

  // 그라데이션 바 업데이트
  updateGradientBar();

  // 설정 창으로 상태 전달
  if (window.require) {
    try {
      const { ipcRenderer } = window.require('electron');
      if (ipcRenderer) {
        ipcRenderer.send('status-update', 'recording', recording);
      }
    } catch(e) {}
  }
}

function loadOpacity() {
  const saved = localStorage.getItem('panelOpacity');
  updateOpacity(saved || 100);
}

function updateOpacity(value, autoAdjust = false) {
  const transparency = value / 100;
  let bgOpacity = 0.95 * (1 - transparency);

  // 자동 조정 모드일 때 배경 밝기에 따라 보정
  if (autoAdjust && state.backgroundBrightness !== undefined) {
    const brightness = state.backgroundBrightness;

    if (brightness < 80) {
      // 매우 어두운 배경: 투명하게
      bgOpacity = Math.max(0.3, bgOpacity * 0.5);
    } else if (brightness < 150) {
      // 중간 어두운 배경: 약간 투명하게
      bgOpacity = Math.max(0.5, bgOpacity * 0.8);
    } else if (brightness < 200) {
      // 밝은 배경: 불투명하게
      bgOpacity = Math.min(0.95, bgOpacity + 0.15);
    } else {
      // 매우 밝은 배경: 매우 불투명하게
      bgOpacity = Math.min(0.98, bgOpacity + 0.3);
    }

    console.log(`🎨 투명도 조정: 밝기=${brightness.toFixed(0)} -> 불투명도=${bgOpacity.toFixed(2)}`);
  }

  DOM.mainPanel.style.background = `rgba(20,20,20,${bgOpacity})`;
  localStorage.setItem('panelOpacity', value);
}

function loadTextSize() {
  const saved = localStorage.getItem('textSize');
  updateTextSize(saved || 100);
}

function updateTextSize(value) {
  console.log('📏 텍스트 크기 변경:', value);
  const scale = value / 100;
  const originalSize = 20; // 원본 크기
  const translatedOriginalSize = 16; // 번역 텍스트 원본 크기

  DOM.currTextOriginal.style.fontSize = `${originalSize * scale}px`;
  DOM.currTextTranslated.style.fontSize = `${translatedOriginalSize * scale}px`;

  // 350% 이상일 때 패널 너비 증가
  const wrapElement = document.querySelector('.wrap');
  if (wrapElement) {
    if (value >= 350) {
      const widthScale = 1.3; // 350% 이상일 때 1.3배 고정
      wrapElement.style.width = `min(${92 * widthScale}vw, ${1100 * widthScale}px)`;
      console.log('📐 패널 너비 확장:', `${widthScale}x`);
    } else {
      wrapElement.style.width = 'min(92vw, 1100px)';
    }
  }

  localStorage.setItem('textSize', value);
  console.log('✅ 텍스트 크기 적용 완료 - 원본:', `${originalSize * scale}px`, '번역:', `${translatedOriginalSize * scale}px`);
}

function loadTextColor() {
  const saved = localStorage.getItem('textColor');
  updateTextColor(saved || 'white');
}

function updateTextColor(color) {
  console.log('🎨 텍스트 색깔 변경:', color);
  if (color === 'white') {
    DOM.currTextOriginal.style.color = '#fff';
    DOM.currTextTranslated.style.color = 'rgba(255, 255, 255, 0.75)';
    DOM.currTextOriginal.style.textShadow = '0 2px 10px rgba(0, 0, 0, 0.8)';
    DOM.currTextTranslated.style.textShadow = '0 2px 10px rgba(0, 0, 0, 0.8)';
  } else {
    DOM.currTextOriginal.style.color = '#000';
    DOM.currTextTranslated.style.color = 'rgba(0, 0, 0, 0.75)';
    DOM.currTextOriginal.style.textShadow = 'none';
    DOM.currTextTranslated.style.textShadow = 'none';
  }
  localStorage.setItem('textColor', color);
  console.log('✅ 텍스트 색깔 적용 완료:', color);
}

function showLoadingAnimation() {
  let dotCount = 0;
  DOM.currTextOriginal.style.background = 'linear-gradient(135deg, #c9a6ff, #7ad7ff)';
  DOM.currTextOriginal.style.webkitBackgroundClip = 'text';
  DOM.currTextOriginal.style.webkitTextFillColor = 'transparent';
  DOM.currTextOriginal.style.backgroundClip = 'text';

  state.loadingInterval = setInterval(() => {
    dotCount = (dotCount + 1) % 4;
    DOM.currTextOriginal.textContent = '로딩 중' + '.'.repeat(dotCount);
    DOM.currTextTranslated.textContent = '';
  }, 500);
}

function hideLoadingAnimation() {
  if (state.loadingInterval) {
    clearInterval(state.loadingInterval);
    state.loadingInterval = null;
  }
  DOM.currTextOriginal.style.background = 'none';
  DOM.currTextOriginal.style.webkitBackgroundClip = 'initial';
  DOM.currTextOriginal.style.webkitTextFillColor = 'initial';
  DOM.currTextOriginal.style.backgroundClip = 'initial';

  // 텍스트 색깔 복원
  const savedColor = localStorage.getItem('textColor') || 'white';
  updateTextColor(savedColor);
}

function showResult(original, translated) {
  const orig = (original ?? '').trim();
  const trans = (translated ?? '').trim();

  if (!orig && !trans) return;

  // 첫 메시지일 때 로딩 애니메이션 종료
  if (state.isFirstMessage) {
    state.isFirstMessage = false;
    hideLoadingAnimation();
  }

  DOM.current.classList.add('typing');

  // 표시 모드에 따라 텍스트 표시
  if (state.displayMode === 'translateOnly') {
    // 번역만
    DOM.currTextOriginal.textContent = trans || orig;
    DOM.currTextTranslated.textContent = '';
  } else if (state.displayMode === 'transcriptOnly') {
    // 전사만
    DOM.currTextOriginal.textContent = orig;
    DOM.currTextTranslated.textContent = '';
  } else {
    // 번역+전사 (기본)
    DOM.currTextOriginal.textContent = orig;
    DOM.currTextTranslated.textContent = trans;
  }

  state.currentOriginal = orig;
  state.currentTranslated = trans;

  const logText = orig + (trans ? '\n' + trans : '');
  addToLog(logText);
}

function addToLog(text) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('ko-KR', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });

  state.translationHistory.unshift({ time: timeStr, text });
  if (state.translationHistory.length > 50) {
    state.translationHistory.pop();
  }

  // 설정 창으로 로그 전달
  if (window.require) {
    try {
      const { ipcRenderer } = window.require('electron');
      if (ipcRenderer) {
        ipcRenderer.send('status-update', 'translation', state.translationHistory);
      }
    } catch(e) {}
  }
}

function openSettings() {
  if (!window.require) return;

  try {
    const { ipcRenderer } = window.require('electron');
    ipcRenderer.send('open-settings-window');
  } catch(e) {
    console.log('설정 창 열기 실패', e);
  }
}

async function connectWebSocket() {
  if (state.isConnecting) {
    console.log('⚠️ 이미 연결 시도 중');
    return;
  }

  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    console.log('✅ 이미 연결되어 있음');
    return;
  }

  if (state.ws && state.ws.readyState === WebSocket.CONNECTING) {
    console.log('⚠️ 연결 중...');
    return;
  }

  state.isConnecting = true;

  try {
    console.log('🔌 WS 연결 시도:', state.SERVER_URL);
    state.ws = new WebSocket(state.SERVER_URL);
    state.ws.binaryType = 'arraybuffer';

    state.ws.onopen = () => {
      console.log('✅ WS 연결 성공');
      state.isConnecting = false;
      updateServerStatus(true);

      // displayMode에 따라 옵션 설정
      let translate = false;
      let polish = false;

      if (state.displayMode === 'translateOnly') {
        translate = true;
        polish = false;
      } else if (state.displayMode === 'transcriptOnly') {
        translate = false;
        polish = true;
      } else { // 'both'
        translate = true;
        polish = true;
      }

      // Get language hint from settings
      const languageHint = localStorage.getItem('languageHint') || 'auto';
      const targetLang = localStorage.getItem('targetLang') || '';

      const startMsg = {
        type: 'start',
        lang: languageHint,
        targetLang,
        polish: polish,
        translate: translate
      };
      console.log('📤 start 메시지 전송:', startMsg);
      state.ws.send(JSON.stringify(startMsg));
    };

    state.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const t = data.type || data.status;

        console.log('📥 서버 메시지:', t, data);

        if (t === 'hello') {
          // Don't show hello message as subtitle - it's just a connection confirmation
          console.log('👋', data.message);
        } else if (t === 'ready') {
          console.log('✅ 서버 준비 완료', data);
        } else if (t === 'partial') {
          // Partial result - real-time transcription (no translation yet)
          // {
          //   type: 'partial',
          //   original: string,  // Current incomplete transcription
          //   last_translation: string,  // Last completed translation (for display continuity)
          //   language: string
          // }

          const partialText = (data.original || '').trim();
          const lastTranslation = data.last_translation || '';

          console.log('⏳ Partial:', partialText.substring(0, 50) + '...');

          // Show partial transcription with last completed translation
          if (partialText) {
            showResult(partialText, lastTranslation);
          }
        } else if (t === 'final') {
          // Final result - complete sentence with translation
          // {
          //   type: 'final',
          //   start: string,
          //   end: string,
          //   original: string,  // Complete sentence
          //   translation: string,  // Full translation
          //   language: string
          // }

          const finalText = (data.original || '').trim();
          const translation = data.translation || '';

          console.log('✅ Final:', {
            text: finalText.substring(0, 50) + '...',
            translation: translation.substring(0, 30) + '...'
          });

          // Show complete sentence with translation (only once!)
          if (finalText) {
            showResult(finalText, translation);
          }
        } else if (t === 'active_transcription' || t === 'no_audio_detected') {
          // Legacy format support (for backwards compatibility)
          const lines = data.lines || [];
          const bufferText = data.buffer_transcription || '';
          const serverTranslation = data.translation || '';

          let displayText = '';

          if (lines.length > 0) {
            const lastLine = lines[lines.length - 1];
            displayText = (lastLine.text || '').trim();
          }

          if (bufferText.trim()) {
            if (displayText) {
              displayText = displayText + ' ' + bufferText.trim();
            } else {
              displayText = bufferText.trim();
            }
          }

          console.log('🔄 실시간 결과:', {
            lines: lines.length,
            text: displayText.substring(0, 50) + '...',
            translation: serverTranslation.substring(0, 30) + '...'
          });

          if (displayText) {
            showResult(displayText, serverTranslation);
          }
        } else if (t === 'error') {
          console.error('❗ 서버 오류:', data.error || data.message);
        }
      } catch (e) {
        console.error('❌ 메시지 파싱 오류:', e, event.data);
      }
    };

    state.ws.onerror = (error) => {
      console.error('❌ WS 오류:', error);
      state.isConnecting = false;
      updateServerStatus(false);
    };

    state.ws.onclose = (event) => {
      console.log('🔌 WS 연결 종료', event.code, event.reason);
      state.isConnecting = false;
      updateServerStatus(false);
      state.ws = null;
    };

  } catch (error) {
    console.error('❌ 연결 실패:', error);
    state.isConnecting = false;
    updateServerStatus(false);
  }
}

async function initAudioStream() {
  if (state.mediaStream) {
    console.log('✅ 오디오 스트림 재사용');
    return;
  }
  
  try {
    console.log('🎤 마이크 권한 요청 중...');
    state.mediaStream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        channelCount: 1,
        sampleRate: 48000,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      } 
    });
    console.log('✅ 마이크 접근 허용');
  } catch (error) {
    console.error('❌ 마이크 접근 거부:', error);
    alert('마이크 접근이 거부되었습니다.\n브라우저 설정에서 마이크 권한을 허용해주세요.');
    throw error;
  }
}

async function startRecording() {
  if (state.isRecording) {
    console.log('⚠️ 이미 녹음 중');
    return;
  }

  try {
    // WebSocket 연결 확인 및 재연결
    await connectWebSocket();

    // WebSocket이 열릴 때까지 대기
    let retries = 0;
    while ((!state.ws || state.ws.readyState !== WebSocket.OPEN) && retries < 10) {
      console.log(`⏳ WebSocket 연결 대기 중... (${retries + 1}/10)`);
      await new Promise(resolve => setTimeout(resolve, 300));
      retries++;
    }

    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      console.error('❌ WebSocket 연결 타임아웃');
      alert('서버 연결에 실패했습니다. 서버 URL을 확인해주세요.');
      return;
    }

    await initAudioStream();

    // AudioContext 생성
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 48000
    });

    const source = state.audioContext.createMediaStreamSource(state.mediaStream);

    // ScriptProcessorNode 사용 (AudioWorklet은 Electron에서 까다로울 수 있음)
    const bufferSize = 4096;
    const processor = state.audioContext.createScriptProcessor(bufferSize, 1, 1);

    // 500ms마다 전송 (16kHz * 0.5초 = 8000 샘플)
    const targetSamplesPerChunk = 8000;
    state.audioBuffer = [];

    processor.onaudioprocess = (e) => {
      if (!state.isRecording || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
        return;
      }

      const inputData = e.inputBuffer.getChannelData(0);

      // 48kHz -> 16kHz 다운샘플링
      const targetSampleRate = 16000;
      const sourceSampleRate = state.audioContext.sampleRate;
      const ratio = sourceSampleRate / targetSampleRate;
      const newLength = Math.floor(inputData.length / ratio);
      const downsampled = new Float32Array(newLength);

      for (let i = 0; i < newLength; i++) {
        const srcIndex = Math.floor(i * ratio);
        downsampled[i] = inputData[srcIndex];
      }

      // Float32 (-1.0 to 1.0) -> Int16 (-32768 to 32767)
      const pcmData = new Int16Array(downsampled.length);
      for (let i = 0; i < downsampled.length; i++) {
        const s = Math.max(-1, Math.min(1, downsampled[i]));
        pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      // 버퍼에 추가
      state.audioBuffer.push(pcmData);

      // 버퍼에 충분한 샘플이 모이면 전송
      const totalSamples = state.audioBuffer.reduce((sum, arr) => sum + arr.length, 0);
      if (totalSamples >= targetSamplesPerChunk) {
        // 모든 버퍼를 하나로 합치기
        const combinedLength = totalSamples;
        const combinedBuffer = new Int16Array(combinedLength);
        let offset = 0;

        for (const buf of state.audioBuffer) {
          combinedBuffer.set(buf, offset);
          offset += buf.length;
        }

        // 전송
        state.ws.send(combinedBuffer.buffer);
        console.log('📤 오디오 청크 전송:', combinedBuffer.length, 'samples (~' + (combinedBuffer.length / 16000).toFixed(2) + 's)');

        // 버퍼 초기화
        state.audioBuffer = [];
      }
    };

    source.connect(processor);
    processor.connect(state.audioContext.destination);

    state.audioWorkletNode = processor;
    state.isRecording = true;
    updateRecordingStatus(true);
    console.log('🎙️ 녹음 시작 (RAW PCM 16kHz)');

    // 첫 녹음 시작시 로딩 애니메이션 표시
    if (state.isFirstMessage) {
      showLoadingAnimation();
    }

  } catch(error){
    console.error('❌ 녹음 시작 에러:', error);
    state.isRecording = false;
    updateRecordingStatus(false);
  }
}

function stopRecording() {
  console.log('⏹️ 녹음 중지 요청');

  // Send finish command to server to flush remaining buffers
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    const finishMsg = { type: 'finish' };
    state.ws.send(JSON.stringify(finishMsg));
    console.log('📤 Finish 메시지 전송');
  }

  if (state.audioWorkletNode) {
    state.audioWorkletNode.disconnect();
    state.audioWorkletNode = null;
    console.log('✅ AudioProcessor 중지');
  }

  if (state.audioContext && state.audioContext.state !== 'closed') {
    state.audioContext.close();
    state.audioContext = null;
    console.log('✅ AudioContext 종료');
  }

  state.audioBuffer = [];
  state.isRecording = false;
  updateRecordingStatus(false);
}

function cleanupRecording() {
  console.log('🧹 리소스 정리 시작');

  if (state.audioWorkletNode) {
    state.audioWorkletNode.disconnect();
    state.audioWorkletNode = null;
  }

  if (state.audioContext && state.audioContext.state !== 'closed') {
    state.audioContext.close();
    state.audioContext = null;
  }

  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach(track => {
      track.stop();
      console.log('🛑 오디오 트랙 중지:', track.label);
    });
    state.mediaStream = null;
  }

  if (state.ws) {
    if (state.ws.readyState === WebSocket.OPEN) {
      try {
        state.ws.send(JSON.stringify({ type: 'stop' }));
      } catch (e) {
        console.error('⚠️ 종료 신호 전송 실패:', e);
      }
    }
    state.ws.close();
    state.ws = null;
  }

  state.mediaRecorder = null;
  state.isRecording = false;
  state.recordingEnabled = false;
  console.log('✅ 리소스 정리 완료');
}

function loadSettings() {
  const savedUrl = localStorage.getItem('serverUrl');
  if (savedUrl) {
    state.SERVER_URL = savedUrl;
    console.log('📂 서버 URL 불러옴:', savedUrl);
  }

  const savedDisplayMode = localStorage.getItem('displayMode');
  if (savedDisplayMode) {
    state.displayMode = savedDisplayMode;
    console.log('📂 표시 모드 불러옴:', savedDisplayMode);
  }
}

function setupEventListeners() {
  // Space 키를 토글 방식으로 변경
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.target.matches('input, select')) {
      e.preventDefault();

      // 이미 눌린 상태면 무시 (연속 입력 방지)
      if (e.repeat) return;

      console.log('⌨️ Space 키 누름 - 토글');

      if (state.isRecording) {
        // 녹음 중이면 중지
        console.log('⏸️ 녹음 중지');
        stopRecording();
      } else {
        // 녹음 중이 아니면 시작
        console.log('▶️ 녹음 시작');
        startRecording();
      }
    }
  });
}

function setupElectronIntegration() {
  if (!window.require) return;

  try {
    const { ipcRenderer } = window.require('electron');
    let isOverInteractive = false;
    let dragTimer = null;
    let mouseDownPos = null;
    let isDraggingMode = false;
    let isSettingsOpen = false; // 설정창 열림 상태 추적
    const LONG_PRESS_DURATION = 300;

    // 배경 밝기 감지 및 자동 투명도 조정
    async function detectBackgroundBrightness() {
      if (!state.autoAdjustOpacity) {
        return;
      }

      try {
        const { desktopCapturer, screen } = require('electron');

        // 현재 화면의 실제 크기 가져오기
        const primaryDisplay = screen.getPrimaryDisplay();
        const scaleFactor = primaryDisplay.scaleFactor || 1;
        const screenSize = primaryDisplay.size;

        console.log(`🖥️ 화면 정보: ${screenSize.width}x${screenSize.height}, scaleFactor: ${scaleFactor}`);

        // 화면을 실제 해상도로 캡처
        const sources = await desktopCapturer.getSources({
          types: ['screen'],
          thumbnailSize: {
            width: Math.floor(screenSize.width * scaleFactor),
            height: Math.floor(screenSize.height * scaleFactor)
          }
        });

        if (sources.length === 0) {
          console.warn('❌ 화면 캡처 소스가 없음');
          return;
        }

        const thumbnail = sources[0].thumbnail;
        const imgSize = thumbnail.getSize();
        console.log(`📸 캡처된 이미지 크기: ${imgSize.width}x${imgSize.height}`);

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        img.onload = async () => {
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);

          // 패널의 화면상 위치 가져오기 (윈도우 좌표)
          const rect = DOM.mainPanel.getBoundingClientRect();

          // 윈도우의 절대 위치 가져오기
          const windowBounds = await ipcRenderer.invoke('get-window-bounds');

          // 패널의 화면상 절대 위치 계산
          const panelScreenX = windowBounds.x + rect.left;
          const panelScreenY = windowBounds.y + rect.top;

          console.log(`📍 패널 화면 위치: x=${panelScreenX.toFixed(0)}, y=${panelScreenY.toFixed(0)}, w=${rect.width.toFixed(0)}, h=${rect.height.toFixed(0)}`);
          console.log(`📍 윈도우 위치: x=${windowBounds.x}, y=${windowBounds.y}`);

          // 이미지 좌표로 변환 (scaleFactor 고려)
          const imgX = Math.floor((panelScreenX / screenSize.width) * img.width);
          const imgY = Math.floor((panelScreenY / screenSize.height) * img.height);
          const imgW = Math.floor((rect.width / screenSize.width) * img.width);
          const imgH = Math.floor((rect.height / screenSize.height) * img.height);

          console.log(`🎯 이미지 샘플링: x=${imgX}, y=${imgY}, w=${imgW}, h=${imgH} (이미지: ${img.width}x${img.height})`);

          // 유효성 검사
          const safeX = Math.max(0, Math.min(imgX, img.width - 1));
          const safeY = Math.max(0, Math.min(imgY, img.height - 1));
          const safeW = Math.max(10, Math.min(imgW, img.width - safeX));
          const safeH = Math.max(10, Math.min(imgH, img.height - safeY));

          if (safeW <= 0 || safeH <= 0) {
            console.warn('⚠️ 샘플링 영역이 유효하지 않음');
            return;
          }

          console.log(`✅ 안전 샘플링: x=${safeX}, y=${safeY}, w=${safeW}, h=${safeH}`);

          const imageData = ctx.getImageData(safeX, safeY, safeW, safeH);
          const pixels = imageData.data;

          // 평균 밝기 계산
          let totalBrightness = 0;
          const pixelCount = pixels.length / 4;

          for (let i = 0; i < pixels.length; i += 4) {
            const r = pixels[i];
            const g = pixels[i + 1];
            const b = pixels[i + 2];
            const brightness = (r * 0.299 + g * 0.587 + b * 0.114);
            totalBrightness += brightness;
          }

          const avgBrightness = totalBrightness / pixelCount;
          state.backgroundBrightness = avgBrightness;

          console.log(`💡 배경 밝기: ${avgBrightness.toFixed(2)} / 255 (${((avgBrightness/255)*100).toFixed(0)}%)`);

          // 투명도 자동 조정
          const currentOpacity = localStorage.getItem('panelOpacity') || 100;
          updateOpacity(currentOpacity, true);
        };

        img.src = thumbnail.toDataURL();
      } catch (error) {
        console.error('❌ 배경 밝기 감지 실패:', error);
        console.error(error.stack);
      }
    }

    // 자동 투명도가 활성화되어 있으면 즉시 실행
    if (state.autoAdjustOpacity) {
      console.log('🚀 초기 배경 밝기 감지 시작...');
      setTimeout(() => detectBackgroundBrightness(), 1000);
    }

    // 주기적으로 배경 밝기 체크 (3초마다)
    setInterval(() => {
      if (state.autoAdjustOpacity) {
        detectBackgroundBrightness();
      }
    }, 3000);

    document.addEventListener('mousemove', (e) => {
      // 설정창이 열려 있으면 메인 창은 모든 마우스 이벤트 무시
      if (isSettingsOpen) {
        if (isOverInteractive) {
          isOverInteractive = false;
          ipcRenderer.send('set-ignore-mouse-events', true, { forward: true });
        }
        return;
      }

      if (mouseDownPos) {
        if (isDraggingMode) {
          const deltaX = e.screenX - mouseDownPos.screenX;
          const deltaY = e.screenY - mouseDownPos.screenY;
          ipcRenderer.send('move-window', deltaX, deltaY);
          mouseDownPos.screenX = e.screenX;
          mouseDownPos.screenY = e.screenY;
          return;
        }
        if (!isOverInteractive) {
          isOverInteractive = true;
          ipcRenderer.send('set-ignore-mouse-events', false);
        }
        return;
      }

      // 패널 주변에 마진을 두어 더 넓은 클릭 영역 제공
      const rect = DOM.mainPanel.getBoundingClientRect();
      const margin = 30; // 패널 주변 30px까지 클릭 가능
      const isOverPanel = (
        e.clientX >= rect.left - margin &&
        e.clientX <= rect.right + margin &&
        e.clientY >= rect.top - margin &&
        e.clientY <= rect.bottom + margin
      );

      if (isOverPanel !== isOverInteractive) {
        isOverInteractive = isOverPanel;
        ipcRenderer.send('set-ignore-mouse-events', !isOverInteractive, { forward: true });
      }
    });

    DOM.mainPanel.addEventListener('mousedown', (e) => {
      // 설정창이 열려 있으면 메인 패널 클릭 무시
      if (isSettingsOpen) return;

      if (e.button !== 0) return;
      mouseDownPos = { screenX: e.screenX, screenY: e.screenY };
      isDraggingMode = false;

      // 즉시 커서를 grabbing으로 변경
      DOM.mainPanel.style.cursor = 'grabbing';

      if (!isOverInteractive) {
        isOverInteractive = true;
        ipcRenderer.send('set-ignore-mouse-events', false);
      }
      dragTimer = setTimeout(() => {
        if (mouseDownPos) {
          console.log('드래그 모드 시작');
          isDraggingMode = true;
        }
        dragTimer = null;
      }, LONG_PRESS_DURATION);
    });

    document.addEventListener('mouseup', (e) => {
      if (e.button !== 0 || !mouseDownPos) return;
      const wasDragging = isDraggingMode;
      mouseDownPos = null;
      isDraggingMode = false;
      DOM.mainPanel.style.cursor = 'grab';

      if (dragTimer) {
        clearTimeout(dragTimer);
        dragTimer = null;

        console.log('짧은 클릭 -> 설정 열기');
        isSettingsOpen = true; // 설정창 열림 상태 설정
        openSettings();

      } else if (wasDragging) {
        console.log('드래그 종료');
      }

      setTimeout(() => {
        // 마우스업 후에도 패널 주변 영역 확인
        const rect = DOM.mainPanel.getBoundingClientRect();
        const margin = 30;
        const isOverPanelNow = (
          e.clientX >= rect.left - margin &&
          e.clientX <= rect.right + margin &&
          e.clientY >= rect.top - margin &&
          e.clientY <= rect.bottom + margin
        );
        ipcRenderer.send('set-ignore-mouse-events', !isOverPanelNow, { forward: true });
        isOverInteractive = isOverPanelNow;
      }, 10);
    });

    // 설정 창으로부터 변경 사항 수신
    ipcRenderer.on('opacity-changed', (_event, value) => {
      updateOpacity(value);
    });

    ipcRenderer.on('text-size-changed', (_event, value) => {
      console.log('🔔 text-size-changed 이벤트 받음:', value);
      updateTextSize(value);
    });

    ipcRenderer.on('text-color-changed', (_event, value) => {
      console.log('🔔 text-color-changed 이벤트 받음:', value);
      updateTextColor(value);
    });

    ipcRenderer.on('server-url-changed', (_event, value) => {
      state.SERVER_URL = value;
      console.log('💾 서버 URL 변경됨:', value);
    });

    ipcRenderer.on('display-mode-changed', (_event, value) => {
      state.displayMode = value;
      console.log('💾 표시 모드 변경됨:', value);
      // 현재 표시된 텍스트를 새로운 모드로 다시 표시
      if (state.currentOriginal || state.currentTranslated) {
        showResult(state.currentOriginal, state.currentTranslated);
      }
    });

    ipcRenderer.on('language-hint-changed', async (_event, value) => {
      console.log('🌐 언어 힌트 변경됨:', value);
      localStorage.setItem('languageHint', value);

      // 서버 재연결 (새로운 언어 힌트 적용)
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        console.log('🔄 언어 힌트 변경으로 인한 서버 재연결...');
        state.ws.close();
        await new Promise(resolve => setTimeout(resolve, 500));
        await connectWebSocket();
      }
    });

    ipcRenderer.on('auto-opacity-changed', (_event, value) => {
      state.autoAdjustOpacity = value;
      console.log('💾 자동 투명도 조정 변경됨:', value);
      localStorage.setItem('autoAdjustOpacity', value ? 'true' : 'false');

      // 자동 투명도가 활성화되면 즉시 배경 밝기 감지 실행
      if (value) {
        console.log('🚀 배경 밝기 감지 시작...');
        setTimeout(() => detectBackgroundBrightness(), 100);
      } else {
        // 자동 투명도가 비활성화되면 수동 투명도로 복원
        const currentOpacity = localStorage.getItem('panelOpacity') || 100;
        updateOpacity(currentOpacity, false);
      }
    });

    // 설정 창에서 초기 상태 요청 시 응답
    ipcRenderer.on('request-state-for-settings', () => {
      ipcRenderer.send('send-state-to-settings', {
        isServerConnected: state.isServerConnected,
        isRecording: state.isRecording,
        translationHistory: state.translationHistory
      });
    });

    // 설정 창에서 녹음 토글 요청
    ipcRenderer.on('toggle-recording', () => {
      if (state.isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    });

    // 설정 창이 닫힐 때 이벤트 수신
    ipcRenderer.on('settings-window-closed', () => {
      console.log('설정 창 닫힘 - 메인 창 마우스 이벤트 복원');
      isSettingsOpen = false;
      // 마우스 이벤트 처리 재개
      ipcRenderer.send('set-ignore-mouse-events', true, { forward: true });
    });

  } catch(e) {
    console.log('Electron IPC 사용 불가', e);
  }
}

async function init() {
  console.log('🚀 앱 초기화 시작');
  loadOpacity();
  loadTextSize();
  loadTextColor();
  loadSettings();
  updateServerStatus(false);
  updateRecordingStatus(false);
  setupEventListeners();
  setupElectronIntegration();
  
  // 앱 시작 시 자동으로 녹음 시작
  console.log('🎬 자동 녹음 시작...');
  await startRecording();
  
  console.log('✅ 초기화 완료 - Space 키로 녹음 시작/중지를 토글하세요');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
