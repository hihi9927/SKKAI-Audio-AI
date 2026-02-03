const { ipcRenderer } = require('electron');

const DOM = {
  serverIndicatorMini: document.getElementById('serverIndicatorMini'),
  recordingIndicatorMini: document.getElementById('recordingIndicatorMini'),
  closeSettings: document.getElementById('closeSettings'),
  serverUrlInput: document.getElementById('serverUrlInput'),
  modeTranslateOnly: document.getElementById('modeTranslateOnly'),
  modeTranscriptOnly: document.getElementById('modeTranscriptOnly'),
  modeBoth: document.getElementById('modeBoth'),
  opacitySlider: document.getElementById('opacitySlider'),
  opacityValue: document.getElementById('opacityValue'),
  textSizeSlider: document.getElementById('textSizeSlider'),
  textSizeValue: document.getElementById('textSizeValue'),
  textColorWhite: document.getElementById('textColorWhite'),
  textColorBlack: document.getElementById('textColorBlack'),
  languageAuto: document.getElementById('languageAuto'),
  languageEnglish: document.getElementById('languageEnglish'),
  languageKorean: document.getElementById('languageKorean'),
  translationLog: document.getElementById('translationLog'),
  closeAppBtn: document.getElementById('closeAppBtn')
};

// Receive status updates from main window
ipcRenderer.on('update-server-status', (event, connected) => {
  DOM.serverIndicatorMini.classList.toggle('connected', connected);
});

ipcRenderer.on('update-recording-status', (event, recording) => {
  DOM.recordingIndicatorMini.classList.toggle('active', recording);
});

ipcRenderer.on('update-translation-log', (event, history) => {
  updateLogDisplay(history);
});

function updateLogDisplay(history) {
  if (history.length === 0) {
    DOM.translationLog.innerHTML = '<div class="log-empty">No translation history yet</div>';
    return;
  }

  DOM.translationLog.innerHTML = history.map(entry => `
    <div class="log-entry">
      <div class="log-time">${entry.time}</div>
      <div class="log-text">${entry.text.replace(/\n/g, '<br>')}</div>
    </div>
  `).join('');
}

function loadSettings() {
  const savedUrl = localStorage.getItem('serverUrl');
  if (savedUrl) {
    DOM.serverUrlInput.value = savedUrl;
  }

  const savedDisplayMode = localStorage.getItem('displayMode') || 'both';
  if (savedDisplayMode === 'translateOnly') {
    DOM.modeTranslateOnly.checked = true;
  } else if (savedDisplayMode === 'transcriptOnly') {
    DOM.modeTranscriptOnly.checked = true;
  } else {
    DOM.modeBoth.checked = true;
  }

  const savedOpacity = localStorage.getItem('panelOpacity');
  if (savedOpacity) {
    DOM.opacitySlider.value = savedOpacity;
    DOM.opacityValue.textContent = savedOpacity + '%';
  }

  const savedTextSize = localStorage.getItem('textSize');
  if (savedTextSize) {
    DOM.textSizeSlider.value = savedTextSize;
    DOM.textSizeValue.textContent = savedTextSize + '%';
  }

  const savedTextColor = localStorage.getItem('textColor') || 'white';
  if (savedTextColor === 'white') {
    DOM.textColorWhite.checked = true;
  } else {
    DOM.textColorBlack.checked = true;
  }

  const savedLanguageHint = localStorage.getItem('languageHint') || 'auto';
  if (savedLanguageHint === 'auto') {
    DOM.languageAuto.checked = true;
  } else if (savedLanguageHint === 'en') {
    DOM.languageEnglish.checked = true;
  } else {
    DOM.languageKorean.checked = true;
  }
}

function setupEventListeners() {
  // Space key event - toggle recording even in settings window
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.target.matches('input, select, textarea')) {
      e.preventDefault();
      if (e.repeat) return; // Prevent continuous input

      // Send recording toggle request to main window
      ipcRenderer.send('toggle-recording');
    }
  });

  // Close settings
  DOM.closeSettings.addEventListener('click', () => {
    ipcRenderer.send('close-settings-window');
  });

  // Change opacity
  DOM.opacitySlider.addEventListener('input', (e) => {
    const value = e.target.value;
    DOM.opacityValue.textContent = value + '%';
    localStorage.setItem('panelOpacity', value);
    ipcRenderer.send('opacity-changed', value);
  });

  // Change text size
  DOM.textSizeSlider.addEventListener('input', (e) => {
    const value = e.target.value;
    DOM.textSizeValue.textContent = value + '%';
    localStorage.setItem('textSize', value);
    ipcRenderer.send('text-size-changed', value);
  });

  // Change server URL
  DOM.serverUrlInput.addEventListener('change', (e) => {
    const value = e.target.value.trim();
    localStorage.setItem('serverUrl', value);
    ipcRenderer.send('server-url-changed', value);
  });

  // Change text color
  const textColorHandler = (e) => {
    const color = e.target.value;
    localStorage.setItem('textColor', color);
    ipcRenderer.send('text-color-changed', color);
  };

  DOM.textColorWhite.addEventListener('change', textColorHandler);
  DOM.textColorBlack.addEventListener('change', textColorHandler);

  // Change display mode
  const displayModeHandler = (e) => {
    const mode = e.target.value;
    localStorage.setItem('displayMode', mode);
    ipcRenderer.send('display-mode-changed', mode);
  };

  DOM.modeTranslateOnly.addEventListener('change', displayModeHandler);
  DOM.modeTranscriptOnly.addEventListener('change', displayModeHandler);
  DOM.modeBoth.addEventListener('change', displayModeHandler);

  // Change language hint
  const languageHintHandler = (e) => {
    const language = e.target.value;
    localStorage.setItem('languageHint', language);
    ipcRenderer.send('language-hint-changed', language);
  };

  DOM.languageAuto.addEventListener('change', languageHintHandler);
  DOM.languageEnglish.addEventListener('change', languageHintHandler);
  DOM.languageKorean.addEventListener('change', languageHintHandler);

  // Close app
  DOM.closeAppBtn.addEventListener('click', () => {
    ipcRenderer.send('close-app');
  });
}

function init() {
  loadSettings();
  setupEventListeners();

  // Request initial state from main window
  ipcRenderer.send('request-initial-state');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
