const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

let mainWindow = null;
let settingsWindow = null;

function createWindow() {
  const { screen } = require('electron');
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;

  // 창을 화면 하단 중앙에 배치 (하단에서 200px 위)
  const windowWidth = 1000;
  const windowHeight = 275;
  const x = Math.floor((screenWidth - windowWidth) / 2);
  const y = screenHeight - windowHeight - 200; // 하단에서 200px 위

  mainWindow = new BrowserWindow({
    width: windowWidth,
    height: windowHeight,
    x: x,
    y: y,
    transparent: true,
    frame: false,
    hasShadow: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    backgroundColor: '#00000000',
    alwaysOnTop: true,
    resizable: true,
    skipTaskbar: false,
    visualEffectState: 'active',
    vibrancy: null
  });

  // PowerPoint 전체화면 위에도 표시되도록 최상위 레벨 설정
  mainWindow.setAlwaysOnTop(true, 'screen-saver');

  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  // 개발자 도구 (필요시 주석 해제)
  // mainWindow.webContents.openDevTools();

  // 페이지 로딩 완료 후 준비 상태 설정
  mainWindow.webContents.on('did-finish-load', () => {
    // 창이 완전히 로드되었음을 표시
  });

  // 창이 닫힐 때 모든 리소스 정리
  mainWindow.on('closed', () => {
    if (settingsWindow && !settingsWindow.isDestroyed()) {
      settingsWindow.close();
    }
    mainWindow = null;
  });
}

function createSettingsWindow() {
  // 이미 설정 창이 열려 있으면 포커스만 이동
  if (settingsWindow && !settingsWindow.isDestroyed()) {
    settingsWindow.moveTop();
    settingsWindow.focus();
    settingsWindow.show();
    return;
  }

  const { screen } = require('electron');
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;

  // 설정 창 크기
  const windowWidth = 450;
  const windowHeight = 700;

  // 화면 중앙에 배치
  const x = Math.floor((screenWidth - windowWidth) / 2);
  const y = Math.floor((screenHeight - windowHeight) / 2);

  settingsWindow = new BrowserWindow({
    width: windowWidth,
    height: windowHeight,
    x: x,
    y: y,
    transparent: true,
    frame: false,
    hasShadow: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    backgroundColor: '#00000000',
    alwaysOnTop: true,
    resizable: false,
    skipTaskbar: false,
    title: '설정',
    minimizable: true,
    maximizable: false,
    closable: true,
    roundedCorners: true,
    visualEffectState: 'active',
    vibrancy: null,
    parent: mainWindow  // 메인 창을 부모로 설정 - 이것이 핵심!
  });

  settingsWindow.loadFile(path.join(__dirname, 'settings.html'));

  // 개발자 도구 (필요시 주석 해제)
  // settingsWindow.webContents.openDevTools();

  // 페이지 로딩 완료 후 준비 상태 설정
  settingsWindow.webContents.on('did-finish-load', () => {
    // 메인 창에 초기 상태 요청
    if (mainWindow && !mainWindow.isDestroyed() && mainWindow.webContents && !mainWindow.webContents.isDestroyed()) {
      try {
        if (!mainWindow.webContents.isLoading()) {
          mainWindow.webContents.send('request-state-for-settings');
        }
      } catch (e) {
        // 무시
      }
    }
  });

  // 설정 창이 닫힐 때 모든 리소스 정리
  settingsWindow.on('closed', () => {
    // 메인 창에 설정 창이 닫혔음을 알림 (안전하게)
    if (mainWindow && !mainWindow.isDestroyed() && mainWindow.webContents && !mainWindow.webContents.isDestroyed()) {
      try {
        // 페이지 로딩이 완료된 경우에만 전송
        if (!mainWindow.webContents.isLoading()) {
          mainWindow.webContents.send('settings-window-closed');
        }
      } catch (e) {
        // 무시
      }
    }
    settingsWindow = null;
  });

  // 개발자 도구 (필요시 주석 해제)
  // settingsWindow.webContents.openDevTools();
}

// IPC 리스너 설정
function setupIpcListeners() {
  // 'resize-window' 메시지를 받으면 창 크기를 조절
  ipcMain.on('resize-window', (event, height) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      const { screen } = require('electron');
      const primaryDisplay = screen.getPrimaryDisplay();
      const { height: screenHeight } = primaryDisplay.workAreaSize;

      const [width, oldHeight] = mainWindow.getSize(); // ◀◀ 이전 높이 가져오기
      const [x, oldY] = mainWindow.getPosition();     // ◀◀ 이전 Y 위치 가져오기

      // 요청한 높이로 창 크기를 조절 (최소 높이 100px, 최대 화면 높이의 90%)
      const maxHeight = Math.floor(screenHeight * 0.9);
      const newHeight = Math.max(100, Math.min(height, maxHeight));

      // ◀◀ (핵심) 창이 위로 커지도록 새 Y 위치 계산
      let newY = oldY - (newHeight - oldHeight);

      // 창이 화면 위로 벗어나지 않도록 보정
      newY = Math.max(0, newY);

      // ◀◀ setSize 대신 setBounds로 X, Y, Width, Height 동시 설정
      mainWindow.setBounds({ x: x, y: newY, width: width, height: newHeight }, false);
    }
  });

  // 'set-ignore-mouse-events' 메시지를 받으면 마우스 클릭 통과 설정
  ipcMain.on('set-ignore-mouse-events', (event, ignore, options) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.setIgnoreMouseEvents(ignore, options);
    }
  });

  // 'move-window' 메시지를 받으면 창 위치 이동
  ipcMain.on('move-window', (event, deltaX, deltaY) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      const [currentX, currentY] = mainWindow.getPosition();
      mainWindow.setPosition(currentX + deltaX, currentY + deltaY, false);
    }
  });

  // 'toggle-fullscreen-mode' 메시지를 받으면 전체 화면 모드 토글
  let originalBounds = null; // 원래 창 크기/위치 저장용

  ipcMain.on('toggle-fullscreen-mode', (event, isFullscreen) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (isFullscreen) {
        // 원래 위치/크기 저장
        originalBounds = mainWindow.getBounds();

        // 투명도 해제하고 검정 배경 적용
        mainWindow.setBackgroundColor('#000000');

        // 전체 화면으로
        const { screen } = require('electron');
        const primaryDisplay = screen.getPrimaryDisplay();
        const { width, height } = primaryDisplay.bounds;

        mainWindow.setBounds({
          x: 0,
          y: 0,
          width: width,
          height: height
        }, true);
      } else {
        // 투명 배경 복원
        mainWindow.setBackgroundColor('#00000000');

        // 원래 크기로 복원
        if (originalBounds) {
          setTimeout(() => {
            mainWindow.setBounds(originalBounds, true);
          }, 50);
        } else {
          // 기본 위치로
          const { screen } = require('electron');
          const primaryDisplay = screen.getPrimaryDisplay();
          const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;

          const windowWidth = 1000;
          const windowHeight = 275;
          const x = Math.floor((screenWidth - windowWidth) / 2);
          const y = screenHeight - windowHeight - 200;

          setTimeout(() => {
            mainWindow.setBounds({
              x: x,
              y: y,
              width: windowWidth,
              height: windowHeight
            }, true);
          }, 50);
        }
      }
    }
  });

  // 설정 창 열기 요청
  ipcMain.on('open-settings-window', () => {
    createSettingsWindow();
    // 설정 창이 열리면 메인 창의 마우스 이벤트 무시 설정
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.setIgnoreMouseEvents(true, { forward: true });
    }
  });

  // 설정 창 닫기 요청
  ipcMain.on('close-settings-window', () => {
    if (settingsWindow && !settingsWindow.isDestroyed()) {
      settingsWindow.close();
    }
  });

  // 안전하게 메시지 전송하는 헬퍼 함수
  const safelySeendToMain = (channel, ...args) => {
    if (mainWindow && !mainWindow.isDestroyed() && mainWindow.webContents && !mainWindow.webContents.isDestroyed()) {
      try {
        if (!mainWindow.webContents.isLoading()) {
          mainWindow.webContents.send(channel, ...args);
        }
      } catch (e) {
        // 무시
      }
    }
  };

  const safelySendToSettings = (channel, ...args) => {
    if (settingsWindow && !settingsWindow.isDestroyed() && settingsWindow.webContents && !settingsWindow.webContents.isDestroyed()) {
      try {
        if (!settingsWindow.webContents.isLoading()) {
          settingsWindow.webContents.send(channel, ...args);
        }
      } catch (e) {
        // 무시
      }
    }
  };

  // 설정 변경 사항들을 메인 창으로 전달
  ipcMain.on('opacity-changed', (event, value) => {
    safelySeendToMain('opacity-changed', value);
  });

  ipcMain.on('text-size-changed', (_event, value) => {
    safelySeendToMain('text-size-changed', value);
  });

  ipcMain.on('text-color-changed', (_event, value) => {
    safelySeendToMain('text-color-changed', value);
  });

  ipcMain.on('server-url-changed', (event, value) => {
    safelySeendToMain('server-url-changed', value);
  });

  ipcMain.on('lang-changed', (event, value) => {
    safelySeendToMain('lang-changed', value);
  });

  ipcMain.on('polish-changed', (event, value) => {
    safelySeendToMain('polish-changed', value);
  });

  ipcMain.on('translate-changed', (event, value) => {
    safelySeendToMain('translate-changed', value);
  });

  ipcMain.on('display-mode-changed', (_event, value) => {
    safelySeendToMain('display-mode-changed', value);
  });

  ipcMain.on('auto-opacity-changed', (_event, value) => {
    safelySeendToMain('auto-opacity-changed', value);
  });

  // 앱 종료 요청
  ipcMain.on('close-app', () => {
    app.quit();
  });

  // 설정 창에서 녹음 토글 요청
  ipcMain.on('toggle-recording', () => {
    safelySeendToMain('toggle-recording');
  });

  // 윈도우 위치 요청
  ipcMain.handle('get-window-bounds', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      return mainWindow.getBounds();
    }
    return { x: 0, y: 0, width: 0, height: 0 };
  });

  // 초기 상태 요청 (설정 창이 열릴 때)
  ipcMain.on('request-initial-state', (event) => {
    safelySeendToMain('request-state-for-settings');
  });

  // 메인 창으로부터 상태를 받아 설정 창으로 전달
  ipcMain.on('send-state-to-settings', (event, state) => {
    safelySendToSettings('update-server-status', state.isServerConnected);
    safelySendToSettings('update-recording-status', state.isRecording);
    safelySendToSettings('update-translation-log', state.translationHistory);
  });

  // 메인 창의 상태 업데이트를 설정 창으로 전달
  ipcMain.on('status-update', (event, statusType, value) => {
    // statusType: 'server', 'recording', 'translation'
    // 설정 창에서는 'update-server-status', 'update-recording-status', 'update-translation-log'로 받음
    const channelMap = {
      'server': 'update-server-status',
      'recording': 'update-recording-status',
      'translation': 'update-translation-log'
    };
    const channel = channelMap[statusType];
    if (channel) {
      safelySendToSettings(channel, value);
    }
  });
}

// 앱 준비 완료
app.whenReady().then(() => {
  createWindow();
  setupIpcListeners();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// 모든 창이 닫힐 때
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// 앱이 종료되기 전에 모든 리소스 정리
app.on('will-quit', () => {
  // 모든 창 닫기
  if (settingsWindow && !settingsWindow.isDestroyed()) {
    settingsWindow.destroy();
    settingsWindow = null;
  }
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.destroy();
    mainWindow = null;
  }

  // IPC 리스너 정리
  ipcMain.removeAllListeners('resize-window');
  ipcMain.removeAllListeners('set-ignore-mouse-events');
  ipcMain.removeAllListeners('move-window');
  ipcMain.removeAllListeners('toggle-fullscreen-mode');
  ipcMain.removeAllListeners('open-settings-window');
  ipcMain.removeAllListeners('close-settings-window');
  ipcMain.removeAllListeners('opacity-changed');
  ipcMain.removeAllListeners('text-size-changed');
  ipcMain.removeAllListeners('text-color-changed');
  ipcMain.removeAllListeners('server-url-changed');
  ipcMain.removeAllListeners('lang-changed');
  ipcMain.removeAllListeners('polish-changed');
  ipcMain.removeAllListeners('translate-changed');
  ipcMain.removeAllListeners('display-mode-changed');
  ipcMain.removeAllListeners('auto-opacity-changed');
  ipcMain.removeAllListeners('close-app');
  ipcMain.removeAllListeners('toggle-recording');
  ipcMain.removeAllListeners('request-initial-state');
  ipcMain.removeAllListeners('send-state-to-settings');
  ipcMain.removeAllListeners('status-update');
});
