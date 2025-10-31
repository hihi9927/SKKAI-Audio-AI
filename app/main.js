const { app, BrowserWindow, ipcMain } = require('electron');

let win;

function createWindow() {
  win = new BrowserWindow({
    width: 1000,
    height: 200,
    transparent: true,        
    frame: false, 
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    backgroundColor: '#00000000',  
    alwaysOnTop: true,
    resizable: true
  });

  // 투명한 영역 클릭 통과 설정
  win.setIgnoreMouseEvents(true, { forward: true });

  win.loadFile('index.html');

  // 개발자 도구 (필요시 주석 해제)
  // win.webContents.openDevTools();
}

// 렌더러에서 마우스 이벤트 제어 요청 처리
ipcMain.on('set-ignore-mouse-events', (event, ignore, options) => {
  const webContents = event.sender;
  const win = BrowserWindow.fromWebContents(webContents);
  if (win) {
    win.setIgnoreMouseEvents(ignore, options);
  }
});

// 창 위치 요청 (비동기)
ipcMain.on('get-window-position', (event) => {
  const webContents = event.sender;
  const win = BrowserWindow.fromWebContents(webContents);
  if (win) {
    const pos = win.getPosition();
    event.reply('window-position', { x: pos[0], y: pos[1] });
  }
});

// 창 위치 요청 (동기)
ipcMain.on('get-window-position-sync', (event) => {
  const webContents = event.sender;
  const win = BrowserWindow.fromWebContents(webContents);
  if (win) {
    const pos = win.getPosition();
    event.returnValue = { x: pos[0], y: pos[1] };
  } else {
    event.returnValue = { x: 0, y: 0 };
  }
});

// 창 이동
ipcMain.on('move-window', (event, pos) => {
  const webContents = event.sender;
  const win = BrowserWindow.fromWebContents(webContents);
  if (win) {
    win.setPosition(Math.round(pos.x), Math.round(pos.y), false);
  }
});

// 창 높이 조정
ipcMain.on('resize-window', (event, height) => {
  const webContents = event.sender;
  const win = BrowserWindow.fromWebContents(webContents);
  if (win) {
    const [width] = win.getSize();
    win.setSize(width, Math.max(150, Math.min(1000, height)));
  }
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
