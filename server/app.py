#!/usr/bin/env python3
"""
STT 클라이언트 데스크톱 앱
PyWebView 기반 - HTML을 앱처럼 실행
"""
import webview
import os
import json
import sys
import io

# noconsole 모드 대응: stdout/stderr를 안전하게 처리
def setup_console():
    """콘솔 출력을 안전하게 설정"""
    try:
        # Windows noconsole 모드에서는 stdout이 None일 수 있음
        if sys.stdout is None or not hasattr(sys.stdout, 'write'):
            sys.stdout = io.StringIO()
        if sys.stderr is None or not hasattr(sys.stderr, 'write'):
            sys.stderr = io.StringIO()
        
        # UTF-8 인코딩 강제 (이모지 지원)
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except:
        # 모든 출력 무시
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

# 콘솔 설정 먼저 실행
setup_console()

# 설정 파일 경로
CONFIG_FILE = 'stt_config.json'

class API:
    """Python ↔ JavaScript 통신용 API"""
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        """저장된 설정 불러오기"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {'server_url': 'http://localhost:8000/stt'}
    
    def save_config(self, config):
        """설정 저장"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"설정 저장 실패: {e}")
            return False
    
    def get_server_url(self):
        """저장된 서버 주소 가져오기"""
        return self.config.get('server_url', '')
    
    def set_server_url(self, url):
        """서버 주소 저장"""
        self.config['server_url'] = url
        return self.save_config(self.config)

# HTML 내용 (인라인으로 포함)
HTML_CONTENT = '''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>실시간 자막 (데스크톱 앱)</title>
<style>
  :root{ --fg:#fff; --glass:rgba(0,0,0,.35); --shadow:rgba(0,0,0,.55);
         --accent1:#c9a6ff; --accent2:#7ad7ff; --radius:18px; }
  *{box-sizing:border-box}
  html,body{height:100%;margin:0;padding:0}
  body{
    display:flex; flex-direction:column;
    background: radial-gradient(1200px 600px at 50% 20%, #0f1220 0%, #070910 55%, #05070c 100%);
    color:var(--fg);
    font-family: "Malgun Gothic","맑은 고딕","Apple SD Gothic Neo","Nanum Gothic",sans-serif;
    overflow:hidden;
  }
  .header{
    padding:12px 20px; background:rgba(0,0,0,.3); border-bottom:1px solid rgba(255,255,255,.1);
    display:flex; gap:10px; align-items:center;
  }
  .header input{
    flex:1; background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.25);
    color:#fff; padding:8px 12px; border-radius:8px; font-size:.95rem;
  }
  .header button{
    border-radius:8px; border:1px solid rgba(255,255,255,.25);
    background:rgba(255,255,255,.12); color:#fff; padding:8px 16px; cursor:pointer;
  }
  .header button:hover{background:rgba(255,255,255,.2)}
  .content{flex:1; display:flex; align-items:flex-end; justify-content:center; padding:20px; overflow:auto}
  .wrap{width:100%; max-width:1100px; margin:0 auto}
  .panel{
    background:var(--glass); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
    border-radius:var(--radius); box-shadow:0 8px 28px var(--shadow);
    padding:20px 28px;
  }
  .current{
    font-weight:500; font-size:clamp(24px,4vmin,42px); line-height:1.45;
    letter-spacing:.2px; text-shadow:0 2px 10px rgba(0,0,0,.45); min-height:1.2em;
  }
  .current.typing .caret{
    display:inline-block; width:.6ch; height:1.05em; transform:translateY(3px);
    background:linear-gradient(180deg,var(--accent1),var(--accent2)); opacity:.8; margin-left:.1ch;
    animation:blink 1s step-end infinite;
  }
  @keyframes blink{50%{opacity:0}}
  .rule{ margin:10px 0; height:2px; opacity:.55; border-radius:999px;
         background:linear-gradient(90deg,var(--accent1),var(--accent2)); }
  .finals{ max-height:250px; overflow:auto; scrollbar-width:thin; padding-right:4px; }
  .line{ font-size:clamp(18px,3vmin,28px); line-height:1.42; opacity:.95; margin:.3em 0;
         animation:fadeIn .35s ease both; }
  @keyframes fadeIn{from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none}}
  .meta{ display:flex; gap:10px; margin-top:12px; opacity:.9; font-size:.95rem; align-items:center; flex-wrap:wrap }
  .pill{ border:1px solid rgba(255,255,255,.25); border-radius:999px; padding:6px 14px; display:flex; align-items:center }
  .status-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;background:#888;transition:background-color .2s ease}
  .on{background:#35d07f}.err{background:#ff6b6b}.processing{background:#ffaa00}
  .ctrl{ display:flex; gap:10px; flex-wrap:wrap; margin-top:12px }
  .btn{
    border-radius:12px; border:1px solid rgba(255,255,255,.25);
    background:rgba(255,255,255,.08); color:#fff; padding:12px 18px; cursor:pointer; user-select:none;
    font-size:1rem;
  }
  .btn:hover{background:rgba(255,255,255,.15)}
  .btn:disabled{opacity:.5; cursor:not-allowed}
</style>
</head>
<body>
  <!-- 상단 헤더: 서버 주소 설정 -->
  <div class="header">
    <label>서버 주소:</label>
    <input type="text" id="serverUrl" placeholder="http://localhost:8000/stt">
    <button id="saveBtn">💾 저장</button>
    <button id="connectBtn">🔌 연결 테스트</button>
  </div>

  <!-- 메인 컨텐츠 -->
  <div class="content">
    <div class="wrap">
      <div class="panel">
        <!-- 실시간 줄 -->
        <div id="current" class="current typing"><span id="currText"></span><span class="caret"></span></div>
        <div class="rule"></div>
        
        <!-- 확정 라인 -->
        <div id="finals" class="finals"></div>
        
        <!-- 상태 정보 -->
        <div class="meta">
          <div class="pill">
            <span id="recDot" class="status-dot"></span>
            <span id="recState">준비</span>
          </div>
          <div class="pill" id="langInfo" style="display:none">
            🌍 <span id="langText">-</span>
          </div>
        </div>
        
        <!-- 컨트롤 버튼 -->
        <div class="ctrl">
          <button id="micBtn" class="btn">🎤 녹음 시작 (5초)</button>
          <button id="clearBtn" class="btn">🗑️ 초기화</button>
        </div>
      </div>
    </div>
  </div>

<script>
const RECORD_DURATION = 5;

/* DOM */
const currEl = document.getElementById('currText');
const current = document.getElementById('current');
const finalsEl = document.getElementById('finals');
const micBtn = document.getElementById('micBtn');
const clearBtn = document.getElementById('clearBtn');
const recDot = document.getElementById('recDot');
const recState = document.getElementById('recState');
const serverUrlInput = document.getElementById('serverUrl');
const saveBtn = document.getElementById('saveBtn');
const connectBtn = document.getElementById('connectBtn');
const langInfo = document.getElementById('langInfo');
const langText = document.getElementById('langText');

let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

/* UI 함수 */
function renderPartial(text){
  current.classList.add('typing');
  currEl.textContent = text || '';
}

function commitFinal(text, lang = null){
  const t = (text ?? '').trim();
  if(!t) return;
  const el = document.createElement('div');
  el.className = 'line';
  el.textContent = t;
  finalsEl.appendChild(el);
  finalsEl.scrollTop = finalsEl.scrollHeight;
  currEl.textContent = '';
  current.classList.remove('typing');
  if(lang){
    langText.textContent = lang;
    langInfo.style.display = 'flex';
  }
}

function setStatus(status, text){
  recDot.classList.remove('on','err','processing');
  if(status) recDot.classList.add(status);
  recState.textContent = text;
}

function clearAll(){
  finalsEl.innerHTML = '';
  currEl.textContent = '';
  current.classList.remove('typing');
  langInfo.style.display = 'none';
}

/* Python API 통신 */
async function loadServerUrl(){
  if(window.pywebview){
    const url = await window.pywebview.api.get_server_url();
    if(url) serverUrlInput.value = url;
  }
}

async function saveServerUrl(){
  const url = serverUrlInput.value.trim();
  if(!url){
    alert('서버 주소를 입력하세요');
    return;
  }
  if(window.pywebview){
    await window.pywebview.api.set_server_url(url);
  }
  alert('저장되었습니다');
}

/* 연결 테스트 */
async function testConnection(){
  const serverUrl = serverUrlInput.value.trim();
  if(!serverUrl){
    alert('서버 주소를 입력하세요');
    return;
  }
  
  setStatus('processing', '연결 중...');
  
  try {
    const healthUrl = serverUrl.replace('/stt', '/health');
    const response = await fetch(healthUrl, {
      headers: {
        'ngrok-skip-browser-warning': 'true'
      }
    });
    
    if(response.ok){
      const data = await response.json();
      setStatus('on', '연결 성공');
      alert(`✅ 서버 연결 성공!\\n모델: ${data.model}\\n디바이스: ${data.device}`);
      setTimeout(() => setStatus(null, '준비'), 2000);
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch(error){
    setStatus('err', '연결 실패');
    alert(`❌ 서버 연결 실패\\n${error.message}\\n\\n서버가 실행 중인지 확인하세요.`);
  }
}

/* STT 서버 통신 */
async function sendToServer(audioBlob){
  const serverUrl = serverUrlInput.value.trim();
  if(!serverUrl){
    setStatus('err', '서버 주소 입력 필요');
    alert('먼저 서버 주소를 입력하고 저장하세요');
    return;
  }
  
  setStatus('processing', '처리 중...');
  
  try {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    
    const response = await fetch(serverUrl, {
      method: 'POST',
      headers: {
        'ngrok-skip-browser-warning': 'true'
      },
      body: formData
    });
    
    if(!response.ok){
      throw new Error(`HTTP ${response.status}`);
    }
    
    const result = await response.json();
    
    if(result.success){
      commitFinal(result.original, result.language);
      if(result.translated && result.translated !== '(번역 실패)'){
        commitFinal(`→ ${result.translated}`);
      }
      setStatus('on', '완료');
      setTimeout(() => setStatus(null, '준비'), 2000);
    } else {
      throw new Error(result.error || '서버 처리 실패');
    }
    
  } catch(error){
    console.error('STT 에러:', error);
    setStatus('err', `오류: ${error.message}`);
    commitFinal(`[오류: ${error.message}]`);
  }
}

/* 녹음 */
async function startRecording(){
  if(isRecording) return;
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    
    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };
    
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      stream.getTracks().forEach(track => track.stop());
      await sendToServer(audioBlob);
      isRecording = false;
      micBtn.textContent = '🎤 녹음 시작 (5초)';
      micBtn.disabled = false;
    };
    
    mediaRecorder.start();
    isRecording = true;
    
    renderPartial('🎤 녹음 중...');
    setStatus('on', `녹음 중 (${RECORD_DURATION}초)`);
    micBtn.textContent = '⏸️ 녹음 중...';
    micBtn.disabled = true;
    
    let remaining = RECORD_DURATION;
    const countdown = setInterval(() => {
      remaining--;
      if(remaining > 0){
        setStatus('on', `녹음 중 (${remaining}초)`);
      } else {
        clearInterval(countdown);
      }
    }, 1000);
    
    setTimeout(() => {
      if(mediaRecorder && mediaRecorder.state === 'recording'){
        mediaRecorder.stop();
      }
      clearInterval(countdown);
    }, RECORD_DURATION * 1000);
    
  } catch(error){
    console.error('마이크 접근 에러:', error);
    setStatus('err', '마이크 접근 실패');
    alert('마이크 접근이 거부되었습니다.\\n시스템 설정에서 마이크 권한을 허용해주세요.');
  }
}

/* 이벤트 */
micBtn.addEventListener('click', startRecording);
clearBtn.addEventListener('click', clearAll);
saveBtn.addEventListener('click', saveServerUrl);
connectBtn.addEventListener('click', testConnection);

window.addEventListener('keydown', (e) => {
  if(e.code === 'Space' && !e.target.matches('input')){
    e.preventDefault();
    startRecording();
  }
});

/* 초기화 */
window.addEventListener('pywebviewready', () => {
  loadServerUrl();
});

setStatus(null, '준비');
</script>
</body>
</html>
'''

def safe_print(*args, **kwargs):
    """안전한 print (인코딩 에러 무시)"""
    try:
        print(*args, **kwargs)
    except (UnicodeEncodeError, AttributeError, OSError):
        pass  # 조용히 무시

def main():
    """앱 실행"""
    try:
        api = API()
        
        # 윈도우 생성
        window = webview.create_window(
            title='STT 실시간 자막',
            html=HTML_CONTENT,
            js_api=api,
            width=1000,
            height=700,
            resizable=True,
            min_size=(800, 600)
        )
        
        # 앱 실행
        webview.start(debug=False)
        
    except Exception as e:
        # 에러를 파일로 기록 (이모지 없이)
        try:
            with open('error.log', 'w', encoding='utf-8') as f:
                f.write(f"Error: {str(e)}\n")
                import traceback
                traceback.print_exc(file=f)
        except:
            pass  # 로그 실패해도 계속 진행

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()