#!/usr/bin/env python3
"""
STT 클라이언트 데스크톱 앱
음성 감지 자동 녹음 시작 + 침묵 감지 자동 중지
"""
import webview
import os
import json
import sys
import io
import threading
import tempfile
import requests
import sounddevice as sd
import soundfile as sf
import numpy as np
import time

# noconsole 모드 대응
def setup_console():
    """콘솔 출력을 안전하게 설정"""
    try:
        if sys.stdout is None or not hasattr(sys.stdout, 'write'):
            sys.stdout = io.StringIO()
        if sys.stderr is None or not hasattr(sys.stderr, 'write'):
            sys.stderr = io.StringIO()
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

setup_console()

# 설정 파일 경로
def get_config_path():
    """설정 파일 경로 가져오기"""
    if getattr(sys, 'frozen', False):
        if sys.platform == 'darwin':
            app_dir = os.path.dirname(sys.executable)
            app_dir = os.path.join(app_dir, '..', 'Resources')
            app_dir = os.path.abspath(app_dir)
        else:
            app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    config_path = os.path.join(app_dir, 'stt_config.json')
    print(f"설정 파일 경로: {config_path}")
    return config_path

CONFIG_FILE = get_config_path()

class VoiceActivatedRecorder:
    """음성 감지 자동 녹음 및 침묵 감지 중지"""
    
    def __init__(self, api):
        self.api = api
        self.is_recording = False
        self.is_listening = False
        self.sample_rate = 16000
        self.chunk_duration = 0.01  # 10ms
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # 음성 및 침묵 감지 설정
        self.voice_threshold = 0.02      # 음성 감지 임계값
        self.silence_threshold = 0.015   # 침묵 판단 임계값
        self.silence_duration = 0.3      # 침묵 지속 시간 (초)
        self.min_record_duration = 0.5   # 최소 녹음 시간 (초)
        
        # 녹음 상태
        self.audio_buffer = []
        self.silence_start = None
        self.record_start_time = None
        
        print("VoiceActivatedRecorder 초기화 완료")
        print(f"음성 임계값: {self.voice_threshold}, 침묵 임계값: {self.silence_threshold}")
    
    def calculate_volume(self, audio_chunk):
        """오디오 청크의 평균 음량 계산"""
        return np.abs(audio_chunk).mean()
    
    def audio_callback(self, indata, frames, time_info, status):
        """오디오 스트림 콜백"""
        if status:
            print(f"오디오 콜백 상태: {status}")
        
        if not self.is_listening:
            return
        
        # 오디오 데이터 복사
        audio_chunk = indata.copy()
        
        # 음량 계산
        volume = self.calculate_volume(audio_chunk)
        
        # 녹음 중이 아닐 때: 음성 감지 대기
        if not self.is_recording:
            if volume > self.voice_threshold:
                # 음성 감지! 녹음 시작
                print(f"음성 감지 (음량: {volume:.4f}), 녹음 시작")
                self.is_recording = True
                self.audio_buffer = []
                self.record_start_time = time.time()
                self.silence_start = None
                
                # UI 업데이트
                try:
                    self.api.window.evaluate_js("showRecording()")
                except:
                    pass
        
        # 녹음 중일 때
        if self.is_recording:
            # 버퍼에 추가
            self.audio_buffer.append(audio_chunk)
            
            # 침묵 감지
            if volume < self.silence_threshold:
                if self.silence_start is None:
                    self.silence_start = time.time()
                elif time.time() - self.silence_start >= self.silence_duration:
                    # 최소 녹음 시간 체크
                    if self.record_start_time and time.time() - self.record_start_time >= self.min_record_duration:
                        # 침묵 감지, 녹음 처리
                        print(f"침묵 감지 (음량: {volume:.4f}), 녹음 중지")
                        self.process_recording()
            else:
                # 소리 감지, 침묵 타이머 리셋
                self.silence_start = None
    
    def process_recording(self):
        """녹음된 오디오 처리"""
        if not self.audio_buffer:
            return
        
        print(f"녹음 처리 시작: {len(self.audio_buffer)} 청크")
        
        # 버퍼 복사 및 초기화
        audio_data = np.concatenate(self.audio_buffer)
        self.audio_buffer = []
        self.is_recording = False
        self.silence_start = None
        
        # UI 업데이트
        try:
            self.api.window.evaluate_js("showProcessing()")
        except:
            pass
        
        # 별도 스레드에서 처리
        thread = threading.Thread(target=self._send_audio, args=(audio_data,))
        thread.daemon = True
        thread.start()
    
    def _send_audio(self, audio_data):
        """오디오를 서버로 전송"""
        try:
            # 최소 크기 체크
            if len(audio_data) < self.sample_rate * 0.3:  # 0.3초 이하
                print(f"오디오가 너무 짧습니다: {len(audio_data)} 샘플")
                return
            
            # WAV 파일로 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            # 16-bit PCM WAV로 저장
            sf.write(temp_path, audio_data, self.sample_rate, subtype='PCM_16')
            
            file_size = os.path.getsize(temp_path)
            print(f"오디오 파일 저장: {temp_path} ({file_size} bytes)")
            
            # 최소 파일 크기 체크 (5KB)
            if file_size < 5000:
                print(f"파일이 너무 작습니다: {file_size} bytes")
                os.unlink(temp_path)
                return
            
            # 서버 전송
            server_url = self.api.config.get('server_url', 'http://127.0.0.1:8000/stt')
            print(f"서버로 전송 중: {server_url}")
            
            with open(temp_path, 'rb') as f:
                files = {'audio': ('recording.wav', f, 'audio/wav')}
                headers = {
                    'ngrok-skip-browser-warning': 'true',
                    'Accept': 'application/json'
                }
                
                response = requests.post(
                    server_url,
                    files=files,
                    headers=headers,
                    timeout=30
                )
            
            print(f"서버 응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"STT 결과: {result}")
                
                if result.get('success'):
                    # 번역이 있으면 번역문만, 없으면 원문
                    if result.get('translated') and result['translated'] != '(번역 실패)':
                        text = result['translated']
                    else:
                        text = result.get('original', '')
                    
                    if text.strip():
                        text_escaped = json.dumps(text, ensure_ascii=False)
                        self.api.window.evaluate_js(f"showResult({text_escaped})")
                else:
                    error_msg = result.get('error', '서버 처리 실패')
                    print(f"STT 실패: {error_msg}")
            else:
                # 오류 응답 내용 출력
                try:
                    error_detail = response.text
                    print(f"서버 오류 상세: {error_detail}")
                except:
                    pass
                print(f"서버 오류: HTTP {response.status_code}")
            
            # 임시 파일 삭제
            os.unlink(temp_path)
            
        except requests.exceptions.ConnectionError as e:
            print(f"서버 연결 실패: {e}")
            print("서버가 실행 중인지 확인하세요.")
        except requests.exceptions.Timeout as e:
            print(f"서버 응답 시간 초과: {e}")
        except Exception as e:
            print(f"오디오 전송 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def start_listening(self):
        """음성 대기 시작"""
        if self.is_listening:
            print("이미 리스닝 중입니다")
            return
        
        print("음성 대기 시작")
        self.is_listening = True
        self.audio_buffer = []
        self.silence_start = None
        
        try:
            # InputStream 시작
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self.audio_callback
            )
            self.stream.start()
            print("오디오 스트림 시작됨 - 음성 감지 대기 중...")
            
        except Exception as e:
            print(f"리스닝 시작 실패: {e}")
            import traceback
            traceback.print_exc()
            self.is_listening = False
    
    def stop_listening(self):
        """음성 대기 중지"""
        print("음성 대기 중지")
        self.is_listening = False
        
        # 현재 녹음 중이면 처리
        if self.is_recording and self.audio_buffer:
            if self.record_start_time and time.time() - self.record_start_time >= self.min_record_duration:
                self.process_recording()
        
        try:
            # 스트림 중지
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            self.is_recording = False
            self.audio_buffer = []
            self.silence_start = None
            
            print("오디오 스트림 중지됨")
            
        except Exception as e:
            print(f"리스닝 중지 실패: {e}")
            import traceback
            traceback.print_exc()

class API:
    """Python ↔ JavaScript 통신용 API"""
    
    def __init__(self):
        print("API 초기화 시작")
        self.config = self.load_config()
        self.recorder = VoiceActivatedRecorder(self)
        self.window = None
        print(f"설정 파일 경로: {CONFIG_FILE}")
        print(f"로드된 설정: {self.config}")
        print("API 초기화 완료")
    
    def load_config(self):
        """저장된 설정 불러오기"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    print(f"설정 파일 로드 성공: {config}")
                    return config
            except Exception as e:
                print(f"설정 파일 로드 실패: {e}")
        else:
            print(f"설정 파일이 없습니다: {CONFIG_FILE}")
        return {'server_url': 'http://127.0.0.1:8000/stt'}
    
    def save_config(self, config):
        """설정 저장"""
        try:
            config_dir = os.path.dirname(CONFIG_FILE)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"설정 저장 성공: {config}")
            return True
        except Exception as e:
            print(f"설정 저장 실패: {e}")
            return False
    
    def toggle_listening(self):
        """음성 대기 토글"""
        print(f"toggle_listening 호출됨, 현재 상태: {self.recorder.is_listening}")
        
        if not self.recorder.is_listening:
            # 리스닝 시작
            self.recorder.start_listening()
            return {'listening': True}
        else:
            # 리스닝 중지
            self.recorder.stop_listening()
            return {'listening': False}
    
    def test_server_connection(self):
        """서버 연결 테스트"""
        server_url = self.config.get('server_url', 'http://127.0.0.1:8000/stt')
        health_url = server_url.replace('/stt', '/health')
        
        try:
            print(f"서버 연결 테스트: {health_url}")
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"서버 연결 성공: {data}")
                return {
                    'success': True,
                    'model': data.get('model', 'unknown'),
                    'device': data.get('device', 'unknown')
                }
            else:
                print(f"서버 응답 오류: {response.status_code}")
                return {'success': False, 'error': f'HTTP {response.status_code}'}
        except Exception as e:
            print(f"서버 연결 실패: {e}")
            return {'success': False, 'error': str(e)}

# HTML 내용 - 음성 감지 자동 녹음 UI
HTML_CONTENT = '''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>실시간 자막 입력창 (음성 감지 자동 녹음)</title>
<style>
  :root{ --fg:#fff; --glass:rgba(0,0,0,.35); --shadow:rgba(0,0,0,.55);
         --accent1:#c9a6ff; --accent2:#7ad7ff; --radius:18px; }
  html,body{height:100%;margin:0}
  body{
    display:flex; align-items:flex-end; justify-content:center;
    background: radial-gradient(1200px 600px at 50% 20%, #0f1220 0%, #070910 55%, #05070c 100%);
    color:var(--fg);
    font-family: "Malgun Gothic","맑은 고딕","Apple SD Gothic Neo","Nanum Gothic",sans-serif;
    overflow:hidden;
  }
  :lang(en), .en{ font-family: Arial, Helvetica, sans-serif; }
  body::before{content:"";position:fixed;inset:0;box-shadow:inset 0 0 220px rgba(0,0,0,.65);pointer-events:none}
  .wrap{width:min(92vw,1100px); margin:0 auto 10vh; padding:0 12px}
  .panel{
    background:var(--glass); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
    border-radius:var(--radius); box-shadow:0 8px 28px var(--shadow);
    padding: clamp(10px,2.2vmin,20px) clamp(14px,3vmin,32px);
  }
  .current{
    font-weight:500;
    font-size: clamp(20px,3.6vmin,40px); line-height:1.45;
    letter-spacing:.2px; text-shadow:0 2px 10px rgba(0,0,0,.45);
    min-height:1.2em;
  }
  .current.typing .caret{
    display:inline-block; width:.6ch; height:1.05em; transform:translateY(3px);
    background:linear-gradient(180deg,var(--accent1),var(--accent2)); opacity:.8; margin-left:.1ch;
    animation:blink 1s step-end infinite;
  }
  @keyframes blink{50%{opacity:0}}
  .rule{ margin:8px 0 10px; height:2px; opacity:.55; border-radius:999px;
         background:linear-gradient(90deg,var(--accent1),var(--accent2)); }
  .status{ font-size:0.8em; opacity:0.7; margin-top:8px; }
  .listening { color: #7ad7ff; }
  .recording { color: #ff6b6b; }
  .processing { color: #ffaa00; }
</style>
</head>
<body>
  <div class="wrap">
    <div class="panel" role="region" aria-live="polite" aria-atomic="false">
      <div id="current" class="current typing"><span id="currText"></span><span class="caret"></span></div>
      <div class="rule"></div>
      <div id="status" class="status"></div>
    </div>
  </div>

<script>
console.log('JavaScript 초기화 시작');

const currEl = document.getElementById('currText');
const current = document.getElementById('current');
const statusEl = document.getElementById('status');

let isListening = false;

function showResult(text){
  console.log('showResult 호출됨:', text);
  const t = (text ?? '').trim();
  if(!t) return;
  current.classList.add('typing');
  currEl.textContent = t;
}

function showRecording(){
  console.log('녹음 시작');
  statusEl.innerHTML = '<span class="recording">● 녹음 중</span>';
}

function showProcessing(){
  console.log('처리 중');
  statusEl.innerHTML = '<span class="processing">⏳ 처리 중</span>';
}

async function toggleListening(){
  console.log('toggleListening 호출됨, 현재 상태:', isListening);
  
  try {
    if(window.pywebview && window.pywebview.api){
      const result = await window.pywebview.api.toggle_listening();
      console.log('toggle 결과:', result);
      
      isListening = result.listening;
      
      if(isListening){
        statusEl.innerHTML = '<span class="listening">🎤 음성 대기 중</span>';
        currEl.textContent = '말씀하세요...';
      } else {
        statusEl.textContent = '준비';
        currEl.textContent = 'Space 키를 눌러 음성 대기를 시작하세요';
      }
      
    } else {
      console.error('pywebview API를 사용할 수 없습니다');
      showResult('[오류: API를 사용할 수 없습니다]');
    }
  } catch(error) {
    console.error('리스닝 토글 실패:', error);
    showResult('[오류: ' + error.message + ']');
  }
}

async function testServer(){
  console.log('서버 연결 테스트 시작');
  try {
    const result = await window.pywebview.api.test_server_connection();
    console.log('서버 테스트 결과:', result);
    
    if(result.success){
      statusEl.textContent = `서버 연결됨 (${result.model})`;
    } else {
      statusEl.textContent = `서버 연결 실패: ${result.error}`;
    }
  } catch(error) {
    console.error('서버 테스트 실패:', error);
    statusEl.textContent = '서버 테스트 실패';
  }
}

// 키보드 이벤트
window.addEventListener('keydown', (e) => {
  if(e.code === 'Space' && !e.target.matches('input')){
    console.log('Space 키 인식됨');
    e.preventDefault();
    toggleListening();
  }
  
  // T 키: 서버 연결 테스트
  if(e.code === 'KeyT' && !e.target.matches('input')){
    console.log('T 키 인식됨 - 서버 테스트');
    e.preventDefault();
    testServer();
  }
});

// pywebview 준비 완료
window.addEventListener('pywebviewready', () => {
  console.log('pywebviewready 이벤트 발생');
  currEl.textContent = 'Space 키를 눌러 음성 대기를 시작하세요';
  statusEl.textContent = '준비 완료 (T 키로 서버 테스트)';
  
  // 자동으로 서버 연결 테스트
  setTimeout(testServer, 500);
});

// 페이지 로드
if(document.readyState === 'complete'){
  console.log('페이지 이미 로드됨');
  setTimeout(() => {
    if(!window.pywebview){
      console.log('초기화 중...');
      currEl.textContent = '초기화 중...';
    }
  }, 100);
}

console.log('JavaScript 초기화 완료');
</script>
</body>
</html>
'''

def main():
    """앱 실행"""
    print("=" * 50)
    print("음성 감지 자동 녹음 앱 시작")
    print("=" * 50)
    
    try:
        api = API()
        
        print("윈도우 생성 중...")
        
        window = webview.create_window(
            title='STT 실시간 자막 (음성 감지)',
            html=HTML_CONTENT,
            js_api=api,
            width=1100,
            height=300,
            resizable=True,
            on_top=True,
            frameless=False
        )
        
        api.window = window
        print("윈도우 생성 완료")
        
        print("webview 시작...")
        webview.start(debug=False)
        print("webview 종료됨")
        
    except Exception as e:
        print(f"앱 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            error_log_path = os.path.join(os.path.dirname(CONFIG_FILE), 'error.log')
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Error: {str(e)}\n")
                traceback.print_exc(file=f)
            print(f"에러 로그 저장됨: {error_log_path}")
        except Exception as log_error:
            print(f"에러 로그 저장 실패: {log_error}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
