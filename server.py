from flask import Flask, request, jsonify
import whisper
from deep_translator import GoogleTranslator
import os
import tempfile
import torch

app = Flask(__name__)

# CORS 설정 (웹 클라이언트에서 접근 가능하도록)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# 서버 시작 시 1회만 모델 로드
print("🤖 Whisper 모델 로딩 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)
print(f"✅ 모델 로드 완료 (device={device})")

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인용 엔드포인트"""
    return jsonify({
        'status': 'ok',
        'model': 'base',
        'device': device
    })

@app.route('/stt', methods=['POST'])
def speech_to_text():
    temp_path = None
    try:
        # 1. 클라이언트로부터 오디오 파일 받기
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']

        if audio_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        print(f"📥 오디오 파일 수신: {audio_file.filename}")

        # 2. 임시 파일로 저장 (Whisper는 파일 경로가 필요)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            temp_path = tmp_file.name
            audio_file.save(temp_path)

        print("🎤 음성 인식 처리 중...")

        # 3. Whisper로 STT 처리
        result = model.transcribe(temp_path, fp16=False)
        text = result['text'].strip()
        lang = result['language']

        print(f"🌍 감지 언어: {lang}")
        print(f"📝 인식 결과: {text}")

        # 4. 번역
        translated = ""
        try:
            if lang == 'ko':
                translated = GoogleTranslator(source='ko', target='en').translate(text)
            elif lang == 'en':
                translated = GoogleTranslator(source='en', target='ko').translate(text)
            else:
                # 기타 언어는 영어로 번역 후 한국어로
                eng_text = GoogleTranslator(source=lang, target='en').translate(text)
                translated = GoogleTranslator(source='en', target='ko').translate(eng_text)

            print(f"🌐 번역 완료: {translated}")
        except Exception as e:
            print(f"⚠️ 번역 실패: {e}")
            translated = "(번역 실패)"

        # 5. 결과 반환
        return jsonify({
            'success': True,
            'original': text,
            'translated': translated,
            'language': lang
        })

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

    finally:
        # 임시 파일 삭제
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # macOS에서 포트 5000은 AirPlay Receiver가 사용 중일 수 있음
    PORT = 8000

    print("="*60)
    print("🚀 STT 서버 시작")
    print("="*60)
    print(f"📡 접속 주소: http://0.0.0.0:{PORT}")
    print(f"🔗 Health Check: http://0.0.0.0:{PORT}/health")
    print(f"🎤 STT 엔드포인트: http://0.0.0.0:{PORT}/stt")
    print("="*60)
    app.run(host='0.0.0.0', port=PORT, debug=False)