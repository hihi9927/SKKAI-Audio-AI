#!/usr/bin/env python3
import requests
import pyaudio
import wave
import argparse
import os

def record_audio(filename, duration=5):
    """마이크로 음성을 녹음하여 WAV 파일로 저장"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

        print(f"🎤 녹음 시작... ({duration}초)")
        frames = []

        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
            # 진행률 표시
            if i % (RATE // CHUNK) == 0:
                print(f"   {i // (RATE // CHUNK) + 1}초...")

        print("✅ 녹음 완료")
        stream.stop_stream()
        stream.close()
        p.terminate()

        # WAV 파일로 저장
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        return True

    except Exception as e:
        print(f"❌ 녹음 실패: {e}")
        return False

def check_server_health(server_url):
    """서버 상태 확인"""
    try:
        health_url = server_url.replace('/stt', '/health')
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 서버 연결 성공")
            print(f"   모델: {data.get('model', 'unknown')}")
            print(f"   디바이스: {data.get('device', 'unknown')}")
            return True
        else:
            print(f"⚠️ 서버 응답 이상: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 서버 연결 실패: {e}")
        print(f"   서버 주소를 확인하세요: {server_url}")
        return False

def send_to_server(audio_file, server_url='http://localhost:5000/stt'):
    """서버로 오디오 파일 전송 및 결과 수신"""
    try:
        print(f"📤 서버로 전송 중: {server_url}")

        with open(audio_file, 'rb') as f:
            files = {'audio': (os.path.basename(audio_file), f, 'audio/wav')}
            response = requests.post(server_url, files=files, timeout=60)

        if response.status_code == 200:
            result = response.json()

            if result.get('success'):
                print("\n" + "="*60)
                print("✅ STT 처리 완료")
                print("="*60)
                print(f"🌍 감지 언어: {result.get('language', 'unknown')}")
                print(f"📝 원문: {result.get('original', '')}")
                print(f"🌐 번역: {result.get('translated', '')}")
                print("="*60)
                return result
            else:
                print(f"❌ 서버 처리 실패: {result.get('error', 'Unknown error')}")
                return None
        else:
            print(f"❌ HTTP 에러: {response.status_code}")
            print(f"   응답: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print("❌ 서버 응답 시간 초과 (60초)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 전송 실패: {e}")
        return None
    except Exception as e:
        print(f"❌ 예상치 못한 에러: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='STT 클라이언트')
    parser.add_argument('--server', type=str, default='http://localhost:8000/stt',
                        help='서버 주소 (예: http://192.168.0.10:8000/stt)')
    parser.add_argument('--duration', type=int, default=5,
                        help='녹음 시간 (초)')
    parser.add_argument('--audio', type=str, default=None,
                        help='기존 오디오 파일 전송 (녹음 건너뛰기)')
    args = parser.parse_args()

    print("="*60)
    print("🎙️  STT 클라이언트")
    print("="*60)

    # 서버 상태 확인
    if not check_server_health(args.server):
        print("\n⚠️ 서버가 실행 중인지 확인하세요.")
        print("   서버 실행: python server.py")
        return

    # 오디오 파일 준비
    if args.audio:
        # 기존 파일 사용
        if not os.path.exists(args.audio):
            print(f"❌ 파일을 찾을 수 없습니다: {args.audio}")
            return
        audio_file = args.audio
        print(f"📁 파일 사용: {audio_file}")
    else:
        # 새로 녹음
        audio_file = 'temp_recording.wav'
        if not record_audio(audio_file, duration=args.duration):
            return

    # 서버로 전송
    result = send_to_server(audio_file, server_url=args.server)

    # 임시 파일 삭제
    if not args.audio and os.path.exists(audio_file):
        os.remove(audio_file)
        print(f"\n🧹 임시 파일 삭제: {audio_file}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()