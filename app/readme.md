## 1) 사전 준비

### A. Node.js 설치

* [nodejs.org](https://nodejs.org/) 에서 **LTS 버전** 다운로드 후 설치
* 설치 확인:

  ```bash
  node -v
  npm -v
  ```

  * 권장: Node 18+ / npm 9+

## 2) 프로젝트 가져오기

```bash
# (선택) 새 폴더에서 시작
mkdir 원하는_파일명 && cd 원하는_파일명

# 이 저장소를 복제하거나, 제공받은 파일을 이 폴더에 넣습니다.
# 아래 3개 필수 파일이 루트에 있어야 합니다.
#  - package.json
#  - main.js
#  - index.html
# (선택) Windows 아이콘 파일: icon.ico
```

> **중요**: Windows에서 `C:\Windows\System32` 같은 시스템 폴더에서 명령을 실행하지 말고, **프로젝트 폴더**로 이동해서 작업하세요.

---

## 3) 의존성 설치

```bash
npm install
```

* 내부적으로 `electron`(런타임)과 `electron-builder`(패키징)가 설치됩니다.

---

## 4) 단순 실행

```bash
npm start
```

* 첫 실행 시 마이크 권한을 요청할 수 있습니다. 허용해 주세요.
* Space 키로 녹음을 시작/중지할 수 있습니다.
* 우측 상단 **투명도 슬라이더**로 자막 패널 투명도를 조절할 수 있습니다(로컬 저장됨).

---

## 5) 빌드 / 패키징 (exe 파일 만들기)

### Windows 설치 파일(NSIS) 만들기

```bash
# 모든 플랫폼 공통 빌드(설정에 따라 대상 생성)
npm run build

# Windows x64 전용 빌드
npm run build-win
```

* 산출물: `dist/` 폴더에 `.exe`(설치형) 또는 `.exe/.zip`(설정에 따라) 생성
* 설치 옵션: 설치 경로 선택, 바탕화면/시작 메뉴 바로가기 생성
* 앱 이름·아이콘 등은 `package.json`의 `build` 섹션으로 제어합니다.

> **아이콘**: Windows 전용 아이콘(`icon.ico`) 파일을 **프로젝트 루트**에 두세요. 없으면 기본 아이콘이 적용됩니다.

---

## 6) 구조 요약

```
project-root/
├─ package.json         # 스크립트/빌드 설정 (electron, electron-builder)
├─ main.js              # Electron 메인 프로세스: 투명·프레임리스·항상위 창
├─ index.html           # UI/로직: 녹음/침묵감지/서버 업로드/자막 렌더링
└─ icon.ico             # (선택) Windows 아이콘
```

---
