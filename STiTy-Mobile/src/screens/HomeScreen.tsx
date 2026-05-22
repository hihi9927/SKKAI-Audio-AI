import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  AppState,
  TouchableOpacity,
  Modal,
  Animated,
  Easing,
  StatusBar,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Circle, Path, Rect, Ellipse, Text as SvgText } from 'react-native-svg';
import { LinearGradient } from 'expo-linear-gradient';
import { LANGUAGES, Language, CONVERSATION_MODES } from '../constants/languages';
import { useWebSocketContext } from '../context/WebSocketContext';
import { useAudioRecording } from '../hooks/useAudioRecording';
import { initTtsEngine, ttsSpeak, ttsStop } from '../utils/tts';
import { setSpeakerphoneOn, releaseAudioMode, isEarphoneConnected } from '../utils/audioRouting';

// ─── Types ────────────────────────────────────────────────────────────────────
interface TranscriptionEntry {
  id: string;
  language: string;
  text: string;
  translatedText: string;
  timestamp: number;
}

// ─── UI string localizations ──────────────────────────────────────────────────
const UI_STRINGS: Record<string, {
  peerLanguage: string; mode: string;
  modes: Record<string, string>;
  collapse: string; configureAndStart: string; startSpeaking: string;
  listening: string; start: string; connectingServer: string; startingServer: string;
  stop: string; resume: string; back: string; cancel: string;
  endTitle: string; endMsg: string; end: string;
  errTitle: string; langMustDiffer: string; peerPickerTitle: string; connectionFailed: string;
  waitingForSlot: string; capacityFullMsg: string; capacityFreeMsg: string;
  restartApp: string;
  translation: string; fast: string; accurate: string;
  fastSub: string; accurateSub: string;
  fastDesc: string; accurateDesc: string;
  settingsTitle: string; settingsGeneral: string;
  settingsAbout: string; settingsAboutSub: string;
  settingsLegal: string; settingsPrivacy: string; settingsPrivacySub: string;
  settingsTerms: string; settingsTermsSub: string;
  settingsData: string; settingsDelete: string; settingsDeleteSub: string;
  settingsDisclaimer: string; settingsBack: string;
  privacyDate: string; privacyH1: string; privacyP1: string;
  privacyH2: string; privacyB2aStart: string; privacyB2aBold: string; privacyB2aEnd: string;
  privacyB2b: string; privacyB2c: string;
  privacyH3: string; privacyP3: string; privacyB3a: string; privacyB3b: string; privacyP3b: string;
  privacyH4: string; privacyP4: string; privacyH5: string; privacyP5: string;
  termsDate: string; termsH1: string; termsP1: string;
  termsH2: string; termsP2: string;
  termsH3: string; termsB3a: string; termsB3b: string; termsB3c: string;
  termsH4: string; termsP4: string; termsH5: string; termsP5: string;
  deleteTitle: string; deleteP1: string;
  deleteB1: string; deleteB2: string; deleteB3: string;
  deleteP2: string; deleteBtn: string;
  deleteAlertTitle: string; deleteAlertMsg: string;
  aboutP: string; aboutCopyright: string;
}> = {
  en: { peerLanguage: 'PEER LANGUAGE', mode: 'MODE', modes: { 'mode-1': 'Speaker', 'mode-2': 'Earphone', 'mode-3': 'Both' }, collapse: 'Collapse ▲', configureAndStart: 'Configure and tap Start', startSpeaking: 'Start speaking', listening: 'Listening...', start: 'Start', connectingServer: 'Connecting to server...', startingServer: 'Starting server...', stop: 'Stop', resume: 'Resume', back: 'Back', cancel: 'Cancel', endTitle: 'End conversation', endMsg: 'Do you want to end the conversation?', end: 'End', errTitle: 'Error', langMustDiffer: 'My language and peer language must be different.', peerPickerTitle: 'Peer Language', connectionFailed: 'Connection failed', waitingForSlot: 'Waiting for a slot…', capacityFullMsg: 'Server is full. Start will activate when a slot frees up.', capacityFreeMsg: 'A slot just opened — you can start now.', restartApp: 'Please restart the app.', translation: 'TRANSLATION', fast: 'Fast', accurate: 'Accurate', fastSub: 'Lower latency', accurateSub: 'Context-aware', fastDesc: 'Translates in near real time with low latency.', accurateDesc: 'Uses surrounding context to produce more natural translations.', settingsTitle: 'Settings', settingsGeneral: 'General', settingsAbout: 'About STiTy', settingsAboutSub: 'Version, credits', settingsLegal: 'Legal', settingsPrivacy: 'Privacy Policy', settingsPrivacySub: 'How we handle your voice & data', settingsTerms: 'Terms of Service', settingsTermsSub: 'Rules and limits of use', settingsData: 'Data', settingsDelete: 'Delete my data', settingsDeleteSub: 'Erase conversations & settings', settingsDisclaimer: '⚠ STiTy provides AI-generated translations and does not guarantee accuracy. Do not rely on it for medical, legal, or emergency communication.', settingsBack: '‹ Back', privacyDate: 'Last updated: May 17, 2026', privacyH1: '1. Voice data', privacyP1: 'STiTy records your voice through the device microphone only while a session is active. Audio is streamed to our translation servers, transcribed, and translated in real time.', privacyH2: '2. What we store', privacyB2aStart: '• Audio recordings are ', privacyB2aBold: 'not stored', privacyB2aEnd: ' after the session ends.', privacyB2b: '• Translated text transcripts are kept on your device only.', privacyB2c: '• We log anonymized session metadata (duration, languages used) for service quality.', privacyH3: '3. Third-party services', privacyP3: 'We use the following third parties to process audio:', privacyB3a: '• Google Cloud Speech-to-Text', privacyB3b: '• OpenAI / Anthropic translation APIs', privacyP3b: 'Their privacy policies apply to data they process.', privacyH4: '4. Your rights', privacyP4: 'You can delete all data tied to your device at any time via Settings → Delete my data. Contact privacy@stity.app for any request.', privacyH5: '5. Children', privacyP5: 'STiTy is not intended for users under 13.', termsDate: 'Last updated: May 17, 2026', termsH1: '1. Service description', termsP1: 'STiTy is a real-time speech translation tool. Translation quality depends on audio clarity, language pair, and AI model performance.', termsH2: '2. No guarantee', termsP2: 'Translations are produced automatically and may be inaccurate. Do not rely on STiTy for medical, legal, financial, or emergency communication.', termsH3: '3. Acceptable use', termsB3a: '• Do not record people without their consent.', termsB3b: '• Do not use STiTy to harass, defame, or deceive others.', termsB3c: '• Do not attempt to reverse-engineer the service.', termsH4: '4. Capacity limits', termsP4: 'STiTy currently supports up to 20 concurrent users. Sessions may be queued during peak times.', termsH5: '4. Termination', termsP5: 'We may suspend access at any time for policy violations.', deleteTitle: 'Delete my data', deleteP1: 'This will permanently erase:', deleteB1: '• All saved conversation transcripts on this device', deleteB2: '• Language preferences and settings', deleteB3: '• Anonymized usage metadata tied to your device ID', deleteP2: 'This action cannot be undone. Audio recordings are never stored, so nothing else remains on our servers.', deleteBtn: 'Permanently delete my data', deleteAlertTitle: 'Deleted', deleteAlertMsg: 'All local data erased.', aboutP: 'STiTy is a real-time speech translation app that lets people speak in their own languages and stay in one continuous conversation.', aboutCopyright: '© 2026 STiTy. All rights reserved.' },
  ko: { peerLanguage: '상대방 언어', mode: '모드', modes: { 'mode-1': '스피커', 'mode-2': '이어폰', 'mode-3': '둘 다' }, collapse: '접기 ▲', configureAndStart: '설정 후 시작을 누르세요', startSpeaking: '말씀해 주세요', listening: '듣는 중...', start: '시작', connectingServer: '서버 연결 중...', startingServer: '서버 시작 중...', stop: '중지', resume: '재개', back: '종료', cancel: '취소', endTitle: '대화 종료', endMsg: '대화를 종료하시겠습니까?', end: '종료', errTitle: '오류', langMustDiffer: '내 언어와 상대방 언어가 달라야 합니다.', peerPickerTitle: '상대방 언어', connectionFailed: '연결 실패', waitingForSlot: '슬롯 대기 중…', capacityFullMsg: '서버가 꽉 찼습니다. 슬롯이 생기면 시작됩니다.', capacityFreeMsg: '슬롯이 생겼어요 — 지금 시작하세요!', restartApp: '앱을 다시 시작해 주세요.', translation: '번역 설정', fast: '빠른 번역', accurate: '정확한 번역', fastSub: '낮은 지연', accurateSub: '문맥 반영', fastDesc: '낮은 지연으로 거의 실시간 번역합니다.', accurateDesc: '주변 문맥을 반영해 더 자연스러운 번역을 제공합니다.', settingsTitle: '설정', settingsGeneral: '일반', settingsAbout: 'STiTy 소개', settingsAboutSub: '버전, 크레딧', settingsLegal: '법적 고지', settingsPrivacy: '개인정보 처리방침', settingsPrivacySub: '음성 및 데이터 처리 방식', settingsTerms: '이용약관', settingsTermsSub: '이용 규칙 및 제한사항', settingsData: '데이터', settingsDelete: '내 데이터 삭제', settingsDeleteSub: '대화 및 설정 초기화', settingsDisclaimer: '⚠ STiTy는 AI 번역을 제공하며 정확성을 보장하지 않습니다. 의료, 법률, 응급 상황에는 사용하지 마세요.', settingsBack: '‹ 뒤로', privacyDate: '최종 업데이트: 2026년 5월 17일', privacyH1: '1. 음성 데이터', privacyP1: 'STiTy는 세션이 활성화된 동안에만 기기 마이크를 통해 음성을 녹음합니다. 음성은 번역 서버로 스트리밍되어 실시간으로 전사 및 번역됩니다.', privacyH2: '2. 저장 데이터', privacyB2aStart: '• 음성 녹음은 세션 종료 후 ', privacyB2aBold: '저장되지 않습니다', privacyB2aEnd: '.', privacyB2b: '• 번역된 텍스트 기록은 기기에만 저장됩니다.', privacyB2c: '• 서비스 품질을 위해 익명화된 세션 메타데이터(시간, 사용 언어)를 기록합니다.', privacyH3: '3. 제3자 서비스', privacyP3: '음성 처리를 위해 다음 제3자를 사용합니다:', privacyB3a: '• Google Cloud Speech-to-Text', privacyB3b: '• OpenAI / Anthropic 번역 API', privacyP3b: '이들의 개인정보 처리방침이 해당 데이터에 적용됩니다.', privacyH4: '4. 귀하의 권리', privacyP4: '설정 → 내 데이터 삭제를 통해 언제든지 기기에 연결된 모든 데이터를 삭제할 수 있습니다. 문의는 privacy@stity.app으로 연락하세요.', privacyH5: '5. 아동', privacyP5: 'STiTy는 13세 미만 사용자를 대상으로 하지 않습니다.', termsDate: '최종 업데이트: 2026년 5월 17일', termsH1: '1. 서비스 설명', termsP1: 'STiTy는 실시간 음성 번역 도구입니다. 번역 품질은 음성 명확도, 언어 쌍, AI 모델 성능에 따라 다릅니다.', termsH2: '2. 정확성 미보장', termsP2: '번역은 자동으로 생성되며 부정확할 수 있습니다. 의료, 법률, 금융, 응급 상황에 STiTy를 사용하지 마세요.', termsH3: '3. 허용되는 사용', termsB3a: '• 동의 없이 타인을 녹음하지 마세요.', termsB3b: '• STiTy를 이용해 타인을 괴롭히거나 비방하거나 속이지 마세요.', termsB3c: '• 서비스를 역공학적으로 분석하려 시도하지 마세요.', termsH4: '4. 용량 제한', termsP4: 'STiTy는 현재 최대 20명의 동시 사용자를 지원합니다. 사용량이 많은 시간에는 세션이 대기할 수 있습니다.', termsH5: '4. 서비스 중단', termsP5: '정책 위반 시 언제든지 접근을 차단할 수 있습니다.', deleteTitle: '내 데이터 삭제', deleteP1: '다음 항목이 영구적으로 삭제됩니다:', deleteB1: '• 이 기기에 저장된 모든 대화 기록', deleteB2: '• 언어 설정 및 기타 설정', deleteB3: '• 기기 ID에 연결된 익명화된 사용 메타데이터', deleteP2: '이 작업은 취소할 수 없습니다. 음성 녹음은 저장되지 않으므로 서버에 남은 데이터가 없습니다.', deleteBtn: '내 데이터 영구 삭제', deleteAlertTitle: '삭제 완료', deleteAlertMsg: '모든 로컬 데이터가 삭제되었습니다.', aboutP: 'STiTy는 각자의 언어로 자유롭게 대화할 수 있도록 해주는 실시간 음성 번역 앱입니다.', aboutCopyright: '© 2026 STiTy. All rights reserved.' },
  ja: { peerLanguage: '相手の言語', mode: 'モード', modes: { 'mode-1': 'スピーカー', 'mode-2': 'イヤホン', 'mode-3': '両方' }, collapse: '閉じる ▲', configureAndStart: '設定して開始をタップ', startSpeaking: '話してください', listening: '聴いています...', start: '開始', connectingServer: 'サーバー接続中...', startingServer: 'サーバー起動中...', stop: '停止', resume: '再開', back: '終了', cancel: 'キャンセル', endTitle: '会話を終了', endMsg: '会話を終了しますか？', end: '終了', errTitle: 'エラー', langMustDiffer: '自分と相手の言語が異なる必要があります。', peerPickerTitle: '相手の言語', connectionFailed: '接続失敗', waitingForSlot: 'スロット待機中…', capacityFullMsg: 'サーバーが満員です。空き次第、開始できます。', capacityFreeMsg: 'スロットが空きました — 開始できます。', restartApp: 'アプリを再起動してください。', translation: '翻訳設定', fast: '高速翻訳', accurate: '精密翻訳', fastSub: '低遅延', accurateSub: '文脈対応', fastDesc: '低遅延でほぼリアルタイムに翻訳します。', accurateDesc: '前後の文脈を活用し、より自然な翻訳を提供します。', settingsTitle: '設定', settingsGeneral: '一般', settingsAbout: 'STiTyについて', settingsAboutSub: 'バージョン・クレジット', settingsLegal: '法的情報', settingsPrivacy: 'プライバシーポリシー', settingsPrivacySub: '音声とデータの取り扱い', settingsTerms: '利用規約', settingsTermsSub: '利用ルールと制限事項', settingsData: 'データ', settingsDelete: 'データを削除', settingsDeleteSub: '会話と設定を消去', settingsDisclaimer: '⚠ STiTyはAI翻訳を提供しており、正確性を保証しません。医療・法律・緊急時の通信には使用しないでください。', settingsBack: '‹ 戻る', privacyDate: '最終更新: 2026年5月17日', privacyH1: '1. 音声データ', privacyP1: 'STiTyは、セッションがアクティブな間のみ、デバイスのマイクを通じて音声を録音します。音声は翻訳サーバーにストリーミングされ、リアルタイムで文字起こしおよび翻訳されます。', privacyH2: '2. 保存データ', privacyB2aStart: '• 音声録音はセッション終了後、', privacyB2aBold: '保存されません', privacyB2aEnd: '。', privacyB2b: '• 翻訳済みテキストはデバイス上にのみ保存されます。', privacyB2c: '• サービス品質向上のため、匿名化されたセッションメタデータ（所要時間、使用言語）を記録します。', privacyH3: '3. 第三者サービス', privacyP3: '音声処理に以下の第三者を利用しています:', privacyB3a: '• Google Cloud Speech-to-Text', privacyB3b: '• OpenAI / Anthropic 翻訳API', privacyP3b: '各社のプライバシーポリシーが、処理されるデータに適用されます。', privacyH4: '4. お客様の権利', privacyP4: '設定 → データを削除 から、いつでもデバイスに紐づくすべてのデータを削除できます。ご要望はprivacy@stity.appまでお問い合わせください。', privacyH5: '5. 子どもの利用', privacyP5: 'STiTyは13歳未満のユーザーを対象としていません。', termsDate: '最終更新: 2026年5月17日', termsH1: '1. サービス説明', termsP1: 'STiTyはリアルタイム音声翻訳ツールです。翻訳品質は音声の明瞭さ、言語ペア、AIモデルの性能によって異なります。', termsH2: '2. 保証なし', termsP2: '翻訳は自動生成されるため、不正確な場合があります。医療・法律・金融・緊急の通信にSTiTyを使用しないでください。', termsH3: '3. 適切な使用', termsB3a: '• 同意なしに他者を録音しないでください。', termsB3b: '• STiTyを使用して他者を嫌がらせ、中傷、または欺かないでください。', termsB3c: '• サービスのリバースエンジニアリングを試みないでください。', termsH4: '4. 利用制限', termsP4: 'STiTyは現在、最大20名の同時利用者をサポートしています。ピーク時にはセッションが待機する場合があります。', termsH5: '4. 利用停止', termsP5: 'ポリシー違反があった場合、いつでもアクセスを停止することがあります。', deleteTitle: 'データを削除', deleteP1: '以下のデータが完全に消去されます:', deleteB1: '• このデバイスに保存されたすべての会話記録', deleteB2: '• 言語設定と各種設定', deleteB3: '• デバイスIDに紐づく匿名化された利用メタデータ', deleteP2: 'この操作は取り消せません。音声録音は保存されないため、サーバー上にデータは残りません。', deleteBtn: 'データを完全に削除', deleteAlertTitle: '削除完了', deleteAlertMsg: 'すべてのローカルデータが消去されました。', aboutP: 'STiTyは、それぞれが自分の言語で話せ、ひとつの会話を続けられるリアルタイム音声翻訳アプリです。', aboutCopyright: '© 2026 STiTy. All rights reserved.' },
  zh: { peerLanguage: '对方语言', mode: '模式', modes: { 'mode-1': '扬声器', 'mode-2': '耳机', 'mode-3': '两者' }, collapse: '收起 ▲', configureAndStart: '设置后点击开始', startSpeaking: '请开始说话', listening: '正在聆听...', start: '开始', connectingServer: '正在连接服务器...', startingServer: '正在启动服务器...', stop: '停止', resume: '继续', back: '结束', cancel: '取消', endTitle: '结束对话', endMsg: '确定要结束对话吗？', end: '结束', errTitle: '错误', langMustDiffer: '我的语言和对方语言必须不同。', peerPickerTitle: '对方语言', connectionFailed: '连接失败', waitingForSlot: '等待空位…', capacityFullMsg: '服务器已满，有空位时即可开始。', capacityFreeMsg: '有空位了 — 现在可以开始。', restartApp: '请重新启动应用。', translation: '翻译设置', fast: '快速翻译', accurate: '精准翻译', fastSub: '低延迟', accurateSub: '结合语境', fastDesc: '以低延迟实现近实时翻译。', accurateDesc: '结合上下文语境，提供更自然流畅的翻译。', settingsTitle: '设置', settingsGeneral: '通用', settingsAbout: '关于 STiTy', settingsAboutSub: '版本与鸣谢', settingsLegal: '法律信息', settingsPrivacy: '隐私政策', settingsPrivacySub: '我们如何处理您的语音与数据', settingsTerms: '服务条款', settingsTermsSub: '使用规则与限制', settingsData: '数据', settingsDelete: '删除我的数据', settingsDeleteSub: '清除对话记录与设置', settingsDisclaimer: '⚠ STiTy 提供 AI 生成的翻译，不保证准确性。请勿用于医疗、法律或紧急通信。', settingsBack: '‹ 返回', privacyDate: '最后更新：2026年5月17日', privacyH1: '1. 语音数据', privacyP1: 'STiTy 仅在会话期间通过设备麦克风录制您的语音。音频将流式传输至翻译服务器，并实时转录和翻译。', privacyH2: '2. 存储数据', privacyB2aStart: '• 音频录音在会话结束后', privacyB2aBold: '不会被存储', privacyB2aEnd: '。', privacyB2b: '• 翻译后的文字记录仅保存在您的设备上。', privacyB2c: '• 我们记录匿名化的会话元数据（时长、所用语言）以改善服务质量。', privacyH3: '3. 第三方服务', privacyP3: '我们使用以下第三方来处理音频：', privacyB3a: '• Google Cloud 语音转文字', privacyB3b: '• OpenAI / Anthropic 翻译 API', privacyP3b: '这些第三方的隐私政策适用于其所处理的数据。', privacyH4: '4. 您的权利', privacyP4: '您可随时通过 设置 → 删除我的数据 删除与您设备绑定的所有数据。如需帮助，请联系 privacy@stity.app。', privacyH5: '5. 儿童', privacyP5: 'STiTy 不面向 13 岁以下用户。', termsDate: '最后更新：2026年5月17日', termsH1: '1. 服务说明', termsP1: 'STiTy 是一款实时语音翻译工具。翻译质量取决于音频清晰度、语言对及 AI 模型性能。', termsH2: '2. 不作保证', termsP2: '翻译由系统自动生成，可能存在不准确之处。请勿将 STiTy 用于医疗、法律、金融或紧急通信。', termsH3: '3. 合规使用', termsB3a: '• 未经他人同意，请勿录制他人。', termsB3b: '• 请勿利用 STiTy 骚扰、诽谤或欺骗他人。', termsB3c: '• 请勿尝试对服务进行逆向工程。', termsH4: '4. 容量限制', termsP4: 'STiTy 目前最多支持 20 名并发用户。高峰期间会话可能需要排队等待。', termsH5: '4. 终止服务', termsP5: '违反政策时，我们可随时暂停您的访问权限。', deleteTitle: '删除我的数据', deleteP1: '以下内容将被永久删除：', deleteB1: '• 本设备上保存的所有对话记录', deleteB2: '• 语言偏好及相关设置', deleteB3: '• 与您设备 ID 绑定的匿名化使用元数据', deleteP2: '此操作无法撤销。由于音频录音从不存储，服务器上不会保留任何数据。', deleteBtn: '永久删除我的数据', deleteAlertTitle: '已删除', deleteAlertMsg: '所有本地数据已清除。', aboutP: 'STiTy 是一款实时语音翻译应用，让人们用各自的语言自由交流，融入同一段对话。', aboutCopyright: '© 2026 STiTy. 保留所有权利。' },
  es: { peerLanguage: 'IDIOMA DEL OTRO', mode: 'MODO', modes: { 'mode-1': 'Altavoz', 'mode-2': 'Auricular', 'mode-3': 'Ambos' }, collapse: 'Contraer ▲', configureAndStart: 'Configura y pulsa Iniciar', startSpeaking: 'Empieza a hablar', listening: 'Escuchando...', start: 'Iniciar', connectingServer: 'Conectando al servidor...', startingServer: 'Iniciando servidor...', stop: 'Detener', resume: 'Reanudar', back: 'Volver', cancel: 'Cancelar', endTitle: 'Finalizar conversación', endMsg: '¿Quieres finalizar la conversación?', end: 'Finalizar', errTitle: 'Error', langMustDiffer: 'Mi idioma y el del otro deben ser distintos.', peerPickerTitle: 'Idioma del otro', connectionFailed: 'Conexión fallida', waitingForSlot: 'Esperando espacio…', capacityFullMsg: 'Servidor lleno. Inicio disponible al liberarse un espacio.', capacityFreeMsg: '¡Hay espacio libre — puedes iniciar!', restartApp: 'Por favor, reinicia la aplicación.', translation: 'TRADUCCIÓN', fast: 'Rápido', accurate: 'Preciso', fastSub: 'Baja latencia', accurateSub: 'Con contexto', fastDesc: 'Traduce casi en tiempo real con baja latencia.', accurateDesc: 'Usa el contexto circundante para producir traducciones más naturales.', settingsTitle: 'Ajustes', settingsGeneral: 'General', settingsAbout: 'Acerca de STiTy', settingsAboutSub: 'Versión, créditos', settingsLegal: 'Legal', settingsPrivacy: 'Política de privacidad', settingsPrivacySub: 'Cómo gestionamos tu voz y datos', settingsTerms: 'Términos de servicio', settingsTermsSub: 'Reglas y límites de uso', settingsData: 'Datos', settingsDelete: 'Eliminar mis datos', settingsDeleteSub: 'Borrar conversaciones y ajustes', settingsDisclaimer: '⚠ STiTy ofrece traducciones generadas por IA y no garantiza su exactitud. No lo uses para comunicaciones médicas, legales o de emergencia.', settingsBack: '‹ Volver', privacyDate: 'Última actualización: 17 de mayo de 2026', privacyH1: '1. Datos de voz', privacyP1: 'STiTy graba tu voz a través del micrófono del dispositivo solo mientras la sesión está activa. El audio se transmite a nuestros servidores de traducción, se transcribe y se traduce en tiempo real.', privacyH2: '2. Qué almacenamos', privacyB2aStart: '• Las grabaciones de audio ', privacyB2aBold: 'no se almacenan', privacyB2aEnd: ' una vez finalizada la sesión.', privacyB2b: '• Las transcripciones de texto traducido se guardan únicamente en tu dispositivo.', privacyB2c: '• Registramos metadatos de sesión anonimizados (duración, idiomas usados) para la calidad del servicio.', privacyH3: '3. Servicios de terceros', privacyP3: 'Usamos los siguientes terceros para procesar el audio:', privacyB3a: '• Google Cloud Speech-to-Text', privacyB3b: '• APIs de traducción de OpenAI / Anthropic', privacyP3b: 'Sus políticas de privacidad se aplican a los datos que procesan.', privacyH4: '4. Tus derechos', privacyP4: 'Puedes eliminar todos los datos vinculados a tu dispositivo en cualquier momento desde Ajustes → Eliminar mis datos. Contacta privacy@stity.app para cualquier solicitud.', privacyH5: '5. Menores', privacyP5: 'STiTy no está destinado a usuarios menores de 13 años.', termsDate: 'Última actualización: 17 de mayo de 2026', termsH1: '1. Descripción del servicio', termsP1: 'STiTy es una herramienta de traducción de voz en tiempo real. La calidad de la traducción depende de la claridad del audio, el par de idiomas y el rendimiento del modelo de IA.', termsH2: '2. Sin garantía', termsP2: 'Las traducciones se generan automáticamente y pueden ser inexactas. No confíes en STiTy para comunicaciones médicas, legales, financieras o de emergencia.', termsH3: '3. Uso aceptable', termsB3a: '• No grabes a personas sin su consentimiento.', termsB3b: '• No uses STiTy para acosar, difamar o engañar a otros.', termsB3c: '• No intentes realizar ingeniería inversa del servicio.', termsH4: '4. Límites de capacidad', termsP4: 'STiTy admite actualmente hasta 20 usuarios simultáneos. Las sesiones pueden ponerse en cola en horas punta.', termsH5: '4. Rescisión', termsP5: 'Podemos suspender el acceso en cualquier momento por infracciones de la política.', deleteTitle: 'Eliminar mis datos', deleteP1: 'Se eliminará permanentemente:', deleteB1: '• Todos los registros de conversación guardados en este dispositivo', deleteB2: '• Preferencias de idioma y ajustes', deleteB3: '• Metadatos de uso anonimizados vinculados al ID de tu dispositivo', deleteP2: 'Esta acción no se puede deshacer. Las grabaciones de audio nunca se almacenan, por lo que no queda nada en nuestros servidores.', deleteBtn: 'Eliminar mis datos permanentemente', deleteAlertTitle: 'Eliminado', deleteAlertMsg: 'Todos los datos locales han sido borrados.', aboutP: 'STiTy es una aplicación de traducción de voz en tiempo real que permite a las personas hablar en sus propios idiomas y mantener una conversación continua.', aboutCopyright: '© 2026 STiTy. Todos los derechos reservados.' },
  fr: { peerLanguage: "LANGUE DE L'AUTRE", mode: 'MODE', modes: { 'mode-1': 'Haut-parleur', 'mode-2': 'Écouteur', 'mode-3': 'Les deux' }, collapse: 'Réduire ▲', configureAndStart: 'Configurez et appuyez sur Démarrer', startSpeaking: 'Commencez à parler', listening: 'En écoute...', start: 'Démarrer', connectingServer: 'Connexion au serveur...', startingServer: 'Démarrage du serveur...', stop: 'Arrêter', resume: 'Reprendre', back: 'Retour', cancel: 'Annuler', endTitle: 'Terminer la conversation', endMsg: 'Voulez-vous terminer la conversation ?', end: 'Terminer', errTitle: 'Erreur', langMustDiffer: "Ma langue et celle de l'autre doivent être différentes.", peerPickerTitle: "Langue de l'autre", connectionFailed: 'Connexion échouée', waitingForSlot: 'En attente…', capacityFullMsg: "Serveur plein. Démarrage possible dès qu'un créneau se libère.", capacityFreeMsg: 'Un créneau est libre — commencez !', restartApp: "Veuillez redémarrer l'application.", translation: 'TRADUCTION', fast: 'Rapide', accurate: 'Précis', fastSub: 'Faible latence', accurateSub: 'Contextuel', fastDesc: 'Traduit en quasi temps réel avec une faible latence.', accurateDesc: 'Exploite le contexte environnant pour des traductions plus naturelles.', settingsTitle: 'Paramètres', settingsGeneral: 'Général', settingsAbout: 'À propos de STiTy', settingsAboutSub: 'Version, crédits', settingsLegal: 'Mentions légales', settingsPrivacy: 'Politique de confidentialité', settingsPrivacySub: 'Comment nous gérons votre voix et vos données', settingsTerms: "Conditions d'utilisation", settingsTermsSub: "Règles et limites d'utilisation", settingsData: 'Données', settingsDelete: 'Supprimer mes données', settingsDeleteSub: 'Effacer conversations et paramètres', settingsDisclaimer: "⚠ STiTy fournit des traductions générées par IA et n'en garantit pas l'exactitude. Ne l'utilisez pas pour des communications médicales, juridiques ou d'urgence.", settingsBack: '‹ Retour', privacyDate: 'Dernière mise à jour : 17 mai 2026', privacyH1: '1. Données vocales', privacyP1: "STiTy enregistre votre voix via le microphone de l'appareil uniquement lorsqu'une session est active. L'audio est transmis en streaming à nos serveurs de traduction, transcrit et traduit en temps réel.", privacyH2: '2. Ce que nous stockons', privacyB2aStart: '• Les enregistrements audio ne sont ', privacyB2aBold: 'pas conservés', privacyB2aEnd: ' après la fin de la session.', privacyB2b: '• Les transcriptions traduites sont conservées uniquement sur votre appareil.', privacyB2c: '• Nous enregistrons des métadonnées de session anonymisées (durée, langues utilisées) pour la qualité du service.', privacyH3: '3. Services tiers', privacyP3: "Nous faisons appel aux prestataires suivants pour traiter l'audio :", privacyB3a: '• Google Cloud Speech-to-Text', privacyB3b: '• API de traduction OpenAI / Anthropic', privacyP3b: "Leurs politiques de confidentialité s'appliquent aux données qu'ils traitent.", privacyH4: '4. Vos droits', privacyP4: 'Vous pouvez supprimer toutes les données liées à votre appareil à tout moment via Paramètres → Supprimer mes données. Contactez privacy@stity.app pour toute demande.', privacyH5: '5. Enfants', privacyP5: "STiTy n'est pas destiné aux utilisateurs de moins de 13 ans.", termsDate: 'Dernière mise à jour : 17 mai 2026', termsH1: '1. Description du service', termsP1: "STiTy est un outil de traduction vocale en temps réel. La qualité de traduction dépend de la clarté audio, de la paire de langues et des performances du modèle d'IA.", termsH2: '2. Aucune garantie', termsP2: "Les traductions sont produites automatiquement et peuvent être inexactes. Ne vous fiez pas à STiTy pour des communications médicales, juridiques, financières ou d'urgence.", termsH3: '3. Usage acceptable', termsB3a: '• Ne pas enregistrer des personnes sans leur consentement.', termsB3b: '• Ne pas utiliser STiTy pour harceler, diffamer ou tromper autrui.', termsB3c: '• Ne pas tenter de désassembler ou rétroconcevoir le service.', termsH4: '4. Limites de capacité', termsP4: "STiTy supporte actuellement jusqu'à 20 utilisateurs simultanés. Les sessions peuvent être mises en file d'attente aux heures de pointe.", termsH5: '4. Résiliation', termsP5: "Nous pouvons suspendre l'accès à tout moment en cas de violation de la politique.", deleteTitle: 'Supprimer mes données', deleteP1: 'Les éléments suivants seront définitivement effacés :', deleteB1: '• Toutes les transcriptions de conversations enregistrées sur cet appareil', deleteB2: '• Préférences de langue et paramètres', deleteB3: "• Métadonnées d'utilisation anonymisées liées à l'identifiant de votre appareil", deleteP2: "Cette action est irréversible. Les enregistrements audio ne sont jamais stockés, il ne reste donc rien sur nos serveurs.", deleteBtn: 'Supprimer définitivement mes données', deleteAlertTitle: 'Supprimé', deleteAlertMsg: 'Toutes les données locales ont été effacées.', aboutP: "STiTy est une application de traduction vocale en temps réel qui permet à chacun de parler dans sa propre langue tout en participant à une même conversation continue.", aboutCopyright: '© 2026 STiTy. Tous droits réservés.' },
  id: { peerLanguage: 'BAHASA MITRA', mode: 'MODE', modes: { 'mode-1': 'Speaker', 'mode-2': 'Earphone', 'mode-3': 'Keduanya' }, collapse: 'Tutup ▲', configureAndStart: 'Atur dan ketuk Mulai', startSpeaking: 'Mulai berbicara', listening: 'Mendengarkan...', start: 'Mulai', connectingServer: 'Menghubungkan ke server...', startingServer: 'Memulai server...', stop: 'Berhenti', resume: 'Lanjutkan', back: 'Kembali', cancel: 'Batal', endTitle: 'Akhiri percakapan', endMsg: 'Apakah Anda ingin mengakhiri percakapan?', end: 'Akhiri', errTitle: 'Kesalahan', langMustDiffer: 'Bahasa saya dan bahasa mitra harus berbeda.', peerPickerTitle: 'Bahasa Mitra', connectionFailed: 'Koneksi gagal', waitingForSlot: 'Menunggu slot…', capacityFullMsg: 'Server penuh. Mulai tersedia saat ada slot kosong.', capacityFreeMsg: 'Slot tersedia — mulai sekarang!', restartApp: 'Silakan mulai ulang aplikasi.', translation: 'TERJEMAHAN', fast: 'Cepat', accurate: 'Akurat', fastSub: 'Latensi rendah', accurateSub: 'Berbasis konteks', fastDesc: 'Menerjemahkan hampir secara real time dengan latensi rendah.', accurateDesc: 'Menggunakan konteks sekitar untuk menghasilkan terjemahan yang lebih alami.', settingsTitle: 'Pengaturan', settingsGeneral: 'Umum', settingsAbout: 'Tentang STiTy', settingsAboutSub: 'Versi, kredit', settingsLegal: 'Hukum', settingsPrivacy: 'Kebijakan Privasi', settingsPrivacySub: 'Cara kami mengelola suara & data Anda', settingsTerms: 'Syarat Layanan', settingsTermsSub: 'Aturan dan batasan penggunaan', settingsData: 'Data', settingsDelete: 'Hapus data saya', settingsDeleteSub: 'Hapus percakapan & pengaturan', settingsDisclaimer: '⚠ STiTy menyediakan terjemahan yang dihasilkan AI dan tidak menjamin keakuratannya. Jangan gunakan untuk komunikasi medis, hukum, atau darurat.', settingsBack: '‹ Kembali', privacyDate: 'Terakhir diperbarui: 17 Mei 2026', privacyH1: '1. Data suara', privacyP1: 'STiTy merekam suara Anda melalui mikrofon perangkat hanya saat sesi aktif. Audio dialirkan ke server terjemahan kami, ditranskripsikan, dan diterjemahkan secara real time.', privacyH2: '2. Yang kami simpan', privacyB2aStart: '• Rekaman audio ', privacyB2aBold: 'tidak disimpan', privacyB2aEnd: ' setelah sesi berakhir.', privacyB2b: '• Transkrip teks terjemahan hanya disimpan di perangkat Anda.', privacyB2c: '• Kami mencatat metadata sesi yang dianonimkan (durasi, bahasa yang digunakan) untuk kualitas layanan.', privacyH3: '3. Layanan pihak ketiga', privacyP3: 'Kami menggunakan pihak ketiga berikut untuk memproses audio:', privacyB3a: '• Google Cloud Speech-to-Text', privacyB3b: '• API terjemahan OpenAI / Anthropic', privacyP3b: 'Kebijakan privasi mereka berlaku untuk data yang mereka proses.', privacyH4: '4. Hak Anda', privacyP4: 'Anda dapat menghapus semua data yang terkait dengan perangkat Anda kapan saja melalui Pengaturan → Hapus data saya. Hubungi privacy@stity.app untuk permintaan apapun.', privacyH5: '5. Anak-anak', privacyP5: 'STiTy tidak ditujukan untuk pengguna di bawah 13 tahun.', termsDate: 'Terakhir diperbarui: 17 Mei 2026', termsH1: '1. Deskripsi layanan', termsP1: 'STiTy adalah alat terjemahan ucapan secara real time. Kualitas terjemahan bergantung pada kejernihan audio, pasangan bahasa, dan kinerja model AI.', termsH2: '2. Tidak ada jaminan', termsP2: 'Terjemahan dihasilkan secara otomatis dan mungkin tidak akurat. Jangan mengandalkan STiTy untuk komunikasi medis, hukum, keuangan, atau darurat.', termsH3: '3. Penggunaan yang diizinkan', termsB3a: '• Jangan merekam orang tanpa persetujuan mereka.', termsB3b: '• Jangan gunakan STiTy untuk melecehkan, mencemarkan nama baik, atau menipu orang lain.', termsB3c: '• Jangan mencoba merekayasa balik layanan ini.', termsH4: '4. Batas kapasitas', termsP4: 'STiTy saat ini mendukung hingga 20 pengguna serentak. Sesi mungkin antri saat jam sibuk.', termsH5: '4. Penghentian', termsP5: 'Kami dapat menangguhkan akses kapan saja atas pelanggaran kebijakan.', deleteTitle: 'Hapus data saya', deleteP1: 'Ini akan menghapus secara permanen:', deleteB1: '• Semua transkrip percakapan yang tersimpan di perangkat ini', deleteB2: '• Preferensi bahasa dan pengaturan', deleteB3: '• Metadata penggunaan yang dianonimkan terkait ID perangkat Anda', deleteP2: 'Tindakan ini tidak dapat dibatalkan. Rekaman audio tidak pernah disimpan, sehingga tidak ada yang tersisa di server kami.', deleteBtn: 'Hapus data saya secara permanen', deleteAlertTitle: 'Dihapus', deleteAlertMsg: 'Semua data lokal telah dihapus.', aboutP: 'STiTy adalah aplikasi terjemahan ucapan real time yang memungkinkan orang berbicara dalam bahasa masing-masing dan tetap terlibat dalam satu percakapan yang berkesinambungan.', aboutCopyright: '© 2026 STiTy. Semua hak dilindungi.' },
  vi: { peerLanguage: 'NGÔN NGỮ ĐỐI TÁC', mode: 'CHẾ ĐỘ', modes: { 'mode-1': 'Loa ngoài', 'mode-2': 'Tai nghe', 'mode-3': 'Cả hai' }, collapse: 'Thu gọn ▲', configureAndStart: 'Cài đặt và nhấn Bắt đầu', startSpeaking: 'Bắt đầu nói', listening: 'Đang nghe...', start: 'Bắt đầu', connectingServer: 'Đang kết nối máy chủ...', startingServer: 'Đang khởi động máy chủ...', stop: 'Dừng', resume: 'Tiếp tục', back: 'Quay lại', cancel: 'Hủy', endTitle: 'Kết thúc cuộc trò chuyện', endMsg: 'Bạn có muốn kết thúc cuộc trò chuyện không?', end: 'Kết thúc', errTitle: 'Lỗi', langMustDiffer: 'Ngôn ngữ của tôi và đối tác phải khác nhau.', peerPickerTitle: 'Ngôn ngữ đối tác', connectionFailed: 'Kết nối thất bại', waitingForSlot: 'Đang chờ chỗ trống…', capacityFullMsg: 'Máy chủ đầy. Sẽ bắt đầu khi có chỗ trống.', capacityFreeMsg: 'Có chỗ trống rồi — bắt đầu ngay!', restartApp: 'Vui lòng khởi động lại ứng dụng.', translation: 'DỊCH THUẬT', fast: 'Nhanh', accurate: 'Chính xác', fastSub: 'Độ trễ thấp', accurateSub: 'Dựa trên ngữ cảnh', fastDesc: 'Dịch gần như thời gian thực với độ trễ thấp.', accurateDesc: 'Dựa vào ngữ cảnh xung quanh để tạo ra bản dịch tự nhiên hơn.', settingsTitle: 'Cài đặt', settingsGeneral: 'Chung', settingsAbout: 'Giới thiệu STiTy', settingsAboutSub: 'Phiên bản, ghi công', settingsLegal: 'Pháp lý', settingsPrivacy: 'Chính sách quyền riêng tư', settingsPrivacySub: 'Cách chúng tôi xử lý giọng nói và dữ liệu của bạn', settingsTerms: 'Điều khoản dịch vụ', settingsTermsSub: 'Quy tắc và giới hạn sử dụng', settingsData: 'Dữ liệu', settingsDelete: 'Xóa dữ liệu của tôi', settingsDeleteSub: 'Xóa cuộc trò chuyện và cài đặt', settingsDisclaimer: '⚠ STiTy cung cấp bản dịch do AI tạo ra và không đảm bảo độ chính xác. Không sử dụng cho thông tin liên lạc y tế, pháp lý hoặc khẩn cấp.', settingsBack: '‹ Quay lại', privacyDate: 'Cập nhật lần cuối: 17 tháng 5 năm 2026', privacyH1: '1. Dữ liệu giọng nói', privacyP1: 'STiTy ghi âm giọng nói của bạn qua micrô của thiết bị chỉ khi phiên đang hoạt động. Âm thanh được phát trực tuyến đến máy chủ dịch của chúng tôi, phiên âm và dịch theo thời gian thực.', privacyH2: '2. Những gì chúng tôi lưu trữ', privacyB2aStart: '• Bản ghi âm ', privacyB2aBold: 'không được lưu trữ', privacyB2aEnd: ' sau khi phiên kết thúc.', privacyB2b: '• Bản ghi văn bản đã dịch chỉ được lưu trên thiết bị của bạn.', privacyB2c: '• Chúng tôi ghi lại siêu dữ liệu phiên được ẩn danh (thời lượng, ngôn ngữ sử dụng) để cải thiện chất lượng dịch vụ.', privacyH3: '3. Dịch vụ bên thứ ba', privacyP3: 'Chúng tôi sử dụng các bên thứ ba sau để xử lý âm thanh:', privacyB3a: '• Google Cloud Speech-to-Text', privacyB3b: '• API dịch thuật OpenAI / Anthropic', privacyP3b: 'Chính sách quyền riêng tư của họ áp dụng cho dữ liệu họ xử lý.', privacyH4: '4. Quyền của bạn', privacyP4: 'Bạn có thể xóa tất cả dữ liệu liên kết với thiết bị của mình bất kỳ lúc nào qua Cài đặt → Xóa dữ liệu của tôi. Liên hệ privacy@stity.app cho bất kỳ yêu cầu nào.', privacyH5: '5. Trẻ em', privacyP5: 'STiTy không dành cho người dùng dưới 13 tuổi.', termsDate: 'Cập nhật lần cuối: 17 tháng 5 năm 2026', termsH1: '1. Mô tả dịch vụ', termsP1: 'STiTy là công cụ dịch giọng nói theo thời gian thực. Chất lượng dịch phụ thuộc vào độ rõ ràng của âm thanh, cặp ngôn ngữ và hiệu suất mô hình AI.', termsH2: '2. Không bảo đảm', termsP2: 'Bản dịch được tạo tự động và có thể không chính xác. Không dựa vào STiTy cho các thông tin liên lạc y tế, pháp lý, tài chính hoặc khẩn cấp.', termsH3: '3. Sử dụng chấp nhận được', termsB3a: '• Không ghi âm người khác khi không có sự đồng ý của họ.', termsB3b: '• Không sử dụng STiTy để quấy rối, phỉ báng hoặc lừa dối người khác.', termsB3c: '• Không cố gắng dịch ngược kỹ thuật dịch vụ.', termsH4: '4. Giới hạn công suất', termsP4: 'STiTy hiện hỗ trợ tối đa 20 người dùng đồng thời. Các phiên có thể xếp hàng trong giờ cao điểm.', termsH5: '4. Chấm dứt', termsP5: 'Chúng tôi có thể đình chỉ quyền truy cập bất kỳ lúc nào vì vi phạm chính sách.', deleteTitle: 'Xóa dữ liệu của tôi', deleteP1: 'Thao tác này sẽ xóa vĩnh viễn:', deleteB1: '• Tất cả bản ghi cuộc trò chuyện đã lưu trên thiết bị này', deleteB2: '• Tùy chọn ngôn ngữ và cài đặt', deleteB3: '• Siêu dữ liệu sử dụng được ẩn danh liên kết với ID thiết bị của bạn', deleteP2: 'Hành động này không thể hoàn tác. Bản ghi âm không bao giờ được lưu trữ, vì vậy không còn gì trên máy chủ của chúng tôi.', deleteBtn: 'Xóa vĩnh viễn dữ liệu của tôi', deleteAlertTitle: 'Đã xóa', deleteAlertMsg: 'Tất cả dữ liệu cục bộ đã được xóa.', aboutP: 'STiTy là ứng dụng dịch giọng nói theo thời gian thực, cho phép mọi người nói bằng ngôn ngữ của mình và duy trì một cuộc trò chuyện liên tục.', aboutCopyright: '© 2026 STiTy. Bảo lưu mọi quyền.' },
  th: { peerLanguage: 'ภาษาของคู่สนทนา', mode: 'โหมด', modes: { 'mode-1': 'ลำโพง', 'mode-2': 'หูฟัง', 'mode-3': 'ทั้งสอง' }, collapse: 'ย่อ ▲', configureAndStart: 'ตั้งค่าแล้วแตะเริ่มต้น', startSpeaking: 'เริ่มพูด', listening: 'กำลังฟัง...', start: 'เริ่มต้น', connectingServer: 'กำลังเชื่อมต่อเซิร์ฟเวอร์...', startingServer: 'กำลังเริ่มเซิร์ฟเวอร์...', stop: 'หยุด', resume: 'ดำเนินการต่อ', back: 'กลับ', cancel: 'ยกเลิก', endTitle: 'สิ้นสุดการสนทนา', endMsg: 'คุณต้องการสิ้นสุดการสนทนาหรือไม่?', end: 'สิ้นสุด', errTitle: 'ข้อผิดพลาด', langMustDiffer: 'ภาษาของฉันและคู่สนทนาต้องแตกต่างกัน', peerPickerTitle: 'ภาษาของคู่สนทนา', connectionFailed: 'การเชื่อมต่อล้มเหลว', waitingForSlot: 'รอช่อง…', capacityFullMsg: 'เซิร์ฟเวอร์เต็ม จะเริ่มได้เมื่อมีช่องว่าง', capacityFreeMsg: 'มีช่องว่างแล้ว — เริ่มได้เลย!', restartApp: 'กรุณารีสตาร์ทแอป', translation: 'การแปล', fast: 'เร็ว', accurate: 'แม่นยำ', fastSub: 'ความหน่วงต่ำ', accurateSub: 'คำนึงถึงบริบท', fastDesc: 'แปลแบบเกือบเรียลไทม์ด้วยความหน่วงต่ำ', accurateDesc: 'ใช้บริบทโดยรอบเพื่อสร้างการแปลที่เป็นธรรมชาติยิ่งขึ้น', settingsTitle: 'การตั้งค่า', settingsGeneral: 'ทั่วไป', settingsAbout: 'เกี่ยวกับ STiTy', settingsAboutSub: 'เวอร์ชัน, เครดิต', settingsLegal: 'กฎหมาย', settingsPrivacy: 'นโยบายความเป็นส่วนตัว', settingsPrivacySub: 'วิธีที่เราจัดการเสียงและข้อมูลของคุณ', settingsTerms: 'ข้อกำหนดการให้บริการ', settingsTermsSub: 'กฎและข้อจำกัดการใช้งาน', settingsData: 'ข้อมูล', settingsDelete: 'ลบข้อมูลของฉัน', settingsDeleteSub: 'ลบการสนทนาและการตั้งค่า', settingsDisclaimer: '⚠ STiTy ให้บริการแปลภาษาโดย AI และไม่รับประกันความแม่นยำ อย่าใช้สำหรับการสื่อสารทางการแพทย์ กฎหมาย หรือฉุกเฉิน', settingsBack: '‹ กลับ', privacyDate: 'อัปเดตล่าสุด: 17 พฤษภาคม 2026', privacyH1: '1. ข้อมูลเสียง', privacyP1: 'STiTy บันทึกเสียงของคุณผ่านไมโครโฟนของอุปกรณ์เฉพาะเมื่อเซสชันทำงานอยู่ เสียงจะถูกส่งไปยังเซิร์ฟเวอร์แปลภาษาของเรา ถอดความและแปลในเวลาจริง', privacyH2: '2. สิ่งที่เราจัดเก็บ', privacyB2aStart: '• การบันทึกเสียง', privacyB2aBold: 'จะไม่ถูกเก็บไว้', privacyB2aEnd: 'หลังจากสิ้นสุดเซสชัน', privacyB2b: '• ข้อความที่แปลแล้วจะถูกเก็บไว้ในอุปกรณ์ของคุณเท่านั้น', privacyB2c: '• เราบันทึกข้อมูลเมตาของเซสชันที่ไม่ระบุตัวตน (ระยะเวลา, ภาษาที่ใช้) เพื่อคุณภาพการบริการ', privacyH3: '3. บริการของบุคคลที่สาม', privacyP3: 'เราใช้บุคคลที่สามต่อไปนี้เพื่อประมวลผลเสียง:', privacyB3a: '• Google Cloud Speech-to-Text', privacyB3b: '• OpenAI / Anthropic Translation APIs', privacyP3b: 'นโยบายความเป็นส่วนตัวของพวกเขาใช้กับข้อมูลที่พวกเขาประมวลผล', privacyH4: '4. สิทธิ์ของคุณ', privacyP4: 'คุณสามารถลบข้อมูลทั้งหมดที่เชื่อมโยงกับอุปกรณ์ของคุณได้ทุกเมื่อผ่าน การตั้งค่า → ลบข้อมูลของฉัน ติดต่อ privacy@stity.app สำหรับคำขอใดๆ', privacyH5: '5. เด็ก', privacyP5: 'STiTy ไม่ได้มีไว้สำหรับผู้ใช้อายุต่ำกว่า 13 ปี', termsDate: 'อัปเดตล่าสุด: 17 พฤษภาคม 2026', termsH1: '1. คำอธิบายบริการ', termsP1: 'STiTy เป็นเครื่องมือแปลเสียงพูดแบบเรียลไทม์ คุณภาพการแปลขึ้นอยู่กับความชัดเจนของเสียง คู่ภาษา และประสิทธิภาพของโมเดล AI', termsH2: '2. ไม่มีการรับประกัน', termsP2: 'การแปลสร้างขึ้นโดยอัตโนมัติและอาจไม่ถูกต้อง อย่าพึ่งพา STiTy สำหรับการสื่อสารทางการแพทย์ กฎหมาย การเงิน หรือฉุกเฉิน', termsH3: '3. การใช้งานที่ยอมรับได้', termsB3a: '• อย่าบันทึกเสียงผู้อื่นโดยไม่ได้รับความยินยอม', termsB3b: '• อย่าใช้ STiTy เพื่อคุกคาม หมิ่นประมาท หรือหลอกลวงผู้อื่น', termsB3c: '• อย่าพยายามทำวิศวกรรมย้อนกลับบริการ', termsH4: '4. ขีดจำกัดความจุ', termsP4: 'STiTy รองรับผู้ใช้พร้อมกันสูงสุด 20 คน เซสชันอาจถูกจัดคิวในช่วงเวลาเร่งด่วน', termsH5: '4. การยุติ', termsP5: 'เราอาจระงับการเข้าถึงได้ทุกเมื่อสำหรับการละเมิดนโยบาย', deleteTitle: 'ลบข้อมูลของฉัน', deleteP1: 'ข้อมูลต่อไปนี้จะถูกลบอย่างถาวร:', deleteB1: '• บันทึกการสนทนาทั้งหมดที่บันทึกไว้ในอุปกรณ์นี้', deleteB2: '• การตั้งค่าภาษาและการตั้งค่าต่างๆ', deleteB3: '• ข้อมูลเมตาการใช้งานที่ไม่ระบุตัวตนที่เชื่อมโยงกับ ID อุปกรณ์ของคุณ', deleteP2: 'การดำเนินการนี้ไม่สามารถยกเลิกได้ การบันทึกเสียงไม่เคยถูกเก็บไว้ ดังนั้นจึงไม่มีสิ่งใดเหลืออยู่บนเซิร์ฟเวอร์ของเรา', deleteBtn: 'ลบข้อมูลของฉันอย่างถาวร', deleteAlertTitle: 'ลบแล้ว', deleteAlertMsg: 'ข้อมูลในเครื่องทั้งหมดถูกลบแล้ว', aboutP: 'STiTy เป็นแอปแปลเสียงพูดแบบเรียลไทม์ที่ช่วยให้ผู้คนสามารถพูดในภาษาของตนเองและอยู่ในการสนทนาเดียวกันได้อย่างต่อเนื่อง', aboutCopyright: '© 2026 STiTy. สงวนลิขสิทธิ์ทั้งหมด' },
  de: { peerLanguage: 'SPRACHE DES ANDEREN', mode: 'MODUS', modes: { 'mode-1': 'Lautsprecher', 'mode-2': 'Kopfhörer', 'mode-3': 'Beide' }, collapse: 'Einklappen ▲', configureAndStart: 'Einstellen und Start tippen', startSpeaking: 'Anfangen zu sprechen', listening: 'Zuhören...', start: 'Start', connectingServer: 'Verbinde mit Server...', startingServer: 'Server wird gestartet...', stop: 'Stopp', resume: 'Fortsetzen', back: 'Zurück', cancel: 'Abbrechen', endTitle: 'Gespräch beenden', endMsg: 'Möchten Sie das Gespräch beenden?', end: 'Beenden', errTitle: 'Fehler', langMustDiffer: 'Meine Sprache und die des anderen müssen unterschiedlich sein.', peerPickerTitle: 'Sprache des anderen', connectionFailed: 'Verbindung fehlgeschlagen', waitingForSlot: 'Warte auf Slot…', capacityFullMsg: 'Server voll. Start möglich, sobald ein Slot frei wird.', capacityFreeMsg: 'Ein Slot ist frei — jetzt starten!', restartApp: 'Bitte starten Sie die App neu.', translation: 'ÜBERSETZUNG', fast: 'Schnell', accurate: 'Präzise', fastSub: 'Niedrige Latenz', accurateSub: 'Kontextsensitiv', fastDesc: 'Übersetzt nahezu in Echtzeit mit niedriger Latenz.', accurateDesc: 'Nutzt den umgebenden Kontext für natürlichere Übersetzungen.', settingsTitle: 'Einstellungen', settingsGeneral: 'Allgemein', settingsAbout: 'Über STiTy', settingsAboutSub: 'Version, Impressum', settingsLegal: 'Rechtliches', settingsPrivacy: 'Datenschutzrichtlinie', settingsPrivacySub: 'Umgang mit Ihrer Stimme und Ihren Daten', settingsTerms: 'Nutzungsbedingungen', settingsTermsSub: 'Regeln und Nutzungsgrenzen', settingsData: 'Daten', settingsDelete: 'Meine Daten löschen', settingsDeleteSub: 'Gespräche und Einstellungen löschen', settingsDisclaimer: '⚠ STiTy stellt KI-generierte Übersetzungen bereit und garantiert keine Genauigkeit. Nicht für medizinische, rechtliche oder Notfallkommunikation verwenden.', settingsBack: '‹ Zurück', privacyDate: 'Zuletzt aktualisiert: 17. Mai 2026', privacyH1: '1. Sprachdaten', privacyP1: 'STiTy nimmt Ihre Stimme über das Gerätmikrofon nur während einer aktiven Sitzung auf. Das Audio wird an unsere Übersetzungsserver gestreamt, transkribiert und in Echtzeit übersetzt.', privacyH2: '2. Was wir speichern', privacyB2aStart: '• Audioaufnahmen werden nach Ende der Sitzung ', privacyB2aBold: 'nicht gespeichert', privacyB2aEnd: '.', privacyB2b: '• Übersetzte Texttranskripte verbleiben nur auf Ihrem Gerät.', privacyB2c: '• Wir protokollieren anonymisierte Sitzungsmetadaten (Dauer, verwendete Sprachen) zur Qualitätssicherung.', privacyH3: '3. Drittanbieterdienste', privacyP3: 'Wir nutzen folgende Drittanbieter zur Audioverarbeitung:', privacyB3a: '• Google Cloud Speech-to-Text', privacyB3b: '• OpenAI / Anthropic Übersetzungs-APIs', privacyP3b: 'Deren Datenschutzrichtlinien gelten für die von ihnen verarbeiteten Daten.', privacyH4: '4. Ihre Rechte', privacyP4: 'Sie können alle mit Ihrem Gerät verknüpften Daten jederzeit über Einstellungen → Meine Daten löschen entfernen. Für Anfragen wenden Sie sich an privacy@stity.app.', privacyH5: '5. Kinder', privacyP5: 'STiTy ist nicht für Nutzer unter 13 Jahren bestimmt.', termsDate: 'Zuletzt aktualisiert: 17. Mai 2026', termsH1: '1. Dienstbeschreibung', termsP1: 'STiTy ist ein Echtzeit-Sprachübersetzungstool. Die Übersetzungsqualität hängt von der Audioklarheit, dem Sprachenpaar und der KI-Modellleistung ab.', termsH2: '2. Keine Garantie', termsP2: 'Übersetzungen werden automatisch erstellt und können ungenau sein. Verlassen Sie sich bei medizinischer, rechtlicher, finanzieller oder Notfallkommunikation nicht auf STiTy.', termsH3: '3. Zulässige Nutzung', termsB3a: '• Nehmen Sie keine Personen ohne deren Zustimmung auf.', termsB3b: '• Verwenden Sie STiTy nicht, um andere zu belästigen, zu verleumden oder zu täuschen.', termsB3c: '• Versuchen Sie nicht, den Dienst zu dekompilieren oder zu reverse-engineeren.', termsH4: '4. Kapazitätsgrenzen', termsP4: 'STiTy unterstützt derzeit bis zu 20 gleichzeitige Nutzer. Sitzungen können zu Spitzenzeiten in die Warteschlange gestellt werden.', termsH5: '4. Kündigung', termsP5: 'Wir können den Zugang jederzeit bei Richtlinienverstößen sperren.', deleteTitle: 'Meine Daten löschen', deleteP1: 'Folgende Daten werden dauerhaft gelöscht:', deleteB1: '• Alle gespeicherten Gesprächsprotokolle auf diesem Gerät', deleteB2: '• Spracheinstellungen und Konfiguration', deleteB3: '• Anonymisierte Nutzungsmetadaten, die mit Ihrer Geräte-ID verknüpft sind', deleteP2: 'Diese Aktion kann nicht rückgängig gemacht werden. Audioaufnahmen werden nie gespeichert, daher verbleiben keine Daten auf unseren Servern.', deleteBtn: 'Meine Daten dauerhaft löschen', deleteAlertTitle: 'Gelöscht', deleteAlertMsg: 'Alle lokalen Daten wurden gelöscht.', aboutP: 'STiTy ist eine Echtzeit-Sprachübersetzungs-App, mit der Menschen in ihrer eigenen Sprache sprechen und trotzdem zusammen in einem Gespräch bleiben können.', aboutCopyright: '© 2026 STiTy. Alle Rechte vorbehalten.' },
};

// ─── Fixed card colors (from prototype) ──────────────────────────────────────
const MY_CARD_COLOR   = '#6080C8';
const PEER_CARD_COLOR = '#9BB4D4';
const MODE_ACTIVE_COLOR = '#C8B830';

// ─── Bubble color palette (by detected language) ──────────────────────────────
const LANG_COLORS: Record<string, { bubble: string; avatar: string }> = {
  en: { bubble: '#6080C8', avatar: '#9BB4D4' },
  ko: { bubble: '#7A9030', avatar: '#A8B870' },
  ja: { bubble: '#C87060', avatar: '#E0A898' },
  zh: { bubble: '#9060C8', avatar: '#B898E0' },
  es: { bubble: '#C8A030', avatar: '#E0C878' },
  fr: { bubble: '#308898', avatar: '#78B8C8' },
  id: { bubble: '#30A070', avatar: '#70C0A0' },
  vi: { bubble: '#B85050', avatar: '#E09080' },
  th: { bubble: '#5080C0', avatar: '#88B0E0' },
  de: { bubble: '#C09050', avatar: '#E0C080' },
};
const getLangColor = (code: string) =>
  LANG_COLORS[code] ?? { bubble: '#909090', avatar: '#B8B8B8' };

// ─── Storage keys ─────────────────────────────────────────────────────────────
const SK = { MY: 'stity_myLang', TARGET: 'stity_targetLang', MODE: 'stity_mode', SPEED: 'stity_speed' };

// ─── Helpers ──────────────────────────────────────────────────────────────────
const langToCode = (lang: string): string => {
  const map: Record<string, string> = {
    Korean: 'ko', English: 'en', Japanese: 'ja', Chinese: 'zh',
    Indonesian: 'id', Vietnamese: 'vi', Thai: 'th',
    French: 'fr', German: 'de', Spanish: 'es',
  };
  return map[lang] ?? lang.toLowerCase().substring(0, 2);
};

const translateText = async (text: string, sl: string, tl: string): Promise<string> => {
  try {
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${sl}&tl=${tl}&dt=t&q=${encodeURIComponent(text)}`;
    const res = await fetch(url);
    const data = await res.json();
    return data[0].map((i: any) => i[0]).join('');
  } catch {
    return '';
  }
};

// ─── Sub-components ───────────────────────────────────────────────────────────

// Logo: S(#6080C8) T(#6080C8) i(#9BB4D4) T(#7A9030) y(#E8E0A0) — Prototype.html 기준
const STiTyLogo = () => (
  <Text style={S.logo}>
    <Text style={{ color: '#6080C8' }}>ST</Text>
    <Text style={{ color: '#9BB4D4' }}>i</Text>
    <Text style={{ color: '#7A9030' }}>T</Text>
    <Text style={{ color: '#E8E0A0' }}>y</Text>
  </Text>
);

const StatusDot = ({ active }: { active: boolean }) => {
  const ringOpacity = useRef(new Animated.Value(0.3)).current;
  const animRef = useRef<Animated.CompositeAnimation | null>(null);

  useEffect(() => {
    if (active) {
      animRef.current = Animated.loop(
        Animated.sequence([
          Animated.timing(ringOpacity, { toValue: 0.06, duration: 750, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
          Animated.timing(ringOpacity, { toValue: 0.3, duration: 750, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        ])
      );
      animRef.current.start();
    } else {
      animRef.current?.stop();
      ringOpacity.setValue(0.3);
    }
    return () => { animRef.current?.stop(); };
  }, [active]);

  return (
    <View style={S.dotWrap}>
      {active && <Animated.View style={[S.dotRing, { opacity: ringOpacity }]} />}
      <View style={[S.dot, { backgroundColor: active ? '#22c55e' : '#ccc' }]} />
    </View>
  );
};

const WaveBar = ({ delay }: { delay: number }) => {
  const h = useRef(new Animated.Value(3)).current;
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(h, { toValue: 14, duration: 400, delay, useNativeDriver: false }),
        Animated.timing(h, { toValue: 3, duration: 400, useNativeDriver: false }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, []);
  return <Animated.View style={[S.waveBar, { height: h }]} />;
};

const RecordingBar = ({ label }: { label: string }) => (
  <View style={S.recBar}>
    <View style={S.recDot} />
    <Text style={S.recLabel}>{label}</Text>
    <View style={S.waveRow}>
      {[0, 100, 200, 300, 400, 300, 200, 100, 0].map((d, i) => (
        <WaveBar key={i} delay={d} />
      ))}
    </View>
  </View>
);

// Mode-based idle illustrations (from prototype)
const SpeakerIllustration = () => (
  <Svg width={150} height={125} viewBox="0 0 180 150">
    <Circle cx="90" cy="82" r="62" fill="#E8E0A0" opacity="0.3" />
    <Ellipse cx="90" cy="132" rx="58" ry="7" fill="#7A9030" opacity="0.22" />
    <Rect x="84" y="70" width="12" height="28" rx="2.5" fill="#1a1a1a" />
    <Rect x="85.4" y="74" width="9.2" height="20" rx="1" fill="#6080C8" />
    <Rect x="87.5" y="71.8" width="5" height="1.2" rx="0.6" fill="#555" />
    <Path d="M87 84 Q88.5 81 90 84 Q91.5 87 93 84" stroke="#E8E0A0" strokeWidth="1" fill="none" strokeLinecap="round" />
    <Path d="M80 76 Q75 84 80 92" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.75" />
    <Path d="M75 72 Q67 84 75 96" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.4" />
    <Path d="M100 76 Q105 84 100 92" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.75" />
    <Path d="M105 72 Q113 84 105 96" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.4" />
    <Ellipse cx="55" cy="114" rx="14" ry="15" fill="#9BB4D4" />
    <Circle cx="55" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="59" cy="92" r="1.5" fill="#6080C8" />
    <Circle cx="64" cy="92" r="1.5" fill="#6080C8" />
    <Path d="M58 97 Q61 99 64 97" stroke="#6080C8" strokeWidth="1.3" strokeLinecap="round" fill="none" />
    <Ellipse cx="125" cy="114" rx="14" ry="15" fill="#6080C8" />
    <Circle cx="125" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="116" cy="92" r="1.5" fill="#7A9030" />
    <Circle cx="121" cy="92" r="1.5" fill="#7A9030" />
    <Path d="M116 97 Q119 99 122 97" stroke="#7A9030" strokeWidth="1.3" strokeLinecap="round" fill="none" />
  </Svg>
);

const EarphoneIllustration = () => (
  <Svg width={150} height={125} viewBox="0 0 180 150">
    <Circle cx="90" cy="82" r="62" fill="#9BB4D4" opacity="0.22" />
    <Ellipse cx="90" cy="132" rx="58" ry="7" fill="#7A9030" opacity="0.22" />
    <Circle cx="74" cy="76" r="1.6" fill="#9BB4D4" />
    <Circle cx="84" cy="70" r="2.2" fill="#E8E0A0" />
    <Circle cx="96" cy="70" r="2.2" fill="#E8E0A0" />
    <Circle cx="106" cy="76" r="1.6" fill="#6080C8" />
    <Ellipse cx="55" cy="114" rx="14" ry="15" fill="#9BB4D4" />
    <Circle cx="55" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="59" cy="92" r="1.5" fill="#6080C8" />
    <Circle cx="64" cy="92" r="1.5" fill="#6080C8" />
    <Path d="M58 97 Q61 99 64 97" stroke="#6080C8" strokeWidth="1.3" strokeLinecap="round" fill="none" />
    <Ellipse cx="43" cy="94" rx="2.6" ry="3.6" fill="#fff" stroke="#6080C8" strokeWidth="1.3" />
    <Rect x="42" y="97.5" width="2" height="4.5" rx="1" fill="#fff" stroke="#6080C8" strokeWidth="1" />
    <Ellipse cx="125" cy="114" rx="14" ry="15" fill="#6080C8" />
    <Circle cx="125" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="116" cy="92" r="1.5" fill="#7A9030" />
    <Circle cx="121" cy="92" r="1.5" fill="#7A9030" />
    <Path d="M116 97 Q119 99 122 97" stroke="#7A9030" strokeWidth="1.3" strokeLinecap="round" fill="none" />
    <Ellipse cx="137" cy="94" rx="2.6" ry="3.6" fill="#fff" stroke="#7A9030" strokeWidth="1.3" />
    <Rect x="136" y="97.5" width="2" height="4.5" rx="1" fill="#fff" stroke="#7A9030" strokeWidth="1" />
  </Svg>
);

const BothIllustration = () => (
  <Svg width={150} height={125} viewBox="0 0 180 150">
    <Circle cx="90" cy="82" r="62" fill="#E8E0A0" opacity="0.25" />
    <Ellipse cx="90" cy="132" rx="58" ry="7" fill="#7A9030" opacity="0.22" />
    <Ellipse cx="55" cy="114" rx="14" ry="15" fill="#9BB4D4" />
    <Circle cx="55" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="59" cy="92" r="1.5" fill="#6080C8" />
    <Circle cx="64" cy="92" r="1.5" fill="#6080C8" />
    <Path d="M58 97 Q61 99 64 97" stroke="#6080C8" strokeWidth="1.3" strokeLinecap="round" fill="none" />
    <Rect x="44" y="64" width="22" height="11" rx="5.5" fill="#6080C8" />
    <Path d="M53 75 L55 78 L57 75 Z" fill="#6080C8" />
    <SvgText x="55" y="72.5" textAnchor="middle" fontSize="7.5" fontWeight="800" fill="#fff" letterSpacing="0.5">ME</SvgText>
    <Ellipse cx="43" cy="94" rx="2.6" ry="3.6" fill="#fff" stroke="#6080C8" strokeWidth="1.3" />
    <Rect x="42" y="97.5" width="2" height="4.5" rx="1" fill="#fff" stroke="#6080C8" strokeWidth="1" />
    <Rect x="84" y="76" width="12" height="28" rx="2.5" fill="#1a1a1a" />
    <Rect x="85.4" y="80" width="9.2" height="20" rx="1" fill="#6080C8" />
    <Rect x="87.5" y="77.8" width="5" height="1.2" rx="0.6" fill="#555" />
    <Path d="M87 90 Q88.5 87 90 90 Q91.5 93 93 90" stroke="#E8E0A0" strokeWidth="1" fill="none" strokeLinecap="round" />
    <Path d="M100 82 Q105 90 100 98" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.75" />
    <Path d="M105 78 Q113 90 105 102" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.4" />
    <Ellipse cx="125" cy="114" rx="14" ry="15" fill="#6080C8" />
    <Circle cx="125" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="116" cy="92" r="1.5" fill="#7A9030" />
    <Circle cx="121" cy="92" r="1.5" fill="#7A9030" />
    <Path d="M116 97 Q119 99 122 97" stroke="#7A9030" strokeWidth="1.3" strokeLinecap="round" fill="none" />
  </Svg>
);

const ModeIllustration = ({ modeId }: { modeId: string }) => {
  if (modeId === 'mode-2') return <EarphoneIllustration />;
  if (modeId === 'mode-3') return <BothIllustration />;
  return <SpeakerIllustration />;
};

// Pulsing dot for capacity notice
const CapacityPulse = ({ isOk }: { isOk: boolean }) => {
  const ringOpacity = useRef(new Animated.Value(0.2)).current;
  useEffect(() => {
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(ringOpacity, { toValue: 0.05, duration: 750, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        Animated.timing(ringOpacity, { toValue: 0.2, duration: 750, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
      ])
    );
    anim.start();
    return () => anim.stop();
  }, []);
  const color = isOk ? '#7A9030' : '#C8A030';
  return (
    <View style={{ width: 14, height: 14, alignItems: 'center', justifyContent: 'center' }}>
      <Animated.View style={{ position: 'absolute', width: 14, height: 14, borderRadius: 7, backgroundColor: color, opacity: ringOpacity }} />
      <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: color }} />
    </View>
  );
};

const LangPickerSheet = ({
  visible, title, selectedCode, excludeCode, onSelect, onClose, myLangCode,
  cancelLabel = 'Cancel', forceEnglish = false,
}: {
  visible: boolean;
  title: string;
  selectedCode: string;
  excludeCode: string;
  onSelect: (lang: Language) => void;
  onClose: () => void;
  myLangCode: string;
  cancelLabel?: string;
  forceEnglish?: boolean;
}) => {
  const { bottom } = useSafeAreaInsets();
  const filtered = LANGUAGES.filter(l => l.code !== excludeCode);
  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <View style={S.pickerOverlay}>
        <TouchableOpacity style={{ flex: 1 }} activeOpacity={1} onPress={onClose} />
        <View style={[S.pickerSheet, { paddingBottom: Math.max(bottom, 16) + 16 }]}>
          <Text style={S.pickerTitle}>{title}</Text>
          <ScrollView>
            {filtered.map(lang => {
              const name = forceEnglish
                ? lang.name
                : (lang.translations[myLangCode] || lang.nativeName);
              const selected = lang.code === selectedCode;
              return (
                <TouchableOpacity
                  key={lang.code}
                  style={[S.pickerOption, selected && S.pickerOptionSel]}
                  onPress={() => { onSelect(lang); onClose(); }}
                >
                  <Text style={[S.pickerOptionTxt, selected && S.pickerOptionTxtSel]}>
                    {name} ({lang.code})
                  </Text>
                </TouchableOpacity>
              );
            })}
          </ScrollView>
          <TouchableOpacity style={S.pickerCancel} onPress={onClose}>
            <Text style={S.pickerCancelTxt}>{cancelLabel}</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
};

const BubbleItem = ({ entry, myLangCode, targetLangCode }: {
  entry: TranscriptionEntry; myLangCode: string; targetLangCode: string;
}) => {
  const isMine = entry.language === myLangCode;
  const isPeer = entry.language === targetLangCode;
  const clr = isMine
    ? { bubble: '#6080C8', avatar: '#9BB4D4', text: '#fff', subText: 'rgba(255,255,255,0.65)' }
    : isPeer
    ? { bubble: '#f2f2f2', avatar: '#E8E0A0', text: '#1a1a1a', subText: '#aaa' }
    : { ...getLangColor(entry.language), text: '#fff', subText: 'rgba(255,255,255,0.65)' };

  const slideAnim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(slideAnim, {
      toValue: 1,
      duration: 300,
      easing: Easing.bezier(0.4, 0, 0.2, 1),
      useNativeDriver: true,
    }).start();
  }, []);

  return (
    <Animated.View style={[S.bubbleRow, isMine && S.bubbleRowMine, {
      opacity: slideAnim,
      transform: [{ translateY: slideAnim.interpolate({ inputRange: [0, 1], outputRange: [12, 0] }) }],
    }]}>
      <View style={[S.avatar, { backgroundColor: clr.avatar }]}>
        <Text style={[S.avatarTxt, { color: isMine ? '#fff' : '#7A9030' }]}>
          {entry.language.toUpperCase()}
        </Text>
      </View>
      <View style={[
        S.bubble,
        {
          backgroundColor: clr.bubble,
          borderBottomRightRadius: isMine ? 4 : 18,
          borderBottomLeftRadius: isMine ? 18 : 4,
        },
      ]}>
        <Text style={[S.bubbleMain, { color: clr.text }]}>{entry.text}</Text>
        {!!entry.translatedText && (
          <Text style={[S.bubbleSub, { color: clr.subText }]}>{entry.translatedText}</Text>
        )}
      </View>
    </Animated.View>
  );
};

// ─── Main screen ──────────────────────────────────────────────────────────────
export const HomeScreen: React.FC<{ navigation: any }> = () => {
  const insets = useSafeAreaInsets();

  type SessionState = 'idle' | 'recording' | 'paused';
  const [sessionState, setSessionState] = useState<SessionState>('idle');
  const sessionStateRef = useRef<SessionState>('idle');

  const [myLang, setMyLang] = useState<Language>(LANGUAGES.find(l => l.code === 'en')!);
  const [targetLang, setTargetLang] = useState<Language>(LANGUAGES.find(l => l.code === 'ko')!);
  const [modeId, setModeId] = useState('mode-1');
  const [loaded, setLoaded] = useState(false);

  const [showSetup, setShowSetup] = useState(false);
  const [picker, setPicker] = useState<null | 'my' | 'peer' | 'speed'>(null);
  const [speed, setSpeed] = useState<'fast' | 'accurate'>('fast');
  const [isTTSMuted, setIsTTSMuted] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuPage, setMenuPage] = useState<null | 'privacy' | 'terms' | 'delete' | 'about'>(null);

  const [transcriptions, setTranscriptions] = useState<TranscriptionEntry[]>([]);
  const [isInitializing, setIsInitializing] = useState(false);
  const [sessionError, setSessionError] = useState('');
  const [serverCapacity, setServerCapacity] = useState<{ active: number; max: number } | null>(null);
  const [justFreed, setJustFreed] = useState(false);
  const prevServerFullRef = useRef(false);
  const [connectPct, setConnectPct] = useState(0);
  const progressAnim = useRef(new Animated.Value(0)).current;
  const progressAnimRef = useRef<Animated.CompositeAnimation | null>(null);

  // chips / setup 패널 애니메이션
  const chipsAnim = useRef(new Animated.Value(0)).current;
  const setupOpacityAnim = useRef(new Animated.Value(1)).current;
  const setupHeightAnim = useRef(new Animated.Value(1000)).current;
  const setupNaturalHeight = useRef(1000);
  const isSetupAnimatingRef = useRef(false);
  const isSetupVisibleRef = useRef(true);
  const setupAnimRef = useRef<Animated.CompositeAnimation | null>(null);

  const scrollRef = useRef<ScrollView>(null);
  const entryIdRef = useRef(0);
  const ttsQueueRef = useRef<{ text: string; lang: string }[]>([]);
  const isSpeakingRef = useRef(false);

  const modeRef = useRef(modeId);
  const myLangRef = useRef(myLang);
  const targetLangRef = useRef(targetLang);
  const isTTSMutedRef = useRef(isTTSMuted);

  const { isConnected, connect, sendAudio, disconnect, addMessageListener, sendMessage, serverStatus, probeServer } = useWebSocketContext();
  const isConnectedRef = useRef(isConnected);
  const sendAudioRef = useRef(sendAudio);
  const sendMessageRef = useRef(sendMessage);
  const serverStatusRef = useRef(serverStatus);

  useEffect(() => { isConnectedRef.current = isConnected; }, [isConnected]);
  useEffect(() => { sendAudioRef.current = sendAudio; }, [sendAudio]);
  useEffect(() => { sendMessageRef.current = sendMessage; }, [sendMessage]);
  useEffect(() => { serverStatusRef.current = serverStatus; }, [serverStatus]);
  useEffect(() => { isTTSMutedRef.current = isTTSMuted; }, [isTTSMuted]);
  useEffect(() => { modeRef.current = modeId; }, [modeId]);
  useEffect(() => { myLangRef.current = myLang; }, [myLang]);
  useEffect(() => { targetLangRef.current = targetLang; }, [targetLang]);
  useEffect(() => { sessionStateRef.current = sessionState; }, [sessionState]);

  const wasAutoSuspendedRef = useRef(false);

  const { startRecording, stopRecording } = useAudioRecording({
    onAudioData: (audioData) => {
      if (isConnectedRef.current && sessionStateRef.current === 'recording') {
        sendAudioRef.current(audioData);
      }
    },
  });

  // ── 초기화 ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    void initTtsEngine();
    probeServer();

    AsyncStorage.multiGet([SK.MY, SK.TARGET, SK.SPEED]).then(async pairs => {
      const [myVal, targetVal, speedVal] = pairs.map(p => p[1]);
      if (myVal) {
        const f = LANGUAGES.find(l => l.code === myVal);
        if (f) { setMyLang(f); myLangRef.current = f; }
      }
      if (targetVal) {
        const f = LANGUAGES.find(l => l.code === targetVal);
        if (f) { setTargetLang(f); targetLangRef.current = f; }
      }
      if (speedVal === 'accurate') setSpeed('accurate');
      // 이어폰 연결 여부로 초기 모드 자동 설정
      const earphone = await isEarphoneConnected();
      const initialMode = earphone ? 'mode-2' : 'mode-1';
      setModeId(initialMode);
      modeRef.current = initialMode;
      setLoaded(true);
    }).catch(() => setLoaded(true));

    return () => {
      stopRecording();
      disconnect();
      ttsQueueRef.current = [];
      isSpeakingRef.current = false;
      ttsStop();
      releaseAudioMode();
    };
  }, []);

  // ── Server connect progress ───────────────────────────────────────────────────
  useEffect(() => {
    const id = progressAnim.addListener(({ value }) => setConnectPct(Math.round(value * 100)));
    return () => progressAnim.removeListener(id);
  }, []);

  useEffect(() => {
    progressAnimRef.current?.stop();
    if (serverStatus === 'ec2-starting') {
      progressAnim.setValue(0);
    } else if (serverStatus === 'connecting') {
      progressAnim.setValue(0);
      progressAnimRef.current = Animated.timing(progressAnim, { toValue: 0.85, duration: 80000, useNativeDriver: false });
      progressAnimRef.current.start();
    } else if (serverStatus === 'ready') {
      Animated.timing(progressAnim, { toValue: 1, duration: 400, useNativeDriver: false }).start();
    }
  }, [serverStatus]);

  // ── Setup panel ↔ Chips 전환 애니메이션 (maxHeight + opacity parallel) ────────
  const prevSetupVisibleRef = useRef(true);
  useEffect(() => {
    const visible = sessionState === 'idle' || showSetup;
    if (visible === prevSetupVisibleRef.current) return;
    prevSetupVisibleRef.current = visible;
    isSetupVisibleRef.current = visible;

    setupAnimRef.current?.stop();
    isSetupAnimatingRef.current = true;

    if (visible) {
      setupHeightAnim.setValue(0);
      setupOpacityAnim.setValue(0);
      // non-idle 상태에서는 collapse 버튼이 추가돼 높이가 늘어나므로 여유 확보
      const showTargetH = sessionStateRef.current !== 'idle'
        ? setupNaturalHeight.current + 48
        : setupNaturalHeight.current;
      setupAnimRef.current = Animated.parallel([
        Animated.timing(setupHeightAnim, {
          toValue: showTargetH,
          duration: 420,
          easing: Easing.bezier(0.4, 0, 0.2, 1),
          useNativeDriver: false,
        }),
        Animated.timing(setupOpacityAnim, {
          toValue: 1,
          duration: 340,
          easing: Easing.ease,
          useNativeDriver: false,
        }),
      ]);
      setupAnimRef.current.start(() => { isSetupAnimatingRef.current = false; });
      Animated.timing(chipsAnim, { toValue: 0, duration: 250, easing: Easing.ease, useNativeDriver: true }).start();
    } else {
      setupAnimRef.current = Animated.parallel([
        Animated.timing(setupHeightAnim, {
          toValue: 0,
          duration: 380,
          easing: Easing.bezier(0.4, 0, 0.2, 1),
          useNativeDriver: false,
        }),
        Animated.timing(setupOpacityAnim, {
          toValue: 0,
          duration: 260,
          easing: Easing.ease,
          useNativeDriver: false,
        }),
      ]);
      setupAnimRef.current.start(() => { isSetupAnimatingRef.current = false; });
      Animated.timing(chipsAnim, { toValue: 1, duration: 380, easing: Easing.ease, useNativeDriver: true }).start();
    }
  }, [sessionState, showSetup]);

  // ── AppState ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    const prevRef = { current: AppState.currentState };
    const sub = AppState.addEventListener('change', next => {
      const prev = prevRef.current;
      prevRef.current = next;
      if ((next === 'background' || next === 'inactive') && sessionStateRef.current === 'recording') {
        wasAutoSuspendedRef.current = true;
        stopRecording();
        disconnect();
        sessionStateRef.current = 'paused';
        setSessionState('paused');
      } else if (next === 'active' && (prev === 'background' || prev === 'inactive') && sessionStateRef.current === 'paused') {
        if (wasAutoSuspendedRef.current) doResume();
      } else if (next === 'active' && (prev === 'background' || prev === 'inactive') && sessionStateRef.current === 'idle') {
        probeServer(); // keepalive 끊겼으면 재기동, 살아있으면 즉시 return
      }
    });
    return () => sub.remove();
  }, []);

  // ── keepalive 실패 감지: serverStatus 'ready'→'idle' 전환 시 자동 재기동 ────────
  const prevServerStatusRef = useRef<string>(serverStatus);
  useEffect(() => {
    const prev = prevServerStatusRef.current;
    prevServerStatusRef.current = serverStatus;
    if (serverStatus === 'idle' && prev !== 'idle' && sessionStateRef.current === 'idle') {
      probeServer();
    }
  }, [serverStatus]);

  // ── Message listener ─────────────────────────────────────────────────────────
  const handleMessageRef = useRef((msg: any) => {
    const text = (msg.original || '').trim();
    if (!text || msg.type !== 'final') return;
    const lang = langToCode(msg.language || 'auto');
    const serverTrans = (msg.translation || '').trim();
    addTranscription(lang, text, serverTrans || undefined);
  });

  useEffect(() => {
    return addMessageListener((msg: any) => {
      if (msg.type === 'capacity' && typeof msg.active === 'number' && typeof msg.max === 'number') {
        setServerCapacity({ active: msg.active, max: msg.max });
      }
      handleMessageRef.current(msg);
    });
  }, [addMessageListener]);

  // ── Server capacity: full→available transition 감지 ───────────────────────────
  const serverFull = serverCapacity !== null && serverCapacity.active >= serverCapacity.max;
  useEffect(() => {
    if (prevServerFullRef.current && !serverFull) {
      setJustFreed(true);
      const t = setTimeout(() => setJustFreed(false), 2500);
      prevServerFullRef.current = serverFull;
      return () => clearTimeout(t);
    }
    prevServerFullRef.current = serverFull;
  }, [serverFull]);

  // ── Auto-scroll ───────────────────────────────────────────────────────────────
  useEffect(() => {
    setTimeout(() => scrollRef.current?.scrollToEnd({ animated: false }), 50);
  }, [transcriptions.length]);

  // ── TTS ───────────────────────────────────────────────────────────────────────
  // mode-1: speaker, TTS for all languages
  // mode-2: earphone, TTS only for peer language
  // mode-3: if TTS lang == myLang → earphone (I hear peer's translation)
  //         if TTS lang != myLang → speaker (peer hears my translation)
  const processNextTTS = useCallback(() => {
    if (ttsQueueRef.current.length === 0) { isSpeakingRef.current = false; return; }
    isSpeakingRef.current = true;
    const next = ttsQueueRef.current.shift()!;
    const ttsStart = new Date().toISOString();

    let isEarphone: boolean;
    if (modeRef.current === 'mode-1') {
      isEarphone = false;
    } else if (modeRef.current === 'mode-2') {
      isEarphone = true;
    } else {
      // mode-3: earphone when TTS is in my language, speaker when it's in peer's language
      isEarphone = next.lang === myLangRef.current.code;
    }
    setSpeakerphoneOn(!isEarphone);
    ttsSpeak(
      next.text, next.lang, 1.3,
      () => {
        sendMessageRef.current({ type: 'tts_log', text: next.text, lang: next.lang, start: ttsStart, end: new Date().toISOString() });
        processNextTTS();
      },
      () => processNextTTS(),
      isEarphone, // true → STREAM_VOICE_CALL(이어폰), false → STREAM_MUSIC(스피커)
    );
  }, []);

  const speakTranslation = (text: string, lang: string) => {
    if (isTTSMutedRef.current || !text) return;
    ttsQueueRef.current.push({ text, lang });
    if (!isSpeakingRef.current) processNextTTS();
  };

  const shouldPlayTTS = (detectedLang: string) => {
    if (isTTSMutedRef.current) return false;
    if (modeRef.current === 'mode-1' || modeRef.current === 'mode-3') return true;
    if (modeRef.current === 'mode-2') return detectedLang !== myLangRef.current.code;
    return false;
  };

  const getTTSTarget = (detectedLang: string) =>
    detectedLang !== myLangRef.current.code ? myLangRef.current.code : targetLangRef.current.code;

  const getTransTarget = (detectedLang: string) =>
    detectedLang === myLangRef.current.code ? targetLangRef.current.code : myLangRef.current.code;

  const addTranscription = (lang: string, text: string, serverTrans?: string) => {
    entryIdRef.current += 1;
    const id = String(entryIdRef.current);
    setTranscriptions(prev => [...prev, { id, language: lang, text, translatedText: serverTrans ?? '', timestamp: Date.now() }]);
    if (serverTrans) {
      if (shouldPlayTTS(lang)) speakTranslation(serverTrans, getTTSTarget(lang));
      return;
    }
    const tTarget = getTransTarget(lang);
    if (tTarget && tTarget !== lang) {
      translateText(text, lang, tTarget).then(translated => {
        if (!translated) return;
        setTranscriptions(prev => prev.map(e => e.id === id ? { ...e, translatedText: translated } : e));
        if (shouldPlayTTS(lang)) speakTranslation(translated, getTTSTarget(lang));
      });
    }
  };

  // ── Session actions ───────────────────────────────────────────────────────────
  const stopTTS = () => {
    ttsQueueRef.current = [];
    isSpeakingRef.current = false;
    ttsStop();
  };

  // mode-2: earphone default, mode-1 & mode-3: speaker default
  // (mode-3 switches dynamically per TTS item inside processNextTTS)
  const applySpeakerRouting = (id = modeRef.current) =>
    setSpeakerphoneOn(id !== 'mode-2');

  const doStart = async () => {
    if (myLang.code === targetLang.code) {
      const s = UI_STRINGS[myLang.code] ?? UI_STRINGS.en;
      Alert.alert(s.errTitle, s.langMustDiffer);
      return;
    }
    if (isInitializing) return;
    const s = UI_STRINGS[myLang.code] ?? UI_STRINGS.en;
    setSessionError('');
    setIsInitializing(true);
    try {
      // 서버가 준비 안 됐거나 오류 상태면 재시작
      if (serverStatusRef.current !== 'ready') {
        const ok = await probeServer(serverStatusRef.current === 'error');
        if (!ok) throw new Error(s.connectionFailed);
      }
      try {
        await connect({ lang: myLang.code, targetLang: targetLang.code, speed });
      } catch {
        // connect 실패 = serverStatus는 'ready'였지만 서버 실제로 죽어있음 → 강제 재시작
        const ok = await probeServer(true);
        if (!ok) throw new Error(s.connectionFailed);
        await connect({ lang: myLang.code, targetLang: targetLang.code, speed });
      }
      await startRecording();
      applySpeakerRouting();
      sessionStateRef.current = 'recording';
      setSessionState('recording');
      setShowSetup(false);
    } catch (e: any) {
      setSessionError(e?.message ?? s.connectionFailed);
    } finally {
      setIsInitializing(false);
    }
  };

  const doStop = async () => {
    wasAutoSuspendedRef.current = false;
    stopTTS();
    await stopRecording();
    sessionStateRef.current = 'paused';
    setSessionState('paused');
  };

  const doResume = async () => {
    setSessionError('');
    try {
      if (!isConnectedRef.current) {
        if (serverStatusRef.current !== 'ready') {
          setIsInitializing(true);
          const ok = await probeServer();
          if (!ok) throw new Error('Connection failed');
        }
        try {
          await connect({ lang: myLangRef.current.code, targetLang: targetLangRef.current.code });
        } catch {
          // connect 실패 = 서버가 이미 종료됨. 강제로 서버 재시작 후 재연결
          setIsInitializing(true);
          const ok = await probeServer(true);
          if (!ok) throw new Error('Connection failed');
          await connect({ lang: myLangRef.current.code, targetLang: targetLangRef.current.code });
        }
      }
      await startRecording();
      applySpeakerRouting();
      sessionStateRef.current = 'recording';
      setSessionState('recording');
    } catch (e: any) {
      setSessionError(e?.message || 'Connection failed');
    } finally {
      setIsInitializing(false);
    }
  };

  const doBack = () => {
    const s = UI_STRINGS[myLangRef.current.code] ?? UI_STRINGS.en;
    Alert.alert(s.endTitle, s.endMsg, [
      { text: s.cancel, style: 'cancel' },
      {
        text: s.end, style: 'destructive',
        onPress: async () => {
          stopTTS();
          await stopRecording();
          disconnect();
          setTranscriptions([]);
          releaseAudioMode();
          sessionStateRef.current = 'idle';
          setSessionState('idle');
          setShowSetup(false);
        },
      },
    ]);
  };

  // ── Settings updates ──────────────────────────────────────────────────────────
  const updateMyLang = (lang: Language) => {
    setMyLang(lang); myLangRef.current = lang;
    AsyncStorage.setItem(SK.MY, lang.code);
  };
  const updateTargetLang = (lang: Language) => {
    setTargetLang(lang); targetLangRef.current = lang;
    AsyncStorage.setItem(SK.TARGET, lang.code);
  };
  const updateMode = (id: string) => {
    setModeId(id); modeRef.current = id;
    AsyncStorage.setItem(SK.MODE, id);
    if (sessionStateRef.current !== 'idle') {
      stopTTS();
      applySpeakerRouting(id);
    }
  };
  const updateSpeed = (s: 'fast' | 'accurate') => {
    setSpeed(s);
    AsyncStorage.setItem(SK.SPEED, s);
  };

  if (!loaded) return null;

  const ui = UI_STRINGS[myLang.code] ?? UI_STRINGS.en;
  const isIdle = sessionState === 'idle';
  const isRecording = sessionState === 'recording';
  const isPaused = sessionState === 'paused';
  const modeObj = CONVERSATION_MODES.find(m => m.id === modeId) ?? CONVERSATION_MODES[0];
  const bottomPad = Math.max(insets.bottom, 20);

  return (
    <SafeAreaView style={S.container} edges={['top', 'left', 'right']}>
      <StatusBar barStyle="dark-content" backgroundColor="#fff" />

      {/* ── Header ── */}
      <View style={S.header}>
        <View style={S.headerLeft}>
          <TouchableOpacity onPress={() => { setMenuOpen(true); setMenuPage(null); }} style={S.iconBtn}>
            <Ionicons name="menu-outline" size={24} color="#aaa" />
          </TouchableOpacity>
        </View>
        <View style={S.headerLogoWrap} pointerEvents="none">
          <STiTyLogo />
        </View>
        <View style={S.headerRight}>
          {!isIdle && (
            <TouchableOpacity onPress={() => setShowSetup(v => !v)} style={S.iconBtn}>
              <Ionicons name="settings-outline" size={18} color="#ccc" />
            </TouchableOpacity>
          )}
          <StatusDot active={isRecording} />
          <TouchableOpacity
            style={S.iconBtn}
            onPress={() => setIsTTSMuted(v => {
              const next = !v;
              isTTSMutedRef.current = next;
              if (next) stopTTS();
              return next;
            })}
          >
            <Ionicons
              name={isTTSMuted ? 'volume-mute-outline' : 'volume-high-outline'}
              size={20}
              color={isTTSMuted ? '#ccc' : '#7A9030'}
            />
          </TouchableOpacity>
        </View>
      </View>

      {/* ── Setup panel (maxHeight + opacity parallel animation) ── */}
      <Animated.View
        style={{ overflow: 'hidden', maxHeight: setupHeightAnim, opacity: setupOpacityAnim }}
        onLayout={(e) => {
          const h = e.nativeEvent.layout.height;
          if (h > 10 && !isSetupAnimatingRef.current && isSetupVisibleRef.current) {
            if (Math.abs(h - setupNaturalHeight.current) > 2) {
              setupNaturalHeight.current = h;
              setupHeightAnim.setValue(h);
            }
          }
        }}
      >
        <View style={S.setupPanel}>
          <View style={S.langRow}>
            <TouchableOpacity
              style={[S.langCard, { backgroundColor: MY_CARD_COLOR }]}
              onPress={() => setPicker('my')}
              activeOpacity={0.85}
            >
              <Text style={S.langCardLabel}>MY LANGUAGE</Text>
              <Text style={S.langCardValue}>{myLang.name}</Text>
              <Text style={S.langCardArrow}>▾</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={S.langSwapBtn}
              onPress={() => {
                const tmp = myLangRef.current;
                updateMyLang(targetLangRef.current);
                updateTargetLang(tmp);
              }}
              activeOpacity={0.7}
            >
              <Text style={S.langSwap}>⇄</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[S.langCard, { backgroundColor: PEER_CARD_COLOR }]}
              onPress={() => setPicker('peer')}
              activeOpacity={0.85}
            >
              <Text style={S.langCardLabel}>{ui.peerLanguage}</Text>
              <Text style={S.langCardValue}>
                {targetLang.translations[myLang.code] ?? targetLang.name}
              </Text>
              <Text style={S.langCardArrow}>▾</Text>
            </TouchableOpacity>
          </View>

          <View style={S.modeSection}>
            <Text style={S.modeSectionLabel}>{ui.mode}</Text>
            <View style={S.modeSegment}>
              {CONVERSATION_MODES.map(m => (
                <TouchableOpacity
                  key={m.id}
                  style={[S.modeOption, modeId === m.id && S.modeOptionActive]}
                  onPress={() => updateMode(m.id)}
                  activeOpacity={0.8}
                >
                  <Text style={[S.modeOptionTxt, modeId === m.id && S.modeOptionTxtActive]} numberOfLines={1}>
                    {ui.modes[m.id] ?? m.name}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Translation speed pill */}
          <View style={S.speedSection}>
            <Text style={S.speedLabel}>{ui.translation}</Text>
            <TouchableOpacity
              style={[S.speedPill, speed === 'fast' ? S.speedPillFast : S.speedPillAccurate]}
              onPress={() => setPicker('speed')}
              activeOpacity={0.85}
            >
              <View style={[S.speedPillIco, { backgroundColor: speed === 'fast' ? '#6080C8' : '#7A9030' }]}>
                {speed === 'fast' ? (
                  <Svg width={16} height={16} viewBox="0 0 24 24">
                    <Path d="M13 2 L4 14 H11 L10 22 L20 9 H13 Z" fill="#fff" />
                  </Svg>
                ) : (
                  <Svg width={16} height={16} viewBox="0 0 24 24">
                    <Circle cx="12" cy="12" r="9" stroke="#fff" strokeWidth="2.2" fill="none" />
                    <Circle cx="12" cy="12" r="5" stroke="#fff" strokeWidth="2.2" fill="none" />
                    <Circle cx="12" cy="12" r="1.5" fill="#fff" />
                  </Svg>
                )}
              </View>
              <View style={S.speedPillMeta}>
                <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                  <Text style={[S.speedPillName, { color: speed === 'fast' ? '#4865A8' : '#5B7022' }]}>
                    {speed === 'fast' ? ui.fast : ui.accurate}
                  </Text>
                  <View style={[S.speedPillTag, speed === 'fast' ? S.speedPillTagFast : S.speedPillTagAccurate]}>
                    <Text style={[S.speedPillTagTxt, { color: speed === 'fast' ? '#6080C8' : '#7A9030' }]}>
                      {speed === 'fast' ? '≈0.5s' : '≈1.5s'}
                    </Text>
                  </View>
                </View>
                <Text style={S.speedPillSub}>
                  {speed === 'fast' ? ui.fastSub : ui.accurateSub}
                </Text>
              </View>
              <Text style={S.speedPillChev}>›</Text>
            </TouchableOpacity>
          </View>

          {!isIdle && (
            <TouchableOpacity style={S.collapseBtn} onPress={() => setShowSetup(false)}>
              <Text style={S.collapseBtnTxt}>{ui.collapse}</Text>
            </TouchableOpacity>
          )}
        </View>
      </Animated.View>

      {/* ── Session chips (opacity 애니메이션) ── */}
      {!isIdle && !showSetup && (
        <Animated.View style={{
          opacity: chipsAnim,
          transform: [{ translateY: chipsAnim.interpolate({ inputRange: [0, 1], outputRange: [-6, 0] }) }],
        }}>
          <TouchableOpacity style={S.chips} onPress={() => setShowSetup(true)} activeOpacity={0.8}>
            <View style={[S.chip, { backgroundColor: '#f0f4e8' }]}>
              <Text style={[S.chipTxt, { color: '#7A9030' }]}>{myLang.name}</Text>
            </View>
            <Text style={S.chipSep}>↔</Text>
            <View style={[S.chip, { backgroundColor: '#eaf2fb' }]}>
              <Text style={[S.chipTxt, { color: '#6080C8' }]}>
                {targetLang.translations[myLang.code] ?? targetLang.name}
              </Text>
            </View>
            <View style={[S.chip, { backgroundColor: '#f0f0f0' }]}>
              <Text style={[S.chipTxt, { color: '#777' }]}>{ui.modes[modeObj.id] ?? modeObj.name}</Text>
            </View>
            <View style={[S.chip, { backgroundColor: '#f0f0f0' }]}>
              <Text style={[S.chipTxt, { color: '#777' }]}>{speed === 'fast' ? `⚡ ${ui.fast}` : `◎ ${ui.accurate}`}</Text>
            </View>
          </TouchableOpacity>
        </Animated.View>
      )}

      {/* ── Error banner ── */}
      {!!sessionError && serverStatus !== 'ready' && (
        <View style={S.errorBanner}>
          <Text style={S.errorBannerTxt}>{ui.restartApp}</Text>
        </View>
      )}

      {/* ── Conversation area ── */}
      <ScrollView
        ref={scrollRef}
        style={S.convArea}
        contentContainerStyle={[
          S.convContent,
          (isIdle || transcriptions.length === 0) && S.convContentFlex,
        ]}
      >
        {isIdle && (
          <View style={S.idleWrap}>
            <ModeIllustration modeId={modeId} />
            <Text style={S.idleTxt}>{ui.configureAndStart}</Text>
            {__DEV__ && (
              <View style={{ flexDirection: 'row', gap: 6, marginTop: 10 }}>
                <TouchableOpacity
                  style={S.dbgBtn}
                  onPress={() => setServerCapacity({ active: 20, max: 20 })}
                >
                  <Text style={[S.dbgBtnTxt, { color: '#996600' }]}>Full</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[S.dbgBtn, { backgroundColor: '#D4F0D4', borderColor: '#70C070' }]}
                  onPress={() => {
                    setServerCapacity({ active: 20, max: 20 });
                    setTimeout(() => setServerCapacity({ active: 19, max: 20 }), 150);
                  }}
                >
                  <Text style={[S.dbgBtnTxt, { color: '#336633' }]}>Freed</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[S.dbgBtn, { backgroundColor: '#f0f0f0', borderColor: '#ccc' }]}
                  onPress={() => {
                    prevServerFullRef.current = false;
                    setJustFreed(false);
                    setServerCapacity(null);
                  }}
                >
                  <Text style={[S.dbgBtnTxt, { color: '#777' }]}>Clear</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        )}

        {!isIdle && transcriptions.length === 0 && (
          <View style={S.idleWrap}>
            <LinearGradient
              colors={['rgba(122,144,48,0.1)', 'rgba(6,182,212,0.1)']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={S.micIconWrap}
            >
              <Ionicons name="mic" size={26} color="#7A9030" />
            </LinearGradient>
            <Text style={S.idleTxt}>{ui.startSpeaking}</Text>
          </View>
        )}

        {transcriptions.map(entry => (
          <BubbleItem key={entry.id} entry={entry} myLangCode={myLang.code} targetLangCode={targetLang.code} />
        ))}
      </ScrollView>

      {/* ── Recording bar ── */}
      {isRecording && <RecordingBar label={ui.listening} />}


      {/* ── Capacity notice (idle only) ── */}
      {isIdle && (serverFull || justFreed) && (
        <View style={[S.capacityNotice, !serverFull && justFreed && S.capacityNoticeOk]}>
          <CapacityPulse isOk={!serverFull && justFreed} />
          <Text style={[S.capText, !serverFull && justFreed && S.capTextOk]}>
            {serverFull ? ui.capacityFullMsg : ui.capacityFreeMsg}
          </Text>
          {serverCapacity && (
            <View style={[S.capBadge, !serverFull && justFreed && S.capBadgeOk]}>
              <Text style={[S.capBadgeTxt, !serverFull && justFreed && S.capBadgeTxtOk]}>
                {Math.round((serverCapacity.active / serverCapacity.max) * 100)}%
              </Text>
            </View>
          )}
        </View>
      )}

      {/* ── Bottom bar ── */}
      <View style={[S.bottomBar, { paddingBottom: bottomPad + 8 }]}>
        {isIdle && (
          <TouchableOpacity
            style={[S.btn, S.btnOutline,
              serverFull && S.btnFull,
              (serverStatus === 'ec2-starting' || serverStatus === 'connecting') && S.btnDisabled]}
            onPress={doStart}
            activeOpacity={0.85}
            disabled={serverStatus === 'ec2-starting' || serverStatus === 'connecting' || isInitializing || serverFull}
          >
            <Text style={[S.btnTxt, { color: serverFull ? '#bbb' : '#7A9030' }]}>
              {serverFull
                ? ui.waitingForSlot
                : serverStatus === 'ready' || serverStatus === 'idle' || serverStatus === 'error'
                ? ui.start
                : serverStatus === 'ec2-starting'
                ? ui.startingServer
                : `${ui.connectingServer} (${connectPct}%)`}
            </Text>
          </TouchableOpacity>
        )}

        {isRecording && (
          <>
            <TouchableOpacity style={[S.btn, S.btnStop]} onPress={doStop} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#ef4444' }]}>{ui.stop}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[S.btn, S.btnGhost]} onPress={doBack} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#999' }]}>✕</Text>
            </TouchableOpacity>
          </>
        )}

        {isPaused && (
          <>
            <TouchableOpacity style={[S.btn, S.btnPaused]} onPress={doResume} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#aaa' }]}>{ui.resume}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[S.btn, S.btnPaused]} onPress={doBack} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#aaa' }]}>{ui.back}</Text>
            </TouchableOpacity>
          </>
        )}
      </View>

      {/* ── Hamburger menu ── */}
      <Modal visible={menuOpen} animationType="slide" transparent onRequestClose={() => setMenuOpen(false)}>
        <View style={S.pickerOverlay}>
          <TouchableOpacity style={{ flex: 1 }} activeOpacity={1} onPress={() => setMenuOpen(false)} />
          <View style={[S.pickerSheet, { paddingBottom: Math.max(insets.bottom, 16) + 16 }]}>
            {menuPage === null && (
              <ScrollView showsVerticalScrollIndicator={false}>
                <Text style={S.pickerTitle}>{ui.settingsTitle}</Text>

                <Text style={S.menuSectionTitle}>{ui.settingsGeneral}</Text>
                <TouchableOpacity style={S.menuItem} onPress={() => setMenuPage('about')}>
                  <View style={[S.menuIco, { backgroundColor: '#6080C8' }]}>
                    <Ionicons name="information-circle-outline" size={16} color="#fff" />
                  </View>
                  <View style={S.menuMeta}>
                    <Text style={S.menuItemTxt}>{ui.settingsAbout}</Text>
                    <Text style={S.menuItemSub}>{ui.settingsAboutSub}</Text>
                  </View>
                  <Text style={S.menuChev}>›</Text>
                </TouchableOpacity>

                <Text style={S.menuSectionTitle}>{ui.settingsLegal}</Text>
                <TouchableOpacity style={S.menuItem} onPress={() => setMenuPage('privacy')}>
                  <View style={[S.menuIco, { backgroundColor: '#7A9030' }]}>
                    <Ionicons name="shield-checkmark-outline" size={16} color="#fff" />
                  </View>
                  <View style={S.menuMeta}>
                    <Text style={S.menuItemTxt}>{ui.settingsPrivacy}</Text>
                    <Text style={S.menuItemSub}>{ui.settingsPrivacySub}</Text>
                  </View>
                  <Text style={S.menuChev}>›</Text>
                </TouchableOpacity>
                <TouchableOpacity style={S.menuItem} onPress={() => setMenuPage('terms')}>
                  <View style={[S.menuIco, { backgroundColor: '#9BB4D4' }]}>
                    <Ionicons name="document-text-outline" size={16} color="#fff" />
                  </View>
                  <View style={S.menuMeta}>
                    <Text style={S.menuItemTxt}>{ui.settingsTerms}</Text>
                    <Text style={S.menuItemSub}>{ui.settingsTermsSub}</Text>
                  </View>
                  <Text style={S.menuChev}>›</Text>
                </TouchableOpacity>

                <Text style={S.menuSectionTitle}>{ui.settingsData}</Text>
                <TouchableOpacity style={S.menuItem} onPress={() => setMenuPage('delete')}>
                  <View style={[S.menuIco, { backgroundColor: '#ef4444' }]}>
                    <Ionicons name="trash-outline" size={16} color="#fff" />
                  </View>
                  <View style={S.menuMeta}>
                    <Text style={[S.menuItemTxt, { color: '#ef4444' }]}>{ui.settingsDelete}</Text>
                    <Text style={[S.menuItemSub, { color: '#f4a0a0' }]}>{ui.settingsDeleteSub}</Text>
                  </View>
                  <Text style={[S.menuChev, { color: '#f4a0a0' }]}>›</Text>
                </TouchableOpacity>

                <Text style={S.menuDisclaimer}>{ui.settingsDisclaimer}</Text>
                <Text style={S.menuVersion}>STiTy v1.0.0 · Build 2026.05</Text>
              </ScrollView>
            )}

            {menuPage === 'privacy' && (
              <ScrollView showsVerticalScrollIndicator={false}>
                <TouchableOpacity onPress={() => setMenuPage(null)} style={S.detailBack}>
                  <Text style={S.detailBackTxt}>{ui.settingsBack}</Text>
                </TouchableOpacity>
                <Text style={S.detailTitle}>{ui.settingsPrivacy}</Text>
                <Text style={S.detailDate}>{ui.privacyDate}</Text>
                <Text style={S.detailH4}>{ui.privacyH1}</Text>
                <Text style={S.detailP}>{ui.privacyP1}</Text>
                <Text style={S.detailH4}>{ui.privacyH2}</Text>
                <Text style={S.detailBullet}>{ui.privacyB2aStart}<Text style={{ fontWeight: '700' }}>{ui.privacyB2aBold}</Text>{ui.privacyB2aEnd}</Text>
                <Text style={S.detailBullet}>{ui.privacyB2b}</Text>
                <Text style={S.detailBullet}>{ui.privacyB2c}</Text>
                <Text style={S.detailH4}>{ui.privacyH3}</Text>
                <Text style={S.detailP}>{ui.privacyP3}</Text>
                <Text style={S.detailBullet}>{ui.privacyB3a}</Text>
                <Text style={S.detailBullet}>{ui.privacyB3b}</Text>
                <Text style={S.detailP}>{ui.privacyP3b}</Text>
                <Text style={S.detailH4}>{ui.privacyH4}</Text>
                <Text style={S.detailP}>{ui.privacyP4}</Text>
                <Text style={S.detailH4}>{ui.privacyH5}</Text>
                <Text style={S.detailP}>{ui.privacyP5}</Text>
              </ScrollView>
            )}

            {menuPage === 'terms' && (
              <ScrollView showsVerticalScrollIndicator={false}>
                <TouchableOpacity onPress={() => setMenuPage(null)} style={S.detailBack}>
                  <Text style={S.detailBackTxt}>{ui.settingsBack}</Text>
                </TouchableOpacity>
                <Text style={S.detailTitle}>{ui.settingsTerms}</Text>
                <Text style={S.detailDate}>{ui.termsDate}</Text>
                <Text style={S.detailH4}>{ui.termsH1}</Text>
                <Text style={S.detailP}>{ui.termsP1}</Text>
                <Text style={S.detailH4}>{ui.termsH2}</Text>
                <Text style={S.detailP}>{ui.termsP2}</Text>
                <Text style={S.detailH4}>{ui.termsH3}</Text>
                <Text style={S.detailBullet}>{ui.termsB3a}</Text>
                <Text style={S.detailBullet}>{ui.termsB3b}</Text>
                <Text style={S.detailBullet}>{ui.termsB3c}</Text>
                <Text style={S.detailH4}>{ui.termsH5}</Text>
                <Text style={S.detailP}>{ui.termsP5}</Text>
              </ScrollView>
            )}

            {menuPage === 'delete' && (
              <ScrollView showsVerticalScrollIndicator={false}>
                <TouchableOpacity onPress={() => setMenuPage(null)} style={S.detailBack}>
                  <Text style={S.detailBackTxt}>{ui.settingsBack}</Text>
                </TouchableOpacity>
                <Text style={[S.detailTitle, { color: '#ef4444' }]}>{ui.deleteTitle}</Text>
                <Text style={S.detailP}>{ui.deleteP1}</Text>
                <Text style={S.detailBullet}>{ui.deleteB1}</Text>
                <Text style={S.detailBullet}>{ui.deleteB2}</Text>
                <Text style={S.detailBullet}>{ui.deleteB3}</Text>
                <Text style={[S.detailP, { marginTop: 12, color: '#666' }]}>{ui.deleteP2}</Text>
                <TouchableOpacity
                  style={S.deleteBtn}
                  onPress={() => {
                    Alert.alert(ui.deleteAlertTitle, ui.deleteAlertMsg);
                    setMenuOpen(false);
                    setMenuPage(null);
                  }}
                >
                  <Text style={S.deleteBtnTxt}>{ui.deleteBtn}</Text>
                </TouchableOpacity>
                <TouchableOpacity style={S.pickerCancel} onPress={() => setMenuPage(null)}>
                  <Text style={S.pickerCancelTxt}>{ui.cancel}</Text>
                </TouchableOpacity>
              </ScrollView>
            )}

            {menuPage === 'about' && (
              <ScrollView showsVerticalScrollIndicator={false}>
                <TouchableOpacity onPress={() => setMenuPage(null)} style={S.detailBack}>
                  <Text style={S.detailBackTxt}>{ui.settingsBack}</Text>
                </TouchableOpacity>
                <View style={{ alignItems: 'center', paddingVertical: 20 }}>
                  <Text style={S.aboutLogo}>
                    <Text style={{ color: '#6080C8' }}>ST</Text>
                    <Text style={{ color: '#9BB4D4' }}>i</Text>
                    <Text style={{ color: '#7A9030' }}>T</Text>
                    <Text style={{ color: '#E8E0A0' }}>y</Text>
                  </Text>
                  <Text style={S.aboutVersion}>v1.0.0 · Build 2026.05</Text>
                </View>
                <Text style={S.detailP}>{ui.aboutP}</Text>
                <Text style={[S.detailP, { marginTop: 12, color: '#888', fontSize: 11 }]}>{ui.aboutCopyright}</Text>
              </ScrollView>
            )}
          </View>
        </View>
      </Modal>

      {/* ── Language pickers ── */}
      <LangPickerSheet
        visible={picker === 'my'}
        title="My Language"
        selectedCode={myLang.code}
        excludeCode={targetLang.code}
        onSelect={updateMyLang}
        onClose={() => setPicker(null)}
        myLangCode={myLang.code}
        cancelLabel="Cancel"
        forceEnglish
      />
      <LangPickerSheet
        visible={picker === 'peer'}
        title={ui.peerPickerTitle}
        selectedCode={targetLang.code}
        excludeCode={myLang.code}
        onSelect={updateTargetLang}
        onClose={() => setPicker(null)}
        myLangCode={myLang.code}
        cancelLabel={ui.cancel}
      />

      {/* ── Speed picker ── */}
      <Modal visible={picker === 'speed'} animationType="slide" transparent onRequestClose={() => setPicker(null)}>
        <View style={S.pickerOverlay}>
          <TouchableOpacity style={{ flex: 1 }} activeOpacity={1} onPress={() => setPicker(null)} />
          <View style={[S.pickerSheet, { paddingBottom: Math.max(insets.bottom, 16) + 16 }]}>
            <Text style={S.pickerTitle}>{ui.translation}</Text>

            <TouchableOpacity
              style={[S.speedPickerOption, speed === 'fast' && S.speedPickerOptionSelFast]}
              onPress={() => { updateSpeed('fast'); setPicker(null); }}
              activeOpacity={0.85}
            >
              <View style={[S.speedPickerIco, { backgroundColor: '#6080C8' }]}>
                <Svg width={18} height={18} viewBox="0 0 24 24">
                  <Path d="M13 2 L4 14 H11 L10 22 L20 9 H13 Z" fill="#fff" />
                </Svg>
              </View>
              <View style={S.speedPickerMeta}>
                <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                  <Text style={S.speedPickerName}>{ui.fast}</Text>
                  <View style={[S.speedPickerTag, { backgroundColor: 'rgba(96,128,200,0.12)' }]}>
                    <Text style={[S.speedPickerTagTxt, { color: '#6080C8' }]}>≈0.5s</Text>
                  </View>
                </View>
                <Text style={S.speedPickerDesc}>{ui.fastDesc}</Text>
              </View>
              <View style={S.speedPickerCheck}>
                {speed === 'fast' && (
                  <Svg width={18} height={18} viewBox="0 0 24 24">
                    <Path d="M20 6 L9 17 L4 12" stroke="#6080C8" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                  </Svg>
                )}
              </View>
            </TouchableOpacity>

            <TouchableOpacity
              style={[S.speedPickerOption, speed === 'accurate' && S.speedPickerOptionSelAccurate]}
              onPress={() => { updateSpeed('accurate'); setPicker(null); }}
              activeOpacity={0.85}
            >
              <View style={[S.speedPickerIco, { backgroundColor: '#7A9030' }]}>
                <Svg width={18} height={18} viewBox="0 0 24 24">
                  <Circle cx="12" cy="12" r="9" stroke="#fff" strokeWidth="2.2" fill="none" />
                  <Circle cx="12" cy="12" r="5" stroke="#fff" strokeWidth="2.2" fill="none" />
                  <Circle cx="12" cy="12" r="1.5" fill="#fff" />
                </Svg>
              </View>
              <View style={S.speedPickerMeta}>
                <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                  <Text style={S.speedPickerName}>{ui.accurate}</Text>
                  <View style={[S.speedPickerTag, { backgroundColor: 'rgba(122,144,48,0.12)' }]}>
                    <Text style={[S.speedPickerTagTxt, { color: '#7A9030' }]}>≈1.5s</Text>
                  </View>
                </View>
                <Text style={S.speedPickerDesc}>{ui.accurateDesc}</Text>
              </View>
              <View style={S.speedPickerCheck}>
                {speed === 'accurate' && (
                  <Svg width={18} height={18} viewBox="0 0 24 24">
                    <Path d="M20 6 L9 17 L4 12" stroke="#7A9030" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                  </Svg>
                )}
              </View>
            </TouchableOpacity>

            <TouchableOpacity style={S.pickerCancel} onPress={() => setPicker(null)}>
              <Text style={S.pickerCancelTxt}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────
const S = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },

  // Header
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 20, paddingTop: 14, paddingBottom: 2 },
  logo: { fontSize: 28, fontWeight: '900', letterSpacing: -1 },
  headerLeft: { flexDirection: 'row', alignItems: 'center' },
  headerLogoWrap: { position: 'absolute', left: 0, right: 0, alignItems: 'center' },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 10, marginLeft: 'auto' },
  iconBtn: { padding: 4 },
  dotWrap: { width: 20, height: 20, alignItems: 'center', justifyContent: 'center' },
  dotRing: { position: 'absolute', width: 20, height: 20, borderRadius: 10, backgroundColor: '#22c55e' },
  dot: { width: 8, height: 8, borderRadius: 4 },

  // Setup panel
  setupPanel: { paddingHorizontal: 20, paddingTop: 18 },
  langRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 14 },
  langCard: { flex: 1, borderRadius: 16, paddingTop: 12, paddingBottom: 14, paddingHorizontal: 14 },
  langCardLabel: { fontSize: 9, fontWeight: '700', color: 'rgba(255,255,255,0.7)', letterSpacing: 0.5, marginBottom: 5 },
  langCardValue: { fontSize: 16, fontWeight: '700', color: '#fff' },
  langCardArrow: { position: 'absolute', top: 10, right: 12, color: 'rgba(255,255,255,0.6)', fontSize: 12 },
  langSwapBtn: { padding: 8 },
  langSwap: { fontSize: 20, color: '#ccc' },
  modeSection: { marginBottom: 4 },
  modeSectionLabel: { fontSize: 9, fontWeight: '700', color: '#aaa', letterSpacing: 0.5, marginBottom: 6 },
  modeSegment: { flexDirection: 'row', backgroundColor: '#f0f0f0', borderRadius: 12, padding: 3, gap: 2 },
  modeOption: { flex: 1, paddingVertical: 8, borderRadius: 10, alignItems: 'center' },
  modeOptionActive: { backgroundColor: '#fff', shadowColor: '#000', shadowOpacity: 0.08, shadowRadius: 4, elevation: 2 },
  modeOptionTxt: { fontSize: 11, fontWeight: '600', color: '#999', textAlign: 'center', lineHeight: 14 },
  modeOptionTxtActive: { color: MODE_ACTIVE_COLOR },
  collapseBtn: { alignItems: 'flex-end', paddingTop: 8, paddingBottom: 2 },
  collapseBtnTxt: { fontSize: 12, color: '#aaa', fontWeight: '600' },

  // Session chips
  chips: { flexDirection: 'row', paddingHorizontal: 20, paddingTop: 12, paddingBottom: 2, alignItems: 'center', gap: 6, flexWrap: 'wrap' },
  chip: { borderRadius: 100, paddingVertical: 3, paddingHorizontal: 10 },
  chipTxt: { fontSize: 11, fontWeight: '600' },
  chipSep: { fontSize: 11, color: '#aaa' },

  // Error
  errorBanner: { backgroundColor: '#fff0f0', marginHorizontal: 16, marginTop: 6, padding: 10, borderRadius: 12 },
  errorBannerTxt: { color: '#ef4444', fontSize: 12, textAlign: 'center' },

  // Conversation
  convArea: { flex: 1 },
  convContent: { paddingVertical: 12 },
  convContentFlex: { flexGrow: 1 },
  idleWrap: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 40, gap: 12 },
  micIconWrap: { width: 56, height: 56, borderRadius: 28, alignItems: 'center', justifyContent: 'center' },
  idleTxt: { fontSize: 13, color: '#bbb', textAlign: 'center' },

  // Bubbles
  bubbleRow: { flexDirection: 'row', paddingHorizontal: 16, paddingVertical: 5, gap: 8, alignItems: 'flex-end' },
  bubbleRowMine: { flexDirection: 'row-reverse' },
  avatar: { width: 28, height: 28, borderRadius: 14, alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  avatarTxt: { fontSize: 8, fontWeight: '700', color: '#fff' },
  bubble: { maxWidth: '72%', paddingVertical: 10, paddingHorizontal: 14, borderRadius: 18 },
  bubbleMain: { fontSize: 15, fontWeight: '600', color: '#fff', lineHeight: 21 },
  bubbleSub: { fontSize: 11, color: 'rgba(255,255,255,0.65)', marginTop: 3 },

  // Recording bar
  recBar: { flexDirection: 'row', alignItems: 'center', marginHorizontal: 16, marginBottom: 8, backgroundColor: 'rgba(96,128,200,0.07)', borderRadius: 16, paddingVertical: 12, paddingHorizontal: 16, gap: 10 },
  recDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: '#ef4444' },
  recLabel: { flex: 1, fontSize: 12, fontWeight: '600', color: '#6080C8' },
  waveRow: { flexDirection: 'row', alignItems: 'center', gap: 3 },
  waveBar: { width: 3, borderRadius: 2, backgroundColor: '#6080C8' },

  // Bottom bar
  bottomBar: { flexDirection: 'row', paddingHorizontal: 20, paddingTop: 10 , gap: 10 },
  btn: { flex: 1, height: 52, borderRadius: 100, alignItems: 'center', justifyContent: 'center' },
  btnTxt: { fontSize: 15, fontWeight: '600' },
  btnOutline: { borderWidth: 2, borderColor: '#7A9030', backgroundColor: '#fff' },
  btnOutlineCyan: { borderWidth: 2, borderColor: '#6080C8', backgroundColor: '#fff' },
  btnPaused: { borderWidth: 2, borderColor: '#ccc', backgroundColor: 'transparent' },
  btnStop: { backgroundColor: '#fff0f0' },
  btnGhost: { flex: 0, width: 52, backgroundColor: '#f5f5f5' },
  btnDisabled: { opacity: 0.45 },

  // Capacity notice
  capacityNotice: { flexDirection: 'row', alignItems: 'center', gap: 10, marginHorizontal: 20, marginBottom: 4, paddingVertical: 10, paddingHorizontal: 14, backgroundColor: '#fffaf0', borderWidth: 1, borderColor: '#f3e6c4', borderRadius: 14 },
  capacityNoticeOk: { backgroundColor: '#f3faf0', borderColor: '#d8ebc8' },
  capText: { flex: 1, fontSize: 12, fontWeight: '600', color: '#7A6010', lineHeight: 18 },
  capTextOk: { color: '#56711F' },
  capBadge: { paddingVertical: 3, paddingHorizontal: 8, borderRadius: 100, backgroundColor: 'rgba(200,160,48,0.12)' },
  capBadgeOk: { backgroundColor: 'rgba(122,144,48,0.12)' },
  capBadgeTxt: { fontSize: 12, fontWeight: '700', color: '#7A6010' },
  capBadgeTxtOk: { color: '#56711F' },
  btnFull: { borderColor: '#e2e2e2' },

  // Hamburger menu
  menuSectionTitle: { fontSize: 10, fontWeight: '700' as const, color: '#bbb', letterSpacing: 0.5, textTransform: 'uppercase' as const, marginTop: 16, marginBottom: 4, paddingHorizontal: 2 },
  menuItem: { flexDirection: 'row' as const, alignItems: 'center' as const, paddingVertical: 11, gap: 12 },
  menuIco: { width: 32, height: 32, borderRadius: 8, alignItems: 'center' as const, justifyContent: 'center' as const },
  menuMeta: { flex: 1 },
  menuItemTxt: { fontSize: 15, fontWeight: '600' as const, color: '#1a1a1a' },
  menuItemSub: { fontSize: 11, color: '#aaa', marginTop: 1 },
  menuChev: { fontSize: 20, color: '#ccc', lineHeight: 24 },
  menuDisclaimer: { fontSize: 11, color: '#aaa', lineHeight: 16, marginTop: 20, paddingTop: 14, borderTopWidth: 1, borderTopColor: '#f0f0f0' },
  menuVersion: { fontSize: 11, color: '#ccc', textAlign: 'center' as const, marginTop: 10, marginBottom: 4 },
  // Detail pages
  detailBack: { paddingBottom: 8 },
  detailBackTxt: { fontSize: 15, color: '#6080C8', fontWeight: '600' as const },
  detailTitle: { fontSize: 20, fontWeight: '800' as const, color: '#1a1a1a', marginBottom: 4 },
  detailDate: { fontSize: 11, color: '#aaa', marginBottom: 14 },
  detailH4: { fontSize: 13, fontWeight: '700' as const, color: '#1a1a1a', marginTop: 14, marginBottom: 4 },
  detailP: { fontSize: 13, color: '#555', lineHeight: 20 },
  detailBullet: { fontSize: 13, color: '#555', lineHeight: 20, marginLeft: 6 },
  deleteBtn: { marginTop: 16, paddingVertical: 14, borderRadius: 14, backgroundColor: '#ef4444', alignItems: 'center' as const },
  deleteBtnTxt: { fontSize: 14, fontWeight: '700' as const, color: '#fff' },
  aboutLogo: { fontSize: 42, fontWeight: '900' as const, letterSpacing: -1.5 },
  aboutVersion: { fontSize: 12, color: '#aaa', marginTop: 4 },

  // Speed pill
  speedSection: { marginBottom: 4, paddingTop: 6 },
  speedLabel: { fontSize: 9, fontWeight: '700' as const, color: '#aaa', letterSpacing: 0.5, textTransform: 'uppercase' as const, marginBottom: 6 },
  speedPill: { flexDirection: 'row' as const, alignItems: 'center' as const, gap: 10, paddingVertical: 10, paddingHorizontal: 12, borderRadius: 12, borderWidth: 1.5, borderColor: '#ececec', backgroundColor: '#fafafa' },
  speedPillFast: { borderColor: '#d6e3f6', backgroundColor: '#f4f8fd' },
  speedPillAccurate: { borderColor: '#d9e6c6', backgroundColor: '#f6faef' },
  speedPillIco: { width: 30, height: 30, borderRadius: 8, alignItems: 'center' as const, justifyContent: 'center' as const },
  speedPillMeta: { flex: 1 },
  speedPillName: { fontSize: 14, fontWeight: '700' as const, lineHeight: 16 },
  speedPillTag: { paddingVertical: 2, paddingHorizontal: 6, borderRadius: 100 },
  speedPillTagFast: { backgroundColor: 'rgba(96,128,200,0.12)' },
  speedPillTagAccurate: { backgroundColor: 'rgba(122,144,48,0.12)' },
  speedPillTagTxt: { fontSize: 9.5, fontWeight: '700' as const },
  speedPillSub: { fontSize: 11, fontWeight: '500' as const, color: '#888', marginTop: 2 },
  speedPillChev: { color: '#bbb', fontSize: 18, lineHeight: 20 },
  // Speed picker
  speedPickerOption: { flexDirection: 'row' as const, alignItems: 'flex-start' as const, gap: 12, paddingVertical: 14, paddingHorizontal: 12, borderRadius: 14, borderWidth: 1.5, borderColor: 'transparent', marginBottom: 4 },
  speedPickerOptionSelFast: { backgroundColor: '#f4f8fd', borderColor: '#d6e3f6' },
  speedPickerOptionSelAccurate: { backgroundColor: '#f6faef', borderColor: '#d9e6c6' },
  speedPickerIco: { width: 36, height: 36, borderRadius: 10, alignItems: 'center' as const, justifyContent: 'center' as const },
  speedPickerMeta: { flex: 1, paddingTop: 1 },
  speedPickerName: { fontSize: 15, fontWeight: '700' as const, color: '#1a1a1a', lineHeight: 18 },
  speedPickerTag: { paddingVertical: 2, paddingHorizontal: 7, borderRadius: 100 },
  speedPickerTagTxt: { fontSize: 10, fontWeight: '700' as const },
  speedPickerDesc: { fontSize: 12, fontWeight: '500' as const, color: '#666', lineHeight: 17 },
  speedPickerCheck: { width: 22, height: 22, alignItems: 'center' as const, justifyContent: 'center' as const, alignSelf: 'center' as const },

  // Debug (dev-only)
  dbgBtn: { paddingVertical: 4, paddingHorizontal: 10, borderRadius: 8, backgroundColor: '#FFF3CD', borderWidth: 1, borderColor: '#F0C040' },
  dbgBtnTxt: { fontSize: 11, fontWeight: '600' as const },

  // Language picker
  pickerOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.3)', justifyContent: 'flex-end' },
  pickerSheet: { backgroundColor: '#fff', borderTopLeftRadius: 24, borderTopRightRadius: 24, paddingTop: 20, paddingHorizontal: 20, maxHeight: '75%' },
  pickerTitle: { fontSize: 13, fontWeight: '700', color: '#aaa', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 12 },
  pickerOption: { paddingVertical: 13, paddingHorizontal: 14, borderRadius: 12 },
  pickerOptionSel: { backgroundColor: '#eaf2fb' },
  pickerOptionTxt: { fontSize: 15, fontWeight: '600', color: '#1a1a1a' },
  pickerOptionTxtSel: { color: '#6080C8' },
  pickerCancel: { marginTop: 8, paddingVertical: 14, borderRadius: 14, backgroundColor: '#f5f5f5', alignItems: 'center' },
  pickerCancelTxt: { fontSize: 15, fontWeight: '600', color: '#aaa' },
});

export default HomeScreen;
