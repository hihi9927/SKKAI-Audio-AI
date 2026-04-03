import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Speech from 'expo-speech';
import { COLORS, FONTS, SPACING } from '../constants/theme';

interface TranslationItemProps {
  sourceLang: string;
  targetLang: string;
  sourceText: string;
  targetText: string;
  isLatest?: boolean;
  translationOnly?: boolean;
}

export const TranslationItem: React.FC<TranslationItemProps> = ({
  sourceLang,
  targetLang,
  sourceText,
  targetText,
  isLatest = false,
  translationOnly = false,
}) => {
  const getLabelColor = (lang: string) => {
    switch (lang) {
      case 'ko':
        return COLORS.gradientStart;  // Purple
      case 'en':
        return COLORS.gradientMiddle; // Blue
      case 'es':
        return '#E94E77';
      case 'ja':
        return '#FF6B6B';
      case 'zh':
        return '#4ECDC4';
      case 'id':
        return COLORS.gradientEnd;
      case 'vi':
        return '#45B7D1';
      case 'th':
        return '#96CEB4';
      case 'fr':
        return '#DDA0DD';
      case 'de':
        return '#FFB347';
      default:
        return COLORS.gradientMiddle;
    }
  };

  const toBCP47 = (code: string): string => {
    const map: Record<string, string> = {
      ko: 'ko-KR', en: 'en-US', ja: 'ja-JP', zh: 'zh-CN',
      id: 'id-ID', vi: 'vi-VN', th: 'th-TH',
      es: 'es-ES', fr: 'fr-FR', de: 'de-DE',
    };
    return map[code] ?? code;
  };

  const speakText = (text: string, lang: string) => {
    Speech.stop();
    const bcp = toBCP47(lang);
    Speech.getAvailableVoicesAsync().then(voices => {
      const voiceId = (
        voices.find(v => v.language === bcp && v.identifier.toLowerCase().includes('google')) ??
        voices.find(v => v.language.startsWith(lang) && v.identifier.toLowerCase().includes('google')) ??
        voices.find(v => v.language === bcp)
      )?.identifier;
      Speech.speak(text, { language: bcp, ...(voiceId ? { voice: voiceId } : {}), rate: 1.6 });
    }).catch(() => {
      Speech.speak(text, { language: bcp, rate: 1.6 });
    });
  };

  return (
    <View style={[styles.container, isLatest && styles.latestContainer]}>
      {/* Source language row - translationOnly면 숨김 */}
      {!translationOnly && (
        <View style={styles.rowWrapper}>
          <View style={styles.row}>
            <Text style={[styles.langLabel, { color: getLabelColor(sourceLang) }]}>
              {sourceLang}
            </Text>
            <Text style={styles.text}>{sourceText}</Text>
            <TouchableOpacity
              onPress={() => speakText(sourceText, sourceLang)}
              style={styles.speakerBtn}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
            >
              <Ionicons name="volume-medium-outline" size={23} color={getLabelColor(sourceLang)} />
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* Target language row - 번역이 있을 때만 표시 */}
      {targetText && targetText.length > 0 && (
        <View style={styles.rowWrapper}>
          <View style={styles.row}>
            <Text style={[styles.langLabel, { color: getLabelColor(targetLang) }]}>
              {targetLang}
            </Text>
            <Text style={[styles.translatedText, isLatest && styles.latestTranslatedText]}>
              {targetText}
            </Text>
            <TouchableOpacity
              onPress={() => speakText(targetText, targetLang)}
              style={styles.speakerBtn}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
            >
              <Ionicons name="volume-medium-outline" size={23} color={getLabelColor(targetLang)} />
            </TouchableOpacity>
          </View>
          {/* 원본 - 번역 아래에 원문 표시 */}
          {sourceText ? (
            <Text style={styles.originalText}>{sourceText}</Text>
          ) : null}
        </View>
      )}

      {/* translationOnly + 번역 없는 live 항목: 원문만 표시 */}
      {translationOnly && (!targetText || targetText.length === 0) && (
        <View style={styles.rowWrapper}>
          <View style={styles.row}>
            <Text style={[styles.langLabel, { color: getLabelColor(sourceLang) }]}>
              {sourceLang}
            </Text>
            <Text style={styles.text}>{sourceText}</Text>
            <TouchableOpacity
              onPress={() => speakText(sourceText, sourceLang)}
              style={styles.speakerBtn}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
            >
              <Ionicons name="volume-medium-outline" size={23} color={getLabelColor(sourceLang)} />
            </TouchableOpacity>
          </View>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingVertical: SPACING.lg,
    paddingHorizontal: SPACING.md,
  },
  latestContainer: {},
  rowWrapper: {
    marginVertical: SPACING.xs,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  langLabel: {
    fontSize: FONTS.sizes.lg,
    fontWeight: '700',
    width: 30,
    marginRight: SPACING.md,
  },
  text: {
    flex: 1,
    fontSize: FONTS.sizes.lg,
    color: COLORS.textPrimary,
    fontWeight: '500',
    textAlign: 'center',
  },
  translatedText: {
    flex: 1,
    fontSize: FONTS.sizes.lg,
    color: COLORS.textPrimary,
    fontWeight: '500',
    textAlign: 'center',
  },
  latestTranslatedText: {
    fontSize: FONTS.sizes.lg,
  },
  speakerBtn: {
    paddingLeft: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  originalText: {
    marginTop: 4,
    fontSize: 14,  // 18 * 0.75
    color: COLORS.textMuted,
    textAlign: 'center',
  },
});

export default TranslationItem;
