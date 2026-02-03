import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { COLORS, FONTS, SPACING } from '../constants/theme';

interface TranslationItemProps {
  sourceLang: string;
  targetLang: string;
  sourceText: string;
  targetText: string;
  isLatest?: boolean;
}

export const TranslationItem: React.FC<TranslationItemProps> = ({
  sourceLang,
  targetLang,
  sourceText,
  targetText,
  isLatest = false,
}) => {
  const getLabelColor = (lang: string) => {
    switch (lang) {
      case 'ko':
        return COLORS.gradientStart;  // Purple
      case 'en':
        return COLORS.gradientMiddle; // Blue
      case 'es':
        return '#E94E77';             // Spanish - Red/Pink
      case 'ja':
        return '#FF6B6B';             // Japanese - Coral
      case 'zh':
        return '#4ECDC4';             // Chinese - Teal
      case 'id':
        return COLORS.gradientEnd;    // Indonesian - Cyan
      case 'vi':
        return '#45B7D1';             // Vietnamese - Sky Blue
      case 'th':
        return '#96CEB4';             // Thai - Sage Green
      case 'fr':
        return '#DDA0DD';             // French - Plum
      case 'de':
        return '#FFB347';             // German - Orange
      default:
        return COLORS.gradientMiddle;
    }
  };

  return (
    <View style={[styles.container, isLatest && styles.latestContainer]}>
      {/* Source language row (transcription) */}
      <View style={styles.row}>
        <Text style={[styles.langLabel, { color: getLabelColor(sourceLang) }]}>
          {sourceLang}
        </Text>
        <Text style={[styles.text, isLatest && styles.latestText]}>
          {sourceText}
        </Text>
      </View>

      {/* Target language row - 번역이 있을 때만 표시 */}
      {targetText && targetText.length > 0 && (
        <View style={styles.row}>
          <Text style={[styles.langLabel, { color: getLabelColor(targetLang) }]}>
            {targetLang}
          </Text>
          <Text style={[styles.translatedText, isLatest && styles.latestTranslatedText]}>
            {targetText}
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingVertical: SPACING.md,
    paddingHorizontal: SPACING.md,
  },
  latestContainer: {
    // Slightly emphasized for latest item
  },
  row: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginVertical: SPACING.xs,
  },
  langLabel: {
    fontSize: FONTS.sizes.sm,
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
  latestText: {
    fontSize: FONTS.sizes.xl,
    fontWeight: '600',
  },
  translatedText: {
    flex: 1,
    fontSize: FONTS.sizes.md,
    color: COLORS.textPrimary,
    textAlign: 'center',
  },
  latestTranslatedText: {
    fontSize: FONTS.sizes.lg,
  },
});

export default TranslationItem;
