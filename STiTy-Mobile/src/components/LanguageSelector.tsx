import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Modal,
  FlatList,
  StyleSheet,
  SafeAreaView,
} from 'react-native';
import { COLORS, FONTS, SPACING, BORDER_RADIUS } from '../constants/theme';
import { LANGUAGES, Language, formatLanguageDisplay } from '../constants/languages';

interface LanguageSelectorProps {
  label: string;
  selectedLanguage: Language;
  onSelect: (language: Language) => void;
  excludeLanguage?: string;
}

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  label,
  selectedLanguage,
  onSelect,
  excludeLanguage,
}) => {
  const [modalVisible, setModalVisible] = useState(false);

  const filteredLanguages = excludeLanguage
    ? LANGUAGES.filter(lang => lang.code !== excludeLanguage)
    : LANGUAGES;

  const handleSelect = (language: Language) => {
    onSelect(language);
    setModalVisible(false);
  };

  // label이 비어있으면 HomeScreen에서 직접 label을 관리하는 것
  const showLabel = label && label.length > 0;

  return (
    <View style={[styles.container, !showLabel && styles.containerNoLabel]}>
      {showLabel && <Text style={styles.label}>{label}</Text>}
      <TouchableOpacity
        style={styles.selector}
        onPress={() => setModalVisible(true)}
        activeOpacity={0.7}
      >
        <Text style={styles.selectedText}>
          {formatLanguageDisplay(selectedLanguage)}
        </Text>
        <Text style={styles.arrow}>▼</Text>
      </TouchableOpacity>

      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <SafeAreaView style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>언어 선택</Text>
              <TouchableOpacity
                onPress={() => setModalVisible(false)}
                style={styles.closeButton}
              >
                <Text style={styles.closeText}>✕</Text>
              </TouchableOpacity>
            </View>

            <FlatList
              data={filteredLanguages}
              keyExtractor={(item) => item.code}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={[
                    styles.languageItem,
                    item.code === selectedLanguage.code && styles.selectedItem,
                  ]}
                  onPress={() => handleSelect(item)}
                >
                  <Text
                    style={[
                      styles.languageText,
                      item.code === selectedLanguage.code && styles.selectedLanguageText,
                    ]}
                  >
                    {item.nativeName}
                  </Text>
                  <Text style={styles.languageCode}>({item.code})</Text>
                </TouchableOpacity>
              )}
              ItemSeparatorComponent={() => <View style={styles.separator} />}
            />
          </View>
        </SafeAreaView>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: SPACING.sm,
  },
  containerNoLabel: {
    flex: 1,
    justifyContent: 'flex-start',
    marginVertical: 0,
  },
  label: {
    fontSize: FONTS.sizes.md,
    fontWeight: '600',
    color: COLORS.gradientStart,
    marginRight: SPACING.md,
    minWidth: 80,
  },
  selector: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: SPACING.xs,
  },
  selectedText: {
    fontSize: FONTS.sizes.md,
    color: COLORS.textPrimary,
    marginRight: SPACING.sm,
  },
  arrow: {
    fontSize: 10,
    color: COLORS.textMuted,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: COLORS.background,
    borderTopLeftRadius: BORDER_RADIUS.xl,
    borderTopRightRadius: BORDER_RADIUS.xl,
    maxHeight: '70%',
    paddingBottom: SPACING.xl,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: SPACING.lg,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  modalTitle: {
    fontSize: FONTS.sizes.lg,
    fontWeight: '600',
    color: COLORS.textPrimary,
  },
  closeButton: {
    padding: SPACING.sm,
  },
  closeText: {
    fontSize: FONTS.sizes.lg,
    color: COLORS.textMuted,
  },
  languageItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: SPACING.md,
    paddingHorizontal: SPACING.lg,
  },
  selectedItem: {
    backgroundColor: COLORS.backgroundSecondary,
  },
  languageText: {
    fontSize: FONTS.sizes.md,
    color: COLORS.textPrimary,
  },
  selectedLanguageText: {
    color: COLORS.gradientMiddle,
    fontWeight: '600',
  },
  languageCode: {
    fontSize: FONTS.sizes.sm,
    color: COLORS.textMuted,
  },
  separator: {
    height: 1,
    backgroundColor: COLORS.border,
    marginHorizontal: SPACING.lg,
  },
});

export default LanguageSelector;
