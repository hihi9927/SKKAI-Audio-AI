import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ViewStyle, TextStyle, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import MaskedView from '@react-native-masked-view/masked-view';

interface GradientButtonProps {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  variant?: 'filled' | 'outline';
  style?: ViewStyle;
  textStyle?: TextStyle;
}

export const GradientButton: React.FC<GradientButtonProps> = ({
  title,
  onPress,
  disabled = false,
  style,
  textStyle,
}) => {
  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled}
      activeOpacity={0.8}
      style={[disabled && styles.disabled, style]}
    >
      <LinearGradient
        colors={['#8E54E9', '#4776E6', '#00CFEF']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
        style={styles.gradient}
      >
        <View style={styles.inner}>
          <MaskedView
            maskElement={<Text style={[styles.text, textStyle]}>{title}</Text>}
          >
            <LinearGradient
              colors={['#8E54E9', '#4776E6', '#00CFEF']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
            >
              <Text style={[styles.text, textStyle, { opacity: 0 }]}>{title}</Text>
            </LinearGradient>
          </MaskedView>
        </View>
      </LinearGradient>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  gradient: {
    borderRadius: 28,
    padding: 1.5,
  },
  inner: {
    backgroundColor: '#FFFFFF',
    borderRadius: 26.5,
    paddingVertical: 10,
    paddingHorizontal: 36,
    alignItems: 'center',
  },
  text: {
    fontSize: 17,
    fontWeight: 'bold',
  },
  disabled: {
    opacity: 0.5,
  },
});

export default GradientButton;
