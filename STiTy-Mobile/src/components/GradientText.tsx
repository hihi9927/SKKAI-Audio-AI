import React from 'react';
import { Text, View, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

// PDF 그라데이션 색상: 0% #8E54E9, 50% #4776E6, 100% #00CFEF
// 글자별 그라데이션 적용: S(보라) → T(보라-파랑) → i(파랑) → T(파랑-청록) → y(청록)

// Logo component with the STiTy branding - PDF 디자인 정확히 반영
export const Logo: React.FC<{ size?: 'small' | 'medium' | 'large' }> = ({ size = 'large' }) => {
  const fontSize = size === 'large' ? 48 : size === 'medium' ? 36 : 24;

  return (
    <View style={styles.logoContainer}>
      <Text style={[styles.logoText, { fontSize }]}>
        <Text style={{ color: '#8E54E9' }}>S</Text>
        <Text style={{ color: '#6A65E8' }}>T</Text>
        <Text style={{ color: '#4776E6' }}>i</Text>
        <Text style={{ color: '#23A6EA' }}>T</Text>
        <Text style={{ color: '#00CFEF' }}>y</Text>
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  logoContainer: {
    alignItems: 'center',
  },
  logoText: {
    fontWeight: 'bold',
  },
});

export default Logo;
