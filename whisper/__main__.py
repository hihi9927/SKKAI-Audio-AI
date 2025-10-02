"""
Whisper Offline Real-time STT Example

This example demonstrates how to use Whisper for offline real-time speech-to-text.
Run with: python -m whisper
"""

import sys
import os

def main():
    print("🎤 Whisper Offline Real-time STT")
    print("=" * 40)
    print()
    
    try:
        import whisper
        print("✓ Whisper module loaded successfully")
        
        # Check available models
        models = whisper.available_models()
        print(f"✓ Available models: {len(models)}")
        print(f"  Recommended for real-time: 'tiny', 'base', 'small'")
        print()
        
        # Check if weights directory exists
        weights_dir = os.path.join(os.getcwd(), "weights")
        if os.path.exists(weights_dir):
            weights_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
            if weights_files:
                print(f"✓ Found {len(weights_files)} model files in weights/:")
                for f in weights_files:
                    print(f"  - {f}")
            else:
                print("⚠️  weights/ directory exists but no .pt files found")
        else:
            print("⚠️  weights/ directory not found")
            print("   Create weights/ directory and download model files")
        
        print()
        print("📖 Basic Usage:")
        print("   import whisper")
        print("   model = whisper.load_model('base', download_root='./weights')")
        print("   result = whisper.transcribe(model, 'audio.wav')")
        print("   print(result['text'])")
        print()
        print("🚀 Ready for real-time STT integration!")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("   Please install required dependencies")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
