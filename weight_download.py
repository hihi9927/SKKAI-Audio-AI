import whisper

# Download and load whisper model
for model_size in ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo']:
    model = whisper.load_model(model_size, model_dir="weights")