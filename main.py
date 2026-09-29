import warnings
import matplotlib.pyplot as plt

# Suppress warnings for clean console output
warnings.filterwarnings("ignore")
plt.switch_backend('Agg')

from av_pipeline.ui import build_gradio_app
from av_pipeline.audio import AVTSEEngine, SileroVADProcessor, StreamingASREngine
from av_pipeline.brain import JevBrainRouter
from av_pipeline.vision import PixelJevGrounder, ActionSerializer

def main():
    print("Initializing components...")
    # Initialize the core pipeline instances (this runs on CPU/GPU as detected)
    _avtse = AVTSEEngine()
    _avtse.load_model()
    _vad = SileroVADProcessor()
    _asr = StreamingASREngine()
    _asr.load_model()
    _jev = JevBrainRouter()
    _grounder = PixelJevGrounder()
    _serializer = ActionSerializer(screen_width=1920, screen_height=1080)
    
    print("Building Gradio App...")
    app = build_gradio_app(
        avtse=_avtse,
        vad=_vad,
        asr=_asr,
        jev=_jev,
        grounder=_grounder,
        serializer=_serializer
    )
    
    print("Launching Gradio Server...")
    # share=False for local use, set to True if tunneling is needed.
    app.launch(share=False, show_error=True, inbrowser=True)

if __name__ == "__main__":
    main()
