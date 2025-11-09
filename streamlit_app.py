import streamlit as st
import sys
import os
import tempfile
from PIL import Image

st.set_page_config(
    page_title="AI Hair Transformation",
    page_icon="💇",
    layout="wide"
)

st.title("💇 AI Hair Transformation - Model Test")

# Test 1: OpenCV (already confirmed working)
try:
    import cv2
    st.success(f"✅ OpenCV: {cv2.__version__}")
except Exception as e:
    st.error(f"❌ OpenCV: {e}")
    st.stop()

# Test 2: PyTorch and Transformers
try:
    import torch
    import transformers
    st.success(f"✅ PyTorch: {torch.__version__}")
    st.success(f"✅ Transformers: {transformers.__version__}")
except Exception as e:
    st.error(f"❌ AI frameworks: {e}")
    st.stop()

# Test 3: Hair transformation model
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from hair_transformation.utils.hair_ai import StreamlitHairTransformation
    st.success("✅ Hair transformation model imported!")
    
    # Test initialization
    with st.spinner("🔄 Initializing AI models..."):
        transformer = StreamlitHairTransformation()
        st.success("✅ AI models initialized successfully!")
        
except Exception as e:
    st.error(f"❌ Hair transformation model failed: {e}")
    st.info("This might be due to large model downloads. Let's try a simple upload test first.")

# Basic file upload functionality
st.header("📤 Test Image Upload")
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🚀 Test Basic Processing"):
        with st.spinner("Processing..."):
            try:
                # Save to temp file and test processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                # Try basic processing
                if 'transformer' in locals():
                    results = transformer.process_image(tmp_path, "test_session")
                    if results:
                        st.success("✅ Full processing successful!")
                        st.json({
                            "skin_tone": results['analysis_data']['skin_tone'],
                            "face_shape": results['analysis_data']['face_shape'],
                            "hair_length": results['analysis_data']['hair_length']
                        })
                    else:
                        st.warning("⚠️ Processing returned no results (might be expected for test)")
                else:
                    st.info("🤖 AI models not fully loaded, but basic functionality works!")
                    
            except Exception as e:
                st.error(f"❌ Processing error: {e}")
            finally:
                # Clean up
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

st.info("🎯 Next: If this works, we'll add the full transformation interface!")