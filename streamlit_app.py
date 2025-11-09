import streamlit as st
import sys
import os
import tempfile
from PIL import Image
import traceback

st.set_page_config(
    page_title="AI Hair Transformation",
    page_icon="💇",
    layout="wide"
)

st.title("💇 AI Hair Transformation")
st.info("🚀 Professional Hair Analysis & Styling Recommendations")

# Initialize the transformer with better error handling
@st.cache_resource
def load_transformer():
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from hair_transformation.utils.hair_ai import StreamlitHairTransformation
        
        with st.spinner("🔄 Loading AI models... This may take a minute."):
            transformer = StreamlitHairTransformation()
            st.success("✅ AI models loaded successfully!")
            return transformer
    except Exception as e:
        st.error(f"❌ Failed to load AI models: {e}")
        st.info("The app will use basic functionality.")
        return None

transformer = load_transformer()

# File upload section
st.header("📤 Upload Your Photo")
uploaded_file = st.file_uploader("Choose a clear face photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🚀 Analyze & Recommend Hairstyles", type="primary"):
        with st.spinner("🔍 Analyzing your features and generating recommendations..."):
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                if transformer:
                    results = transformer.process_image(tmp_path, "user_session")
                    
                    if results:
                        st.success("✅ Analysis Complete!")
                        
                        # Display analysis results
                        st.header("📊 Your Analysis Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Skin Tone", results['analysis_data']['skin_tone'])
                            st.metric("Face Shape", results['analysis_data']['face_shape'])
                        with col2:
                            st.metric("Hair Length", results['analysis_data']['hair_length'])
                            st.metric("Hair Texture", results['analysis_data']['hair_texture'])
                        with col3:
                            st.metric("Hair Coverage", f"{results['analysis_data']['hair_coverage']:.1f}%")
                            st.metric("Ethnicity", results['analysis_data']['ethnicity'])
                        
                        # Display style recommendations
                        st.header("💡 Recommended Hairstyles")
                        for i, style in enumerate(results['recommendations']['styles'], 1):
                            st.write(f"{i}. **{style}**")
                        
                        # Display transformations
                        st.header("🎨 Visualized Hairstyles")
                        
                        # Show all images
                        for title, img in results['images']:
                            st.subheader(title)
                            st.image(img, use_column_width=True)
                            
                    else:
                        st.error("❌ Analysis failed. Please try with a different image.")
                else:
                    st.warning("⚠️ AI models not available. Using basic analysis.")
                    # Show basic image info
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    st.info("For full AI-powered analysis, please ensure all models are properly loaded.")
                    
            except Exception as e:
                st.error(f"❌ Processing error: {e}")
                st.code(traceback.format_exc())
            finally:
                # Clean up
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Sidebar information
st.sidebar.header("ℹ️ About This App")
st.sidebar.info("""
This AI-powered app analyzes your facial features and provides personalized hairstyle recommendations.

**Features:**
- Face shape detection
- Skin tone analysis  
- Hair type classification
- Personalized style recommendations
- Visual hair transformations

**Tips for best results:**
- Use clear, well-lit photos
- Face should be clearly visible
- Remove hats or heavy accessories
- Natural hair color works best
""")

st.sidebar.header("🔧 Technical Info")
st.sidebar.code("""
Models Used:
- SegFormer (Hair Segmentation)
- OpenCV (Face Detection)
- Custom AI (Style Recommendations)
""")