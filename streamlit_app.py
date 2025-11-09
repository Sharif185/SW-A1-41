import streamlit as st
import sys
import os
import tempfile
from PIL import Image

st.set_page_config(
    page_title="Hair Style Advisor",
    page_icon="💇",
    layout="wide"
)

st.title("💇 Hair Style Advisor")
st.info("Get personalized hairstyle recommendations based on your features")

# Initialize the transformer
@st.cache_resource
def load_transformer():
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from hair_transformation.utils.hair_ai import StreamlitHairTransformation
        return StreamlitHairTransformation()
    except Exception as e:
        st.error(f"Failed to load: {e}")
        return None

transformer = load_transformer()

# File upload
st.header("📤 Upload Your Photo")
uploaded_file = st.file_uploader("Choose a clear face photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🚀 Get Recommendations", type="primary"):
        with st.spinner("Analyzing your features..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                if transformer:
                    results = transformer.process_image(tmp_path, "user_session")
                    
                    if results:
                        st.success("✅ Analysis Complete!")
                        
                        # Display analysis
                        st.header("📊 Your Analysis")
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Skin Tone", results['analysis_data']['skin_tone'])
                            st.metric("Face Shape", results['analysis_data']['face_shape'])
                        with cols[1]:
                            st.metric("Hair Length", results['analysis_data']['hair_length'])
                            st.metric("Hair Texture", results['analysis_data']['hair_texture'])
                        with cols[2]:
                            st.metric("Hair Coverage", results['analysis_data']['hair_coverage'])
                            st.metric("Ethnicity", results['analysis_data']['ethnicity'])
                        
                        # Recommendations
                        st.header("💡 Recommended Hairstyles")
                        for i, style in enumerate(results['recommendations']['styles'], 1):
                            st.write(f"{i}. **{style}**")
                        
                        st.header("🎨 Recommended Hair Colors")
                        st.write(" • ".join([f"**{color}**" for color in results['recommendations']['colors']]))
                        
                        # Images
                        st.header("🖼️ Style Visualizations")
                        for title, img in results['images']:
                            st.subheader(title)
                            st.image(img, use_column_width=True)
                            
                    else:
                        st.error("Analysis failed. Please try with a different image.")
                else:
                    st.error("Service not available.")
                    
            except Exception as e:
                st.error(f"Processing error: {e}")
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Info
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
This app provides personalized hairstyle recommendations based on basic feature analysis.

**Features:**
- Face detection
- Skin tone analysis  
- Style recommendations
- Color suggestions

Upload a clear face photo to get started!
""")