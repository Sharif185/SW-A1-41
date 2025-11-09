import streamlit as st
import os
import uuid
import sys
from PIL import Image
import tempfile
import json

# Add your Django app to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your AI model
from hair_transformation.utils.hair_ai import DjangoHairTransformation

# Page configuration
st.set_page_config(
    page_title="AI Hair Transformation",
    page_icon="💇",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("💇 AI-Powered Hair Transformation")
    st.markdown("Upload your photo to see how different hairstyles would look on you!")
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Sidebar for upload
    with st.sidebar:
        st.header("Upload Your Photo")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear front-facing photo for best results"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🚀 Analyze & Transform Hair", type="primary"):
                process_image(uploaded_file)
    
    # Display results if available
    if st.session_state.results:
        display_results()

def process_image(uploaded_file):
    """Process the uploaded image with hair transformation"""
    st.session_state.processing = True
    st.session_state.session_id = str(uuid.uuid4())
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Show processing status
        progress_placeholder = st.empty()
        progress_placeholder.info("🔄 Starting hair analysis...")
        
        # Initialize transformer
        progress_placeholder.info("📦 Loading AI models...")
        transformer = DjangoHairTransformation()
        
        # Process image
        progress_placeholder.info("🔍 Analyzing your features...")
        results = transformer.process_image(tmp_path, st.session_state.session_id)
        
        if results:
            st.session_state.results = results
            progress_placeholder.success("✅ Analysis complete!")
        else:
            progress_placeholder.error("❌ Processing failed. Please try another image.")
            
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        st.session_state.processing = False

def display_results():
    """Display the transformation results"""
    results = st.session_state.results
    
    st.header("🎯 Your Personal Hair Analysis")
    
    # Analysis results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Skin & Face Analysis")
        st.write(f"**Skin Tone:** {results['analysis_data']['skin_tone']}")
        st.write(f"**Ethnicity:** {results['analysis_data']['ethnicity']}")
        st.write(f"**Face Shape:** {results['analysis_data']['face_shape']}")
    
    with col2:
        st.subheader("Hair Analysis")
        st.write(f"**Hair Length:** {results['analysis_data']['hair_length']}")
        st.write(f"**Hair Texture:** {results['analysis_data']['hair_texture']}")
        st.write(f"**Coverage:** {results['analysis_data']['hair_coverage']}%")
    
    with col3:
        st.subheader("Analysis Images")
        if 'hair_analysis' in results['images']:
            st.image(results['images']['hair_analysis'], caption="Hair Analysis", use_column_width=True)
    
    # Style recommendations
    st.header("💡 Style Recommendations")
    
    st.subheader("Recommended Hair Colors")
    colors = results['recommendations']['colors']
    st.write(", ".join(colors))
    
    # Transformations
    st.header("🎨 Virtual Hair Transformations")
    
    transformations = results['images']['transformations']
    
    # Group by style type
    long_styles = [t for t in transformations if t['style_type'] == 'Long']
    short_styles = [t for t in transformations if t['style_type'] == 'Short']
    
    # Display long styles
    if long_styles:
        st.subheader("Long Styles")
        cols = st.columns(len(long_styles))
        for idx, style in enumerate(long_styles):
            with cols[idx]:
                st.image(style['image'], caption=style['title'], use_column_width=True)
    
    # Display short styles
    if short_styles:
        st.subheader("Short Styles")
        cols = st.columns(len(short_styles))
        for idx, style in enumerate(short_styles):
            with cols[idx]:
                st.image(style['image'], caption=style['title'], use_column_width=True)
    
    # Reset button
    if st.button("🔄 Analyze Another Photo"):
        st.session_state.results = None
        st.session_state.session_id = None
        st.rerun()

if __name__ == "__main__":
    main()