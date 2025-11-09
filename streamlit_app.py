
import streamlit as st
import os
import sys
from PIL import Image
import tempfile
import traceback

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure the page first
st.set_page_config(
    page_title="AI Hair Transformation",
    page_icon="💇",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("💇 AI-Powered Hair Transformation")
    st.markdown("Upload your photo to see how different hairstyles would look on you!")
    
    try:
        # Test OpenCV import first
        import cv2
        st.success("✅ OpenCV loaded successfully!")
    except ImportError as e:
        st.error(f"❌ OpenCV failed to load: {e}")
        return
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # Sidebar for upload
    with st.sidebar:
        st.header("📤 Upload Your Photo")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear front-facing photo for best results"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🚀 Analyze & Transform Hair", type="primary", use_container_width=True):
                process_image(uploaded_file)

def process_image(uploaded_file):
    """Process the uploaded image with error handling"""
    try:
        with st.spinner("🔄 Loading AI models... This may take a few minutes for first use."):
            # Import here to catch errors
            try:
                from hair_transformation.utils.hair_ai import StreamlitHairTransformation
                transformer = StreamlitHairTransformation()
                st.success("✅ AI models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load AI models: {str(e)}")
                st.info("This might be due to large model downloads. Please wait and try again.")
                return
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process the image
        with st.spinner("🔍 Analyzing your features..."):
            results = transformer.process_image(tmp_path, "session_123")
        
        if results:
            st.session_state.processed = True
            st.session_state.results = results
            st.success("✅ Analysis complete!")
            display_results(results)
        else:
            st.error("❌ Processing failed. Please try another image.")
            
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.code(traceback.format_exc())
    finally:
        # Clean up temporary file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def display_results(results):
    """Display the transformation results"""
    st.header("🎯 Your Personal Hair Analysis")
    
    # Analysis results in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Skin & Face")
        st.metric("Skin Tone", results['analysis_data']['skin_tone'])
        st.metric("Ethnicity", results['analysis_data']['ethnicity'])
        st.metric("Face Shape", results['analysis_data']['face_shape'])
    
    with col2:
        st.subheader("💇 Hair Analysis")
        st.metric("Hair Length", results['analysis_data']['hair_length'])
        st.metric("Hair Texture", results['analysis_data']['hair_texture'])
        st.metric("Coverage", f"{results['analysis_data']['hair_coverage']}%")
    
    with col3:
        st.subheader("📊 Analysis Images")
        if 'hair_analysis' in results['images']:
            st.image(results['images']['hair_analysis'], caption="Hair Analysis", use_column_width=True)
    
    # Style recommendations
    st.header("💡 Style Recommendations")
    
    st.subheader("🎨 Recommended Hair Colors")
    colors = results['recommendations']['colors']
    for color in colors:
        st.write(f"• {color}")
    
    # Transformations
    st.header("🎨 Virtual Hair Transformations")
    
    transformations = results['images']['transformations']
    
    if transformations:
        # Group by style type
        long_styles = [t for t in transformations if t['style_type'] == 'Long']
        short_styles = [t for t in transformations if t['style_type'] == 'Short']
        
        # Display long styles
        if long_styles:
            st.subheader("💫 Long Styles")
            cols = st.columns(len(long_styles))
            for idx, style in enumerate(long_styles):
                with cols[idx]:
                    st.image(style['image'], caption=style['title'], use_column_width=True)
        
        # Display short styles
        if short_styles:
            st.subheader("✂️ Short Styles")
            cols = st.columns(len(short_styles))
            for idx, style in enumerate(short_styles):
                with cols[idx]:
                    st.image(style['image'], caption=style['title'], use_column_width=True)
    else:
        st.info("No transformations available. This might be due to model loading issues.")
    
    # Reset button
    if st.button("🔄 Analyze Another Photo", use_container_width=True):
        st.session_state.processed = False
        st.session_state.results = None
        st.rerun()

if __name__ == "__main__":
    main()