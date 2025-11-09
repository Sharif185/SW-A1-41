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
    
    # Initialize AI models with progress tracking
    try:
        from hair_transformation.utils.hair_ai import StreamlitHairTransformation
        
        with st.spinner("🔄 Loading AI models... This may take 3-5 minutes for first use."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Show loading progress
            for i in range(5):
                progress_bar.progress((i + 1) * 20)
                steps = [
                    "Loading hair segmentation...",
                    "Loading face detection...", 
                    "Loading skin analysis...",
                    "Loading transformation models...",
                    "Finalizing..."
                ]
                status_text.text(steps[i])
                # Small delay to show progress
                import time
                time.sleep(2)
            
            transformer = StreamlitHairTransformation()
            
        progress_bar.progress(100)
        status_text.text("✅ AI models loaded successfully!")
        
        # Show which models are available
        if hasattr(transformer.transformer, 'models_used'):
            st.info(f"**Loaded models:** {', '.join(transformer.transformer.models_used)}")
        
        # Check if AI transformations are available
        if not transformer.transformer.use_hairstyle_ai:
            st.warning("⚠️ AI transformations limited - using enhanced basic transformations")
        else:
            st.success("🎨 Full AI transformations available!")
        
    except Exception as e:
        st.error(f"❌ Failed to load AI models: {e}")
        st.info("Please refresh the page and try again.")
        transformer = None
        return
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
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
                if transformer:
                    process_image(uploaded_file, transformer)
                else:
                    st.error("AI models not available.")

def process_image(uploaded_file, transformer):
    """Process the uploaded image with error handling"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process the image with progress indication
        with st.spinner("🔍 Analyzing your features... This may take 2-3 minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress (you can replace with actual progress callbacks)
            steps = ["Detecting face", "Analyzing skin tone", "Segmenting hair", "Generating transformations"]
            for i, step in enumerate(steps):
                progress_bar.progress((i + 1) * 25)
                status_text.text(f"{step}...")
                # Add small delay to show progress
                import time
                time.sleep(1)
            
            results = transformer.process_image(tmp_path, "user_session")
        
        if results:
            st.session_state.processed = True
            st.session_state.results = results
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
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
        st.info(f"**Skin Tone:** {results['analysis_data']['skin_tone']}")
        st.info(f"**Ethnicity:** {results['analysis_data']['ethnicity']}")
        st.info(f"**Face Shape:** {results['analysis_data']['face_shape']}")
    
    with col2:
        st.subheader("💇 Hair Analysis")
        st.info(f"**Hair Length:** {results['analysis_data']['hair_length']}")
        st.info(f"**Hair Texture:** {results['analysis_data']['hair_texture']}")
        st.info(f"**Coverage:** {results['analysis_data']['hair_coverage']}%")
    
    with col3:
        st.subheader("📊 Analysis Preview")
        if 'hair_analysis' in results['images']:
            st.image(results['images']['hair_analysis'], caption="Hair Analysis", use_column_width=True)
    
    # Style recommendations
    st.header("💡 Style Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Recommended Hair Colors")
        colors = results['recommendations']['colors']
        for color in colors[:4]:  # Show top 4 colors
            st.write(f"• {color}")
    
    with col2:
        st.subheader("💫 Recommended Styles")
        styles = results['recommendations']['styles']
        for style in styles[:4]:  # Show top 4 styles
            st.write(f"• {style}")
    
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
        st.info("No transformations generated. This might be due to image quality or model limitations.")
    
    # Reset button
    st.markdown("---")
    if st.button("🔄 Analyze Another Photo", type="primary", use_container_width=True):
        st.session_state.processed = False
        st.session_state.results = None
        st.rerun()

# Display existing results if available
if 'processed' in st.session_state and st.session_state.processed and st.session_state.results:
    display_results(st.session_state.results)

if __name__ == "__main__":
    main()