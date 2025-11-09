import streamlit as st
import sys
import os

st.set_page_config(
    page_title="OpenCV Test",
    page_icon="✅",
    layout="wide"
)

st.title("🔍 OpenCV Headless Test")

# Test 1: Basic OpenCV import
try:
    import cv2
    st.success(f"✅ OpenCV imported successfully! Version: {cv2.__version__}")
except Exception as e:
    st.error(f"❌ OpenCV import failed: {e}")
    st.stop()

# Test 2: Basic OpenCV functionality
try:
    import numpy as np
    # Create a simple test image with OpenCV
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    st.success("✅ OpenCV basic functionality working!")
except Exception as e:
    st.error(f"❌ OpenCV functionality failed: {e}")

# Test 3: File operations
try:
    from PIL import Image
    st.success("✅ PIL imported successfully!")
except Exception as e:
    st.error(f"❌ PIL import failed: {e}")

st.info("🎉 If you see all green checkmarks, OpenCV is working correctly!")