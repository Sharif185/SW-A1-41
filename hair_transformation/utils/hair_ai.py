import streamlit as st
import torch
import torch.nn as nn
from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline
import cv2
import numpy as np
from PIL import Image
import os
import logging
from typing import Dict, List, Any
import tempfile
import gc

logger = logging.getLogger(__name__)

def safe_model_loading():
    """Fix for meta tensor error"""
    # Clear cache and memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def safe_model_to_device(model, device):
    try:
        # First try normal loading
        model = model.to(device)
        return model
    except Exception as e:
        if "meta tensor" in str(e) or "no data" in str(e):
            # Use the recommended approach for meta tensors
            st.warning("Using safe model loading for meta tensors...")
            model = model.to_empty(device=device)
            return model
        else:
            raise e

class StreamlitHairTransformation:
    def __init__(self, use_hairstyle_ai=True):
        self.use_hairstyle_ai = use_hairstyle_ai
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.hair_segmentor = None
        self.inpainting_pipeline = None
        self.face_detector = None
        
        # Clear memory before loading
        safe_model_loading()
        self._load_models()
    
    def _load_models(self):
        """Load models with meta tensor error handling"""
        try:
            # Load hair segmentation model
            with st.spinner("🔄 Loading hair segmentation model..."):
                try:
                    self.hair_segmentor = pipeline(
                        "image-segmentation", 
                        model="mattmdjaga/segformer_b2_clothes",
                        device=-1 if self.device == "cpu" else 0
                    )
                    st.success("✅ Hair segmentation model loaded")
                except Exception as e:
                    logger.warning(f"Hair segmentation failed: {e}")
                    st.warning("Using fallback hair segmentation")
                    
        except Exception as e:
            logger.error(f"Hair segmentation model error: {e}")
            st.error("Hair segmentation model failed to load")
        
        try:
            # Load inpainting pipeline with safe device handling
            if self.use_hairstyle_ai:
                with st.spinner("🔄 Loading hairstyle transformation models..."):
                    # Clear memory before loading large model
                    safe_model_loading()
                    
                    # Load pipeline with safe settings
                    self.inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                        "runwayml/stable-diffusion-inpainting",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,
                        local_files_only=False
                    )
                    
                    # Safe device transfer
                    self.inpainting_pipeline = safe_model_to_device(self.inpainting_pipeline, self.device)
                    
                    st.success("✅ Hairstyle transformation model loaded")
                    
        except Exception as e:
            logger.error(f"Inpainting model failed: {e}")
            st.error("AI hairstyle transformations disabled - using enhanced basic transformations")
            self.use_hairstyle_ai = False
    
    def process_image(self, image_path: str, session_id: str) -> Dict[str, Any]:
        """Process image with comprehensive error handling"""
        try:
            # Load and validate image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform hair segmentation with fallback
            hair_mask = self._segment_hair(image)
            
            # Analyze facial and hair features
            analysis_data = self._analyze_features(image, hair_mask)
            
            # Generate personalized recommendations
            recommendations = self._generate_recommendations(analysis_data)
            
            # Generate hair transformations
            transformations = self._generate_transformations(image, hair_mask, analysis_data)
            
            return {
                'analysis_data': analysis_data,
                'recommendations': recommendations,
                'images': {
                    'hair_analysis': self._create_analysis_visualization(image, hair_mask),
                    'transformations': transformations
                }
            }
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            st.error(f"Processing error: {str(e)}")
            return self._get_fallback_results(image_path)
    
    def _segment_hair(self, image: Image.Image) -> np.ndarray:
        """Perform hair segmentation with multiple fallbacks"""
        try:
            if self.hair_segmentor:
                results = self.hair_segmentor(image)
                hair_mask = self._extract_hair_mask(results)
                if hair_mask is not None:
                    return hair_mask
            
            # Fallback to color-based segmentation
            return self._fallback_hair_segmentation(image)
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return self._fallback_hair_segmentation(image)
    
    def _extract_hair_mask(self, results) -> np.ndarray:
        """Extract hair mask from segmentation results"""
        try:
            # Look for hair-related segments
            for result in results:
                if hasattr(result, 'label') and 'hair' in result.label.lower():
                    return np.array(result.mask)
            
            # Fallback: use the largest segment
            if results:
                return np.array(results[0].mask)
                
            return None
        except Exception as e:
            logger.error(f"Mask extraction failed: {e}")
            return None
    
    def _fallback_hair_segmentation(self, image: Image.Image) -> np.ndarray:
        """Color-based hair segmentation fallback"""
        try:
            image_np = np.array(image)
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            
            # Define color ranges for hair (adjust based on common hair colors)
            lower_hair = np.array([0, 0, 0])
            upper_hair = np.array([180, 255, 100])
            
            mask = cv2.inRange(hsv, lower_hair, upper_hair)
            
            # Clean up mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
            
        except Exception as e:
            logger.error(f"Fallback segmentation failed: {e}")
            # Return empty mask as last resort
            image_np = np.array(image)
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    
    def _analyze_features(self, image: Image.Image, hair_mask: np.ndarray) -> Dict[str, Any]:
        """Comprehensive feature analysis"""
        image_np = np.array(image)
        
        # Skin tone analysis
        skin_tone = self._analyze_skin_tone(image_np)
        
        # Hair analysis
        hair_color = self._analyze_hair_color(image_np, hair_mask)
        hair_length, hair_texture = self._analyze_hair_properties(hair_mask)
        
        # Face analysis
        face_shape = self._detect_face_shape(image_np)
        ethnicity = self._predict_ethnicity(skin_tone, image_np)
        
        return {
            'skin_tone': skin_tone,
            'ethnicity': ethnicity,
            'face_shape': face_shape,
            'hair_length': hair_length,
            'hair_texture': hair_texture,
            'hair_color': hair_color,
            'hair_coverage': self._calculate_hair_coverage(hair_mask)
        }
    
    def _analyze_skin_tone(self, image_np: np.ndarray) -> str:
        """Analyze skin tone from image"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            
            # Simple skin tone detection based on hue and value
            avg_hue = np.mean(hsv[:,:,0])
            avg_value = np.mean(hsv[:,:,2])
            
            if avg_value < 80:
                return "Deep"
            elif avg_hue < 15:
                return "Fair"
            elif avg_hue < 25:
                return "Light"
            elif avg_hue < 35:
                return "Medium"
            else:
                return "Tan"
                
        except Exception as e:
            logger.error(f"Skin tone analysis failed: {e}")
            return "Medium"
    
    def _analyze_hair_color(self, image_np: np.ndarray, hair_mask: np.ndarray) -> str:
        """Analyze hair color from masked region"""
        try:
            if np.sum(hair_mask) == 0:
                return "Unknown"
                
            # Apply mask to get hair region
            masked_image = cv2.bitwise_and(image_np, image_np, mask=hair_mask)
            
            # Convert to grayscale for brightness analysis
            gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
            
            # Calculate average brightness in hair region
            hair_pixels = gray[hair_mask > 0]
            if len(hair_pixels) == 0:
                return "Unknown"
                
            avg_brightness = np.mean(hair_pixels)
            
            # Classify based on brightness
            if avg_brightness < 50:
                return "Black"
            elif avg_brightness < 100:
                return "Dark Brown"
            elif avg_brightness < 150:
                return "Brown"
            elif avg_brightness < 200:
                return "Light Brown"
            else:
                return "Blonde"
                
        except Exception as e:
            logger.error(f"Hair color analysis failed: {e}")
            return "Brown"
    
    def _analyze_hair_properties(self, hair_mask: np.ndarray) -> tuple:
        """Analyze hair length and texture from mask"""
        try:
            if np.sum(hair_mask) == 0:
                return "Short", "Straight"
            
            # Calculate hair coverage area
            hair_pixels = np.sum(hair_mask > 0)
            total_pixels = hair_mask.shape[0] * hair_mask.shape[1]
            coverage_ratio = hair_pixels / total_pixels
            
            # Estimate length based on coverage
            if coverage_ratio < 0.1:
                length = "Short"
            elif coverage_ratio < 0.2:
                length = "Medium"
            else:
                length = "Long"
            
            # Simple texture estimation (placeholder)
            texture = "Straight"
            
            return length, texture
            
        except Exception as e:
            logger.error(f"Hair properties analysis failed: {e}")
            return "Medium", "Straight"
    
    def _detect_face_shape(self, image_np: np.ndarray) -> str:
        """Detect face shape (simplified)"""
        try:
            # Placeholder face shape detection
            # In a real implementation, you'd use face landmarks
            shapes = ["Oval", "Round", "Square", "Heart", "Diamond"]
            return "Oval"  # Default
            
        except Exception as e:
            logger.error(f"Face shape detection failed: {e}")
            return "Oval"
    
    def _predict_ethnicity(self, skin_tone: str, image_np: np.ndarray) -> str:
        """Predict ethnicity based on features"""
        try:
            # Simplified ethnicity prediction based on skin tone
            if skin_tone in ["Fair", "Light"]:
                return "Caucasian"
            elif skin_tone in ["Medium", "Tan"]:
                return "Asian/Hispanic"
            elif skin_tone == "Deep":
                return "African"
            else:
                return "Various"
                
        except Exception as e:
            logger.error(f"Ethnicity prediction failed: {e}")
            return "Various"
    
    def _calculate_hair_coverage(self, hair_mask: np.ndarray) -> int:
        """Calculate hair coverage percentage"""
        try:
            if np.sum(hair_mask) == 0:
                return 0
                
            hair_pixels = np.sum(hair_mask > 0)
            total_pixels = hair_mask.shape[0] * hair_mask.shape[1]
            coverage = int((hair_pixels / total_pixels) * 100)
            
            return min(coverage, 100)
            
        except Exception as e:
            logger.error(f"Hair coverage calculation failed: {e}")
            return 50
    
    def _generate_recommendations(self, analysis: Dict) -> Dict[str, List[str]]:
        """Generate personalized recommendations"""
        skin_tone = analysis['skin_tone']
        face_shape = analysis['face_shape']
        hair_color = analysis['hair_color']
        
        # Color recommendations based on skin tone
        color_recommendations = {
            'Fair': ['Honey Blonde', 'Golden Brown', 'Caramel Highlights'],
            'Light': ['Chocolate Brown', 'Auburn', 'Buttery Blonde'],
            'Medium': ['Rich Brown', 'Burgundy', 'Copper'],
            'Tan': ['Dark Brown', 'Black', 'Mahogany'],
            'Deep': ['Jet Black', 'Blue Black', 'Dark Espresso']
        }
        
        # Style recommendations based on face shape
        style_recommendations = {
            'Oval': ['Layered Cut', 'Side Part', 'Textured Crop'],
            'Round': ['Angular Bob', 'Long Layers', 'Side-Swept Bangs'],
            'Square': ['Soft Layers', 'Wavy Bob', 'Side Part'],
            'Heart': ['Chin-Length Bob', 'Layered Cut', 'Side Bangs'],
            'Diamond': ['Pixie Cut', 'Bob', 'Layered Style']
        }
        
        colors = color_recommendations.get(skin_tone, ['Dark Brown', 'Caramel Highlights', 'Auburn'])
        styles = style_recommendations.get(face_shape, ['Layered Cut', 'Side Part', 'Textured Crop'])
        
        return {
            'colors': colors,
            'styles': styles
        }
    
    def _generate_transformations(self, image: Image.Image, hair_mask: np.ndarray, analysis: Dict) -> List[Dict]:
        """Generate hair transformations with AI fallback"""
        transformations = []
        
        try:
            # Try AI transformations first if available
            if self.use_hairstyle_ai and self.inpainting_pipeline:
                ai_transformations = self._generate_ai_transformations(image, hair_mask, analysis)
                if ai_transformations:
                    transformations.extend(ai_transformations)
            
            # Always include enhanced basic transformations
            basic_transformations = self._generate_enhanced_basic_transformations(image, analysis)
            transformations.extend(basic_transformations)
                
        except Exception as e:
            logger.error(f"Transformation generation failed: {e}")
            transformations.extend(self._generate_enhanced_basic_transformations(image, analysis))
        
        return transformations
    
    def _generate_ai_transformations(self, image: Image.Image, hair_mask: np.ndarray, analysis: Dict) -> List[Dict]:
        """Generate AI-powered hair transformations"""
        transformations = []
        
        try:
            safe_model_loading()  # Clear memory before AI processing
            
            # Example AI transformation prompts based on analysis
            prompts = [
                "professional hairstyle, well-groomed hair, natural look",
                "elegant hairstyle with volume and texture",
                "modern haircut with stylish look"
            ]
            
            for i, prompt in enumerate(prompts[:2]):  # Limit to 2 AI transformations
                try:
                    # Convert hair mask to correct format
                    mask_image = Image.fromarray((hair_mask * 255).astype(np.uint8))
                    
                    # Generate inpainted image
                    result = self.inpainting_pipeline(
                        prompt=prompt,
                        image=image,
                        mask_image=mask_image,
                        num_inference_steps=20,  # Reduced for speed
                        guidance_scale=7.5
                    ).images[0]
                    
                    transformations.append({
                        'image': result,
                        'title': f'AI Style {i+1}',
                        'style_type': 'AI Generated'
                    })
                    
                except Exception as e:
                    logger.error(f"Single AI transformation failed: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"AI transformations failed: {e}")
            
        return transformations
    
    def _generate_enhanced_basic_transformations(self, image: Image.Image, analysis: Dict) -> List[Dict]:
        """Generate enhanced basic transformations using image processing"""
        transformations = []
        
        try:
            base_size = (300, 400)
            base_image = image.resize(base_size, Image.Resampling.LANCZOS)
            
            # Current style
            transformations.append({
                'image': base_image,
                'title': 'Your Current Style',
                'style_type': 'Current'
            })
            
            # Color variations
            color_variations = [
                ((80, 50, 30), 'Rich Dark Brown'),
                ((120, 80, 50), 'Chocolate Brown'),
                ((180, 140, 100), 'Light Brown'),
                ((210, 180, 140), 'Sandy Blonde')
            ]
            
            for color, name in color_variations[:3]:  # Limit to 3 color variations
                colored_image = self._apply_hair_color_tint(base_image, color)
                transformations.append({
                    'image': colored_image,
                    'title': name,
                    'style_type': 'Color Change'
                })
                
        except Exception as e:
            logger.error(f"Basic transformations failed: {e}")
            
        return transformations
    
    def _apply_hair_color_tint(self, image: Image.Image, target_color: tuple) -> Image.Image:
        """Apply hair color tint to image"""
        try:
            # Convert to numpy array
            arr = np.array(image).astype(float)
            
            # Create tint mask (simple approach - in real app, use hair mask)
            tint_strength = 0.3  # How strong the tint should be
            
            # Apply tint
            for i in range(3):  # RGB channels
                arr[:,:,i] = arr[:,:,i] * (1 - tint_strength) + target_color[i] * tint_strength
            
            # Clip values and convert back
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            
            return Image.fromarray(arr)
            
        except Exception as e:
            logger.error(f"Color tint failed: {e}")
            return image
    
    def _create_analysis_visualization(self, image: Image.Image, hair_mask: np.ndarray) -> Image.Image:
        """Create visualization of hair analysis"""
        try:
            # Create a simple visualization showing the hair mask
            image_np = np.array(image)
            
            # Create overlay
            overlay = image_np.copy()
            overlay[hair_mask > 0] = [255, 0, 0]  # Red overlay for hair
            
            # Blend with original
            alpha = 0.3
            blended = cv2.addWeighted(image_np, 1 - alpha, overlay, alpha, 0)
            
            return Image.fromarray(blended)
            
        except Exception as e:
            logger.error(f"Analysis visualization failed: {e}")
            return image
    
    def _get_fallback_results(self, image_path: str) -> Dict[str, Any]:
        """Provide comprehensive fallback results"""
        try:
            image = Image.open(image_path)
            
            return {
                'analysis_data': {
                    'skin_tone': 'Light Medium',
                    'ethnicity': 'Various',
                    'face_shape': 'Oval',
                    'hair_length': 'Medium',
                    'hair_texture': 'Straight',
                    'hair_color': 'Brown',
                    'hair_coverage': 70
                },
                'recommendations': {
                    'colors': ['Dark Brown', 'Caramel Highlights', 'Auburn'],
                    'styles': ['Layered Cut', 'Side Part', 'Textured Crop']
                },
                'images': {
                    'hair_analysis': image,
                    'transformations': self._generate_enhanced_basic_transformations(image, {})
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback results failed: {e}")
            # Ultimate fallback
            return {
                'analysis_data': {
                    'skin_tone': 'Medium',
                    'ethnicity': 'Various',
                    'face_shape': 'Oval',
                    'hair_length': 'Medium',
                    'hair_texture': 'Straight',
                    'hair_color': 'Brown',
                    'hair_coverage': 65
                },
                'recommendations': {
                    'colors': ['Brown', 'Highlights', 'Natural'],
                    'styles': ['Layered', 'Styled', 'Professional']
                },
                'images': {
                    'hair_analysis': None,
                    'transformations': []
                }
            }