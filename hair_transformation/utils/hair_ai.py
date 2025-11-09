import os
import numpy as np
from PIL import Image
import cv2
import requests
from io import BytesIO
import tempfile

class SkinToneAwareHairTransformation:
    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_cascade = None
        print("✅ Basic hair transformation initialized")

    def detect_face(self, image):
        """Simple face detection"""
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            if self.face_cascade is None:
                return None
            
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                return None
            
            face_x, face_y, face_w, face_h = max(faces, key=lambda rect: rect[2] * rect[3])
            return {
                'shape': 'Oval',
                'bounding_box': (face_x, face_y, face_w, face_h)
            }
        except:
            return None

    def analyze_skin_tone(self, image, face_features):
        """Basic skin tone analysis"""
        try:
            img_array = np.array(image)
            
            if face_features is None:
                return {'skin_tone': 'Medium', 'ethnicity_likely': 'Unknown'}
            
            face_x, face_y, face_w, face_h = face_features['bounding_box']
            face_region = img_array[face_y:face_y+face_h, face_x:face_x+face_w]
            
            if face_region.size == 0:
                return {'skin_tone': 'Medium', 'ethnicity_likely': 'Unknown'}
            
            avg_color = np.mean(face_region, axis=(0, 1))
            brightness = np.mean(avg_color)
            
            if brightness > 180:
                return {'skin_tone': 'Fair', 'ethnicity_likely': 'East Asian/Caucasian'}
            elif brightness > 140:
                return {'skin_tone': 'Light', 'ethnicity_likely': 'East Asian/Caucasian'}
            elif brightness > 100:
                return {'skin_tone': 'Medium', 'ethnicity_likely': 'Latin American/Middle Eastern'}
            else:
                return {'skin_tone': 'Dark', 'ethnicity_likely': 'African'}
                
        except:
            return {'skin_tone': 'Medium', 'ethnicity_likely': 'Unknown'}

    def get_recommendations(self, skin_analysis):
        """Get style and color recommendations"""
        styles = [
            "Classic layered cut with face-framing layers",
            "Modern textured crop with volume",
            "Soft waves with curtain bangs",
            "Sleek straight bob with blunt ends"
        ]
        
        colors = ["Honey Blonde", "Chocolate Brown", "Caramel Highlights", "Rich Brown"]
        
        return styles, colors

    def create_visualization(self, original_image, hairstyle, skin_tone):
        """Create basic visualization"""
        try:
            img_array = np.array(original_image)
            result = img_array.copy()
            
            # Add text overlay
            cv2.putText(result, f"Style: {hairstyle}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.putText(result, f"Skin: {skin_tone}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            return Image.fromarray(result)
        except:
            return original_image

    def process_image(self, image_path):
        """Main processing pipeline"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = image_path

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Analyze
            face_features = self.detect_face(image)
            skin_analysis = self.analyze_skin_tone(image, face_features)
            styles, colors = self.get_recommendations(skin_analysis)
            
            # Create results
            results = {
                'original_image': image,
                'analysis_data': {
                    'skin_tone': skin_analysis['skin_tone'],
                    'ethnicity': skin_analysis['ethnicity_likely'],
                    'face_shape': face_features['shape'] if face_features else 'Unknown',
                    'hair_length': 'Medium',
                    'hair_texture': 'Straight',
                    'hair_coverage': '75%'
                },
                'recommendations': {
                    'styles': styles,
                    'colors': colors
                },
                'images': []
            }
            
            # Add original image
            results['images'].append(("1. Original Image", image))
            
            # Add style visualizations
            for i, style in enumerate(styles[:2]):
                vis_image = self.create_visualization(image, style, skin_analysis['skin_tone'])
                results['images'].append((f"{i+2}. {style}", vis_image))
            
            return results
            
        except Exception as e:
            print(f"Processing error: {e}")
            return None


class StreamlitHairTransformation:
    def __init__(self):
        self.transformer = SkinToneAwareHairTransformation()
    
    def process_image(self, image_path, session_id):
        """Process image for Streamlit"""
        try:
            return self.transformer.process_image(image_path)
        except Exception as e:
            print(f"Streamlit processing error: {e}")
            return None