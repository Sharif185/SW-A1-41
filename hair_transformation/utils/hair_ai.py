import os
import uuid
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
import requests
from io import BytesIO
from sklearn.cluster import KMeans
import warnings
import logging
import tempfile

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SkinToneAwareHairTransformation:
    def __init__(self, use_hairstyle_ai=False):  # DISABLE heavy models for deployment
        self.hairstyles_dataset = []
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            print(f"⚠ Face cascade loading warning: {e}")
            self.face_cascade = None
        
        # Model tracking
        self.models_used = []
        
        # Load the SegFormer model for hair segmentation
        print("📦 Loading SegFormer model for hair segmentation...")
        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            self.processor = AutoImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
            self.models_used.append("mattmdjaga/segformer_b2_clothes (Hair Segmentation)")
            print("✅ SegFormer model loaded successfully")
        except Exception as e:
            print(f"   ⚠ Could not load SegFormer: {e}")
            self.processor = None
            self.model = None
        
        # DISABLE hairstyle transformation models for deployment
        self.use_hairstyle_ai = False  # Force disable
        self.hairstyle_pipe = None
        
        print("✅ Lightweight hair transformation initialized (AI models disabled for deployment)")

    def _get_head_hair_mask(self, image_np, face_features, all_masks):
        """Extract head hair mask while excluding facial hair (beards)"""
        if face_features is None:
            print("   ⚠ No face features for head hair extraction")
            return all_masks
            
        face_x, face_y, face_w, face_h = face_features['bounding_box']
        
        # Create head region mask (above face) - focus on scalp hair only
        head_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        
        # Define head region: above face, wider than face
        head_top = max(0, face_y - int(face_h * 1.5))
        head_bottom = face_y + int(face_h * 0.3)
        head_left = max(0, face_x - int(face_w * 0.3))
        head_right = min(image_np.shape[1], face_x + face_w + int(face_w * 0.3))
        
        head_mask[head_top:head_bottom, head_left:head_right] = 1
        
        # Filter masks to only include hair in head region (scalp hair only)
        head_hair_mask = np.zeros_like(all_masks)
        
        # Find contours in the original mask
        contours, _ = cv2.findContours(all_masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding box of contour
            x, y, w, h = cv2.boundingRect(contour)
            contour_center_y = y + h // 2
            
            # Check if this contour is in the head region (above face)
            is_in_head_region = (y < head_bottom and contour_center_y < face_y + face_h * 0.4)
            
            # Additional check: contour should be above the face center (exclude beard)
            is_above_face_center = y < (face_y + face_h // 2)
            
            # Check if contour overlaps significantly with head region
            contour_mask = np.zeros_like(all_masks)
            cv2.fillPoly(contour_mask, [contour], 255)
            overlap_with_head = np.sum((contour_mask > 0) & (head_mask > 0))
            total_contour_area = np.sum(contour_mask > 0)
            
            significant_overlap = (overlap_with_head > total_contour_area * 0.3) if total_contour_area > 0 else False
            
            # Only include if it's clearly head hair (not facial hair)
            if (is_in_head_region or is_above_face_center or significant_overlap):
                cv2.fillPoly(head_hair_mask, [contour], 255)
        
        # If no head hair found, fall back to original mask but remove very low contours (beards)
        if np.sum(head_hair_mask > 0) < 1000:
            print("   ⚠ Using fallback head hair detection")
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Only include contours that are in upper part of image (exclude beards)
                if y < image_np.shape[0] * 0.6:
                    cv2.fillPoly(head_hair_mask, [contour], 255)
        
        print(f"   🎯 Head hair pixels: {np.sum(head_hair_mask > 0)} (beards excluded)")
        return head_hair_mask

    def _choose_hair_class_from_logits(self, upsampled_logits, image_np):
        """Heuristic selection of hair class focusing on HEAD hair (not facial hair)"""
        try:
            cfg = getattr(self.model, "config", None)
            if cfg and getattr(cfg, "id2label", None):
                # Prefer hair-related classes, prioritize head hair
                hair_classes = []
                for idx, label in cfg.id2label.items():
                    if isinstance(label, str):
                        label_lower = label.lower()
                        if "hair" in label_lower and "face" not in label_lower:
                            hair_classes.append((idx, label, 3))
                        elif "hair" in label_lower:
                            hair_classes.append((idx, label, 2))
                        elif "head" in label_lower or "scalp" in label_lower:
                            hair_classes.append((idx, label, 2))
                        elif "hat" in label_lower or "cap" in label_lower:
                            hair_classes.append((idx, label, 1))
                
                if hair_classes:
                    hair_classes.sort(key=lambda x: x[2], reverse=True)
                    print(f"   🎯 Selected hair class: {hair_classes[0][1]} (priority {hair_classes[0][2]})")
                    return int(hair_classes[0][0])
        except Exception:
            pass

        # Fallback
        h, w = image_np.shape[:2]
        upper_region = np.zeros((h, w), dtype=np.uint8)
        upper_region[:h//2, :] = 1

        probs = torch.softmax(upsampled_logits[0], dim=0).cpu().numpy()
        best_idx = 0
        best_upper_overlap = -1
        
        for c in range(probs.shape[0]):
            mask_c = (probs[c] > 0.3).astype(np.uint8)
            upper_overlap = np.sum(mask_c & upper_region)
            if upper_overlap > best_upper_overlap:
                best_upper_overlap = upper_overlap
                best_idx = c
                
        print(f"   🎯 Fallback selected class {best_idx} with upper overlap: {best_upper_overlap}")
        return int(best_idx)

    def analyze_skin_tone(self, image, face_features):
        """Comprehensive skin tone analysis with ethnicity awareness"""
        try:
            img_array = np.array(image)
            
            if face_features is None:
                return {
                    'skin_tone': 'unknown', 'tone_category': 'unknown',
                    'dominant_color': [128, 128, 128], 'ethnicity_likely': 'unknown', 'warmth': 'neutral'
                }
            
            face_x, face_y, face_w, face_h = face_features['bounding_box']
            face_region = img_array[face_y:face_y+face_h, face_x:face_x+face_w]
            
            if face_region.size == 0:
                return {
                    'skin_tone': 'unknown', 'tone_category': 'unknown',
                    'dominant_color': [128, 128, 128], 'ethnicity_likely': 'unknown', 'warmth': 'neutral'
                }
            
            # Simple analysis for deployment
            avg_color = np.mean(face_region, axis=(0, 1))
            brightness = np.mean(avg_color)
            
            if brightness > 180:
                skin_tone, ethnicity_likely = "Fair", "East Asian/Caucasian"
            elif brightness > 140:
                skin_tone, ethnicity_likely = "Light", "East Asian/Caucasian"
            elif brightness > 100:
                skin_tone, ethnicity_likely = "Medium", "Latin American/Middle Eastern"
            else:
                skin_tone, ethnicity_likely = "Dark", "African"
            
            return {
                'skin_tone': skin_tone, 'tone_category': skin_tone,
                'dominant_color': avg_color.astype(int).tolist(),
                'ethnicity_likely': ethnicity_likely, 'warmth': 'neutral', 'confidence': 'Medium'
            }
            
        except Exception as e:
            print(f"   ❌ Skin tone analysis error: {e}")
            return {
                'skin_tone': 'unknown', 'tone_category': 'unknown',
                'dominant_color': [128, 128, 128], 'ethnicity_likely': 'unknown', 'warmth': 'neutral'
            }

    def enhanced_hair_segmentation(self, image_path):
        """Enhanced hair segmentation focusing on HEAD hair only"""
        try:
            if isinstance(image_path, str):
                if image_path.startswith('http'):
                    response = requests.get(image_path, timeout=10)
                    image = Image.open(BytesIO(response.content))
                else:
                    image = Image.open(image_path)
            else:
                image = image_path

            if image.mode != 'RGB':
                image = image.convert('RGB')

            original_size = image.size
            print(f"   Original image size: {original_size}")

            max_size = 1024
            if max(original_size) > max_size:
                scale_factor = max_size / max(original_size)
                new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
                image = image.resize(new_size, Image.LANCZOS)
                original_size = new_size

            if self.processor is None or self.model is None:
                return self.fallback_segmentation(image)

            print("   🔍 Detecting face for head hair extraction...")
            face_features, _, _ = self.detect_face_comprehensive(image)
            
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=image.size[::-1], mode="bilinear", align_corners=False
            )

            image_np = np.array(image)
            hair_class_idx = self._choose_hair_class_from_logits(upsampled_logits, image_np)

            probs = torch.softmax(upsampled_logits[0], dim=0).cpu().numpy()
            hair_prob = probs[hair_class_idx]
            all_hair_mask = (hair_prob >= 0.35).astype(np.uint8) * 255

            print("   🎯 Extracting head hair (excluding beards)...")
            head_hair_mask = self._get_head_hair_mask(image_np, face_features, all_hair_mask)

            # Clean up mask
            kernel_size = max(3, min(original_size) // 200)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            hair_mask_closed = cv2.morphologyEx(head_hair_mask, cv2.MORPH_CLOSE, kernel)
            hair_mask_clean = cv2.morphologyEx(hair_mask_closed, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(hair_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(hair_mask_clean)
            min_area = max(1000, original_size[0] * original_size[1] * 0.001)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    cv2.fillPoly(clean_mask, [contour], 255)

            clean_mask = cv2.GaussianBlur(clean_mask, (5, 5), 0)
            clean_mask = (clean_mask > 127).astype(np.uint8) * 255

            total_pixels = clean_mask.size
            hair_pixels = int((clean_mask > 0).sum())
            hair_coverage = (hair_pixels / total_pixels) * 100

            # Analyze hair type
            hair_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if hair_contours:
                largest_contour = max(hair_contours, key=cv2.contourArea)
                hair_x, hair_y, hair_w, hair_h = cv2.boundingRect(largest_contour)
                hair_bbox = (hair_x, hair_y, hair_w, hair_h)
                hair_area = cv2.contourArea(largest_contour)
                hair_perimeter = cv2.arcLength(largest_contour, True)
                complexity = (hair_perimeter ** 2) / (4 * np.pi * hair_area) if hair_perimeter > 0 else 1.0

                if complexity > 2.0:
                    hair_type = "very curly"
                elif complexity > 1.6:
                    hair_type = "curly"
                elif complexity > 1.3:
                    hair_type = "wavy"
                else:
                    hair_type = "straight"

                length_ratio = hair_h / max(hair_w, 1)
                if length_ratio > 2.5:
                    hair_length = "very long"
                elif length_ratio > 1.8:
                    hair_length = "long"
                elif length_ratio > 1.2:
                    hair_length = "medium"
                else:
                    hair_length = "short"
            else:
                hair_bbox = (0, 0, 0, 0)
                hair_type = "unknown"
                hair_length = "unknown"

            # Create visualization
            vis_image = np.array(image.copy())
            if hair_contours:
                cv2.drawContours(vis_image, [largest_contour], -1, (0, 255, 0), 2)
                x, y, w, h = hair_bbox
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(vis_image, f"HEAD HAIR: {hair_coverage:.1f}%", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_image, f"Type: {hair_type}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis_image, f"Length: {hair_length}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            segmentation_stats = {
                'hair_pixels': hair_pixels, 'total_pixels': total_pixels,
                'hair_coverage_percent': hair_coverage, 'hair_bbox': hair_bbox,
                'has_hair': hair_pixels > min_area, 'hair_type': hair_type,
                'hair_length': hair_length, 'complexity_score': complexity,
                'mask_quality': 'High' if hair_pixels > min_area else 'Low',
                'hair_class_idx': hair_class_idx
            }

            return {
                'hair_mask': Image.fromarray(clean_mask),
                'visualization': Image.fromarray(vis_image),
                'stats': segmentation_stats,
                'original_image': image
            }

        except Exception as e:
            print(f"   ❌ Enhanced segmentation error: {e}")
            return self.fallback_segmentation(image_path if not isinstance(image_path, Image.Image) else image_path)

    def fallback_segmentation(self, image):
        """Fallback segmentation"""
        try:
            if isinstance(image, str):
                image = Image.open(image)
            
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            hair_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
            
            kernel = np.ones((5,5), np.uint8)
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
            
            stats = {
                'hair_pixels': np.sum(hair_mask > 0), 'total_pixels': hair_mask.size,
                'hair_coverage_percent': (np.sum(hair_mask > 0) / hair_mask.size) * 100,
                'has_hair': True, 'hair_type': 'unknown', 'hair_length': 'unknown'
            }
            
            return {
                'hair_mask': Image.fromarray(hair_mask),
                'visualization': image,
                'stats': stats,
                'original_image': image
            }
        except Exception as e:
            print(f"   ❌ Fallback segmentation failed: {e}")
            return None

    def detect_face_comprehensive(self, image):
        """Face detection"""
        try:
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            if self.face_cascade is None:
                return None, None, None
            
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None, None, None
            
            face_x, face_y, face_w, face_h = max(faces, key=lambda rect: rect[2] * rect[3])
            
            face_ratio = face_w / face_h
            if face_ratio > 1.15:
                face_shape = "Round"
            elif face_ratio < 0.8:
                face_shape = "Oval"
            else:
                face_shape = "Square"
            
            face_features = {
                'shape': face_shape, 'bounding_box': (face_x, face_y, face_w, face_h), 'ratio': face_ratio
            }
            
            vis_image = img_array.copy()
            cv2.rectangle(vis_image, (face_x, face_y), (face_x+face_w, face_y+face_h), (0, 255, 0), 3)
            cv2.putText(vis_image, f"Face: {face_shape}", (face_x, face_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return face_features, gray[face_y:face_y+face_h, face_x:face_x+face_w], Image.fromarray(vis_image)
            
        except Exception as e:
            print(f"   ❌ Face detection error: {e}")
            return None, None, None

    def get_balanced_diverse_styles(self, skin_analysis, face_shape, current_hair_analysis):
        """Get style recommendations"""
        ethnicity = skin_analysis['ethnicity_likely']
        skin_tone = skin_analysis['skin_tone']
        
        # Basic style recommendations based on analysis
        styles = [
            "Classic layered cut with face-framing layers",
            "Modern textured crop with volume at crown",
            "Soft waves with curtain bangs for face framing",
            "Sleek straight bob with blunt ends",
            "Beachy waves with textured ends for movement",
            "Professional blowout with voluminous roots",
            "Elegant updo with face-framing tendrils",
            "Trendy shag cut with heavy textured bangs"
        ]
        
        return styles[:4]

    def get_hair_color_recommendations(self, skin_analysis):
        """Get hair color recommendations"""
        skin_tone = skin_analysis['skin_tone']
        
        if skin_tone in ["Fair", "Light"]:
            return ["Honey Blonde", "Golden Brown", "Caramel Highlights", "Ash Blonde", "Natural Brown"]
        elif skin_tone == "Medium":
            return ["Chocolate Brown", "Auburn", "Chestnut", "Mahogany", "Caramel Balayage"]
        else:
            return ["Rich Brown", "Espresso", "Blue Black", "Burgundy", "Dark Chocolate"]

    def basic_ethnicity_aware_transformation(self, original_image, hair_mask, hairstyle, skin_analysis):
        """Basic transformation without AI"""
        try:
            original_array = np.array(original_image)
            hair_array = np.array(hair_mask)
            
            result = original_array.copy()
            hair_region = hair_array > 128
            
            if np.sum(hair_region) < 500:
                return original_image
            
            # Simple color transformation based on skin tone
            skin_tone = skin_analysis['skin_tone']
            
            if skin_tone in ["Fair", "Light"]:
                base_color = [80, 50, 30]  # Light brown
            elif skin_tone == "Medium":
                base_color = [60, 35, 20]  # Medium brown
            else:
                base_color = [40, 25, 15]  # Dark brown
            
            # Apply color transformation
            for channel in range(3):
                result[hair_region, channel] = np.clip(
                    0.3 * result[hair_region, channel] + 0.7 * base_color[channel],
                    0, 255
                ).astype(np.uint8)
            
            # Add annotation
            annotated = result.copy()
            cv2.putText(annotated, f"Style: {hairstyle}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Skin: {skin_tone}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return Image.fromarray(annotated)
            
        except Exception as e:
            print(f"   ❌ Basic transformation failed: {e}")
            return original_image

    def full_balanced_transformation_pipeline(self, image_path, use_ai=False):  # Disable AI for deployment
        """Complete pipeline"""
        try:
            print("🔍 Step 1: Enhanced Hair Analysis...")
            seg_results = self.enhanced_hair_segmentation(image_path)
            
            if seg_results is None:
                return None
            
            original_image = seg_results['original_image']
            hair_mask = seg_results['hair_mask']
            hair_stats = seg_results['stats']
            
            print("🔍 Step 2: Face Analysis...")
            face_features, face_region, face_vis = self.detect_face_comprehensive(original_image)
            
            if face_features is None:
                img_array = np.array(original_image)
                h, w = img_array.shape[:2]
                face_features = {
                    'shape': 'Oval', 'bounding_box': (w//4, h//4, w//2, h//2)
                }
                face_vis_array = img_array.copy()
                cv2.rectangle(face_vis_array, (w//4, h//4), (w//4 + w//2, h//4 + h//2), (0, 255, 0), 2)
                face_vis = Image.fromarray(face_vis_array)
            
            print("🔍 Step 3: Skin Tone Analysis...")
            skin_analysis = self.analyze_skin_tone(original_image, face_features)
            
            print("🔍 Step 4: Style Recommendations...")
            current_hair_analysis = {
                'length': hair_stats['hair_length'], 'texture': hair_stats['hair_type']
            }
            
            style_recommendations = self.get_balanced_diverse_styles(
                skin_analysis, face_features['shape'], current_hair_analysis)
            
            print("🔍 Step 5: Color Recommendations...")
            color_recommendations = self.get_hair_color_recommendations(skin_analysis)
            
            print("🔍 Step 6: Generate Transformations...")
            results = []
            
            # Add analysis visualizations
            results.append(("1. Original Image", original_image))
            results.append(("2. Hair Analysis", seg_results['visualization']))
            results.append(("3. Face Analysis", face_vis))
            
            # Generate transformations
            for i, hairstyle in enumerate(style_recommendations[:3]):
                print(f"   🎯 Generating style {i+1}: {hairstyle}")
                
                transformed = self.basic_ethnicity_aware_transformation(
                    original_image, hair_mask, hairstyle, skin_analysis)
                
                if transformed is not None:
                    results.append((f"{i+4}. {hairstyle}", transformed))
            
            return {
                'original_image': original_image,
                'segmentation_results': seg_results,
                'face_features': face_features,
                'skin_analysis': skin_analysis,
                'style_recommendations': style_recommendations,
                'color_recommendations': color_recommendations,
                'results': results,
                'hair_stats': hair_stats
            }
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return None


class StreamlitHairTransformation:
    def __init__(self):
        self.transformer = SkinToneAwareHairTransformation(use_hairstyle_ai=False)  # Disable AI
    
    def process_image(self, image_path, session_id):
        """Process image for Streamlit"""
        try:
            results = self.transformer.full_balanced_transformation_pipeline(image_path, use_ai=False)
            
            if results is None:
                return None
            
            return {
                'original_image': results['original_image'],
                'analysis_data': {
                    'skin_tone': results['skin_analysis']['skin_tone'],
                    'ethnicity': results['skin_analysis']['ethnicity_likely'],
                    'face_shape': results['face_features']['shape'],
                    'hair_length': results['hair_stats']['hair_length'],
                    'hair_texture': results['hair_stats']['hair_type'],
                    'hair_coverage': results['hair_stats']['hair_coverage_percent'],
                },
                'recommendations': {
                    'styles': results['style_recommendations'][:4],
                    'colors': results['color_recommendations'][:5]
                },
                'images': results['results']
            }
            
        except Exception as e:
            print(f"Error in Streamlit processing: {e}")
            return None