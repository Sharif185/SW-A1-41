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

# Load a hair segmentation model
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

class SkinToneAwareHairTransformation:
    def _init_(self, use_hairstyle_ai=True):
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
            self.processor = AutoImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
            self.models_used.append("mattmdjaga/segformer_b2_clothes (Hair Segmentation)")
            print("✅ SegFormer model loaded successfully")
        except Exception as e:
            print(f"   ⚠ Could not load SegFormer: {e}")
            self.processor = None
            self.model = None
        
        # Hairstyle transformation models
        self.use_hairstyle_ai = use_hairstyle_ai
        self.hairstyle_pipe = None
        
        if use_hairstyle_ai:
            self._initialize_hairstyle_models()

    def _initialize_hairstyle_models(self):
        """Initialize hairstyle transformation models (use inpainting pipeline)"""
        try:
            print("🔄 Initializing hairstyle transformation models...")
            from diffusers import StableDiffusionInpaintPipeline

            model_name = "runwayml/stable-diffusion-inpainting"
            self.hairstyle_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            # move to device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.hairstyle_pipe = self.hairstyle_pipe.to(self.device)

            # try to enable memory optimizations if available
            try:
                self.hairstyle_pipe.enable_model_cpu_offload()
                self.hairstyle_pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

            self.models_used.append(f"{model_name} (Hairstyle Transformation)")
            print(f"✅ Loaded: {model_name}")

        except Exception as e:
            print(f"🚫 Model initialization failed: {e}")
            self.use_hairstyle_ai = False

    def _get_head_hair_mask(self, image_np, face_features, all_masks):
        """Extract head hair mask while excluding facial hair (beards)"""
        if face_features is None:
            print("   ⚠ No face features for head hair extraction")
            return all_masks
            
        face_x, face_y, face_w, face_h = face_features['bounding_box']
        
        # Create head region mask (above face) - focus on scalp hair only
        head_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        
        # Define head region: above face, wider than face
        head_top = max(0, face_y - int(face_h * 1.5))  # Extended above face
        head_bottom = face_y + int(face_h * 0.3)  # Just below forehead (exclude beard area)
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
                if y < image_np.shape[0] * 0.6:  # Upper 60% of image
                    cv2.fillPoly(head_hair_mask, [contour], 255)
        
        print(f"   🎯 Head hair pixels: {np.sum(head_hair_mask > 0)} (beards excluded)")
        return head_hair_mask

    def _choose_hair_class_from_logits(self, upsampled_logits, image_np):
        """
        Heuristic selection of hair class focusing on HEAD hair (not facial hair)
        """
        try:
            cfg = getattr(self.model, "config", None)
            if cfg and getattr(cfg, "id2label", None):
                # Prefer hair-related classes, prioritize head hair
                hair_classes = []
                for idx, label in cfg.id2label.items():
                    if isinstance(label, str):
                        label_lower = label.lower()
                        if "hair" in label_lower and "face" not in label_lower:
                            hair_classes.append((idx, label, 3))  # Highest priority
                        elif "hair" in label_lower:
                            hair_classes.append((idx, label, 2))  # Medium priority
                        elif "head" in label_lower or "scalp" in label_lower:
                            hair_classes.append((idx, label, 2))
                        elif "hat" in label_lower or "cap" in label_lower:
                            hair_classes.append((idx, label, 1))
                
                if hair_classes:
                    # Sort by priority and return highest priority class
                    hair_classes.sort(key=lambda x: x[2], reverse=True)
                    print(f"   🎯 Selected hair class: {hair_classes[0][1]} (priority {hair_classes[0][2]})")
                    return int(hair_classes[0][0])
        except Exception:
            pass

        # Fallback: use class with most overlap with upper image region (head area)
        h, w = image_np.shape[:2]
        upper_region = np.zeros((h, w), dtype=np.uint8)
        upper_region[:h//2, :] = 1  # Upper half of image (head area)

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
                print("   ⚠ No face features for skin tone analysis")
                return {
                    'skin_tone': 'unknown',
                    'tone_category': 'unknown',
                    'dominant_color': [128, 128, 128],
                    'ethnicity_likely': 'unknown',
                    'warmth': 'neutral'
                }
            
            face_x, face_y, face_w, face_h = face_features['bounding_box']
            
            # Extract face region for skin tone analysis
            face_region = img_array[face_y:face_y+face_h, face_x:face_x+face_w]
            
            if face_region.size == 0:
                return {
                    'skin_tone': 'unknown',
                    'tone_category': 'unknown',
                    'dominant_color': [128, 128, 128],
                    'ethnicity_likely': 'unknown',
                    'warmth': 'neutral'
                }
            
            # Convert to LAB color space for better skin tone analysis
            lab_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2LAB)
            
            # Sample multiple points for robust analysis
            height, width = face_region.shape[:2]
            
            # Sample from cheek areas (avoiding eyes, nose, mouth)
            cheek_samples = []
            
            # Left cheek area
            l_cheek_x1 = width // 4
            l_cheek_x2 = width // 2
            l_cheek_y1 = height // 3
            l_cheek_y2 = 2 * height // 3
            
            # Right cheek area
            r_cheek_x1 = width // 2
            r_cheek_x2 = 3 * width // 4
            r_cheek_y1 = height // 3
            r_cheek_y2 = 2 * height // 3
            
            # Sample from cheek regions
            for x in range(l_cheek_x1, l_cheek_x2, 5):
                for y in range(l_cheek_y1, l_cheek_y2, 5):
                    if y < height and x < width:
                        cheek_samples.append(face_region[y, x])
            
            for x in range(r_cheek_x1, r_cheek_x2, 5):
                for y in range(r_cheek_y1, r_cheek_y2, 5):
                    if y < height and x < width:
                        cheek_samples.append(face_region[y, x])
            
            if not cheek_samples:
                # Fallback: sample entire face
                cheek_samples = face_region.reshape(-1, 3)[::10]
            
            cheek_samples = np.array(cheek_samples)
            
            # Use K-means to find dominant skin tone
            if len(cheek_samples) > 10:
                kmeans = KMeans(n_clusters=3, random_state=42)
                labels = kmeans.fit_predict(cheek_samples)
                dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(labels))]
            else:
                dominant_color = np.mean(cheek_samples, axis=0)
            
            # Convert to LAB for better tone analysis
            dominant_lab = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2LAB)[0][0]
            
            # Analyze skin tone characteristics
            L, A, B = dominant_lab
            
            # Categorize skin tone
            if L > 180:  # Very light
                tone_category = "Very Light"
                ethnicity_likely = "East Asian/Caucasian"
            elif L > 160:
                tone_category = "Light"
                ethnicity_likely = "East Asian/Caucasian"
            elif L > 140:
                tone_category = "Medium Light"
                ethnicity_likely = "Mediterranean/Latin American"
            elif L > 120:
                tone_category = "Medium"
                ethnicity_likely = "Latin American/Middle Eastern"
            elif L > 100:
                tone_category = "Medium Dark"
                ethnicity_likely = "South Asian/Middle Eastern"
            elif L > 80:
                tone_category = "Dark"
                ethnicity_likely = "African/Deep South Asian"
            else:
                tone_category = "Very Dark"
                ethnicity_likely = "African"
            
            # Determine warmth (A channel in LAB indicates green-red)
            if A > 130:
                warmth = "Warm"
            elif A > 115:
                warmth = "Neutral Warm"
            elif A > 100:
                warmth = "Neutral"
            else:
                warmth = "Cool"
            
            # Simple tone description
            if L > 160:
                skin_tone = "Fair"
            elif L > 140:
                skin_tone = "Light"
            elif L > 120:
                skin_tone = "Medium"
            elif L > 100:
                skin_tone = "Olive"
            elif L > 80:
                skin_tone = "Brown"
            else:
                skin_tone = "Dark"
            
            skin_analysis = {
                'skin_tone': skin_tone,
                'tone_category': tone_category,
                'dominant_color': dominant_color.astype(int).tolist(),
                'ethnicity_likely': ethnicity_likely,
                'warmth': warmth,
                'lab_values': [L, A, B],
                'confidence': 'High' if len(cheek_samples) > 50 else 'Medium'
            }
            
            print(f"   ✅ Skin tone: {skin_tone} ({tone_category}), {warmth} undertones")
            print(f"   📊 Likely ethnicity: {ethnicity_likely}")
            
            return skin_analysis
            
        except Exception as e:
            print(f"   ❌ Skin tone analysis error: {e}")
            return {
                'skin_tone': 'unknown',
                'tone_category': 'unknown',
                'dominant_color': [128, 128, 128],
                'ethnicity_likely': 'unknown',
                'warmth': 'neutral',
                'confidence': 'Low'
            }

    def get_balanced_diverse_styles(self, skin_analysis, face_shape, current_hair_analysis):
        """Get balanced recommendations: 2 long styles and 2 short styles, all completely different"""
        ethnicity = skin_analysis['ethnicity_likely']
        current_length = current_hair_analysis['length']
        current_texture = current_hair_analysis['texture']
        
        # Expanded hairstyle database with more options
        expanded_styles = {
            "African": {
                "Short": [
                    "Tapered fade with geometric temple designs",
                    "Buzz cut with sharp shape-up and line-up",
                    "Short textured afro with defined edges",
                    "Bald fade with intricate hairline patterns"
                ],
                "Medium": [
                    "Twist out with uniform curl definition",
                    "Finger coils with geometric parting",
                    "Bantu knots with creative patterns",
                    "Cornrows with intricate zigzag designs"
                ],
                "Long": [
                    "Box braids with middle part and golden cuffs",
                    "Senegalese twists with decorative beads",
                    "Goddess locs with golden hoop accessories",
                    "Fulani braids with cowrie shell decorations"
                ],
                "Very Long": [
                    "Waist-length box braids with ombre effect",
                    "Butt-length senegalese twists with beads",
                    "Long goddess locs with multiple accessories",
                    "Micro braids with intricate scalp designs"
                ]
            },
            "East Asian/Caucasian": {
                "Short": [
                    "Textured pixie with choppy layered ends",
                    "Asymmetrical bob with undercut design",
                    "French crop with disconnected textured top",
                    "Modern mullet with shaved temple design"
                ],
                "Medium": [
                    "Blunt bob with face-framing curtain layers",
                    "Shag cut with wispy bangs and texture",
                    "Layered lob with choppy ends and movement",
                    "Modern wolf cut with heavy layers and bangs"
                ],
                "Long": [
                    "Long layers with face-framing curtain bangs",
                    "Beachy waves with textured ends and volume",
                    "Blunt cut with minimal face-framing layers",
                    "Soft romantic curls with layered ends"
                ],
                "Very Long": [
                    "Mermaid waves with long layered ends",
                    "Long curtain bangs with face-framing layers",
                    "Waterfall layers throughout the length",
                    "Long balayage with soft face-framing"
                ]
            },
            "Latin American/Middle Eastern": {
                "Short": [
                    "Textured crop with voluminous crown",
                    "Short curly pixie with side-swept bangs",
                    "Modern mullet with textured curly top",
                    "Asymmetrical bob with curly ends"
                ],
                "Medium": [
                    "Curly lob with face-framing layers",
                    "Layered bob with curly texture and volume",
                    "Shag cut with curly ends and bangs",
                    "Modern wolf cut with curly layers"
                ],
                "Long": [
                    "Long bouncy curls with layered ends",
                    "Voluminous blowout with face-framing",
                    "Layered curls with curtain bangs",
                    "Beachy waves with curly volume"
                ],
                "Very Long": [
                    "Waterfall of long bouncy curls",
                    "Long mermaid waves with volume",
                    "Voluminous extra long curls",
                    "Long layers with curly movement"
                ]
            },
            "South Asian/Middle Eastern": {
                "Short": [
                    "Textured bob with side-swept bangs",
                    "Short shag with layered bangs",
                    "Modern pixie with textured crown",
                    "Asymmetrical crop with choppy ends"
                ],
                "Medium": [
                    "Layered bob with long curtain bangs",
                    "Shag cut with heavy face-framing",
                    "Textured lob with movement and layers",
                    "Modern wolf cut with textured ends"
                ],
                "Long": [
                    "Long sleek straight hair with blunt ends",
                    "Beachy waves with textured movement",
                    "Layered cut with face-framing volume",
                    "Soft romantic waves with layers"
                ],
                "Very Long": [
                    "Waterfall of long straight hair",
                    "Long mermaid waves with layers",
                    "Cascading layers with face-framing",
                    "Extra long curtain bangs"
                ]
            }
        }
        
        # Get base styles for ethnicity
        base_styles = expanded_styles.get(ethnicity, expanded_styles["East Asian/Caucasian"])
        
        # Select 2 long and 2 short styles (completely different from current)
        selected_styles = []
        
        # Get 2 long styles (from Long or Very Long categories)
        long_styles = []
        long_styles.extend(base_styles.get("Long", []))
        long_styles.extend(base_styles.get("Very Long", []))
        
        # Remove styles that might be similar to current
        filtered_long = [style for style in long_styles if self.is_style_different(style, current_length, current_texture)]
        
        # Select 2 diverse long styles
        if len(filtered_long) >= 2:
            selected_styles.extend(filtered_long[:2])
        else:
            selected_styles.extend(long_styles[:2])
        
        # Get 2 short styles (from Short category)
        short_styles = base_styles.get("Short", [])
        
        # Remove styles that might be similar to current
        filtered_short = [style for style in short_styles if self.is_style_different(style, current_length, current_texture)]
        
        # Select 2 diverse short styles
        if len(filtered_short) >= 2:
            selected_styles.extend(filtered_short[:2])
        else:
            selected_styles.extend(short_styles[:2])
        
        # If we need more styles, add from medium category
        if len(selected_styles) < 4:
            medium_styles = base_styles.get("Medium", [])
            filtered_medium = [style for style in medium_styles if self.is_style_different(style, current_length, current_texture)]
            needed = 4 - len(selected_styles)
            selected_styles.extend(filtered_medium[:needed])
        
        # Ensure we have exactly 4 styles
        return selected_styles[:4]

    def is_style_different(self, style, current_length, current_texture):
        """Check if a style is significantly different from current hair"""
        style_lower = style.lower()
        
        # If current is long, any short style is different
        if current_length in ["long", "very long"] and any(word in style_lower for word in ["short", "crop", "pixie", "buzz", "fade"]):
            return True
        
        # If current is short, any long style is different
        if current_length in ["short", "medium"] and any(word in style_lower for word in ["long", "waist", "butt", "rapunzel", "extra long"]):
            return True
        
        # Different texture patterns
        current_texture_lower = current_texture.lower()
        if current_texture_lower in ["straight", "wavy"] and any(word in style_lower for word in ["curly", "afro", "braid", "twist", "dread"]):
            return True
        
        if current_texture_lower in ["curly", "very curly"] and any(word in style_lower for word in ["straight", "sleek", "blunt", "smooth"]):
            return True
        
        return False

    def get_hair_color_recommendations(self, skin_analysis):
        """Get hair color recommendations based on skin tone and undertones"""
        skin_tone = skin_analysis['skin_tone']
        warmth = skin_analysis['warmth']
        ethnicity = skin_analysis['ethnicity_likely']
        
        color_recommendations = {
            "Fair": {
                "Warm": ["Honey Blonde", "Golden Brown", "Strawberry Blonde", "Copper", "Caramel Highlights"],
                "Cool": ["Ash Blonde", "Platinum", "Silver", "Ash Brown", "Champagne Blonde"],
                "Neutral": ["Beige Blonde", "Natural Brown", "Hazelnut", "Sandy Blonde", "Taupe"]
            },
            "Light": {
                "Warm": ["Golden Blonde", "Butterscotch", "Amber", "Cinnamon", "Warm Brown"],
                "Cool": ["Ash Brown", "Sandstone", "Pearl Blonde", "Mushroom Brown", "Icy Blonde"],
                "Neutral": ["Honey Brown", "Natural Blonde", "Chestnut", "Tawny", "Wheat"]
            },
            "Medium": {
                "Warm": ["Caramel", "Bronze", "Auburn", "Cognac", "Copper Brown"],
                "Cool": ["Chocolate Brown", "Mocha", "Espresso", "Cool Brown", "Black Brown"],
                "Neutral": ["Rich Brown", "Tobacco", "Cedar", "Mahogany", "Cappuccino"]
            },
            "Olive": {
                "Warm": ["Warm Brown", "Chestnut", "Auburn", "Bronze", "Golden Black"],
                "Cool": ["Cool Black", "Dark Brown", "Blue Black", "Ash Brown", "Graphite"],
                "Neutral": ["Dark Chocolate", "Natural Black", "Deep Brown", "Ebony", "Sable"]
            },
            "Brown": {
                "Warm": ["Rich Brown", "Mahogany", "Burgundy", "Copper", "Cinnamon"],
                "Cool": ["Blue Black", "Cool Brown", "Plum", "Eggplant", "Charcoal"],
                "Neutral": ["Natural Black", "Dark Brown", "Expresso", "Mocha", "Dark Chocolate"]
            },
            "Dark": {
                "Warm": ["Blue Black", "Red Black", "Burgundy", "Mahogany", "Purple Black"],
                "Cool": ["Jet Black", "Cool Black", "Blue Black", "Graphite", "Slate"],
                "Neutral": ["Natural Black", "Soft Black", "Dark Brown", "Ebony", "Raven"]
            }
        }
        
        # Get base color recommendations
        tone_recs = color_recommendations.get(skin_tone, color_recommendations["Medium"])
        colors = tone_recs.get(warmth, tone_recs["Neutral"])
        
        # Adjust for ethnicity preferences
        if ethnicity == "African" and skin_tone in ["Brown", "Dark"]:
            # Add culturally appropriate colors
            colors = ["Jet Black", "Blue Black", "Burgundy", "Purple", "Brown Black", "Natural Black"]
        elif ethnicity == "East Asian/Caucasian" and skin_tone in ["Fair", "Light"]:
            # Add popular Asian/Caucasian colors
            colors = ["Ash Brown", "Natural Black", "Dark Brown", "Chocolate", "Honey Blonde", "Caramel"]
        
        return colors[:6]  # Return top 6 colors

    def enhanced_hair_segmentation(self, image_path):
        """Enhanced hair segmentation focusing on HEAD hair only (excludes beards)"""
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

            # Resize very large images to improve processing speed and stability
            max_size = 1024
            if max(original_size) > max_size:
                scale_factor = max_size / max(original_size)
                new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
                image = image.resize(new_size, Image.LANCZOS)
                print(f"   Resized to: {new_size}")
                original_size = new_size

            if self.processor is None or self.model is None:
                # Fallback segmentation
                return self.fallback_segmentation(image)

            # First detect face for head hair extraction
            print("   🔍 Detecting face for head hair extraction...")
            face_features, _, _ = self.detect_face_comprehensive(image)
            
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # (1, C, h, w)

            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False
            )

            image_np = np.array(image)
            # choose class index for hair using improved method
            hair_class_idx = self._choose_hair_class_from_logits(upsampled_logits, image_np)

            # build prob map and mask
            probs = torch.softmax(upsampled_logits[0], dim=0).cpu().numpy()
            hair_prob = probs[hair_class_idx]
            all_hair_mask = (hair_prob >= 0.35).astype(np.uint8) * 255

            # Extract HEAD hair only (exclude facial hair/beards)
            print("   🎯 Extracting head hair (excluding beards)...")
            head_hair_mask = self._get_head_hair_mask(image_np, face_features, all_hair_mask)

            # morphological cleanup (adaptive)
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
                complexity = 1.0

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
                cv2.putText(vis_image, "BEARDS EXCLUDED", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            segmentation_stats = {
                'hair_pixels': hair_pixels,
                'total_pixels': total_pixels,
                'hair_coverage_percent': hair_coverage,
                'hair_bbox': hair_bbox,
                'has_hair': hair_pixels > min_area,
                'hair_type': hair_type,
                'hair_length': hair_length,
                'complexity_score': complexity,
                'mask_quality': 'High' if hair_pixels > min_area else 'Low',
                'hair_class_idx': hair_class_idx
            }

            segmentation_results = {
                'hair_mask': Image.fromarray(clean_mask),
                'visualization': Image.fromarray(vis_image),
                'stats': segmentation_stats,
                'original_image': image
            }

            print(f"   ✅ HEAD HAIR detection: {hair_pixels} pixels ({hair_coverage:.1f}%)")
            print(f"   📊 Hair analysis: {hair_length} {hair_type} hair (class {hair_class_idx})")
            print(f"   🚫 Beards/facial hair excluded from segmentation")

            return segmentation_results

        except Exception as e:
            print(f"   ❌ Enhanced segmentation error: {e}")
            return self.fallback_segmentation(image_path if not isinstance(image_path, Image.Image) else image_path)

    def fallback_segmentation(self, image):
        """Fallback segmentation when main model fails"""
        try:
            if isinstance(image, str):
                image = Image.open(image)
            
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Simple hair detection based on color and texture
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel = lab[:,:,0]
            
            # Adaptive thresholding
            hair_mask = cv2.adaptiveThreshold(l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
            
            # Clean up the mask
            kernel = np.ones((5,5), np.uint8)
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
            
            stats = {
                'hair_pixels': np.sum(hair_mask > 0),
                'total_pixels': hair_mask.size,
                'hair_coverage_percent': (np.sum(hair_mask > 0) / hair_mask.size) * 100,
                'hair_bbox': (0, 0, img_array.shape[1], img_array.shape[0] // 3),
                'has_hair': True,
                'hair_type': 'unknown',
                'hair_length': 'unknown',
                'complexity_score': 1.0,
                'mask_quality': 'Low'
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
        """Comprehensive face detection with detailed analysis"""
        try:
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            if self.face_cascade is None:
                print("   ⚠ Face cascade not available")
                return None, None, None
            
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                print("   ⚠ No faces detected")
                return None, None, None
            
            # Take the largest face
            face_x, face_y, face_w, face_h = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Comprehensive face shape detection
            face_ratio = face_w / face_h
            
            # Calculate facial proportions
            jaw_width = face_w
            forehead_width = int(face_w * 0.85)
            cheekbone_width = int(face_w * 0.95)
            
            if face_ratio > 1.15:
                face_shape = "Round"
                shape_confidence = "High"
            elif face_ratio < 0.8:
                face_shape = "Oval"
                shape_confidence = "High"
            elif abs(face_w - face_h) < 0.08 * max(face_w, face_h):
                face_shape = "Square"
                shape_confidence = "Medium"
            elif jaw_width > forehead_width * 1.15:
                face_shape = "Triangle"
                shape_confidence = "Medium"
            elif forehead_width > jaw_width * 1.1:
                face_shape = "Heart"
                shape_confidence = "Medium"
            else:
                face_shape = "Diamond"
                shape_confidence = "Medium"
            
            face_features = {
                'shape': face_shape,
                'confidence': shape_confidence,
                'bounding_box': (face_x, face_y, face_w, face_h),
                'ratio': face_ratio,
                'jaw_width': jaw_width,
                'forehead_width': forehead_width,
                'cheekbone_width': cheekbone_width,
                'face_length': face_h,
                'landmarks': {
                    'center': (face_x + face_w//2, face_y + face_h//2),
                    'forehead': (face_x + face_w//2, face_y + face_h//6),
                    'chin': (face_x + face_w//2, face_y + face_h)
                }
            }
            
            # Create detailed visualization
            vis_image = img_array.copy()
            
            # Draw bounding box
            cv2.rectangle(vis_image, (face_x, face_y), (face_x+face_w, face_y+face_h), (0, 255, 0), 3)
            
            # Draw facial landmarks
            landmarks = face_features['landmarks']
            cv2.circle(vis_image, landmarks['center'], 6, (255, 0, 0), -1)
            cv2.circle(vis_image, landmarks['forehead'], 5, (0, 255, 255), -1)
            cv2.circle(vis_image, landmarks['chin'], 5, (255, 255, 0), -1)
            
            # Add detailed annotations
            cv2.putText(vis_image, f"Face: {face_shape}", (face_x, face_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_image, f"Ratio: {face_ratio:.2f}", (face_x, face_y-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return face_features, gray[face_y:face_y+face_h, face_x:face_x+face_w], Image.fromarray(vis_image)
            
        except Exception as e:
            print(f"   ❌ Face detection error: {e}")
            return None, None, None

    def extract_hair_texture_features(self, original_image, hair_mask):
        """Extract the person's natural hair texture for realistic transformations"""
        try:
            img_array = np.array(original_image)
            mask_array = np.array(hair_mask)
            
            # Extract hair region
            hair_region = mask_array > 128
            hair_pixels = img_array[hair_region]
            
            if len(hair_pixels) == 0:
                return None
            
            # Analyze hair color characteristics
            avg_color = np.mean(hair_pixels, axis=0)
            std_color = np.std(hair_pixels, axis=0)
            
            # Convert to LAB for better texture analysis
            lab_hair = cv2.cvtColor(np.uint8([hair_pixels]), cv2.COLOR_RGB2LAB)[0]
            
            # Calculate texture complexity
            gray_hair = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            hair_gray = gray_hair[hair_region]
            
            texture_features = {
                'avg_color': avg_color.tolist(),
                'std_color': std_color.tolist(),
                'avg_lab': np.mean(lab_hair, axis=0).tolist(),
                'texture_complexity': np.std(hair_gray) if len(hair_gray) > 0 else 0,
                'hair_density': np.sum(hair_region) / hair_region.size
            }
            
            return texture_features
            
        except Exception as e:
            print(f"   ❌ Texture extraction error: {e}")
            return None

    def create_texture_preserving_prompt(self, hairstyle, skin_analysis, face_shape, texture_features):
        """Create prompt that preserves natural hair texture while changing style"""
        ethnicity = skin_analysis['ethnicity_likely']
        skin_tone = skin_analysis['skin_tone']
        
        # Base structure emphasizing texture preservation
        texture_keywords = ""
        if texture_features and texture_features['texture_complexity'] > 25:
            texture_keywords = "natural hair texture, realistic strands, detailed texture, "
        elif texture_features and texture_features['texture_complexity'] > 15:
            texture_keywords = "defined texture, natural movement, realistic hair, "
        
        base_prompt = f"professional photo, {hairstyle}, {texture_keywords}high quality, detailed, realistic hair transformation"
        
        # Ethnicity-specific enhancements
        if ethnicity == "African":
            prompt = f"{base_prompt}, maintaining natural curl pattern, defined texture, professional styling, realistic afro-textured hair"
        elif ethnicity == "East Asian/Caucasian":
            prompt = f"{base_prompt}, natural hair movement, realistic strands, professional salon quality, smooth texture"
        elif ethnicity == "Latin American/Middle Eastern":
            prompt = f"{base_prompt}, voluminous texture, natural waves, professional styling, defined curls"
        else:
            prompt = base_prompt
        
        # Face shape consideration
        prompt += f", flattering for {face_shape.lower()} face shape"
        
        # Skin tone context for better color matching
        if skin_tone in ["Dark", "Brown"]:
            prompt += f", on person with {skin_tone.lower()} skin tone, natural hair colors"
        elif skin_tone in ["Fair", "Light"]:
            prompt += f", on person with {skin_tone.lower()} skin tone, natural hair tones"
        
        return prompt

    def texture_preserving_transformation(self, original_image, hair_mask, face_features, hairstyle, skin_analysis, texture_features):
        """Transformation that preserves natural hair texture while changing style"""
        if not self.use_hairstyle_ai or self.hairstyle_pipe is None:
            return self.basic_ethnicity_aware_transformation(original_image, hair_mask, hairstyle, skin_analysis)

        try:
            print(f"   🎨 Texture-Preserving Transformation: {hairstyle}")

            # Store original dimensions
            original_size = original_image.size
            print(f"   Original image size: {original_size}")

            # create prompt
            prompt = self.create_texture_preserving_prompt(hairstyle, skin_analysis, face_features['shape'], texture_features)
            init_image = original_image.convert("RGB")
            mask_image = hair_mask.convert("L")

            # ensure sizes match
            if mask_image.size != init_image.size:
                mask_image = mask_image.resize(init_image.size, resample=Image.NEAREST)

            # mask convention: white (255) = inpaint region — ensure it's white where hair is
            nonzero = (np.array(mask_image) > 127).sum()
            total = mask_image.size[0] * mask_image.size[1]
            pct = 100.0 * nonzero / max(1, total)
            print(f"   Mask white pixels: {nonzero} / {total} ({pct:.2f}%)")

            if pct < 0.2:
                # if mask is tiny maybe threshold too high or inverted — invert if needed
                mask_image = ImageOps.invert(mask_image)
                nonzero = (np.array(mask_image) > 127).sum()
                pct = 100.0 * nonzero / max(1, total)
                print(f"   After invert: mask white pixels {nonzero} ({pct:.2f}%)")

            # Resize images to 512x512 for the model (Stable Diffusion's expected input size)
            target_size = (512, 512)
            init_image_resized = init_image.resize(target_size, Image.LANCZOS)
            mask_image_resized = mask_image.resize(target_size, Image.NEAREST)

            device = getattr(self, "device", "cpu")
            # run inpainting
            generator = torch.manual_seed(42)
            output = self.hairstyle_pipe(
                prompt=prompt,
                image=init_image_resized,
                mask_image=mask_image_resized,
                guidance_scale=7.5,
                num_inference_steps=20,  # Reduced for faster processing
                generator=generator,
                negative_prompt="blurry, low quality, artifacts, deformed, bad anatomy"
            )

            edited_resized = output.images[0]
            
            # Resize back to original dimensions
            edited = edited_resized.resize(original_size, Image.LANCZOS)
            
            # Convert to numpy arrays for blending
            edited_np = np.array(edited)
            orig_np = np.array(init_image)
            mask_np = (np.array(mask_image) > 127).astype(np.uint8)[:, :, None]

            # Ensure all arrays have the same dimensions
            if edited_np.shape != orig_np.shape:
                print(f"   ⚠ Shape mismatch - edited: {edited_np.shape}, original: {orig_np.shape}")
                # Resize edited to match original if needed
                edited_np = cv2.resize(edited_np, (orig_np.shape[1], orig_np.shape[0]))
            
            if mask_np.shape[:2] != orig_np.shape[:2]:
                print(f"   ⚠ Mask shape mismatch - mask: {mask_np.shape}, original: {orig_np.shape}")
                # Resize mask to match original if needed
                mask_np = cv2.resize(mask_np, (orig_np.shape[1], orig_np.shape[0]))
                mask_np = (mask_np > 0.5).astype(np.uint8)

            # Blend edited hair back onto original using mask to preserve face/background
            composed = (edited_np * mask_np + orig_np * (1 - mask_np)).astype(np.uint8)
            composed_pil = Image.fromarray(composed)
            
            print(f"   ✅ Transformation completed successfully")
            return composed_pil

        except Exception as e:
            print(f"   ❌ Texture-preserving transformation failed: {e}")
            import traceback
            traceback.print_exc()
            return self.basic_ethnicity_aware_transformation(original_image, hair_mask, hairstyle, skin_analysis)

    def basic_ethnicity_aware_transformation(self, original_image, hair_mask, hairstyle, skin_analysis):
        """Basic ethnicity-aware transformation without AI"""
        try:
            original_array = np.array(original_image)
            hair_array = np.array(hair_mask)
            
            result = original_array.copy()
            hair_region = hair_array > 128
            
            if np.sum(hair_region) < 500:
                return original_image
            
            # Choose culturally appropriate colors
            ethnicity = skin_analysis['ethnicity_likely']
            skin_tone = skin_analysis['skin_tone']
            
            if ethnicity == "African" and skin_tone in ["Brown", "Dark"]:
                # Natural black hair colors for African ethnicity
                base_color = [30, 20, 10]  # Rich black
                texture_strength = 40
            elif ethnicity == "East Asian/Caucasian":
                base_color = [60, 40, 20]  # Dark brown
                texture_strength = 25
            else:
                base_color = [50, 30, 15]  # Neutral dark
                texture_strength = 30
            
            # Apply color transformation
            for channel in range(3):
                result[hair_region, channel] = np.clip(
                    0.4 * result[hair_region, channel] + 0.6 * base_color[channel],
                    0, 255
                ).astype(np.uint8)
            
            # Add ethnicity-appropriate texture
            height, width = result.shape[:2]
            
            if ethnicity == "African":
                # Stronger texture for African hair
                texture1 = np.sin(np.indices((height, width))[1] / 15) * 30
                texture2 = np.random.rand(height, width) * 35
                texture = texture1 + texture2 - 32
            else:
                # Moderate texture for other ethnicities
                texture = np.random.rand(height, width) * 25 - 12
            
            for channel in range(3):
                result[hair_region, channel] = np.clip(
                    result[hair_region, channel] + texture[hair_region],
                    0, 255
                ).astype(np.uint8)
            
            # Add annotation
            annotated = result.copy()
            cv2.putText(annotated, f"{ethnicity} Style", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, hairstyle[:30], (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return Image.fromarray(annotated)
            
        except Exception as e:
            print(f"   ❌ Basic ethnicity-aware transformation failed: {e}")
            return original_image

    def full_balanced_transformation_pipeline(self, image_path, use_ai=True):
        """Complete pipeline with balanced long/short style recommendations"""
        try:
            print("🔍 Step 1: Enhanced Hair Analysis...")
            seg_results = self.enhanced_hair_segmentation(image_path)
            
            if seg_results is None:
                print("❌ Hair analysis failed")
                return None
            
            original_image = seg_results['original_image']
            hair_mask = seg_results['hair_mask']
            hair_stats = seg_results['stats']
            
            print("🔍 Step 2: Detailed Face Analysis...")
            face_features, face_region, face_vis = self.detect_face_comprehensive(original_image)
            
            if face_features is None:
                print("   ⚠ Using comprehensive default face features")
                img_array = np.array(original_image)
                h, w = img_array.shape[:2]
                face_features = {
                    'shape': 'Oval',
                    'confidence': 'Default',
                    'bounding_box': (w//4, h//4, w//2, h//2),
                    'ratio': 0.8
                }
                face_vis_array = img_array.copy()
                cv2.rectangle(face_vis_array, (w//4, h//4), (w//4 + w//2, h//4 + h//2), (0, 255, 0), 2)
                face_vis = Image.fromarray(face_vis_array)
            
            print("🔍 Step 3: Skin Tone & Ethnicity Analysis...")
            skin_analysis = self.analyze_skin_tone(original_image, face_features)
            
            print("🔍 Step 4: Natural Hair Texture Analysis...")
            texture_features = self.extract_hair_texture_features(original_image, hair_mask)
            
            print("🔍 Step 5: Balanced Style Recommendations (2 Long + 2 Short)...")
            current_hair_analysis = {
                'length': hair_stats['hair_length'],
                'texture': hair_stats['hair_type'],
                'coverage': hair_stats['hair_coverage_percent']
            }
            
            style_recommendations = self.get_balanced_diverse_styles(
                skin_analysis, face_features['shape'], current_hair_analysis)
            
            print("🔍 Step 6: Enhanced Color Recommendations...")
            color_recommendations = self.get_hair_color_recommendations(skin_analysis)
            
            print("🔍 Step 7: Generate Balanced Transformations...")
            results = []
            transformation_details = []
            
            # Add analysis visualizations
            results.append(("1. Original Image", original_image))
            results.append(("2. Hair Analysis", seg_results['visualization']))
            results.append(("3. Face & Skin Analysis", face_vis))
            
            # Generate balanced transformations (2 long, 2 short)
            for i, hairstyle in enumerate(style_recommendations[:4]):
                style_type = "Long" if i < 2 else "Short"
                print(f"   🎯 Generating {style_type} {skin_analysis['ethnicity_likely']} style {i+1}/4: {hairstyle}")
                
                transformed = self.texture_preserving_transformation(
                    original_image, hair_mask, face_features, hairstyle, skin_analysis, texture_features)
                
                if transformed is not None:
                    results.append((f"{i+4}. {style_type}: {hairstyle}", transformed))
                    
                    transformation_details.append({
                        'style': hairstyle,
                        'type': style_type,
                        'ethnicity': skin_analysis['ethnicity_likely'],
                        'skin_tone': skin_analysis['skin_tone'],
                        'method': 'Texture-Preserved AI' if use_ai and self.use_hairstyle_ai else 'Basic',
                        'texture_preserved': texture_features is not None
                    })
                else:
                    print(f"   ❌ Failed to generate {style_type} style {i+1}")
            
            return {
                'original_image': original_image,
                'segmentation_results': seg_results,
                'face_features': face_features,
                'skin_analysis': skin_analysis,
                'texture_features': texture_features,
                'style_recommendations': style_recommendations,
                'color_recommendations': color_recommendations,
                'results': results,
                'transformation_details': transformation_details,
                'hair_stats': hair_stats
            }
            
        except Exception as e:
            print(f"❌ Balanced transformation pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return None


class StreamlitHairTransformation:
    def _init_(self):
        self.transformer = SkinToneAwareHairTransformation(use_hairstyle_ai=True)
    
    def process_image(self, image_path, session_id):
        """
        Process image and return results in Streamlit-compatible format
        """
        try:
            # Run the transformation pipeline
            results = self.transformer.full_balanced_transformation_pipeline(image_path, use_ai=True)
            
            if results is None:
                return None
            
            # Convert results to Streamlit-compatible format
            streamlit_results = {
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
                    'colors': results['color_recommendations'][:6]
                },
                'images': {
                    'hair_analysis': results['segmentation_results']['visualization'],
                    'face_analysis': results.get('face_vis', results['segmentation_results']['visualization']),
                    'transformations': []
                }
            }
            
            # Extract transformation images
            for title, img in results['results']:
                if "Long:" in title or "Short:" in title:
                    streamlit_results['images']['transformations'].append({
                        'title': title,
                        'image': img,
                        'style_type': 'Long' if 'Long:' in title else 'Short'
                    })
            
            return streamlit_results
            
        except Exception as e:
            print(f"Error in Streamlit processing: {e}")
            return None