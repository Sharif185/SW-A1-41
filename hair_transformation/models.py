from django.db import models
import os
import uuid

def user_upload_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('uploads', filename)

def result_upload_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('results', filename)

class HairTransformation(models.Model):
    original_image = models.ImageField(upload_to=user_upload_path)
    created_at = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(max_length=100, unique=True, default=uuid.uuid4)
    
    # Analysis results (store as JSON)
    skin_tone = models.CharField(max_length=50, blank=True)
    ethnicity = models.CharField(max_length=100, blank=True)
    face_shape = models.CharField(max_length=50, blank=True)
    hair_length = models.CharField(max_length=50, blank=True)
    hair_texture = models.CharField(max_length=50, blank=True)
    
    # Style recommendations
    style_recommendations = models.JSONField(default=list)
    color_recommendations = models.JSONField(default=list)
    
    # Processed images
    hair_analysis_image = models.ImageField(upload_to=result_upload_path, null=True, blank=True)
    face_analysis_image = models.ImageField(upload_to=result_upload_path, null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']

class TransformationResult(models.Model):
    hair_transformation = models.ForeignKey(HairTransformation, on_delete=models.CASCADE, related_name='results')
    style_name = models.CharField(max_length=200)
    style_type = models.CharField(max_length=50)  # Long or Short
    transformed_image = models.ImageField(upload_to=result_upload_path)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['style_type', 'style_name']