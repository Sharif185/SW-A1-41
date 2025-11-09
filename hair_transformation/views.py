import os
import uuid
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views import View
from django.conf import settings
from django.core.files.base import ContentFile
from PIL import Image
import json

from .forms import ImageUploadForm
from .models import HairTransformation, TransformationResult
from .utils.hair_ai import DjangoHairTransformation

class HomeView(View):
    def get(self, request):
        form = ImageUploadForm()
        return render(request, 'hair_transformation/home.html', {'form': form})
    
    def post(self, request):
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            hair_transformation = form.save(commit=False)
            hair_transformation.session_id = str(uuid.uuid4())
            hair_transformation.save()
            
            # Process the image
            return redirect('processing_view', session_id=hair_transformation.session_id)
        
        return render(request, 'hair_transformation/home.html', {'form': form})

class ProcessingView(View):
    def get(self, request, session_id):
        try:
            hair_transformation = HairTransformation.objects.get(session_id=session_id)
            
            # Initialize AI processor
            processor = DjangoHairTransformation()
            
            # Process the image
            image_path = hair_transformation.original_image.path
            results = processor.process_image(image_path, session_id)
            
            if results:
                # Save analysis data
                hair_transformation.skin_tone = results['analysis_data']['skin_tone']
                hair_transformation.ethnicity = results['analysis_data']['ethnicity']
                hair_transformation.face_shape = results['analysis_data']['face_shape']
                hair_transformation.hair_length = results['analysis_data']['hair_length']
                hair_transformation.hair_texture = results['analysis_data']['hair_texture']
                hair_transformation.style_recommendations = results['recommendations']['styles']
                hair_transformation.color_recommendations = results['recommendations']['colors']
                
                # Save analysis images
                hair_analysis_file = processor.pil_to_django_file(
                    results['images']['hair_analysis'], 
                    f"{session_id}_hair_analysis.png"
                )
                hair_transformation.hair_analysis_image.save(
                    f"{session_id}_hair_analysis.png", 
                    hair_analysis_file
                )
                
                face_analysis_file = processor.pil_to_django_file(
                    results['images']['face_analysis'], 
                    f"{session_id}_face_analysis.png"
                )
                hair_transformation.face_analysis_image.save(
                    f"{session_id}_face_analysis.png", 
                    face_analysis_file
                )
                
                hair_transformation.save()
                
                # Save transformation results
                for transformation in results['images']['transformations']:
                    transformed_file = processor.pil_to_django_file(
                        transformation['image'],
                        f"{session_id}_{transformation['style_type']}_{uuid.uuid4().hex[:8]}.png"
                    )
                    
                    TransformationResult.objects.create(
                        hair_transformation=hair_transformation,
                        style_name=transformation['title'],
                        style_type=transformation['style_type'],
                        transformed_image=transformed_file
                    )
                
                return redirect('results_view', session_id=session_id)
            else:
                return render(request, 'hair_transformation/error.html', {
                    'error': 'Image processing failed. Please try again with a different image.'
                })
                
        except HairTransformation.DoesNotExist:
            return render(request, 'hair_transformation/error.html', {
                'error': 'Session not found.'
            })
        except Exception as e:
            return render(request, 'hair_transformation/error.html', {
                'error': f'Processing error: {str(e)}'
            })

class ResultsView(View):
    def get(self, request, session_id):
        try:
            hair_transformation = HairTransformation.objects.get(session_id=session_id)
            transformation_results = hair_transformation.results.all()
            
            # Separate long and short styles
            long_styles = transformation_results.filter(style_type='Long')
            short_styles = transformation_results.filter(style_type='Short')
            
            context = {
                'transformation': hair_transformation,
                'long_styles': long_styles,
                'short_styles': short_styles,
                'analysis_data': {
                    'skin_tone': hair_transformation.skin_tone,
                    'ethnicity': hair_transformation.ethnicity,
                    'face_shape': hair_transformation.face_shape,
                    'hair_length': hair_transformation.hair_length,
                    'hair_texture': hair_transformation.hair_texture,
                }
            }
            
            return render(request, 'hair_transformation/results.html', context)
            
        except HairTransformation.DoesNotExist:
            return render(request, 'hair_transformation/error.html', {
                'error': 'Results not found.'
            })

class AjaxProcessingView(View):
    def get(self, request, session_id):
        """AJAX endpoint to check processing status"""
        try:
            hair_transformation = HairTransformation.objects.get(session_id=session_id)
            results_exist = hair_transformation.results.exists()
            
            return JsonResponse({
                'processed': results_exist,
                'status': 'complete' if results_exist else 'processing'
            })
        except HairTransformation.DoesNotExist:
            return JsonResponse({'error': 'Session not found'}, status=404)