from django.contrib import admin
from .models import HairTransformation, TransformationResult

class TransformationResultInline(admin.TabularInline):
    model = TransformationResult
    extra = 0
    readonly_fields = ['transformed_image']

@admin.register(HairTransformation)
class HairTransformationAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'skin_tone', 'ethnicity', 'face_shape', 'created_at']
    list_filter = ['skin_tone', 'ethnicity', 'face_shape', 'created_at']
    search_fields = ['session_id', 'ethnicity']
    readonly_fields = ['session_id', 'created_at']
    inlines = [TransformationResultInline]

@admin.register(TransformationResult)
class TransformationResultAdmin(admin.ModelAdmin):
    list_display = ['style_name', 'style_type', 'hair_transformation', 'created_at']
    list_filter = ['style_type', 'created_at']
    search_fields = ['style_name', 'hair_transformation__session_id']