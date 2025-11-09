from django import forms
from .models import HairTransformation

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = HairTransformation
        fields = ['original_image']
        widgets = {
            'original_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            })
        }
    
    def clean_original_image(self):
        image = self.cleaned_data.get('original_image')
        if image:
            # Validate file size (5MB max)
            if image.size > 5 * 1024 * 1024:
                raise forms.ValidationError("Image file too large ( > 5MB )")
            
            # Validate file extension
            ext = image.name.split('.')[-1].lower()
            if ext not in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                raise forms.ValidationError("Unsupported file extension")
                
        return image