from django import forms
from .models import Dataset

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name','file']

class MessageClassificationForm(forms.Form):
    message = forms.CharField(widget = forms.Textarea,label="Enter message",required=True)        
        