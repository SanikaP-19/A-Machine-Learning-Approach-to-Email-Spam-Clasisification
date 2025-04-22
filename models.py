from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Dataset(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    name = models.CharField(max_length=255)#Dataset name
    file = models.FileField(upload_to='datasets/') #FIle upload field
    uploaded_at = models.DateTimeField(auto_now_add = True)
    analysis_history = models.JSONField(default=list,blank=True)
    def __str__(self):
        return self.name
    
    