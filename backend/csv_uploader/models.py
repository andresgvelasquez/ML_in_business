from django.db import models

# Create your models here.

class Dataset(models.Model):
    id = models.CharField(max_length=10, primary_key=True)
    f0 = models.FloatField()
    f1 = models.FloatField()
    f2 = models.FloatField()
    product = models.FloatField(default=99999)