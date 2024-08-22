from django.db import models

# Create your models here.
class tbl_register(models.Model):
    user_name=models.CharField(max_length=50,null=True)
    email=models.EmailField(null=True)
    password=models.CharField(max_length=50,null=True)
    phone_number=models.IntegerField(null=True)


class tbl_contact(models.Model):
    name=models.CharField(max_length=50,null=True)
    email=models.EmailField(null=True)
    subject=models.CharField(max_length=50,null=True)
    message=models.CharField(max_length=100,null=True)
