from django.contrib.auth.models import AbstractUser
from django.db import models
from django.conf import settings

# Create your models here.

class User(AbstractUser):

    def __str__(self):
        return self.username


class EmailCheck(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)  # 유저 연결 (옵션)
    email_content = models.TextField()  # 이메일 내용
    result = models.CharField(max_length=50)  # 스팸 여부 (스팸 / 정상 메일)
    probability = models.FloatField()  # 확률 (0~100)
    created_at = models.DateTimeField(auto_now_add=True)  # 저장 시간

    def __str__(self):
        return f"{self.result} ({self.probability}%)"