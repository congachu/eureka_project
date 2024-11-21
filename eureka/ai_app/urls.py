from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name='home'), # 127.0.0.1:8000
    path("check", views.check, name='check'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout, name='logout'),
    path('profile/', views.profile, name='profile'),
]