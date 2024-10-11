from django.shortcuts import render, get_object_or_404, redirect
from .spam_detector.aiModel import classify_email


# Create your views here.


def home(request):
    return render(request, 'home.html')


def check(request):
    if request.method == 'POST':
        email_content = request.POST.get('email-content')
        result = classify_email(email_content)
        return render(request, 'home.html', {"result":result})
    return redirect('home')
