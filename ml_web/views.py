from django.shortcuts import render
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from asd_ml.helpers import *
from collections import namedtuple
from django.db import connection
from urllib.parse import parse_qs

now = timezone.now()
cursor = connection.cursor()

# Create your views here.
def empty(request):
    page = { "title": "empty", "sub": "" }
    return render(request, 'html/empty.html', { "currentTime": now, "page": page })

def home(request):
    page = { "title": "empty", "sub": "" }
    return render(request, 'html/empty.html', { "currentTime": now, "page": page })

def client_form(request):
    page = { "title": "empty", "sub": "" }
    return render(request, 'html/empty.html', { "currentTime": now, "page": page })

def list_report(request):
    page = { "title": "empty", "sub": "" }
    return render(request, 'html/empty.html', { "currentTime": now, "page": page })
