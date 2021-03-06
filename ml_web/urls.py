from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('empty', views.empty, name="empty"),
    path('', views.home, name="home"),
    path('model', views.model, name="model"),
    path('evaluate', views.evaluate, name="evaluate"),
]
