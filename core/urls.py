from django.urls import path
from core import views

urlpatterns = [
    path("", views.home, name="home"),
    path("analyses-form/", views.analyses_form, name="analyses-form"),
]