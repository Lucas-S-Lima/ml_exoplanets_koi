from django.urls import path
from core import views

urlpatterns = [
    path("", views.home, name="home"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("analyses-form/", views.analyses_form, name="analyses-form"),
    path("new-analyses/", views.analyses_form, name="new-analyses"),
    path("new-analysis/", views.analyses_form, name="new-analysis"),
    path("my-analyses/", views.my_analyses, name="my-analyses"),
    path("analyses-result/", views.analyses_result, name="analyses-result"),
    path("analyses-result/<int:analysis_id>/", views.analyses_result, name="analyses-result-detail"),
    path("analyses-result/<int:analysis_id>/download/", views.download_analysis_result, name="download-analysis-result"),
    path("analyses-result/<int:analysis_id>/delete/", views.delete_analysis_result, name="delete-analysis-result"),
]