from django.urls import path
from core import views

urlpatterns = [
    # Endpoints for exoplanet candidate registration and prediction
    #path("register-exoplanet/", views.register_exoplanet, name="register_exoplanet"),
    #path("register-csv-exoplanets/", views.register_csv_exoplanets, name="register_csv_exoplanets"),
    #path("predict-exoplanets/", views.predict_exoplanet, name="predict_exoplanet"),
    #path("predict-exoplanets/<int:candidate_id>/", views.predict_exoplanet, name="predict_exoplanet"),
    #path("get-prediction-results/<int:candidate_id>/", views.get_predictions, name="get_predictions"),
    #path("get-prediction-results/", views.get_user_exoplanets, name="get_user_exoplanets"),

    # Endpoints to get evaluated exoplanets information
    path("evaluated-exoplanets/<int:id>/", views.get_individual_evaluated_exoplanet, name="get_individual_evaluated_exoplanet"),
    path("evaluated-exoplanets/", views.get_evaluated_exoplanets, name="get_evaluated_exoplanets"),

    # Endpoints for authentication
    path("register-user/", views.register_user, name="register_user"),
    path("obtain-token/", views.obtain_token, name="obtain_token"),
    path("refresh-token/", views.refresh_token, name="refresh_token"),
]