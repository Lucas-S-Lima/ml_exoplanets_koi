from django.contrib import admin
from .models import ModelRun, PredictionResult, ExoplanetCandidate, User


@admin.register(ModelRun)
class ModelRunAdmin(admin.ModelAdmin):
    list_display = ["algorithm", "trained_at", "accuracy", "precision", "recall", "f1_score"]


@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ["label", "candidate", "probability", "created_at"]
    list_filter = ["label", "created_at"]
    search_fields = ["candidate__name"]


@admin.register(ExoplanetCandidate)
class ExoplanetCandidateAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "user",
        "orbital_period_days",
        "transit_duration_hours",
        "transit_depth_ppm",
        "planet_radius_earth",
        "insolation_flux_earth",
        "equilibrium_temperature_k",
        "impact_parameter",
        "transit_signal_to_noise",
        "stellar_effective_temperature_k",
        "stellar_surface_gravity",
        "stellar_radius_solar",
        "kepler_magnitude",
        "created_at",
    ]

    list_filter = ["created_at"]
    search_fields = ["name"]


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ["username", "email", "first_name", "last_name", "password"]
    search_fields = ["username", "email"]
    readonly_fields = ["date_joined", "is_staff", "is_active", "password"]
