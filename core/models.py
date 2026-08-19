from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    email = models.EmailField(unique=True)

    def __str__(self):
        return self.username
    

class ExoplanetCandidate(models.Model):
    name = models.CharField(max_length=80)
    user = models.ForeignKey('User', on_delete=models.CASCADE, related_name='exoplanet_candidates')
    orbital_period_days = models.FloatField()
    transit_duration_hours = models.FloatField()
    transit_depth_ppm = models.FloatField()
    planet_radius_earth = models.FloatField()
    insolation_flux_earth = models.FloatField()
    equilibrium_temperature_k = models.FloatField()
    impact_parameter = models.FloatField()
    transit_signal_to_noise = models.FloatField()
    stellar_effective_temperature_k = models.FloatField()
    stellar_surface_gravity = models.FloatField()
    stellar_radius_solar = models.FloatField()
    kepler_magnitude = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)


class PredictionResult(models.Model):
    LABEL_CHOICES = [
        ("exoplanet", "EXOPLANET"),
        ("false_positive", "FALSE POSITIVE"),
    ]

    label = models.CharField(
        max_length=20,
        choices=LABEL_CHOICES,
    )

    candidate = models.ForeignKey(
        ExoplanetCandidate,
        on_delete=models.CASCADE,
        related_name="predictions",
    )

    probability = models.DecimalField(
        max_digits=5,
        decimal_places=4,
    )   
    created_at = models.DateTimeField(auto_now_add=True)


class ModelRun(models.Model):
    ALGORITHM_CHOICES = [
        ('logistic_regression', 'Logistic Regression'),
        ('random_forest', 'Random Forest'),
        ('svm', 'SVM'),
        ('gradient_boosting', 'Gradient Boosting'),
    ]

    algorithm = models.CharField(max_length=50, choices=ALGORITHM_CHOICES)
    trained_at = models.DateTimeField(auto_now_add=True)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    true_positive = models.IntegerField()
    true_negative = models.IntegerField()
    false_positive = models.IntegerField()
    false_negative = models.IntegerField()
    hyperparameters = models.JSONField(default=dict, blank=True)
    model_artifact_path = models.CharField(max_length=255)

    class Meta:
        ordering = ['-trained_at']

    def __str__(self):
        return f"{self.algorithm} — {self.trained_at:%Y-%m-%d %H:%M}"