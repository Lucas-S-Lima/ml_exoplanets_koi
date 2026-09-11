from rest_framework import serializers
from core.models import ExoplanetCandidate


class ExoplanetSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExoplanetCandidate
        fields = "__all__"

    def validate_float_fields(self, value):
        if value < 0:
            raise serializers.ValidationError("This field must be a non-negative float.")
        return value
