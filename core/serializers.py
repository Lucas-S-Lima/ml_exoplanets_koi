from rest_framework import serializers
from core.models import ExoplanetCandidate


class ExoplanetSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExoplanetCandidate
        fields = "__all__"

    # def validate_float_fields(self, value):
    #     if value < 0:
    #         raise serializers.ValidationError("This field must be a non-negative float.")
    #     return value

class ObtainedTokenSerializer(serializers.Serializer):
    token = serializers.CharField(read_only=True)

    class Meta:
        fields = ["token"]

class RefreshTokenSerializer(serializers.Serializer):
    refresh_token = serializers.CharField(write_only=True)

    class Meta:
        fields = ["refresh_token"]

class UserRegistrationSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=60)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        fields = ["name", "email", "password"]
