from core.models import ExoplanetCandidate, User
from core.serializers import ExoplanetSerializer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from django.shortcuts import get_object_or_404


@api_view(["GET"])
def get_evaluated_exoplanets(request):
    exoplanets = ExoplanetCandidate.objects.all()
    serializer = ExoplanetSerializer(exoplanets, many=True)

    data = serializer.data

    return Response(
        data=data, 
        status=status.HTTP_200_OK, 
    )


@api_view(["GET"])
def get_individual_evaluated_exoplanet(request, id):
    exoplanet = get_object_or_404(ExoplanetCandidate, id=id)
    serializer = ExoplanetSerializer(exoplanet)

    data = serializer.data

    return Response(
        data=data, 
        status=status.HTTP_200_OK, 
    )


@api_view(["POST"])
def register_user(request):
    username = request.data.get("name")
    email = request.data.get("email")
    password = request.data.get("password")

    existing_user, new_user = User.objects.get_or_create(username=username, email=email, password=password)

    if existing_user:
        return Response(
            {
                "message": "User already exists"
            }, 
            status=status.HTTP_400_BAD_REQUEST
        )

    return Response(
        {
            "message": "User registered successfully"
        }, 
        status=status.HTTP_200_OK)


@api_view(["GET"])
def obtain_token(request):
    email = request.data.get("email")
    password = request.data.get("password")

    user = User.objects.filter(email=email).first()

    if user is None or not user.check_password(password):
        return Response(
            {
                "message": "Invalid credentials"
            }, 
            status=status.HTTP_401_UNAUTHORIZED
        )

    token = user.generate_token()

    return Response({"token": token}, status=status.HTTP_200_OK)


@api_view(["POST"])
def refresh_token(request):
    token = request.data.get("token")

    user = User.objects.filter(token=token).first()

    if user is None:
        return Response({"message": "Invalid token"}, 
                        status=status.HTTP_401_UNAUTHORIZED
        )

    new_token = user.generate_token()

    return Response({"token": new_token}, status=status.HTTP_200_OK)


