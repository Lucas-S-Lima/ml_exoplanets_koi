from core.models import ExoplanetCandidate
from core.serializers import ExoplanetSerializer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status


@api_view(["GET"])
def get_evaluated_exoplanets(request):
    exoplanets = ExoplanetCandidate.objects.all()
    serializer = ExoplanetSerializer(exoplanets, many=True)

    data = serializer.data

    return Response(
        data=data, 
        status=status.HTTP_200_OK, 
    )





