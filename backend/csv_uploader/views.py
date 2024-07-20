from django.shortcuts import render
from rest_framework import generics, viewsets
from .models import Dataset
from .serializers import DatasetSerializer
import pandas as pd
from django.http import JsonResponse

# Create your views here.

from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from rest_framework.decorators import api_view
from django.core.files.base import ContentFile

from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

@api_view(['POST'])
def upload_file(request):
    parser_classes = (MultiPartParser, FormParser)
    
    if 'file' not in request.FILES:
        return Response({'message': 'No file uploaded.'}, status=status.HTTP_400_BAD_REQUEST)
    
    file = request.FILES['file']
    
    # Aquí puedes procesar el archivo como desees
    # Ejemplo de procesamiento (solo como referencia):
    try:
        # Procesa el archivo CSV
        import pandas as pd
        data = pd.read_csv(file)
        
        data_head = data.head()
        # Aquí podrías hacer algo con los datos
        return Response({'message': 'File uploaded successfully.', 'data': data_head.to_dict()}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'message': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer

class DatasetList(generics.ListCreateAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer

class DatasetHeadView(generics.GenericAPIView):
    def get(self, request, *args, **kwargs):
        # Obtener todos los objetos Dataset
        datasets = Dataset.objects.all().values()
        
        # Crear un DataFrame de Pandas
        df = pd.DataFrame(list(datasets))
        
        # Obtener las primeras filas del DataFrame
        df_head = df.head()
        
        # Convertir el DataFrame a JSON
        df_head_json = df_head.to_json(orient='records')
        
        return JsonResponse(df_head_json, safe=False)