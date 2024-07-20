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

from .utils.functions import save_to_postgres, generate_summary

import logging

# Configura el logging
logging.basicConfig(level=logging.INFO)

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
        df_region_1 = pd.read_csv(file)
        
        # Extraer las primeras filas.
        head_region_1 = df_region_1.head()

        # Generar un summary
        general_summary_json, column_summary_json = generate_summary(df_region_1)

        # Enviar el dataframe a la base de datos Postgres
        logging.info('Enviando data a postgre....')
        save_to_postgres(df_region_1, 'region_1')
        # Aquí podrías hacer algo con los datos
        return Response({'message': 'File uploaded successfully.', 'data': head_region_1.to_dict(), 'general_summary': general_summary_json, 'column_summary': column_summary_json}, status=status.HTTP_200_OK)
        # return Response({
        #     'message': 'File uploaded successfully.',
        #     'data': head_region_1.to_dict(),
        #     'general_summary': general_summary_dict,
        #     'column_summary': column_summary_dict
        # }, status=status.HTTP_200_OK)

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