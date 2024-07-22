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

from .utils.functions import save_to_postgres, generate_summary, data_viz_overview, train_valid_split_scaled, LinearReg_predict

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

        # Enviar el dataframe a la base de datos Postgres
        save_to_postgres(df_region_1, 'region_1')

        # Extraer las primeras filas.
        head_region_1 = df_region_1.head()

        # Generar tablas con resumen general y por columna
        general_summary_json, column_summary_json = generate_summary(df_region_1)

        # Generar información para el histograma y boxplot
        bin_counts_df_json, boxplot_data = data_viz_overview(df_region_1)

        # Generar la tabla de correlacion
        heatmap_data_json = df_region_1.corr().to_json()

        # Preprocesamiento de datos=========================================

        # Didivir el dataframe en train y valid (para las características y objetivo)
        # y escalar las características
        features_train_scaled_region_1, target_train_region_1, features_valid_scaled_region_1, target_valid_region_1 = train_valid_split_scaled(df_region_1,
                                                                                                                                                'product',
                                                                                                                                                0.25) 

        # Predicciones para la región 0
        predicts_region_1, model_region_1, volumen_predictions = LinearReg_predict(
            features_train_scaled_region_1, 
            target_train_region_1, 
            features_valid_scaled_region_1, 
            target_valid_region_1
        )
        
        # Carga de datos====================================================

        # Aquí podrías hacer algo con los datos
        return Response({'message': 'File uploaded successfully.', 
                        'data': head_region_1.to_dict(), 
                        'general_summary': general_summary_json, 
                        'histogram_data': bin_counts_df_json,
                        'boxplot_data': boxplot_data,
                        'heatmap_data': heatmap_data_json,
                        'column_summary': column_summary_json,
                        'volumen_predictions': volumen_predictions},
                        status=status.HTTP_200_OK
        )


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