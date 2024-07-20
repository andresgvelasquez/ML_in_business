# Dockerfile para Django
FROM python:3.9

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requisitos y el resto del código
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/

# Exponer el puerto de Django
EXPOSE 8000

# Comando para ejecutar el servidor
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "myproject.wsgi:application"]