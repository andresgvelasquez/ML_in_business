# # Dockerfile para React
# FROM node:18

# # Configurar el directorio de trabajo
# WORKDIR /app

# # Copiar los archivos de dependencias
# COPY package*.json ./
# RUN npm install

# # Copiar el resto del código
# COPY . .

# # Construir el proyecto
# RUN npm run build

# # Servir la aplicación
# CMD ["npx", "serve", "-s", "build"]

# # Exponer el puerto
# EXPOSE 3000


# Usa una imagen base de Node.js
FROM node:18

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar los archivos de dependencias
COPY package*.json ./
RUN npm install

# Copiar el resto del código
COPY . .

# Establece la variable de entorno para desarrollo
ENV NODE_ENV=development

# Exponer el puerto
EXPOSE 3000

# Comando para iniciar el servidor de desarrollo
CMD ["npm", "start"]