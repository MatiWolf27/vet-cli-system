# 🏥 Sistema de Recomendaciones Veterinarias CLI

Sistema de machine learning para recomendaciones de servicios veterinarios basado en características de mascotas. Desarrollado como prototipo para demostrar la aplicación de ML en sistemas de recomendación para marketplaces.

## 🚀 Características

- **Predicción inteligente**: Recomienda servicios veterinarios usando múltiples modelos de ML
- **CLI interactivo**: Interfaz de línea de comandos con colores y menús intuitivos
- **Múltiples modelos**: Regresión Logística, Árboles de Decisión y Random Forest
- **Datos sintéticos realistas**: Generación automática de datos de entrenamiento
- **Visualización avanzada**: Gráficos de probabilidades y barras de confianza

## 📋 Requisitos

- Python 3.8+
- pip (gestor de paquetes de Python)

## 🛠️ Instalación

1. **Clona el repositorio:**
```bash
git clone https://github.com/tu-usuario/vet-cli-system.git
cd vet-cli-system
```

2. **Crea un entorno virtual:**
```bash
python -m venv venv
```

3. **Activa el entorno virtual:**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. **Instala las dependencias:**
```bash
pip install -r requirements.txt
```

## 🎮 Uso

### Modo Interactivo (Recomendado)
```bash
python vet_cli.py
```

### Modo Rápido con Parámetros
```bash
python vet_cli.py --quick --animal Perro --edad 24 --peso 15.0
```

### Parámetros Disponibles
- `--quick`: Ejecuta predicción rápida
- `--animal`: Tipo de animal (Perro, Gato, Conejo, Hamster, Pez)
- `--edad`: Edad en meses
- `--peso`: Peso en kilogramos

## 🤖 Modelos de Machine Learning

### Regresión Logística
- **Ventaja**: Interpretabilidad y robustez
- **Uso**: Clasificación binaria y multiclase
- **Aplicación**: Predicción de servicios veterinarios

### Árboles de Decisión
- **Ventaja**: Captura relaciones no lineales
- **Uso**: Clasificación con reglas claras
- **Aplicación**: Explicación de recomendaciones

### Random Forest
- **Ventaja**: Reduce sobreajuste y mejora precisión
- **Uso**: Ensamble de múltiples árboles
- **Aplicación**: Predicción final con alta confianza

## 📊 Características Analizadas

- **Tipo de animal**: Perro, Gato, Conejo, Hamster, Pez
- **Edad**: En meses
- **Peso**: En kilogramos
- **Género**: Macho/Hembra
- **Ubicación**: Norte, Sur, Centro, Este, Oeste
- **Historial de enfermedades**: Número de enfermedades previas
- **Último servicio**: Días desde el último servicio
- **Costo promedio**: Costo histórico de servicios
- **Satisfacción anterior**: Rating de satisfacción (1-5)
- **Score de urgencia**: Nivel de urgencia (0-1)

## 🎯 Servicios Recomendados

- **Consulta General**: Revisión rutinaria
- **Vacunación**: Servicios de inmunización
- **Cirugía**: Procedimientos quirúrgicos
- **Emergencia**: Atención urgente
- **Preventivo**: Cuidado preventivo

## 🏗️ Arquitectura del Sistema

```
VetCLISystem
├── Generación de datos sintéticos
├── Preprocesamiento (encoding, escalado)
├── Entrenamiento de modelos
├── Predicción y evaluación
└── Interfaz CLI interactiva
```

## 🔧 Desarrollo

### Estructura del Proyecto
```
vet-cli-system/
├── vet_cli.py          # Sistema principal
├── requirements.txt     # Dependencias
├── README.md          # Documentación
├── .gitignore         # Archivos a ignorar
└── venv/              # Entorno virtual (no se sube)
```

### Agregar Nuevos Modelos
Para agregar nuevos algoritmos de ML, modifica la función `prepare_and_train()` en `vet_cli.py`.

### Personalizar Datos
Ajusta la función `create_training_data()` para generar datos con diferentes distribuciones o características.

## 📈 Métricas de Rendimiento

El sistema evalúa automáticamente:
- **Accuracy**: Precisión general de cada modelo
- **Consenso**: Acuerdo entre modelos
- **Confianza**: Nivel de certeza en las predicciones

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **Platzi**: Por el curso de Fundamentos de Machine Learning que inspiró este proyecto
- **Scikit-learn**: Por las herramientas de ML utilizadas
- **Comunidad Python**: Por las librerías y herramientas de desarrollo

## 📞 Contacto

- **LinkedIn**: https://www.linkedin.com/in/matias-lobos-396725208/
---

⭐ Si este proyecto te resulta útil, ¡dale una estrella en GitHub! 
