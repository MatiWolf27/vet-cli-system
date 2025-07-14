#!/usr/bin/env python3
"""
Sistema CLI Interactivo para Recomendaciones Veterinarias
Uso: python vet_cli.py
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class VetCLISystem:
    """Sistema de CLI para recomendaciones veterinarias"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.models = {}
        self.service_mapping = {}
        self.is_trained = False
        
        # Colores para terminal
        self.COLORS = {
            'GREEN': '\033[92m',
            'BLUE': '\033[94m',
            'YELLOW': '\033[93m',
            'RED': '\033[91m',
            'PURPLE': '\033[95m',
            'CYAN': '\033[96m',
            'WHITE': '\033[97m',
            'BOLD': '\033[1m',
            'END': '\033[0m'
        }
    
    def print_colored(self, text, color='WHITE'):
        """Imprime texto con color"""
        print(f"{self.COLORS[color]}{text}{self.COLORS['END']}")
    
    def print_header(self, title):
        """Imprime un header bonito"""
        width = 60
        self.print_colored("=" * width, 'CYAN')
        self.print_colored(f"{title:^{width}}", 'BOLD')
        self.print_colored("=" * width, 'CYAN')
    
    def create_training_data(self, n_samples=1500):
        """Crea datos de entrenamiento realistas"""
        np.random.seed(42)
        
        print("📊 Generando datos de entrenamiento...")
        
        data = []
        servicios = ['Consulta_General', 'Vacunación', 'Cirugía', 'Emergencia', 'Preventivo']
        
        for i in range(n_samples):
            # Generar datos por tipo de animal
            animal_prob = np.random.random()
            if animal_prob < 0.5:
                tipo_animal = 'Perro'
                peso_base = np.random.normal(20, 10)
                edad_max = 180
            elif animal_prob < 0.8:
                tipo_animal = 'Gato'
                peso_base = np.random.normal(4, 2)
                edad_max = 200
            elif animal_prob < 0.9:
                tipo_animal = 'Conejo'
                peso_base = np.random.normal(1.5, 0.5)
                edad_max = 120
            elif animal_prob < 0.95:
                tipo_animal = 'Hamster'
                peso_base = np.random.normal(0.15, 0.05)
                edad_max = 36
            else:
                tipo_animal = 'Pez'
                peso_base = np.random.normal(0.05, 0.02)
                edad_max = 60
            
            edad_meses = np.random.randint(1, edad_max)
            peso_kg = max(0.01, peso_base)
            
            # Lógica de recomendación realista
            if edad_meses < 12:  # Animales jóvenes
                servicio = np.random.choice(['Vacunación', 'Consulta_General'], p=[0.7, 0.3])
                urgencia = np.random.uniform(0.1, 0.4)
            elif edad_meses > edad_max * 0.8:  # Animales senior
                servicio = np.random.choice(['Consulta_General', 'Emergencia', 'Preventivo'], p=[0.5, 0.3, 0.2])
                urgencia = np.random.uniform(0.4, 0.8)
            else:  # Animales adultos
                servicio = np.random.choice(servicios, p=[0.3, 0.2, 0.15, 0.1, 0.25])
                urgencia = np.random.uniform(0.2, 0.6)
            
            # Historial correlacionado con edad
            historial = max(0, int(np.random.poisson(edad_meses / 50)))
            
            # Costo base por servicio
            costo_base = {
                'Consulta_General': 80, 'Vacunación': 60, 'Preventivo': 100,
                'Cirugía': 400, 'Emergencia': 300
            }
            costo = costo_base[servicio] * (1 + peso_kg/30) * np.random.normal(1, 0.15)
            
            data.append({
                'tipo_animal': tipo_animal,
                'edad_meses': edad_meses,
                'peso_kg': peso_kg,
                'genero': np.random.choice(['Macho', 'Hembra']),
                'ubicacion': np.random.choice(['Norte', 'Sur', 'Centro', 'Este', 'Oeste']),
                'historial_enfermedades': historial,
                'ultimo_servicio_dias': np.random.randint(1, 365),
                'costo_promedio': max(20, costo),
                'satisfaccion_anterior': np.random.uniform(3.0, 5.0),
                'urgencia_score': urgencia,
                'servicio_recomendado': servicio
            })
        
        df = pd.DataFrame(data)
        self.print_colored(f"✅ Generados {len(df)} registros de entrenamiento", 'GREEN')
        return df
    
    def prepare_and_train(self, df):
        """Prepara datos y entrena modelos"""
        print("🔄 Preparando datos y entrenando modelos...")
        
        # Encoding de variables categóricas
        categorical_cols = ['tipo_animal', 'genero', 'ubicacion', 'servicio_recomendado']
        
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
        
        # Crear mapeo de servicios para interpretación
        self.service_mapping = dict(zip(
            self.encoders['servicio_recomendado'].transform(self.encoders['servicio_recomendado'].classes_),
            self.encoders['servicio_recomendado'].classes_
        ))
        
        # Definir características
        feature_cols = ['edad_meses', 'peso_kg', 'historial_enfermedades', 
                       'ultimo_servicio_dias', 'costo_promedio', 'satisfaccion_anterior',
                       'urgencia_score', 'tipo_animal_encoded', 'genero_encoded', 'ubicacion_encoded']
        
        X = df[feature_cols]
        y = df['servicio_recomendado_encoded']
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalado
        self.scalers['scaler'] = StandardScaler()
        X_train_scaled = self.scalers['scaler'].fit_transform(X_train)
        X_test_scaled = self.scalers['scaler'].transform(X_test)
        
        # Entrenar modelos
        models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        }
        
        print("🤖 Entrenando modelos:")
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            self.models[name] = model
            self.print_colored(f"  ✅ {name}: {accuracy:.3f} accuracy", 'GREEN')
        
        self.is_trained = True
        self.feature_columns = feature_cols
        
        return X_test_scaled, y_test
    
    def predict_service(self, pet_data):
        """Predice servicio para una mascota"""
        if not self.is_trained:
            self.print_colored("❌ Error: Modelos no entrenados", 'RED')
            return None
        
        try:
            # Convertir a DataFrame
            pet_df = pd.DataFrame([pet_data])
            
            # Encoding
            for col in ['tipo_animal', 'genero', 'ubicacion']:
                if col in pet_data:
                    try:
                        encoded_val = self.encoders[col].transform([pet_data[col]])[0]
                        pet_df[f'{col}_encoded'] = encoded_val
                    except ValueError:
                        # Valor no visto en entrenamiento
                        pet_df[f'{col}_encoded'] = 0
            
            # Preparar características
            pet_features = pet_df[self.feature_columns].values
            pet_scaled = self.scalers['scaler'].transform(pet_features)
            
            # Predicciones de todos los modelos
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                pred_encoded = model.predict(pet_scaled)[0]
                pred_service = self.service_mapping[pred_encoded]
                predictions[name] = pred_service
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(pet_scaled)[0]
                    # Crear dict de probabilidades por servicio
                    prob_dict = {}
                    for i, service in enumerate(self.encoders['servicio_recomendado'].classes_):
                        prob_dict[service] = proba[i]
                    probabilities[name] = prob_dict
            
            return predictions, probabilities
            
        except Exception as e:
            self.print_colored(f"❌ Error en predicción: {e}", 'RED')
            return None, None
    
    def display_predictions(self, predictions, probabilities, pet_data):
        """Muestra las predicciones de forma bonita"""
        self.print_header("🔮 RECOMENDACIONES VETERINARIAS")
        
        # Información de la mascota
        self.print_colored("\n📋 INFORMACIÓN DE LA MASCOTA:", 'BLUE')
        self.print_colored(f"   🐾 Tipo: {pet_data['tipo_animal']}", 'WHITE')
        self.print_colored(f"   📅 Edad: {pet_data['edad_meses']} meses ({pet_data['edad_meses']/12:.1f} años)", 'WHITE')
        self.print_colored(f"   ⚖️ Peso: {pet_data['peso_kg']} kg", 'WHITE')
        self.print_colored(f"   🏥 Historial: {pet_data['historial_enfermedades']} enfermedades", 'WHITE')
        self.print_colored(f"   🚨 Urgencia: {pet_data['urgencia_score']:.2f}/1.0", 'WHITE')
        
        if predictions:
            # Predicciones por modelo
            self.print_colored("\n🤖 PREDICCIONES POR MODELO:", 'PURPLE')
            
            for model_name, prediction in predictions.items():
                self.print_colored(f"\n   📊 {model_name.upper().replace('_', ' ')}:", 'YELLOW')
                self.print_colored(f"      🎯 Recomendación: {prediction}", 'GREEN')
                
                if model_name in probabilities:
                    self.print_colored("      📈 Probabilidades:", 'CYAN')
                    sorted_probs = sorted(probabilities[model_name].items(), 
                                        key=lambda x: x[1], reverse=True)
                    
                    for service, prob in sorted_probs[:3]:  # Top 3
                        bar_length = int(prob * 20)
                        bar = "█" * bar_length + "░" * (20 - bar_length)
                        self.print_colored(f"         {service:15} {bar} {prob:.1%}", 'WHITE')
            
            # Consenso
            consensus = max(set(predictions.values()), key=list(predictions.values()).count)
            confidence = list(predictions.values()).count(consensus) / len(predictions)
            
            self.print_colored(f"\n🏆 RECOMENDACIÓN FINAL:", 'BOLD')
            self.print_colored(f"   ✅ Servicio: {consensus}", 'GREEN')
            self.print_colored(f"   📊 Confianza: {confidence:.1%}", 'GREEN')
            
            # Interpretación
            self.print_colored(f"\n💡 INTERPRETACIÓN:", 'BLUE')
            if pet_data['edad_meses'] < 12:
                self.print_colored("   🐣 Mascota joven - Enfoque en prevención y vacunas", 'YELLOW')
            elif pet_data['edad_meses'] > 100:
                self.print_colored("   👴 Mascota senior - Monitoreo regular recomendado", 'YELLOW')
            
            if pet_data['urgencia_score'] > 0.6:
                self.print_colored("   🚨 Nivel de urgencia alto - Atención prioritaria", 'RED')
            elif pet_data['urgencia_score'] < 0.3:
                self.print_colored("   😌 Nivel de urgencia bajo - Rutina preventiva", 'GREEN')
            
            return consensus
        
        return None
    
    def interactive_mode(self):
        """Modo interactivo para cambiar parámetros"""
        self.print_header("🎮 MODO INTERACTIVO")
        
        # Valores por defecto
        default_pet = {
            'tipo_animal': 'Perro',
            'edad_meses': 24,
            'peso_kg': 15.0,
            'genero': 'Macho',
            'ubicacion': 'Centro',
            'historial_enfermedades': 1,
            'ultimo_servicio_dias': 90,
            'costo_promedio': 150.0,
            'satisfaccion_anterior': 4.0,
            'urgencia_score': 0.3
        }
        
        current_pet = default_pet.copy()
        
        while True:
            # Mostrar opciones
            self.print_colored("\n🔧 CONFIGURACIÓN ACTUAL:", 'BLUE')
            for i, (key, value) in enumerate(current_pet.items(), 1):
                display_key = key.replace('_', ' ').title()
                self.print_colored(f"   {i:2d}. {display_key:20} = {value}", 'WHITE')
            
            self.print_colored(f"\n   {len(current_pet)+1:2d}. 🔮 HACER PREDICCIÓN", 'GREEN')
            self.print_colored(f"   {len(current_pet)+2:2d}. 📊 EJEMPLOS RÁPIDOS", 'YELLOW')
            self.print_colored(f"   {len(current_pet)+3:2d}. ❌ SALIR", 'RED')
            
            try:
                choice = input(f"\n{self.COLORS['CYAN']}Selecciona opción (1-{len(current_pet)+3}): {self.COLORS['END']}")
                choice = int(choice)
                
                if choice == len(current_pet) + 1:  # Predicción
                    predictions, probabilities = self.predict_service(current_pet)
                    self.display_predictions(predictions, probabilities, current_pet)
                    input(f"\n{self.COLORS['YELLOW']}Presiona Enter para continuar...{self.COLORS['END']}")
                
                elif choice == len(current_pet) + 2:  # Ejemplos
                    self.quick_examples()
                    input(f"\n{self.COLORS['YELLOW']}Presiona Enter para continuar...{self.COLORS['END']}")
                
                elif choice == len(current_pet) + 3:  # Salir
                    self.print_colored("👋 ¡Hasta luego!", 'GREEN')
                    break
                
                elif 1 <= choice <= len(current_pet):  # Modificar parámetro
                    keys = list(current_pet.keys())
                    key_to_modify = keys[choice - 1]
                    
                    self.print_colored(f"\n🔧 Modificando: {key_to_modify.replace('_', ' ').title()}", 'BLUE')
                    self.print_colored(f"   Valor actual: {current_pet[key_to_modify]}", 'WHITE')
                    
                    if key_to_modify == 'tipo_animal':
                        opciones = ['Perro', 'Gato', 'Conejo', 'Hamster', 'Pez']
                        self.print_colored(f"   Opciones: {', '.join(opciones)}", 'YELLOW')
                        new_value = input(f"   Nuevo valor: ").strip().title()
                        if new_value in opciones:
                            current_pet[key_to_modify] = new_value
                        else:
                            self.print_colored("   ❌ Opción no válida", 'RED')
                    
                    elif key_to_modify == 'genero':
                        opciones = ['Macho', 'Hembra']
                        self.print_colored(f"   Opciones: {', '.join(opciones)}", 'YELLOW')
                        new_value = input(f"   Nuevo valor: ").strip().title()
                        if new_value in opciones:
                            current_pet[key_to_modify] = new_value
                        else:
                            self.print_colored("   ❌ Opción no válida", 'RED')
                    
                    elif key_to_modify == 'ubicacion':
                        opciones = ['Norte', 'Sur', 'Centro', 'Este', 'Oeste']
                        self.print_colored(f"   Opciones: {', '.join(opciones)}", 'YELLOW')
                        new_value = input(f"   Nuevo valor: ").strip().title()
                        if new_value in opciones:
                            current_pet[key_to_modify] = new_value
                        else:
                            self.print_colored("   ❌ Opción no válida", 'RED')
                    
                    else:  # Valores numéricos
                        try:
                            if key_to_modify in ['historial_enfermedades', 'ultimo_servicio_dias', 'edad_meses']:
                                new_value = int(input(f"   Nuevo valor: "))
                            else:
                                new_value = float(input(f"   Nuevo valor: "))
                            
                            current_pet[key_to_modify] = new_value
                            self.print_colored("   ✅ Valor actualizado", 'GREEN')
                        except ValueError:
                            self.print_colored("   ❌ Valor no válido", 'RED')
                
                else:
                    self.print_colored("❌ Opción no válida", 'RED')
            
            except (ValueError, KeyboardInterrupt):
                self.print_colored("\n👋 Saliendo...", 'YELLOW')
                break
    
    def quick_examples(self):
        """Ejemplos rápidos predefinidos"""
        self.print_header("⚡ EJEMPLOS RÁPIDOS")
        
        examples = [
            {
                'name': 'Cachorro Labrador',
                'data': {
                    'tipo_animal': 'Perro', 'edad_meses': 4, 'peso_kg': 5.0,
                    'genero': 'Macho', 'ubicacion': 'Norte', 'historial_enfermedades': 0,
                    'ultimo_servicio_dias': 15, 'costo_promedio': 80.0,
                    'satisfaccion_anterior': 4.5, 'urgencia_score': 0.2
                }
            },
            {
                'name': 'Gato Senior (12 años)',
                'data': {
                    'tipo_animal': 'Gato', 'edad_meses': 144, 'peso_kg': 4.2,
                    'genero': 'Hembra', 'ubicacion': 'Centro', 'historial_enfermedades': 3,
                    'ultimo_servicio_dias': 30, 'costo_promedio': 200.0,
                    'satisfaccion_anterior': 3.8, 'urgencia_score': 0.7
                }
            },
            {
                'name': 'Conejo Adulto',
                'data': {
                    'tipo_animal': 'Conejo', 'edad_meses': 36, 'peso_kg': 1.8,
                    'genero': 'Hembra', 'ubicacion': 'Sur', 'historial_enfermedades': 1,
                    'ultimo_servicio_dias': 120, 'costo_promedio': 100.0,
                    'satisfaccion_anterior': 4.2, 'urgencia_score': 0.4
                }
            }
        ]
        
        for i, example in enumerate(examples, 1):
            self.print_colored(f"\n{i}. 🐾 {example['name']}", 'YELLOW')
            predictions, probabilities = self.predict_service(example['data'])
            
            if predictions:
                consensus = max(set(predictions.values()), key=list(predictions.values()).count)
                confidence = list(predictions.values()).count(consensus) / len(predictions)
                
                # Info resumida
                data = example['data']
                self.print_colored(f"   📅 {data['edad_meses']} meses | ⚖️ {data['peso_kg']} kg | 🚨 {data['urgencia_score']:.1f}", 'WHITE')
                self.print_colored(f"   ✅ Recomendación: {consensus} ({confidence:.0%} confianza)", 'GREEN')

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Sistema de Recomendaciones Veterinarias CLI')
    parser.add_argument('--quick', action='store_true', help='Ejecutar ejemplo rápido')
    parser.add_argument('--animal', type=str, default='Perro', help='Tipo de animal')
    parser.add_argument('--edad', type=int, default=24, help='Edad en meses')
    parser.add_argument('--peso', type=float, default=15.0, help='Peso en kg')
    
    args = parser.parse_args()
    
    # Crear sistema
    vet_cli = VetCLISystem()
    
    # Header principal
    vet_cli.print_header("🏥 SISTEMA VETERINARIO CLI")
    vet_cli.print_colored("Sistema de Recomendaciones basado en Machine Learning", 'BLUE')
    vet_cli.print_colored("Desarrollado para SAAS Veterinario\n", 'BLUE')
    
    # Generar y entrenar
    print("🚀 Inicializando sistema...")
    training_data = vet_cli.create_training_data()
    vet_cli.prepare_and_train(training_data)
    
    if args.quick:
        # Modo rápido con parámetros de línea de comandos
        pet_data = {
            'tipo_animal': args.animal,
            'edad_meses': args.edad,
            'peso_kg': args.peso,
            'genero': 'Macho',
            'ubicacion': 'Centro',
            'historial_enfermedades': 1,
            'ultimo_servicio_dias': 90,
            'costo_promedio': 150.0,
            'satisfaccion_anterior': 4.0,
            'urgencia_score': 0.3
        }
        
        predictions, probabilities = vet_cli.predict_service(pet_data)
        vet_cli.display_predictions(predictions, probabilities, pet_data)
    else:
        # Modo interactivo
        vet_cli.interactive_mode()

if __name__ == "__main__":
    main()