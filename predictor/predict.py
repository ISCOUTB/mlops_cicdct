#!/usr/bin/env python3
"""
Script para hacer predicciones usando el modelo Iris entrenado
"""

import pickle
import numpy as np
import os
import argparse


def load_model(model_path="models/iris_model_latest.pkl"):
    """Carga el modelo entrenado desde el archivo pickle"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def predict_iris(sepal_length, sepal_width, petal_length, petal_width, model_path="models/iris_model_latest.pkl"):
    """
    Hace una predicción para una muestra de Iris
    
    Args:
        sepal_length: Longitud del sépalo
        sepal_width: Ancho del sépalo  
        petal_length: Longitud del pétalo
        petal_width: Ancho del pétalo
        model_path: Ruta al modelo guardado
    
    Returns:
        dict: Predicción y probabilidades
    """
    # Cargar modelo
    model_data = load_model(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    target_names = model_data['target_names']
    
    # Preparar datos de entrada
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    
    # Hacer predicción
    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]
    
    # Preparar resultado
    result = {
        'input': {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        },
        'prediction': target_names[prediction],
        'prediction_index': int(prediction),
        'probabilities': {
            target_names[i]: float(prob) for i, prob in enumerate(probabilities)
        },
        'confidence': float(max(probabilities))
    }
    
    return result


def main():
    """Función principal para línea de comandos"""
    parser = argparse.ArgumentParser(description='Predicción de especies de Iris')
    parser.add_argument('--sepal_length', type=float, required=True, help='Longitud del sépalo')
    parser.add_argument('--sepal_width', type=float, required=True, help='Ancho del sépalo')
    parser.add_argument('--petal_length', type=float, required=True, help='Longitud del pétalo')
    parser.add_argument('--petal_width', type=float, required=True, help='Ancho del pétalo')
    parser.add_argument('--model_path', type=str, default='models/iris_model_latest.pkl', help='Ruta al modelo')
    
    args = parser.parse_args()
    
    try:
        result = predict_iris(
            args.sepal_length,
            args.sepal_width,
            args.petal_length,
            args.petal_width,
            args.model_path
        )
        
        print("=" * 50)
        print("PREDICCIÓN DE ESPECIE DE IRIS")
        print("=" * 50)
        print(f"Entrada:")
        print(f"  Longitud sépalo: {result['input']['sepal_length']}")
        print(f"  Ancho sépalo:    {result['input']['sepal_width']}")
        print(f"  Longitud pétalo: {result['input']['petal_length']}")
        print(f"  Ancho pétalo:    {result['input']['petal_width']}")
        print()
        print(f"Predicción: {result['prediction']}")
        print(f"Confianza: {result['confidence']:.4f}")
        print()
        print("Probabilidades por clase:")
        for species, prob in result['probabilities'].items():
            print(f"  {species}: {prob:.4f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()