# sepsis_detection_system/main.py
import sys
import uvicorn
from data.loader import MedicalDataLoader
from data.preprocessor import SepsisDataPreprocessor
from ensemble.trainer import train_pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from api.main import app

def main():
    """Función principal para entrenar modelo localmente"""
    try:
        print("=== ENTRENAMIENTO LOCAL DEL SISTEMA AVANZADO ===")
        
        # Usar la función train_pipeline que ya maneja todo
        filepath = r"C:\Users\axelc\Downloads\STACK\dataset\Dataset.csv"
        ensemble, results = train_pipeline(filepath, nrows=5000)
        
        print("\n=== RESULTADOS ===")
        print(f"Ensemble AUC: {results.get('ensemble_auc', 0):.4f}")
        print(f"Ensemble AUPRC: {results.get('ensemble_auprc', 0):.4f}")
        
        print("\nModelo guardado exitosamente!")
        print("Para usar la API, ejecuta: python -m uvicorn api.main:app --reload --port 8000")
        
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        logger.error(f"Error en main: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        main()
    else:
        print("Uso:")
        print("  python -m main train")
        print("  python -m main api")