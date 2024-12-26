from fastapi import FastAPI
import naive_bayes
import time
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[str]]

# Definir el modelo para el vector
class VectorS(BaseModel):
    vector: List[str]

class VectorI(BaseModel):
    vector: List[int]

@app.post("/naive-bayes")
async def calculo(X: Matrix, y: VectorS,
                  prediccion: VectorS):
    start = time.time()

    # Datos de ejemplo
    #X = [["sunny", "hot", "high"], ["sunny", "hot", "high"], ["overcast", "hot", "high"], ["rainy", "mild", "high"]]
    #y = ["no", "no", "yes", "yes"]

    model = naive_bayes.NaiveBayes()
    model.fit(X.matrix, y.vector)

    # Predicción ["sunny", "hot", "high"]
    proba = model.predict_proba(prediccion.vector)
    
    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "Prediccion": proba
    }
    jj = json.dumps(str(j1))

    return jj