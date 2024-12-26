Clase NaiveBayes:

    Implementa el modelo de Naive Bayes.
    Métodos principales:
        fit: Entrena el modelo usando datos de entrada categóricos (X) y etiquetas (y).
            Cuenta las frecuencias de etiquetas (label_count) y características por etiqueta (feature_count).
        predict_proba: Calcula las probabilidades de pertenencia a cada clase para un conjunto de características dadas.
            Aplica el teorema de Bayes y normaliza las probabilidades.

Optimización Matemática:

    Se usaron funciones inline en C++ para las operaciones de suma y multiplicación, en lugar del ensamblador. Estas funciones proporcionan simplicidad, portabilidad y rendimiento similar:

    inline double multiply(double a, double b) {
        return a * b;
    }

    inline double add(double a, double b) {
        return a + b;
    }

Normalización de Probabilidades:

    Calcula el total de las probabilidades para todas las etiquetas y las divide por este valor para garantizar que las probabilidades sumen 1.

Interfaz con Python (Pybind11):

    La clase NaiveBayes es expuesta como un módulo de Python llamado naive_bayes.
    Métodos disponibles en Python:
        fit(X, y): Entrena el modelo.
        predict_proba(X): Retorna un diccionario con las probabilidades para cada etiqueta.

Uso en Python: Una vez compilado como módulo, el modelo se utiliza desde Python para entrenar y predecir probabilidades con un flujo de trabajo sencillo:
