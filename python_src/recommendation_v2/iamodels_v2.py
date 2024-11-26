import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recomendar(df, columnas_descripcion, prompt, pares_columna_valor, columnas_outputs, umbral_similitud=0.1, num_recomendaciones=1):
    # Concatenar listas de síntomas en cadenas de texto

    for columna in columnas_descripcion:
        if isinstance(df[columna].iloc[0], list):
            df[columna] = df[columna].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))
    
    # Filtrar el DataFrame basado en los pares columna-valor
    if pares_columna_valor:
        def check_value(row, columna, valores):
            if isinstance(row[columna], list):
                return any(val in row[columna] for val in valores) if isinstance(valores, list) else valores in row[columna]
            else:
                return row[columna] in valores if isinstance(valores, list) else row[columna] == valores

        mascara = df.apply(lambda row: all(check_value(row, columna, valores) for columna, valores in pares_columna_valor), axis=1)
        df_filtrado = df[mascara]
    
        if df_filtrado.empty:
            print(f"Error: No se encontraron filas que coincidan con todos los criterios.")
            return pd.DataFrame()

        # Eliminar filas con NaN en las columnas de descripción
        df_filtrado = df_filtrado.dropna(subset=columnas_descripcion)
        indices_filtrados = df_filtrado.index.tolist()
    else:
        df_filtrado = df.dropna(subset=columnas_descripcion)
        indices_filtrados = df_filtrado.index.tolist()

    # Inicializar matriz de similitud para texto y numérico
    matriz_similitud = None

    # Procesar columnas de texto y numéricas
    for columna in columnas_descripcion:
        if pd.api.types.is_numeric_dtype(df_filtrado[columna]):  # Si la columna es numérica
            valores = df_filtrado[columna].astype(float).tolist() + [float(prompt[columna][0])]
            valores = np.array(valores).reshape(-1, 1)
            similitudes = cosine_similarity(valores)[-1, :-1]
        else:  # Si la columna es texto
            vectorizer = TfidfVectorizer()
            textos = df_filtrado[columna].str.lower().tolist() + [str(prompt[columna][0]).lower()]
            textos_vectorizados = vectorizer.fit_transform(textos)
            similitudes = cosine_similarity(textos_vectorizados)[-1, :-1]

        if matriz_similitud is None:
            matriz_similitud = similitudes
        else:
            matriz_similitud = np.minimum(matriz_similitud, similitudes)

    if matriz_similitud is None:
        print(f"Error: No se pudieron calcular similitudes.")
        return pd.DataFrame()

    # Asegurarse de que los tamaños de las listas coincidan
    if len(matriz_similitud) != len(indices_filtrados):
        print(f"Error: La longitud de la matriz de similitud ({len(matriz_similitud)}) no coincide con la de los índices filtrados ({len(indices_filtrados)}).")
        return pd.DataFrame()

    # Buscar usuarios similares
    recomendaciones = [(indices_filtrados[i], similitud) for i, similitud in enumerate(matriz_similitud) if similitud > umbral_similitud]

    # Ordenar las recomendaciones por similitud y seleccionar las mejores
    recomendaciones.sort(key=lambda x: x[1], reverse=True)
    # recomendaciones = recomendaciones[:num_recomendaciones]

    # Obtener los índices recomendados y los datos correspondientes
    indices_recomendados = [indice for indice, _ in recomendaciones]
    df_recomendaciones = df.loc[indices_recomendados, columnas_outputs].reset_index(drop=True)
    df_recomendaciones_unicas = df_recomendaciones.drop_duplicates(subset=columnas_outputs)
    
    # Imprimir las recomendaciones finales
    # print("Recomendaciones finales:")
    #print(df_recomendaciones_unicas)

    return df_recomendaciones_unicas[:num_recomendaciones], [similitud for _, similitud in recomendaciones]