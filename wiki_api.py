import os
import wikipediaapi
import re

# Definir los temas y los idiomas
topics = ["Second World War"]  # Varios temas de interés
languages = ['en']  # Varios idiomas (inglés, español, francés)

# Definir el User-Agent
user_agent = 'FactCheckIIWW/1.0 (100504162@alumnos.uc3m.es)'

# Función para limpiar caracteres no válidos en los nombres de archivo
def clean_filename(filename):
    # Reemplaza los caracteres no válidos para los nombres de archivo en Windows
    return re.sub(r'[\/:*?"<>|]', '_', filename)

# Función para limpiar el texto de enlaces, imágenes y otros elementos no deseados
def clean_text(text):
    # Eliminar enlaces (URLs) y direcciones de imágenes (imágenes con <img>)
    text = re.sub(r'http[s]?://\S+', '', text)  # Eliminar enlaces completos
    text = re.sub(r'<img [^>]*>', '', text)    # Eliminar etiquetas de imagen HTML
    text = re.sub(r'<a [^>]*>', '', text)      # Eliminar etiquetas de enlace HTML
    text = re.sub(r'</a>', '', text)           # Eliminar cierre de etiquetas de enlace
    text = re.sub(r'<[^>]+>', '', text)        # Eliminar otras etiquetas HTML no deseadas
    
    # Eliminar referencias de formato (como las tablas o cualquier otro contenido no textual)
    text = re.sub(r'\[\d+\]', '', text)  # Eliminar referencias tipo [1], [2], etc.

    # Eliminar caracteres innecesarios
    text = re.sub(r'\s+', ' ', text)  # Reemplaza múltiples espacios por uno solo
    text = text.strip()  # Eliminar espacios al principio y al final del texto
    
    return text

# Función para descargar, limpiar, y guardar artículos
def download_and_save_articles(topic, language):
    # Inicializa la API de Wikipedia con un User-Agent personalizado
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language=language)

    # Obtiene la página relacionada con el tema
    page = wiki_wiki.page(topic)

    # Obtén el directorio actual donde se ejecuta el script
    current_directory = os.getcwd()

    # Crea la carpeta de destino relativa al directorio actual
    folder_path = os.path.join(current_directory, f'corpus/{topic}/{language}')
    os.makedirs(folder_path, exist_ok=True)  # Crea la carpeta si no existe

    # Verifica si la página existe
    if page.exists():
        # Limpiar y guardar el artículo principal
        clean_title = clean_filename(page.title)  # Limpia el título del archivo
        clean_text_content = clean_text(page.text)  # Limpiar el texto

        # Guardar el artículo limpio
        with open(os.path.join(folder_path, f"{clean_title}.txt"), "w", encoding="utf-8") as file:
            file.write(clean_text_content)
        print(f"Artículo principal guardado como: {os.path.join(folder_path, f'{clean_title}.txt')}")

        # Buscar todos los artículos relacionados con el tema (enlaces dentro de la página)
        print(f"\nDescargando artículos relacionados con '{topic}' en idioma '{language}'...")
        for link in page.links:
            related_page = wiki_wiki.page(link)
            if related_page.exists():
                clean_related_title = clean_filename(related_page.title)  # Limpia el título del archivo relacionado
                clean_related_text_content = clean_text(related_page.text)  # Limpiar el texto relacionado

                # Guardar el artículo relacionado limpio
                with open(os.path.join(folder_path, f"{clean_related_title}.txt"), "w", encoding="utf-8") as file:
                    file.write(clean_related_text_content)
                print(f"Artículo guardado como: {os.path.join(folder_path, f'{clean_related_title}.txt')}")

    else:
        print(f"La página de Wikipedia sobre '{topic}' en idioma '{language}' no existe.")

# Descargar artículos para cada combinación de tema e idioma
for topic in topics:
    for language in languages:
        download_and_save_articles(topic, language)
