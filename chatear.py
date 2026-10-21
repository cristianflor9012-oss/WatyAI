import os
import sys
import time
import threading
from mlx_lm import load, stream_generate
from ddgs import DDGS

# Desactivar avisos de paralelismo en los tokenizadores
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

class CargandoAnimacion:
    """Muestra indicadores animados mientras el modelo procesa o busca en la web."""
    def __init__(self):
        self.cargando = False
        self.hilo = None
        self.mensaje_base = "Pensando"

    def _animar(self):
        iconos = ["   ", ".  ", ".. ", "..."]
        i = 0
        while self.cargando:
            sys.stdout.write(f"\r\033[1;33mWatyAI:\033[0m {self.mensaje_base}{iconos[i % len(iconos)]}")
            sys.stdout.flush()
            time.sleep(0.3)
            i += 1
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

    def iniciar(self, mensaje="Pensando"):
        self.mensaje_base = mensaje
        self.cargando = True
        self.hilo = threading.Thread(target=self._animar)
        self.hilo.start()

    def detener(self):
        self.cargando = False
        if self.hilo:
            self.hilo.join()

def buscar_en_internet(query: str):
    """Busca información actualizada en la web usando ddgs."""
    textos = []
    fuentes = []
    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(query, max_results=3))
            for r in resultados:
                fuentes.append(r.get("title", "Web"))
                textos.append(r.get("body", ""))
    except Exception as e:
        pass
    return textos, fuentes

def main():
    limpiar_pantalla()
    print("------------------------------------------------------------------")
    print("  ExeLux Workspace - Consola Interactiva Local (Qwen 2.5 7B + Web)")
    print("------------------------------------------------------------------")
    print("Cargando el modelo Qwen 2.5 7B (3-bit) en la memoria unificada...")

    model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-3bit")

    print("\n¡WatyAI está listo! Escribe '/salir' para terminar o '/reset' para limpiar memoria.")
    print("------------------------------------------------------------------\n")

    # Prompt del sistema para ExeLux
    SYSTEM_PROMPT = (
        "Eres WatyAI, un motor de ejecución de código y asistente técnico desarrollado para ExeLux.\n"
        "REGLA OBLIGATORIA: Debes responder EXCLUSIVAMENTE en idioma español latino.\n"
        "Fecha actual del sistema: Año 2026.\n"
        "Se neutral, directo y preciso. Si generas código, especifica siempre el lenguaje (ejemplo: ```python, ```javascript, ```html).\n"
        "Utiliza formato Markdown claro para títulos, negritas y listas.\n"
        "Si se te proporciona información de la web, úsala para enriquecer tu respuesta técnica de forma precisa."
    )

    historial_mensajes = []
    animacion = CargandoAnimacion()

    while True:
        try:
            user_input = input("\033[1;36mTú:\033[0m ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["/salir", "exit", "quit"]:
                print("\n\033[1;33mWatyAI:\033[0m ¡Nos vemos! Que tengas un excelente día de desarrollo en ExeLux.\n")
                break
            
            if user_input.lower() in ["/reset", "reiniciar"]:
                historial_mensajes.clear()
                print("\n\033[1;33mWatyAI:\033[0m ¡Memoria limpiada! Empezamos desde cero.\n")
                continue

            # Detectar si la pregunta requiere buscar en la web
            palabras_clave = ["busca", "último", "noticia", "qué es", "quién es", "cómo funciona", "versión", "actual", "documentación"]
            contexto_web = ""
            fuentes_encontradas = []

            if any(k in user_input.lower() for k in palabras_clave) or len(user_input) > 20:
                animacion.iniciar("Buscando en internet")
                textos_web, fuentes_encontradas = buscar_en_internet(user_input)
                animacion.detener()
                if textos_web:
                    contexto_web = "\n".join(textos_web)
                    print(f"\033[1;90m[Fuentes web consultadas: {', '.join(fuentes_encontradas)}]\033[0m")

            historial_mensajes.append({"role": "user", "content": user_input})

            # Construcción del prompt
            prompt_actual = f"<|im_start|>system\n{SYSTEM_PROMPT}\n"
            if contexto_web:
                prompt_actual += f"\n[Información relevante encontrada en internet:\n{contexto_web}]\n"
            prompt_actual += "<|im_end|>\n"

            for msg in historial_mensajes:
                role = msg["role"]
                content = msg["content"]
                prompt_actual += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            prompt_actual += "<|im_start|>assistant\n"

            # Animar pensamiento
            animacion.iniciar("Pensando")

            # Generación en streaming corregida (sin argumentos incompatibles)
            stream = stream_generate(
                model, 
                tokenizer, 
                prompt=prompt_actual, 
                max_tokens=1536
            )

            respuesta_completa = ""
            primera_iteracion = True

            for chunk in stream:
                if primera_iteracion:
                    animacion.detener()
                    print("\033[1;33mWatyAI:\033[0m ", end="", flush=True)
                    primera_iteracion = False
                
                texto_chunk = chunk.text.replace("<|im_end|>", "").replace("<|im_start|>", "")
                
                if texto_chunk:
                    sys.stdout.write(texto_chunk)
                    sys.stdout.flush()
                    respuesta_completa += texto_chunk

            print("\n")

            historial_mensajes.append({"role": "assistant", "content": respuesta_completa.strip()})

        except KeyboardInterrupt:
            animacion.detener()
            print("\n\n\033[1;33mWatyAI:\033[0m Sesión interrumpida. ¡Hasta luego!\n")
            break
        except Exception as e:
            animacion.detener()
            print(f"\n\033[1;31m[Error de ejecución]: {str(e)}\033[0m\n")

if __name__ == "__main__":
    main()