# Agente de IA con LangChain y DeepSeek_Light_V1

Este proyecto implementa un **agente de IA interactivo en línea de comandos** utilizando **LangChain** y el modelo `DeepSeek_Light_V1` de Hugging Face. 

## 🔥 Sobre el modelo DeepSeek_Light_V1
Este modelo ha sido optimizado por **David Sánchez Alonso**, logrando una reducción del **50% en el uso de VRAM** sin afectar significativamente su rendimiento. Es una herramienta muy potente, aunque puede beneficiarse de mejoras adicionales como **fine-tuning y prompts más específicos** para maximizar su potencial.

🔹 **Optimización aplicada:**
- Cuantización en 4-bit BFloat16 para reducir VRAM sin perder precisión.
- Pruning para eliminar pesos innecesarios y mejorar eficiencia.

## 🚀 Características
- **Modelo de lenguaje ligero** basado en `DeepSeek_Light_V1`, optimizado para eficiencia.
- **Integración con LangChain** para una mejor gestión de consultas y respuestas.
- **Interfaz CLI interactiva** para conversar con el agente de manera intuitiva.
- **Respuestas directas**, sin generación de preguntas adicionales a menos que se soliciten.

## 🛠️ Instalación
### 1️⃣ Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_PROYECTO>
```

### 2️⃣ Crear un entorno virtual (Opcional pero recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Linux/macOS
venv\Scripts\activate    # En Windows
```

### 3️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4️⃣ Instalar PyTorch con soporte para CUDA
Si tienes una GPU compatible con CUDA, instala PyTorch optimizado:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Si usas CPU, instala PyTorch sin CUDA:
```bash
pip install torch torchvision torchaudio
```

## 🎯 Uso
Ejecuta el script principal para iniciar el asistente de IA:
```bash
python mini_deepseek.py
```
### Comandos dentro del agente:
- **Escribe tu pregunta** para recibir una respuesta concisa.
- **Escribe `salir`** para terminar la sesión.

## 📂 Estructura del Proyecto
```
📂 <NOMBRE_DEL_PROYECTO>
├── mini_deepseek.py    # Código principal del agente de IA
├── requirements.txt    # Lista de dependencias
└── README.md           # Este documento
```

## 🔧 Configuración y Personalización
Si deseas modificar el comportamiento del asistente, puedes editar el archivo `mini_deepseek.py`, especialmente el `PromptTemplate` para ajustar el tono y estilo de las respuestas.

## 🛑 Solución de Problemas
### 🔹 CUDA no es detectado
Ejecuta en Python:
```python
import torch
print(torch.cuda.is_available())
```
Si devuelve `False`, revisa la instalación de CUDA y PyTorch.

### 🔹 LangChain no encuentra `LLMChain`
Modifica la importación:
```python
from langchain.chains.llm import LLMChain
```

## 🏆 Contribución
Si deseas contribuir con mejoras, ¡sientete libre de hacer un fork y enviar un pull request!

## 📜 Licencia
Este proyecto está bajo la licencia MIT. Revisa el archivo `LICENSE` para más detalles.

