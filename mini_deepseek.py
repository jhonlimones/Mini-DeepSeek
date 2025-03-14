from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain.llms.base import LLM
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# Configurar el modelo y el tokenizador
model_name = "sanchezalonsodavid17/DeepSeek_Light_V1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config
)

# Definir clase personalizada para integrar Hugging Face con LangChain
class HuggingFaceLLM(LLM):
    def _call(self, prompt: str, stop=None, max_length=500, **kwargs) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()

    @property
    def _identifying_params(self):
        return {"model_name": model_name}

    @property
    def _llm_type(self):
        return "huggingface"

# Crear la cadena LLM con un prompt template
prompt_template = (
    "Eres un asistente de IA. Responde únicamente la pregunta del usuario de manera clara y concisa. "
    "No generes preguntas adicionales a menos que el usuario lo solicite. "
    "Si el usuario pregunta sobre un tema amplio, proporciona una respuesta clara sin extenderte demasiado. "
    "Mantén las respuestas dentro de los 500 caracteres.\n\n"
    "Pregunta: {question}\n"
    "Respuesta:"
)
prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
llm = HuggingFaceLLM()
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Interfaz de línea de comandos para interactuar con el modelo
print("Agente de IA interactivo. Escribe 'salir' para terminar.")
while True:
    user_input = input("Tú: ")
    if user_input.lower() == "salir":
        print("Cerrando el agente...")
        break
    response = llm_chain.invoke(user_input)  # Eliminamos espacios extra
    print(response)
