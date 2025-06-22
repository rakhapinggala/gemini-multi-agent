from utils.gemini import gemini_prompt

def get_router_chain():
    def run(self, query, *args, **kwargs):
        prompt = f"""
Klasifikasikan pertanyaan ini menjadi salah satu kategori: IT, HR, Finance, Investor.

Pertanyaan: {query}
Jawaban hanya salah satu kata: IT / HR / Finance / Investor
"""
        return gemini_prompt(prompt)
    return type("RouterChain", (), {"run": run})()
