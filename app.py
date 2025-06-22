from flask import Flask, render_template, request
from agents.it_agent import get_it_chain
from agents.hr_agent import get_hr_chain
from agents.finance_agent import get_finance_chain
from agents.router_agent import get_router_chain
from agents.invest_agent import get_invest_chain
from markdown import markdown
from generate_report import generate_report
from langchain.schema import Document

app = Flask(__name__)

# Inisialisasi semua agen
it_agent = get_it_chain()
hr_agent = get_hr_chain()
finance_agent = get_finance_chain()
invest_agent = get_invest_chain()
router_agent = get_router_chain()

report_filename = ""
fake_sources = []

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    category = ""
    if request.method == "POST":
        query = request.form["query"]
        category = router_agent.run(query).strip().lower() # untuk mengklasifikasikan pertanyaan dengan agen router

        if category == "it":
            answer = it_agent.run(query)
        elif category == "hr":
            answer = hr_agent.run(query)
        elif category == "finance":
            answer = finance_agent.run(query)
        elif category == "investor":
            answer = invest_agent.run(query)
        else:
            answer = "Maaf, saya tidak bisa mengklasifikasikan pertanyaan Anda."

    answer_html = markdown(answer)
    category = category.upper()
    category_descriptions = {
    "IT": "Orang yang tidak percaya diri",
    "FINANCE": "Orang yang licik",
    "HR": "Orang yang subjektif",
    "INVESTOR": "Orang yang berorientasi pada investasi"
}
    sources = {
    "IT": "troubleshooting IT.txt",
    "FINANCE": "kebijakan reimbursemen.txt",
    "HR": "kebijakan cuti.txt",
    "INVESTOR": "investasi_emas.txt"
}
    source_doc = sources.get(category, "")
    fake_sources = []
    if source_doc:
        fake_sources.append(Document(page_content="", metadata={"source": source_doc}))

    label = f"{category} - {category_descriptions.get(category, '')} - {sources.get(category, '')}"
    report_filename = generate_report(query, answer, label, fake_sources)
    return render_template("index.html", answer=answer_html, category_label=label, report_filename=report_filename, fake_sources=fake_sources)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
