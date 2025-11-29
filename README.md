📘 Knowledge Base + Task Manager AI Agent

A lightweight AI-powered web application built using Flask, Groq LLLM, and SQLite, featuring:
📚 PDF Knowledge Base (RAG engine)
💬 AI Question Answering with citations
📝 Task Manager (Add, Mark Done, Delete)
🎨 Modern Bootstrap UI with dark mode
⚡ Fast, local embedding + retrieval

🚀 Demo Features

🔹 Knowledge Base (RAG)
Upload text-based PDFs
Text → Chunking → Embeddings (MiniLM-L6-v2)
Ask questions and receive:
AI answers
Inline citations like [source p3]
Top relevant text snippets
Auto bullet formatting
Clean HTML rendering

🔹 Task Manager

Create tasks
Add notes, due date, priority
Mark tasks as Done
Delete tasks
SQLite persistent storage
Includes badges + action buttons

UI Features

Polished Bootstrap 5 UI
Dark/Light mode toggle
Copy Answer button
Mini recent-questions history(localStorage)
Safe delete confirmation popup

🛠️ Tech Stack
Layer	    Technology
Backend	    Flask (Python)
AI Model    Groq API (LLaMA 3.x models)
Embeddings  SentenceTransformer (MiniLM-L6-v2)
Database	SQLite
Frontend	Bootstrap 5 + Icons
PDF Parsing	PyPDF
Storage	    Local filesystem + SQLite

Project Structure 

Knowledge Base and Task Manager Agent/
│
├── app_flask.py
├── requirements.txt
├── README.md
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── ask.html
│   ├── tasks.html
│   └── kb.html
│
├── tasks/
│   ├── db.py
│   └── service.py
│
└── .gitignore

⚙️ Installation & Setup

1️⃣ Clone repository
git clone https://github.com/<username>/<repo>.git
cd <repo>

2️⃣ Create virtual environment
Windows (CMD):
python -m venv .venv
.venv\Scripts\activate.bat

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set your Groq API key
Windows CMD:
set GROQ_API_KEY=your_key_here

5️⃣ Run the application
python app_flask.py
Now open:
http://127.0.0.1:5000


📘 Usage Guide


📚 Index Page
Upload one or more PDFs
Index them (page-limited for speed)
Clear KB if needed
Check stored content at /kb

💬 Ask Page
Ask any question
Select top-k chunks
Choose LLM model
View answer + citations
Use mini chat history
One-click “Copy Answer”

📝 Tasks Page
Add task
Add notes & priority
Mark tasks done
Delete tasks
Clean table view of all tasks

🚧 Limitations

No login or user accounts
Only supports text-based PDFs
No OCR for scanned documents
Retrieval uses cosine similarity only
Stored data is local (SQLite)

🔮 Future Enhancements

Authentication (JWT or OAuth)
Deployment on Render / Railway
OCR support for scanned PDFs
BM25 hybrid retriever
Editable tasks (update feature)
Upload history & user profiles

✨ Author

Nitheesh Gowda G S
AI Engineer
Building end-to-end intelligent systems.