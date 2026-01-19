Github URl: https://github.com/2024aa05026-eng/ConversationAI.git

Pre-requisite:
To get URL for fixed and random
1) python collect_urls.py


Install and Running commands:

1) pip3 install -r requirements.txt
2) python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
3) python3 src/ingest.py
   Output of Step 3:- data/corpus_chunks.json

4) python3 src/embed_index.py
   Output of step 4:- data/faiss.index
                      data/embeddings.npy

5) python3 src/bm25_index.py
   Output of step 5: data/bm25.pkl

6) Launch web interface
    streamlit run app.py --server.runOnSave=false --server.fileWatcherType=none
    streamlit run app.py --server.fileWatcherType=none

    Open browser: http://localhost:8501
    
7) You can now:
    Enter questions
    View generated answers
    View retrieved Wikipedia sources


Optional:
If you see error related 
ValueError: Your currently installed version of Keras is Keras 3,
but this is not yet supported in Transformers.
Please install tf-keras


1) pip uninstall -y tensorflow keras tf-keras
2) pip install torch sentence-transformers --upgrade


Preferred version
Python max 3.11
pip uninstall -y numpy
pip install numpy==1.26.4
pip uninstall -y faiss-cpu
pip install faiss-cpu==1.7.4
