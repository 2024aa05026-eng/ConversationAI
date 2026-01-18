Install and Running commands:1

1) pip3 install -r requirements.txt
2) python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
3) python3 code/ingest.py
   Output of Step 3:- data/corpus_chunks.json

4) python code/embed_index.py
   Output of step 4:- data/faiss.index
                      data/embeddings.npy

5) python code/bm25_index.py
   Output of step 5: data/bm25.pkl

6) Launch web interface
    streamlit run app.py
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
