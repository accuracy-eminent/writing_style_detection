# Writing Style Detection with Machine Learning

Many historical documents exist with no known author, or have a list of possible authors with no one clear candidate.
Stylometry, or writing style analysis, can solve this problem.
Various metrics can be calculated, such as word length, sentence length, and frequency of n-grams (strings of n words), 
and richness of vocabulary can be used to establish a style of an author that can be statistically differentiated from 
the authors. By finding the text where the stylistic metrics match closest to an author, we can estimate who wrote the text.

The style of three nineteeth century authors (Charles Dickens, Jane Austen, and Herman Melville) were analyzed and a LSTM based algorithm was built
to classify a given text as being written by one
of the authors.

## Usage

To run the app, build the Docker container and open
the web app in a web browser:
```bash
docker build -t writing_style_detection .
docker run -p 8995:8501 writing_style_detection
# Navigate to http://127.0.0.1:8995
```

Alternatively, to run locally:
```bash
pip install -r requirements.txt
cd src
python3 train_model.py
streamlit run app.py
```

## Data analysis

[Here](src/eda.ipynb) is a more detailed analysis
of the authors' writing styles.