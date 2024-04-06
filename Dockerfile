FROM python:3.10-slim
WORKDIR /src
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY src /src
COPY data /data
# Train the model, model weights are saved to data/
RUN python3 train_model.py
RUN python3 -m pip install streamlit
CMD python3 -m streamlit run --server.port 8501 app.py