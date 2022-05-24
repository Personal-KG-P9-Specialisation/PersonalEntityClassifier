FROM python:3.7-slim-buster
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN apt -y update && apt -y install git
RUN git clone https://github.com/fastnlp/fastNLP.git && cd fastNLP/ && python setup.py install
RUN git clone https://github.com/fastnlp/fitlog.git && cd fitlog/ && python setup.py install
RUN pip install transformers==2.11 && pip install sklearn
RUN pip install networkx==2.6.3 && pip install matplotlib