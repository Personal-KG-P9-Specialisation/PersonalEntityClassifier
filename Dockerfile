FROM python:3.7-slim-buster
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN apt -y update && apt -y install git
RUN git clone https://github.com/fastnlp/fastNLP.git && cd fastNLP/ && python setup.py install
RUN git clone https://github.com/fastnlp/fitlog.git && cd fitlog/ && python setup.py install
RUN pip install transformers==2.11 && pip install sklearn
RUN pip install networkx==2.6.3 && pip install matplotlib
RUN mkdir code
#RUN cd code && git clone https://github.com/Personal-KG-P9-Specialisation/PersonalEntityClassifier.git .



#get annotations data
RUN git clone https://github.com/Personal-KG-P9-Specialisation/PKGAnnotationSystem.git
RUN apt install git-lfs && cd PKGAnnotationSystem && git-lfs pull
#temp code
#RUN cd code && git pull origin colake

#Get data
RUN mkdir data
RUN cp /PKGAnnotationSystem/annotations_data/final_updated_filtered_relation_annotated_triples.jsonl /data/final_updated_filtered_relation_annotated_triples.jsonl
RUN cp /PKGAnnotationSystem/annotations_data/final_annotated_conceptnet_entities.jsonl /data/final_annotated_conceptnet_entities.jsonl
RUN cp /PKGAnnotationSystem/annotations_data/final_annotated_personal_entities.jsonl /data/final_annotated_personal_entities.jsonl
RUN rm -rf PKGAnnotationSystem
#RUN cd code/data && python -m preprocessor

#RUN python -m utils
RUN pip install spacy && pip install neuralcoref && pip install stanfordcorenlp
COPY / code/
RUN chmod +x code/run.sh
CMD ['code/run.sh']