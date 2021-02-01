FROM python:3.9.1-alpine

ADD . /milestone2WebApp
WORKDIR /milestone2WebApp
# You will need this if you need PostgreSQL, otherwise just skip this
#RUN apk update && apk add postgresql-dev gcc python3-dev musl-dev libffi-dev
RUN apk add gcc libc-dev linux-headers
#RUN apk add gfortran libopenblas-dev liblapack-dev
RUN pip install uwsgi
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV PORT=8000
EXPOSE 8000
# Runner script here
CMD ["./runner.sh"]