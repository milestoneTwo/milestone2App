from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    return HttpResponse("Hello, world. You're at the milestone 2 index.")

def newPage(request):
    return HttpResponse(f"Page Number 2 just for testy test")