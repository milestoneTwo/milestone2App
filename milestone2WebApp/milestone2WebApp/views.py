from django.http import HttpResponse
from configurations import ROOT_DIR

def home(request):
    return HttpResponse(ROOT_DIR)

def api(request):
    pass