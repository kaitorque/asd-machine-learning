from django.shortcuts import render
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from asd_ml.helpers import *
from collections import namedtuple
from django.db import connection
from urllib.parse import parse_qs
from django.core.files.storage import default_storage

now = timezone.now()
cursor = connection.cursor()

# Create your views here.
def empty(request):
    page = { "title": "empty", "sub": "" }
    return render(request, 'html/empty.html', { "currentTime": now, "page": page })

def home(request):
    page = { "title": "empty", "sub": "" }
    return render(request, 'html/empty.html', { "currentTime": now, "page": page })

def model(request):
    page = { "title": "model", "sub": "" }
    if request.method == 'GET':
        return render(request, 'html/model.html', { "currentTime": now, "page": page })
    elif request.method == 'POST':
        errorMsg = []
        if request.POST['step'] == "add":
            Input = namedtuple('Input',['name','value'])
            # tbfile = request.POST['tbfile']
            trainsize = request.POST['trainsize']
            numneuron = request.POST.getlist('numneuron')
            maxepoch = request.POST['maxepoch']
            inputlist = [
                # Input("File", tbfile),
                Input("Taining Size", trainsize),
                # Input("Number of Neutron", numneutron),
                Input("Max Epoch", maxepoch),
            ]
            errorMsg.extend(checkEmpty(inputlist))
            inputlist = [
                Input("Taining Size", trainsize),
                # Input("Number of Neutron", numneutron),
                Input("Max Epoch", maxepoch),
            ]
            errorMsg.extend(checkDigit(inputlist))
            if not numneuron:
                errorMsg.append("Neuron cannot be empty.")
            if not errorMsg:
                cursor.execute("INSERT INTO model "
                "(trainsize, maxepoch)"
                " VALUES "
                "(%s, %s)", [int(trainsize), int(maxepoch)])
                lastid = cursor.lastrowid
                cursor.execute("UPDATE model "
                "SET filename = %s"
                " WHERE id = %s ", [str(lastid)+".csv", lastid])
                insertstr = "INSERT INTO layer (model_id, numneuron) VALUES "
                valuearr = []
                paramarr = []
                for x in numneuron:
                    valuearr.append("(%s, %s)")
                    paramarr.extend([lastid, x])
                valuestr = ",".join(valuearr)
                insertstr = insertstr+valuestr
                # print(valuearr)
                cursor.execute(insertstr, paramarr);
                csv_file = request.FILES["tbfile"]
                csv_file_name = default_storage.save("ml_web/static/file/"+str(lastid)+".csv", csv_file)
                return JsonResponse({"success": True, "response": "Data Set & Model successfully added!"})
        else:
            errorMsg.append("Request Error")
        return JsonResponse({"success": False, "response": errorMsg})
