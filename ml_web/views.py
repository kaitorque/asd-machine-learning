from django.shortcuts import render
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from asd_ml.helpers import *
from collections import namedtuple
from django.db import connection
from urllib.parse import parse_qs
from django.core.files.storage import default_storage
from asd_ml.model import *

now = timezone.now()
cursor = connection.cursor()

# Create your views here.
def empty(request):
    page = { "title": "empty", "sub": "" }
    return render(request, 'html/empty.html', { "currentTime": now, "page": page })

def home(request):
    page = { "title": "home", "sub": "" }
    if request.method == 'GET':
        if 'step' in request.GET and request.GET['step'] == "modeltable":
            cursor.execute("SELECT m.id, m.filename, trainsize, maxepoch, group_concat(l.numneuron) neuron_layer, tracc, teacc FROM model m "
            "INNER JOIN layer l ON m.id = l.model_id "
            "GROUP BY m.id")
            datax = dictfetchall(cursor)
            datay = list(map(cLink , datax))
            return JsonResponse({"data": datay })
        else:
            return render(request, 'html/home.html', { "currentTime": now, "page": page })
    elif request.method == 'POST':
        #Declare array of errorMsg
        errorMsg = []
        delid = parse_qs(decrypt(request.POST['delid']))
        if request.POST['step'] == "delete":
            cursor.execute("DELETE FROM model WHERE id = %s", delid["id"])
            cursor.execute("DELETE FROM layer WHERE model_id = %s", delid["id"])
            default_storage.delete("ml_web/static/file/"+str(delid["id"][0])+".csv");
            default_storage.delete("ml_web/static/file/"+str(delid["id"][0])+".h5");
            default_storage.delete("ml_web/static/file/"+str(delid["id"][0])+".json");
            if cursor.rowcount > 0:
                response = "Model record successfully deleted!"
            else:
                response = "No change detected. No record updated!"
            return JsonResponse({"success": True, "response": response})
        else:
            errorMsg.append("Request Error")
        return JsonResponse({"success": False, "response": errorMsg})


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
                acc = createModel(lastid, trainsize, maxepoch, numneuron)
                cursor.execute("UPDATE model SET tracc = %s, teacc = %s WHERE id = %s", [str(acc['training']),str(acc['testing']),lastid])
                return JsonResponse({"success": True, "response": "Data Set & Model successfully added!"})
        else:
            errorMsg.append("Request Error")
        return JsonResponse({"success": False, "response": errorMsg})

def evaluate(request):
    page = { "title": "home", "sub": "" }
    errorMsg = []

    if request.method == 'GET':
        delid = parse_qs(decrypt(request.GET['q']))
        # cursor.execute("SELECT * FROM model WHERE id = %s", delid["id"])
        # datax = dictfetchone(cursor)
        # fuzzy = fuzzy_calc(datax)
        # cursor.execute("UPDATE client SET coping_level = %s WHERE id = %s", [fuzzy['coping'], datax["id"]])
        # link = "cog/"+str(datax["id"])+".png"
        return render(request, 'html/evaluate.html', { "currentTime": now, "page": page, "delid": delid["id"][0] })
    elif request.method == 'POST':
        if request.POST['step'] == 'evaluate':
            id = request.POST['id']
            csv_file = request.FILES["tbfile"]
            prediction = loadModel(id, csv_file)
            return JsonResponse({"success": True, "response": "Successfuly evaluate", "data": prediction })
        elif request.POST['step'] == 'qevaluate':
            id = request.POST['id']
            q1 = request.POST['q1']
            q2 = request.POST['q2']
            q3 = request.POST['q3']
            q4 = request.POST['q4']
            q5 = request.POST['q5']
            q6 = request.POST['q6']
            q7 = request.POST['q7']
            q8 = request.POST['q8']
            q9 = request.POST['q9']
            q10 = request.POST['q10']
            qage = request.POST['age']
            qjaundice = request.POST['jaundice']
            qautism = request.POST['autism']
            qgender = request.POST['gender']

            question = {"q1" : q1,
                        "q2" : q2,
                        "q3" : q3,
                        "q4" : q4,
                        "q5" : q5,
                        "q6" : q6,
                        "q7" : q7,
                        "q8" : q8,
                        "q9" : q9,
                        "q10" : q10,
                        "age" : qage,
                        "jaundice" :  qjaundice ,
                        "autism" : qautism,
                        "gender" : qgender}
            prediction = loadModel2(id,question)
            return JsonResponse({"success": True, "response": "Successfuly evaluate", "data": prediction})
        else:
            errorMsg.append("Request Error")
    else:
        errorMsg.append("Request Error")

    return JsonResponse({"success": False, "response": errorMsg})