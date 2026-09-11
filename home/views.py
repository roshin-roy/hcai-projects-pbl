from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Moniya Mohan", "matriculation": "675659"},
        {"name": "Roshin Roy", "matriculation": "674412"},
    ]
    
    projects = [
        {"name": "1 - Supervised Learning Interface", "url_name": "project1:index"},
        {"name": "2 - Explainability", "url_name": "project2:index"},
        {"name": "3 - Learning-to-Defer", "url_name": "project3:index"},
        {"name": "4 - Preference Elicitation", "url_name": "project4:index"}
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))