from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout


def home(request):
    return render(request, "home.html")


def login_view(request):
    return render(request, "login.html")


@login_required
def analyses_form(request):
    # renders the template for creating a new analysis
    return render(request, "new_analyses.html")


@login_required
def my_analyses(request):
    return render(request, "my_analyses.html")


@login_required
def analyses_result(request, analysis_id=None):
    # show the result template; analysis_id is optional for demo
    return render(request, "result.html", {"analysis_id": analysis_id})


@login_required
def download_analysis_result(request, analysis_id):
    # placeholder: real implementation should return a file response
    return redirect("my-analyses")


@login_required
def delete_analysis_result(request, analysis_id):
    # placeholder: real implementation should delete and redirect
    return redirect("my-analyses")


def logout_view(request):
    logout(request)
    return redirect("home")
