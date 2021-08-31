from django.http.response import HttpResponse
from .result import main
from django.shortcuts import render, redirect
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.exceptions import NotFound
from django.urls import include

@api_view(['GET', 'POST'])
def get_df(request):
    answer,graph1,graph2,pie1,pie2,prediction = main(request.data)
    result = "Buy" if answer== 1 else "Sell"
    print(result)
    return JsonResponse({'result': result, 'df1':graph1, 'df2': graph2, 'pie1': pie1, 'pie2': pie2, 'prediction': prediction})

