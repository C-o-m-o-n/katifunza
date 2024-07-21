import os
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
# from backend.katifunza_server import settings
from data_magic.data_job import PreProcess
from main import OurCrew
from dotenv import load_dotenv
from textwrap import dedent
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import AgentPromptSerializer


# Create your views here.
preprocessor = PreProcess()
pdf_path = os.path.join(settings.MEDIA_ROOT, "constitution.pdf")

vector_store = preprocessor.store_embeddings(pdf_path)

class AgentPromptView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = AgentPromptSerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            crew = OurCrew(question, vector_store)
            result = crew.run()
            return Response(result, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

