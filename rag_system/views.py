from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework import status
from .rag_utils import process_json_with_rag, query_rag_system, get_conversation_history, clear_conversation_history
import json

# Create your views here.

class RAGAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def post(self, request, *args, **kwargs):
        # Try to get file upload first
        uploaded_file = request.FILES.get('file')
        
        if uploaded_file:
            # Handle file upload
            try:
                json_data = json.load(uploaded_file)
            except Exception as e:
                return Response({'error': f'Invalid JSON file: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Handle raw JSON data
            json_data = request.data if request.data else None
            # If neither file nor JSON, json_data will be None
        result = process_json_with_rag(json_data)
        return Response(result)

class QueryAPIView(APIView):
    parser_classes = (JSONParser,)

    def post(self, request, *args, **kwargs):
        question = request.data.get('question')
        if not question:
            return Response({'error': 'No question provided.'}, status=status.HTTP_400_BAD_REQUEST)
        
        result = query_rag_system(question)
        return Response(result)

class HistoryAPIView(APIView):
    def get(self, request, *args, **kwargs):
        """Get conversation history"""
        history = get_conversation_history()
        return Response({
            "conversation_history": history,
            "total_conversations": len(history)
        })
    
    def delete(self, request, *args, **kwargs):
        """Clear conversation history"""
        result = clear_conversation_history()
        return Response(result)
