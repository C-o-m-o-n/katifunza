from rest_framework import serializers

class AgentPromptSerializer(serializers.Serializer):
    question = serializers.CharField()