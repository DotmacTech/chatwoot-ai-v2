"""
Handlers package for processing messages through the standardized pipeline.
"""
from .base import BaseHandler
from .chatwoot import ChatwootHandler

__all__ = ['BaseHandler', 'ChatwootHandler']
