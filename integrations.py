# integrations.py
import os
import logging
from datetime import datetime
import json
from typing import Optional, Dict, Any

import aiohttp
from telegram import Update
from telegram.ext import ContextTypes
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

class IntegrationsManager:
    """Manages all external service integrations for the Telegram bot"""
    
    def __init__(self):
        self.google_creds = None
        self.session = None
        
    async def initialize(self):
        """Initialize HTTP session and authenticate services"""
        self.session = aiohttp.ClientSession()
        await self.authenticate_google()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

    async def authenticate_google(self):
        """Authenticate with Google services"""
        SCOPES = ['https://www.googleapis.com/auth/calendar.readonly',
                 'https://www.googleapis.com/auth/tasks.readonly',
                 'https://www.googleapis.com/auth/gmail.readonly']
        
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
                
        self.google_creds = creds

    async def get_calendar_events(self, days: int = 7) -> list:
        """Fetch upcoming calendar events"""
        try:
            service = build('calendar', 'v3', credentials=self.google_creds)
            now = datetime.utcnow().isoformat() + 'Z'
            
            events_result = service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=10,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            formatted_events = []
            
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                formatted_events.append({
                    'summary': event['summary'],
                    'start': start,
                    'link': event.get('htmlLink', '')
                })
                
            return formatted_events
            
        except HttpError as error:
            logger.error(f"Error fetching calendar events: {error}")
            return []

    async def get_tasks(self) -> list:
        """Fetch tasks from Google Tasks"""
        try:
            service = build('tasks', 'v1', credentials=self.google_creds)
            results = service.tasklists().list(maxResults=10).execute()
            items = results.get('items', [])
            
            all_tasks = []
            for item in items:
                task_list_id = item['id']
                tasks = service.tasks().list(tasklist=task_list_id, maxResults=10).execute()
                
                for task in tasks.get('items', []):
                    all_tasks.append({
                        'title': task['title'],
                        'due': task.get('due', ''),
                        'status': task['status']
                    })
                    
            return all_tasks
            
        except HttpError as error:
            logger.error(f"Error fetching tasks: {error}")
            return []

    async def get_unread_emails(self, max_results: int = 5) -> list:
        """Fetch unread emails from Gmail"""
        try:
            service = build('gmail', 'v1', credentials=self.google_creds)
            results = service.users().messages().list(
                userId='me',
                labelIds=['UNREAD'],
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            email_list = []
            
            for msg in messages:
                email = service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='metadata',
                    metadataHeaders=['Subject', 'From']
                ).execute()
                
                headers = email['payload']['headers']
                subject = next(h['value'] for h in headers if h['name'] == 'Subject')
                sender = next(h['value'] for h in headers if h['name'] == 'From')
                
                email_list.append({
                    'subject': subject,
                    'from': sender,
                    'id': msg['id']
                })
                
            return email_list
            
        except HttpError as error:
            logger.error(f"Error fetching emails: {error}")
            return []
