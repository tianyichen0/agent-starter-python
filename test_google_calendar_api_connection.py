# test_connection.py
import json
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar"]

creds = Credentials.from_authorized_user_file("token.json", SCOPES)

if not creds.valid:
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # save refreshed token
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    else:
        raise RuntimeError("Invalid credentials and no refresh token available.")

service = build("calendar", "v3", credentials=creds)

# Test 1: list calendars (confirms auth works)
calendar_list = service.calendarList().list().execute()
print("Calendars:")
for cal in calendar_list.get("items", []):
    print(f"  - {cal['summary']} ({cal['id']})")

# Test 2: list upcoming events on primary calendar
import datetime
now = datetime.datetime.utcnow().isoformat() + "Z"
events_result = service.events().list(
    calendarId="primary",
    timeMin=now,
    maxResults=5,
    singleEvents=True,
    orderBy="startTime",
).execute()

events = events_result.get("items", [])
print("\nUpcoming events:")
if not events:
    print("  No upcoming events found.")
for event in events:
    start = event["start"].get("dateTime", event["start"].get("date"))
    print(f"  - {event.get('summary')} at {start}")