import uuid
from typing import Dict, List, Any
from openai import OpenAI

from fastapi import FastAPI
from dataclasses import dataclass, field

app = FastAPI()
client = OpenAI(
    api_key = ""
)

def generate_response(messages: List[Dict[str, str]]) -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return completion.choices[0].message

@dataclass
class Userport:
    name: str
    authentication: "Function"


@dataclass
class User:
    name: str
    phoneNumber: str
    email: str
    Userports: list[Userport] = field(default_factory=list)

    def addPhonePort(self):
        self.Userports.append(Userport("Phone", lambda number: self.phoneNumber == number))
        Users.append(self)

    def authenticate(self, method, arg):
        for port in self.Userports:
            if port.name == method:
                return port.authentication(arg)
        return False


@dataclass
class Session:
    user: User
    id: str


@dataclass()
class Machine:
    name: str


@dataclass
class Problem:
    name: str
    description: str


@dataclass
class Session:
    id: str
    username: str
    machine: Machine
    problem: Problem
    messages: List[str] = field(default_factory=list)


Sessions = {}
closedSessions = {}
Machines = {"machine1": Machine("machine1")}
Users = [User("John", "1234567890", "john@johnsmail")]
Users[0].addPhonePort()


@app.get("/")
def test():
    return {"Hello": "World"}


@app.post("/createProblemSession")
def createProblemSession(username: str, problem: Problem, machineName: str, authMethod: str, arg) -> dict[str, str]:
    for user in Users:
        if user.name == username:
            if user.authenticate(authMethod, arg):
                session_id = str(uuid.uuid4())
                Sessions[session_id] = Session(session_id, username, Machines[machineName], problem)
                return {"session_id": session_id}
    return {"error": "Invalid credentials"}


@app.get("/getUserSessions")
def getUserSessions(username: str, authMethod: str, arg) -> dict[str, list[Session]] | dict[str, str]:
    for user in Users:
        if user.name == username:
            if user.authenticate(authMethod, arg):
                return {"sessions": [session for session in Sessions.values() if session.username == username]}
    return {"error": "Invalid credentials"}

@app.post("/closeSession")
def closeSession(session_id: str) -> dict[str, str]:
    if session_id in Sessions:
        closedSessions[session_id] = Sessions.pop(session_id)
        return {"message": "Closed"}
    return {"error": "Invalid session id"}


@app.post("/sendMessage")
def sendMessage(session_id: str, message: str) -> dict[str, str]:
    if session_id in Sessions:
        Sessions[session_id].messages.append(message)
        return {"message": "Sent"}
    return {"error": "Invalid session id"}


@app.get("/getMessages")
def getMessages(session_id: str) -> dict[str, Session] | dict[str, str]:
    if session_id in Sessions:
        return {"messages": Sessions[session_id].messages}
    return {"error": "Invalid session id"}


@app.get("/getSession")
def getSession(session_id: str) -> dict[str, Session] | dict[str, str]:
    if session_id in Sessions:
        session = Sessions[session_id]
        return session
    return {"error": "Invalid session id"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
