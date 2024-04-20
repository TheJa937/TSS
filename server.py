import os
import uuid
from typing import Dict, List, Any
from openai import OpenAI
from enum import Enum

from fastapi import FastAPI
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv("./env.env")

apiKey = os.getenv("OPENAI_API_KEY")
app = FastAPI()
client = OpenAI(
    api_key=apiKey
)

with open("prompt", "r") as f:
    prompt = f.read()


class MachineStatus(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


def generate_response(messages: List[Dict[str, str]]) -> str:
    """
    Generate AI response based on the conversation history.

    Args:
        messages (List[Dict[str, str]]): List of messages in the conversation history.

    Returns:
        str: Generated response from AI model.
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return completion.choices[0].message.content


@dataclass
class Userport:
    """
    Represents a user port. A user port is a way a user can be authenticated to the system.

    Attributes:
        name (str): Name of the port.
        authentication (Function): Authentication function for the port.
    """
    name: str
    authentication: "Function"


@dataclass
class User:
    """
    Represents a user. A user can have multiple ports for authentication.

    Attributes:
        name (str): Name of the user.
        phoneNumber (str): Phone number of the user.
        email (str): Email of the user.
        Userports (list[Userport]): List of user ports.
        currentSessionId (str): Current session ID of the user.
    """
    name: str
    phoneNumber: str
    email: str
    Userports: list[Userport] = field(default_factory=list)
    currentSessionId: str = ""

    def addPhonePort(self):
        """
        Add phone port for the user.
        """
        self.Userports.append(Userport("Phone", lambda number: self.phoneNumber == number))
        Users.append(self)

    def authenticate(self, method, arg):
        """
        Authenticate the user. using the method declared by method

        Args:
            method: Authentication method.
            arg: Argument for authentication.

        Returns:
            bool: True if authentication is successful, False otherwise.
        """
        for port in self.Userports:
            if port.name == method:
                return port.authentication(arg)
        return False


@dataclass
class Session:
    """
    Represents a session. So a problem of a user on a certain machine

    Attributes:
        user (User): User associated with the session.
        id (str): Session ID.
        machine (Machine): Machine associated with the session.
        problem (Problem): Problem associated with the session.
        messages (List[dict[str, str]]): List of messages in the session.
    """
    user: User
    id: str
    machine: "Machine"
    problem: "Problem"
    messages: List[dict[str, str]] = field(default_factory=list)


Sessions = {}
closedSessions = {}
Machines = {"machine1": Machine("machine1")}
Users = [User("John", "1234567890", "john@johnsmail")]
Users[0].addPhonePort()
Sessions["123"] = Session("123", "John", Machines["machine1"], Problem("Problem1", "This is a problem"))
Sessions["123"].messages.append({"role": "system", "content": prompt})


def getUsersCurrentSession(username: str):
    """
    Get the current session of the user.

    Args:
        username (str): Username.

    Returns:
        str: Session ID of the user.
    """
    for user in Users:
        if user.name == username:
            return user.currentSessionId
    return None


@dataclass
class AiFunction:
    """
    Represents an AI function.

    Attributes:
        name (str): Name of the AI function.
        function (Function): Function associated with the AI function.
    """
    name: str
    function: "Function"


aiFunctions = []


def listSessionStates(_):
    """
    List session states.

    Args:
        _ : Unused argument.

    Returns:
        dict[str, str]: Session states.
    """
    return {session.machine.name: session.machine.status.name for session in Sessions.values()}
aiFunctions.append(AiFunction("listSessionStates", listSessionStates))


def parseAiFunction(call: str):
    """
    Parse AI function call.

    Args:
        call (str): AI function call.

    Returns:
        Any: Result of the AI function call.
    """
    try:
        functionName, arg = call.split("(")
        arg = arg[:-1]
        for function in aiFunctions:
            if function.name == functionName:
                return function.function(arg)
    except:
        return call


@app.get("/")
def test():
    """
    Test API endpoint.
    """
    return {"Hello": "World"}


@app.post("/setMachineStatus")
def setMachineStatus(session_id: str, status: str) -> dict[str, str]:
    """
    Set machine status.

    Args:
        session_id (str): Session ID.
        status (str): Status.

    Returns:
        dict[str, str]: Response message.
    """
    if session_id in Sessions:
        session = Sessions[session_id]
        if status not in MachineStatus.__members__:
            return {"error": "Invalid status"}
        session.machine.status = MachineStatus[status]
        return {"message": "Status updated"}
    return {"error": "Invalid Session ID"}


@app.post("/createProblemSession")
def createProblemSession(username: str, problem: "Problem", machineName: str, authMethod: str, arg) -> dict[str, str]:
    """
    Create a problem session.

    Args:
        username (str): Username.
        problem (Problem): Problem details.
        machineName (str): Machine name.
        authMethod (str): Authentication method.
        arg : Authentication argument.

    Returns:
        dict[str, str]: Response message.
    """
    for user in Users:
        if user.name == username:
            if user.authenticate(authMethod, arg):
                session_id = str(uuid.uuid4())
                Sessions[session_id] = Session(session_id, username, Machines[machineName], problem)
                Sessions[session_id].messages.append({"role": "system", "content": prompt})
                user.currentSessionId = session_id
                return {"session_id": session_id}
    return {"error": "Invalid credentials"}


@app.get("/getUserSessions")
def getUserSessions(username: str, authMethod: str, arg) -> dict[str, list[Session]] | dict[str, str]:
    """
    Get user sessions.

    Args:
        username (str): Username.
        authMethod (str): Authentication method.
        arg : Authentication argument.

    Returns:
        dict[str, list[Session]] | dict[str, str]: User sessions or error message.
    """
    for user in Users:
        if user.name == username:
            if user.authenticate(authMethod, arg):
                return {"sessions": [session for session in Sessions.values() if session.username == username]}
    return {"error": "Invalid credentials"}


@app.post("/closeSession")
def closeSession(session_id: str) -> dict[str, str]:
    """
    Close a session.

    Args:
        session_id (str): Session ID.

    Returns:
        dict[str, str]: Response message.
    """
    if session_id in Sessions:
        closedSessions[session_id] = Sessions.pop(session_id)
        return {"message": "Closed"}
    return {"error": "Invalid session id"}


@app.post("/sendMessage")
def sendMessage(username: str, message: str) -> dict[str, str] | dict[str, dict]:
    """
    Send a message.

    Args:
        username (str): Username.
        message (str): Message content.

    Returns:
        dict[str, str] | dict[str, dict]: Response message or AI generated response.
    """
    sessionId = getUsersCurrentSession(username)
    if sessionId in Sessions:
        session = Sessions[sessionId]
        session.messages.append({"role": "user", "content": message})
        response = parseAiFunction(generate_response(session.messages))
        return {"message": response}
    return {"error": "Invalid session id"}


@app.get("/getMessages")
def getMessages(username: str) -> dict[str, Session] | dict[str, str]:
    """
    Get messages.

    Args:
        username (str): Username.

    Returns:
        dict[str, Session] | dict[str, str]: Messages or error message.
    """
    session_id = getUsersCurrentSession(username)
    if session_id in Sessions:
        return {"messages": Sessions[session_id].messages}
    return {"error": "Invalid session id"}


@app.get("/getCurrentSession")
def getCurrentSession(username: str) -> Session:
    """
    Get current session.

    Args:
        username (str): Username.

    Returns:
        Session: Current session.
    """
    session_id = getUsersCurrentSession(username)
    if session_id in Sessions:
        session = Sessions[session_id]
        return session
    return {"error": "Invalid session id"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
