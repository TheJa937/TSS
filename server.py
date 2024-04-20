import os
import uuid
from typing import Dict, List, Any, Callable
from openai import OpenAI
from enum import Enum

from fastapi import FastAPI, UploadFile, File
from dataclasses import dataclass, field
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import askPDF


load_dotenv("./env.env")

openai_apiKey = os.getenv("OPENAI_API_KEY")
app = FastAPI()
client = OpenAI(
    api_key=openai_apiKey
)

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("prompt", "r") as f:
    prompt = f.read()

ADMIN_PASSWORD = "apfel"

class MachineStatus(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

async def generate_response(messages: List[Dict[str, str]]) -> str:
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
    authentication: Callable


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


class SessionState(Enum):
    WaitingForServicetechnician = "Waiting for Servicetechnician"
    AwaitingUserResponse = "Awaiting user response"
    PleaseScheduleService = "Please schedule service"
    Closed = "Closed"

    def __str__(self):
        emojies = {self.WaitingForServicetechnician: "🟡",
                   self.AwaitingUserResponse: "🔵",
                   self.PleaseScheduleService: "🔴"}
        return self.value + emojies.get(self, "")


@dataclass
class Session:
    """
    Represents a session. So a problem of a user on a certain machine

    Attributes:
        username (User): User associated with the session.
        id (str): Session ID.
        machine (Machine): Machine associated with the session.
        problem (Problem): Problem associated with the session.
        messages (List[dict[str, str]]): List of messages in the session.
    """
    username: str
    id: str
    machine: "Machine"
    problem: "Problem"
    state: SessionState = SessionState.WaitingForServicetechnician
    messages: List[dict[str, str]] = field(default_factory=list)
    files: List[str] = field(default_factory=list)


@dataclass()
class Machine:
    name: str
    status: MachineStatus = MachineStatus.GREEN


@dataclass
class Problem:
    name: str
    description: str

NAME = "Henrik"
NUMBER = "4916095848582"
Sessions = {}
closedSessions = {}
Machines = {"machine1": Machine("machine1"), "machine2": Machine("machine2"), "machine3": Machine("machine3")}
Users = [User(NAME, NUMBER, "Henrik@johnsmail")]
Users[0].addPhonePort()


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


def authenticateUser(username: str, method: str, arg: Any):
    """
    Authenticate the user.

    Args:
        username (str): Username.
        method (str): Authentication method.
        arg (Any): Argument for authentication.

    Returns:
        bool: True if authentication is successful, False otherwise.
    """
    for user in Users:
        if user.name == username:
            return user.authenticate(method, arg)
    return False


@dataclass
class AiFunction:
    """
    Represents an AI function.

    Attributes:
        name (str): Name of the AI function.
        function (Function): Function associated with the AI function.
    """
    name: str
    function: Callable


aiFunctions = []


class Specials(Enum):
    Button = "Button"


@dataclass
class Response:
    message: str
    specials: dict[str, str]


def listSessionStates(context, _) -> list[Response]:
    """
    List session states.

    Args:
        _ : Unused argument.

    Returns:
        dict[str, str]: Session states.
    """
    responses = []
    i = 1
    for session in Sessions.values():
        if session.username == context["user"].name:
            response = Response(f"{session.machine.name}: {session.state}", {Specials.Button: True})
            responses.append(response)
            i += 1
    return responses


def parsePDF(_, question) -> str:
    return askPDF.askQuestion(question)


aiFunctions.append(AiFunction("listSessionStates", listSessionStates))
aiFunctions.append(AiFunction("parsePDF", parsePDF))

def parseAiFunction(call: str, context):
    """
    Parse AI function call.

    Args:
        call (str): AI function call.

    Returns:
        Any: Result of the AI function call.
    """
    if len(call.split("(")) != 2:
        return [Response(call, {})]
    functionName, arg = call.split("(")
    arg = arg[:-1]
    for function in aiFunctions:
        if function.name == functionName:
            return function.function(context, arg)


@app.get("/")
async def test():
    """
    Test API endpoint.
    """
    return {"Hello": "World"}


@app.post("/setMachineStatus")
async def setMachineStatus(session_id: str, status: str) -> dict[str, str]:
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
async def createProblemSession(username: str, problem: "Problem", machineName: str, authMethod: str, arg) -> dict[
    str, str]:
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
async def getUserSessions(username: str, authMethod: str, arg) -> dict[str, list[Session]] | dict[str, str]:
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
                for session in Sessions.values():
                    print(session)
                return {"sessions": [session for session in Sessions.values() if session.username == username]}
    return {"error": "Invalid credentials"}


@app.post("/closeSession")
async def closeSession(session_id: str) -> dict[str, str]:
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
async def sendMessage(username: str, message: str, authMethod: str, arg: str) -> dict[str, str] | dict[str, list[Any]]:
    """
    Send a message.

    Args:
        username (str): Username.
        message (str): Message content.

    Returns:
        dict[str, str] | dict[str, dict]: Response message or AI generated response.
    """
    if not authenticateUser(username, authMethod, arg):
        return {"error": "Invalid credentials"}
    sessionId = getUsersCurrentSession(username)
    if sessionId in Sessions:
        session = Sessions[sessionId]
        session.messages.append({"role": "user", "content": message})
        context = {"user": Users[0]}
        response = parseAiFunction(await generate_response(session.messages), context)
        return {"message": response}
    return {"error": "Invalid session id"}


@app.get("/getMessages")
async def getMessages(username: str, authMethod: str, arg) -> dict[str, Session] | dict[str, str]:
    """
    Get messages.

    Args:
        username (str): Username.

    Returns:
        dict[str, Session] | dict[str, str]: Messages or error message.
    """
    if not authenticateUser(username, authMethod, arg):
        return {"error": "Invalid credentials"}
    session_id = getUsersCurrentSession(username)
    if session_id in Sessions:
        return {"messages": Sessions[session_id].messages}
    return {"error": "Invalid session id"}


@app.get("/getCurrentSession")
async def getCurrentSession(username: str, authMethod: str, arg) -> Session | dict[str, str]:
    """
    Get current session.

    Args:
        username (str): Username.

    Returns:
        Session: Current session.
    """
    if not authenticateUser(username, authMethod, arg):
        return {"error": "Invalid credentials"}
    session_id = getUsersCurrentSession(username)
    if session_id in Sessions:
        session = Sessions[session_id]
        return session
    return {"error": "Invalid session id"}

@app.get("/allActiveSessions")
def getAllActiveSessions(adminPassword: str) -> List[Session] | dict[str, str]:
    if adminPassword == ADMIN_PASSWORD:
        return Sessions
    return {"error": "Invalid Admin Password"}


@app.post("/upload/")
async def uploadfile(file: UploadFile, username: str, authMethod: str, arg):
    if not authenticateUser(username, authMethod, arg):
        return {"error": "Invalid credentials"}
    sessionId = getUsersCurrentSession(username)
    if sessionId in Sessions:
        session = Sessions[sessionId]
        try:
            file_path = f"./UploadedFiles/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            session.files.append(file_path)
        except Exception as e:
            return {"message": e.args}
        finally:
            file.file.close()
            return {"message": "File saved successfully"}
    else:
        file.file.close()
        return {"error": "user has no active Session"}


if __name__ == "__main__":
    import uvicorn

    Sessions["1"] = Session(NAME, "1", Machines["machine1"], Problem("Problem1", "This is a problem"))
    Sessions["1"].messages.append({"role": "system", "content": prompt})
    Sessions["2"] = Session(NAME, "2", Machines["machine2"], Problem("Problem2", "This is another problem"))
    Sessions["2"].messages.append({"role": "system", "content": prompt})
    Sessions["2"].state = SessionState.AwaitingUserResponse

    Users[0].currentSessionId = "1"

    uvicorn.run(app, host="0.0.0.0", port=8000)
