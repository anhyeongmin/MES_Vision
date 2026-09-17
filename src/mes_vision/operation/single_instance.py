"""Focus an existing desktop instance without opening another device owner."""
import hashlib
import os
from pathlib import Path
from PySide6.QtCore import QObject
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication


def server_name(runtime):
    path=os.path.normcase(str(Path(runtime).resolve()))
    return 'mes-vision-'+hashlib.sha256(path.encode('utf-8')).hexdigest()[:32]


def focus_existing(runtime,timeout_ms=1500):
    socket=QLocalSocket()
    socket.connectToServer(server_name(runtime))
    if not socket.waitForConnected(timeout_ms): return False
    socket.write(b'FOCUS\n')
    sent=socket.bytesToWrite()==0 or socket.waitForBytesWritten(timeout_ms)
    socket.disconnectFromServer()
    return sent


class DesktopActivation(QObject):
    """Create only while holding desktop.lock. Messages can only raise the UI."""
    def __init__(self,runtime,window):
        super().__init__(window); self.window=window; self.clients=set()
        name=server_name(runtime); QLocalServer.removeServer(name)
        self.server=QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.UserAccessOption)
        if not self.server.listen(name): raise RuntimeError(self.server.errorString())
        self.server.newConnection.connect(self.accept)

    def accept(self):
        while self.server.hasPendingConnections():
            client=self.server.nextPendingConnection(); self.clients.add(client)
            client.readyRead.connect(lambda c=client:self.read(c))
            client.disconnected.connect(lambda c=client:self.release(c))
            if client.bytesAvailable(): self.read(client)

    def read(self,client):
        if client.bytesAvailable()>64:
            client.abort(); return
        if not client.canReadLine(): return
        if bytes(client.readLine(64))==b'FOCUS\n':
            if self.window.isMinimized(): self.window.showNormal()
            else: self.window.show()
            target=QApplication.activeModalWidget() or self.window
            target.show(); target.raise_(); target.activateWindow()
        client.disconnectFromServer()

    def release(self,client):
        self.clients.discard(client); client.deleteLater()
