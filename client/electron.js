const { app, BrowserWindow } = require('electron');
const isDev = require('electron-is-dev');
const { spawn } = require('child_process');
let mainWindow, serverProcess: { stdout: { pipe: (arg0: NodeJS.WriteStream & { fd: 1; }) => void; }; stderr: { pipe: (arg0: NodeJS.WriteStream & { fd: 2; }) => void; }; kill: () => void; };

function startServer() {
  serverProcess = spawn('node', ['server.js'], { cwd: __dirname });
  serverProcess.stdout.pipe(process.stdout);
  serverProcess.stderr.pipe(process.stderr);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1024,
    height: 768,
    webPreferences: { nodeIntegration: false }
  });
  const startUrl = isDev
    ? 'http://localhost:3000/'
    : `file://${__dirname}/client/build/index.html`;
  mainWindow.loadURL(startUrl);
}

app.whenReady().then(() => {
  startServer();
  createWindow();
});

app.on('window-all-closed', () => {
  if (serverProcess) serverProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});