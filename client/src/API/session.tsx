import * as ftrack from '@ftrack/api';

// Replace with your real credentials (or better, load from env)
const SERVER_URL = 'https://mycompany.ftrackapp.com';
const USERNAME = 'john.doe@example.com';
const API_KEY = '7434384e-b653-11f1-a84d-f44c22dd25eu';

// Create the session once
const session = new ftrack.Session(SERVER_URL, USERNAME, API_KEY);

export default session;
