import * as ftrack from '@ftrack/api';
import dotenv from 'dotenv';

dotenv.config();

export async function getFtrackSession(SERVER_URL?: string, USERNAME?: string, API_KEY?: string): Promise<ftrack.Session> {

    if (!SERVER_URL || !USERNAME || !API_KEY) {
        throw new Error("Missing ftrack credentials in .env file");
    }

    // Add protocol if missing
    const fullServerUrl = SERVER_URL.startsWith('http') 
        ? SERVER_URL 
        : `https://${SERVER_URL}`;

    // Create session with explicit authentication
    const session = new ftrack.Session(
        fullServerUrl,
        USERNAME,
        API_KEY
    );

    try {
        // Test connection with a simple request
        console.log(`✓ Connected to ftrack API ${fullServerUrl}`);
        return session;
    } catch (error) {
        console.error('❌ Connection test failed:');
        console.error('Server URL:', fullServerUrl);
        console.error('Username:', USERNAME);
        throw new Error('Failed to authenticate with ftrack API');
    }
}