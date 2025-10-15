# Client Environment Configuration Guide

This document describes environment variables needed for the IDara client application.

## Quick Setup

Create a `.env` file in the client directory:

```env
# Backend API URL
REACT_APP_BACKEND_URL=http://localhost:5000

# Enable mock data mode (when backend is unavailable)
REACT_APP_USE_MOCK_DATA=true

# Application Environment
REACT_APP_ENV=development

# ftrack Integration (optional)
REACT_APP_FTRACK_URL=https://your-company.ftrackapp.com

# Feature Flags
REACT_APP_ENABLE_MFA=true
REACT_APP_ENABLE_FEEDBACK=true
REACT_APP_ENABLE_ANALYTICS=false
```

## Configuration Options

### API Configuration
- **REACT_APP_BACKEND_URL**: Backend server URL
  - Development: `http://localhost:5000`
  - Production: Your production API URL

### Mock Data Mode
- **REACT_APP_USE_MOCK_DATA**: Enable mock data when API is unavailable
  - Set to `true` for demo mode
  - Set to `false` to always use real API

### Application Settings
- **REACT_APP_ENV**: Application environment
  - Options: development, staging, production

### Integration URLs
- **REACT_APP_FTRACK_URL**: Your ftrack server URL (if using ftrack)

### Feature Flags
- **REACT_APP_ENABLE_MFA**: Enable multi-factor authentication
- **REACT_APP_ENABLE_FEEDBACK**: Enable user feedback features
- **REACT_APP_ENABLE_ANALYTICS**: Enable analytics tracking

## Mock Data System

The application includes a comprehensive mock data system that automatically activates when:
1. The backend API is unavailable
2. `REACT_APP_USE_MOCK_DATA` is set to `true`
3. API calls fail (automatic fallback)

### Mock Data Includes:
- **User Accounts**: Admin and regular user profiles
- **Organizations**: Sample VFX studios with team members
- **Projects**: Movie/show projects with sequences and shots
- **VFX Analysis Results**: Complexity scores and predictions
- **Contact Messages**: Sample support inquiries
- **Feedback**: User ratings and feature suggestions

### Demo Login Credentials

```
Email: john.doe@vfxstudio.com
Password: Demo123!
```

### Using Mock Data

The mock data system is transparent - it automatically:
1. Intercepts failed API calls
2. Returns appropriate mock data
3. Simulates realistic API delays
4. Logs when mock data is being used

To force mock data mode:
```javascript
import MockApiService from 'utils/mockApiService';

MockApiService.enableMockMode();
```

To check if mock mode is active:
```javascript
if (MockApiService.isMockMode()) {
  console.log('Running in mock data mode');
}
```

## Building for Production

When building for production without a backend:

```bash
# Set mock data mode
echo "REACT_APP_USE_MOCK_DATA=true" > .env.production

# Build the application
npm run build
```

The built application will work standalone with mock data, perfect for:
- Demonstrations
- Portfolio showcases
- Client presentations
- Offline usage

## Available Mock Data

### Accounts (5 users)
- 2 Admin accounts (different organizations)
- 3 Regular users
- Mix of verified and pending invitations

### Projects (3 projects)
- "Galactic Warriors" - Sci-fi action (145 shots)
- "Ocean's Mystery" - Underwater fantasy (89 shots)
- "City of Tomorrow" - Futuristic cityscape (210 shots)

### VFX Results (5 analysis results)
- Easy, Medium, and Hard complexity classifications
- Complete complexity score breakdowns
- Predicted VFX hours
- Processing time metrics

### Organizations
- "VFX Studio Alpha" - North America (3 members)
- "Creative VFX House" - Asia Pacific (1 member)

## Troubleshooting

### Mock data not loading
1. Check browser console for errors
2. Verify `REACT_APP_USE_MOCK_DATA=true` in .env
3. Clear browser cache and localStorage
4. Restart development server

### API calls still failing
1. Check REACT_APP_BACKEND_URL is correct
2. Ensure backend server is running (if not using mock mode)
3. Check network tab in browser dev tools
4. Verify CORS settings on backend

### Authentication issues
1. Clear localStorage: `localStorage.clear()`
2. Use demo credentials provided above
3. Check that mock mode is enabled
4. Verify token is being stored correctly

