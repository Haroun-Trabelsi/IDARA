# Mock Data System - Demo Mode Documentation

This document explains the mock data system implemented in IDara to showcase the application's functionality when API and database access is unavailable.

## 🎯 Purpose

The mock data system provides:
- **Standalone demonstration** of the application without backend dependencies
- **Automatic fallback** when API calls fail
- **Realistic data** for showcasing features
- **Portfolio/presentation** capability
- **Development** without constant backend connectivity

## 📦 What's Included

### Frontend Mock Data (`client/src/utils/mockData.ts`)
Complete dummy data for all application features:
- ✅ User accounts (5 users, 2 organizations)
- ✅ VFX projects (3 projects with sequences)
- ✅ Complexity analysis results (5 shot analyses)
- ✅ Contact messages (3 inquiries)
- ✅ User feedback (3 reviews)
- ✅ Authentication tokens

### Backend Seed Data (`server/src/utils/seedData.ts`)
MongoDB-ready seed data matching the frontend:
- ✅ Account documents with proper schemas
- ✅ Result documents with complexity scores
- ✅ ContactMessage documents
- ✅ Feedback documents linked to accounts

### Mock API Service (`client/src/utils/mockApiService.ts`)
Intelligent API interceptor that:
- ✅ Automatically detects API failures
- ✅ Returns appropriate mock data
- ✅ Simulates realistic delays
- ✅ Supports all major endpoints
- ✅ Logs mock mode status

## 🚀 Quick Start

### Frontend Only (No Backend)

1. **Enable mock data mode:**
```bash
cd client
echo "REACT_APP_USE_MOCK_DATA=true" > .env
```

2. **Start the application:**
```bash
npm start
```

3. **Login with demo credentials:**
```
Email: john.doe@vfxstudio.com
Password: Demo123!
```

The app will run entirely on mock data!

### With Backend (Using Seed Data)

1. **Configure the server:**
```bash
cd server
# Create .env with your MongoDB connection
echo "MONGODB_URI=mongodb://localhost:27017/idara" > .env
```

2. **Seed the database:**
```bash
npm run seed
```

3. **Start both servers:**
```bash
# Terminal 1 - Backend
cd server
npm run dev

# Terminal 2 - Frontend
cd client
npm start
```

4. **Login with seeded credentials:**
```
Email: john.doe@vfxstudio.com
Password: Demo123!
```

## 📊 Mock Data Overview

### User Accounts

| Name | Email | Role | Organization | Status |
|------|-------|------|--------------|--------|
| John Doe | john.doe@vfxstudio.com | Admin | VFX Studio Alpha | Active |
| Sarah Johnson | sarah.johnson@vfxstudio.com | User | VFX Studio Alpha | Active |
| Mike Chen | mike.chen@creativevfx.com | Admin | Creative VFX House | Active |
| Emma Williams | emma.williams@vfxstudio.com | User | VFX Studio Alpha | Active |
| Alex Rodriguez | alex.rodriguez@vfxstudio.com | User | VFX Studio Alpha | Pending |

### Projects

| Project | Organization | Shots | Status | Description |
|---------|--------------|-------|--------|-------------|
| Galactic Warriors | VFX Studio Alpha | 145 | Active | Sci-fi action movie |
| Ocean's Mystery | VFX Studio Alpha | 89 | Active | Underwater fantasy |
| City of Tomorrow | Creative VFX House | 210 | Planning | Futuristic cityscape |

### VFX Analysis Results

| Filename | Project | Complexity | Confidence | VFX Hours |
|----------|---------|------------|------------|-----------|
| GW_SQ010_SH0010_v001.mov | Galactic Warriors | Hard | 0.87 | 48.5 |
| GW_SQ010_SH0020_v001.mov | Galactic Warriors | Medium | 0.76 | 24.2 |
| GW_SQ020_SH0010_v002.mov | Galactic Warriors | Hard | 0.92 | 67.8 |
| OM_SQ010_SH0005_v001.mov | Ocean's Mystery | Easy | 0.82 | 8.5 |
| OM_SQ020_SH0012_v001.mov | Ocean's Mystery | Medium | 0.71 | 28.3 |

## 🔧 How It Works

### Automatic Fallback

The axios interceptor automatically catches failed API calls:

```typescript
// client/src/utils/axios.ts
axios.interceptors.response.use(
  (response) => response,
  async (error) => {
    // On API failure, return mock data
    MockApiService.enableMockMode();
    return mockResponse;
  }
);
```

### Manual Control

You can manually control mock mode:

```typescript
import MockApiService from 'utils/mockApiService';

// Enable mock mode
MockApiService.enableMockMode();

// Disable mock mode
MockApiService.disableMockMode();

// Check status
if (MockApiService.isMockMode()) {
  console.log('Using mock data');
}
```

### Supported Endpoints

The mock API service supports:
- ✅ `POST /auth/login` - Authentication
- ✅ `GET /auth/login` - Get current user
- ✅ `POST /auth/register` - Registration
- ✅ `GET /results` - VFX analysis results
- ✅ `GET /projects` - Project list
- ✅ `GET /collaborators` - Organization members
- ✅ `GET /contact` - Contact messages
- ✅ `GET /feedback` - User feedback

## 🎨 Use Cases

### 1. Portfolio Showcase
Build and deploy the frontend alone to showcase your work:
```bash
cd client
REACT_APP_USE_MOCK_DATA=true npm run build
# Deploy the 'build' folder to any static hosting
```

### 2. Client Presentations
Demo the full application without needing database access:
- All features work with realistic data
- No setup required
- Can run offline

### 3. Development
Work on frontend features without running the backend:
- Faster iteration
- No database setup needed
- Consistent test data

### 4. Testing
Test UI/UX with predictable data:
- Known user scenarios
- Edge cases covered
- Repeatable tests

## 📝 Customizing Mock Data

### Adding New Mock Data

Edit `client/src/utils/mockData.ts`:

```typescript
export const mockProjects = [
  ...existingProjects,
  {
    _id: "proj-004",
    name: "Your New Project",
    organizationId: "ORG-001",
    description: "Project description",
    status: "active",
    // ... other fields
  }
];
```

### Adding New Endpoints

Edit `client/src/utils/mockApiService.ts`:

```typescript
private async getMockResponse(config: any): Promise<any> {
  // Add your endpoint
  if (url.includes('/your-new-endpoint')) {
    return await yourMockHandler();
  }
  // ... existing handlers
}
```

### Backend Seed Data

Edit `server/src/utils/seedData.ts`:

```typescript
export const seedYourNewModel = [
  {
    // Your data structure
  }
];

// Add to seedDatabase() function
await YourModel.insertMany(seedYourNewModel);
```

## 🔍 Debugging

### Check Mock Mode Status

Open browser console:
```javascript
// You'll see warnings when mock mode is active
🔄 Mock API Mode Enabled - Using dummy data for demonstration
```

### Verify Mock Data Loading

```javascript
import mockData from 'utils/mockData';

console.log('Mock accounts:', mockData.mockAccounts);
console.log('Mock projects:', mockData.mockProjects);
console.log('Mock results:', mockData.mockResults);
```

### Test Individual Mock API Calls

```javascript
import { mockApiResponses } from 'utils/mockData';

// Test login
const loginResult = await mockApiResponses.login(
  'john.doe@vfxstudio.com',
  'Demo123!'
);
console.log('Login result:', loginResult);

// Test getting results
const results = await mockApiResponses.getResults('Galactic Warriors');
console.log('Results:', results);
```

## ⚙️ Configuration

### Enable/Disable Mock Mode

**Environment Variable:**
```env
# .env file
REACT_APP_USE_MOCK_DATA=true
```

**Code-level:**
```typescript
// client/src/utils/mockApiService.ts
export const useMockData = true; // Change to false
```

### Mock API Delay

Adjust simulation delay in `mockData.ts`:
```typescript
export const simulateDelay = (ms: number = 500) => 
  new Promise(resolve => setTimeout(resolve, ms));
```

## 📚 Files Reference

### Client Files
- `client/src/utils/mockData.ts` - All mock data definitions
- `client/src/utils/mockApiService.ts` - Mock API service logic
- `client/src/utils/axios.ts` - Axios interceptor with fallback
- `client/ENV_CONFIGURATION.md` - Client configuration guide

### Server Files
- `server/src/utils/seedData.ts` - Database seed data
- `server/src/scripts/seed.ts` - Database seeding script
- `server/ENV_CONFIGURATION.md` - Server configuration guide

### Documentation
- `MOCK_DATA_README.md` - This file
- `README.md` - Main project README

## 🎓 Best Practices

1. **Keep mock data synchronized** - Ensure frontend and backend mock data match
2. **Use realistic data** - Make mock data representative of real use cases
3. **Update regularly** - When adding features, add corresponding mock data
4. **Document changes** - Note any mock data additions in commits
5. **Test both modes** - Test with both real API and mock data
6. **Clear indicators** - Make it obvious when mock mode is active

## 🚨 Important Notes

- Mock data passwords are for demonstration only: `Demo123!`
- Mock tokens are not cryptographically secure
- Don't use mock credentials in production
- Mock mode should be disabled for production builds (unless intentional)
- Seed data includes placeholder API keys that won't work with real services

## 🤝 Contributing

When adding new features:
1. Add corresponding mock data to `mockData.ts`
2. Add backend seed data to `seedData.ts`
3. Update mock API service if new endpoints are added
4. Update this documentation
5. Test with both mock and real data

## 📞 Support

If you encounter issues with the mock data system:
1. Check browser console for mock mode warnings
2. Verify environment variables are set correctly
3. Clear localStorage and browser cache
4. Review this documentation
5. Check that mock data matches your schema

---

**Last Updated:** October 2025  
**Version:** 1.0.0

