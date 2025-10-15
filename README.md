# IDara - Development Phase Setup

This document outlines the setup and running instructions for the IDara MERN stack application during development.

## Prerequisites

- Node.js (v14 or later)
- npm (v7 or later)
- MongoDB Atlas account (or local MongoDB instance)
- Git (optional)

## Tech Stack

- **MongoDB** (Database)
- **Express.js** (Backend framework)
- **React.js** (Frontend library)
- **Node.js** (JavaScript runtime)
- **Tailwind CSS** (Styling framework)

## Project Structure

```
idara/
├── client/    # Frontend (React)
├── server/    # Backend (Express/Node)
```

## Setup Instructions

### 1. Clone the Repository (if applicable)

```bash
git clone <repository-url>
cd idara
```

### 2. Install Dependencies for Both Client and Server

```bash
cd client
npm install

cd ../server
npm install
```

### 3. Environment Setup

## Running the Application

### Frontend (Client)

Navigate to the `client` folder:

```bash
cd client
```

Run Tailwind CSS compiler in watch mode:

```bash
npx tailwindcss -i ./input.css -o ./output.css --watch
```

In a separate terminal (while still in the `client` folder), start the React development server:

```bash
npm run dev
```

### Backend (Server)

Navigate to the `server` folder:

```bash
cd server
```

Start the Express server:

```bash
npm run dev
```

## Accessing the Application

- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://localhost:5000](http://localhost:5000)

## Database Connection Issues

If you encounter connection issues to MongoDB Atlas and the console suggests whitelisting your IP:

- Send your MongoDB account email to the project maintainer to get added with access  

## Features Implemented

- User authentication (registration/login)
- Multi-factor authentication (MFA)
- Organization and team management
- VFX shot complexity analysis
- Project and sequence tracking
- Ftrack integration support
- Session management
- Contact form and feedback system
- Admin dashboard
- **Mock Data System** for demonstrations

## 🎭 Demo Mode / Mock Data System

IDara includes a comprehensive mock data system that allows the application to run standalone without backend or database access. Perfect for:
- **Demonstrations** and presentations
- **Portfolio showcases**
- **Development** without backend setup
- **Offline usage**

### Quick Demo Setup

```bash
cd client
echo "REACT_APP_USE_MOCK_DATA=true" > .env
npm start
```

**Demo Login:**
- Email: `john.doe@vfxstudio.com`
- Password: `Demo123!`

### With Database Seeding

```bash
# Setup and seed database
cd server
echo "MONGODB_URI=mongodb://localhost:27017/idara" > .env
npm run seed

# Run the application
npm run dev
```

📖 **Full Documentation:** See [MOCK_DATA_README.md](./MOCK_DATA_README.md) for complete details.

### Mock Data Includes:
- ✅ 5 User accounts (admins and users)
- ✅ 3 VFX projects with 400+ shots
- ✅ 5 Complexity analysis results
- ✅ Contact messages and feedback
- ✅ Automatic API fallback

## Troubleshooting

### If frontend doesn't connect to backend:

- Ensure both servers are running
- Ensure API endpoints match between frontend and backend

### For styling issues:

- Ensure Tailwind CSS is compiling
- Verify classes are correctly applied in components

## Next Steps

- Implement API key integration
- Connect Ftrack URL
- Enhance authentication features
- Develop additional application-specific functionality
