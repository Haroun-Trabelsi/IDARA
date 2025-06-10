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
- Session management
- Basic website template
- API key placeholder (to be populated)
- Ftrack URL placeholder (to be populated)

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
