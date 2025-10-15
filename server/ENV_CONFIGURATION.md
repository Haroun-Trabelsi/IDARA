# Environment Configuration Guide

This document describes all environment variables needed for the IDara server.

## Quick Setup for Demo Mode

Create a `.env` file in the server directory with these values:

```env
# Server Configuration
PORT=5000
NODE_ENV=development

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/idara

# JWT Configuration
JWT_SECRET=demo_jwt_secret_key_for_development_only_12345
JWT_EXPIRES_IN=7d

# Email Configuration
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_SECURE=false
EMAIL_USER=demo@idara.com
EMAIL_PASSWORD=demo_password
EMAIL_FROM=noreply@idara.com

# Frontend URL
FRONTEND_URL=http://localhost:3000

# Demo Mode
DEMO_MODE=true
```

## Complete Configuration Options

### Server Configuration
- **PORT**: Server port (default: 5000)
- **NODE_ENV**: Environment (development/production)

### MongoDB Configuration
- **MONGODB_URI**: MongoDB connection string
  - Local: `mongodb://localhost:27017/idara`
  - Atlas: `mongodb+srv://username:password@cluster.mongodb.net/database`

### JWT Configuration
- **JWT_SECRET**: Secret key for JWT token generation
- **JWT_EXPIRES_IN**: Token expiration time (e.g., 7d, 24h)

### Email Configuration
- **EMAIL_HOST**: SMTP server host
- **EMAIL_PORT**: SMTP server port
- **EMAIL_SECURE**: Use SSL/TLS (true/false)
- **EMAIL_USER**: SMTP username
- **EMAIL_PASSWORD**: SMTP password
- **EMAIL_FROM**: Sender email address

### Frontend Configuration
- **FRONTEND_URL**: Frontend application URL for CORS

### ftrack Integration (Optional)
- **FTRACK_SERVER_URL**: Your ftrack server URL
- **FTRACK_API_USER**: ftrack API username
- **FTRACK_API_KEY**: ftrack API key

### AWS S3 (Optional)
- **AWS_ACCESS_KEY_ID**: AWS access key
- **AWS_SECRET_ACCESS_KEY**: AWS secret key
- **AWS_REGION**: AWS region
- **AWS_S3_BUCKET**: S3 bucket name

### Security
- **ENCRYPTION_KEY**: 32-character encryption key
- **SESSION_SECRET**: Session secret key

### Rate Limiting
- **RATE_LIMIT_WINDOW_MS**: Time window in milliseconds
- **RATE_LIMIT_MAX_REQUESTS**: Max requests per window

### File Upload
- **MAX_FILE_SIZE**: Maximum file size in bytes (default: 2GB)

### Demo Mode
- **DEMO_MODE**: Enable demo mode with mock data (true/false)

## Demo Credentials

When using seeded data:
- **Email**: john.doe@vfxstudio.com
- **Password**: Demo123!

## Seeding the Database

To populate the database with demo data:

```bash
cd server
npm run seed
```

This will create sample:
- User accounts (admin and regular users)
- VFX shot analysis results
- Contact messages
- Feedback entries

