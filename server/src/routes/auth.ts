import express from 'express'
import checkBearerToken from '../middlewares/check-bearer-token'
import errorHandler from '../middlewares/error-handler'
import register from '../controllers/auth/register'
import login from '../controllers/auth/login'
import loginWithToken from '../controllers/auth/login-with-token'
import verify from '../controllers/auth/verify'
import { setupMFA, verifyMFA } from '../controllers/auth/mfaController'
// initialize router
const router = express.Router()

// POST at route: http://localhost:8080/auth/register
router.post('/register', [], register, errorHandler)

// POST at path: http://localhost:8080/auth/login
router.post('/login', [], login, errorHandler)

// GET at path: http://localhost:8080/auth/account
router.get('/login', [checkBearerToken], loginWithToken, errorHandler)


router.get('/verify/:token', verify)

router.post('/verify', verify) 

router.post('/mfa/setup', [checkBearerToken], setupMFA, errorHandler) // Requiert un token pour sécurité

// POST at path: http://localhost:8080/auth/mfa/verify
router.post('/mfa/verify', [checkBearerToken], verifyMFA, errorHandler) // Requiert un token pour sécurité

export default router
