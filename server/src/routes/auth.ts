// routes/auth.ts
import express, { Request, Response, NextFunction, RequestHandler } from 'express';
import checkBearerToken from '../middlewares/check-bearer-token';
import errorHandler from '../middlewares/error-handler';
import {register ,updateAccount } from '../controllers/auth/register';
import login from '../controllers/auth/login';
import loginWithToken from '../controllers/auth/login-with-token';
import { verify, resendVerification } from '../controllers/auth/verify';
import { setupMFA, verifyMFA } from '../controllers/auth/mfaController';
import forgotPassword from '../controllers/auth/forgotPassword';
import { editProfile, changePassword } from '../controllers/auth/editProfile';
import jwt from '../utils/jwt';
import resetPassword from '../controllers/auth/resetPassword';

// initialize router
const router = express.Router();

// POST at route: http://localhost:8080/auth/register
router.post('/register', [], register, errorHandler);
router.put('/update-account', updateAccount,errorHandler);
// POST at path: http://localhost:8080/auth/login
router.post('/login', [], login, errorHandler);

// GET at path: http://localhost:8080/auth/account
router.get('/login', [checkBearerToken], loginWithToken, errorHandler);

router.get('/verify-email/:token', verify); // Pour GET avec token
router.post('/verify-email', verify);     // Pour POST avec code et email
router.post('/resend-verification', resendVerification);

router.post('/mfa/setup', [checkBearerToken], setupMFA, errorHandler); // Requiert un token pour sécurité

// POST at path: http://localhost:8080/auth/mfa/verify
router.post('/mfa/verify', [checkBearerToken], verifyMFA, errorHandler); // Requiert un token pour sécurité

// PUT at path: http://localhost:8080/auth/edit-profile
router.put('/edit-profile', [checkBearerToken], editProfile, errorHandler);

// PUT at path: http://localhost:8080/auth/edit-profile/password
router.put('/edit-profile/password', [checkBearerToken], changePassword, errorHandler);

router.post('/forgot-password', [], forgotPassword, errorHandler);
router.post('/reset-password', [], resetPassword, errorHandler);

// Nouvelle route GET pour valider le token et rediriger
const resetPasswordHandler: RequestHandler<{ token: string }> = (req, res, next) => {
  try {
    const { token } = req.params;
    // Valider le token
    const decoded = jwt.verifyToken(token);
    if (decoded === null || typeof decoded === 'string' || !('uid' in decoded)) {
      res.status(400).json({ message: 'Invalid or expired reset token' });
      return; // Arrêter l'exécution après la réponse
    }
    // Rediriger vers le frontend avec le token
    res.status(302).set('Location', `${process.env.REACT_APP_FRONTEND_URL || 'http://localhost:3000'}/reset-password/${token}`).end();
    return; // Arrêter l'exécution après la redirection
  } catch (error) {
    next(error);
  }
};

router.get('/reset-password/:token', resetPasswordHandler);

export default router;