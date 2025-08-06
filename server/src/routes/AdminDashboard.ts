import express from 'express';
import { getAdminDashboardData, sendEmail } from '../controllers/admin/AdminDashboard';
import checkBearerToken from '../middlewares/check-bearer-token';
import errorHandler from '../middlewares/error-handler';

const router = express.Router();

// Récupérer les données du tableau de bord
router.get('/dashboard', [checkBearerToken], getAdminDashboardData, errorHandler);

// Envoyer un email
router.post('/send-email', [checkBearerToken], sendEmail, errorHandler);

export default router;