
import express from 'express';
import checkBearerToken from '../middlewares/check-bearer-token';
import errorHandler from '../middlewares/error-handler';
import { inviteAccount, verifyInvitation, getAccounts,updateProfile, deleteAccount } from '../controllers/HeaderComponent/collaboratorController';

// initialize router
const router = express.Router();


// Account invitation routes
router.post('/invite-account', [checkBearerToken], inviteAccount, errorHandler);
router.get('/verify-invitation/:token', verifyInvitation);
router.get('/accounts', [checkBearerToken], getAccounts, errorHandler);
router.post('/update-profile', [checkBearerToken], updateProfile, errorHandler);


router.delete('/accounts/:id', [checkBearerToken], deleteAccount, errorHandler);
export default router;
