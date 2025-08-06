
import express from 'express';
import checkBearerToken from '../middlewares/check-bearer-token';
import errorHandler from '../middlewares/error-handler';
import { inviteAccount, verifyInvitation, getAccounts,updateProfile, deleteAccount } from '../controllers/ManageOrganization/collaboratorController';
import { updateOrganizationSettings, submitFeedback,deleteOrganization,getOrganizationSettings } from '../controllers/ManageOrganization/Settings-FeedbackController';

// initialize router
const router = express.Router();


// Account invitation routes
router.post('/invite-account', [checkBearerToken], inviteAccount, errorHandler);
router.get('/verify-invitation/:token', verifyInvitation);
router.get('/accounts', [checkBearerToken], getAccounts, errorHandler);
router.post('/update-profile', [checkBearerToken], updateProfile, errorHandler);


router.delete('/accounts/:id', [checkBearerToken], deleteAccount, errorHandler);




// Settings routes
router.put('/organization', [checkBearerToken], updateOrganizationSettings, errorHandler);

// Feedback routes
router.post('/feedback', [checkBearerToken], submitFeedback, errorHandler);
router.get('/organization', [checkBearerToken], getOrganizationSettings, errorHandler); // Nouvelle route GET
router.delete('/organization', [checkBearerToken], deleteOrganization, errorHandler);

export default router;
