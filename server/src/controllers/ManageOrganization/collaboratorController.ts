import { type RequestHandler } from 'express';
import { sendInvitationEmail } from '../../utils/nodemailer';
import Account from '../../models/Account';
import crypt from '../../utils/crypt'; // Importer crypt pour hacher le mot de passe
import crypto from 'crypto';

// Options valides pour region et teamSize (doivent correspondre au frontend)
const teamSizeOptions = ['1', '2-10', '11-20', '21-50', '51-100', '101-200', '201-500', '500+'];
const regionOptions = ['Europe', 'US East', 'US West', 'South America', 'East Asia', 'Australia', 'Singapore', 'China'];

export const inviteAccount: RequestHandler = async (req, res, next) => {
  try {
    const { email, language = 'en', region, teamSize } = req.body;
    console.log('Request received for /col/invite-account:', { email, language, region, teamSize }); // Log 1: Requête reçue

    if (!req.auth?.uid) {
      console.log('Failure: req.auth.uid missing'); // Log 2: UID manquant
      return next({
        statusCode: 401,
        message: 'Authentication required.',
      });
    }

    const invitingAccount = await Account.findById(req.auth.uid);
    console.log('User found in database:', invitingAccount); // Log 3: Utilisateur trouvé
    if (!invitingAccount) {
      console.log('Failure: user not found for ID:', req.auth.uid); // Log 4: Utilisateur non trouvé
      return next({
        statusCode: 401,
        message: 'User not found.',
      });
    }

    if (!invitingAccount.canInvite) {
      console.log('Failure: user does not have canInvite permission'); // Log 5: Pas de permission
      return next({
        statusCode: 403,
        message: 'You do not have permission to invite collaborators.',
      });
    }

    if (!invitingAccount.ID_Organization) {
      console.log('Failure: inviting account has no ID_Organization'); // Log 6: Pas d'ID_Organization
      return next({
        statusCode: 400,
        message: 'Cannot invite: organization ID is missing for the inviting account.',
      });
    }

    const organizationName = invitingAccount.organizationName || 'DefaultOrg';
    console.log('OrganizationName:', organizationName); // Log 7: Nom d'organisation

    const totalMembers = await Account.countDocuments({ ID_Organization: invitingAccount.ID_Organization });
    console.log('Total members in organization:', totalMembers); // Log 8: Nombre de membres
    if (totalMembers >= 2) {
      console.log('Failure: Organization already has the maximum number of members (2)'); // Log 9: Limite atteinte
      return next({
        statusCode: 403,
        message: 'This organization has reached the maximum number of members (2).',
      });
    }

    const existingAccount = await Account.findOne({ email });
    if (existingAccount) {
      console.log('Failure: account already exists with email:', email); // Log 10: Email existant
      return next({
        statusCode: 400,
        message: 'An account with this email already exists.',
      });
    }

    // Valider region et teamSize
    const selectedRegion = region && regionOptions.includes(region) ? region : invitingAccount.region;
    const selectedTeamSize = teamSize && teamSizeOptions.includes(teamSize) ? teamSize : invitingAccount.teamSize;
    if (!selectedRegion || !selectedTeamSize) {
      console.log('Failure: invalid region or teamSize', { region, teamSize, selectedRegion, selectedTeamSize }); // Log 11: Validation échouée
      return next({
        statusCode: 400,
        message: 'Invalid region or teamSize provided.',
      });
    }

    const verificationToken = crypto.randomBytes(32).toString('hex');
    const verificationTokenExpires = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 heures
    const tempPassword = crypto.randomBytes(8).toString('hex');
    const hashedPassword = await crypt.hash(tempPassword); // Hacher le mot de passe
    console.log('Verification token generated:', verificationToken); // Log 12: Token généré
    console.log('Temporary password (hashed):', hashedPassword); // Log 13: Mot de passe haché

    const newAccount = new Account({
      name: "Guest",
      email,
      password: hashedPassword, // Utiliser le mot de passe haché
      role: 'user',
      isVerified: false,
      organizationName,
      invitedBy: req.auth.uid,
      region: selectedRegion,
      teamSize: selectedTeamSize,
      ID_Organization: invitingAccount.ID_Organization, // Hériter de l'ID_Organization
      canInvite: false,
      status: 'pending',
      invitedDate: new Date(),
      verificationToken,
      verificationTokenExpires,
      mustCompleteProfile: true,
    });
    await newAccount.save();
    console.log('New account created:', newAccount._id); // Log 14: Compte créé

    const verificationLink = `${process.env.FRONTEND_URL}/verify-invite/${verificationToken}`;
    console.log('Verification link:', verificationLink); // Log 15: Lien de vérification

    await sendInvitationEmail(email, verificationToken, tempPassword, organizationName);
    console.log('Invitation email sent to:', email); // Log 16: Email envoyé

    res.status(201).json({
      message: 'Invitation sent successfully.',
      data: { email, status: 'pending' },
    });
    console.log('Response 201 sent'); // Log 17: Réponse envoyée
  } catch (error: any) {
    console.error('Error in inviteAccount:', error.message, error.stack); // Log 18: Erreur globale
    next({
      statusCode: 500,
      message: 'Server error during invitation',
      error: error.message,
    });
  }
};

export const verifyInvitation: RequestHandler = async (req, res, next) => {
  try {
    const { token } = req.params;
    console.log('Request received for /col/verify-invite:', { token }); // Log 19: Requête reçue

    const account = await Account.findOne({
      verificationToken: token,
      verificationTokenExpires: { $gt: new Date() },
    });

    if (!account) {
      console.log('Failure: invalid or expired token'); // Log 20: Token invalide
      return res.redirect(303, `${process.env.FRONTEND_URL}/error`);
    }

    account.status = 'accepted';
    account.isVerified = true;
    account.verificationToken = undefined;
    account.verificationTokenExpires = undefined;
    await account.save();
    console.log('Account verified and updated:', account._id); // Log 21: Compte vérifié

    res.redirect(303, `${process.env.FRONTEND_URL}/login`);
    console.log('Redirected to login'); // Log 22: Redirection
  } catch (error: any) {
    console.error('Error in verifyInvitation:', error.message, error.stack); // Log 23: Erreur globale
    next({
      statusCode: 500,
      message: 'Server error during invitation verification',
      error: error.message,
    });
  }
};

export const getAccounts: RequestHandler = async (req, res, next) => {
  try {
    console.log('Request received for /col/accounts', req.auth); // Log 24: Requête reçue
    if (!req.auth?.uid) {
      console.log('Failure: req.auth.uid missing'); // Log 25: UID manquant
      return next({
        statusCode: 401,
        message: 'Authentication required.',
      });
    }

    const currentAccount = await Account.findById(req.auth.uid);
    console.log('Current account found:', currentAccount); // Log 26: Compte trouvé
    if (!currentAccount) {
      console.log('Failure: user not found for UID:', req.auth.uid); // Log 27: Compte non trouvé
      return next({
        statusCode: 401,
        message: 'User not found.',
      });
    }

    if (!currentAccount.ID_Organization) {
      console.log('Failure: current account has no ID_Organization'); // Log 28: Pas d'ID_Organization
      return next({
        statusCode: 400,
        message: 'Cannot fetch accounts: organization ID is missing.',
      });
    }

    const accounts = await Account.find({ ID_Organization: currentAccount.ID_Organization });
    console.log('Accounts found for ID_Organization:', accounts.length); // Log 29: Comptes trouvés

    const formattedAccounts = accounts.map(a => ({
      id: a._id.toString(),
      name: a.name,
      surname: a.surname,
      email: a.email,
      role: a.role,
      status: a.status,
      invitedDate: a.invitedDate ? a.invitedDate.toISOString().split('T')[0] : null,
      canInvite: a.canInvite,
      invitedBy: a.invitedBy,
    }));

    res.status(200).json(formattedAccounts);
    console.log('Response 200 sent with accounts'); // Log 30: Réponse envoyée
  } catch (error: any) {
    console.error('Error in getAccounts:', error.message, error.stack); // Log 31: Erreur globale
    next({
      statusCode: 500,
      message: 'Server error during fetching accounts',
      error: error.message,
    });
  }
};

export const deleteAccount: RequestHandler = async (req, res, next) => {
  try {
    const { id } = req.params;
    console.log('Request received for /col/accounts/:id:', { id }); // Log 32: Requête reçue
    console.log('req.auth:', req.auth);

    if (!req.auth?.uid) {
      console.log('Failure: req.auth.uid missing'); // Log 33: UID manquant
      return next({
        statusCode: 401,
        message: 'Authentication required.',
      });
    }

    const deletingAccount = await Account.findById(req.auth.uid);
    console.log('User found in database:', deletingAccount); // Log 34: Utilisateur trouvé
    if (!deletingAccount) {
      console.log('Failure: user not found for UID:', req.auth.uid); // Log 35: Utilisateur non trouvé
      return next({
        statusCode: 401,
        message: 'User not found.',
      });
    }

    if (!deletingAccount.canInvite) {
      console.log('Failure: user does not have canInvite permission'); // Log 36: Pas de permission
      return next({
        statusCode: 403,
        message: 'You do not have permission to delete collaborators.',
      });
    }

    const accountToDelete = await Account.findById(id);
    if (!accountToDelete) {
      console.log('Failure: account to delete not found for ID:', id); // Log 37: Compte non trouvé
      return next({
        statusCode: 404,
        message: 'Account not found.',
      });
    }

    await Account.deleteOne({ _id: id });
    console.log('Account deleted:', id); // Log 38: Compte supprimé

    res.status(200).json({
      message: 'Account deleted successfully.',
    });
    console.log('Response 200 sent'); // Log 39: Réponse envoyée
  } catch (error: any) {
    console.error('Error in deleteAccount:', error.message, error.stack); // Log 40: Erreur globale
    next({
      statusCode: 500,
      message: 'Server error during account deletion',
      error: error.message,
    });
  }
};

export const updateProfile: RequestHandler = async (req, res, next) => {
  try {
    const { name, surname, newPassword, receiveUpdates } = req.body;
    console.log('Request received for /update-profile:', { name, surname, newPassword, receiveUpdates }); // Log 41: Requête reçue
    console.log('req.auth:', req.auth);

    if (!req.auth?.uid) {
      console.log('Failure: req.auth.uid missing'); // Log 42: UID manquant
      return next({
        statusCode: 401,
        message: 'Authentication required.',
      });
    }

    const account = await Account.findById(req.auth.uid);
    console.log('User found in database:', account); // Log 43: Utilisateur trouvé
    if (!account) {
      console.log('Failure: user not found for UID:', req.auth.uid); // Log 44: Utilisateur non trouvé
      return next({
        statusCode: 401,
        message: 'User not found.',
      });
    }

    account.name = name || account.name;
    account.surname = surname || account.surname;
    if (newPassword) {
      account.password = await crypt.hash(newPassword); // Hacher le nouveau mot de passe
      console.log('New password hashed and updated'); // Log 45: Mot de passe haché
    }
    account.receiveUpdates = receiveUpdates !== undefined ? receiveUpdates : account.receiveUpdates;
    account.mustCompleteProfile = false;
    await account.save();
    console.log('Profile updated:', account._id); // Log 46: Profil mis à jour

    res.status(200).json({
      message: 'Profile updated successfully.',
      redirect: 'http://localhost:3000',
    });
    console.log('Response 200 sent'); // Log 47: Réponse envoyée
  } catch (error: any) {
    console.error('Error in updateProfile:', error.message, error.stack); // Log 48: Erreur globale
    next({
      statusCode: 500,
      message: 'Server error during profile update',
      error: error.message,
    });
  }
};