import { type RequestHandler } from 'express';
import { sendInvitationEmail } from '../../utils/nodemailer';
import Account from '../../models/Account';
import crypto from 'crypto';

export const inviteAccount: RequestHandler = async (req, res, next) => {
  try {
    const { email, language = 'en' } = req.body; // Changé 'fr' en 'en' pour aligner avec l'application en anglais
    console.log('Request received for /col/invite-account:', { email, language });
    console.log('req.auth:', req.auth);

    // Vérifier si l'utilisateur est authentifié
    if (!req.auth?.uid) {
      console.log('Failure: req.auth.uid missing');
      return next({
        statusCode: 401,
        message: 'Authentication required.',
      });
    }

    const invitingAccount = await Account.findById(req.auth.uid);
    console.log('User found in database:', invitingAccount);
    if (!invitingAccount) {
      console.log('Failure: user not found for ID:', req.auth.uid);
      return next({
        statusCode: 401,
        message: 'User not found.',
      });
    }

    if (!invitingAccount.canInvite) {
      console.log('Failure: user does not have canInvite permission');
      return next({
        statusCode: 403,
        message: 'You do not have permission to invite collaborators.',
      });
    }

    const organizationName = invitingAccount.organizationName || 'DefaultOrg';
    console.log('OrganizationName:', organizationName);

    const existingAccount = await Account.findOne({ email });
    if (existingAccount) {
      console.log('Failure: account already exists with email:', email);
      return next({
        statusCode: 400,
        message: 'An account with this email already exists.',
      });
    }

    const verificationToken = crypto.randomBytes(32).toString('hex');
    const verificationTokenExpires = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 heures
    const tempPassword = crypto.randomBytes(8).toString('hex');
    console.log('Verification token generated:', verificationToken);
    console.log('Temporary password:', tempPassword);

    const newAccount = new Account({
     // name: email.split('@')[0].replace(/[._]/g, ' '),
      email,
      password: tempPassword,
      role: 'user',
      isVerified: false,
      organizationName,
      invitedBy: req.auth.uid,
      canInvite: false,
      status: 'pending',
      invitedDate: new Date(),
      verificationToken,
      verificationTokenExpires,
      mustCompleteProfile: true,
    });
    await newAccount.save();
    console.log('New account created:', newAccount);

    const verificationLink = `${process.env.FRONTEND_URL}/verify-invite/${verificationToken}`;
    console.log('Verification link:', verificationLink);

    await sendInvitationEmail(email, verificationToken, tempPassword, organizationName);
    console.log('Invitation email sent to:', email);

    res.status(201).json({
      message: 'Invitation sent successfully.',
      data: { email, status: 'pending' },
    });
  } catch (error) {
    console.error('Error in inviteAccount:', error);
    next(error);
  }
};

export const verifyInvitation: RequestHandler = async (req, res, next) => {
  try {
    const { token } = req.params;

    const account = await Account.findOne({
      verificationToken: token,
      verificationTokenExpires: { $gt: new Date() },
    });

    if (!account) {
      // Rediriger vers /error si le token est invalide ou expiré
      return res.redirect(303, `${process.env.FRONTEND_URL}/error`);
    }

    // Mettre à jour le compte après vérification
    account.status = 'accepted';
    account.isVerified = true;
    account.verificationToken = undefined;
    account.verificationTokenExpires = undefined;
    await account.save();

    // Rediriger vers /login si le token est valide
    res.redirect(303, `${process.env.FRONTEND_URL}/login`);
  } catch (error) {
    next(error);
  }
};

export const getAccounts: RequestHandler = async (req, res, next) => {
  try {
    const accounts = await Account.find({ status: { $in: ['pending', 'accepted'] } });
    res.status(200).json(accounts.map(a => ({
      id: a._id.toString(),
      name: a.name,
      email: a.email,
      role: a.role,
      status: a.status,
      invitedDate: a.invitedDate ? a.invitedDate.toISOString().split('T')[0] : null,
    })));
  } catch (error) {
    next(error);
  }
};

export const deleteAccount: RequestHandler = async (req, res, next) => {
  try {
    const { id } = req.params;
    console.log('Request received for /col/accounts/:id:', { id });
    console.log('req.auth:', req.auth);

    if (!req.auth?.uid) {
      console.log('Failure: req.auth.uid missing');
      return next({
        statusCode: 401,
        message: 'Authentication required.',
      });
    }

    const deletingAccount = await Account.findById(req.auth.uid);
    console.log('User found in database:', deletingAccount);
    if (!deletingAccount) {
      console.log('Failure: user not found for UID:', req.auth.uid);
      return next({
        statusCode: 401,
        message: 'User not found.',
      });
    }

    if (!deletingAccount.canInvite) {
      console.log('Failure: user does not have canInvite permission');
      return next({
        statusCode: 403,
        message: 'You do not have permission to delete collaborators.',
      });
    }

    const accountToDelete = await Account.findById(id);
    if (!accountToDelete) {
      console.log('Failure: account to delete not found for ID:', id);
      return next({
        statusCode: 404,
        message: 'Account not found.',
      });
    }

    await Account.deleteOne({ _id: id });
    console.log('Account deleted:', id);

    res.status(200).json({
      message: 'Account deleted successfully.',
    });
  } catch (error) {
    console.error('Error in deleteAccount:', error);
    next(error);
  }
};


export const updateProfile: RequestHandler = async (req, res, next) => {
  try {
    const { name, surname, newPassword, receiveUpdates } = req.body;
    console.log('Request received for /update-profile:', { name, surname, newPassword, receiveUpdates });
    console.log('req.auth:', req.auth);

    if (!req.auth?.uid) {
      console.log('Failure: req.auth.uid missing');
      return next({
        statusCode: 401,
        message: 'Authentication required.',
      });
    }

    const account = await Account.findById(req.auth.uid);
    if (!account) {
      console.log('Failure: user not found for UID:', req.auth.uid);
      return next({
        statusCode: 401,
        message: 'User not found.',
      });
    }

    // Mettre à jour les champs
    account.name = name || account.name;
    account.surname = surname || account.surname;
    if (newPassword) {
      account.password = newPassword; // Note: Tu devrais hasher le mot de passe ici avec bcrypt ou une librairie similaire
    }
    account.receiveUpdates = receiveUpdates !== undefined ? receiveUpdates : account.receiveUpdates;
    account.mustCompleteProfile = false; // Mettre à false après mise à jour
    await account.save();

    console.log('Profile updated:', account);
    res.status(200).json({
      message: 'Profile updated successfully.',
      redirect: 'http://localhost:3000', // Redirection vers la page principale
    });
  } catch (error) {
    console.error('Error in updateProfile:', error);
    next(error);
  }
};