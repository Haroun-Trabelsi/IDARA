import { type RequestHandler } from 'express';
import joi from '../../utils/joi';
import crypt from '../../utils/crypt';
import { sendVerificationEmail } from '../../utils/nodemailer';
import Account from '../../models/Account';
import crypto from 'crypto';

// Fonction pour générer un ID_Organization unique (3 chiffres + 3 majuscules + 3 chiffres)
const generateUniqueIDOrganization = async () => {
  let id;
  let isUnique = false;

  while (!isUnique) {
    const digitsStart = crypto.randomInt(100, 999).toString().padStart(3, '0');
    const letters = Array.from({ length: 3 }, () => String.fromCharCode(crypto.randomInt(65, 91))).join('');
    const digitsEnd = crypto.randomInt(100, 999).toString().padStart(3, '0');
    id = digitsStart + letters + digitsEnd;
    const existing = await Account.findOne({ ID_Organization: id });
    if (!existing) {
      isUnique = true;
    }
  }

  console.log('ID_Organization généré :', id);
  return id;
};

const register: RequestHandler = async (req, res, next) => {
  try {
    console.log('Requête d\'inscription reçue :', req.body);

    const validationError = await joi.validate(
      {
        name: joi.instance.string().required(),
        surname: joi.instance.string().required(),
        organizationName: joi.instance.string().required(),
        email: joi.instance.string().email().required(),
        password: joi.instance.string().min(8).required(),
        role: joi.instance.string().valid('user', 'admin').default('user'),
        canInvite: joi.instance.boolean().default(true),
        isVerified: joi.instance.boolean().default(false),
        mfaEnabled: joi.instance.boolean().default(false),
        teamSize: joi.instance.string().required(),
        region: joi.instance.string().required(),
      },
      req.body,
      { stripUnknown: true }
    );

    if (validationError) {
      console.log('Erreur de validation :', validationError);
      return next(validationError);
    }

    const { name, surname, organizationName, email, password, role, canInvite, isVerified, mfaEnabled, teamSize, region } = req.body;

    console.log('Vérification si organizationName existe déjà...');
    const existingOrg = await Account.findOne({ organizationName });
    if (existingOrg) {
      console.log('organizationName existe déjà :', organizationName);
      return next({
        statusCode: 400,
        message: 'Your organization has been created in ftrack, connect to your organization admin.',
      });
    }

    console.log('Vérification si email existe déjà...');
    const found = await Account.findOne({ email });
    if (found) {
      console.log('Email existe déjà :', email);
      return next({
        statusCode: 400,
        message: 'An account already exists with this email',
      });
    }

    console.log('Hachage du mot de passe...');
    const hash = await crypt.hash(password);

    const verificationCode = crypto.randomInt(100000, 999999).toString();
    const verificationCodeExpires = new Date(Date.now() + 30 * 60 * 1000); // 30 minutes

    console.log('Génération de ID_Organization pour AdministratorOrganization...');
    let ID_Organization = null;
    const status = 'AdministratorOrganization';
    if (status === 'AdministratorOrganization') {
      ID_Organization = await generateUniqueIDOrganization();
    }

    console.log('Création du compte...');
    const account = new Account({
      name,
      surname,
      organizationName,
      email,
      password: hash,
      role,
      isVerified,
      verificationCode,
      verificationCodeExpires,
      mfaEnabled,
      canInvite,
      teamSize,
      region,
      status,
      ID_Organization,
    });
    await account.save();
    console.log('Compte créé avec succès :', account._id);

    console.log('Envoi de l\'email de vérification...');
    await sendVerificationEmail(email, name, verificationCode);

    const { password: _password, verificationCode: _code, verificationCodeExpires: _expires, ...data } = account.toObject();

    res.status(201).json({
      message: 'Signup successful! Please check your email for a verification code.',
      data: { ...data, email, _id: account._id },
    });
  } catch (error: any) {
    console.error('Erreur serveur lors de l\'inscription :', error);
    if (error.code === 11000) {
      console.log('Erreur de duplication (email ou autre champ unique)');
      return next({
        statusCode: 400,
        message: 'An account already exists with this email',
      });
    }
    next({
      statusCode: 500,
      message: 'Server error during signup',
      error: error.message,
    });
  }
};

const updateAccount: RequestHandler = async (req, res, next) => {
  try {
    console.log('Requête de mise à jour reçue :', req.body);

    const validationError = await joi.validate(
      {
        accountId: joi.instance.string().required(),
        name: joi.instance.string().required(),
        surname: joi.instance.string().required(),
        organizationName: joi.instance.string().required(),
        email: joi.instance.string().email().required(),
        password: joi.instance.string().min(8).required(),
        role: joi.instance.string().valid('user', 'admin').default('user'),
        canInvite: joi.instance.boolean().default(true),
        isVerified: joi.instance.boolean().default(false),
        mfaEnabled: joi.instance.boolean().default(false),
        teamSize: joi.instance.string().required(),
        region: joi.instance.string().required(),
      },
      req.body,
      { stripUnknown: true }
    );

    if (validationError) {
      console.log('Erreur de validation :', validationError);
      return next(validationError);
    }

    const { accountId, name, surname, organizationName, email, password, role, canInvite, isVerified, mfaEnabled, teamSize, region } = req.body;

    console.log('Recherche du compte à mettre à jour:', accountId);
    const account = await Account.findById(accountId);
    if (!account) {
      console.log('Compte non trouvé :', accountId);
      return next({
        statusCode: 404,
        message: 'Account not found',
      });
    }

    console.log('Vérification si organizationName existe déjà pour un autre compte...');
    const existingOrg = await Account.findOne({ 
      organizationName, 
      _id: { $ne: accountId } 
    });
    if (existingOrg) {
      console.log('organizationName existe déjà :', organizationName);
      return next({
        statusCode: 400,
        message: 'Your organization has been created in ftrack, connect to your organization admin.',
      });
    }

    console.log('Vérification si email existe déjà pour un autre compte...');
    const found = await Account.findOne({ 
      email, 
      _id: { $ne: accountId } 
    });
    if (found) {
      console.log('Email existe déjà :', email);
      return next({
        statusCode: 400,
        message: 'An account already exists with this email',
      });
    }

    console.log('Hachage du mot de passe...');
    const hash = await crypt.hash(password);

    // Vérifier si l'email a changé
    console.log('Ancien email:', account.email, 'Nouvel email:', email);
    const emailChanged = account.email !== email;
    console.log('Email a changé ?', emailChanged);

    let verificationCode;
    let verificationCodeExpires;

    if (emailChanged) {
      console.log('Email a changé, génération d\'un nouveau code de vérification...');
      verificationCode = crypto.randomInt(100000, 999999).toString();
      verificationCodeExpires = new Date(Date.now() + 30 * 60 * 1000); // 30 minutes
      account.isVerified = false; // Réinitialiser l'état de vérification
    }

    console.log('Mise à jour du compte...');
    account.name = name;
    account.surname = surname;
    account.organizationName = organizationName;
    account.email = email;
    account.password = hash;
    account.role = role;
    account.canInvite = canInvite;
    account.isVerified = emailChanged ? false : isVerified;
    account.verificationCode = emailChanged ? verificationCode : account.verificationCode;
    account.verificationCodeExpires = emailChanged ? verificationCodeExpires : account.verificationCodeExpires;
    account.mfaEnabled = mfaEnabled;
    account.teamSize = teamSize;
    account.region = region;
    await account.save();
    console.log('Compte mis à jour avec succès :', account._id);

    if (emailChanged) {
      console.log('Envoi de l\'email de vérification à la nouvelle adresse :', email);
      try {
        await sendVerificationEmail(email, name, verificationCode!);
        console.log('Email de vérification envoyé avec succès à :', email);
      } catch (emailError: any) {
        console.error('Erreur lors de l\'envoi de l\'email de vérification :', emailError);
        return next({
          statusCode: 500,
          message: 'Failed to send verification email to the new address',
          error: emailError.message,
        });
      }
    }

    const { password: _password, verificationCode: _code, verificationCodeExpires: _expires, ...data } = account.toObject();

    res.status(200).json({
      message: emailChanged 
        ? 'Account updated successfully! Please check your new email for a verification code.' 
        : 'Account updated successfully! Please verify your email.',
      data: { ...data, email },
    });
  } catch (error: any) {
    console.error('Erreur serveur lors de la mise à jour :', error);
    if (error.code === 11000) {
      console.log('Erreur de duplication (email ou autre champ unique)');
      return next({
        statusCode: 400,
        message: 'An account already exists with this email or organization name',
      });
    }
    next({
      statusCode: 500,
      message: 'Server error during account update',
      error: error.message,
    });
  }
};

export { register, updateAccount };