import { type RequestHandler } from 'express';
import joi from '../../utils/joi';
import jwt from '../../utils/jwt';
import crypt from '../../utils/crypt';
import { sendVerificationEmail } from '../../utils/nodemailer';
import Account from '../../models/Account';
import crypto from 'crypto';

// Fonction pour générer un ID_Organization unique (3 chiffres + 3 majuscules + 3 chiffres)
const generateUniqueIDOrganization = async () => {
  let id;
  let isUnique = false;

  while (!isUnique) {
    // 3 chiffres début
    const digitsStart = crypto.randomInt(100, 999).toString().padStart(3, '0');

    // 3 lettres majuscules
    const letters = Array.from({ length: 3 }, () => String.fromCharCode(crypto.randomInt(65, 91))).join('');

    // 3 chiffres fin
    const digitsEnd = crypto.randomInt(100, 999).toString().padStart(3, '0');

    id = digitsStart + letters + digitsEnd;

    // Vérifier unicité
    const existing = await Account.findOne({ ID_Organization: id });
    if (!existing) {
      isUnique = true;
    }
  }

  console.log('ID_Organization généré :', id); // Log pour débogage

  return id;
};

const register: RequestHandler = async (req, res, next) => {
  try {
    console.log('Requête d\'inscription reçue :', req.body); // Log des données reçues

    const validationError = await joi.validate(
      {
        name: joi.instance.string().required(),
        surname: joi.instance.string().required(),
        organizationName: joi.instance.string().required(),
        email: joi.instance.string().email().required(),
        password: joi.instance.string().required(),
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
      console.log('Erreur de validation :', validationError); // Log de l'erreur de validation
      return next(validationError);
    }

    const { name, surname, organizationName, email, password, role, canInvite, isVerified, mfaEnabled, teamSize, region } = req.body;

    console.log('Vérification si organizationName existe déjà...');
    // Check if organizationName already exists
    const existingOrg = await Account.findOne({ organizationName });
    if (existingOrg) {
      console.log('organizationName existe déjà :', organizationName); // Log si organisation existe
      return next({
        statusCode: 400,
        message: 'Your organization has been created in ftrack, connect to your organization admin.',
      });
    }

    console.log('Vérification si email existe déjà...');
    const found = await Account.findOne({ email });
    if (found) {
      console.log('Email existe déjà :', email); // Log si email existe
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
    // Générer ID_Organization si status = 'AdministratorOrganization'
    let ID_Organization = null;
    const status = 'AdministratorOrganization'; // Définir 'status' explicitement car il n'est pas dans req.body
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
      status, // Utilisation de la variable 'status' définie
      ID_Organization, // Nouveau champ
    });
    await account.save();
    console.log('Compte créé avec succès :', account._id); // Log du nouveau compte

    console.log('Envoi de l\'email de vérification...');
    await sendVerificationEmail(email, name, verificationCode);

    const token = jwt.signToken({ uid: account._id, role: account.role });

    const { password: _password, verificationCode: _code, verificationCodeExpires: _expires, ...data } = account.toObject();

    res.status(201).json({
      message: 'Signup successful! Please check your email for a verification code.',
      data: { ...data, email },
      token,
    });
  } catch (error: any) {
    console.error('Erreur serveur lors de l\'inscription :', error); // Log détaillé de l'erreur serveur
    if (error.code === 11000) {
      console.log('Erreur de duplication (email ou autre champ unique)'); // Log spécifique pour duplication
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

export default register;