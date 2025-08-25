import { type RequestHandler } from 'express';
import Account from '../../models/Account';
import Feedback from '../../models/Feedback';

export const updateOrganizationSettings: RequestHandler = async (req, res, next) => {
  try {
  const { organizationName, teamSize, region, username, companyFtrackLink, apiKey } = req.body;
  console.log('Requête reçue pour /col/organization:', { organizationName, teamSize, region, username, companyFtrackLink, apiKey });
    console.log('req.auth:', req.auth);

    if (!req.auth?.uid) {
      console.log('Échec : req.auth.uid manquant');
      return next({
        statusCode: 401,
        message: 'Authentification requise.',
      });
    }

    const account = await Account.findById(req.auth.uid);
    if (!account) {
      console.log('Échec : utilisateur non trouvé pour UID:', req.auth.uid);
      return next({
        statusCode: 401,
        message: 'Utilisateur non trouvé.',
      });
    }
    console.log('Utilisateur authentifié trouvé:', account);

    // Vérifier si l'utilisateur est AdministratorOrganization
    if (account.status !== 'AdministratorOrganization') {
      console.log('Échec : utilisateur non autorisé, status:', account.status);
      return next({
        statusCode: 403,
        message: 'Seul l\'administrateur de l\'organisation peut modifier ces informations.',
      });
    }

    // Mettre à jour les champs de l'utilisateur authentifié
  account.organizationName = organizationName || account.organizationName;
  account.teamSize = teamSize || account.teamSize;
  account.region = region || account.region;
  account.username = username || account.username;
  account.apiKey = apiKey || account.apiKey;
  account.companyFtrackLink = companyFtrackLink || account.companyFtrackLink;
  await account.save();

    // Mettre à jour tous les comptes de l'organisation (ceux ayant invitedBy = UID de l'admin)
    await Account.updateMany(
      { invitedBy: req.auth.uid },
      { $set: { organizationName, teamSize, region, username, companyFtrackLink, apiKey } }
    );

    console.log('Paramètres de l\'organisation mis à jour:', account);
    res.status(200).json({
      message: 'Paramètres de l\'organisation mis à jour avec succès.',
      data: {
        organizationName: account.organizationName,
        teamSize: account.teamSize,
        region: account.region,
        username: account.username,
        companyFtrackLink: account.companyFtrackLink,
        apiKey: account.apiKey ? '**********' : '', // Always show 10 * if exists, never the real value
      },
    });
  } catch (error) {
    console.error('Erreur dans updateOrganizationSettings:', error);
    next(error);
  }
};

export const deleteOrganization: RequestHandler = async (req, res, next) => {
  try {
    if (!req.auth?.uid) {
      console.log('Échec : req.auth.uid manquant');
      return next({
        statusCode: 401,
        message: 'Authentification requise.',
      });
    }

    const account = await Account.findById(req.auth.uid);
    if (!account) {
      console.log('Échec : utilisateur non trouvé pour UID:', req.auth.uid);
      return next({
        statusCode: 401,
        message: 'Utilisateur non trouvé.',
      });
    }

    // Vérifier si l'utilisateur est AdministratorOrganization
    if (account.status !== 'AdministratorOrganization') {
      console.log('Échec : utilisateur non autorisé, status:', account.status);
      return next({
        statusCode: 403,
        message: 'Seul l\'administrateur de l\'organisation peut supprimer l\'organisation.',
      });
    }

    // Supprimer tous les comptes où invitedBy = UID de l'admin, ainsi que le compte de l'admin
    await Account.deleteMany({
      $or: [
        { invitedBy: req.auth.uid },
        { _id: req.auth.uid },
      ],
    });

    console.log('Organisation supprimée avec succès pour:', account.organizationName);
    res.status(200).json({
      message: 'Organisation supprimée avec succès.',
    });
  } catch (error) {
    console.error('Erreur dans deleteOrganization:', error);
    next(error);
  }
};

export const submitFeedback: RequestHandler = async (req, res, next) => {
  try {
    const { rating, feedbackText, suggestFeatures, featureSuggestions } = req.body;
    console.log('Request received for /col/feedback:', { rating, feedbackText, suggestFeatures, featureSuggestions });
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

    // Valider les données
    if (!rating || rating < 1 || rating > 5) {
      return next({
        statusCode: 400,
        message: 'Rating must be between 1 and 5.',
      });
    }

    // Sauvegarder les données dans le modèle Feedback
    const feedback = new Feedback({
      accountId: req.auth.uid,
      rating: Number(rating),
      feedbackText: feedbackText || undefined,
      featureSuggestions: suggestFeatures && Array.isArray(featureSuggestions) ? featureSuggestions.filter((s: string) => s.trim()) : [],
    });
    await feedback.save();

    // Mettre à jour les champs dans Account
    account.rating = Number(rating);
    account.feedbackText = feedbackText || undefined;
    account.featureSuggestions = suggestFeatures && Array.isArray(featureSuggestions) ? featureSuggestions.filter((s: string) => s.trim()) : [];
    await account.save();

    console.log('Feedback submitted:', feedback);
    res.status(201).json({
      message: 'Feedback submitted successfully.',
    });
  } catch (error) {
    console.error('Error in submitFeedback:', error);
    next(error);
  }
};

export const getOrganizationSettings: RequestHandler = async (req, res, next) => {
  try {
    if (!req.auth?.uid) {
      console.log('Échec : req.auth.uid manquant');
      return next({
        statusCode: 401,
        message: 'Authentification requise.',
      });
    }

    const account = await Account.findById(req.auth.uid);
    if (!account) {
      console.log('Échec : utilisateur non trouvé pour UID:', req.auth.uid);
      return next({
        statusCode: 401,
        message: 'Utilisateur non trouvé.',
      });
    }

    res.status(200).json({
      message: 'Données de l\'organisation récupérées avec succès.',
      data: {
        organizationName: account.organizationName,
        teamSize: account.teamSize,
        region: account.region,
        username: account.username,
        companyFtrackLink: account.companyFtrackLink,
        id: account._id,
      },
    });
  } catch (error) {
    console.error('Erreur dans getOrganizationSettings:', error);
    next(error);
  }
};