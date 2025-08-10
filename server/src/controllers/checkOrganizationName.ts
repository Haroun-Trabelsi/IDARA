import { type RequestHandler } from 'express';
import Account from '../models/Account';

export const checkOrganizationName: RequestHandler = async (req, res, next) => {
  try {
    console.log('Requête reçue pour /col/check-organization-name:', req.body);
    const { organizationName } = req.body;

    if (!organizationName) {
      console.log('Erreur: organizationName manquant ou vide');
      res.status(400).json({
        statusCode: 400,
        message: 'Organization name is required.',
      });
      return;
    }

    const existingOrg = await Account.findOne({
      organizationName: { $regex: `^${organizationName}$`, $options: 'i' }
    });
    if (existingOrg) {
      console.log('Erreur: organizationName existe déjà (insensible à la casse):', organizationName);
      res.status(400).json({
        statusCode: 400,
        message: 'Your organization has been created in ftrack, connect to your organization admin.',
      });
      return;
    }

    console.log('Nom d\'organisation disponible:', organizationName);
    res.status(200).json({
      message: 'Organization name is available.',
    });
  } catch (error) {
    console.error('Erreur serveur dans checkOrganizationName:', error);
    next({
      statusCode: 500,
      message: 'Server error during organization name check.',
      error: (error as Error).message,
    });
  }
};